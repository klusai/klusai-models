"""Post-hoc harness coverage diagnostic; never selects models or changes primary gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_gateway_res127 import PROTOCOL
from scorecard_res125 import digest, write

from europriv_bench import metrics


def selected(row, scope):
    if scope == "all_national_id_gold":
        return [span for span in row["spans"] if span["label"] == "NATIONAL_ID"]
    spans = []
    for span in row["spans"]:
        validator = metrics.get_validator(metrics._span_country(row, span))
        if (validator is not None and validator.decode_bearing
                and validator.parse(row["text"][span["start"]:span["end"]]).valid):
            spans.append(span)
    return spans


def audit(scratch):
    names = [f"mdeberta-280m-seed{seed}" for seed in range(3)] + PROTOCOL["comparators"]
    results = []
    for config in ("ro-realskeleton", "pl-realskeleton"):
        gold_path = scratch / "gold" / f"{config}.json"
        gold = json.loads(gold_path.read_text())
        for scope in ("all_national_id_gold", "validator_valid_decode_bearing"):
            rows = [{**row, "spans": selected(row, scope)} for row in gold["rows"]]
            for name in names:
                pred_path = scratch / "predictions" / f"{name}--{config}.json"
                prediction = json.loads(pred_path.read_text())
                if prediction["identity"]["gold_sha256"] != digest(gold_path):
                    raise ValueError("Ordered gold identity mismatch")
                if len(prediction["receipts"]) != len(rows):
                    raise ValueError("Incomplete prediction receipts")
                intervals = [r["intervals"] for r in prediction["receipts"]]
                results.append({"config": config, "scope": scope, "name": name,
                                "training_seed": prediction["identity"].get("training_seed"),
                                "n_documents": len(rows), "gold_sha256": digest(gold_path),
                                "prediction_sha256": digest(pred_path),
                                "residual_leakage_v2": metrics.residual_leakage_v2(rows, intervals)})
    write(scratch / "national-id-v2-audit.json", {
        "post_hoc": True, "harness_commit": PROTOCOL["harness_commit"], "script_sha256": digest(__file__),
        "purpose": "Distinguish the historical any-overlap national-ID floor from exact-character coverage",
        "selection": "No model selection or new win claim; predeclared primary/legacy results remain unchanged",
        "scopes": {
            "all_national_id_gold": "All NATIONAL_ID-labeled gold, including values outside the validated subset",
            "validator_valid_decode_bearing": "Same pinned harness validators as the legacy anchor, across all labels",
        },
        "method": "Keep every document and all native intervals; filter only gold spans for each named diagnostic. "
                  "Original v2 subject grouping and sensitivity metadata remain unchanged; Wilson CIs are descriptive.",
        "results": results,
    })
    print(f"Audited {len(results)} national-ID coverage rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    audit(parser.parse_args().scratch)
