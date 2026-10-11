"""Harness scoring for full gateway coverage and explicitly conditional BIOES F1."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scorecard_res125 import ci, digest, entity_counts, literal_mask, ratio_samples, subject_counts, write
from scorecard_res125 import score_one as score_anchor

from europriv_bench import metrics
from europriv_bench.adapters import DummyAdapter
from europriv_bench.gateway import utility_counts
from europriv_bench.runner import _rows_to_gold, run_spec
from europriv_bench.spec import DatasetRef, EvalSpec, Task


def score_one(scratch, name, config):
    if not config.startswith("gateway-"):
        return score_anchor(scratch, name, config)
    gold_path = scratch / "gold" / f"{config}.json"
    pred_path = scratch / "predictions" / f"{name}--{config}.json"
    frozen, saved = json.loads(gold_path.read_text()), json.loads(pred_path.read_text())
    if saved["identity"]["gold_sha256"] != digest(gold_path):
        raise ValueError("Gold identity mismatch")
    rows, receipts = frozen["rows"], saved["receipts"]
    if len(rows) != len(receipts):
        raise ValueError("Incomplete prediction run")
    intervals = [r["intervals"] for r in receipts]
    if saved["identity"]["kind"] != "shield":
        for row, receipt in zip(rows, receipts, strict=True):
            if literal_mask(row["text"], receipt["intervals"]) != receipt["output"]:
                raise ValueError("Operator output does not match coverage receipt")
    spec = EvalSpec(name=config, task=Task.DETECTION, languages=[frozen["language"]],
                    dataset=DatasetRef(hf_id="klusai/europriv-bench", config="gateway-traffic-v1", loader="gateway"),
                    metrics=["residual_leakage_v2", "gateway_utility"])
    scores = run_spec(spec, DummyAdapter(), rows=rows, redaction_spans=intervals)["scores"]
    gold, pred, supported, excluded = [], [], [], []
    labels = {s["label"] for row in rows for s in row["spans"]}
    for index, (row, receipt) in enumerate(zip(rows, receipts, strict=True)):
        try:
            _, tags = _rows_to_gold([row])
        except ValueError as error:
            if "token collision" not in str(error):
                raise
            excluded.append(row["id"])
            continue
        guesses = [t if t == "O" or t.split("-", 1)[1] in labels else "O" for t in receipt["tags"]]
        if len(tags[0]) != len(guesses):
            raise ValueError("F1 token alignment mismatch")
        gold.append(tags[0])
        pred.append(guesses)
        supported.append(index)
    f1 = metrics.entity_f1(gold, pred)
    f1_counts = [[0, 0, 0] for _ in rows]
    for index, count in zip(supported, entity_counts(gold, pred), strict=True):
        f1_counts[index] = count
    counts = {"entity_f1": f1_counts, "over_redaction": utility_counts(rows, intervals)}
    subjects = metrics.residual_subject_detection_v2(rows, intervals)
    for group in ("all", "direct", "quasi", "unspecified"):
        counts[f"v2_{group}"] = subject_counts(subjects, len(rows), group)
    indices = np.random.default_rng(125).integers(len(rows), size=(2000, len(rows)))
    intervals95 = {k: ci(ratio_samples(c, indices, f1=k == "entity_f1")) for k, c in counts.items()}
    removed, denominator = np.asarray(counts["over_redaction"]).sum(axis=0)
    result = {"name": name, "config": config, "language": frozen["language"], "n_documents": len(rows),
              "gold_sha256": digest(gold_path), "prediction_sha256": digest(pred_path),
              "identity": saved["identity"], "dataset": frozen["dataset"], "entity_f1": f1,
              "f1_scope": {"full_split_available": not excluded, "scored_documents": len(supported),
                           "excluded_documents": len(excluded), "exclusion": "gold whitespace-token collision",
                           "meaning": "conditional diagnostic only; no exclusion from v2 or utility",
                           "excluded_ids_sha256": hashlib.sha256(json.dumps(excluded).encode()).hexdigest()},
              "legacy": {}, "residual_leakage_v2": scores["residual_leakage_v2"],
              "utility": {"removed_nonpii_characters": int(removed), "nonpii_characters": int(denominator),
                          "over_redaction": float(removed / denominator) if denominator else None,
                          "retention": float(1 - removed / denominator) if denominator else None},
              "bootstrap_ci95": intervals95, "counts": counts, "scorer_sha256": digest(__file__)}
    write(scratch / "results" / f"{name}--{config}.json", result)
    print(f"SCORED {name} {config} F1_subset_n={len(supported)} v2_full_n={len(rows)}", flush=True)
