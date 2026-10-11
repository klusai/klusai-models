"""Frozen RES-125 scoring: native operator receipts, harness metrics, paired document CIs.

Use the pinned europriv-bench source on PYTHONPATH. Scratch contains gold, checkpoints,
receipts and sufficient statistics; only the aggregate scorecard is intended for commit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from europriv_bench import metrics
from europriv_bench.adapters import KpModelAdapter, PresidioAdapter, ShieldAdapter, kp_entities_to_bioes
from europriv_bench.runner import _load_gold_rows, _rows_to_gold
from europriv_bench.spec import EvalSpec

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "docs/res125-protocol.json").read_text())
GROUPS = ("all", "direct", "quasi", "unspecified")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def literal_mask(text, intervals):
    """Execute the frozen local operator, rejecting invalid native offsets."""
    covered = bytearray(len(text))
    for start, end in intervals:
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError("Invalid native operator interval")
        covered[start:end] = b"\x01" * (end - start)
    return "".join("█" if covered[i] else c for i, c in enumerate(text))


def utility_counts(row, intervals):
    """Micro character utility, with unioned overlapping gold and operator intervals."""
    text = row["text"]
    covered, protected = bytearray(len(text)), bytearray(len(text))
    literal_mask(text, intervals)  # bounds validation is shared with the actual operator
    for start, end in intervals:
        covered[start:end] = b"\x01" * (end - start)
    for span in row["spans"]:
        protected[span["start"]:span["end"]] = b"\x01" * (span["end"] - span["start"])
    eligible = [i for i, c in enumerate(text) if not c.isspace() and not protected[i]]
    return [sum(covered[i] for i in eligible), len(eligible)]


def entity_counts(gold, pred):
    from seqeval.scheme import IOBES, Entities

    out = []
    for true, guessed in zip(gold, pred, strict=True):
        actual = {(e.tag, e.start, e.end) for e in Entities([true], IOBES).entities[0]}
        found = {(e.tag, e.start, e.end) for e in Entities([guessed], IOBES).entities[0]}
        out.append([len(actual & found), len(actual), len(found)])
    return out


def ratio_samples(counts, indices, *, f1=False):
    total = np.asarray(counts)[indices].sum(axis=1)
    numerator = 2 * total[:, 0] if f1 else total[:, 0]
    denominator = total[:, 1] + total[:, 2] if f1 else total[:, 1]
    # NaN prevents empty support being silently interpreted as measured zero leakage.
    return np.divide(numerator, denominator, out=np.full(len(indices), np.nan), where=denominator != 0)


def ci(samples):
    if not np.isfinite(samples).all():
        return None
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def subject_counts(subjects, n, group="all"):
    counts = np.zeros((n, 2), dtype=int)
    for subject in subjects:
        if group == "all" or subject.get("identifier_type", "").lower() == group:
            counts[subject["doc"]] += [not subject["detected"], 1]
    return counts.tolist()


def freeze(args):
    from huggingface_hub import hf_hub_download
    from pyarrow import parquet

    for config in PROTOCOL["configs"]:
        target = args.scratch / "gold" / f"{config}.json"
        if target.exists():
            continue
        path = args.bench / "evaluations" / f"pii-detection-{config}.yaml"
        spec = EvalSpec.from_yaml(path)
        if spec.dataset.loader == "ai4privacy-openpii":
            rows = _load_gold_rows(spec)
            provenance = {**spec.dataset.provenance(), **rows.source_metadata}
        else:
            source = hf_hub_download(
                spec.dataset.hf_id, f"{spec.dataset.config}/test-00000-of-00001.parquet",
                repo_type="dataset", revision=PROTOCOL["benchmark_revision"],
            )
            rows = parquet.read_table(source).to_pylist()
            provenance = {**spec.dataset.provenance(), "revision": PROTOCOL["benchmark_revision"],
                          "parquet_sha256": digest(source)}
        _rows_to_gold(rows)
        write(target, {"config": config, "language": spec.languages[0], "dataset": provenance,
                       "spec_sha256": digest(path), "rows": rows})
        print(f"FROZEN {config} n={len(rows)}", flush=True)


def predict(args):
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(125)
    if args.name == "shield":
        adapter = ShieldAdapter(base_url=args.shield_url, pin=json.loads(args.shield_pin.read_text()))
        identity = {"kind": "shield", "pin_sha256": adapter.client.pin_sha256}
    elif args.name == "presidio":
        adapter = PresidioAdapter()
        identity = {"kind": "presidio", "language": "en", "threshold": 0.0,
                    "crosswalk": adapter.PRESIDIO_TO_KP}
    else:
        from transformers import pipeline

        adapter = KpModelAdapter(str(args.model))
        adapter._pipe = pipeline("token-classification", model=str(args.model), aggregation_strategy="simple",
                                 device=args.device)
        adapter._pipe.tokenizer.model_max_length = 512
        identity = {"kind": "kp", "checkpoint": str(args.model), "training_seed": args.seed,
                    "max_input_tokens": 512, "stride": 0, "dtype": "float32",
                    "files": {p.name: digest(p) for p in sorted(args.model.iterdir())
                              if p.suffix in {".json", ".safetensors", ".model"}}}
    identity.update(
        packages={k: importlib.metadata.version(k) for k in ("torch", "transformers", "spacy", "presidio-analyzer")
                  if importlib.util.find_spec(k.replace("-", "_")) is not None},
        device=args.device if identity["kind"] == "kp" else "cpu",
        protocol_sha256=digest(ROOT / "docs/res125-protocol.json"),
        script_sha256=digest(__file__),
    )
    for config in args.configs or PROTOCOL["configs"]:
        gold_path = args.scratch / "gold" / f"{config}.json"
        frozen = json.loads(gold_path.read_text())
        rows = frozen["rows"]
        if args.name == "shield":
            adapter.prepare([frozen["language"]])
        ident = {**identity, "gold_sha256": digest(gold_path), "config": config}
        target = args.scratch / "predictions" / f"{args.name}--{config}.json"
        saved = json.loads(target.read_text()) if target.exists() else {"identity": ident, "receipts": []}
        if saved["identity"] != ident:
            raise ValueError(f"Stale prediction identity: {target}")
        receipts = saved["receipts"]
        for offset in range(len(receipts), len(rows), 16):
            texts = [r["text"] for r in rows[offset:offset + 16]]
            if args.name == "shield":
                native = adapter.client.analyze(texts)
                for text, result in zip(texts, native, strict=True):
                    entities = [{"start": e["start"], "end": e["end"],
                                 "label": adapter.SHIELD_TO_KP[e["entity_type"]]}
                                for e in result["items"] if e["entity_type"] in adapter.SHIELD_TO_KP]
                    receipts.append({"tags": kp_entities_to_bioes(text, entities),
                                     "intervals": [[e["start"], e["end"]] for e in result["items"]],
                                     "native": result, "provenance": adapter.provenance()})
            else:
                if args.name == "presidio":
                    entities_batch = [
                        [{"start": int(e.start), "end": int(e.end),
                          "label": adapter.PRESIDIO_TO_KP[e.entity_type]}
                         for e in adapter._analyzer().analyze(text=text, language="en", entities=adapter._entities,
                                                             score_threshold=0.0)
                         if e.entity_type in adapter.PRESIDIO_TO_KP]
                        for text in texts
                    ]
                else:
                    entities_batch = [[{"start": int(e["start"]), "end": int(e["end"]),
                                        "label": e["entity_group"]} for e in entities]
                                      for entities in adapter._pipe(texts, batch_size=16)]
                for text, entities in zip(texts, entities_batch, strict=True):
                    intervals = [[e["start"], e["end"]] for e in entities]
                    receipts.append({"tags": kp_entities_to_bioes(text, entities), "intervals": intervals,
                                     "native": entities, "output": literal_mask(text, intervals)})
            write(target, saved)
            print(f"{args.name} {config} {len(receipts)}/{len(rows)}", flush=True)
        score_one(args.scratch, args.name, config)


def score_one(scratch, name, config):
    gold_path = scratch / "gold" / f"{config}.json"
    pred_path = scratch / "predictions" / f"{name}--{config}.json"
    frozen, saved = json.loads(gold_path.read_text()), json.loads(pred_path.read_text())
    if saved["identity"]["gold_sha256"] != digest(gold_path):
        raise ValueError("Gold identity mismatch")
    rows, receipts = frozen["rows"], saved["receipts"]
    if len(rows) != len(receipts):
        raise ValueError("Incomplete prediction run")
    _, gold = _rows_to_gold(rows)
    labels = {t.split("-", 1)[1] for seq in gold for t in seq if t != "O"}
    pred = [[t if t == "O" or t.split("-", 1)[1] in labels else "O" for t in r["tags"]] for r in receipts]
    if any(len(g) != len(p) for g, p in zip(gold, pred, strict=True)):
        raise ValueError("Token alignment mismatch")
    intervals = [r["intervals"] for r in receipts]
    for row, receipt in zip(rows, receipts, strict=True):
        if saved["identity"]["kind"] != "shield":
            if literal_mask(row["text"], receipt["intervals"]) != receipt["output"]:
                raise ValueError("Operator output does not match coverage receipt")
    f1 = metrics.entity_f1(gold, pred)
    v2 = metrics.residual_leakage_v2(rows, intervals)
    subjects = metrics.residual_subject_detection_v2(rows, intervals)
    counts = {"entity_f1": entity_counts(gold, pred),
              "over_redaction": [utility_counts(r, spans) for r, spans in zip(rows, intervals, strict=True)]}
    for group in GROUPS:
        counts[f"v2_{group}"] = subject_counts(subjects, len(rows), group)
    if config == "tab-echr-legal-en":
        legacy = {"tab_reid_leakage": metrics.tab_reid_leakage(rows, pred)}
        for group in ("all", "direct", "quasi"):
            counts[f"legacy_{group}"] = subject_counts(metrics.tab_reid_subject_detection(rows, pred), len(rows), group)
    else:
        legacy = {"national_id_leakage": metrics.national_id_leakage(rows, pred),
                  "name_in_context_leakage": metrics.name_in_context_leakage(rows, pred)}
        units = [s for s in metrics.national_id_subject_detection(rows, pred) if s["decode_bearing"]]
        counts["legacy_national_id"] = subject_counts(units, len(rows))
    indices = np.random.default_rng(125).integers(len(rows), size=(2000, len(rows)))
    intervals95 = {k: ci(ratio_samples(c, indices, f1=k == "entity_f1")) for k, c in counts.items()}
    fp, nonpii = np.asarray(counts["over_redaction"]).sum(axis=0)
    result = {"name": name, "config": config, "language": frozen["language"], "n_documents": len(rows),
              "gold_sha256": digest(gold_path), "prediction_sha256": digest(pred_path),
              "identity": saved["identity"], "dataset": frozen["dataset"], "entity_f1": f1,
              "legacy": legacy, "residual_leakage_v2": v2,
              "utility": {"removed_nonpii_characters": int(fp), "nonpii_characters": int(nonpii),
                          "over_redaction": float(fp / nonpii) if nonpii else None,
                          "retention": float(1 - fp / nonpii) if nonpii else None},
              "bootstrap_ci95": intervals95, "counts": counts}
    total = np.asarray(counts["entity_f1"]).sum(axis=0)
    if abs(f1["f1"] - 2 * total[0] / (total[1] + total[2])) > 1e-12:
        raise ValueError("Bootstrap sufficient statistics disagree with harness F1")
    write(scratch / "results" / f"{name}--{config}.json", result)
    print(f"SCORED {name} {config} F1={f1['f1']:.4f} v2={v2['all_leak_rate']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "predict", "score"))
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--configs", nargs="+")
    parser.add_argument("--shield-url", default="http://127.0.0.1:11446")
    parser.add_argument("--shield-pin", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args)
    elif args.action == "predict":
        predict(args)
    else:
        for config in args.configs or PROTOCOL["configs"]:
            score_one(args.scratch, args.name, config)


if __name__ == "__main__":
    main()
