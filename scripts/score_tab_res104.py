"""Score one frozen TAB arm through the harness; cache predictions and update the scorecard.

Run baseline adapters in the existing europriv-bench environment (read-only sources), model
checkpoints in the encoder environment. All outputs go to this worktree. No model selection here.
Bootstrap F1 and paired deltas over documents; DIRECT leak uses harness Wilson intervals plus
paired document-cluster bootstrap deltas (subjects within a judgment are not independent).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from score_reid_local import score_predictions
from scorecard_klu106 import _doc_entity_counts, _mask, _predict

from europriv_bench.metrics import tab_reid_subject_detection
from europriv_bench.runner import _rows_to_gold

BOOTSTRAPS = 2000
BOOT_SEED = 104
MARKER = "<!-- RES104-TAB-SUPERVISED -->"


def bootstrap_samples(gold, pred, rows):
    """Return paired document-bootstrap F1 and DIRECT-leak samples, with a fixed draw matrix."""
    counts = np.asarray(_doc_entity_counts(gold, pred)).T
    rng = np.random.default_rng(BOOT_SEED)
    indices = rng.integers(0, len(rows), size=(BOOTSTRAPS, len(rows)))
    pooled = counts[indices].sum(axis=1)
    denom = pooled[:, 1] + pooled[:, 2]
    f1 = np.divide(2 * pooled[:, 0], denom, out=np.zeros(BOOTSTRAPS), where=denom != 0)
    subjects = np.zeros((len(rows), 2), dtype=int)
    for s in tab_reid_subject_detection(rows, pred):
        if s["identifier_type"] == "DIRECT":
            subjects[s["doc"]] += [not s["detected"], 1]
    pooled = subjects[indices].sum(axis=1)
    leakage = np.divide(pooled[:, 0], pooled[:, 1], out=np.zeros(BOOTSTRAPS), where=pooled[:, 1] != 0)
    return f1, leakage


def ci(values):
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def refresh(outdir: Path, rows, gold):
    records, samples = {}, {}
    for path in sorted(outdir.glob("*-predictions.json")):
        obj = json.loads(path.read_text())
        key = obj["name"]
        records[key] = obj["result"]
        samples[key] = bootstrap_samples(gold, obj["predictions"], rows)
    for key, r in records.items():
        r["f1_ci95"] = ci(samples[key][0])
        r["paired_deltas"] = {}
        for reference in ("presidio", "spacy", f"cjeu-seed{r.get('seed')}"):
            if reference not in records or reference == key:
                continue
            ref = records[reference]
            r["paired_deltas"][reference] = {
                "f1": r["entity_f1"]["f1"] - ref["entity_f1"]["f1"],
                "f1_ci95": ci(samples[key][0] - samples[reference][0]),
                "direct_leak": r["tab_reid_leakage"]["direct_leak_rate"] - ref["tab_reid_leakage"]["direct_leak_rate"],
                "direct_leak_ci95": ci(samples[key][1] - samples[reference][1]),
            }
    aggregates = {}
    for arm in ("tab", "cjeu"):
        keys = [k for k in records if k.startswith(f"{arm}-seed")]
        if not keys:
            continue
        f1s = [records[k]["entity_f1"]["f1"] for k in keys]
        leaks = [records[k]["tab_reid_leakage"]["direct_leak_rate"] for k in keys]
        aggregates[arm] = {
            "seeds": [records[k]["seed"] for k in keys],
            "seed_count": len(keys),
            "f1_mean": float(np.mean(f1s)),
            "f1_range": [min(f1s), max(f1s)],
            "direct_mean": float(np.mean(leaks)),
            "direct_range": [min(leaks), max(leaks)],
        }
    result = {
        "n_docs": len(rows),
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": BOOT_SEED,
        "bootstrap_unit": "document (paired across models; DIRECT subjects clustered within document)",
        "interval": "95% percentile; marginal DIRECT intervals use harness Wilson",
        "results": records,
        "seed_spread": aggregates,
    }
    Path("docs/res104-tab-results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        MARKER,
        "## TAB-train supervision (RES-104 climb)",
        "",
        "Fixed TAB-only arm; checkpoint selection uses TAB dev only. "
        "Data licence, hashes, split checks and alignment exclusions: [manifest](res104-tab-manifest.json).",
        "Train 1,010 / dev 126 documents after the board curator's exclusions; test 127 documents, "
        "4,821 spans, 264 DIRECT subjects. No CJEU mix arm was run.",
        "",
        "Three epochs, LoRA r16, LR 3e-4, batch 16 fp32 MPS; full-document 256-token windows. "
        "Minimum dev-loss checkpoint; loss is only a selection signal. Test never selects an arm/epoch.",
        "",
        "Harness CPU scoring (`score_reid_local.score_predictions`); F1 95% CIs and paired deltas "
        "use 2,000 document-bootstrap draws (seed 104); DIRECT marginal CIs are Wilson. "
        "Delta leak CIs cluster subjects by document. Machine-readable [results](res104-tab-results.json).",
        "",
        "| Arm | Seed | F1 [95% CI] | DIRECT leak [95% CI] | ΔF1 vs Presidio [95% CI] |",
        "|---|---:|---|---|---|",
    ]

    def interval(value, bounds):
        return f"{value:.4f} [{bounds[0]:.4f}, {bounds[1]:.4f}]"

    for key, r in records.items():
        leak = r["tab_reid_leakage"]
        delta = r["paired_deltas"].get("presidio")
        delta_text = interval(delta["f1"], delta["f1_ci95"]) if delta else "—"
        seed_text = r["seed"] if r.get("seed") is not None else "—"
        lines.append(
            f"| {key} | {seed_text} | "
            f"{interval(r['entity_f1']['f1'], r['f1_ci95'])} | "
            f"{interval(leak['direct_leak_rate'], [leak['direct_leak_rate_ci_low'], leak['direct_leak_rate_ci_high']])} | "
            f"{delta_text} |"
        )
    lines.extend(["", "Seed spread (training seeds; each row above is one seed):", ""])
    for arm, a in aggregates.items():
        lines.append(
            f"- {arm}: n_seeds={a['seed_count']} {a['seeds']}; F1 mean {a['f1_mean']:.4f}, "
            f"range [{a['f1_range'][0]:.4f}, {a['f1_range'][1]:.4f}]; DIRECT mean {a['direct_mean']:.4f}, "
            f"range [{a['direct_range'][0]:.4f}, {a['direct_range'][1]:.4f}]."
        )
    lines.extend(
        [
            "",
            "| Arm | ΔF1 vs same-seed CJEU [95% CI] | ΔDIRECT vs spaCy [95% CI] | ΔDIRECT vs same-seed CJEU [95% CI] |",
            "|---|---|---|---|",
        ]
    )
    for key, r in records.items():
        if r.get("seed") is None:
            continue
        deltas = r["paired_deltas"]
        same_seed = deltas.get(f"cjeu-seed{r['seed']}")
        cells = [interval(same_seed["f1"], same_seed["f1_ci95"]) if same_seed else "—"]
        for ref in ("spacy", f"cjeu-seed{r['seed']}"):
            d = deltas.get(ref)
            cells.append(interval(d["direct_leak"], d["direct_leak_ci95"]) if d else "—")
        lines.append(f"| {key} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "Scope: supervised held-out TAB EN legal, not zero-shot. Subject identity and near-duplicate "
            "passage disjointness are not established. CIs condition on the trained seed; seed ranges "
            "are reported separately. Other languages/domains and a mixed-data arm were not tested. "
            "Scoring preserves the existing encoder adapter's 512-token truncation (no stride) and its "
            "any-token-overlap per-mention re-id proxy; complete removal of every identifier character "
            "is not verified.",
            "",
        ]
    )
    path = Path("docs/res104-reid-win-scorecard.md")
    original = path.read_text().split(MARKER)[0].rstrip()
    path.write_text(original + "\n\n" + "\n".join(lines))
    print(json.dumps(result, indent=2), flush=True)


def main():
    os.environ["EUROPRIV_DEVICE"] = "cpu"
    import torch

    torch.set_num_threads(4)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--adapter", choices=("presidio", "spacy"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--outdir", type=Path, default=Path("runs/res104-tab-scores"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    payload = Path("artifacts/res104-tab/test.jsonl").read_bytes()
    manifest = json.loads(Path("docs/res104-tab-manifest.json").read_text())
    if hashlib.sha256(payload).hexdigest() != manifest["splits"]["test"]["jsonl_sha256"]:
        raise ValueError("Test data changed after the audit")
    rows = [json.loads(line) for line in payload.decode().splitlines()]
    texts, gold = _rows_to_gold(rows)
    labels = sorted({t.split("-", 1)[1] for seq in gold for t in seq if t != "O"})
    path = args.outdir / f"{args.name}-predictions.json"
    weights_sha256 = None
    if args.model:
        with (Path(args.model) / "model.safetensors").open("rb") as fh:
            weights_sha256 = hashlib.file_digest(fh, "sha256").hexdigest()
    if path.exists():
        old = json.loads(path.read_text())["result"]
        if (
            old["model"] != (args.model or args.adapter)
            or old["seed"] != args.seed
            or old["test_sha256"] != hashlib.sha256(payload).hexdigest()
            or old.get("weights_sha256") != weights_sha256
        ):
            raise ValueError("Cached predictions belong to a different model, seed or test set")
    if not path.exists():
        if args.model:
            pred = _predict(args.model, texts)
        else:
            from europriv_bench.adapters import build

            pred = build(args.adapter).predict_tags(texts)
        pred = _mask(pred, labels)
        result = score_predictions(rows, pred)
        result.update(
            {
                "seed": args.seed,
                "model": args.model or args.adapter,
                "n_docs": len(rows),
                "test_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        if args.model:
            checkpoint = Path(args.model)
            result["weights_sha256"] = weights_sha256
            training_path = checkpoint.with_name(checkpoint.name + "-train.json")
            if not training_path.exists():
                training_path = checkpoint.with_name(checkpoint.name + "-result.json")
            if training_path.exists():
                training = json.loads(training_path.read_text())
                result["training_record"] = str(training_path)
                result["training_record_sha256"] = hashlib.sha256(training_path.read_bytes()).hexdigest()
                result["checkpoint_selection"] = training.get("selection", "fixed RES-72 epoch count")
                result["best_checkpoint"] = training.get("best_checkpoint")
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps({"name": args.name, "result": result, "predictions": pred}))
        partial.replace(path)
    refresh(args.outdir, rows, gold)


if __name__ == "__main__":
    main()
