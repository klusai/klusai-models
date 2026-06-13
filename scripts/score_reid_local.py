"""RES-104: score our LOCAL kp checkpoints on the TAB re-id-risk metric (DIRECT/QUASI leak rate).

Honest completion of the re-id board: the published board scored `kp-model` = the shipped 280M (lowest
recall → most leakage, last place). Here we score our higher-recall models — xlmr-560m (det-F1 0.244),
the CJEU real-structure model (0.340), the real-prose model (0.265) — to see if any leads on re-id risk
vs the board leaders (spacy 0.496 / presidio 0.500 DIRECT-leak). Inference only; gold pulled from HF.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scorecard_klu106 lives here
from datasets import load_dataset
from europriv_bench.metrics import entity_f1, tab_reid_leakage
from europriv_bench.spans import Span, char_spans_to_bioes
from scorecard_klu106 import _mask, _predict

CKPTS = {
    "kp-xlmr-560m (F1 0.244)": "runs/kp-deid-xlmr-560m-seed0",
    "kp-cjeu-structure (F1 0.340)": "runs/res72-cjeu-tab-seed0",
    "kp-cjeu-realprose (F1 0.265)": "runs/res72-realprose-seed0",
}


def main() -> None:
    ds = load_dataset("klusai/europriv-bench", "tab-echr-legal-en-v1", split="test")
    rows = [{"text": r["text"], "spans": r["spans"]} for r in ds]
    texts = [r["text"] for r in rows]
    gold_tags = [char_spans_to_bioes(r["text"], [Span(s["start"], s["end"], s["label"]) for s in r["spans"]])
                 for r in rows]
    eval_labels = sorted({t.split("-", 1)[1] for seq in gold_tags for t in seq if t != "O"})
    print(f"TAB test: {len(rows)} docs, eval_labels={eval_labels}\n")

    out = []
    for name, ckpt in CKPTS.items():
        if not Path(ckpt).exists():
            print(f"[skip] {name}: {ckpt} missing")
            continue
        t0 = time.time()
        pred = _mask(_predict(ckpt, texts), eval_labels)
        f1 = entity_f1(gold_tags, pred)["f1"]
        rl = tab_reid_leakage(rows, pred)
        rec = {"model": name, "entity_f1": round(f1, 4),
               "direct_leak_rate": round(rl["direct_leak_rate"], 4),
               "direct_leak_ci": [round(rl["direct_leak_rate_ci_low"], 4), round(rl["direct_leak_rate_ci_high"], 4)],
               "quasi_leak_rate": round(rl["quasi_leak_rate"], 4)}
        out.append(rec)
        print(f"  {name:32} F1={rec['entity_f1']:.3f}  DIRECT-leak={rec['direct_leak_rate']:.3f} "
              f"CI{rec['direct_leak_ci']}  QUASI-leak={rec['quasi_leak_rate']:.3f}  ({time.time()-t0:.0f}s)")

    Path("runs/res104-reid-local.json").write_text(json.dumps(out, indent=2))
    print("\nvs board: spacy 0.496 / presidio 0.500 / kp-280M 0.674 (DIRECT-leak, ↓ better)")


if __name__ == "__main__":
    main()
