#!/usr/bin/env python3
"""RES-102 — combine the two arm results into the stage-B-transfer scorecard (.md + .json).

Reads the per-arm result JSONs emitted by ``train_res102_stageb_transfer.py`` (each carries a TAB
entity-F1 from the EXACT RES-97 eval path), computes the delta, and writes an honest scorecard. The
delta is the result; absolute levels are low (2000-doc models). A null/negative/within-noise delta
is reported plainly.

    python scripts/scorecard_res102.py \
        --stagea runs/res102-stagea-seed0-result.json \
        --stageb runs/res102-stageb-seed0-result.json \
        --out-json docs/res102-stageb-transfer-scorecard.json \
        --out-md   docs/res102-stageb-transfer-scorecard.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click

# The known prior: mdeberta-280m trained on the full 8-language synthetic corpus scored TAB
# entity-F1 = 0.0489 (RES-97 scorecard). Context only — not part of this matched pair.
RES97_MDEBERTA_TAB_F1 = 0.0489


@click.command()
@click.option("--stagea", required=True, help="arm A (stage-A) result JSON.")
@click.option("--stageb", required=True, help="arm B (stage-B) result JSON.")
@click.option("--stagea-integrity", default="../klusai-datasets/artifacts/europriv/en_stagea_matched_v1_metrics.json")
@click.option("--stageb-integrity", default="../klusai-datasets/artifacts/europriv/en_stageb_v1_metrics.json")
@click.option("--out-json", default="docs/res102-stageb-transfer-scorecard.json")
@click.option("--out-md", default="docs/res102-stageb-transfer-scorecard.md")
def main(stagea, stageb, stagea_integrity, stageb_integrity, out_json, out_md):
    a = json.loads(Path(stagea).read_text())
    b = json.loads(Path(stageb).read_text())
    a_f1 = a["tab_entity_f1"]
    b_f1 = b["tab_entity_f1"]
    delta = round(b_f1 - a_f1, 4)

    integ_a = json.loads(Path(stagea_integrity).read_text()) if Path(stagea_integrity).exists() else {}
    integ_b = json.loads(Path(stageb_integrity).read_text()) if Path(stageb_integrity).exists() else {}

    # Honest reading. At 2000 docs, F1 differences below ~0.01 are within run-to-run noise (no
    # multi-seed variance band was run here — single seed per arm, matched). State the direction and
    # whether it clears that rough floor.
    noise_floor = 0.01
    if abs(delta) < noise_floor:
        verdict = (
            f"WITHIN NOISE. The stage-B arm scored TAB entity-F1 {b_f1:.4f} vs the stage-A arm "
            f"{a_f1:.4f} (delta {delta:+.4f}). At 2000 docs and a single seed per arm, a gap this "
            f"small (|delta| < {noise_floor}) is not distinguishable from run-to-run noise. On this "
            f"setup the structural-diversity effect on REAL-legal transfer is NOT demonstrated."
        )
        effect = "within_noise"
    elif delta > 0:
        verdict = (
            f"POSITIVE (directional). Stage-B transferred better to REAL legal gold: TAB entity-F1 "
            f"{b_f1:.4f} vs stage-A {a_f1:.4f} (delta {delta:+.4f}). The arms share identical PII and "
            f"size; only document STRUCTURE differs, so the gap is attributable to structural "
            f"diversity. Single seed per arm — directional, not a CI-backed claim. Absolute levels "
            f"are low (2000-doc models); the delta is the result."
        )
        effect = "positive_directional"
    else:
        verdict = (
            f"NEGATIVE. Stage-B did NOT transfer better: TAB entity-F1 {b_f1:.4f} vs stage-A "
            f"{a_f1:.4f} (delta {delta:+.4f}). On this matched 2000-doc setup, template-splice data "
            f"matched or beat structurally-diverse data on REAL-legal transfer. Reported as-is."
        )
        effect = "negative"

    scorecard = {
        "issue": "RES-102",
        "schema": 1,
        "question": "Does training kp-deid-mdeberta-280m on structurally-diverse stage-B synthetic EN "
                    "data transfer better to REAL legal gold (TAB ECHR) than template-splice stage-A "
                    "data, at matched size + matched PII (only document STRUCTURE differs)?",
        "eval": {
            "name": "TAB ECHR legal English (REAL peer-reviewed gold)",
            "config": a["tab"]["config"],
            "n": a["tab"]["n"],
            "eval_labels": a["tab"]["eval_labels"],
            "path": "europriv-bench/evaluations/pii-detection-tab-echr-legal-en.yaml (RES-89)",
            "scoring": "entity_f1 via scorecard_klu106 (_predict/_mask/_entity_f1_on_rows) — EXACT "
                       "RES-97 path; apples-to-apples with the mdeberta-280m TAB baseline.",
        },
        "matched_pair": {
            "base_model": a["base_model"],
            "n_train_rows_per_arm": {"stagea": a["n_train_rows"], "stageb": b["n_train_rows"]},
            "hyperparams": a["hyperparams"],
            "identical_hyperparams_and_seed_across_arms": a["hyperparams"] == b["hyperparams"],
            "only_difference": "document body STRUCTURE (template-splice vs LLM-narrated); PII and "
                               "size matched",
        },
        "results": {
            "stagea_tab_entity_f1": a_f1,
            "stageb_tab_entity_f1": b_f1,
            "delta_stageb_minus_stagea": delta,
            "noise_floor": noise_floor,
            "effect": effect,
            "stagea_tab_full": a["tab"],
            "stageb_tab_full": b["tab"],
        },
        "context_prior": {
            "res97_mdeberta_280m_full_corpus_tab_f1": RES97_MDEBERTA_TAB_F1,
            "note": "RES-97: mdeberta-280m trained on the full 8-lang synthetic corpus collapsed on "
                    "TAB (0.0489). Context only; not part of this matched pair.",
        },
        "dataset_integrity": {
            "stagea": _integrity_line(integ_a),
            "stageb": _integrity_line(integ_b),
        },
        "reading": verdict,
        "guards": {
            "config_status": "dev (synthetic training; TAB is real external gold)",
            "single_seed_per_arm": True,
            "no_citable_or_sota_claim": True,
            "negative_or_null_reported_truthfully": True,
        },
        "train_artifacts": {
            "stagea": {k: a[k] for k in ("checkpoint_dir", "train_seconds", "device", "data")},
            "stageb": {k: b[k] for k in ("checkpoint_dir", "train_seconds", "device", "data")},
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")

    md = _render_md(scorecard)
    Path(out_md).write_text(md, encoding="utf-8")
    click.echo(f"wrote {out_json} and {out_md}\n")
    click.echo(f"stage-A TAB-F1={a_f1:.4f}  stage-B TAB-F1={b_f1:.4f}  delta={delta:+.4f}  ({effect})")


def _integrity_line(integ: dict) -> dict:
    gi = integ.get("gold_integrity", {})
    return {
        "n": integ.get("n"),
        "invalid_nino": gi.get("invalid_nino"),
        "invalid_iban": gi.get("invalid_iban"),
        "all_rows_byte_equal_and_bioes_valid": gi.get("all_rows_byte_equal_and_bioes_valid"),
        "matched_pair_pii": integ.get("matched_pair_pii"),
    }


def _render_md(s: dict) -> str:
    r = s["results"]
    mp = s["matched_pair"]
    ia, ib = s["dataset_integrity"]["stagea"], s["dataset_integrity"]["stageb"]
    lines = []
    lines.append("# RES-102 — Stage-B structural-diversity transfer to REAL legal gold (TAB)\n")
    lines.append(f"*{s['timestamp']}* · config_status: **dev** (synthetic training; TAB is real external gold)\n")
    lines.append("## Question\n")
    lines.append(s["question"] + "\n")
    lines.append("## Result (the delta is the headline)\n")
    lines.append("| Arm | Train data | Structure | TAB entity-F1 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| A (stage-A) | `en_stagea_matched_v1.jsonl` ({mp['n_train_rows_per_arm']['stagea']}) | template-splice | **{r['stagea_tab_entity_f1']:.4f}** |")
    lines.append(f"| B (stage-B) | `en_stageb_v1.jsonl` ({mp['n_train_rows_per_arm']['stageb']}) | LLM-narrated | **{r['stageb_tab_entity_f1']:.4f}** |")
    lines.append(f"\n**Delta (stage-B − stage-A) = {r['delta_stageb_minus_stagea']:+.4f}** · effect: **{r['effect']}** (noise floor {r['noise_floor']})\n")
    lines.append("## Reading\n")
    lines.append(s["reading"] + "\n")
    lines.append("## Setup (matched pair — only STRUCTURE differs)\n")
    lines.append(f"- Base model: `{mp['base_model']}` (the shipped kp-deid-mdeberta-280m architecture)")
    lines.append(f"- Hyperparams (identical across arms): `{json.dumps(mp['hyperparams'])}`")
    lines.append(f"- Identical hyperparams + seed across arms: **{mp['identical_hyperparams_and_seed_across_arms']}**")
    lines.append(f"- Eval: {s['eval']['name']} — {s['eval']['n']} docs, labels {s['eval']['eval_labels']}")
    lines.append(f"- Scoring: {s['eval']['scoring']}")
    lines.append(f"- Single seed per arm (seed {mp['hyperparams']['seed']}); no multi-seed CI band.\n")
    lines.append("## Dataset integrity\n")
    lines.append(f"- **stage-A**: n={ia['n']}, invalid NINO={ia['invalid_nino']}, invalid IBAN={ia['invalid_iban']}, "
                 f"all rows byte-equal + BIOES-valid={ia['all_rows_byte_equal_and_bioes_valid']}; "
                 f"matched-pair PII: {json.dumps(ia.get('matched_pair_pii'))}")
    lines.append(f"- **stage-B**: n={ib['n']}, invalid NINO={ib['invalid_nino']}, invalid IBAN={ib['invalid_iban']}, "
                 f"all rows byte-equal + BIOES-valid={ib['all_rows_byte_equal_and_bioes_valid']}\n")
    lines.append("## Context prior\n")
    lines.append(f"- RES-97: mdeberta-280m on the full 8-lang synthetic corpus scored TAB entity-F1 = "
                 f"{s['context_prior']['res97_mdeberta_280m_full_corpus_tab_f1']:.4f} (the known collapse). "
                 f"Context only; not part of this matched pair.\n")
    lines.append("## Guards\n")
    for k, v in s["guards"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
