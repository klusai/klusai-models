#!/usr/bin/env python3
"""RES-104 — combine the two xlmr-560m arm results into the stage-B-transfer scorecard (.md + .json).

The xlmr-560m analog of RES-102. Reads the per-arm result JSONs emitted by
``train_res102_stageb_transfer.py`` run with ``--base FacebookAI/xlm-roberta-large`` (each carries a
TAB entity-F1 from the EXACT RES-97 eval path), computes the within-pair delta, and writes an honest
scorecard. The DELTA is the result; absolute levels are low (2000-doc models). A null / negative /
within-noise delta is reported plainly — that decides whether step 2b (expensive scale-up) is worth
the hours.

Two reference baselines already on the board are reported for context (NOT as direct comparisons):
  * RES-102 280M matched-pair: stage-A 0.0261 / stage-B 0.1006 (delta +0.0745) — same 2000-doc
    matched-pair setup, smaller base. This is the apples-to-apples comparison: does the bigger base
    amplify or dampen the stage-B benefit?
  * Full-corpus xlmr-560m: TAB entity-F1 0.2439 (template-splice, 40k docs). NOT directly comparable
    to these 2000-doc arms in absolute terms — different data volume and recipe; included only to
    locate where these tiny matched arms sit relative to the board entry.

    python scripts/scorecard_res104.py \
        --stagea runs/res104-stagea-seed0-result.json \
        --stageb runs/res104-stageb-seed0-result.json \
        --out-json docs/res104-xlmr560m-stageb-scorecard.json \
        --out-md   docs/res104-xlmr560m-stageb-scorecard.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click

# Reference baselines already on the board (context only — NOT direct comparisons; see module docs).
RES102_280M_STAGEA_TAB_F1 = 0.0261
RES102_280M_STAGEB_TAB_F1 = 0.1006
RES102_280M_DELTA = round(RES102_280M_STAGEB_TAB_F1 - RES102_280M_STAGEA_TAB_F1, 4)  # +0.0745
FULLCORPUS_XLMR560M_TAB_F1 = 0.2439  # template-splice, 40k docs — not size/recipe-matched to these arms.


@click.command()
@click.option("--stagea", required=True, help="arm A (stage-A) result JSON.")
@click.option("--stageb", required=True, help="arm B (stage-B) result JSON.")
@click.option("--stagea-integrity", default="../klusai-datasets/artifacts/europriv/en_stagea_matched_v1_metrics.json")
@click.option("--stageb-integrity", default="../klusai-datasets/artifacts/europriv/en_stageb_v1_metrics.json")
@click.option("--out-json", default="docs/res104-xlmr560m-stageb-scorecard.json")
@click.option("--out-md", default="docs/res104-xlmr560m-stageb-scorecard.md")
def main(stagea, stageb, stagea_integrity, stageb_integrity, out_json, out_md):
    a = json.loads(Path(stagea).read_text())
    b = json.loads(Path(stageb).read_text())
    a_f1 = a["tab_entity_f1"]
    b_f1 = b["tab_entity_f1"]
    delta = round(b_f1 - a_f1, 4)

    integ_a = json.loads(Path(stagea_integrity).read_text()) if Path(stagea_integrity).exists() else {}
    integ_b = json.loads(Path(stageb_integrity).read_text()) if Path(stageb_integrity).exists() else {}

    # Honest reading. At 2000 docs, F1 differences below ~0.01 are within run-to-run noise (no
    # multi-seed variance band here — single seed per arm, matched). State the direction and whether
    # it clears that rough floor.
    noise_floor = 0.01
    if abs(delta) < noise_floor:
        verdict = (
            f"WITHIN NOISE. The stage-B arm scored TAB entity-F1 {b_f1:.4f} vs the stage-A arm "
            f"{a_f1:.4f} (delta {delta:+.4f}). At 2000 docs and a single seed per arm, a gap this "
            f"small (|delta| < {noise_floor}) is not distinguishable from run-to-run noise. On this "
            f"setup the structural-diversity effect on REAL-legal transfer is NOT demonstrated for "
            f"xlmr-560m. Step 2b (scale-up) is NOT justified by this run alone."
        )
        effect = "within_noise"
    elif delta > 0:
        verdict = (
            f"POSITIVE (directional). Stage-B transferred better to REAL legal gold for xlmr-560m: "
            f"TAB entity-F1 {b_f1:.4f} vs stage-A {a_f1:.4f} (delta {delta:+.4f}). The arms share "
            f"identical PII and size; only document STRUCTURE differs, so the gap is attributable to "
            f"structural diversity. Single seed per arm — directional, not a CI-backed claim. "
            f"Absolute levels are low (2000-doc models); the within-pair delta is the result."
        )
        effect = "positive_directional"
    else:
        verdict = (
            f"NEGATIVE. Stage-B did NOT transfer better for xlmr-560m: TAB entity-F1 {b_f1:.4f} vs "
            f"stage-A {a_f1:.4f} (delta {delta:+.4f}). On this matched 2000-doc setup, template-"
            f"splice data matched or beat structurally-diverse data on REAL-legal transfer. Reported "
            f"as-is; step 2b (scale-up) is NOT justified by this run."
        )
        effect = "negative"

    # How does the bigger base compare to the 280M's RES-102 delta?
    delta_vs_280m = round(delta - RES102_280M_DELTA, 4)
    if abs(delta) < noise_floor and abs(RES102_280M_DELTA) >= noise_floor:
        amplification = (
            f"DAMPENED. The 280M showed a clear positive matched-pair delta (+{RES102_280M_DELTA:.4f}); "
            f"at xlmr-560m the same matched-pair delta is {delta:+.4f}, within the {noise_floor} noise "
            f"floor. The bigger base did NOT reproduce the stage-B benefit at this data scale."
        )
    elif delta > 0 and delta > RES102_280M_DELTA + noise_floor:
        amplification = (
            f"AMPLIFIED. xlmr-560m's stage-B matched-pair delta ({delta:+.4f}) exceeds the 280M's "
            f"(+{RES102_280M_DELTA:.4f}) by {delta_vs_280m:+.4f} — the bigger base appears to magnify "
            f"the structural-diversity benefit (single seed; directional)."
        )
    elif delta > 0:
        amplification = (
            f"COMPARABLE. xlmr-560m's stage-B matched-pair delta ({delta:+.4f}) is the same sign and "
            f"within ~{noise_floor} of the 280M's (+{RES102_280M_DELTA:.4f}); the benefit broadly "
            f"holds at the bigger base (directional)."
        )
    else:
        amplification = (
            f"REVERSED. The 280M showed a positive matched-pair delta (+{RES102_280M_DELTA:.4f}); "
            f"xlmr-560m's is {delta:+.4f} (opposite sign). The bigger base did not reproduce — and "
            f"here reversed — the 280M stage-B benefit (single seed; directional)."
        )

    scorecard = {
        "issue": "RES-104",
        "schema": 1,
        "question": "Does training the BEST detector architecture (xlmr-560m, base FacebookAI/"
                    "xlm-roberta-large) on structurally-diverse stage-B synthetic EN data transfer "
                    "better to REAL legal gold (TAB ECHR) than template-splice stage-A data, at "
                    "matched size + matched PII (only document STRUCTURE differs)? [xlmr-560m analog "
                    "of RES-102]",
        "eval": {
            "name": "TAB ECHR legal English (REAL peer-reviewed gold)",
            "config": a["tab"]["config"],
            "n": a["tab"]["n"],
            "eval_labels": a["tab"]["eval_labels"],
            "path": "europriv-bench/evaluations/pii-detection-tab-echr-legal-en.yaml (RES-89)",
            "scoring": "entity_f1 via scorecard_klu106 (_predict/_mask/_entity_f1_on_rows) — EXACT "
                       "RES-97 path; apples-to-apples with the xlmr-560m TAB baseline.",
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
        "baselines_on_board": {
            "res102_280m_matched_pair": {
                "stagea_tab_f1": RES102_280M_STAGEA_TAB_F1,
                "stageb_tab_f1": RES102_280M_STAGEB_TAB_F1,
                "delta": RES102_280M_DELTA,
                "note": "Same 2000-doc matched-pair setup, smaller base (mdeberta-280m). The "
                        "apples-to-apples comparison for the within-pair delta.",
            },
            "fullcorpus_xlmr560m": {
                "tab_f1": FULLCORPUS_XLMR560M_TAB_F1,
                "recipe": "template-splice, 40k docs",
                "note": "NOT directly comparable to these 2000-doc arms in absolute terms (different "
                        "data volume + recipe). Included only to locate these tiny matched arms "
                        "relative to the board entry.",
            },
            "delta_vs_280m": delta_vs_280m,
            "amplification_reading": amplification,
        },
        "comparability_caveat": "These 2000-doc arms are NOT directly comparable in absolute terms "
                                "to the 40k full-corpus xlmr-560m board entry (0.2439, template-"
                                "splice). The within-pair DELTA is the result of this experiment.",
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
            "absolute_not_comparable_to_40k_board_entry": True,
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
    click.echo(f"vs 280M RES-102 delta +{RES102_280M_DELTA:.4f}  ->  delta_vs_280m={delta_vs_280m:+.4f}")


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
    bl = s["baselines_on_board"]
    ia, ib = s["dataset_integrity"]["stagea"], s["dataset_integrity"]["stageb"]
    lines = []
    lines.append("# RES-104 — Stage-B structural-diversity transfer to REAL legal gold (TAB), xlmr-560m\n")
    lines.append(f"*{s['timestamp']}* · config_status: **dev** (synthetic training; TAB is real external gold)\n")
    lines.append("## Question\n")
    lines.append(s["question"] + "\n")
    lines.append("## Result (the within-pair delta is the headline)\n")
    lines.append("| Arm | Train data | Structure | TAB entity-F1 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| A (stage-A) | `en_stagea_matched_v1.jsonl` ({mp['n_train_rows_per_arm']['stagea']}) | template-splice | **{r['stagea_tab_entity_f1']:.4f}** |")
    lines.append(f"| B (stage-B) | `en_stageb_v1.jsonl` ({mp['n_train_rows_per_arm']['stageb']}) | LLM-narrated | **{r['stageb_tab_entity_f1']:.4f}** |")
    lines.append(f"\n**Delta (stage-B − stage-A) = {r['delta_stageb_minus_stagea']:+.4f}** · effect: **{r['effect']}** (noise floor {r['noise_floor']})\n")
    lines.append("## Reading\n")
    lines.append(s["reading"] + "\n")
    lines.append("## Comparison to baselines on the board\n")
    lines.append("| Baseline | Setup | TAB entity-F1 | Matched-pair delta |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **RES-104 xlmr-560m** (this) | 2000-doc matched pair | A {r['stagea_tab_entity_f1']:.4f} / B {r['stageb_tab_entity_f1']:.4f} | **{r['delta_stageb_minus_stagea']:+.4f}** |")
    lines.append(f"| RES-102 mdeberta-280m | 2000-doc matched pair | A {bl['res102_280m_matched_pair']['stagea_tab_f1']:.4f} / B {bl['res102_280m_matched_pair']['stageb_tab_f1']:.4f} | +{bl['res102_280m_matched_pair']['delta']:.4f} |")
    lines.append(f"| Full-corpus xlmr-560m | 40k template-splice | {bl['fullcorpus_xlmr560m']['tab_f1']:.4f} | n/a (single arm) |")
    lines.append(f"\n**Does the bigger base amplify or dampen the stage-B benefit?** {bl['amplification_reading']}\n")
    lines.append(f"> **Comparability caveat.** {s['comparability_caveat']}\n")
    lines.append("## Setup (matched pair — only STRUCTURE differs)\n")
    lines.append(f"- Base model: `{mp['base_model']}` (the BEST detector architecture, xlmr-560m)")
    lines.append(f"- Hyperparams (identical across arms): `{json.dumps(mp['hyperparams'])}`")
    lines.append(f"- Identical hyperparams + seed across arms: **{mp['identical_hyperparams_and_seed_across_arms']}**")
    lines.append(f"- Eval: {s['eval']['name']} — {s['eval']['n']} docs, labels {s['eval']['eval_labels']}")
    lines.append(f"- Scoring: {s['eval']['scoring']}")
    lines.append(f"- Single seed per arm (seed {mp['hyperparams']['seed']}); no multi-seed CI band — directional.\n")
    lines.append("## Dataset integrity\n")
    lines.append(f"- **stage-A**: n={ia['n']}, invalid NINO={ia['invalid_nino']}, invalid IBAN={ia['invalid_iban']}, "
                 f"all rows byte-equal + BIOES-valid={ia['all_rows_byte_equal_and_bioes_valid']}; "
                 f"matched-pair PII: {json.dumps(ia.get('matched_pair_pii'))}")
    lines.append(f"- **stage-B**: n={ib['n']}, invalid NINO={ib['invalid_nino']}, invalid IBAN={ib['invalid_iban']}, "
                 f"all rows byte-equal + BIOES-valid={ib['all_rows_byte_equal_and_bioes_valid']}\n")
    lines.append("## Guards\n")
    for k, v in s["guards"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
