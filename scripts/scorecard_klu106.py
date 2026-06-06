#!/usr/bin/env python3
"""KLU-106 — produce the held-out scorecard for kp-deid v2 vs the zero-shot KLU-51 control.

Acceptance is "produce the scorecard on the FIXED held-out eval set", not "train a good model".
This scores both models on the SAME clean held-out surfaces v2 never trained on and reports rigorous,
contamination-labelled numbers:

  * **Held-out clean general split** (per language) carved BEFORE training (template- AND
    subject-disjoint, KLU-106) — entity-F1, plus an identifier-surface-form holdout subset.
  * **Real-skeleton tracks** (ro/pl/it) from EuroPriv-Bench — entity-F1 + per-subject national-ID
    re-id leak with a Wilson upper bound. (it-realskeleton-v1 is skip-and-reported if not yet
    published on the hub — ConfigUnavailableError.)

Rigorous claims (KLU-106):
  * **F1 gain = bootstrap-CI'd Δ(v2 − control)** on the SAME held-out set, per language; the headline
    requires the Δ CI to **exclude 0**. The control is the zero-shot KLU-51 ``kp-deid`` scored on the
    SAME held-out set (NOT the published 0.46–0.52).
  * **>=2-seed variance** for the headline delta (bf16/MPS nondeterminism — a gain inside seed-noise
    is not a gain): we score every completed v2 seed and report mean/min/max F1 + Δ.
  * **re-id leak ~0** on clean_held_out real-skeleton, per-subject, Wilson UB vs a pre-registered
    threshold.
  * **detection-F1 gain AND re-id-leak Δ side-by-side per track**, surfacing any config where F1 rises
    but leak ALSO rises (the dissociation breaking) — flagged, not averaged away.
  * Every cell schema-3 labelled (``contamination``, ``config_status=dev``).

Run from the repo root after training (``scripts/train_v2_klu106.py``):

    python scripts/scorecard_klu106.py \
        --manifest runs/klu106-train-manifest.json \
        --control klusai/kp-deid-mdeberta-280m \
        --suite ../europriv-bench/evaluations \
        --out runs/klu106-scorecard.json
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import click

from klusai.privacy.models.logger import get_logger

logger = get_logger("scorecard_klu106")

# Pre-registered re-id leak Wilson upper-bound threshold on clean_held_out real-skeleton (per-subject).
# The v1 board sustained leak_rate=0.000 with Wilson UB ~0.0034 on ro-realskeleton (n=1123 subjects);
# we require the v2 UB to stay at/under this band — "~0", not point-0 (KLU-106).
LEAK_WILSON_UB_THRESHOLD = 0.01

BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 12345


def _entity_f1_on_rows(rows: list[dict], pred_tags) -> dict:
    """Strict entity-F1 over a row subset using the harness scoring path (eval-label fairness mask)."""
    from europriv_bench.metrics import entity_f1
    from europriv_bench.runner import _rows_to_gold

    _, gold_tags = _rows_to_gold(rows)
    eval_labels = {t.split("-", 1)[1] for seq in gold_tags for t in seq if t != "O"}
    masked = [
        [t if (t == "O" or t.split("-", 1)[1] in eval_labels) else "O" for t in seq]
        for seq in pred_tags
    ]
    return entity_f1(gold_tags, masked)


def _doc_entity_counts(gold_tags, pred_tags):
    """Per-document strict (IOBES) entity counts ``(tp, n_pred, n_gold)`` for every document.

    seqeval's ``f1_score`` (no ``average=`` override) is corpus-level **micro**-F1: it pools all
    entities across documents, so micro-F1 over any multiset of documents is exactly reconstructable
    from the summed per-document ``(tp, n_pred, n_gold)``. Precomputing these counts once turns each
    bootstrap iteration into O(n) integer sums instead of an O(n) seqeval re-parse — mathematically
    identical to ``entity_f1`` on the resample, but ~10^3x cheaper (the full-corpus bootstrap is
    otherwise day-scale on CPU). Strict IOBES entity extraction is per-sequence, so the counts sum.
    """
    from seqeval.scheme import IOBES, Entities

    tps: list[int] = []
    npreds: list[int] = []
    ngolds: list[int] = []
    for g, p in zip(gold_tags, pred_tags):
        eg = Entities([g], scheme=IOBES).entities[0]
        ep = Entities([p], scheme=IOBES).entities[0]
        sg = {(e.tag, e.start, e.end) for e in eg}
        sp = {(e.tag, e.start, e.end) for e in ep}
        tps.append(len(sg & sp))
        npreds.append(len(sp))
        ngolds.append(len(sg))
    return tps, npreds, ngolds


def _micro_f1(tp: int, n_pred: int, n_gold: int) -> float:
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _bootstrap_delta_ci(gold_tags, pred_a, pred_b, *, iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    """Paired bootstrap CI for F1(a) − F1(b) over the SAME rows (resample documents with replacement).

    Returns ``(delta, low, high)`` for the 95% percentile interval. Paired: each bootstrap sample
    draws the same document indices for both models, so the CI reflects the per-document paired
    difference (the correct uncertainty for "did v2 beat the control on this fixed eval set").

    Implemented over precomputed per-document micro-F1 counts (see ``_doc_entity_counts``): each
    iteration sums the resampled ``(tp, n_pred, n_gold)`` for both models and recomputes micro-F1 —
    identical to recomputing seqeval on the resample, but fast enough for the full corpus."""
    tp_a, np_a, ng_a = _doc_entity_counts(gold_tags, pred_a)
    tp_b, np_b, ng_b = _doc_entity_counts(gold_tags, pred_b)
    n = len(gold_tags)
    rng = random.Random(seed)

    def f1_at(idx, tp, npred, ngold):
        s_tp = sum(tp[i] for i in idx)
        s_np = sum(npred[i] for i in idx)
        s_ng = sum(ngold[i] for i in idx)
        return _micro_f1(s_tp, s_np, s_ng)

    full = range(n)
    base = f1_at(full, tp_a, np_a, ng_a) - f1_at(full, tp_b, np_b, ng_b)
    deltas = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(f1_at(idx, tp_a, np_a, ng_a) - f1_at(idx, tp_b, np_b, ng_b))
    deltas.sort()
    lo = deltas[int(0.025 * iters)]
    hi = deltas[min(iters - 1, int(0.975 * iters))]
    return base, lo, hi


def _predict(model_id: str, texts: list[str]):
    from europriv_bench.adapters import KpModelAdapter

    return KpModelAdapter(model_id=model_id).predict_tags(texts)


def _classify(config: str) -> str:
    from europriv_bench.leaderboard import classify_contamination

    return classify_contamination("kp-model", config)


def _score_general_heldout(heldout_dir, surface_idx, v2_models, control, langs_filter):
    """Per-language held-out general scorecard: F1(v2 seeds) vs F1(control), bootstrap Δ CI."""
    from datasets import load_from_disk

    from europriv_bench.runner import _rows_to_gold

    ds = load_from_disk(heldout_dir)
    by_lang: dict[str, list[int]] = {}
    for i in range(ds.num_rows):
        by_lang.setdefault(ds[i]["language"], []).append(i)

    surface_set = set(surface_idx or [])
    cells = []
    for lang in sorted(by_lang):
        if langs_filter and lang not in langs_filter:
            continue
        idx = by_lang[lang]
        rows = [ds[i] for i in idx]
        texts = [r["text"] for r in rows]
        _, gold_tags = _rows_to_gold(rows)
        eval_labels = sorted({t.split("-", 1)[1] for seq in gold_tags for t in seq if t != "O"})

        # Control predictions (zero-shot KLU-51 model) on the SAME rows.
        ctrl_pred = _mask(_predict(control, texts), eval_labels)
        ctrl_f1 = _entity_f1_on_rows(rows, ctrl_pred)["f1"]

        # Each completed v2 seed.
        seed_cells = []
        per_seed_pred = {}
        for sd, mid in v2_models.items():
            pred = _mask(_predict(mid, texts), eval_labels)
            per_seed_pred[sd] = pred
            f1 = _entity_f1_on_rows(rows, pred)["f1"]
            d, lo, hi = _bootstrap_delta_ci(_mask_gold(gold_tags, eval_labels), pred, ctrl_pred)
            seed_cells.append({
                "seed": sd, "model_id": mid, "f1": round(f1, 4),
                "delta_vs_control": round(d, 4),
                "delta_ci95_low": round(lo, 4), "delta_ci95_high": round(hi, 4),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            })

        # >=2-seed variance on the headline delta.
        seed_deltas = [c["delta_vs_control"] for c in seed_cells]
        seed_f1s = [c["f1"] for c in seed_cells]
        all_exclude_zero = all(c["ci_excludes_zero"] and c["delta_vs_control"] > 0 for c in seed_cells)

        # Identifier-surface-form holdout subset for this language.
        sub_local = [j for j, gi in enumerate(idx) if gi in surface_set]
        surface_cell = None
        if sub_local:
            sub_rows = [rows[j] for j in sub_local]
            sub_texts = [r["text"] for r in sub_rows]
            ctrl_sub = _mask(_predict(control, sub_texts), eval_labels)
            surface_cell = {"n": len(sub_rows), "control_f1": round(_entity_f1_on_rows(sub_rows, ctrl_sub)["f1"], 4)}
            for sd, pred in per_seed_pred.items():
                # re-predict on the subset (offsets differ) for honesty
                sp = _mask(_predict(v2_models[sd], sub_texts), eval_labels)
                surface_cell[f"v2_seed{sd}_f1"] = round(_entity_f1_on_rows(sub_rows, sp)["f1"], 4)

        cells.append({
            "track": "general-heldout-clean",
            "language": lang,
            "n": len(rows),
            "eval_labels": eval_labels,
            "contamination": "clean_held_out",   # carved before training, template+subject disjoint
            "contamination_basis": "KLU-106 per-language held-out general (template- AND subject-disjoint)",
            "config_status": "dev",
            "control_model": control,
            "control_f1": round(ctrl_f1, 4),
            "v2_seeds": seed_cells,
            "v2_f1_mean": round(sum(seed_f1s) / len(seed_f1s), 4) if seed_f1s else None,
            "v2_f1_min": round(min(seed_f1s), 4) if seed_f1s else None,
            "v2_f1_max": round(max(seed_f1s), 4) if seed_f1s else None,
            "delta_mean": round(sum(seed_deltas) / len(seed_deltas), 4) if seed_deltas else None,
            "delta_min": round(min(seed_deltas), 4) if seed_deltas else None,
            "delta_max": round(max(seed_deltas), 4) if seed_deltas else None,
            "headline_gain_all_seeds_ci_exclude_0": all_exclude_zero,
            "identifier_surface_form_holdout": surface_cell,
        })
    return cells


def _mask(pred_tags, eval_labels):
    s = set(eval_labels)
    return [[t if (t == "O" or t.split("-", 1)[1] in s) else "O" for t in seq] for seq in pred_tags]


def _mask_gold(gold_tags, eval_labels):
    s = set(eval_labels)
    return [[t if (t == "O" or t.split("-", 1)[1] in s) else "O" for t in seq] for seq in gold_tags]


def _score_realskeleton(suite, v2_models, control, only_filter):
    """Real-skeleton tracks: F1 + per-subject national-ID re-id leak (Wilson UB), side-by-side."""
    from europriv_bench.adapters import KpModelAdapter
    from europriv_bench.runner import ConfigUnavailableError, run_spec
    from europriv_bench.spec import EvalSpec, Task

    specs = sorted(Path(suite).glob("*realskeleton*.yaml"))
    if only_filter:
        specs = [p for p in specs if only_filter in p.name]
    ts = datetime.now(timezone.utc).isoformat()

    cells = []
    for path in specs:
        spec = EvalSpec.from_yaml(path)
        # Only DETECTION real-skeleton tracks carry entity_f1 + a national-ID re-id leak metric; the
        # anonymization (Track C) spec on the same data has a different score shape, so skip it here.
        if spec.task is not Task.DETECTION:
            continue
        models = {"control": control, **{f"v2_seed{sd}": mid for sd, mid in v2_models.items()}}
        per_model = {}
        unavailable = False
        for tag, mid in models.items():
            try:
                res = run_spec(spec, KpModelAdapter(model_id=mid), timestamp=ts)
            except ConfigUnavailableError as e:
                logger.warning("skip-and-report %s: %s", spec.name, e)
                unavailable = True
                break
            per_model[tag] = res
        if unavailable:
            cells.append({"track": "real-skeleton", "spec": spec.name,
                          "config": spec.dataset.config, "status": "unavailable_on_hub_skipped"})
            continue

        # Leak metric key (cnp_leakage on ro, national_id_leakage on pl/it).
        leak_key = "cnp_leakage" if "cnp_leakage" in spec.metrics else "national_id_leakage"
        ctrl = per_model["control"]
        ctrl_f1 = ctrl["scores"]["entity_f1"]["f1"]
        ctrl_leak = ctrl["scores"].get(leak_key, {})
        v2_rows = []
        for sd in v2_models:
            r = per_model[f"v2_seed{sd}"]
            f1 = r["scores"]["entity_f1"]["f1"]
            leak = r["scores"].get(leak_key, {})
            f1_gain = f1 - ctrl_f1
            leak_delta = leak.get("leak_rate", 0.0) - ctrl_leak.get("leak_rate", 0.0)
            ub = leak.get("leak_rate_ci_high", 0.0)
            v2_rows.append({
                "seed": sd, "model_id": v2_models[sd],
                "f1": round(f1, 4), "f1_gain_vs_control": round(f1_gain, 4),
                "leak_rate": round(leak.get("leak_rate", 0.0), 5),
                "leak_rate_wilson_ub": round(ub, 5),
                "leak_rate_delta_vs_control": round(leak_delta, 5),
                "subjects_total": leak.get("decode_bearing_total", leak.get("cnp_total")),
                "leak_ub_under_threshold": bool(ub <= LEAK_WILSON_UB_THRESHOLD),
                # Dissociation breaks if F1 rises AND leak rises together.
                "dissociation_break": bool(f1_gain > 0 and leak_delta > 1e-9),
            })
        cells.append({
            "track": "real-skeleton",
            "spec": spec.name,
            "config": spec.dataset.config,
            "languages": spec.languages,
            "leak_metric": leak_key,
            "contamination": ctrl.get("contamination"),
            "config_status": ctrl.get("config_status", "dev"),
            "n": ctrl["n"],
            "control_f1": round(ctrl_f1, 4),
            "control_leak_rate": round(ctrl_leak.get("leak_rate", 0.0), 5),
            "control_leak_wilson_ub": round(ctrl_leak.get("leak_rate_ci_high", 0.0), 5),
            "leak_wilson_ub_threshold": LEAK_WILSON_UB_THRESHOLD,
            "v2_seeds": v2_rows,
            "any_dissociation_break": any(r["dissociation_break"] for r in v2_rows),
        })
    return cells


@click.command()
@click.option("--manifest", default="runs/klu106-train-manifest.json",
              help="Training manifest from scripts/train_v2_klu106.py (held-out dir + seed dirs).")
@click.option("--control", default="klusai/kp-deid-mdeberta-280m",
              help="Zero-shot KLU-51 kp-deid control (scored on the SAME held-out set).")
@click.option("--suite", default="../europriv-bench/evaluations")
@click.option("--out", default="runs/klu106-scorecard.json")
@click.option("--only-realskeleton", default=None, help="Substring filter on realskeleton spec names.")
@click.option("--langs", default=None, help="Comma list to restrict general-heldout languages (debug).")
@click.option("--heldout-general", "heldout_general_override", default=None,
              help="Override the held-out general dir (RES-19: re-score against the harder set).")
@click.option("--skip-realskeleton", is_flag=True, default=False,
              help="Skip the real-skeleton tracks (RES-19: this is a general-eval-quality re-score).")
@click.option("--threads", type=int, default=4)
def main(manifest, control, suite, out, only_realskeleton, langs, heldout_general_override,
         skip_realskeleton, threads):
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass
    os.environ.setdefault("EUROPRIV_DEVICE", "cpu")  # scoring on CPU is fine + deterministic-ish

    man = json.loads(Path(manifest).read_text())
    # RES-19: re-score the EXISTING trained v2 seeds + control against a harder held-out general set,
    # without touching the KLU-106 carve. ``--heldout-general`` overrides the dir read from the
    # manifest; the surface-form-holdout indices (computed against the KLU-106 carve) do not apply to
    # an overridden set, so they are dropped.
    heldout_dir = heldout_general_override or man["heldout_general_dir"]
    surface_idx = [] if heldout_general_override else (
        man.get("identifier_surface_form_holdout", {}).get("indices") or [])
    v2_models = {a["seed"]: a["output_dir"] for a in man.get("seed_artifacts", [])}
    if not v2_models:
        raise click.UsageError("manifest has no completed seed_artifacts to score.")
    logger.info("scoring v2 seeds %s vs control %s on held-out %s", sorted(v2_models), control, heldout_dir)

    langs_filter = set(langs.split(",")) if langs else None

    general_cells = _score_general_heldout(heldout_dir, surface_idx, v2_models, control, langs_filter)
    realskeleton_cells = (
        [] if skip_realskeleton
        else _score_realskeleton(suite, v2_models, control, only_realskeleton)
    )

    # Headline summary: per-language held-out F1 gain (mean over seeds), CI-exclude-0 status.
    headline = {
        c["language"]: {
            "control_f1": c["control_f1"],
            "v2_f1_mean": c["v2_f1_mean"],
            "delta_mean": c["delta_mean"],
            "delta_min": c["delta_min"], "delta_max": c["delta_max"],
            "all_seeds_ci_exclude_0": c["headline_gain_all_seeds_ci_exclude_0"],
        }
        for c in general_cells
    }
    leak_summary = [
        {"spec": c.get("spec"), "config": c.get("config"),
         "status": c.get("status", "scored"),
         "control_leak_wilson_ub": c.get("control_leak_wilson_ub"),
         "v2_leak_wilson_ubs": [r["leak_rate_wilson_ub"] for r in c.get("v2_seeds", [])],
         "all_under_threshold": all(r["leak_ub_under_threshold"] for r in c.get("v2_seeds", [])) if c.get("v2_seeds") else None,
         "any_dissociation_break": c.get("any_dissociation_break")}
        for c in realskeleton_cells
    ]

    scorecard = {
        "issue": "RES-19 (re-score of KLU-106 v2 seeds)" if heldout_general_override else "KLU-106",
        "schema": 3,
        "heldout_general_dir": heldout_dir,
        "heldout_general_override": bool(heldout_general_override),
        "control_model": control,
        "control_role": "zero-shot KLU-51 kp-deid scored on the SAME held-out set (NOT the published 0.46-0.52)",
        "v2_seed_models": v2_models,
        "leak_wilson_ub_threshold": LEAK_WILSON_UB_THRESHOLD,
        "bootstrap": {"iters": BOOTSTRAP_ITERS, "seed": BOOTSTRAP_SEED, "ci": "95% percentile, paired by document"},
        "carve_out_from_manifest": man.get("carve_out"),
        "post_downsample_subject_disjoint": man.get("post_downsample_subject_disjoint"),
        "post_downsample_subject_intersection_per_language": man.get("post_downsample_subject_intersection_per_language"),
        "identifier_surface_form_holdout": man.get("identifier_surface_form_holdout"),
        "general_heldout_cells": general_cells,
        "real_skeleton_cells": realskeleton_cells,
        "headline_f1_delta_per_language": headline,
        "reid_leak_summary": leak_summary,
        "guards": {
            "model_card_status": "dev",
            "no_sota_or_best_protector_or_validated_claim": True,
            "validated_gated_on": "KLU-27",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote scorecard -> %s", out)
    click.echo(json.dumps({"headline_f1_delta_per_language": headline,
                           "reid_leak_summary": leak_summary}, indent=2))


if __name__ == "__main__":
    main()
