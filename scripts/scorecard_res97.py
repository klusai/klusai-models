#!/usr/bin/env python3
"""RES-97 — held-out scorecard: kp-deid-xlmr-560m vs kp-deid-mdeberta-280m (and the board).

The question RES-97 asks: does the 560M XLM-R encoder beat the 280M mDeBERTa AND/OR close the
detection-F1 gap to GLiNER, on the DISCRIMINATING evals (NOT the saturating general split, which
pins at F1=1.000 — RES-19)? This scores every model on the SAME held-out surfaces and reports the
deltas HONESTLY with bootstrap CIs. A null/modest result is fine and reportable.

Discriminating evals (each contamination-labelled):
  * **RES-19 hard-general** (clean held-out, our taxonomy, template+subject-disjoint from training):
    the de-saturated synthetic general set; per-language entity-F1 + paired bootstrap Δ CI.
  * **Ai4Privacy openpii** (RES-93, EXTERNAL synthetic — Ai4Privacy's LLM generator, NOT ours):
    per-language entity-F1. openmed/tabularisai trained on Ai4Privacy → in_distribution for them.
  * **TAB ECHR legal en** (RES-89, REAL peer-reviewed gold): the cross-domain real-data anchor.
  * **Real-skeleton** (ro/pl/it, RES-89/CITABLE-gated): entity-F1 + national-ID re-id leak (Wilson UB).

For every detection cell we score: xlmr-560m (each seed) and the mdeberta-280m comparand, plus any
baseline adapters available on this machine (gliner, etc. — skipped cleanly if the extra isn't
installed). The headline Δ = paired bootstrap CI of F1(xlmr-560m seed) − F1(mdeberta-280m) on the
SAME rows; a real win requires the Δ CI to exclude 0 across BOTH seeds. We reuse the KLU-106
bootstrap utilities verbatim (paired-by-document, micro-F1 over precomputed entity counts).

config_status=dev for synthetic; TAB is real-external-gold. No citable claim is made (RES-77 gate).
All numbers come from real runs against the trained merged checkpoints.

Run from the repo root after training (scripts/train_xlmr560m_res97.py):

    python scripts/scorecard_res97.py \
        --manifest runs/res97-train-manifest.json \
        --comparand runs/kp-deid-mdeberta-280m-v2-seed0 \
        --hard-general runs/res19-heldout-hard-general \
        --suite ../europriv-bench/evaluations \
        --out runs/res97-scorecard.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import click

# Reuse the KLU-106 bootstrap machinery verbatim so the CI is computed identically to the 280M run.
from scorecard_klu106 import (  # type: ignore  # noqa: E402  (sibling script, same dir on sys.path)
    BOOTSTRAP_ITERS,
    BOOTSTRAP_SEED,
    LEAK_WILSON_UB_THRESHOLD,
    _bootstrap_delta_ci,
    _entity_f1_on_rows,
    _mask,
    _mask_gold,
    _predict,
)

from klusai.privacy.models.logger import get_logger

logger = get_logger("scorecard_res97")

# Baseline adapters to also place on the board if their optional extra is importable on this Mac.
# Each is attempted with a clean skip-and-report if the extra/model isn't present (no hard dep).
OPTIONAL_BASELINES = ("gliner",)


def _f2(prec: float, rec: float) -> float:
    return 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0


def _score_detection_set(rows, label, contamination, config_status, xlmr_models, comparand,
                         baselines):
    """Score one detection eval (a list of {text, spans} rows) for every model, with Δ CIs.

    ``xlmr_models`` = {seed: dir}; ``comparand`` = the mdeberta-280m dir; ``baselines`` = {name: adapter}.
    Returns a cell dict with per-model F1/F2, and per-seed paired bootstrap Δ(xlmr − comparand).
    """
    from europriv_bench.runner import _rows_to_gold

    texts = [r["text"] for r in rows]
    _, gold_tags = _rows_to_gold(rows)
    eval_labels = sorted({t.split("-", 1)[1] for seq in gold_tags for t in seq if t != "O"})
    gold_masked = _mask_gold(gold_tags, eval_labels)

    def _f1f2(pred):
        m = _entity_f1_on_rows(rows, pred)
        return round(m["f1"], 4), round(_f2(m.get("precision", 0.0), m.get("recall", 0.0)), 4)

    # Comparand (mdeberta-280m) predictions on the SAME rows.
    comparand_pred = _mask(_predict(comparand, texts), eval_labels)
    comparand_f1, comparand_f2 = _f1f2(comparand_pred)

    # Each xlmr-560m seed.
    seed_cells = []
    seed_f1s = []
    for sd, mid in xlmr_models.items():
        pred = _mask(_predict(mid, texts), eval_labels)
        f1, f2 = _f1f2(pred)
        d, lo, hi = _bootstrap_delta_ci(gold_masked, pred, comparand_pred)
        seed_f1s.append(f1)
        seed_cells.append({
            "seed": sd, "model_id": mid, "f1": f1, "f2": f2,
            "delta_f1_vs_comparand": round(d, 4),
            "delta_ci95_low": round(lo, 4), "delta_ci95_high": round(hi, 4),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "beats_comparand": bool(lo > 0),
        })

    # Optional baselines (gliner, ...) for board context (no Δ CI — just F1/F2 reference points).
    baseline_cells = []
    for name, adapter in baselines.items():
        try:
            pred = _mask(adapter.predict_tags(texts), eval_labels)
            f1, f2 = _f1f2(pred)
            baseline_cells.append({"name": name, "model_id": adapter.model_id, "f1": f1, "f2": f2})
        except Exception as e:  # extra missing / model download failure — skip-and-report
            baseline_cells.append({"name": name, "status": f"skipped: {str(e).splitlines()[0][:120]}"})

    all_seeds_beat = bool(seed_cells) and all(c["beats_comparand"] for c in seed_cells)
    return {
        "track": label,
        "n": len(rows),
        "eval_labels": eval_labels,
        "contamination": contamination,
        "config_status": config_status,
        "comparand_model": comparand,
        "comparand_f1": comparand_f1,
        "comparand_f2": comparand_f2,
        "xlmr_560m_seeds": seed_cells,
        "xlmr_560m_f1_mean": round(sum(seed_f1s) / len(seed_f1s), 4) if seed_f1s else None,
        "xlmr_560m_f1_min": round(min(seed_f1s), 4) if seed_f1s else None,
        "xlmr_560m_f1_max": round(max(seed_f1s), 4) if seed_f1s else None,
        "all_seeds_beat_comparand_ci": all_seeds_beat,
        "baselines": baseline_cells,
    }


def _hard_general_cells(hard_general_dir, xlmr_models, comparand, baselines, langs_filter):
    from datasets import load_from_disk

    ds = load_from_disk(hard_general_dir)
    by_lang: dict[str, list[int]] = {}
    for i in range(ds.num_rows):
        by_lang.setdefault(ds[i]["language"], []).append(i)

    cells = []
    for lang in sorted(by_lang):
        if langs_filter and lang not in langs_filter:
            continue
        rows = [ds[i] for i in by_lang[lang]]
        cell = _score_detection_set(
            rows, f"hard-general:{lang}", "clean_held_out", "dev",
            xlmr_models, comparand, baselines,
        )
        cell["language"] = lang
        cell["contamination_basis"] = "RES-19 hard-general: template- AND subject-disjoint from training"
        cells.append(cell)
        logger.info("hard-general %s: comparand_f1=%.4f xlmr_mean=%s all_seeds_beat=%s",
                    lang, cell["comparand_f1"], cell["xlmr_560m_f1_mean"], cell["all_seeds_beat_comparand_ci"])
    return cells


def _hf_detection_cells(suite, glob, track_prefix, xlmr_models, comparand, baselines):
    """Score every detection spec matching ``glob`` (ai4privacy openpii, TAB) via the harness loader."""
    from europriv_bench.runner import ConfigUnavailableError, _load_gold_rows
    from europriv_bench.spec import EvalSpec, Task

    cells = []
    for path in sorted(Path(suite).glob(glob)):
        spec = EvalSpec.from_yaml(path)
        if spec.task is not Task.DETECTION:
            continue
        try:
            rows = _load_gold_rows(spec)
        except ConfigUnavailableError as e:
            logger.warning("skip-and-report %s: %s", spec.name, e)
            cells.append({"track": f"{track_prefix}:{spec.dataset.config}", "spec": spec.name,
                          "config": spec.dataset.config, "status": "unavailable_on_hub_skipped"})
            continue
        from europriv_bench.leaderboard import classify_contamination
        from europriv_bench.runner import config_status_for

        contamination = classify_contamination("kp-model", spec.dataset.config)
        cell = _score_detection_set(
            rows, f"{track_prefix}:{spec.dataset.config}", contamination,
            config_status_for(spec.dataset.config), xlmr_models, comparand, baselines,
        )
        cell["spec"] = spec.name
        cell["config"] = spec.dataset.config
        cell["languages"] = spec.languages
        cells.append(cell)
        logger.info("%s: comparand_f1=%.4f xlmr_mean=%s contamination=%s",
                    spec.dataset.config, cell["comparand_f1"], cell["xlmr_560m_f1_mean"], contamination)
    return cells


def _realskeleton_cells(suite, xlmr_models, comparand, baselines):
    """Real-skeleton detection tracks: F1 + national-ID re-id leak (Wilson UB), side-by-side."""
    from europriv_bench.adapters import KpModelAdapter
    from europriv_bench.runner import ConfigUnavailableError, run_spec
    from europriv_bench.spec import EvalSpec, Task

    ts = datetime.now(timezone.utc).isoformat()
    cells = []
    for path in sorted(Path(suite).glob("*realskeleton*.yaml")):
        spec = EvalSpec.from_yaml(path)
        if spec.task is not Task.DETECTION:
            continue
        models = {"comparand": comparand, **{f"xlmr_seed{sd}": mid for sd, mid in xlmr_models.items()}}
        per_model = {}
        unavailable = False
        for tag, mid in models.items():
            try:
                per_model[tag] = run_spec(spec, KpModelAdapter(model_id=mid), timestamp=ts)
            except ConfigUnavailableError as e:
                logger.warning("skip-and-report %s: %s", spec.name, e)
                unavailable = True
                break
        if unavailable:
            cells.append({"track": "real-skeleton", "spec": spec.name,
                          "config": spec.dataset.config, "status": "unavailable_on_hub_skipped"})
            continue

        leak_key = "cnp_leakage" if "cnp_leakage" in spec.metrics else "national_id_leakage"
        cmp = per_model["comparand"]
        cmp_f1 = cmp["scores"]["entity_f1"]["f1"]
        cmp_leak = cmp["scores"].get(leak_key, {})
        seed_rows = []
        for sd in xlmr_models:
            r = per_model[f"xlmr_seed{sd}"]
            f1 = r["scores"]["entity_f1"]["f1"]
            leak = r["scores"].get(leak_key, {})
            ub = leak.get("leak_rate_ci_high", 0.0)
            leak_delta = leak.get("leak_rate", 0.0) - cmp_leak.get("leak_rate", 0.0)
            seed_rows.append({
                "seed": sd, "model_id": xlmr_models[sd],
                "f1": round(f1, 4), "f1_gain_vs_comparand": round(f1 - cmp_f1, 4),
                "leak_rate": round(leak.get("leak_rate", 0.0), 5),
                "leak_rate_wilson_ub": round(ub, 5),
                "leak_rate_delta_vs_comparand": round(leak_delta, 5),
                "subjects_total": leak.get("decode_bearing_total", leak.get("cnp_total")),
                "leak_ub_under_threshold": bool(ub <= LEAK_WILSON_UB_THRESHOLD),
                "dissociation_break": bool((f1 - cmp_f1) > 0 and leak_delta > 1e-9),
            })
        cells.append({
            "track": "real-skeleton",
            "spec": spec.name,
            "config": spec.dataset.config,
            "languages": spec.languages,
            "leak_metric": leak_key,
            "contamination": cmp.get("contamination"),
            "config_status": cmp.get("config_status", "dev"),
            "n": cmp["n"],
            "comparand_f1": round(cmp_f1, 4),
            "comparand_leak_rate": round(cmp_leak.get("leak_rate", 0.0), 5),
            "comparand_leak_wilson_ub": round(cmp_leak.get("leak_rate_ci_high", 0.0), 5),
            "leak_wilson_ub_threshold": LEAK_WILSON_UB_THRESHOLD,
            "xlmr_560m_seeds": seed_rows,
            "any_dissociation_break": any(r["dissociation_break"] for r in seed_rows),
        })
        logger.info("real-skeleton %s: comparand_f1=%.4f xlmr_f1=%s", spec.dataset.config, cmp_f1,
                    [r["f1"] for r in seed_rows])
    return cells


def _build_baselines(names):
    """Instantiate optional baseline adapters; skip cleanly if the extra isn't installed."""
    from europriv_bench.adapters import build

    out = {}
    for name in names:
        try:
            out[name] = build(name)
        except Exception as e:
            logger.warning("baseline %s unavailable, skipping: %s", name, str(e).splitlines()[0][:120])
    return out


@click.command()
@click.option("--manifest", default="runs/res97-train-manifest.json",
              help="Training manifest from scripts/train_xlmr560m_res97.py (seed dirs).")
@click.option("--comparand", default="runs/kp-deid-mdeberta-280m-v2-seed0",
              help="kp-deid-mdeberta-280m checkpoint dir/id (the model we ask the 560M to beat).")
@click.option("--hard-general", "hard_general_dir", default="runs/res19-heldout-hard-general",
              help="RES-19 hard held-out general dataset dir (de-saturated synthetic).")
@click.option("--suite", default="../europriv-bench/evaluations")
@click.option("--out", default="runs/res97-scorecard.json")
@click.option("--langs", default=None, help="Comma list to restrict hard-general languages (debug).")
@click.option("--baselines", "baseline_names", default=",".join(OPTIONAL_BASELINES),
              help="Comma list of optional baseline adapters to place on the board (skipped if absent).")
@click.option("--skip-realskeleton", is_flag=True, default=False)
@click.option("--skip-baselines", is_flag=True, default=False,
              help="Skip the optional baselines entirely (faster; xlmr-vs-mdeberta only).")
@click.option("--threads", type=int, default=4)
def main(manifest, comparand, hard_general_dir, suite, out, langs, baseline_names,
         skip_realskeleton, skip_baselines, threads):
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass
    os.environ.setdefault("EUROPRIV_DEVICE", "cpu")  # deterministic CPU scoring

    man = json.loads(Path(manifest).read_text())
    xlmr_models = {a["seed"]: a["output_dir"] for a in man.get("seed_artifacts", [])}
    if not xlmr_models:
        raise click.UsageError("manifest has no completed seed_artifacts to score.")
    logger.info("scoring xlmr-560m seeds %s vs comparand %s", sorted(xlmr_models), comparand)

    langs_filter = set(langs.split(",")) if langs else None
    baselines = {} if skip_baselines else _build_baselines(
        [b for b in baseline_names.split(",") if b]
    )

    hard_general = _hard_general_cells(hard_general_dir, xlmr_models, comparand, baselines, langs_filter)
    ai4privacy = _hf_detection_cells(
        suite, "pii-detection-ai4privacy-openpii-*.yaml", "ai4privacy-openpii",
        xlmr_models, comparand, baselines,
    )
    tab = _hf_detection_cells(
        suite, "pii-detection-tab-*.yaml", "tab", xlmr_models, comparand, baselines,
    )
    realskeleton = [] if skip_realskeleton else _realskeleton_cells(
        suite, xlmr_models, comparand, baselines
    )

    # Headline: per discriminating track, does xlmr-560m beat mdeberta-280m with the Δ CI excluding 0?
    def _headline(cell):
        return {
            "track": cell.get("track"),
            "comparand_f1": cell.get("comparand_f1"),
            "xlmr_f1_mean": cell.get("xlmr_560m_f1_mean"),
            "xlmr_f1_min": cell.get("xlmr_560m_f1_min"),
            "xlmr_f1_max": cell.get("xlmr_560m_f1_max"),
            "all_seeds_beat_comparand_ci": cell.get("all_seeds_beat_comparand_ci"),
            "contamination": cell.get("contamination"),
            "config_status": cell.get("config_status"),
        }

    detection_cells = hard_general + ai4privacy + tab
    headline = [_headline(c) for c in detection_cells if "xlmr_560m_seeds" in c]

    scorecard = {
        "issue": "RES-97",
        "schema": 3,
        "question": "Does kp-deid-xlmr-560m beat kp-deid-mdeberta-280m and/or close the F1 gap to "
                    "GLiNER, on the discriminating evals (NOT the saturating general split)?",
        "comparand_model": comparand,
        "comparand_role": "kp-deid-mdeberta-280m (the 280M to beat), scored on the SAME held-out rows",
        "xlmr_560m_seed_models": xlmr_models,
        "training_manifest": manifest,
        "base_model": man.get("base_model"),
        "device": man.get("device"),
        "total_wall_minutes": man.get("total_wall_minutes"),
        "seed_artifacts": man.get("seed_artifacts"),
        "leak_wilson_ub_threshold": LEAK_WILSON_UB_THRESHOLD,
        "bootstrap": {"iters": BOOTSTRAP_ITERS, "seed": BOOTSTRAP_SEED,
                      "ci": "95% percentile, paired by document"},
        "hard_general_cells": hard_general,
        "ai4privacy_openpii_cells": ai4privacy,
        "tab_cells": tab,
        "real_skeleton_cells": realskeleton,
        "headline_per_track": headline,
        "guards": {
            "model_card_status": "dev",
            "no_citable_or_sota_claim": True,
            "citable_gated_on": "RES-77",
            "contamination_labelled": True,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote scorecard -> %s", out)
    click.echo(json.dumps({"headline_per_track": headline,
                           "real_skeleton": [
                               {"config": c.get("config"), "status": c.get("status", "scored"),
                                "comparand_f1": c.get("comparand_f1"),
                                "xlmr_f1": [r["f1"] for r in c.get("xlmr_560m_seeds", [])],
                                "any_dissociation_break": c.get("any_dissociation_break")}
                               for c in realskeleton]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
