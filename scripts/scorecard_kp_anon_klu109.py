#!/usr/bin/env python3
"""KLU-109 — held-out privacy-utility frontier scorecard for kp-anon vs the redaction baseline.

Acceptance is "produce the rigorous scorecard + a committed Pareto-frontier figure", NOT "train a
great model" (KLU-109). This scores, on a FIXED held-out eval set (the EuroPriv-Bench ro/pl
real-skeleton Track-C configs — gold offsets, never re-detected on the output), TWO anonymizers that
share the SAME trained detector so the comparison isolates the *anonymization policy*:

  * **control = the redaction baseline** (KLU-104): the trained kp-anon detector used as a plain
    redactor — ``europriv_bench.adapters.KpModelAdapter.anonymize`` blanket-masks every detected span
    with the ``█`` placeholder.
  * **kp-anon** (KLU-109): the SAME detector wrapped with the pseudonymization policy
    (``klusai.privacy.models.anon.KpAnonAdapter``) — detected spans become type-consistent surrogates.

Because both use the identical detector, the **privacy axis** (``redaction_leakage.leak_rate``, read
from gold offsets) is the SAME for the two at equal recall — that is the honest, load-bearing point:
pseudonymization does not cost privacy. The **utility axis** is where they separate:

  * ``information_retention`` (↑ non-PII tokens preserved) and ``1 − structural_disruption.mask_token_ratio``
    (↑ less mask-glyph fragmentation). The redaction baseline shreds the document with ``█`` masks;
    kp-anon keeps it readable/joinable.

Rigor (the program's hard guards):
  * **Held-out** fixed eval set; numbers are **bootstrap-CI'd** (paired, by document) deltas of the
    utility proxy ``kp-anon − redaction-baseline``, and the leak carries its 95% Wilson CI.
  * **>=2-seed variance** — every completed training seed is scored; a utility gain counts as
    clearing seed-noise only if the paired Δ CI excludes 0 for **every** seed AND the per-seed point
    deltas agree in sign. (A gain inside seed-noise is not a gain.)
  * Dominance on the privacy-utility plane = leak not worse (within CI) AND utility strictly better
    (Δ CI excludes 0), reported PER frontier language.
  * config_status=dev; NO SOTA/best/validated claim (gated on KLU-27); every cell contamination- and
    status-labelled.

Run from the repo root after ``scripts/train_kp_anon_klu109.py``:

    python scripts/scorecard_kp_anon_klu109.py \
        --manifest runs/klu109-kp-anon-train-manifest.json \
        --suite ../europriv-bench/evaluations \
        --out runs/klu109-kp-anon-scorecard.json
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import click

from klusai.privacy.models.logger import get_logger

logger = get_logger("scorecard_kp_anon_klu109")

BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 12345

# Track-C anonymization specs that carry decode-bearing national IDs → a non-trivial privacy axis.
# (ro published as anonymization-ro-realskeleton.yaml; pl added as anonymization-pl-realskeleton.yaml.)
FRONTIER_SPECS = ["anonymization-ro-realskeleton.yaml", "anonymization-pl-realskeleton.yaml"]


def _per_doc_info_retention(rows, redacted):
    """Per-document non-PII-token retention counts ``(retained, total)`` — sums to the corpus metric.

    Lets a paired document bootstrap recompute ``information_retention`` as a ratio of resampled sums
    (identical to recomputing the metric on the resample, but O(n) per iteration)."""
    from europriv_bench.metrics import _non_pii_token_mask
    from europriv_bench.spans import whitespace_tokens

    out = []
    for i, row in enumerate(rows):
        text = row["text"]
        toks = whitespace_tokens(text)
        keep = _non_pii_token_mask(text, row.get("spans", []))
        red = redacted[i] if i < len(redacted) else ""
        out_counts: dict[str, int] = {}
        for tok, _, _ in whitespace_tokens(red):
            out_counts[tok] = out_counts.get(tok, 0) + 1
        retained = total = 0
        for (tok, _, _), is_keep in zip(toks, keep):
            if not is_keep:
                continue
            total += 1
            if out_counts.get(tok, 0) > 0:
                retained += 1
                out_counts[tok] -= 1
        out.append((retained, total))
    return out


def _per_doc_mask_counts(rows, redacted):
    """Per-document ``(mask_tokens, output_tokens)`` — sums to ``structural_disruption.mask_token_ratio``."""
    from europriv_bench.metrics import structural_disruption  # reuse its _is_mask via a 1-doc call
    from europriv_bench.spans import whitespace_tokens

    out = []
    for i, row in enumerate(rows):
        red = redacted[i] if i < len(redacted) else ""
        # Reuse the exact mask classifier by scoring this single doc.
        single = structural_disruption([row], [red])
        out.append((int(single["mask_tokens"]), len(whitespace_tokens(red))))
    return out


def _bootstrap_ratio_delta_ci(a_counts, b_counts, *, iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    """Paired document bootstrap CI for ratio(a) − ratio(b), each ratio = sum(num)/sum(den).

    ``a_counts``/``b_counts`` are per-document ``(num, den)`` lists over the SAME documents (paired:
    each resample draws the same indices for both). Returns ``(delta, low, high)`` 95% percentile."""
    n = len(a_counts)
    rng = random.Random(seed)

    def ratio_at(idx, counts):
        num = sum(counts[i][0] for i in idx)
        den = sum(counts[i][1] for i in idx)
        return (num / den) if den else 0.0

    full = range(n)
    base = ratio_at(full, a_counts) - ratio_at(full, b_counts)
    deltas = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(ratio_at(idx, a_counts) - ratio_at(idx, b_counts))
    deltas.sort()
    lo = deltas[int(0.025 * iters)]
    hi = deltas[min(iters - 1, int(0.975 * iters))]
    return base, lo, hi


def _score_one_spec(spec_path, seed_models, suite):
    """Frontier cell for one Track-C spec: leak (shared detector) + bootstrap-CI'd utility Δ per seed."""
    from europriv_bench.adapters import KpModelAdapter
    from europriv_bench.leaderboard import classify_contamination
    from europriv_bench.metrics import (
        information_retention,
        pseudonymization_consistency,
        redaction_leakage,
        structural_disruption,
    )
    from europriv_bench.runner import ConfigUnavailableError, _load_gold_rows
    from europriv_bench.spec import EvalSpec
    from klusai.privacy.models.anon import KpAnonAdapter

    spec = EvalSpec.from_yaml(Path(suite) / spec_path)
    try:
        rows = _load_gold_rows(spec)
    except ConfigUnavailableError as e:
        logger.warning("skip-and-report %s: %s", spec.name, e)
        return {"track": "track-c-frontier", "spec": spec.name,
                "config": spec.dataset.config, "status": "unavailable_on_hub_skipped"}
    texts = [r["text"] for r in rows]

    seed_cells = []
    for sd, mid in seed_models.items():
        det = KpModelAdapter(model_id=mid)          # SAME detector, blanket-mask policy = redaction baseline
        anon = KpAnonAdapter(model_id=mid)          # SAME detector, pseudonymization policy = kp-anon

        red_baseline = det.anonymize(texts)
        red_anon = anon.anonymize(texts)
        maps_anon = anon.pseudonymize(texts)

        leak_baseline = redaction_leakage(rows, red_baseline)
        leak_anon = redaction_leakage(rows, red_anon)
        ir_baseline = information_retention(rows, red_baseline)
        ir_anon = information_retention(rows, red_anon)
        sd_baseline = structural_disruption(rows, red_baseline)
        sd_anon = structural_disruption(rows, red_anon)
        bij_anon = pseudonymization_consistency(rows, maps_anon)

        # Paired bootstrap Δ(kp-anon − baseline) for the two utility proxies.
        ir_a = _per_doc_info_retention(rows, red_anon)
        ir_b = _per_doc_info_retention(rows, red_baseline)
        d_ir, ir_lo, ir_hi = _bootstrap_ratio_delta_ci(ir_a, ir_b)

        # Utility on the mask axis = (1 − mask_token_ratio); Δ(anon − baseline) of that = baseline_mask − anon_mask.
        mk_a = _per_doc_mask_counts(rows, red_anon)
        mk_b = _per_doc_mask_counts(rows, red_baseline)
        d_mask_ratio, mlo, mhi = _bootstrap_ratio_delta_ci(mk_a, mk_b)  # anon_maskratio − baseline_maskratio
        d_unmask = -d_mask_ratio                                        # utility ↑ = less masking

        leak_same = abs(leak_anon["leak_rate"] - leak_baseline["leak_rate"]) < 1e-12
        ir_gain_ci_excl0 = bool(ir_lo > 0 or ir_hi < 0)
        unmask_gain_ci_excl0 = bool(mlo > 0 or mhi < 0)  # the mask-ratio Δ CI excluding 0
        seed_cells.append({
            "seed": sd,
            "model_id": mid,
            # Privacy axis — identical at equal recall (the load-bearing honest point).
            "leak_rate_baseline": round(leak_baseline["leak_rate"], 5),
            "leak_rate_kp_anon": round(leak_anon["leak_rate"], 5),
            "leak_rate_wilson_ub_kp_anon": round(leak_anon["leak_rate_ci_high"], 5),
            "leak_unchanged_by_policy": leak_same,
            "subjects_total": leak_anon["subjects_total"],
            # Utility axis — information retention (token preservation).
            "info_retention_baseline": round(ir_baseline["information_retention"], 4),
            "info_retention_kp_anon": round(ir_anon["information_retention"], 4),
            "info_retention_delta": round(d_ir, 4),
            "info_retention_delta_ci95_low": round(ir_lo, 4),
            "info_retention_delta_ci95_high": round(ir_hi, 4),
            "info_retention_gain_ci_excludes_0": ir_gain_ci_excl0,
            # Utility axis — mask-token ratio (lower = less structural disruption).
            "mask_token_ratio_baseline": round(sd_baseline["mask_token_ratio"], 4),
            "mask_token_ratio_kp_anon": round(sd_anon["mask_token_ratio"], 4),
            "unmasking_utility_delta": round(d_unmask, 4),                # (1−mask)_anon − (1−mask)_base
            "mask_ratio_delta_ci95_low": round(mlo, 4),
            "mask_ratio_delta_ci95_high": round(mhi, 4),
            "mask_ratio_gain_ci_excludes_0": unmask_gain_ci_excl0,
            # Pseudonymization consistency (bijection).
            "bijection_in_doc": round(bij_anon["in_doc_bijection_rate"], 4),
            "bijection_cross_doc": round(bij_anon["cross_doc_bijection_rate"], 4),
            # Dominance: privacy not worse (equal leak) AND utility strictly better (mask Δ CI excl 0).
            "dominates_baseline": bool(leak_same and unmask_gain_ci_excl0 and d_unmask > 0),
        })

    # >=2-seed aggregate.
    unmask_deltas = [c["unmasking_utility_delta"] for c in seed_cells]
    ir_deltas = [c["info_retention_delta"] for c in seed_cells]
    all_dominate = all(c["dominates_baseline"] for c in seed_cells)
    # A gain clears seed-noise iff every seed's mask-ratio Δ CI excludes 0 AND all point deltas agree in sign.
    unmask_clears_noise = (
        len(seed_cells) >= 2
        and all(c["mask_ratio_gain_ci_excludes_0"] for c in seed_cells)
        and (all(d > 0 for d in unmask_deltas) or all(d < 0 for d in unmask_deltas))
    )

    return {
        "track": "track-c-frontier",
        "spec": spec.name,
        "config": spec.dataset.config,
        "languages": spec.languages,
        "n": len(rows),
        "control_role": "redaction baseline (KLU-104) = SAME kp-anon detector, blanket-mask policy",
        "privacy_metric": "redaction_leakage.leak_rate (gold-offset, Wilson CI)",
        "utility_metrics": ["information_retention (↑)", "1 − structural_disruption.mask_token_ratio (↑)"],
        "contamination": classify_contamination("kp-anon", spec.dataset.config),
        "config_status": "dev",
        "seeds": seed_cells,
        "unmasking_utility_delta_min": round(min(unmask_deltas), 4) if unmask_deltas else None,
        "unmasking_utility_delta_max": round(max(unmask_deltas), 4) if unmask_deltas else None,
        "info_retention_delta_min": round(min(ir_deltas), 4) if ir_deltas else None,
        "info_retention_delta_max": round(max(ir_deltas), 4) if ir_deltas else None,
        "all_seeds_dominate_baseline": all_dominate,
        "utility_gain_clears_seed_noise": unmask_clears_noise,
    }


@click.command()
@click.option("--manifest", default="runs/klu109-kp-anon-train-manifest.json")
@click.option("--suite", default="../europriv-bench/evaluations")
@click.option("--out", default="runs/klu109-kp-anon-scorecard.json")
@click.option("--threads", type=int, default=4)
def main(manifest, suite, out, threads):
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass
    os.environ.setdefault("EUROPRIV_DEVICE", "cpu")  # scoring on CPU is fine + deterministic-ish

    man = json.loads(Path(manifest).read_text())
    seed_models = {a["seed"]: a["output_dir"] for a in man.get("seed_artifacts", [])}
    if not seed_models:
        raise click.UsageError("manifest has no completed seed_artifacts to score.")
    logger.info("scoring frontier for kp-anon seeds %s", sorted(seed_models))

    cells = [_score_one_spec(sp, seed_models, suite) for sp in FRONTIER_SPECS]

    frontier_langs = sorted({lg for c in cells if c.get("languages") for lg in c["languages"]})
    dominated = [c for c in cells if c.get("all_seeds_dominate_baseline")]
    headline = {
        "frontier_languages_scored": frontier_langs,
        "n_languages_kp_anon_dominates_redaction_baseline_all_seeds": len(dominated),
        "dominates_on": [c["config"] for c in dominated],
        "per_config": {
            c.get("config"): {
                "status": c.get("status", "scored"),
                "leak_unchanged_by_policy": all(s["leak_unchanged_by_policy"] for s in c.get("seeds", []))
                if c.get("seeds") else None,
                "unmasking_utility_delta_min": c.get("unmasking_utility_delta_min"),
                "unmasking_utility_delta_max": c.get("unmasking_utility_delta_max"),
                "all_seeds_dominate": c.get("all_seeds_dominate_baseline"),
                "utility_gain_clears_seed_noise": c.get("utility_gain_clears_seed_noise"),
            }
            for c in cells
        },
    }

    scorecard = {
        "issue": "KLU-109",
        "model": "kp-anon-mdeberta-280m",
        "model_kind": "span-replacement anonymizer (LoRA mDeBERTa detector + pseudonymization policy)",
        "schema": 1,
        "control_role": "redaction baseline (KLU-104): SAME trained detector, blanket-mask anonymize",
        "seed_models": seed_models,
        "bootstrap": {"iters": BOOTSTRAP_ITERS, "seed": BOOTSTRAP_SEED,
                      "ci": "95% percentile, paired by document"},
        "train_manifest": {
            "device": man.get("device"),
            "bounds": man.get("bounds"),
            "seeds_completed": man.get("seeds_completed"),
            "total_wall_minutes": man.get("total_wall_minutes"),
            "stopped_early": man.get("stopped_early"),
            "utilization": man.get("utilization"),
            "max_util_profile": man.get("max_util_profile"),
            "balanced_train_rows": man.get("balanced_train_rows"),
        },
        "frontier_cells": cells,
        "headline": headline,
        "guards": {
            "config_status": "dev",
            "no_sota_or_best_or_validated_claim": True,
            "validated_gated_on": "KLU-27",
            "privacy_axis_attributable_to_detection_recall": True,
            "substitution_introduces_zero_leak_by_construction": True,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote scorecard -> %s", out)
    click.echo(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
