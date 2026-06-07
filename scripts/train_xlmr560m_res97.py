#!/usr/bin/env python3
"""RES-97 — kp-deid-xlmr-560m: a more capable multilingual detector, trained on the Mac (MPS/LoRA).

RES-63/RES-53 assumed XLM-R-560m needs a GPU burst. It does not: a 560M encoder + LoRA trains
comfortably on the 96 GB M3 Ultra via PyTorch-MPS — we already train the 280M mDeBERTa on MPS, this
is ~2x params, well within unified memory. The only cost is wall-clock, not capability.

This is the SAME ``xlmr-ner`` token-classification path (transformers+peft LoRA, transformers==4.57.6,
NOT the MoE/transformers>=5.9 path), reusing the load-bearing KLU-106 contamination carve-out and
KLU-54 template-disjoint in-training eval. The ONE architecture difference vs the 280M run: XLM-R
(RoBERTa) names its attention projections ``query``/``key``/``value`` rather than mDeBERTa's
``*_proj`` — handled by ``lora_target_modules`` so the LoRA adapter is non-empty (RES-97).

It is deliberately a thin wrapper over the KLU-106 trainer so the two members of the family are
trained by identical machinery (same carve, same disjoint eval, same MPS profile, same manifest
shape) and are therefore comparable on the held-out scorecard. Scoring is a separate step
(``scripts/scorecard_res97.py``).

Hard runaway bounds (same as KLU-106): ``--max-samples`` total balanced; ``--epochs`` <= 3;
``--wall-clock-stop`` seconds with stop-and-report (never start a seed that cannot finish; a partial
scorecard is still produced). >=2 seeds for the headline-delta variance.

Run from the repo root after ``make install``:

    python scripts/train_xlmr560m_res97.py --device mps --epochs 3 --seeds 0 1 \
        --max-samples 40000 --wall-clock-stop 25200
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click

from klusai.privacy.models.logger import get_logger
from klusai.privacy.models.training.token_classification import (
    MAX_UTIL_BATCH_CANDIDATES,
    SUBJECT_LABELS,
    _bioes_from_spans,
    _subject_surface_forms,
    carve_heldout_general,
    identifier_surface_form_holdout,
    lora_target_modules,
    resolve_device,
    resolve_max_util_profile,
    template_disjoint_split,
)
from klusai.privacy.models.util_sampler import GpuUtilSampler, peak_mps_mem_gb

logger = get_logger("train_xlmr560m_res97")

# Same 8 languages as the KLU-106 280M run so the two models are compared on identical data.
LANGUAGES = ["ro", "en", "pl", "de", "fr", "es", "it", "nl"]
DATASET_TMPL = "klusai/ds-kp-general-{lang}-50k"


def _load_all(seed: int):
    from datasets import concatenate_datasets, load_dataset

    parts = []
    for lang in LANGUAGES:
        # Load a capped per-language pool, not the full 50k. The carve_heldout_general pass is
        # O(pool) Python (template_skeleton + nested-spans materialization over every row); the full
        # 8x50k=400k pool made it grind 11+ min single-core before training. We only down-sample a
        # balanced <=40k out of this pool anyway, and the ~6 generator templates/language are all
        # present in 10k, so a 10k/lang (80k) pool is scientifically equivalent for the carve +
        # balanced draw while making prep ~5x faster.
        d = load_dataset(DATASET_TMPL.format(lang=lang), split="train[:10000]")
        keep = [c for c in d.column_names if c in ("text", "spans", "language", "domain")]
        d = d.select_columns(keep)
        parts.append(d)
        logger.info("loaded %s: %d rows", lang, d.num_rows)
    # NOTE: do NOT ``.shuffle(seed)`` the full 400k corpus here. A shuffle installs an Arrow
    # indices-mapping, and every subsequent column read (``rows["text"]`` / the nested ``rows["spans"]``)
    # then gathers per-row through that permutation — pathologically slow on 400k nested rows (it hung
    # data-prep for 11+ min at 100% one core, never reaching training). Order is irrelevant to every
    # correctness invariant downstream: the carve/template split are deterministic seeded *hashes* (not
    # positional), and ``_balanced_downsample`` does its own per-language seeded RNG pick + shuffle. The
    # HF ``Trainer`` also reshuffles each epoch. So we keep the corpus contiguous for fast column reads.
    merged = concatenate_datasets(parts)
    logger.info("merged 8-language corpus: %d rows", merged.num_rows)
    return merged


def _balanced_downsample(rows, *, max_samples: int, language_key: str, seed: int):
    import collections
    import random

    langs = rows[language_key]
    by_lang: dict[str, list[int]] = collections.defaultdict(list)
    for i in range(rows.num_rows):
        by_lang[langs[i]].append(i)
    present = sorted(by_lang)
    quota = max(1, max_samples // len(present))
    rng = random.Random(seed)
    picked: list[int] = []
    per_lang_counts: dict[str, int] = {}
    for lang in present:
        idxs = by_lang[lang][:]
        rng.shuffle(idxs)
        take = idxs[:quota]
        per_lang_counts[lang] = len(take)
        picked.extend(take)
    rng.shuffle(picked)
    return rows.select(sorted(picked)), per_lang_counts


def _encoder(tok, label2id, max_length):
    def _encode(batch: dict) -> dict:
        enc = tok(batch["text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
        out = []
        for i, spans in enumerate(batch["spans"]):
            out.append(
                _bioes_from_spans(
                    batch["text"][i], spans, enc["offset_mapping"][i], enc.word_ids(batch_index=i), label2id
                )
            )
        enc.pop("offset_mapping")
        enc["labels"] = out
        return enc

    return _encode


def _train_one_seed(
    *, base_model, train_ds, eval_ds, tok, labels, id2label, label2id, resolved_device,
    epochs, lr, lora_rank, batch_size, output_dir, seed, max_util, max_util_batch_size,
    num_workers, bf16,
):
    """Train one LoRA finetune, merge, save to ``output_dir``. Returns (eval_loss, profile, sps)."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    targets = lora_target_modules(base_model)
    logger.info("LoRA target_modules for %s: %s", base_model, targets)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.TOKEN_CLS, r=lora_rank, lora_alpha=lora_rank * 2,
            lora_dropout=0.1, target_modules=targets,
        ),
    )
    trainable = model.get_nb_trainable_parameters()
    model.print_trainable_parameters()
    # Fail loud rather than silently train an empty adapter (the bug arch-aware targets prevents).
    if trainable[0] == 0:
        raise RuntimeError(f"LoRA matched 0 trainable params on {base_model} (targets={targets})")
    collator = DataCollatorForTokenClassification(tok)

    probe_features = [eval_ds[i] for i in range(min(len(eval_ds), MAX_UTIL_BATCH_CANDIDATES[0]))]
    profile = resolve_max_util_profile(
        resolved_device, max_util=(max_util or False), batch_size=batch_size,
        batch_override=max_util_batch_size, num_workers=num_workers, bf16=bf16,
        model=model, collator=collator, sample_features=probe_features,
    )
    if profile.enabled:
        logger.info("max-util ON: batch %d->%d workers=%d bf16=%s | %s", batch_size,
                    profile.batch_size, profile.num_workers, profile.bf16, "; ".join(profile.probe_log))

    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=profile.batch_size,
        per_device_eval_batch_size=profile.eval_batch_size,
        learning_rate=lr, eval_strategy="epoch", save_strategy="no", logging_steps=50,
        seed=seed, report_to=[], use_cpu=(resolved_device == "cpu"),
        dataloader_num_workers=profile.num_workers,
        dataloader_persistent_workers=profile.persistent_workers,
        dataloader_pin_memory=profile.pin_memory, bf16=profile.bf16,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=collator)
    train_out = trainer.train()
    sps = float(train_out.metrics.get("train_samples_per_second", 0.0) or 0.0)
    eval_loss = float(trainer.evaluate().get("eval_loss"))

    model = model.to("cpu")
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    logger.info("seed %d: eval_loss=%.6f saved -> %s", seed, eval_loss, output_dir)
    return eval_loss, profile, sps


@click.command()
@click.option("--base", "base_model", default="FacebookAI/xlm-roberta-large")
@click.option("--out-prefix", default="runs/kp-deid-xlmr-560m",
              help="Output dir prefix; per-seed dirs get '-seed<N>' appended.")
@click.option("--device", default="mps", type=click.Choice(["auto", "cpu", "mps", "cuda"]))
@click.option("--epochs", type=int, default=3, help="<=3 (hard-capped).")
@click.option("--lr", type=float, default=2e-4,
              help="LoRA LR. Slightly below the 280M 3e-4: the 560M base is more capacity-rich, a "
              "touch lower LR is steadier for the larger adapter.")
@click.option("--lora-rank", type=int, default=16)
@click.option("--batch-size", type=int, default=16)
@click.option("--max-length", type=int, default=256)
@click.option("--max-samples", type=int, default=40000,
              help="Total balanced training samples across 8 langs (NOT per language). Hard bound.")
@click.option("--heldout-templates-per-language", type=int, default=1)
@click.option("--eval-fraction", type=float, default=0.04)
@click.option("--seeds", multiple=True, type=int, default=(0, 1),
              help="Training seeds (>=2 for the headline-delta variance).")
@click.option("--wall-clock-stop", type=float, default=25200.0,
              help="Stop-and-report wall-clock budget in seconds (default 7h — the 560M is ~2x the "
              "280M's ~10min/seed, plus headroom). No seed is started that cannot finish within it.")
@click.option("--metrics-out", default="runs/res97-train-manifest.json")
@click.option("--max-util/--no-max-util", "max_util", default=None)
@click.option("--max-util-batch-size", type=int, default=None)
@click.option("--num-workers", type=int, default=None)
@click.option("--bf16/--no-bf16", "bf16", default=False)
def main(base_model, out_prefix, device, epochs, lr, lora_rank, batch_size, max_length,
         max_samples, heldout_templates_per_language, eval_fraction, seeds, wall_clock_stop,
         metrics_out, max_util, max_util_batch_size, num_workers, bf16):
    import torch
    from transformers import AutoTokenizer

    from europriv_bench.taxonomy import bioes_labels

    if epochs > 3:
        raise click.UsageError("epochs is hard-capped at 3 (runaway guard).")
    seeds = list(seeds) or [0, 1]
    if len(seeds) < 2:
        logger.warning("only %d seed(s) — RES-97 wants >=2 for the headline-delta variance", len(seeds))

    resolved_device = resolve_device(device, cpu=(device == "cpu"))
    if resolved_device != "cpu":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    logger.info("resolved device=%s | base=%s | seeds=%s | max_samples=%d | epochs=%d | wall-stop=%.0fs",
                resolved_device, base_model, seeds, max_samples, epochs, wall_clock_stop)

    labels = bioes_labels()
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}
    tok = AutoTokenizer.from_pretrained(base_model)
    if not tok.is_fast:
        raise RuntimeError(f"{base_model} needs a fast tokenizer for span alignment")

    wall0 = time.time()

    carve_seed = seeds[0]
    raw = _load_all(seed=carve_seed)
    train_pool, heldout_rows, carve_info = carve_heldout_general(
        raw, heldout_templates_per_language=heldout_templates_per_language, seed=carve_seed
    )
    logger.info("carve-out: %s", json.dumps(carve_info))

    heldout_dir = Path(out_prefix).parent / "res97-heldout-general"
    heldout_rows.save_to_disk(str(heldout_dir))
    logger.info("saved clean held-out general split (%d rows) -> %s", heldout_rows.num_rows, heldout_dir)

    train_bal, per_lang_counts = _balanced_downsample(
        train_pool, max_samples=max_samples, language_key="language", seed=carve_seed
    )
    logger.info("balanced down-sample: %d rows | per-language: %s",
                train_bal.num_rows, json.dumps(per_lang_counts))

    import collections

    def _subjects_by_lang(rows):
        # Vectorized column reads (single Arrow materialization each), then index Python lists —
        # NOT 400k/40k random ``rows[i]["col"]`` __getitem__s into the Arrow table (the hang).
        texts = rows["text"]
        spanss = rows["spans"]
        langs = rows["language"]
        out = collections.defaultdict(set)
        for text, spans, lang in zip(texts, spanss, langs):
            out[lang] |= _subject_surface_forms(text, spans, labels=SUBJECT_LABELS)
        return out

    held_subj = _subjects_by_lang(heldout_rows)
    bal_subj = _subjects_by_lang(train_bal)
    post_intersection = {lg: sorted(bal_subj.get(lg, set()) & held_subj.get(lg, set())) for lg in held_subj}
    post_nonempty = {lg: len(x) for lg, x in post_intersection.items() if x}
    if post_nonempty:
        raise RuntimeError(
            f"post-downsample train∩heldout subject intersection NON-empty: {post_nonempty} "
            "— the balanced draw re-absorbed held-out subjects (contamination)."
        )
    logger.info("post-downsample subject intersection per language: EMPTY (asserted) %s",
                {lg: len(x) for lg, x in post_intersection.items()})

    surface_holdout = identifier_surface_form_holdout(train_bal, heldout_rows)
    logger.info("identifier-surface-form holdout: %d/%d held-out rows have NO PII string seen in train",
                surface_holdout["n"], surface_holdout["heldout_total"])

    train_rows, intrain_eval_rows, intrain_split = template_disjoint_split(
        train_bal, eval_fraction=eval_fraction, seed=carve_seed
    )
    logger.info("in-training template-disjoint split: %s", json.dumps(intrain_split))

    encode = _encoder(tok, label2id, max_length)
    train_ds = train_rows.map(encode, batched=True, remove_columns=train_rows.column_names)
    intrain_eval_ds = intrain_eval_rows.map(encode, batched=True, remove_columns=intrain_eval_rows.column_names)

    seed_artifacts: list[dict] = []
    profile_summary = None
    sampler = GpuUtilSampler().start() if resolved_device == "mps" else None
    elapsed_per_seed = 0.0
    for k, sd in enumerate(seeds):
        remaining = wall_clock_stop - (time.time() - wall0)
        # Conservative prior before we have a measurement: 560M ~ a few x the 280M's ~11 min/seed.
        need = elapsed_per_seed if elapsed_per_seed else 3600.0
        if k > 0 and remaining < need:
            logger.warning("STOP-AND-REPORT: %.0fs left < ~%.0fs needed for seed %d; stopping with "
                           "%d seed(s) done.", remaining, need, sd, k)
            break
        out_dir = f"{out_prefix}-seed{sd}"
        t0 = time.time()
        logger.info(">>> seed %d -> %s (epochs=%d lr=%g r=%d)", sd, out_dir, epochs, lr, lora_rank)
        eval_loss, profile, sps = _train_one_seed(
            base_model=base_model, train_ds=train_ds, eval_ds=intrain_eval_ds, tok=tok,
            labels=labels, id2label=id2label, label2id=label2id, resolved_device=resolved_device,
            epochs=epochs, lr=lr, lora_rank=lora_rank, batch_size=batch_size, output_dir=out_dir,
            seed=sd, max_util=max_util, max_util_batch_size=max_util_batch_size,
            num_workers=num_workers, bf16=bf16,
        )
        elapsed_per_seed = time.time() - t0
        peak_gb = peak_mps_mem_gb()
        seed_artifacts.append({
            "seed": sd, "output_dir": out_dir, "intrain_eval_loss": eval_loss,
            "train_seconds": round(elapsed_per_seed, 1),
            "effective_batch_size": profile.batch_size, "num_workers": profile.num_workers,
            "bf16": profile.bf16, "train_samples_per_second": round(sps, 1),
            "peak_mps_mem_gb": round(peak_gb, 1) if peak_gb else None,
        })
        profile_summary = {"effective_batch_size": profile.batch_size,
                           "num_workers": profile.num_workers, "bf16": profile.bf16}
        if resolved_device == "mps":
            torch.mps.empty_cache()

    util_report = sampler.stop() if sampler else {"available": False, "note": "non-mps device"}
    total_wall = time.time() - wall0

    manifest = {
        "issue": "RES-97",
        "model": "kp-deid-xlmr-560m",
        "base_model": base_model,
        "lora_target_modules": lora_target_modules(base_model),
        "languages": LANGUAGES,
        "datasets": [DATASET_TMPL.format(lang=lg) for lg in LANGUAGES],
        "device": resolved_device,
        "bounds": {
            "max_samples_total_balanced": max_samples,
            "epochs": epochs,
            "wall_clock_stop_seconds": wall_clock_stop,
            "epochs_hard_cap": 3,
        },
        "carve_out": carve_info,
        "post_downsample_subject_intersection_per_language": {lg: len(x) for lg, x in post_intersection.items()},
        "post_downsample_subject_disjoint": True,
        "balanced_train_per_language": per_lang_counts,
        "balanced_train_rows": train_bal.num_rows,
        "train_rows_after_intrain_split": train_rows.num_rows,
        "intrain_eval_split": intrain_split,
        "identifier_surface_form_holdout": {
            "n": surface_holdout["n"], "heldout_total": surface_holdout["heldout_total"],
            "indices": surface_holdout["indices"],
        },
        "heldout_general_dir": str(heldout_dir),
        "subject_labels": sorted(SUBJECT_LABELS),
        "seeds_requested": seeds,
        "seeds_completed": [a["seed"] for a in seed_artifacts],
        "seed_artifacts": seed_artifacts,
        "max_util_profile": profile_summary,
        "utilization": util_report,
        "total_wall_seconds": round(total_wall, 1),
        "total_wall_minutes": round(total_wall / 60.0, 1),
        "stopped_early": len(seed_artifacts) < len(seeds),
        "lr": lr, "lora_rank": lora_rank, "batch_size": batch_size, "max_length": max_length,
        "config_status": "dev",
    }
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("wrote training manifest -> %s | total wall %.1f min | seeds done: %s",
                metrics_out, total_wall / 60.0, [a["seed"] for a in seed_artifacts])
    click.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
