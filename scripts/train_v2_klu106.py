#!/usr/bin/env python3
"""KLU-106 — kp-deid v2: multilingual protector on ro/en/pl + T1 (de/fr/es/it/nl), Mac/MPS.

Trains an improved multilingual ``kp-deid`` (mDeBERTa-280m, LoRA) on **8 languages** on-device
(PyTorch-MPS, M3 Ultra) under hard runaway bounds, with the **load-bearing contamination carve-out**
(KLU-106): a clean held-out general split is carved out per language BEFORE training (template- AND
subject-disjoint) and the job ASSERTS an empty train∩heldout subject intersection per language. The
training pool is then balanced and down-sampled to ``--max-samples`` total across the 8 langs (NOT
per language) and split template-disjoint (KLU-54) for the in-training eval-loss tripwire.

Hard bounds (runaway guard): ``--max-samples`` (default 40000) total balanced; ``--epochs`` <= 3;
``--wall-clock-stop`` seconds (default 3h) with stop-and-report (we never start a seed we cannot
finish within the budget; a partial scorecard is still produced from whatever finished). MPS is
saturated per KLU-48 (the measured optimum for this 280M encoder is batch-16 single-process fp32; we
still report GPU util / power / throughput). ``--seeds`` trains >=2 seeds for the headline-delta
variance.

This script ONLY trains and writes artifacts + a manifest (which held-out config, which seeds, the
carve-out assertion result, utilization). Scoring + the CI'd F1-Δ scorecard is a separate step
(``scripts/scorecard_klu106.py``) so training and the fixed-eval scorecard are decoupled.

Run from the repo root after ``make install``:

    python scripts/train_v2_klu106.py --device mps --epochs 3 --seeds 0 1 \
        --max-samples 40000 --wall-clock-stop 10800
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click

from klusai.privacy.models.logger import get_logger
from klusai.privacy.models.training.token_classification import (
    SUBJECT_LABELS,
    _bioes_from_spans,
    carve_heldout_general,
    identifier_surface_form_holdout,
    resolve_device,
    resolve_max_util_profile,
    template_disjoint_split,
)
from klusai.privacy.models.util_sampler import GpuUtilSampler, peak_mps_mem_gb

logger = get_logger("train_v2_klu106")

# All 8 languages: the 3 v1 languages (ro/en/pl) + the 5 T1 LocalePacks (de/fr/es/it/nl), KLU-107.
LANGUAGES = ["ro", "en", "pl", "de", "fr", "es", "it", "nl"]
DATASET_TMPL = "klusai/ds-kp-general-{lang}-50k"


def _load_all(seed: int):
    """Load + concatenate the 8 general corpora (each tagged with its language)."""
    from datasets import concatenate_datasets, load_dataset

    parts = []
    for lang in LANGUAGES:
        d = load_dataset(DATASET_TMPL.format(lang=lang), split="train")
        # The corpora already carry a `language` column; keep only the columns we use so the
        # concatenation has a uniform schema across packs.
        keep = [c for c in d.column_names if c in ("text", "spans", "language", "domain")]
        d = d.select_columns(keep)
        parts.append(d)
        logger.info("loaded %s: %d rows", lang, d.num_rows)
    merged = concatenate_datasets(parts).shuffle(seed=seed)
    logger.info("merged 8-language corpus: %d rows", merged.num_rows)
    return merged


def _balanced_downsample(rows, *, max_samples: int, language_key: str, seed: int):
    """Down-sample the train POOL to <= max_samples total, balanced across present languages.

    Balanced = an equal per-language quota = max_samples // n_langs (capped at what each language
    has). NOT max_samples per language (the runaway the bound guards against). Returns the
    down-sampled rows + a per-language count dict for honest logging.
    """
    import collections

    langs = rows[language_key]
    by_lang: dict[str, list[int]] = collections.defaultdict(list)
    for i in range(rows.num_rows):
        by_lang[langs[i]].append(i)
    present = sorted(by_lang)
    quota = max(1, max_samples // len(present))

    import random

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

    from klusai.privacy.models.training.token_classification import MAX_UTIL_BATCH_CANDIDATES

    model = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.TOKEN_CLS, r=lora_rank, lora_alpha=lora_rank * 2,
            lora_dropout=0.1, target_modules=["query_proj", "key_proj", "value_proj"],
        ),
    )
    model.print_trainable_parameters()
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
    logger.info("seed %d: eval_loss=%.4f saved -> %s", seed, eval_loss, output_dir)
    return eval_loss, profile, sps


@click.command()
@click.option("--base", "base_model", default="microsoft/mdeberta-v3-base")
@click.option("--out-prefix", default="runs/kp-deid-mdeberta-280m-v2",
              help="Output dir prefix; per-seed dirs get '-seed<N>' appended.")
@click.option("--device", default="mps", type=click.Choice(["auto", "cpu", "mps", "cuda"]))
@click.option("--epochs", type=int, default=3, help="<=3 (hard-capped).")
@click.option("--lr", type=float, default=3e-4)
@click.option("--lora-rank", type=int, default=16)
@click.option("--batch-size", type=int, default=16)
@click.option("--max-length", type=int, default=256)
@click.option("--max-samples", type=int, default=40000,
              help="Total balanced training samples across 8 langs (NOT per language). Hard bound.")
@click.option("--heldout-templates-per-language", type=int, default=1,
              help="Whole generator templates held out per language for the clean general split.")
@click.option("--eval-fraction", type=float, default=0.04,
              help="In-training template-disjoint eval fraction (KLU-54 tripwire) of the train pool.")
@click.option("--seeds", multiple=True, type=int, default=(0, 1),
              help="Training seeds (>=2 for the headline-delta variance).")
@click.option("--wall-clock-stop", type=float, default=10800.0,
              help="Stop-and-report wall-clock budget in seconds (~3h). No seed is started that "
              "cannot plausibly finish within it.")
@click.option("--metrics-out", default="runs/klu106-train-manifest.json")
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
        raise click.UsageError("epochs is hard-capped at 3 (KLU-106 runaway guard).")
    seeds = list(seeds) or [0, 1]
    if len(seeds) < 2:
        logger.warning("only %d seed(s) — KLU-106 wants >=2 for the headline-delta variance", len(seeds))

    resolved_device = resolve_device(device, cpu=(device == "cpu"))
    if resolved_device != "cpu":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    logger.info("resolved device=%s | seeds=%s | max_samples=%d | epochs=%d | wall-stop=%.0fs",
                resolved_device, seeds, max_samples, epochs, wall_clock_stop)

    labels = bioes_labels()
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}
    tok = AutoTokenizer.from_pretrained(base_model)
    if not tok.is_fast:
        raise RuntimeError(f"{base_model} needs a fast tokenizer for span alignment")

    wall0 = time.time()

    # --- Contamination carve-out (load-bearing): per-language clean held-out, BEFORE down-sample.
    # Use seed of the FIRST training seed for the carve so the held-out set is FIXED across seeds
    # (a fixed eval set is required; the headline delta is bootstrap-CI'd over a fixed eval).
    carve_seed = seeds[0]
    raw = _load_all(seed=carve_seed)
    train_pool, heldout_rows, carve_info = carve_heldout_general(
        raw, heldout_templates_per_language=heldout_templates_per_language, seed=carve_seed
    )
    logger.info("carve-out: %s", json.dumps(carve_info))

    # Persist the clean held-out general split to disk (a FIXED eval set the scorecard reads).
    heldout_dir = Path(out_prefix).parent / "klu106-heldout-general"
    heldout_rows.save_to_disk(str(heldout_dir))
    logger.info("saved clean held-out general split (%d rows) -> %s", heldout_rows.num_rows, heldout_dir)

    # --- Balanced down-sample of the TRAIN POOL to <= max_samples total across 8 langs.
    train_bal, per_lang_counts = _balanced_downsample(
        train_pool, max_samples=max_samples, language_key="language", seed=carve_seed
    )
    logger.info("balanced down-sample: %d rows | per-language: %s",
                train_bal.num_rows, json.dumps(per_lang_counts))

    # Subject-level exclusion re-assert AFTER the down-sample (the bound the bug bit): the held-out
    # subjects must not have been re-absorbed by the balanced draw.
    import collections

    def _subjects_by_lang(rows):
        from klusai.privacy.models.training.token_classification import _subject_surface_forms
        out = collections.defaultdict(set)
        for i in range(rows.num_rows):
            out[rows[i]["language"]] |= _subject_surface_forms(
                rows[i]["text"], rows[i]["spans"], labels=SUBJECT_LABELS
            )
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

    # Identifier-surface-form holdout (surface-form-memorization tripwire) vs the balanced train set.
    surface_holdout = identifier_surface_form_holdout(train_bal, heldout_rows)
    logger.info("identifier-surface-form holdout: %d/%d held-out rows have NO PII string seen in train",
                surface_holdout["n"], surface_holdout["heldout_total"])

    # --- In-training template-disjoint eval (KLU-54 tripwire) carved from the BALANCED train set.
    train_rows, intrain_eval_rows, intrain_split = template_disjoint_split(
        train_bal, eval_fraction=eval_fraction, seed=carve_seed
    )
    logger.info("in-training template-disjoint split: %s", json.dumps(intrain_split))

    encode = _encoder(tok, label2id, max_length)
    train_ds = train_rows.map(encode, batched=True, remove_columns=train_rows.column_names)
    intrain_eval_ds = intrain_eval_rows.map(encode, batched=True, remove_columns=intrain_eval_rows.column_names)

    # --- Train >=2 seeds, each within the wall-clock budget (stop-and-report).
    seed_artifacts: list[dict] = []
    profile_summary = None
    sampler = GpuUtilSampler().start() if resolved_device == "mps" else None
    elapsed_per_seed = 0.0
    for k, sd in enumerate(seeds):
        remaining = wall_clock_stop - (time.time() - wall0)
        # Don't start a seed we cannot plausibly finish: require the budget left to exceed the
        # observed per-seed cost (or a conservative 25 min prior before we have a measurement).
        need = elapsed_per_seed if elapsed_per_seed else 1500.0
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
        "issue": "KLU-106",
        "base_model": base_model,
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
            # Indices into the saved held-out general split whose every PII surface string is
            # absent from train — the scorecard scores F1 on exactly this subset.
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
    }
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("wrote training manifest -> %s | total wall %.1f min | seeds done: %s",
                metrics_out, total_wall / 60.0, [a["seed"] for a in seed_artifacts])
    click.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
