#!/usr/bin/env python3
"""KLU-44 — full multilingual finetune of kp-deid-mdeberta-280m on the Mac GPU (MPS).

Trains the `xlmr-ner` mDeBERTa LoRA token-classifier on the FULL multilingual
`klusai/ds-kp-general-{ro,en,pl}-50k` (150k examples), with a small held-out hyperparameter
sweep (LR x LoRA-r), then retrains the best config on all data, merges LoRA into the base, and
(optionally) publishes the merged model to the Hub.

This is the full-run companion to the bounded smoke path in
`klusai.privacy.models.training.token_classification`. It reuses that module's span->BIOES
projection (`_bioes_from_spans`) and device resolution (`resolve_device`) so the trained label
space stays byte-identical to the benchmark's, and runs on `--device mps` by default (KLU-45),
with CPU as the guaranteed fallback.

Run from the repo root after `make install`:

    python scripts/full_run_klu44.py --device mps --epochs 3 --push \
        --out klusai/kp-deid-mdeberta-280m
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import click

from klusai.privacy.models.logger import get_logger
from klusai.privacy.models.training.token_classification import (
    _bioes_from_spans,
    resolve_device,
    resolve_max_util_profile,
    template_disjoint_split,
)

logger = get_logger("full_run_klu44")

DATASETS = [
    "klusai/ds-kp-general-ro-50k",
    "klusai/ds-kp-general-en-50k",
    "klusai/ds-kp-general-pl-50k",
]

# Small sweep grid (LR, LoRA-r). The smoke used lr=3e-4 / r=16. We bracket it with a slightly
# gentler LR and a larger rank — more capacity for the multilingual mix and the IBAN-style
# ACCOUNT_ID over-fragmentation that limited the smoke F1.
SWEEP = [
    {"lr": 3e-4, "lora_rank": 16},
    {"lr": 2e-4, "lora_rank": 16},
    {"lr": 2e-4, "lora_rank": 32},
]


@dataclass
class SweepResult:
    lr: float
    lora_rank: int
    eval_loss: float
    seconds: float


def _build_dataset(seed: int):
    from datasets import concatenate_datasets, load_dataset

    parts = []
    for hf_id in DATASETS:
        d = load_dataset(hf_id, split="train")
        parts.append(d)
        logger.info("loaded %s: %d rows", hf_id, d.num_rows)
    merged = concatenate_datasets(parts).shuffle(seed=seed)
    logger.info("merged multilingual corpus: %d rows", merged.num_rows)
    return merged


def _encoder(tok, label2id, max_length):
    def _encode(batch: dict) -> dict:
        enc = tok(
            batch["text"], truncation=True, max_length=max_length, return_offsets_mapping=True
        )
        all_labels = []
        for i, spans in enumerate(batch["spans"]):
            word_ids = enc.word_ids(batch_index=i)
            all_labels.append(
                _bioes_from_spans(
                    batch["text"][i], spans, enc["offset_mapping"][i], word_ids, label2id
                )
            )
        enc.pop("offset_mapping")
        enc["labels"] = all_labels
        return enc

    return _encode


def _make_trainer(base_model, train_ds, eval_ds, tok, labels, id2label, label2id, *,
                  lr, lora_rank, epochs, batch_size, output_dir, resolved_device, seed,
                  max_util=None, max_util_batch_size=None, num_workers=None, bf16=False,
                  eval_only=False):
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
    peft_cfg = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.1,
        target_modules=["query_proj", "key_proj", "value_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForTokenClassification(tok)

    # KLU-48: max-util profile is OPT-IN (off by default) — for this 280M encoder plain batch-16
    # single-process fp32 is already the measured throughput optimum on MPS; scaling the batch /
    # adding workers regresses it (docs/klu-48-max-util.md). When opted in, auto-probe the largest
    # batch (memory-guarded) off the eval slice.
    if max_util is None:
        max_util = False
    from klusai.privacy.models.training.token_classification import MAX_UTIL_BATCH_CANDIDATES

    probe_features = [eval_ds[i] for i in range(min(len(eval_ds), MAX_UTIL_BATCH_CANDIDATES[0]))]
    profile = resolve_max_util_profile(
        resolved_device,
        max_util=max_util,
        batch_size=batch_size,
        batch_override=max_util_batch_size,
        num_workers=num_workers,
        bf16=bf16,
        model=model,
        collator=collator,
        sample_features=probe_features,
    )
    if profile.enabled:
        logger.info(
            "max-util ON (mps): batch %d -> %d, num_workers=%d, bf16=%s | probe: %s",
            batch_size, profile.batch_size, profile.num_workers, profile.bf16,
            "; ".join(profile.probe_log),
        )
    eff_batch_size = profile.batch_size

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=eff_batch_size,
        per_device_eval_batch_size=profile.eval_batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=seed,
        report_to=[],
        use_cpu=(resolved_device == "cpu"),
        dataloader_num_workers=profile.num_workers,
        dataloader_persistent_workers=profile.persistent_workers,
        dataloader_pin_memory=profile.pin_memory,
        bf16=profile.bf16,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    return model, trainer


@click.command()
@click.option("--base", "base_model", default="microsoft/mdeberta-v3-base")
@click.option("--out", "publish_id", default="klusai/kp-deid-mdeberta-280m")
@click.option("--output-dir", default="runs/kp-deid-mdeberta-280m")
@click.option("--device", default="mps", type=click.Choice(["auto", "cpu", "mps", "cuda"]))
@click.option("--epochs", type=int, default=3, help="Epochs for the final full-data run.")
@click.option("--sweep-epochs", type=int, default=2)
@click.option("--sweep-subset", type=int, default=30000, help="Multilingual subset size for the sweep.")
@click.option("--batch-size", type=int, default=16)
@click.option("--max-length", type=int, default=256)
@click.option("--eval-fraction", type=float, default=0.03)
@click.option("--seed", type=int, default=0)
@click.option("--push/--no-push", default=False)
@click.option("--metrics-out", default="runs/klu44-train-metrics.json")
@click.option("--max-util/--no-max-util", "max_util", default=None,
              help="KLU-48 max-utilization profile (MPS), opt-in. OFF by default — batch-16 is the "
              "measured optimum for this encoder (docs/klu-48-max-util.md).")
@click.option("--max-util-batch-size", type=int, default=None,
              help="Override the auto-probed max-util batch size (skips the probe).")
@click.option("--num-workers", type=int, default=None,
              help="DataLoader workers (max-util). Default 0 (opt-in; macOS spawn exit-hang).")
@click.option("--bf16/--no-bf16", "bf16", default=False,
              help="Enable bf16 autocast on MPS. OFF by default (no win; fp32 matches CPU; KLU-45/48).")
def main(base_model, publish_id, output_dir, device, epochs, sweep_epochs, sweep_subset,
         batch_size, max_length, eval_fraction, seed, push, metrics_out,
         max_util, max_util_batch_size, num_workers, bf16):
    import os

    import torch
    from transformers import AutoTokenizer

    from europriv_bench.taxonomy import bioes_labels

    resolved_device = resolve_device(device, cpu=(device == "cpu"))
    if resolved_device != "cpu":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    logger.info("resolved device=%s", resolved_device)

    labels = bioes_labels()
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}

    tok = AutoTokenizer.from_pretrained(base_model)
    if not tok.is_fast:
        raise RuntimeError(f"{base_model} needs a fast tokenizer for span alignment")

    raw = _build_dataset(seed)
    # KLU-54: template-disjoint held-out eval. The old shuffled-head split re-used the corpus's ~6
    # templates (per language) in both train and eval, so eval-loss measured memorization of fixed
    # skeletons (final_eval_loss ~7.2e-10), not generalization. Hold out whole templates instead so
    # every eval row's structure is absent from train across the multilingual mix.
    train_rows, eval_rows, split_info = template_disjoint_split(
        raw, eval_fraction=eval_fraction, seed=seed
    )
    train_rows = train_rows.shuffle(seed=seed)
    eval_rows = eval_rows.shuffle(seed=seed)
    logger.info(
        "split: %d train / %d eval | template-disjoint: %s",
        train_rows.num_rows, eval_rows.num_rows, split_info,
    )

    encode = _encoder(tok, label2id, max_length)
    eval_ds = eval_rows.map(encode, batched=True, remove_columns=eval_rows.column_names)
    full_train_ds = train_rows.map(encode, batched=True, remove_columns=train_rows.column_names)

    # Sweep on a fixed multilingual subset of the training split.
    sweep_train = full_train_ds.select(range(min(sweep_subset, full_train_ds.num_rows)))
    logger.info("sweep on %d train rows, %d eval rows, %d configs",
                sweep_train.num_rows, eval_ds.num_rows, len(SWEEP))

    sweep_results: list[SweepResult] = []
    wall0 = time.time()
    for cfg in SWEEP:
        t0 = time.time()
        logger.info(">>> sweep config lr=%g lora_rank=%d", cfg["lr"], cfg["lora_rank"])
        _, trainer = _make_trainer(
            base_model, sweep_train, eval_ds, tok, labels, id2label, label2id,
            lr=cfg["lr"], lora_rank=cfg["lora_rank"], epochs=sweep_epochs,
            batch_size=batch_size, output_dir=f"{output_dir}-sweep", resolved_device=resolved_device,
            seed=seed, max_util=max_util, max_util_batch_size=max_util_batch_size,
            num_workers=num_workers, bf16=bf16,
        )
        trainer.train()
        metrics = trainer.evaluate()
        el = float(metrics.get("eval_loss"))
        dt = time.time() - t0
        sweep_results.append(SweepResult(lr=cfg["lr"], lora_rank=cfg["lora_rank"], eval_loss=el, seconds=dt))
        logger.info("<<< lr=%g r=%d eval_loss=%.6f (%.1fs)", cfg["lr"], cfg["lora_rank"], el, dt)
        del trainer
        if resolved_device == "mps":
            torch.mps.empty_cache()

    best = min(sweep_results, key=lambda r: r.eval_loss)
    logger.info("BEST sweep config: lr=%g lora_rank=%d eval_loss=%.6f",
                best.lr, best.lora_rank, best.eval_loss)

    # Final full-data run with the best config.
    logger.info(">>> FINAL full-data run: %d train rows, %d epochs, lr=%g, r=%d",
                full_train_ds.num_rows, epochs, best.lr, best.lora_rank)
    t0 = time.time()
    model, trainer = _make_trainer(
        base_model, full_train_ds, eval_ds, tok, labels, id2label, label2id,
        lr=best.lr, lora_rank=best.lora_rank, epochs=epochs,
        batch_size=batch_size, output_dir=output_dir, resolved_device=resolved_device, seed=seed,
        max_util=max_util, max_util_batch_size=max_util_batch_size,
        num_workers=num_workers, bf16=bf16,
    )
    trainer.train()
    final_metrics = trainer.evaluate()
    final_eval_loss = float(final_metrics.get("eval_loss"))
    final_seconds = time.time() - t0
    logger.info("<<< FINAL eval_loss=%.6f (%.1fs)", final_eval_loss, final_seconds)

    # Merge LoRA into base on CPU (device-independent save).
    model = model.to("cpu")
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    logger.info("saved merged model + tokenizer to %s", output_dir)

    pushed = False
    if push:
        logger.info("pushing merged model + tokenizer to %s", publish_id)
        merged.push_to_hub(publish_id)
        tok.push_to_hub(publish_id)
        pushed = True

    total_wall = time.time() - wall0
    out = {
        "device": resolved_device,
        "datasets": DATASETS,
        "train_rows": full_train_ds.num_rows,
        "eval_rows": eval_ds.num_rows,
        "split": "template-disjoint (KLU-54)",
        "split_info": split_info,
        "epochs": epochs,
        "sweep_epochs": sweep_epochs,
        "sweep_subset": sweep_train.num_rows,
        "batch_size": batch_size,
        "effective_batch_size": trainer.args.per_device_train_batch_size,  # KLU-48: post max-util
        "max_util": trainer.args.dataloader_num_workers > 0,
        "num_workers": trainer.args.dataloader_num_workers,
        "bf16": bool(trainer.args.bf16),
        "max_length": max_length,
        "sweep": [asdict(r) for r in sweep_results],
        "best": asdict(best),
        "final_eval_loss": final_eval_loss,
        "final_train_seconds": final_seconds,
        "total_wall_seconds": total_wall,
        "pushed": pushed,
        "publish_id": publish_id,
    }
    with open(metrics_out, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("wrote training metrics → %s", metrics_out)
    logger.info("TOTAL wall-clock: %.1f min", total_wall / 60.0)
    click.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
