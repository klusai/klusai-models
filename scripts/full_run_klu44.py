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
                  lr, lora_rank, epochs, batch_size, output_dir, resolved_device, seed, eval_only=False):
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

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=seed,
        report_to=[],
        use_cpu=(resolved_device == "cpu"),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tok),
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
def main(base_model, publish_id, output_dir, device, epochs, sweep_epochs, sweep_subset,
         batch_size, max_length, eval_fraction, seed, push, metrics_out):
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
    n = raw.num_rows
    n_eval = max(1, int(n * eval_fraction))
    eval_rows = raw.select(range(n_eval))
    train_rows = raw.select(range(n_eval, n))
    logger.info("split: %d train / %d eval", train_rows.num_rows, eval_rows.num_rows)

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
            seed=seed,
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
        "epochs": epochs,
        "sweep_epochs": sweep_epochs,
        "sweep_subset": sweep_train.num_rows,
        "batch_size": batch_size,
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
