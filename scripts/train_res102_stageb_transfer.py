#!/usr/bin/env python3
"""RES-102 — stage-B structural-transfer ablation: train ONE arm on a local JSONL, score on TAB.

The question: does training the shipped detector (kp-deid-mdeberta-280m, base
``microsoft/mdeberta-v3-base``) on structurally-diverse **stage-B** synthetic EN data transfer
better to REAL legal gold (TAB ECHR) than training on template-splice **stage-A** data — at matched
size + matched PII (only document STRUCTURE differs)?

This is a matched-pair ablation. Two arms, IDENTICAL machinery / hyperparams / seed; the ONLY
difference is the input JSONL:
  * arm A: ``en_stagea_matched_v1.jsonl`` (2000 template-splice docs)
  * arm B: ``en_stageb_v1.jsonl``        (2000 LLM-narrated docs, same per-doc PII)

Each invocation trains exactly ONE arm (single process, single log — RES-97 concurrency lesson),
merges the LoRA adapter, saves the checkpoint, then scores it on the TAB eval **reusing the exact
RES-97 path** (``scorecard_klu106._predict`` / ``_entity_f1_on_rows`` / ``_mask`` over the gold rows
``europriv_bench`` loads from ``pii-detection-tab-echr-legal-en.yaml``) so the number is apples-to-
apples with the known mdeberta-280m TAB baseline (0.049, RES-97).

The same token-classification LoRA pieces the shipped detector is trained with are reused verbatim
(``_bioes_from_spans`` span->BIOES, ``lora_target_modules`` arch-aware targets, ``resolve_device``).
Absolute F1 will be low (2000-doc models); the DELTA between arms is the result.

    python scripts/train_res102_stageb_transfer.py \
        --arm stagea --data ../klusai-datasets/artifacts/europriv/en_stagea_matched_v1.jsonl \
        --out runs/res102-stagea-seed0 --epochs 3 --seed 0
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import click

# scorecard_klu106 lives in this same scripts/ dir (RES-97 reused it via sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from klusai.privacy.models.logger import get_logger
from klusai.privacy.models.training.token_classification import (
    _bioes_from_spans,
    lora_target_modules,
    resolve_device,
)

logger = get_logger("train_res102")

TAB_SPEC = "pii-detection-tab-echr-legal-en.yaml"


def _load_jsonl_dataset(path: str):
    from datasets import Dataset

    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({"text": r["text"], "spans": r["spans"]})
    logger.info("loaded %d rows from %s", len(rows), path)
    return Dataset.from_list(rows)


def _encoder(tok, label2id, max_length):
    def _encode(batch: dict) -> dict:
        enc = tok(batch["text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
        out = []
        for i, spans in enumerate(batch["spans"]):
            out.append(
                _bioes_from_spans(
                    batch["text"][i], spans, enc["offset_mapping"][i],
                    enc.word_ids(batch_index=i), label2id,
                )
            )
        enc.pop("offset_mapping")
        enc["labels"] = out
        return enc

    return _encode


def _train(base_model, train_ds, tok, labels, id2label, label2id, resolved_device,
           epochs, lr, lora_rank, batch_size, output_dir, seed):
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
    if trainable[0] == 0:
        raise RuntimeError(f"LoRA matched 0 trainable params on {base_model} (targets={targets})")

    collator = DataCollatorForTokenClassification(tok)
    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr,
        eval_strategy="no", save_strategy="no", logging_steps=50,
        seed=seed, report_to=[], use_cpu=(resolved_device == "cpu"),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
    t0 = time.time()
    train_out = trainer.train()
    train_seconds = time.time() - t0
    sps = float(train_out.metrics.get("train_samples_per_second", 0.0) or 0.0)

    model = model.to("cpu")
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    logger.info("saved merged checkpoint -> %s (%.1fs, %.1f sps)", output_dir, train_seconds, sps)
    return train_seconds, sps


def _eval_tab(model_dir: str, suite: str) -> dict:
    """Score ``model_dir`` on TAB ECHR gold via the EXACT RES-97 path (entity-F1, masked to gold)."""
    os.environ.setdefault("EUROPRIV_DEVICE", "cpu")  # deterministic CPU scoring, as RES-97
    from scorecard_klu106 import _entity_f1_on_rows, _mask, _predict

    from europriv_bench.runner import _load_gold_rows, _rows_to_gold
    from europriv_bench.spec import EvalSpec

    spec = EvalSpec.from_yaml(str(Path(suite) / TAB_SPEC))
    rows = _load_gold_rows(spec)
    texts = [r["text"] for r in rows]
    _, gold_tags = _rows_to_gold(rows)
    eval_labels = sorted({t.split("-", 1)[1] for seq in gold_tags for t in seq if t != "O"})
    pred = _mask(_predict(model_dir, texts), eval_labels)
    m = _entity_f1_on_rows(rows, pred)

    def _f2(p, r):
        return 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0

    out = {
        "config": spec.dataset.config,
        "n": len(rows),
        "eval_labels": eval_labels,
        "entity_f1": round(m["f1"], 4),
        "entity_f2": round(_f2(m.get("precision", 0.0), m.get("recall", 0.0)), 4),
        "precision": round(m.get("precision", 0.0), 4),
        "recall": round(m.get("recall", 0.0), 4),
    }
    logger.info("TAB %s: entity_f1=%.4f (P=%.4f R=%.4f n=%d)", out["config"], out["entity_f1"],
                out["precision"], out["recall"], out["n"])
    return out


@click.command()
@click.option("--arm", type=click.Choice(["stagea", "stageb"]), required=True)
@click.option("--data", required=True, help="Local JSONL for this arm ({text, spans} per line).")
@click.option("--out", "output_dir", required=True, help="Merged-checkpoint output dir.")
@click.option("--base", "base_model", default="microsoft/mdeberta-v3-base",
              help="The SHIPPED detector base (kp-deid-mdeberta-280m). Same for both arms.")
@click.option("--device", default="mps", type=click.Choice(["auto", "cpu", "mps", "cuda"]))
@click.option("--epochs", type=int, default=3, help="<=3 (modest for 2000 docs).")
@click.option("--lr", type=float, default=3e-4, help="LoRA LR (mdeberta-280m default).")
@click.option("--lora-rank", type=int, default=16)
@click.option("--batch-size", type=int, default=16)
@click.option("--max-length", type=int, default=256)
@click.option("--seed", type=int, default=0)
@click.option("--max-train-samples", type=int, default=None,
              help="Cap train rows (dry-run uses 100).")
@click.option("--suite", default="../europriv-bench/evaluations")
@click.option("--metrics-out", default=None, help="Where to write this arm's result JSON.")
def main(arm, data, output_dir, base_model, device, epochs, lr, lora_rank, batch_size,
         max_length, seed, max_train_samples, suite, metrics_out):
    if epochs > 3:
        raise click.UsageError("epochs is hard-capped at 3 (runaway guard).")
    import torch
    from transformers import AutoTokenizer

    from europriv_bench.taxonomy import bioes_labels

    resolved_device = resolve_device(device, cpu=(device == "cpu"))
    if resolved_device != "cpu":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    logger.info("RES-102 arm=%s device=%s base=%s epochs=%d lr=%g r=%d bs=%d seed=%d",
                arm, resolved_device, base_model, epochs, lr, lora_rank, batch_size, seed)

    labels = bioes_labels()
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}
    tok = AutoTokenizer.from_pretrained(base_model)
    if not tok.is_fast:
        raise RuntimeError(f"{base_model} needs a fast tokenizer for span alignment")

    ds = _load_jsonl_dataset(data)
    if max_train_samples is not None:
        ds = ds.select(range(min(max_train_samples, ds.num_rows)))
        logger.info("capped train rows -> %d", ds.num_rows)
    encode = _encoder(tok, label2id, max_length)
    train_ds = ds.map(encode, batched=True, remove_columns=ds.column_names)

    wall0 = time.time()
    train_seconds, sps = _train(
        base_model=base_model, train_ds=train_ds, tok=tok, labels=labels, id2label=id2label,
        label2id=label2id, resolved_device=resolved_device, epochs=epochs, lr=lr,
        lora_rank=lora_rank, batch_size=batch_size, output_dir=output_dir, seed=seed,
    )
    if resolved_device == "mps":
        torch.mps.empty_cache()

    tab = _eval_tab(output_dir, suite)
    total_wall = time.time() - wall0

    result = {
        "issue": "RES-102",
        "arm": arm,
        "data": str(Path(data).resolve()),
        "n_train_rows": train_ds.num_rows,
        "base_model": base_model,
        "device": resolved_device,
        "hyperparams": {"epochs": epochs, "lr": lr, "lora_rank": lora_rank,
                        "batch_size": batch_size, "max_length": max_length, "seed": seed},
        "lora_target_modules": lora_target_modules(base_model),
        "checkpoint_dir": output_dir,
        "train_seconds": round(train_seconds, 1),
        "train_samples_per_second": round(sps, 1),
        "total_wall_seconds": round(total_wall, 1),
        "tab": tab,
        "tab_entity_f1": tab["entity_f1"],
        "config_status": "dev",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    metrics_out = metrics_out or f"{output_dir}-result.json"
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote result -> %s | TAB entity_f1=%.4f | wall %.1fs",
                metrics_out, tab["entity_f1"], total_wall)
    click.echo(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
