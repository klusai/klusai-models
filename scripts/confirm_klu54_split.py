#!/usr/bin/env python3
"""KLU-54 — short before/after confirmation that the eval split is no longer trivial/leaky.

Runs two SHORT, identical bounded finetunes of the xlmr-ner mDeBERTa LoRA token-classifier on
`klusai/ds-kp-general-ro-50k`, differing ONLY in how the eval split is formed:

  * "leaky"   — the OLD split: shuffle the corpus and take the head as eval (shares the corpus's
                ~6 generator templates with train -> eval-loss measures memorization, ~7e-10).
  * "disjoint"— the NEW split: `template_disjoint_split` holds out whole templates so eval shares
                no template+content with train (a genuine held-out generalization set).

Same base model, LoRA config, LR, epochs, batch, seed, and train/eval row budget for both, so the
only moving part is the split. We expect: leaky eval-loss collapses toward ~0; disjoint eval-loss
lands in a clearly non-trivial band. This is a confirmation run, NOT a sweep — keep it bounded.

The eval-loss is a *split-sanity* number only. It is NOT evidence of model quality — only the
EuroPriv-Bench harness leaderboard (entity F1 / leak-rate on the contamination-free real-skeleton)
measures quality (KLU-54).

    python scripts/confirm_klu54_split.py --device mps --epochs 2 \
        --max-train 4000 --max-eval 800 --out runs/klu54-split-confirm.json
"""

from __future__ import annotations

import json
import time

import click

from klusai.privacy.models.logger import get_logger
from klusai.privacy.models.training.token_classification import (
    _bioes_from_spans,
    resolve_device,
    template_disjoint_split,
)

logger = get_logger("confirm_klu54")


def _leaky_split(raw, *, eval_fraction: float, seed: int):
    """The OLD (pre-KLU-54) split: shuffled head of one corpus -> eval (template-overlapping)."""
    raw = raw.shuffle(seed=seed)
    n = raw.num_rows
    n_eval = max(1, int(n * eval_fraction))
    return raw.select(range(n_eval, n)), raw.select(range(n_eval))


def _run_one(
    label, base_model, train_rows, eval_rows, tok, labels, id2label, label2id, *,
    lr, lora_rank, epochs, batch_size, max_length, resolved_device, seed,
):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    def _encode(batch):
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

    train_ds = train_rows.map(_encode, batched=True, remove_columns=train_rows.column_names)
    eval_ds = eval_rows.map(_encode, batched=True, remove_columns=eval_rows.column_names)

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
    args = TrainingArguments(
        output_dir=f"runs/klu54-{label}",
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
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorForTokenClassification(tok))
    t0 = time.time()
    train_out = trainer.train()
    metrics = trainer.evaluate()
    dt = time.time() - t0
    sps = float(train_out.metrics.get("train_samples_per_second", 0.0) or 0.0)
    el = float(metrics.get("eval_loss"))
    logger.info("[%s] eval_loss=%.6g  train_samples/s=%.1f  (%.1fs)", label, el, sps, dt)
    return {
        "label": label,
        "train_rows": train_ds.num_rows,
        "eval_rows": eval_ds.num_rows,
        "eval_loss": el,
        "train_samples_per_second": sps,
        "seconds": dt,
    }


@click.command()
@click.option("--base", "base_model", default="microsoft/mdeberta-v3-base")
@click.option("--dataset", default="klusai/ds-kp-general-ro-50k")
@click.option("--device", default="mps", type=click.Choice(["auto", "cpu", "mps", "cuda"]))
@click.option("--epochs", type=int, default=2)
@click.option("--lr", type=float, default=3e-4)
@click.option("--lora-rank", type=int, default=16)
@click.option("--batch-size", type=int, default=16)
@click.option("--max-length", type=int, default=256)
@click.option("--eval-fraction", type=float, default=0.2)
@click.option("--max-train", type=int, default=4000)
@click.option("--max-eval", type=int, default=800)
@click.option("--seed", type=int, default=0)
@click.option("--out", "out_path", default="runs/klu54-split-confirm.json")
def main(base_model, dataset, device, epochs, lr, lora_rank, batch_size, max_length,
         eval_fraction, max_train, max_eval, seed, out_path):
    import os

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from europriv_bench.taxonomy import bioes_labels

    resolved = resolve_device(device, cpu=(device == "cpu"))
    if resolved != "cpu":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    logger.info("device=%s", resolved)

    labels = bioes_labels()
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}
    tok = AutoTokenizer.from_pretrained(base_model)

    raw = load_dataset(dataset, split="train")

    def _cap(tr, ev):
        tr = tr.shuffle(seed=seed).select(range(min(max_train, tr.num_rows)))
        ev = ev.shuffle(seed=seed).select(range(min(max_eval, ev.num_rows)))
        return tr, ev

    # OLD leaky split.
    l_tr, l_ev = _leaky_split(raw, eval_fraction=eval_fraction, seed=seed)
    l_tr, l_ev = _cap(l_tr, l_ev)
    leaky = _run_one(
        "leaky", base_model, l_tr, l_ev, tok, labels, id2label, label2id,
        lr=lr, lora_rank=lora_rank, epochs=epochs, batch_size=batch_size,
        max_length=max_length, resolved_device=resolved, seed=seed,
    )

    # NEW disjoint split.
    d_tr, d_ev, split_info = template_disjoint_split(raw, eval_fraction=eval_fraction, seed=seed)
    d_tr, d_ev = _cap(d_tr, d_ev)
    disjoint = _run_one(
        "disjoint", base_model, d_tr, d_ev, tok, labels, id2label, label2id,
        lr=lr, lora_rank=lora_rank, epochs=epochs, batch_size=batch_size,
        max_length=max_length, resolved_device=resolved, seed=seed,
    )

    out = {
        "device": resolved,
        "dataset": dataset,
        "base_model": base_model,
        "epochs": epochs,
        "lr": lr,
        "lora_rank": lora_rank,
        "batch_size": batch_size,
        "max_length": max_length,
        "seed": seed,
        "split_info_disjoint": split_info,
        "leaky": leaky,
        "disjoint": disjoint,
        "note": (
            "eval_loss is a split-sanity number only, NOT a quality metric — only the EuroPriv-Bench "
            "harness leaderboard measures model quality (KLU-54)."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("wrote %s", out_path)
    click.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
