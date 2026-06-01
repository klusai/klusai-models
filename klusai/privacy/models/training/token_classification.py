"""Token-classification training backend (transformers + peft LoRA).

The `xlmr-ner` family — XLM-R / mDeBERTa encoders fine-tuned to emit the harmonized KP BIOES
label space (``europriv_bench.taxonomy.bioes_labels()``) so a trained model is *directly*
scoreable on EuroPriv-Bench with no native→KP crosswalk (see the ``kp-model`` adapter).

Decided in KLU-11 (MLX rejected for this encoder family): ``AutoModelForTokenClassification`` +
PEFT ``TaskType.TOKEN_CLS`` LoRA on ``query_proj/key_proj/value_proj``. Compute is CPU-friendly
for a bounded smoke run; the same code runs on a CUDA droplet for a full run (KLU-14).

Training data is the merged LocalePack / ``klusai/ds-kp-general-*`` schema:
``{text, spans:[{start,end,label}], language, domain}`` — gold spans already carry KP labels.
We align those char spans onto the model tokenizer's subwords (offset mapping) to produce
per-subword BIOES labels, the standard HF token-classification setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from klusai.privacy.models.logger import get_logger

logger = get_logger("train.token_classification")


@dataclass
class TokenClassResult:
    """What a (smoke or full) run produces — enough to publish + report honestly."""

    output_dir: str
    publish_id: str
    num_labels: int
    train_examples: int
    eval_examples: int
    epochs: int
    eval_loss: float | None
    pushed: bool


def _bioes_from_spans(
    text: str,
    spans: list[dict],
    offsets: list[tuple[int, int]],
    word_ids: list[int | None],
    label2id: dict[str, str],
) -> list[int]:
    """Project gold char spans onto subword tokens as BIOES label ids (-100 on specials/continuations).

    Span→token: a *word* (group of subwords sharing a word_id) belongs to a span if the word's
    char range overlaps it. We BIOES at the word level then assign the tag to the word's first
    subword; continuation subwords get -100 (ignored by the loss), the canonical HF scheme. This
    keeps the trained label space byte-identical to the benchmark's whitespace-token BIOES at
    eval time (the harness re-tokenizes on whitespace, so only the *types* must match — they do).
    """
    n = len(offsets)
    # Group subword indices by word id, tracking each word's char span.
    words: dict[int, list[int]] = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        words.setdefault(wid, []).append(i)

    # First pass: per-word KP label (first overlapping span wins; gold spans are non-overlapping).
    word_label: dict[int, str | None] = {}
    for wid, idxs in words.items():
        ws = offsets[idxs[0]][0]
        we = offsets[idxs[-1]][1]
        lab: str | None = None
        for sp in spans:
            if ws < sp["end"] and we > sp["start"]:
                lab = sp["label"]
                break
        word_label[wid] = lab

    # Second pass: BIOES over the contiguous word sequence, then place on first subword.
    out = [-100] * n
    ordered = sorted(words)
    for pos, wid in enumerate(ordered):
        lab = word_label[wid]
        first_sub = words[wid][0]
        if lab is None:
            out[first_sub] = label2id["O"]
            continue
        prev_same = pos > 0 and word_label[ordered[pos - 1]] == lab
        next_same = pos + 1 < len(ordered) and word_label[ordered[pos + 1]] == lab
        if prev_same and next_same:
            tag = f"I-{lab}"
        elif prev_same and not next_same:
            tag = f"E-{lab}"
        elif not prev_same and next_same:
            tag = f"B-{lab}"
        else:
            tag = f"S-{lab}"
        out[first_sub] = label2id.get(tag, label2id["O"])
    return out


def train_token_classification(
    *,
    base_model: str,
    dataset: str,
    publish_id: str,
    output_dir: str,
    epochs: int = 1,
    lr: float = 3e-4,
    batch_size: int = 8,
    lora_rank: int = 8,
    max_length: int = 256,
    max_train: int | None = None,
    max_eval: int | None = None,
    eval_fraction: float = 0.05,
    seed: int = 0,
    push: bool = False,
    cpu: bool = True,
    threads: int = 4,
) -> TokenClassResult:
    """Fine-tune an encoder for KP token classification (LoRA), optionally push a merged model.

    Bounded by ``max_train``/``max_eval``/``epochs`` so a real-but-small CPU smoke run is feasible;
    drop the caps + raise epochs on a GPU for a full run. Publishes a *merged* model (LoRA folded
    into the base) so the europriv-bench ``kp-model`` adapter can ``from_pretrained`` it directly.
    """
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    from europriv_bench.taxonomy import bioes_labels

    if cpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        torch.set_num_threads(threads)

    labels = bioes_labels()
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}

    logger.info("loading %s (%d labels) + tokenizer", base_model, len(labels))
    tok = AutoTokenizer.from_pretrained(base_model)
    if not tok.is_fast:
        raise RuntimeError(f"{base_model} needs a fast tokenizer (offset mapping) for span alignment")

    raw = load_dataset(dataset, split="train")
    raw = raw.shuffle(seed=seed)
    n = raw.num_rows
    n_eval = max(1, int(n * eval_fraction))
    eval_rows = raw.select(range(n_eval))
    train_rows = raw.select(range(n_eval, n))
    if max_train is not None:
        train_rows = train_rows.select(range(min(max_train, train_rows.num_rows)))
    if max_eval is not None:
        eval_rows = eval_rows.select(range(min(max_eval, eval_rows.num_rows)))
    logger.info("dataset %s: %d train / %d eval examples", dataset, train_rows.num_rows, eval_rows.num_rows)

    def _encode(batch: dict) -> dict:
        enc = tok(
            batch["text"],
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        all_labels = []
        for i, spans in enumerate(batch["spans"]):
            word_ids = enc.word_ids(batch_index=i)
            all_labels.append(
                _bioes_from_spans(batch["text"][i], spans, enc["offset_mapping"][i], word_ids, label2id)
            )
        enc.pop("offset_mapping")
        enc["labels"] = all_labels
        return enc

    train_ds = train_rows.map(_encode, batched=True, remove_columns=train_rows.column_names)
    eval_ds = eval_rows.map(_encode, batched=True, remove_columns=eval_rows.column_names)

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
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        seed=seed,
        report_to=[],
        use_cpu=cpu,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    logger.info("training: epochs=%d lr=%g batch=%d lora_rank=%d", epochs, lr, batch_size, lora_rank)
    trainer.train()
    metrics = trainer.evaluate()
    eval_loss = float(metrics.get("eval_loss")) if "eval_loss" in metrics else None
    logger.info("eval_loss=%s", eval_loss)

    # Merge LoRA into the base so the published model is a plain token-classifier the benchmark
    # pipeline can load with from_pretrained (no peft required at inference).
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

    return TokenClassResult(
        output_dir=output_dir,
        publish_id=publish_id,
        num_labels=len(labels),
        train_examples=train_rows.num_rows,
        eval_examples=eval_rows.num_rows,
        epochs=epochs,
        eval_loss=eval_loss,
        pushed=pushed,
    )
