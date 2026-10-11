"""RES-104 TAB train-only LoRA, using RES-97 alignment and RES-72 hyperparameters.

Fixed single arm: three epochs, LR 3e-4, rank 16, batch 16, fp32, 256-token windows.
Select the lowest TAB-dev loss checkpoint (selection signal only, never a quality claim).
Overflow windows cover whole documents; no TAB test rows enter this process.
Epoch checkpoints include optimizer/RNG state and are resumed automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from klusai.privacy.models.training.token_classification import _bioes_from_spans, lora_target_modules

BASE_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"


def encode_windows(rows, tokenizer, label2id):
    from datasets import Dataset

    features = []
    for row in rows:
        enc = tokenizer(
            row["text"],
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=0,
        )
        for i, ids in enumerate(enc["input_ids"]):
            features.append(
                {
                    "input_ids": ids,
                    "attention_mask": enc["attention_mask"][i],
                    "labels": _bioes_from_spans(
                        row["text"], row["spans"], enc["offset_mapping"][i], enc.word_ids(batch_index=i), label2id
                    ),
                }
            )
    return Dataset.from_list(features)


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    from europriv_bench.taxonomy import bioes_labels

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    ap.add_argument("--data", type=Path, default=Path("artifacts/res104-tab"))
    ap.add_argument("--manifest", type=Path, default=Path("docs/res104-tab-manifest.json"))
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["status"] != "license_and_document_split_gate_passed":
        raise ValueError("Data provenance gate must pass before training")
    if not torch.backends.mps.is_available():
        raise RuntimeError("This registered run requires local MPS")
    torch.set_num_threads(4)
    set_seed(args.seed)  # before initializing the randomly initialized classifier and LoRA
    output = Path(f"runs/res104-tab-seed{args.seed}")
    work = Path(f"runs/res104-tab-seed{args.seed}-checkpoints")
    result_path = Path(f"runs/res104-tab-seed{args.seed}-train.json")
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if (
            previous["manifest_sha256"] != hashlib.sha256(args.manifest.read_bytes()).hexdigest()
            or not (output / "model.safetensors").exists()
        ):
            raise ValueError("Completed run has a different manifest or missing weights; refusing stale reuse")
        print(f"Already complete: {result_path}", flush=True)
        return
    rows = {}
    for split in ("train", "dev"):
        payload = (args.data / f"{split}.jsonl").read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["splits"][split]["jsonl_sha256"]:
            raise ValueError(f"{split} hash changed after audit")
        rows[split] = [json.loads(line) for line in payload.decode().splitlines()]
    base = "FacebookAI/xlm-roberta-large"
    labels = bioes_labels()
    label2id = {label: i for i, label in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(base, revision=BASE_REVISION)
    datasets = {s: encode_windows(r, tok, label2id) for s, r in rows.items()}
    print({s: {"docs": len(rows[s]), "windows": len(ds)} for s, ds in datasets.items()}, flush=True)
    model = AutoModelForTokenClassification.from_pretrained(
        base, revision=BASE_REVISION, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=lora_target_modules(base),
        ),
    )
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(work),
        num_train_epochs=3,
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        fp16=False,
        bf16=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        data_collator=DataCollatorForTokenClassification(tok),
    )
    resume = get_last_checkpoint(str(work)) if work.exists() else None
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=resume)
    model.to("cpu").merge_and_unload().save_pretrained(output)
    tok.save_pretrained(output)
    (output / "TAB_LICENSE.txt").write_text((args.data / "LICENSE.txt").read_text())
    result = {
        "arm": "tab_train_only",
        "seed": args.seed,
        "base": base,
        "base_revision": BASE_REVISION,
        "output": str(output),
        "selection": "minimum TAB-dev loss over epochs 1/2/3; not a quality metric",
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "dev_selection_loss": trainer.state.best_metric,
        "train_docs": len(rows["train"]),
        "dev_docs": len(rows["dev"]),
        "train_windows": len(datasets["train"]),
        "dev_windows": len(datasets["dev"]),
        "epochs": 3,
        "lr": 3e-4,
        "rank": 16,
        "batch_size": 16,
        "dtype": "float32",
        "device": "mps",
        "max_length": 256,
        "overflow_stride": 0,
        "test_used_for_selection": False,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "resumed_from": resume,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "log_history": trainer.state.log_history,
    }
    partial = result_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    partial.replace(result_path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
