"""RES-127: pinned gateway + original KP corpus, resumable local MPS training.

No test split is read by this trainer. Select checkpoints by gateway-dev loss only;
quality claims belong to the separate EuroPriv harness scorecard.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def checked(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Input hash mismatch: {path}")
    return actual


def prepare(args):
    from datasets import Dataset, concatenate_datasets
    from huggingface_hub import hf_hub_download
    from pyarrow import parquet
    from train_xlmr560m_res97 import _encoder
    from transformers import AutoTokenizer

    from europriv_bench.gateway import flatten_conversation
    from europriv_bench.taxonomy import bioes_labels

    train_path = args.data / "artifacts/gateway-traffic-v1/train-kp.jsonl"
    audit_path = args.data / "docs/gateway-authoring/train-audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "pass" or audit["train_rows"] != 40000:
        raise ValueError("Gateway train audit has not passed")
    sources = [{"id": "gateway-train", "rows": 40000, "license": "CC-BY-4.0",
                "url": "https://github.com/klusai/klusai-datasets",
                "sha256": checked(train_path, audit["files"]["artifacts/gateway-traffic-v1/train-kp.jsonl"])}]
    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    if len(rows) != 40000:
        raise ValueError("Wrong gateway train count")
    columns = ["text", "spans", "language", "domain"]
    parts = [Dataset.from_list(rows).select_columns(columns)]
    inventory = json.loads((args.data / "conf/gateway_traffic_training_inventory.json").read_text())
    national_ids = {}
    for lang in ("ro", "en", "pl"):
        item = next(r for r in inventory["corpora"] if r["id"] == f"klusai/ds-kp-general-{lang}-50k")
        path = hf_hub_download(item["id"], item["artifact_path"], repo_type="dataset",
                               revision=item["revision"], local_files_only=True)
        checked(path, item["file_sha256"])
        table = parquet.read_table(path)
        legacy = Dataset(table).select_columns(columns)
        if len(legacy) != 50000:
            raise ValueError("Wrong legacy corpus count")
        national_ids[lang] = sum(s["label"] == "NATIONAL_ID" for spans in legacy["spans"] for s in spans)
        parts.append(legacy)
        sources.append({**item, "license": "CC-BY-4.0", "selected_rows": len(legacy)})
    if not national_ids["ro"] or not national_ids["pl"]:
        raise ValueError("Missing CNP/PESEL training coverage")
    dev_path = args.dev / "dev.jsonl"
    checked(dev_path, audit["files"]["artifacts/gateway-traffic-v1/dev.jsonl"])
    raw_dev = [json.loads(line) for line in dev_path.read_text().splitlines()]
    if len(raw_dev) != 1600 or any(r["split"] != "dev" for r in raw_dev):
        raise ValueError("Wrong dev split")
    dev_rows = [flatten_conversation(row) for row in raw_dev]
    # Keep the span schema identical across legacy, gateway train and dev.
    dev_rows = [{"text": r["text"], "spans": [{k: s[k] for k in ("start", "end", "label")}
                                            for s in r["spans"]]} for r in dev_rows]
    dev_texts = {r["text"] for r in dev_rows}
    train = concatenate_datasets(parts)
    if any(text in dev_texts for text in train["text"]):
        raise ValueError("Exact train/dev text overlap")
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    labels = bioes_labels()
    encode = _encoder(tok, {v: i for i, v in enumerate(labels)}, 512)
    prepared = args.scratch / "prepared" / args.arch
    prepared.mkdir(parents=True, exist_ok=True)
    for name, dataset in (("train", train), ("dev", Dataset.from_list(dev_rows))):
        encoded = dataset.map(encode, batched=True, remove_columns=dataset.column_names,
                              load_from_cache_file=False)
        encoded.save_to_disk(str(prepared / name))
    tok.save_pretrained(prepared / "tokenizer")
    manifest = {"sources": sources, "train_rows": len(train), "dev_rows": len(dev_rows),
                "legacy_national_id_spans": national_ids,
                "train_languages": dict(collections.Counter(train["language"])),
                "dev_sha256": digest(dev_path), "audit_sha256": digest(audit_path),
                "max_length": 512, "base": args.base, "labels": labels,
                "selection": "minimum gateway-dev token loss; never test",
                "limitations": ["Synthetic unreviewed gateway gold", "Legacy semantic disjointness unverified",
                                "Training truncates at 512; evaluation uses full-input overflow windows",
                                "TRAIN export joins parts with two newlines; harness dev/test uses one"]}
    write(prepared / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def train(args):
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    from klusai.privacy.models.training.token_classification import lora_target_modules

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for this experiment")
    torch.set_num_threads(8)
    set_seed(args.seed)  # Before classifier/LoRA initialization, not only Trainer construction.
    prepared = args.scratch / "prepared" / args.arch
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest["base"] != args.base:
        raise ValueError("Prepared base mismatch")
    labels = manifest["labels"]
    tok = AutoTokenizer.from_pretrained(prepared / "tokenizer")
    model = AutoModelForTokenClassification.from_pretrained(
        args.base, local_files_only=True, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={v: i for i, v in enumerate(labels)},
    )
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.TOKEN_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=lora_target_modules(args.base),
    ))
    out = args.scratch / "runs" / f"{args.arch}-seed{args.seed}"
    if (out / "result.json").exists():
        raise ValueError("Completed seed already exists; refusing overwrite")
    lr = 3e-4 if args.arch == "mdeberta-280m" else 2e-4
    pin = {"prepared_manifest_sha256": digest(prepared / "manifest.json"), "seed": args.seed,
           "epochs": 3, "lr": lr, "batch_size": 16, "dtype": "float32", "device": "mps",
           "group_by_length": True, "pad_to_multiple_of": 32, "torch_empty_cache_steps": 25,
           "gradient_checkpointing": True, "gradient_checkpointing_use_reentrant": False,
           "script_sha256": digest(__file__),
           "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "datasets")}}
    if (out / "pin.json").exists() and json.loads((out / "pin.json").read_text()) != pin:
        raise ValueError("Training resume identity drift")
    timing_path = out / "timing.json"
    if not timing_path.exists():
        write(timing_path, {"wall_started_at": time.time()})
    write(out / "pin.json", pin)
    class Progress(TrainerCallback):
        def on_log(self, training_args, state, control, **kwargs):
            write(out / "progress.json", {"step": state.global_step, "epoch": state.epoch,
                                          "mps_driver_bytes": torch.mps.driver_allocated_memory(),
                                          "mps_tensor_bytes": torch.mps.current_allocated_memory(),
                                          "last": state.log_history[-1], "updated": time.time()})

    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=3,
        per_device_train_batch_size=16, per_device_eval_batch_size=16,
        learning_rate=lr, eval_strategy="steps", eval_steps=1000, save_strategy="steps", save_steps=1000,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, seed=args.seed, data_seed=args.seed, report_to=[],
        dataloader_num_workers=0, dataloader_pin_memory=False, fp16=False, bf16=False, group_by_length=True,
        logging_steps=100, disable_tqdm=True, torch_empty_cache_steps=25,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=load_from_disk(str(prepared / "train")),
                      eval_dataset=load_from_disk(str(prepared / "dev")),
                      data_collator=DataCollatorForTokenClassification(tok, pad_to_multiple_of=32),
                      callbacks=[Progress()])
    if str(trainer.args.device) != "mps":
        raise RuntimeError(f"Unexpected device: {trainer.args.device}")
    checkpoint = get_last_checkpoint(str(out / "checkpoints")) if (out / "checkpoints").exists() else None
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    model.to("cpu").merge_and_unload().save_pretrained(out / "merged")
    tok.save_pretrained(out / "merged")
    elapsed = time.monotonic() - started
    history = json.loads((out / "resume-history.json").read_text()) if (out / "resume-history.json").exists() else []
    write(out / "result.json", {**pin, "seconds_this_session": elapsed,
                               "seconds_total": elapsed + sum(item["elapsed_seconds"] for item in history),
                               "wall_seconds_total": (time.time() -
                                                      json.loads(timing_path.read_text())["wall_started_at"]),
                               "resume_history": history,
                               "resumed_from": checkpoint, "training_metrics_not_quality": result.metrics,
                               "best_checkpoint": trainer.state.best_model_checkpoint,
                               "best_dev_loss_not_quality": trainer.state.best_metric,
                               "checkpoint_files": {p.name: digest(p) for p in (out / "merged").iterdir()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--arch", choices=("mdeberta-280m", "xlmr-560m"), default="mdeberta-280m")
    parser.add_argument("--base", required=True, help="Pinned local base snapshot")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else train)(args)


if __name__ == "__main__":
    main()
