"""Privacy-filter continue-finetuning in the isolated Transformers 5 environment.

Attention LoRA + trainable router/classifier; packed expert tensors stay frozen.
Exports a merged AutoModelForTokenClassification for europriv's kp-model adapter.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def align_spans(spans, offsets, label2id):
    """BIOES on every subtoken; preserve adjacent entities of the same type."""
    owners = []
    for start, end in offsets:
        owners.append(
            next((i for i, s in enumerate(spans) if start < s["end"] and end > s["start"]), None)
            if end > start
            else None
        )
    labels = []
    for i, ((start, end), owner) in enumerate(zip(offsets, owners)):
        if end <= start:
            labels.append(-100)
        elif owner is None:
            labels.append(label2id["O"])
        else:
            before = i > 0 and owners[i - 1] == owner
            after = i + 1 < len(owners) and owners[i + 1] == owner
            prefix = "I" if before and after else "E" if before else "B" if after else "S"
            labels.append(label2id[f"{prefix}-{spans[owner]['label']}"])
    return labels


def load_rows(path, max_train=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if max_train is not None:
        rows = rows[:max_train]
    if not rows:
        raise ValueError("Training data is empty")
    from europriv_bench.taxonomy import bioes_labels

    valid = {label.split("-", 1)[1] for label in bioes_labels() if label != "O"}
    for row in rows:
        end = 0
        for span in sorted(row["spans"], key=lambda s: s["start"]):
            if span["label"] not in valid or not (end <= span["start"] < span["end"] <= len(row["text"])):
                raise ValueError("Unknown label, invalid offset, or overlapping gold spans")
            end = span["end"]
    return rows


def train_moe(
    *,
    base_model,
    dataset,
    output_dir,
    epochs=3,
    lr=1e-4,
    batch_size=4,
    lora_rank=8,
    max_train=None,
    device="cpu",
    threads=4,
    max_length=256,
    max_steps=-1,
    seed=0,
    save_steps=250,
    resume=None,
    gradient_accumulation_steps=1,
    revision=None,
):
    if min(epochs, batch_size, lora_rank, threads, gradient_accumulation_steps, save_steps) < 1 or lr <= 0:
        raise ValueError("Epochs, batch size, rank, threads, accumulation, save interval, and LR must be positive")
    if max_train is not None and max_train < 1:
        raise ValueError("max_train must be positive")
    import torch
    import transformers
    from datasets import Dataset
    from packaging.version import Version

    if Version(transformers.__version__) < Version("5.9"):
        raise RuntimeError("moe-finetune requires the separate .venv-moe: make install-moe (Transformers >=5.9)")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    from europriv_bench.taxonomy import bioes_labels

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device in (None, "auto") else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu for smoke")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    torch.set_num_threads(threads)
    set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(dataset, max_train)
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    config = AutoConfig.from_pretrained(base_model, revision=revision)
    if config.model_type != "openai_privacy_filter":
        raise ValueError("moe-finetune supports only openai_privacy_filter checkpoints")
    labels = bioes_labels()
    config.id2label = dict(enumerate(labels))
    config.label2id = {label: i for i, label in enumerate(labels)}
    # Keep native Viterbi decoding metadata out of the KP checkpoint; generic pipeline uses argmax.
    tok = AutoTokenizer.from_pretrained(base_model, revision=revision)
    if not tok.is_fast:
        raise ValueError("A fast tokenizer is required for character offsets")
    tok.model_max_length = max_length
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    features = []
    for row in rows:
        encoded = tok(
            row["text"],
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )
        for i, offsets in enumerate(encoded["offset_mapping"]):
            feature = {key: encoded[key][i] for key in ("input_ids", "attention_mask")}
            feature["labels"] = align_spans(row["spans"], offsets, config.label2id)
            if any(label != -100 for label in feature["labels"]):
                features.append(feature)
    if not features:
        raise ValueError("No supervised tokens after tokenization")
    manifest = {
        "base_model": base_model,
        "base_revision": getattr(config, "_commit_hash", None),
        "dataset_sha256": fingerprint,
        "rows": len(rows),
        "chunks": len(features),
        "tokens_per_epoch": sum(sum(f["attention_mask"]) for f in features),
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "seed": seed,
        "lr": lr,
        "lora_rank": lora_rank,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "device": device,
        "dtype": "float32",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "adaptation": "attention LoRA + full router and classifier; frozen experts",
    }
    manifest_path = output / "training-manifest.json"
    if resume:
        previous = json.loads(manifest_path.read_text())
        for key in (
            "base_model",
            "base_revision",
            "dataset_sha256",
            "batch_size",
            "max_length",
            "seed",
            "lr",
            "lora_rank",
            "gradient_accumulation_steps",
            "epochs",
        ):
            if previous[key] != manifest[key]:
                raise ValueError(f"Resume configuration changed: {key}")
    elif manifest_path.exists():
        raise ValueError("Output already has a training manifest; use --resume or a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        revision=revision,
        config=config,
        ignore_mismatched_sizes=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=["score", "router"],
        ),
    )
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    if not any("router" in name for name in trainable) or not any(".score." in name for name in trainable):
        raise RuntimeError("Router and KP head must be trainable")
    manifest["trainable_parameters"] = trainable
    manifest["trainable_total"] = sum(trainable.values())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Trainer owns optimizer/scheduler/RNG checkpoints and exact data-position restoration.
    counts = {"tokens": 0, "padded_tokens": 0}
    measurements = []

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    class MeasuredTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            counts["tokens"] += int(inputs["attention_mask"].sum().item())
            counts["padded_tokens"] += inputs["input_ids"].numel()
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch=num_items_in_batch)

    class Timing(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            sync()
            self.start = time.perf_counter()
            self.before = dict(counts)

        def on_step_end(self, args, state, control, **kwargs):
            sync()
            measurement = {
                "step": state.global_step,
                "seconds": time.perf_counter() - self.start,
                **{key: counts[key] - self.before[key] for key in counts},
            }
            measurements.append(measurement)
            with (output / "throughput.jsonl").open("a") as stream:
                stream.write(json.dumps(measurement) + "\n")

    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=1,
        report_to=[],
        seed=seed,
        data_seed=seed,
        use_cpu=device == "cpu",
        dataloader_num_workers=0,
        dataloader_pin_memory=device == "cuda",
        fp16=False,
        bf16=False,
    )
    if args.device.type != device:
        raise ValueError(f"Trainer selected {args.device}, but requested {device}")
    trainer = MeasuredTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(features),
        data_collator=DataCollatorForTokenClassification(tok),
        callbacks=[Timing()],
    )
    if resume == "latest":
        from transformers.trainer_utils import get_last_checkpoint

        resume = get_last_checkpoint(str(output))  # None: interrupted before the first saved optimizer step.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output / f"checkpoint-{trainer.state.global_step}"))
    # Trainer saves optimizer/scheduler/RNG at the final step as well as save_steps boundaries.
    if device == "cuda":
        manifest["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        manifest["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    import resource
    import sys

    manifest["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    merged = model.to("cpu").merge_and_unload()
    merged.save_pretrained(output / "merged", safe_serialization=True)
    tok.save_pretrained(output / "merged")
    warm = measurements[1:] or measurements
    duration = sum(m["seconds"] for m in warm)
    manifest.update(
        {
            "completed_steps": trainer.state.global_step,
            "warm_steps": len(warm),
            "warm_seconds": duration,
            "warm_tokens_per_second": sum(m["tokens"] for m in warm) / duration if duration else None,
            "warm_padded_tokens_per_second": sum(m["padded_tokens"] for m in warm) / duration if duration else None,
            "checkpoint": str(output / "merged"),
            "quality_claim": None,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
