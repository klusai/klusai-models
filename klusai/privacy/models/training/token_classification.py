"""Token-classification training backend (transformers + peft LoRA).

The `xlmr-ner` family — XLM-R / mDeBERTa encoders fine-tuned to emit the harmonized KP BIOES
label space (``europriv_bench.taxonomy.bioes_labels()``) so a trained model is *directly*
scoreable on EuroPriv-Bench with no native→KP crosswalk (see the ``kp-model`` adapter).

Decided in KLU-11 (MLX rejected for this encoder family): ``AutoModelForTokenClassification`` +
PEFT ``TaskType.TOKEN_CLS`` LoRA on ``query_proj/key_proj/value_proj``. Device is selectable via
``resolve_device`` (KLU-45): the Mac GPU (Metal/MPS) is the default Mac-tier device — ~7.7x faster
than CPU and numerically matching it — with CPU as the guaranteed fallback; the same code runs on
a CUDA droplet for a full run (KLU-14).

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
    device: str = "cpu"
    batch_size: int = 0          # KLU-48: effective per-device batch (post max-util probe)
    max_util: bool = False       # KLU-48: was the max-utilization profile active
    num_workers: int = 0         # KLU-48: DataLoader workers used
    bf16: bool = False           # KLU-48: bf16 autocast used on MPS
    train_samples_per_s: float = 0.0  # KLU-48: Trainer's steady-state train throughput


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


def resolve_device(device: str | None, *, cpu: bool) -> str:
    """Resolve the requested training device to one of ``cpu`` / ``mps`` / ``cuda``.

    KLU-45 added the Mac GPU (Metal/MPS) path. Selection precedence:

    * an explicit ``device`` ("cpu"/"mps"/"cuda") is honored if actually available, else falls
      back to CPU with a warning (CPU is the *guaranteed* fallback on every tier);
    * ``device="auto"`` (or ``None``) picks the best Mac-tier device: MPS when present, else CPU
      — CUDA is only chosen by ``auto`` when no MPS but a CUDA GPU is visible (DO droplet);
    * the legacy ``cpu=True`` bool (KLU-17 default) maps to ``device="cpu"`` when ``device`` is
      unset, so existing callers keep their behavior.
    """
    import torch

    def _mps_ok() -> bool:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()

    req = (device or ("auto" if not cpu else "cpu")).lower()

    if req == "cpu":
        return "cpu"
    if req == "mps":
        if _mps_ok():
            return "mps"
        logger.warning("MPS requested but unavailable; falling back to CPU (the guaranteed tier).")
        return "cpu"
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    # auto: best Mac-tier device first (MPS), then CUDA droplet, else CPU.
    if _mps_ok():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --- KLU-48: max-utilization profile for the Mac GPU (MPS) -----------------------------------
#
# At batch 16 the human observed only ~68 W on the M3 Ultra and suspected GPU starvation. KLU-48
# measured it directly (scripts/bench_klu48_max_util.py + docs/klu-48-max-util.md). The honest
# finding is that the premise is WRONG for this model: the 280M mDeBERTa encoder is
# **memory-bandwidth-bound and already near-saturated at batch 16** on this GPU. Measured on real
# ds-kp-general-ro data (2400 train / 2 epochs):
#   * batch 16, single-process      → 189 samples/s   eval_loss 0.0005   (the optimum)
#   * batch 64, 8 workers           →  94 samples/s   eval_loss 0.028    (0.49x — SLOWER)
#   * batch 256 (auto-fill memory)  →  24 samples/s   eval_loss 2.73     (0.14x, 72/96 GB thrash,
#                                       ~13 s/step, optimizer starved → no convergence)
# Synthetic full-length batches confirm throughput is flat (~58–64 samples/s) from batch 16→96.
# bf16 autocast gives no throughput win (bandwidth-bound) and risks drift, so we stay fp32
# (numerically matched to CPU; KLU-45). Multi-process DataLoader workers give no win for this
# light collation AND *deadlock at process exit* under macOS `spawn`.
#
# Conclusion: **plain batch-16 single-process fp32 already IS the max-utilization config for this
# encoder** — the ~68 W is mostly intrinsic to a 280M model on a 76-TFLOP GPU, not starvation, and
# no batch size takes it to 150 W. So the Mac default does NOT scale the batch or add workers (it
# would only regress). The `--max-util` flag, the memory-guarded auto-probe, and the (opt-in)
# workers below are kept as infrastructure for models/regimes that DO benefit — the denser MoE
# track — but for xlmr-ner they are off by default. See the README + doc for the full table.

# Candidate batch sizes to probe (largest first), capped at the measured throughput-stable range.
# The probe steps down until a real fwd+bwd fits AND stays under the memory guardrail, so this is
# a ceiling. 64 is the top because throughput is already flat by 32 and 128+ risks memory thrash.
MAX_UTIL_BATCH_CANDIDATES: tuple[int, ...] = (64, 48, 32, 16)

# Don't let the probe pick a batch whose working set exceeds this fraction of total unified memory
# — past ~50% the MPS allocator starts thrashing (KLU-48: 72 GB/96 GB at batch 256 → 13 s/step).
MAX_UTIL_MEM_FRACTION = 0.5


@dataclass
class MaxUtilProfile:
    """Resolved max-utilization knobs for an MPS training session (KLU-48)."""

    enabled: bool
    batch_size: int
    eval_batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    bf16: bool
    probe_log: list[str]


# Cap the eval batch independently of the (large) train batch. Eval has no backward pass, so it
# doesn't need a huge batch to saturate; more importantly a single giant-shape eval step triggers
# its own one-time MPS graph compile that can dwarf the eval itself. A moderate eval batch keeps
# eval fast and the train batch big (where saturation actually matters).
MAX_UTIL_EVAL_BATCH_CAP = 64


def _probe_mps_batch_size(
    model,
    collator,
    sample_features: list[dict],
    candidates: tuple[int, ...],
    *,
    bf16: bool,
) -> tuple[int, list[str]]:
    """Find the largest batch from ``candidates`` that survives one real fwd+bwd on MPS *and* stays
    under the memory guardrail.

    We build a real padded batch from cached, already-tokenized features and run a forward +
    backward (the memory high-water mark is the backward pass). A candidate is rejected if it OOMs/
    MPS-allocation-fails OR if its peak driver allocation exceeds ``MAX_UTIL_MEM_FRACTION`` of total
    unified memory (the regime where the allocator thrashes — KLU-48). Returns the chosen batch and
    a human-readable probe log. The model is left with grads zeroed; the caller moves it back.
    """
    import torch

    # Total unified memory (bytes) for the guardrail; fall back to a large value if unavailable.
    try:
        import psutil  # optional; not a hard dep

        total_mem = psutil.virtual_memory().total
    except Exception:
        total_mem = int(
            __import__("subprocess").run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            ).stdout.strip()
            or 0
        ) or (96 * 1024**3)
    mem_cap = total_mem * MAX_UTIL_MEM_FRACTION

    log: list[str] = []
    model = model.to("mps")
    model.train()
    pool = sample_features
    for bs in candidates:
        if bs > len(pool):
            # Not enough cached examples to form this batch — tile the pool so the probe is honest
            # about memory (same #tokens) without needing a huge eval slice.
            reps = (bs + len(pool) - 1) // len(pool)
            feats = (pool * reps)[:bs]
        else:
            feats = pool[:bs]
        try:
            torch.mps.empty_cache()
            batch = collator(feats)
            batch = {k: v.to("mps") for k, v in batch.items()}
            ctx = (
                torch.autocast(device_type="mps", dtype=torch.bfloat16)
                if bf16
                else _nullcontext()
            )
            with ctx:
                out = model(**batch)
                loss = out.loss
            loss.backward()
            peak = torch.mps.driver_allocated_memory()
            model.zero_grad(set_to_none=True)
            torch.mps.synchronize()
            torch.mps.empty_cache()
            if peak > mem_cap:
                pct = 100 * peak / total_mem
                log.append(f"batch={bs}: over mem guardrail ({pct:.0f}% > {MAX_UTIL_MEM_FRACTION:.0%})")
                logger.info(
                    "max-util probe: batch=%d uses %.0f%% unified mem (> %.0f%% cap); stepping down",
                    bs, pct, MAX_UTIL_MEM_FRACTION * 100,
                )
                continue
            log.append(f"batch={bs}: OK ({100 * peak / total_mem:.0f}% unified mem)")
            logger.info("max-util probe: batch=%d fits on MPS (%.0f%% unified mem)",
                        bs, 100 * peak / total_mem)
            return bs, log
        except (RuntimeError, MemoryError) as e:  # MPS OOM surfaces as RuntimeError
            msg = str(e).splitlines()[0][:120]
            log.append(f"batch={bs}: failed ({msg})")
            logger.warning("max-util probe: batch=%d did not fit (%s); stepping down", bs, msg)
            model.zero_grad(set_to_none=True)
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    # Nothing fit (unexpected): fall back to the smallest candidate.
    log.append(f"all probes failed; using {candidates[-1]}")
    return candidates[-1], log


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def resolve_max_util_profile(
    resolved_device: str,
    *,
    max_util: bool,
    batch_size: int,
    batch_override: int | None,
    num_workers: int | None,
    bf16: bool,
    model=None,
    collator=None,
    sample_features: list[dict] | None = None,
) -> MaxUtilProfile:
    """Resolve the KLU-48 max-utilization knobs.

    Only active on ``mps`` (CPU/CUDA paths are unchanged — CUDA already saturates and CPU has no GPU
    to feed). When ``max_util`` is on we: (a) bump the per-device batch to the measured throughput
    sweet-spot — an explicit ``batch_override``, else auto-probe ``MAX_UTIL_BATCH_CANDIDATES``
    against real memory with the ``MAX_UTIL_MEM_FRACTION`` guardrail (so we never approach the
    unified-memory limit where MPS thrashes — KLU-48); (b) turn on ``num_workers``>0 so collation
    runs off the main process and never stalls the GPU; (c) optionally enable bf16 autocast (off by
    default — KLU-48 found it gives no MPS throughput win and risks numerical drift). On non-MPS
    devices this is a no-op that returns the incoming ``batch_size`` and zero workers.
    """
    if not max_util or resolved_device != "mps":
        return MaxUtilProfile(
            enabled=False,
            batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            bf16=False,
            probe_log=[],
        )

    # Workers are opt-in: KLU-48 found multi-process DataLoaders give no throughput win for this
    # light-collation encoder and *deadlock at process exit* under macOS `spawn`. Default to 0
    # (single-process); a caller who passes --num-workers explicitly accepts that tradeoff.
    workers = 0 if num_workers is None else num_workers

    if batch_override is not None:
        chosen, probe_log = batch_override, [f"batch={batch_override}: explicit override (no probe)"]
    elif model is not None and collator is not None and sample_features:
        chosen, probe_log = _probe_mps_batch_size(
            model, collator, sample_features, MAX_UTIL_BATCH_CANDIDATES, bf16=bf16
        )
    else:
        chosen, probe_log = MAX_UTIL_BATCH_CANDIDATES[0], ["no probe inputs; using top candidate"]

    return MaxUtilProfile(
        enabled=True,
        batch_size=chosen,
        eval_batch_size=min(chosen, MAX_UTIL_EVAL_BATCH_CAP),
        num_workers=workers,
        # pin_memory is a CUDA concept; on MPS/unified memory it's a no-op, keep it off.
        pin_memory=False,
        # persistent_workers=True deadlocks at process exit with macOS `spawn` for this light
        # workload (KLU-48) — keep it off so workers are torn down cleanly each epoch.
        persistent_workers=False,
        bf16=bf16,
        probe_log=probe_log,
    )


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
    device: str | None = None,
    max_util: bool | None = None,
    max_util_batch_size: int | None = None,
    num_workers: int | None = None,
    bf16: bool = False,
) -> TokenClassResult:
    """Fine-tune an encoder for KP token classification (LoRA), optionally push a merged model.

    Bounded by ``max_train``/``max_eval``/``epochs`` so a real-but-small CPU smoke run is feasible;
    drop the caps + raise epochs on a GPU for a full run. Publishes a *merged* model (LoRA folded
    into the base) so the europriv-bench ``kp-model`` adapter can ``from_pretrained`` it directly.

    **KLU-48 max-utilization profile (MPS only, opt-in).** ``max_util`` auto-probes a larger
    per-device batch (memory-guarded, unless ``max_util_batch_size`` overrides) and can feed the GPU
    from ``num_workers`` DataLoader processes. It is **off by default** on every device: KLU-48
    measured that this 280M encoder is memory-bandwidth-bound and already near-saturated at batch 16
    on the M3 Ultra, so scaling the batch / adding workers does not raise throughput and can regress
    it badly (batch 256 → 0.14x + no convergence). The profile is kept as infrastructure for denser
    models that benefit; for xlmr-ner the Mac default is plain batch-16 single-process fp32 (the
    measured optimum). ``num_workers`` defaults to 0 (workers deadlock at exit under macOS spawn —
    opt in explicitly). ``bf16`` is off by default (no MPS throughput win + drift risk; fp32 is
    numerically matched to CPU, KLU-45/48). See docs/klu-48-max-util.md.
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

    resolved_device = resolve_device(device, cpu=cpu)
    if resolved_device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        torch.set_num_threads(threads)
        logger.info("device=cpu (threads=%d)", threads)
    else:
        # On MPS some ops may not be implemented; PYTORCH_ENABLE_MPS_FALLBACK=1 lets them run on
        # CPU instead of hard-erroring. Harmless on CUDA. Thread cap is a CPU-only knob.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        logger.info("device=%s", resolved_device)

    # KLU-48: the *measured* max-utilization config for this 280M encoder on MPS is plain batch-16
    # single-process fp32 — the Mac GPU is memory-bandwidth-bound and already near-saturated, so
    # scaling the batch and/or adding DataLoader workers does not raise throughput and can regress
    # it badly (docs/klu-48-max-util.md). So the Mac default leaves the batch/workers untouched
    # (max_util OFF) and `--max-util` is an explicit opt-in for models/regimes that do benefit
    # (the denser MoE track). Passing max_util explicitly (True/False) always wins.
    if max_util is None:
        max_util = False

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

    # KLU-48: resolve the max-utilization profile. On MPS this auto-probes the largest batch that
    # fits in unified memory (so the per-step GEMMs are big enough to saturate the GPU) and turns
    # on DataLoader workers; on CPU/CUDA it is a no-op preserving the incoming batch_size.
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
        logging_steps=20,
        seed=seed,
        report_to=[],
        # use_cpu=True pins Trainer to CPU; for mps/cuda we let Trainer place the model on the
        # accelerator it detects (MPS is auto-selected on Apple Silicon when use_cpu is False).
        use_cpu=(resolved_device == "cpu"),
        # KLU-48 max-util knobs (no-ops when profile is disabled / num_workers=0).
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
    logger.info("training: epochs=%d lr=%g batch=%d lora_rank=%d", epochs, lr, eff_batch_size, lora_rank)
    train_out = trainer.train()
    # Steady-state training throughput as the Trainer measures it (excludes model load / probe /
    # save) — the honest figure for the KLU-48 saturation comparison.
    train_samples_per_s = float(train_out.metrics.get("train_samples_per_second", 0.0) or 0.0)
    metrics = trainer.evaluate()
    eval_loss = float(metrics.get("eval_loss")) if "eval_loss" in metrics else None
    logger.info("eval_loss=%s", eval_loss)

    # Merge LoRA into the base so the published model is a plain token-classifier the benchmark
    # pipeline can load with from_pretrained (no peft required at inference). Move back to CPU
    # first so the merge/save is device-independent (and avoids MPS fp32 save quirks).
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

    return TokenClassResult(
        output_dir=output_dir,
        publish_id=publish_id,
        num_labels=len(labels),
        train_examples=train_rows.num_rows,
        eval_examples=eval_rows.num_rows,
        epochs=epochs,
        eval_loss=eval_loss,
        pushed=pushed,
        device=resolved_device,
        batch_size=eff_batch_size,
        max_util=profile.enabled,
        num_workers=profile.num_workers,
        bf16=profile.bf16,
        train_samples_per_s=train_samples_per_s,
    )
