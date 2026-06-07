#!/usr/bin/env python3
"""RES-98 spike — can the ``openai/privacy-filter`` MoE TRAIN/finetune on the Mac?

Bounded feasibility SPIKE (GO/NO-GO), NOT a production trainer. It answers the one open
question that justifies (or kills) the RES-53 GPU-burst budget: the sparse-MoE
(1.5B total / 50M active, top-4 of 128 experts, ``model_type=openai_privacy_filter``)
already does inference on this box (CPU-best per the M3U perf note) — but does its TRAINING
stack run under PyTorch-MPS, or does a MoE routing op fall off Metal and force a CPU path /
an MLX port (as KLU-46 spiked for the DeBERTa encoder)?

What it does (capped, a handful of steps — never a full finetune):
  1. Forward-pass parity: MPS vs CPU on the real MoE (does the sparse arch run on MPS at all,
     or error / silently fall back?). Reports max|Δ| / argmax agreement on token logits.
  2. A few LoRA training steps (experts + router) on a tiny synthetic token-class slice — does
     backprop work on MPS? Captures warm steps/sec + peak memory. CPU is run for comparison.
  3. If MPS chokes on a MoE op, the exception is caught and reported as the BLOCKING OP (the
     thing that genuinely needs GPU burst).

Honesty guards: config_status=dev; this is a feasibility probe, NO accuracy/SOTA claim.

Requires transformers>=5.9 for the ``openai_privacy_filter`` arch (the repo's main .venv is
pinned at 4.57 for the encoder/SDK paths, which does NOT recognise this arch — run this in the
dedicated ``.venv-spike-res98`` per docs/res-98-moe-train-spike.md).

    python scripts/spike_res98_moe_train.py            # parity + MPS & CPU train steps
    python scripts/spike_res98_moe_train.py --steps 5  # cap the number of train steps
"""

from __future__ import annotations

import os
import time

import click

# Cap CPU BLAS threads (M3U perf note: 4 beats 28 for this small MoE). Set before torch import.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
# Do NOT silently fall MoE ops back to CPU — we want to SEE a Metal gap, not mask it.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

from klusai.privacy.models.logger import get_logger  # noqa: E402

logger = get_logger("spike-res98")

BASE_MODEL = "openai/privacy-filter"


def _tiny_batch(tok, model, n: int, seq_len: int, device, seed: int = 0):
    import torch

    rng = torch.Generator().manual_seed(seed)
    sentences = [
        f"Contact person {i} at account {i:06d} in city {i}, phone 070000{i:04d}." for i in range(n)
    ]
    enc = tok(sentences, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    n_labels = model.config.num_labels
    enc["labels"] = torch.randint(0, n_labels, enc["input_ids"].shape, generator=rng)
    enc["labels"][enc["attention_mask"] == 0] = -100  # ignore padding in CE loss
    return {k: v.to(device) for k, v in enc.items()}


def forward_parity(tok, seq_len: int) -> None:
    """Run the SAME inputs through the MoE on CPU and MPS; compare token logits."""
    import torch
    from transformers import AutoModelForTokenClassification

    logger.info("--- forward-pass parity: MPS vs CPU ---")
    model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, dtype=torch.float32).eval()
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "loaded %s: %.2fB params, %d experts top-%d, %d labels",
        BASE_MODEL, total / 1e9, model.config.num_local_experts,
        model.config.num_experts_per_tok, model.config.num_labels,
    )

    cpu_batch = _tiny_batch(tok, model, n=4, seq_len=seq_len, device="cpu")
    with torch.no_grad():
        cpu_logits = model.to("cpu")(**{k: v for k, v in cpu_batch.items() if k != "labels"}).logits.float()

    try:
        mps_model = model.to("mps")
        mps_batch = {k: v.to("mps") for k, v in cpu_batch.items()}
        with torch.no_grad():
            mps_logits = mps_model(**{k: v for k, v in mps_batch.items() if k != "labels"}).logits.float().cpu()
    except Exception as e:  # noqa: BLE001
        logger.error("MPS FORWARD BLOCKED — op fell off Metal: %r", e)
        logger.error("BLOCKING OP (forward): %s", str(e).splitlines()[0][:300])
        return

    dmax = (cpu_logits - mps_logits).abs().max().item()
    dmean = (cpu_logits - mps_logits).abs().mean().item()
    agree = (cpu_logits.argmax(-1) == mps_logits.argmax(-1)).float().mean().item()
    logger.info("forward parity PASS: max|Δ|=%.2e mean|Δ|=%.2e argmax-agreement=%.2f%%", dmax, dmean, 100 * agree)


def train_steps(tok, device_str: str, steps: int, n: int, seq_len: int, bs: int) -> dict | None:
    """Attach LoRA to experts+router, run a few real backward/opt steps. Returns metrics or None on block."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForTokenClassification

    if device_str == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available; skipping MPS train.")
        return None
    if device_str == "cpu":
        torch.set_num_threads(4)

    torch.manual_seed(0)
    device = torch.device(device_str)
    model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, dtype=torch.float32)

    # LoRA onto the MoE expert MLPs + the attention projections + the router gate. The expert
    # and router module names are the whole point of the spike — if backprop through the sparse
    # top-k gather works on MPS, these are the gradients that flow.
    lora = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    try:
        model = get_peft_model(model, lora)
    except Exception as e:  # noqa: BLE001
        logger.error("LoRA attach failed on %s: %r", device_str, e)
        return None
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("[%s] LoRA: %d trainable / %d total (%.3f%%)", device_str, trainable, total, 100 * trainable / total)

    try:
        model = model.to(device)
    except Exception as e:  # noqa: BLE001
        logger.error("[%s] .to(device) failed: %r", device_str, e)
        return None

    batch = _tiny_batch(tok, model, n=n, seq_len=seq_len, device=device)
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)

    if device_str == "mps":
        torch.mps.empty_cache()

    step_times: list[float] = []
    last_loss = float("nan")
    for step in range(steps):
        sl = slice((step * bs) % n, (step * bs) % n + bs)
        sub = {k: v[sl] for k, v in batch.items()}
        s = time.perf_counter()
        opt.zero_grad()
        try:
            out = model(**sub)
            out.loss.backward()
            opt.step()
        except Exception as e:  # noqa: BLE001
            logger.error("[%s] TRAIN STEP BLOCKED — backward/op fell off device: %r", device_str, e)
            logger.error("[%s] BLOCKING OP (backward): %s", device_str, str(e).splitlines()[0][:300])
            return {"device": device_str, "blocked": True, "error": str(e).splitlines()[0][:300]}
        if device_str == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - s
        step_times.append(dt)
        last_loss = float(out.loss.item())
        logger.info("[%s] step %d/%d loss=%.4f %.3fs/step (bs=%d seq=%d)", device_str, step + 1, steps, last_loss, dt, bs, seq_len)

    warm = step_times[1:] or step_times  # drop first (lazy alloc / graph build)
    s_per_step = sum(warm) / len(warm)
    peak_gb = None
    if device_str == "mps":
        # torch 2.12 mps has no reset_peak_memory_stats; driver_allocated_memory is the total
        # the MPS allocator has reserved after the run — a fair peak proxy for this capped probe.
        peak_gb = torch.mps.driver_allocated_memory() / 1e9
    metrics = {
        "device": device_str,
        "blocked": False,
        "s_per_step": s_per_step,
        "it_per_s": 1.0 / s_per_step,
        "bs": bs,
        "seq_len": seq_len,
        "samples_per_s": bs / s_per_step,
        "peak_mem_gb": peak_gb,
        "final_loss": last_loss,
    }
    logger.info(
        "[%s] DONE — warm ~%.3f s/step (%.2f it/s, %.1f samp/s) peak_mem=%s",
        device_str, s_per_step, metrics["it_per_s"], metrics["samples_per_s"],
        f"{peak_gb:.2f}GB" if peak_gb else "n/a (cpu)",
    )
    return metrics


@click.command()
@click.option("--steps", default=5, show_default=True, help="Train steps per device (CAP — never a full run).")
@click.option("--n-examples", default=16, show_default=True)
@click.option("--seq-len", default=128, show_default=True)
@click.option("--batch-size", default=4, show_default=True)
@click.option("--skip-cpu", is_flag=True, help="Skip the CPU comparison train run.")
def main(steps: int, n_examples: int, seq_len: int, batch_size: int, skip_cpu: bool) -> None:
    import torch
    from transformers import AutoTokenizer

    logger.info("RES-98 MoE-on-Mac train spike — transformers MPS path. CAP=%d steps/device.", steps)
    logger.info("mps_available=%s", torch.backends.mps.is_available())
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    forward_parity(tok, seq_len=seq_len)

    logger.info("--- MPS training steps ---")
    mps_m = train_steps(tok, "mps", steps=steps, n=n_examples, seq_len=seq_len, bs=batch_size)

    cpu_m = None
    if not skip_cpu:
        logger.info("--- CPU training steps (comparison / fallback) ---")
        cpu_m = train_steps(tok, "cpu", steps=steps, n=n_examples, seq_len=seq_len, bs=batch_size)

    logger.info("=== VERDICT INPUTS ===")
    logger.info("MPS: %s", mps_m)
    logger.info("CPU: %s", cpu_m)


if __name__ == "__main__":
    main()
