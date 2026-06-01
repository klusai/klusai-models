#!/usr/bin/env python3
"""KLU-48 — Benchmark the max-utilization MPS profile vs the old batch-16 config.

The human observed only ~68 W (package/GPU) training the kp-deid mDeBERTa finetune at batch 16 on
the M3 Ultra and suspected GPU starvation. This harness tests that hypothesis by measuring
**before vs after** the KLU-48 max-utilization profile (auto-probed larger batch + DataLoader
workers) on the *same* kp-deid finetune slice:

    * ``before``  — batch 16, no workers (the KLU-45/KLU-17 config; ``--no-max-util``)
    * ``after``   — ``--max-util``: auto-probed larger batch + ``num_workers`` workers

The finding (docs/klu-48-max-util.md) is that the after-config is *slower* — this 280M encoder is
memory-bandwidth-bound and already near-saturated at batch 16 — so max-util ships **off by default**
for xlmr-ner; this harness is what produced that evidence.

For each run we capture throughput (train samples/s), wall-clock, final eval loss (numerics must
match the before run within noise — that's the precision gate), the effective batch, worker count,
and **peak MPS unified memory** (``torch.mps.driver_allocated_memory``). Power (W) is read via
``sudo powermetrics`` when sudo is non-interactive; otherwise we print the exact command for the
human to run alongside the after-run (and report GPU memory + throughput as the saturation proxy).

Run (from repo root, after ``make install``):

    python scripts/bench_klu48_max_util.py                       # before + after, default slice
    python scripts/bench_klu48_max_util.py --max-train 1500       # bigger slice
    python scripts/bench_klu48_max_util.py --after-batch-size 128 # pin the after batch (skip probe)

Imports the production ``train_token_classification`` backend unchanged (only flips the KLU-48
flags), so it measures the shipped code path, not a fork.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import click

from klusai.privacy.models.logger import get_logger
from klusai.privacy.models.training.token_classification import (
    resolve_device,
    train_token_classification,
)

logger = get_logger("bench.klu48")

# Same kp-deid finetune slice as KLU-45 (mDeBERTa-v3 LoRA on ds-kp-general-ro-50k), so before/after
# are directly comparable to the existing MPS baseline.
BASE_MODEL = "microsoft/mdeberta-v3-base"
DATASET = "klusai/ds-kp-general-ro-50k"

# How a human can read package/GPU power on macOS (needs sudo; -i ms, -n samples).
POWERMETRICS_CMD = "sudo powermetrics --samplers gpu_power -i 1000 -n 10"


def _sudo_noninteractive() -> bool:
    return subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0


def _peak_mps_mem_mb() -> float | None:
    """Peak MPS driver-allocated unified memory in MB (the GPU working set), if on MPS."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / (1024 * 1024)
    except Exception:
        pass
    return None


def _reset_mps_peak() -> None:
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _run_one(label: str, *, max_util: bool, batch_size: int, after_batch_size: int | None,
             num_workers: int | None, bf16: bool, **kw) -> dict:
    logger.info("=== %s (max_util=%s batch=%s workers=%s bf16=%s) ===",
                label, max_util, after_batch_size or batch_size, num_workers, bf16)
    _reset_mps_peak()
    t0 = time.perf_counter()
    res = train_token_classification(
        base_model=BASE_MODEL,
        dataset=DATASET,
        publish_id="klusai/kp-deid-mdeberta-280m",
        output_dir=f"runs/_bench_klu48_{label}",
        device="mps",
        push=False,
        max_util=max_util,
        max_util_batch_size=after_batch_size if max_util else None,
        num_workers=num_workers,
        bf16=bf16,
        batch_size=batch_size,
        **kw,
    )
    wall = time.perf_counter() - t0
    wall_samples_per_s = (res.train_examples * kw.get("epochs", 1)) / wall
    return {
        "label": label,
        "resolved_device": res.device,
        "max_util": res.max_util,
        "effective_batch_size": res.batch_size,
        "num_workers": res.num_workers,
        "bf16": res.bf16,
        "train_examples": res.train_examples,
        "epochs": res.epochs,
        "wall_clock_s": round(wall, 2),
        # Steady-state training throughput (Trainer-reported) is the saturation headline; it
        # excludes one-time model load / batch-probe / save. wall_samples_per_s is the whole-run
        # figure (load+probe+train+eval+save) for reference.
        "samples_per_s": round(res.train_samples_per_s, 2),
        "wall_samples_per_s": round(wall_samples_per_s, 2),
        "eval_loss": res.eval_loss,
        "peak_mps_mem_mb": round(m, 1) if (m := _peak_mps_mem_mb()) is not None else None,
    }


@click.command()
@click.option("--epochs", type=int, default=2)
@click.option("--max-train", type=int, default=1200, help="Bounded train examples for the slice.")
@click.option("--max-eval", type=int, default=300)
@click.option("--before-batch-size", type=int, default=16, help="Old config batch (the ~68 W run).")
@click.option("--after-batch-size", type=int, default=None,
              help="Pin the max-util batch (default: auto-probe the largest that fits).")
@click.option("--num-workers", type=int, default=None, help="Max-util DataLoader workers.")
@click.option("--bf16/--no-bf16", default=False, help="bf16 autocast in the after run (default off).")
@click.option("--lr", type=float, default=5e-4)
@click.option("--lora-rank", type=int, default=16)
@click.option("--seed", type=int, default=0)
@click.option("--out", default="docs/klu-48-max-util.json")
def main(epochs, max_train, max_eval, before_batch_size, after_batch_size, num_workers, bf16,
         lr, lora_rank, seed, out):
    if resolve_device("mps", cpu=False) != "mps":
        raise click.UsageError("MPS not available — this max-util benchmark is Mac-GPU-only.")

    common = dict(
        epochs=epochs, lr=lr, lora_rank=lora_rank,
        max_train=max_train, max_eval=max_eval, seed=seed,
    )

    sudo_ok = _sudo_noninteractive()
    logger.info("sudo non-interactive? %s", sudo_ok)
    if not sudo_ok:
        logger.warning(
            "Cannot read package/GPU power non-interactively. Run this in another terminal "
            "*during the after-run* to capture sustained W:\n    %s", POWERMETRICS_CMD
        )

    before = _run_one(
        "before", max_util=False, batch_size=before_batch_size, after_batch_size=None,
        num_workers=0, bf16=False, **common,
    )
    after = _run_one(
        "after", max_util=True, batch_size=before_batch_size, after_batch_size=after_batch_size,
        num_workers=num_workers, bf16=bf16, **common,
    )

    speedup = round(after["samples_per_s"] / before["samples_per_s"], 2) if before["samples_per_s"] else None
    loss_delta = (
        round(abs((after["eval_loss"] or 0) - (before["eval_loss"] or 0)), 6)
        if before["eval_loss"] is not None and after["eval_loss"] is not None else None
    )

    summary = {
        "base_model": BASE_MODEL,
        "dataset": DATASET,
        "config": common,
        "before": before,
        "after": after,
        "samples_per_s_speedup": speedup,
        "eval_loss_abs_delta_after_vs_before": loss_delta,
        "power_readable": sudo_ok,
        "power_command": POWERMETRICS_CMD,
        "power_tools_present": {
            "powermetrics": shutil.which("powermetrics") is not None,
            "macmon": shutil.which("macmon") is not None,
            "asitop": shutil.which("asitop") is not None,
        },
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2))

    print("\n==== KLU-48 max-util before/after ====")
    hdr = f"{'config':8} {'batch':6} {'workers':8} {'samples/s':10} {'eval_loss':12} {'peakMPS(MB)':12}"
    print(hdr)
    for r in (before, after):
        print(
            f"{r['label']:8} {r['effective_batch_size']:<6} {str(r['num_workers']):8} "
            f"{r['samples_per_s']:<10} {str(r['eval_loss'])[:11]:12} {str(r['peak_mps_mem_mb']):12}"
        )
    print(f"\nthroughput speedup (after/before): {speedup}x ; "
          f"eval_loss |Δ| after vs before: {loss_delta}")
    if not sudo_ok:
        print(f"\n[power] sudo is interactive here — read sustained package/GPU W with:\n    {POWERMETRICS_CMD}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
