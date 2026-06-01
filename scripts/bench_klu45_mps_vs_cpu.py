#!/usr/bin/env python3
"""KLU-45 — Benchmark PyTorch-MPS (Metal) vs CPU for mDeBERTa-v3 LoRA on the Mac Studio.

Runs the *same* bounded token-classification smoke task (subset of ``ds-kp-general-ro-50k``,
same LoRA config as KLU-17) on three device configs and compares throughput + correctness +
peak memory:

    * ``cpu`` @ 4 threads   (the KLU-17 default)
    * ``cpu`` @ 8 threads
    * ``mps``               (Apple Silicon GPU / Metal)

For each run we capture: wall-clock/epoch, train & eval samples/s, final train loss, eval loss
(correctness — MPS must match CPU within noise), the loss curve (logged steps), and peak process
RSS (a proxy for memory pressure; the M3 Ultra's unified memory is shared CPU+GPU so RSS captures
the GPU working set too). Results are written to ``docs/klu-45-mps-vs-cpu.json`` and summarized to
stdout; the prose decision lives in ``docs/klu-45-mps-vs-cpu.md``.

Run (from repo root, after ``make install``):

    python scripts/bench_klu45_mps_vs_cpu.py                  # full benchmark (default caps)
    python scripts/bench_klu45_mps_vs_cpu.py --max-train 200  # quicker
    python scripts/bench_klu45_mps_vs_cpu.py --devices mps    # one config only

This is a one-off measurement harness, not a shipped code path; it imports the production
``train_token_classification`` backend unchanged (only the ``device=`` flag added in KLU-45).
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import click

from klusai.privacy.models.logger import get_logger
from klusai.privacy.models.training.token_classification import (
    resolve_device,
    train_token_classification,
)

logger = get_logger("bench.klu45")

# Same base/dataset/LoRA config as the KLU-17 smoke run, just smaller caps so three back-to-back
# runs are feasible while staying a real (non-toy) measurement.
BASE_MODEL = "microsoft/mdeberta-v3-base"
DATASET = "klusai/ds-kp-general-ro-50k"


def _peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS (kB on Linux); this harness is Mac-only so treat as bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _run_one(label: str, *, device: str, threads: int, **kw) -> dict:
    logger.info("=== benchmark config: %s (device=%s threads=%s) ===", label, device, threads)
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    res = train_token_classification(
        base_model=BASE_MODEL,
        dataset=DATASET,
        publish_id="klusai/kp-deid-mdeberta-280m",
        output_dir=f"runs/_bench_klu45_{label}",
        device=device,
        threads=threads,
        push=False,
        **kw,
    )
    wall = time.perf_counter() - t0
    samples_per_s = (res.train_examples * kw.get("epochs", 1)) / wall
    return {
        "label": label,
        "requested_device": device,
        "resolved_device": res.device,
        "threads": threads if res.device == "cpu" else None,
        "train_examples": res.train_examples,
        "eval_examples": res.eval_examples,
        "epochs": res.epochs,
        "wall_clock_s": round(wall, 2),
        "wall_clock_per_epoch_s": round(wall / max(1, res.epochs), 2),
        "samples_per_s": round(samples_per_s, 2),
        "eval_loss": res.eval_loss,
        "peak_rss_mb_after": round(_peak_rss_mb(), 1),
        "peak_rss_mb_delta": round(_peak_rss_mb() - rss_before, 1),
    }


@click.command()
@click.option("--devices", default="cpu4,cpu8,mps", help="Comma list of configs: cpu4,cpu8,mps.")
@click.option("--epochs", type=int, default=2)
@click.option("--max-train", type=int, default=800, help="Bounded train examples (KLU-17 used 4000).")
@click.option("--max-eval", type=int, default=200)
@click.option("--batch-size", type=int, default=16)
@click.option("--lr", type=float, default=5e-4)
@click.option("--lora-rank", type=int, default=16)
@click.option("--seed", type=int, default=0)
@click.option("--out", default="docs/klu-45-mps-vs-cpu.json")
def main(devices, epochs, max_train, max_eval, batch_size, lr, lora_rank, seed, out):
    configs = {
        "cpu4": dict(label="cpu4", device="cpu", threads=4),
        "cpu8": dict(label="cpu8", device="cpu", threads=8),
        "mps": dict(label="mps", device="mps", threads=4),
    }
    chosen = [configs[c.strip()] for c in devices.split(",") if c.strip()]

    common = dict(
        epochs=epochs, lr=lr, batch_size=batch_size, lora_rank=lora_rank,
        max_train=max_train, max_eval=max_eval, seed=seed,
    )
    logger.info(
        "MPS available? resolve('mps')=%s ; config: %s",
        resolve_device("mps", cpu=False), common,
    )

    results = [_run_one(**cfg, **common) for cfg in chosen]

    # Correctness: do MPS eval losses match CPU within noise?
    by_dev = {r["label"]: r for r in results}
    summary = {
        "base_model": BASE_MODEL,
        "dataset": DATASET,
        "config": common,
        "results": results,
    }
    if "cpu4" in by_dev and "mps" in by_dev:
        cl, ml = by_dev["cpu4"]["eval_loss"], by_dev["mps"]["eval_loss"]
        if cl and ml:
            summary["mps_vs_cpu4_eval_loss_abs_delta"] = round(abs(ml - cl), 6)
            # >1 means MPS is faster than CPU(4 threads).
            summary["mps_vs_cpu4_speedup"] = round(
                by_dev["mps"]["samples_per_s"] / by_dev["cpu4"]["samples_per_s"], 2
            ) if by_dev["cpu4"]["samples_per_s"] else None

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2))

    print("\n==== KLU-45 benchmark summary ====")
    hdr = f"{'config':8} {'device':6} {'wall/epoch(s)':14} {'samples/s':10} {'eval_loss':12} {'peakRSS(MB)':12}"
    print(hdr)
    for r in results:
        print(
            f"{r['label']:8} {r['resolved_device']:6} {r['wall_clock_per_epoch_s']:<14} "
            f"{r['samples_per_s']:<10} {str(r['eval_loss'])[:11]:12} {r['peak_rss_mb_after']:<12}"
        )
    if "mps_vs_cpu4_speedup" in summary:
        print(f"\nMPS speedup vs cpu4: {summary['mps_vs_cpu4_speedup']}x ; "
              f"eval_loss |Δ| vs cpu4: {summary['mps_vs_cpu4_eval_loss_abs_delta']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
