# KLU-48 — Max-utilization Mac training profile (and what actually saturates the M3 Ultra)

**Status:** Done. **Decision: the Mac (`mps`) default stays plain batch-16 single-process fp32 —
that is already the measured max-utilization config for the `xlmr-ner` 280M encoder.** A
`--max-util` opt-in profile (memory-guarded batch auto-probe + optional DataLoader workers) ships
as infrastructure for denser models (the MoE track) but is **off by default** here because for
this encoder it only *regresses* throughput.

Reproduce: `python scripts/bench_klu48_max_util.py` (writes `docs/klu-48-max-util.json`).

## Context

The human observed only ~68 W (package/GPU) training the kp-deid mDeBERTa finetune at batch 16 on
the M3 Ultra and suspected the GPU was *starved* — i.e. that small batch-16 GEMMs were leaving the
Metal cores idle, and that scaling the per-device batch up to fill unified memory (96 GB) plus
DataLoader workers + fp16/bf16 would push utilisation toward ≥150 W and materially higher
samples/s. KLU-48 implemented that profile and **measured it before shipping**.

## What we measured

Machine: Mac Studio **M3 Ultra**, 96 GB unified memory, macOS 15.x arm64, Python 3.13, torch
2.12, transformers 4.57, peft. Same kp-deid finetune slice as KLU-45 (`mdeberta-v3-base` LoRA r=16
on `klusai/ds-kp-general-ro-50k`), `PYTORCH_ENABLE_MPS_FALLBACK=1`. Throughput is the Trainer's
steady-state `train_samples_per_second` (excludes model load / batch-probe / save).

### Before/after on the real finetune slice (2400 train, 2 epochs)

| config | per-device batch | workers | samples/s | eval_loss | peak unified mem |
|---|---|---|---|---|---|
| **before (default)** | 16 | 0 | **189** | 0.0005 | ~7.5 GB |
| after — modest bump | 64 | 8 | 94 (**0.49x**) | 0.028 | ~22 GB |
| after — "fill memory" | 256 (auto-probe ceiling) | 8 | 24 (**0.14x**) | 2.73 | **~72 GB** |

The "after" runs are **slower**, not faster. Batch 256 is catastrophic: it pushes peak unified
memory to ~72 GB of 96 GB, at which point the MPS allocator/graph cache thrashes (~13 s/step after
the first eval), and with only ~5 optimizer steps/epoch the model never converges (train loss
stuck ~0.78, eval_loss 2.73 vs the baseline's 0.0005).

### Why: the GPU is bandwidth-bound, not starved

A pure steady-state micro-benchmark (warmed-up fwd+bwd+step, no DataLoader, full seq-256 batches)
shows throughput is **flat** across batch size — the classic signature of a memory-bandwidth-bound
kernel, not a launch-overhead-bound one:

| batch | samples/s | peak unified mem |
|---|---|---|
| 16 | 60.7 | ~11.6 GB |
| 32 | 64.2 | ~21 GB |
| 64 | 57.7 | ~41 GB |
| 96 | 64.0 | ~60 GB |

From batch 16 to 96 throughput moves only ~58→64 samples/s while memory grows ~5x. A 280M encoder
simply does not have enough arithmetic per byte to keep a 76-TFLOP-class GPU busy; bigger batches
move more bytes for the same flat throughput. **There is no batch size that takes *this* model to
≥150 W** — the ~68 W is largely intrinsic to the model size on this GPU, not starvation.

### bf16 autocast: no win

bf16 autocast on MPS matched fp32 throughput within noise (batch 32: 62.7 vs 64.0 samples/s) — as
expected for a bandwidth-bound, autocast-upcasting path. Since it buys nothing and risks numerical
drift, we keep **fp32** (numerically matched to CPU; KLU-45) and leave `--bf16` off by default.

### DataLoader workers: no win + a macOS bug

Multi-process workers gave no throughput benefit (collation here is light — pad + tensor build)
and, more decisively, **deadlock at process exit under macOS `spawn`** (both `persistent_workers`
on and off). They are therefore opt-in (`--num-workers N`, default 0) and never persistent.

## Decision & wiring

- **Mac default = plain batch-16 single-process fp32** for `xlmr-ner` — it is the measured
  throughput optimum and never regresses the batch-16/fp32 numerics. CPU stays the guaranteed
  fallback (KLU-45).
- **`--max-util` is an explicit opt-in**, off by default on every device. When on (and only on
  `mps`) it auto-probes the largest batch from `(64,48,32,16)` that both fits *and* stays under
  `MAX_UTIL_MEM_FRACTION` (50%) of unified memory — the guardrail that prevents the batch-256
  thrash regime — or honors `--max-util-batch-size`. It is kept as infrastructure for the denser
  MoE/causal-LM track, where large-batch GEMMs *do* saturate the GPU.
- `--num-workers` (default 0, opt-in; macOS spawn exit-hang) and `--bf16` (default off; no win)
  are exposed for experimentation. Eval batch is capped (`MAX_UTIL_EVAL_BATCH_CAP=64`) so a large
  train batch never triggers a giant-shape eval graph compile.
- Wired on `scripts/train.py` and `scripts/full_run_klu44.py`; resolution lives in
  `resolve_max_util_profile()` next to `resolve_device()`. The benchmark harness is
  `scripts/bench_klu48_max_util.py`.

## Power readout

`sudo powermetrics` is the only power reader present on this machine (`macmon`/`asitop` are not
installed), and `sudo` is **not** available non-interactively in the automated run environment, so
this report quantifies saturation via **throughput + peak unified memory** instead of a wattage
number. To read sustained package/GPU watts yourself while a run trains, in a second terminal:

```bash
sudo powermetrics --samplers gpu_power -i 1000 -n 10
```

Expect the default batch-16 run to sit near the ~68 W the human already observed — which, per the
measurements above, is this 280M encoder's natural draw on the M3 Ultra, not a starvation artifact
that a bigger batch could fix.
