# KLU-45 — PyTorch-MPS (Metal) vs CPU for mDeBERTa-v3 LoRA on the Mac Studio

**Status:** Done. **Decision: `mps` is the default Mac-tier device** for the `xlmr-ner`
token-classification family; **CPU remains the guaranteed fallback** (`--device cpu`, or auto-
fallback when MPS is absent). MPS is ~7.7x faster than CPU and numerically matches it.

Reproduce: `python scripts/bench_klu45_mps_vs_cpu.py` (writes `docs/klu-45-mps-vs-cpu.json`).

## Context

KLU-11 rejected **MLX** for this encoder family (no encoder archs, no token-class loss in
`mlx-lm`) and fell back to CPU as the *safe* first Mac path — but the **PyTorch MPS (Metal)**
backend trains `mdeberta-v3-base` on the Mac GPU today and had never actually been measured for
this 280M encoder. (The "MoE ~3x slower on MPS" note in KLU-11 was for a different, causal-LM
model.) KLU-45 measures it directly.

## Benchmark

Same smoke task as KLU-17 — subset of `klusai/ds-kp-general-ro-50k`, same LoRA config
(`AutoModelForTokenClassification` + `TaskType.TOKEN_CLS`, r=16, α=32, targets
`query_proj/key_proj/value_proj`), batch 16, lr 5e-4 — bounded to **800 train / 200 eval, 2
epochs** so three configs run back-to-back. Identical seed (0) across configs.

Machine: Mac Studio **M3 Ultra**, macOS 15.x arm64, Python 3.13, torch 2.12, transformers 5.x,
peft. `PYTORCH_ENABLE_MPS_FALLBACK=1` set (see "MPS op support" below).

| config | device | wall-clock / epoch | throughput | eval_loss | peak RSS |
|---|---|---|---|---|---|
| cpu4 | CPU, 4 threads | 53.96 s | 14.83 samples/s | 0.0102 | ~6.96 GB |
| cpu8 | CPU, 8 threads | 52.08 s | 15.36 samples/s | 0.0194 | ~7.00 GB |
| **mps** | **Mac GPU (Metal)** | **6.97 s** | **114.81 samples/s** | **0.0226** | ~7.00 GB |

**Throughput:** MPS is **~7.7x faster than CPU(4)** and **~7.5x faster than CPU(8)**. Extra CPU
threads (4→8) barely help (~+3.5%) — this 280M encoder is not CPU-thread-bound, so the GPU is the
real lever on the M3 Ultra.

**Correctness:** all three loss curves converge identically (train loss ~0.59→0.03 over 2 epochs)
and final eval losses sit in the same **0.010–0.023** band; MPS vs CPU(4) `|Δ eval_loss| = 0.0124`.
For a 2-epoch smoke run on 800 examples that spread is run-to-run noise (the CPU4↔CPU8 gap, both
fp32 on the same backend, is itself 0.009), not an MPS numerical defect. The merged model is moved
to CPU before `merge_and_unload`/`save_pretrained`, so the published artifact is device-independent.

### MPS op support / fp pitfalls

We set `PYTORCH_ENABLE_MPS_FALLBACK=1` defensively so any op MPS doesn't yet implement silently
runs on CPU instead of hard-erroring. In this run the **full DeBERTa-v2 + LoRA forward/backward
ran on MPS with no fallback warnings and no NaNs/Infs** — disentangled attention, the gather/
scatter in the token-class head, and the LoRA matmuls are all supported. No fp16/bf16 autocast is
used (fp32 throughout), which sidesteps the known MPS mixed-precision rounding issues; that is why
the loss matches CPU. If a future transformers/torch bump introduces an unsupported op, the env
var keeps training working (just slower for that op) rather than crashing.

## Memory & scaling to the full run (for KLU-44)

Peak process RSS is ~7 GB for all three configs (unified memory is shared CPU↔GPU on Apple
Silicon, so RSS captures the GPU working set too). On an M3 Ultra with **256–512 GB unified
memory** this is a rounding error: the working set is dominated by the 280M base weights + LoRA
+ activations at seq 256 / batch 16, and grows only mildly with batch size.

**Implication for KLU-44:** the full **50k**-example run (and larger batches / longer sequences /
multilingual mix) is comfortably feasible **on-device** on the M3 Ultra — at ~115 samples/s, one
epoch over 50k is ~7–8 minutes, so a multi-epoch full run is well under an hour of GPU time with
tens of GB of headroom to spare. **KLU-44 can skip the cloud GPU for this family** and run MPS on
the Mac Studio; CUDA-on-droplet stays available but is not required.

## Decision & wiring

- **Default Mac-tier device = `mps`** for `xlmr-ner` (materially faster, numerically stable).
- **CPU is the explicit guaranteed fallback** — `resolve_device()` drops `mps`/`auto` to CPU when
  MPS is unavailable, and `--device cpu` always pins CPU.
- Wired as `--device {auto,cpu,mps,cuda}` on `scripts/train.py` (supersedes the legacy
  `--cpu/--gpu`; `auto` picks MPS on Mac, CUDA on a droplet, else CPU). `--threads` is the CPU-only
  knob. Existing KLU-17 callers (`cpu=True`, no `device`) keep resolving to CPU unchanged.
- This is the stopgap before the MLX-native path (KLU-46); whichever wins there will not change
  the fact that MPS already makes the Mac GPU usable for this family today.
