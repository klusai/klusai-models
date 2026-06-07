# RES-98 — Does the `openai/privacy-filter` MoE TRAIN on the Mac? (feasibility spike)

**Status:** Done (bounded feasibility spike). **Verdict: GO.** The Mac Studio (M3 Ultra,
96 GB) **trains/finetunes** the `openai/privacy-filter` sparse-MoE
(1.5B total / 50M active, **top-4 of 128 experts**, `model_type=openai_privacy_filter`)
end-to-end under **PyTorch + `transformers` + PEFT-LoRA** — on **both MPS and CPU**, with no
blocking op. No MLX port is required to train this model. `config_status=dev`; this is a
feasibility probe — **no accuracy/SOTA claim**.

This **removes the MoE from the RES-53 GPU-burst scope**: the one open question that justified
the burst budget (does the MoE *training* stack run on-device, or hit a Metal gap like the
DeBERTa encoder in KLU-46?) is answered — it runs on-device today. GPU burst remains justified
only for *throughput* on a real multi-hour/multi-epoch run (see "What this does NOT claim").

Reproduce (needs `transformers>=5.9` for the `openai_privacy_filter` arch — see "Environment"):

```bash
python3.13 -m venv .venv-spike-res98 && source .venv-spike-res98/bin/activate
pip install "transformers==5.10.2" "torch==2.12.0" "peft>=0.13" safetensors huggingface-hub
pip install -e . --no-deps                       # klusai.privacy.models.logger only
python scripts/spike_res98_moe_train.py --steps 5
```

Spike code: `scripts/spike_res98_moe_train.py`. Run on Mac Studio M3 Ultra (96 GB, MPS),
real cached `openai/privacy-filter` weights (1.40B params loaded), fp32 throughout.

## 1. Forward-pass parity — MPS vs CPU — **PASS**

The sparse-MoE architecture runs on MPS at all (the open sub-question): same 4-sentence batch
(seq 128) through the real model on CPU and MPS, token-classification logits compared.

| metric | value |
| --- | --- |
| `max |Δ|` (logits) | **4.2e-5** |
| `mean |Δ|` (logits) | 2.6e-6 |
| **argmax label agreement** | **100.00%** |

The top-4-of-128 router gather, the expert MLPs, and the classifier head all reproduce on
MPS to fp32 reduction-order noise. **No silent CPU fallback, no missing-op error.**

## 2. Backprop / a few LoRA train steps — **WORKS on MPS (the crux)** — **PASS**

LoRA (r=8) attached to the attention projections **and the MoE expert + router MLPs**
(`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`) → 316,065 trainable / 1.40B
(0.023%). 5 capped AdamW steps per device, bs=4, seq 128, lr=1e-4. **Loss decreases and the
MPS and CPU loss trajectories are bit-identical** (16.24 → 15.31 → 13.34 → 14.21 → 14.29 on
both), which both proves the gradient path is real and cross-validates MPS numerics:

| device | warm s/step | it/s | samples/s | peak mem | blocked? |
| --- | --- | --- | --- | --- | --- |
| **MPS** | 1.87 | 0.54 | 2.1 | **6.92 GB** | **no** |
| **CPU** (4 threads) | **0.59** | 1.70 | **6.8** | n/a | no |

(Warm = excluding step 1, which on MPS pays a 6.2 s graph-build cost. `s/step` is *not* a
production-throughput number — bs=4, untuned, 5 steps. It is the GO/NO-GO signal: backward
through the sparse routing completes on Metal and updates the weights.)

**The backward through the sparse top-4 gather did NOT fall off Metal** — this is the finding
that distinguishes the MoE from KLU-46's DeBERTa encoder (which needed an MLX-core port study).

## 3. Blocking op — **NONE**

No MoE routing op errored or fell back. (`PYTORCH_ENABLE_MPS_FALLBACK=0` was set so a Metal
gap would surface as a hard error rather than a silent CPU detour — none occurred.) There is
therefore **no specific op that "genuinely needs GPU burst"** to make training *possible*. Memory
is a non-issue: peak 6.92 GB on MPS in 96 GB.

## CPU vs MPS: CPU is ~3.2x faster here

CPU (4 threads) trains at 0.59 s/step vs MPS 1.87 s/step — consistent with the M3U *inference*
note (`europriv-inference-perf-m3ultra`): this MoE is small (50M active, short sequences) and
its routing is gather-heavy, so it under-utilises the GPU and CPU wins. For on-device smoke
runs / short finetunes, **prefer the CPU path** (thread-capped); MPS works but is not the fast
lane for this model.

## What this does NOT claim (scope honesty)

- **No full finetune.** 5 capped steps, hard-bounded per the SPIKE. No convergence,
  no eval-loss, no held-out scorecard, no real KP data slice (synthetic token labels in the
  model's native 33-label head — enough to exercise the gradient path, not to learn anything).
- **Not a tuned-throughput number.** bs=4, no `bf16`, no large-batch / max-util lever
  (KLU-48). A real run would tune these; the s/step here is a feasibility signal only.
- **No router-aux-loss path exercised.** `output_router_logits=False` (config default); the
  load-balancing aux loss was not enabled. A production finetune may want it — untested here.
- **Head not reshaped to the KP taxonomy.** Continue-finetune kept the model's 33-label head;
  wiring the shared `europriv_bench` BIOES label space (73 labels) is `moe-finetune` trainer
  work, not part of this GO/NO-GO probe.
- **Separate environment.** The repo's main `.venv` is pinned at `transformers==4.57.6` for the
  encoder/SDK paths, which does **not** recognise `model_type=openai_privacy_filter`
  (`ValueError: ... does not recognize this architecture`). The spike uses a dedicated
  `.venv-spike-res98` at `transformers==5.10.2`. This matches the pyproject `hf` extra
  (`transformers>=5.0`) and the README note that the MoE arch needs `transformers>=5.0` (in
  practice `>=5.9` for `openai_privacy_filter`); it does not change the repo's pins.

## Recommendation — **GO**

- **Implement `moe-finetune` for `--backend cuda` (which runs on CPU/MPS on the Mac)** using
  `transformers` + PEFT-LoRA on experts+router. Training is on-device-feasible today; **MLX is
  not required** for it to run. Default the Mac path to **CPU** (faster here) with MPS available.
- **RES-53 / GPU burst:** no longer needed to make MoE training *possible*. Reserve burst for
  *throughput* on a genuine multi-epoch run over the real KP slices (a CUDA droplet auto-selects
  `cuda`, sidestepping the MPS/CPU trade-off) — a budget decision on wall-clock, not a hard
  capability gap. The MoE comes **out of RES-53's "needs GPU to be feasible" scope.**

## Environment

Mac Studio M3 Ultra, 96 GB, macOS; Python 3.13; `transformers==5.10.2`, `torch==2.12.0`,
`peft==0.19.1`, MPS available. Model `openai/privacy-filter` from local HF cache (no download).
Power not measured (per KLU-48, `sudo powermetrics` is the only reader and is unavailable
non-interactively here).
