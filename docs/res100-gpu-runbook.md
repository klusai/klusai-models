# RES-100 — privacy-filter continue-finetune wave

Status: local CPU proof; CUDA throughput and full-run quality remain unverified.
The initial `origin/main` check found only `NotImplementedError` in `scripts/train.py`.
Do not provision hardware or publish from this worktree. These CUDA commands are for the
coordinator to run later on an already authorized machine.

## Environment and model

Keep encoder training in `.venv` with `.[hf]` / Transformers 4.57.6. Run MoE in a separate
Python 3.13 `.venv-moe`: Transformers 5.10.2, Torch 2.12.0, PEFT 0.19.1 (`.[moe]`).
The privacy-filter architecture requires >=5.9. `make install-moe BENCH_DIR=/path/to/europriv-bench`
installs the harness and MoE extra without modifying the encoder environment.

Base: [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter), Apache-2.0,
revision `7ffa9a043d54d1be65afb281eddf0ffbe629385b`. The pretrained representation is retained;
the native 33-label classifier is replaced by the harness KP BIOES taxonomy. Attention
`q/k/v/o_proj` receive rank-8 LoRA; routers and classifier train in full; packed expert tensors
stay frozen. RES-98's module-name list did **not** adapt its raw expert parameters. This
trainer records every trainable parameter rather than claiming expert-LoRA coverage.

Use fp32, eager attention, microbatch 4, sequence length 256, accumulation 4 (effective batch 16),
AdamW at 1e-4, three epochs, seeds 0/1/2 sequentially. No router auxiliary loss, quantization,
FlashAttention, or bf16 is enabled. The encoder batch-16 optimum is not assumed to apply here.

## Data recipe and provenance

```bash
export PYTHONPATH="$PWD:/path/to/klusai-datasets"
.venv-moe/bin/python scripts/prepare_res100_data.py --output-dir runs/res100-data \
  > runs/res100-prepare.log 2>&1
```

The recipe reuses RES-97: first 10,000 rows per language from
`klusai/ds-kp-general-{ro,en,pl,de,fr,es,it,nl}-50k`; seed-0 template/subject carve;
balanced 40,000-row draw; template-disjoint 4% internal validation removed. Add all 1,010
curated TAB **train** documents once, with no oversampling. The data seed stays 0 across model
seeds. Long documents are tokenized into nonoverlapping 256-token chunks, retaining all chunks
and projecting original character offsets onto every subtoken. This is not “truncate TAB to
its first paragraph.” Boundary-crossing entities are labeled within each chunk.

KP source cards declare CC-BY-4.0. TAB's corpus is
[MIT](https://github.com/NorskRegnesentral/text-anonymization-benchmark/blob/master/LICENSE.txt),
not the journal article's CC-BY license. The data preparer verifies the known TAB train SHA256,
uses the existing `klusai-datasets` curation/crosswalk, and records dropped misaligned documents.
It reads HF Parquet through `hf_hub_download` + PyArrow, never streaming datasets. Each source
gets a URL, license, revision, split, and file hash in `manifest.json`; output JSONLs also get
hashes. Unknown or restrictive licenses stop preparation. Keep the manifests and required
attribution with derived artifacts. Piiranha is never loaded for training.

TAB dev/test are excluded from training. TAB test is now **in-domain supervised transfer** for
this model, not a TAB-zero-shot result; any static harness contamination label must be qualified.
Compare on the fixed TAB test config only after the full run, with >=2 seeds and Wilson /
paired bootstrap CIs. A win needs a CI-disjoint held-out delta; losses are not quality evidence.

## Full run on an existing CUDA machine

```bash
mkdir -p runs/res100-logs
for seed in 0 1 2; do
  .venv-moe/bin/python scripts/train.py moe-finetune \
    --base openai/privacy-filter --revision 7ffa9a043d54d1be65afb281eddf0ffbe629385b \
    --dataset runs/res100-data/train.jsonl --out klusai/kp-deid-moe-res100 \
    --backend cuda --device cuda --epochs 3 --lr 0.0001 --batch-size 4 --lora-rank 8 \
    --max-length 256 --gradient-accumulation-steps 4 --save-steps 250 --seed "$seed" \
    --output-dir "runs/res100-seed$seed" > "runs/res100-logs/seed$seed.log" 2>&1
done
```

CPU is the Mac default for this MoE; explicit unavailable devices fail rather than silently
falling back. To resume, repeat the identical command with `--resume latest` (or a specific
`checkpoint-N`). Dataset fingerprint, base revision, seed, and training configuration must
match. Checkpoints include optimizer, scheduler, and RNG states; the final step is saved too.
Only the newest two intermediate checkpoints are retained. `throughput.jsonl` is appended at
every optimizer step; `training-manifest.json` records rows, chunks, token totals, versions,
parameters, and final throughput. `merged/` is the portable scoring artifact; `checkpoint-N/`
is for resumption. Inference stays on CPU. A mid-step interruption resumes from the last save.

## Fresh Ubuntu bootstrap

Prerequisites: Ubuntu with working NVIDIA driver, git, python3, python3-venv, ca-certificates,
GitHub read access to the three repos, and HF read access if datasets become private. Use a
persistent disk and a foreground session or tmux; do not use nohup/setsid. The script installs
Python 3.13 through uv, checks out all three requested git refs, installs the isolated MoE
stack, prepares data, then runs seeds sequentially. It checks CUDA availability, records exact
refs and installed packages, resumes saved seeds, and never provisions or uploads.

```bash
# Copy this script from the reviewed RES-100 commit, then supply that same ref as argument 1.
bash scripts/bootstrap_res100_cuda.sh <REVIEWED_RES100_GIT_REF> \
  43fafaa24cc635c115f1e1cb210e8a2bcb869020 \
  736665aa1c8908c73f6d0dd19b75d4541cf97b46 /work/res100-wave
```

The model ref must be available to the box (coordinator-managed repository access). No push is
performed here. The Ubuntu/CUDA bootstrap is syntax-checked locally, not executed on CUDA.
A first 50-step CUDA calibration is required before relying on the extrapolated schedule.

## Return artifacts

Prefer scp to keep transfer separate from training and preserve resumable optimizer state:

```bash
# From the coordinator's receiving machine, after reviewing the artifacts:
scp -r <GPU_HOST>:/work/res100-wave/klusai-models/runs/res100-seed0 <LOCAL_RESULTS>/
scp -r <GPU_HOST>:/work/res100-wave/klusai-models/runs/res100-seed1 <LOCAL_RESULTS>/
scp -r <GPU_HOST>:/work/res100-wave/klusai-models/runs/res100-seed2 <LOCAL_RESULTS>/
scp -r <GPU_HOST>:/work/res100-wave/klusai-models/runs/res100-logs <LOCAL_RESULTS>/
scp <GPU_HOST>:/work/res100-wave/klusai-models/runs/res100-data/manifest.json <LOCAL_RESULTS>/
```

Record SHA256 before/after transfer. Retain `merged/`, `training-manifest.json`,
`throughput.jsonl`, both resumable checkpoints, environment/git-ref records, data manifest,
and harness output. Private HF upload can be a separate coordinator action after review;
the trainer deliberately rejects `--push`. Nothing has been uploaded by RES-100.

## Local proof and capacity estimate

The bounded CPU run and saved-checkpoint scoring passed; no GPU-hours were consumed.

Measured on M3 Ultra / 96 GB, CPU(4), fp32, batch 4 × accumulation 4, max length 256,
seed 0. Real pretrained weights, 131 documents (123 KP + 8 TAB train), 156 chunks.
Four optimizer steps; warm summary excludes step 1. Another training process was live on the
machine, so shared-memory contention is possible. No MPS training or inference was run here.

| Local result | Value | n / 95% interval |
| --- | --- | --- |
| Warm non-padding tokens/s | 131.43 | 3 steps; bootstrap [119.06, 138.30] |
| Warm padded tokens/s | 247.52 | same 3 steps |
| Peak CPU process RSS | 9.88 GB | one process; not CUDA VRAM |
| Saved merged weights + tokenizer | 5.63 GB | 73-label KP head; 998,089 trainable parameters |
| Harness entity F1 | 0.1768 | 16 held-out KP docs; bootstrap [0.1236, 0.2273] |
| Harness national-ID leak rate | 0.4000 | 6/15 subjects; Wilson [0.1982, 0.6425] |

F1 bootstrap: 1,000 document resamples, RNG seed 0. Throughput bootstrap: 10,000 step
resamples, RNG seed 0; three steps give very weak uncertainty coverage and omit machine/
backend/workload variation. One training seed only. These are **load/score smoke results**,
not a full-quality evaluation, improvement, or win. The actual harness `run_spec` and
`KpModelAdapter` scored the saved merged artifact on CPU. `docs/res100-smoke-evidence.json`
records the results, versions, and artifact hash. A separate tiny-architecture regression test
also verifies exact tensor equality after resume from step 2 through step 4.

Reproduce the bounded smoke (create the log directory first):

```bash
mkdir -p runs/res100-smoke
PYTHONPATH="$PWD:/path/to/klusai-datasets" .venv-moe/bin/python scripts/prepare_res100_data.py \
  --output-dir runs/res100-smoke/data --pool-per-language 100 --kp-max 128 --tab-max 8 \
  > runs/res100-smoke/prepare.log 2>&1
.venv-moe/bin/python scripts/train.py moe-finetune \
  --base openai/privacy-filter --revision 7ffa9a043d54d1be65afb281eddf0ffbe629385b \
  --dataset runs/res100-smoke/data/train.jsonl --out klusai/res100-smoke \
  --backend cuda --device cpu --epochs 3 --lr 0.0001 --batch-size 4 --lora-rank 8 \
  --max-length 256 --gradient-accumulation-steps 4 --max-train 131 --max-steps 4 \
  --save-steps 2 --seed 0 --output-dir runs/res100-smoke/model > runs/res100-smoke/train.log 2>&1
.venv-moe/bin/python scripts/score_res100_smoke.py --model runs/res100-smoke/model/merged \
  --rows runs/res100-smoke/data/heldout-kp.jsonl --limit 16 --out runs/res100-smoke/score.json \
  > runs/res100-smoke/score.log 2>&1
```

The full prepared mix is **39,015 documents, 45,316 chunks, 3,656,801 non-padding tokens per
epoch**. Three epochs = 10,970,403 tokens/seed. At the measured CPU rate that is 23.2 training
hours/seed before preprocessing, saves, scoring, or transfer. No full training was performed.

GPU estimates below are **planning scenarios**, not measured speedups or statistical CIs.
They assume 2× CPU throughput on L40S and 4× on H100, with broad sensitivity ranges because
the eager sparse-expert loop can be dispatch-bound and CUDA can also lose to CPU. The model
uses fp32 expert operations; multiplying advertised BF16/FP8 TFLOPs would be misleading.
Hardware capacity references: [L40S](https://www.nvidia.com/en-us/data-center/l40s/) and
[H100](https://www.nvidia.com/en-us/data-center/h100/). H100 here means an 80 GB model; exact
PCIe/SXM host differences are not benchmarked.

`GPU hours/seed = 10,970,403 / (131.43 × assumed_speedup × 3600) × 1.20`.
The 20% allowance covers setup/checkpoint overhead, not transfer or full harness scoring.

| One GPU | Assumed tokens/s | Hours/seed, 3 epochs | Three-seed wave | Sensitivity hours/seed |
| --- | --- | --- | --- | --- |
| L40S 48 GB | 263 (2× CPU) | 13.9 | 41.7 GPU-h | 7.0–55.6 (0.5–4× CPU) |
| H100 80 GB | 526 (4× CPU) | 7.0 | 20.9 GPU-h | 3.5–55.6 (0.5–8× CPU) |

Expected VRAM at this fp32 profile: **8–16 GB**, provision **24 GB headroom** within either
card. Reasoning: 5.60 GB frozen/base weights, ~4 MB trainable weights, ~12 MB gradients/Adam
states, plus eager-attention/expert activations, temporaries, and allocator reserve. The CPU
RSS and RES-98's 6.92 GB short-sequence MPS allocator proxy support feasibility only; neither
is a CUDA measurement. Provision at least 32 GB host RAM and 80 GB free disk for environment,
model/data caches, three 5.63 GB merged exports, checkpoints, and logs. Calibrate 50 CUDA steps
on the actual mix and replace both throughput and memory estimates before scheduling epochs.

Not verified: Ubuntu/CUDA execution, L40S/H100 speed or memory, full convergence, multi-seed
quality, TAB test performance, GPU kernel optimization, or calibration of the new KP head.
The adapter's ordinary pipeline uses the saved 256-token limit at inference; it does not score
all chunks of long documents. Full TAB scoring must explicitly document this limitation or
use a separately reviewed chunk-aware adapter. Training itself retains every document chunk.
