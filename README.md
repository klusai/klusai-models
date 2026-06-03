# klusai-models

Model layer for the **KlusAI Privacy (KP)** program: finetuning, evaluation, and the
`klusai-privacy` SDK. Publishes `klusai/kp-{task}-{base}-{size}[-variant]`.

## Tracks

- **Primary — continue-finetune `openai/privacy-filter`** (the MoE; proven by OpenMed,
  MLX-friendly, no from-scratch training). Our deltas: real+domain data, deep-European
  languages, GDPR/legal/clinical taxonomy, and we *report numbers* on EuroPriv-Bench.
- **Comparison** — XLM-R / mDeBERTa NER, GLiNER, a LoRA anonymizer LLM, a sensitivity classifier.

## Compute model

**MLX-first on Mac Studio; burst to DigitalOcean GPU droplets** for heavy jobs. Training is
**device-agnostic** (`--backend mlx|cuda`); datasets/checkpoints sync via HF Hub so a droplet
is stateless and disposable. MLX runs publish an extra `-mlx` variant.

### Mac-tier device for the encoder family (KLU-45)

The `xlmr-ner` token-classification family runs on PyTorch (it never uses MLX — KLU-11). On the
Mac Studio it can train on **either the CPU or the GPU (Metal/MPS)**; pick with `--device`:

```bash
python scripts/train.py xlmr-ner ... --device mps   # Mac GPU (Metal)
python scripts/train.py xlmr-ner ... --device cpu --threads 8
python scripts/train.py xlmr-ner ... --device auto  # best Mac-tier device (MPS if present)
```

**Default Mac-tier device: `mps`.** KLU-45 benchmarked `mdeberta-v3-base` LoRA (same smoke task
as KLU-17) on `mps` vs CPU(4)/CPU(8): MPS is materially faster and its loss/eval-loss curve
matches CPU within noise (see [`docs/klu-45-mps-vs-cpu.md`](docs/klu-45-mps-vs-cpu.md)). CPU is
the **guaranteed fallback** — if MPS is unavailable, `--device mps`/`auto` transparently drops to
CPU. `--cpu/--gpu` is the legacy KLU-17 switch and is superseded by `--device`.

> **Note (KLU-54): training `eval_loss` is NOT a model-quality metric.** It is a split-sanity /
> early-stopping signal only. The training eval split is now **template-disjoint** from train
> (`template_disjoint_split`; [`docs/klu-54-eval-split.md`](docs/klu-54-eval-split.md)) so the
> number is meaningful for checkpoint selection — but model **quality** is reported solely by the
> EuroPriv-Bench harness leaderboard (entity F1 / leak-rate on the contamination-free
> `ro-realskeleton-v1` track). The low `eval_loss` figures in the KLU-45/48 tables below are from
> the older leaky split and are used here only to compare *throughput / numerical parity*, never
> quality.

#### Max-utilization on the Mac GPU (KLU-48)

KLU-48 set out to make Mac training **saturate the M3 Ultra** — the human saw only ~68 W at
batch 16 and suspected the GPU was starved. We measured it directly
([`docs/klu-48-max-util.md`](docs/klu-48-max-util.md)) and the **premise turned out to be wrong
for this model**: the 280M mDeBERTa encoder is **memory-bandwidth-bound and already near-saturated
at batch 16** on this GPU. Measured on the same kp-deid finetune slice (real
`ds-kp-general-ro`, 2400 train / 2 epochs):

| config | samples/s | eval_loss | notes |
|---|---|---|---|
| **batch 16, single-process (default)** | **189** | 0.0005 | the optimum |
| batch 64, 8 workers | 94 | 0.028 | **0.49x — slower** |
| batch 256 (auto-fill memory) | 24 | 2.73 | **0.14x**, 72/96 GB, MPS thrash ~13 s/step, no convergence |

Throughput is flat (~58–64 samples/s) from batch 16→96 on full-length synthetic batches; bf16
autocast gives no MPS throughput win; multi-process DataLoader workers give no win for this light
collation **and deadlock at process exit under macOS `spawn`**. So **no batch size takes this
encoder to ≥150 W** — the ~68 W is mostly intrinsic to a 280M model on a 76-TFLOP GPU, not
starvation, and scaling the batch only regresses throughput and breaks convergence.

**Decision: the Mac default stays plain batch-16 single-process fp32 (the measured max-util
config) — it never regresses.** The `--max-util` flag, a memory-guarded batch auto-probe, and
(opt-in) workers ship as **infrastructure for the denser MoE track**, where the large-batch lever
does pay off, but are **off by default** for `xlmr-ner`:

```bash
python scripts/train.py xlmr-ner ... --device mps                        # default: batch-16, optimum
python scripts/train.py xlmr-ner ... --device mps --max-util             # opt in (denser models)
python scripts/train.py xlmr-ner ... --device mps --max-util --max-util-batch-size 48 --num-workers 4
```

Reproduce the numbers: `python scripts/bench_klu48_max_util.py` (writes
`docs/klu-48-max-util.json`). To read sustained package/GPU power yourself while a run trains
(`sudo`; `powermetrics` is the only reader present — `macmon`/`asitop` are not installed):

```bash
sudo powermetrics --samplers gpu_power -i 1000 -n 10
```

## Layout

```
klusai/privacy/models/   package (logger, training/config.py — families + backend)
klusai/privacy/sdk/      the SDK: extract_pii / deidentify / pseudonymize
scripts/                 train.py (unified CLI w/ families), evaluate.py (defers to europriv-bench)
conf/                    models.yaml (families + baselines), training.yaml (hyperparams)
model_card_template.md
tests/
```

Import roots are the PEP 420 namespace `klusai.privacy.models` and `klusai.privacy.sdk`.
Shared taxonomy + span alignment + the eval harness come from `europriv_bench` (a dependency).
Scripts run as `python scripts/x.py` from the repo root (research-repo convention).

## Backends & extras

The training/inference backends are optional extras so the SDK + config layer stay light.
**`hf` and `mlx` are separate, non-combinable install paths — never `pip install .[hf,mlx]`.**
Both transitively pull `transformers`, but they co-constrain it incompatibly: the `hf`
training stack (and the privacy-filter MoE arch) needs `transformers>=5.0`, while older
`mlx-lm` releases cap/exact-pin `transformers` below 5.0 (the 4.57.x band found in KLU-11),
so resolving them together is unsatisfiable. Install whichever single backend the target
environment needs; encoder training (which never uses MLX) and the MLX inference path live
in different environments.

| Model family (`scripts/train.py`) | Extra | Backend(s) |
| --- | --- | --- |
| `moe-finetune` (privacy-filter MoE, primary) | `mlx` (Apple Silicon) **or** `hf` (CUDA) | `mlx` / `cuda` |
| `xlmr-ner` (XLM-R / mDeBERTa token-classification) | `hf` | `cuda` (HF/Torch) |
| `classifier` (sensitivity/doc-level) | `hf` | `cuda` (HF/Torch) |
| `anon-lora` (LoRA anonymizer LLM) | `hf` | `cuda` (HF/Torch) |
| `gliner` | `gliner` | HF/Torch |

Per KLU-11, the encoder families (`xlmr-ner`, `classifier`) do not use MLX — MLX is only for
the MoE / causal-LM track on Mac Studio. Pick `mlx` **or** `hf`, never both.

> **KLU-46 (spike):** an MLX-core proof-of-concept *can* run a DeBERTa-v2 encoder forward at
> exact parity with `transformers` (`klusai/privacy/models/mlx_encoder.py`, behind the `mlx`
> extra) — but the production encoder path stays on PyTorch-MPS. See
> [`docs/klu-46-mlx-encoder-spike.md`](docs/klu-46-mlx-encoder-spike.md) for the GO/DEFER finding.

## Usage

```bash
make install && source .venv/bin/activate
make check

# Token-classification family (xlmr-ner): transformers + peft LoRA, CPU-feasible.
# AutoModelForTokenClassification + TaskType.TOKEN_CLS LoRA on query/key/value_proj;
# labels come from europriv_bench.taxonomy.bioes_labels(). Bounded with --max-train/--epochs.
python scripts/train.py xlmr-ner --base microsoft/mdeberta-v3-base \
    --dataset klusai/ds-kp-general-ro-50k --out klusai/kp-deid-mdeberta-280m \
    --backend cuda --epochs 3 --max-train 4000 --lora-rank 16 --push
# (encoder family never uses MLX — KLU-11. --backend cuda runs on CPU when no GPU is present;
#  no -mlx variant is published for this family.)

# MoE continue-finetune (primary track) — lands later.
python scripts/train.py moe-finetune --base openai/privacy-filter \
    --dataset klusai/ds-kp-legal-ro-50k --out klusai/kp-deid-moe-ro --backend mlx

# Score on EuroPriv-Bench (defers entirely to the harness via its `kp-model` adapter).
python scripts/evaluate.py --model klusai/kp-deid-mdeberta-280m \
    --suite ../europriv-bench/evaluations --only ro-realskeleton
```

Evaluation always defers to the **europriv-bench** harness (single source of truth for
scoring) so results match the public leaderboard. SOTA is claimed only with head-to-head
wins vs Piiranha (baseline-only, CC-BY-NC-ND), tabularisai, MAPA, OpenMed, GLiNER-PII,
and `openai/privacy-filter`.

### `kp-anon` — the Track-C span-replacement anonymizer (KLU-109)

`kp-anon` (`anon-lora` family, `kp-anon-mdeberta-280m`) is a **span-replacement anonymizer**: a
LoRA mDeBERTa-280m PII detector (the KLU-48 MPS-proven batch-16 fp32 profile) whose detected spans
are **pseudonymized** — replaced with deterministic, type-consistent *surrogates*
(`klusai.privacy.models.anon.KpAnonAdapter` / `Pseudonymizer`) — instead of blanket-masked. It is
scored against the **KLU-104 redaction baseline** (the *same* detector used as a plain `█`-masking
redactor) on the **privacy-utility (Pareto) frontier**:

- **privacy** = `redaction_leakage.leak_rate` (per-subject re-id leak, read from gold offsets, Wilson
  CI) — identical for the two at equal detection recall; substitution introduces zero leak by
  construction (a surrogate never re-discloses a fragment of its source);
- **utility** = `information_retention` (↑ non-PII tokens preserved) and `1 − structural_disruption.mask_token_ratio`
  (↑ less mask-glyph fragmentation) — where pseudonymization keeps the document usable and the
  redaction baseline shreds it.

```bash
# Train kp-anon on-device (MPS, ≤3 epochs, capped corpus, ~3h wall-clock stop, ≥2 seeds).
python scripts/train_kp_anon_klu109.py --device mps --epochs 3 --seeds 0 --seeds 1 \
    --max-samples 18000 --wall-clock-stop 10800
# Held-out, bootstrap-CI'd, ≥2-seed frontier scorecard (ro + pl real-skeleton).
python scripts/scorecard_kp_anon_klu109.py --manifest runs/klu109-kp-anon-train-manifest.json
# Render the committed Pareto-frontier figure (dependency-free SVG).
python scripts/figure_kp_anon_frontier_klu109.py
```

`config_status=dev`; **no SOTA / "best" / validated claim** (validation gated on KLU-27). See
[`docs/klu-109-kp-anon-frontier.md`](docs/klu-109-kp-anon-frontier.md) for the scorecard + figure.
