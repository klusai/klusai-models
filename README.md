# klusai-models

Model layer for the **KlusAI Privacy (KP)** program: finetuning, evaluation, and the
`klusai-privacy` SDK. Publishes `klusai/kp-{task}-{base}-{size}[-variant]`.

## Tracks

- **Primary — continue-finetune `openai/privacy-filter`** (the MoE; proven by OpenMed,
  MLX-friendly, no from-scratch training). Our deltas: real+domain data, deep-European
  languages, GDPR/legal/clinical taxonomy, and we *report numbers* on EuroPriv-Bench.
- **Comparison** — XLM-R / mDeBERTa NER, GLiNER, a LoRA anonymizer LLM, a sensitivity classifier.

## Compute model

**PyTorch on Mac Studio; explicitly scheduled CUDA runs for heavy jobs.** Encoder training
uses MPS; the MoE defaults to CPU on Mac, with MPS available. Both implemented trainers use
`--backend cuda` and select hardware with `--device cpu|mps|cuda`. The MoE exports local
checkpoints for separate coordinator-managed transfer. MLX runs publish an extra `-mlx` variant.

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

The backends use **separate environments**. Keep the existing encoder stack in `.venv`
(`.[hf]`: Transformers **4.57.6**) and install privacy-filter into `.venv-moe`
(`.[moe]`: Transformers **5.10.2**, Torch **2.12.0**, PEFT **0.19.1**).
Do not combine `hf`, `moe`, or `mlx` extras. The 4.x encoder stack does not recognize
`openai_privacy_filter`; the MoE architecture requires Transformers >=5.9.

```bash
make install-moe BENCH_DIR=../europriv-bench
.venv-moe/bin/python scripts/train.py moe-finetune --help
```

| Model family | Extra/environment | Training implementation |
| --- | --- | --- |
| `moe-finetune` | `moe` / `.venv-moe` | PyTorch CPU (default on Mac), MPS, CUDA |
| `xlmr-ner` | `hf` / `.venv` | Existing Transformers 4.57 encoder path |
| `classifier`, `anon-lora` | `hf` | Family-dependent; unified CLI still unwired |
| `gliner` | `gliner` | Separate inference environment |
| MLX experiments | `mlx` | Separate environment; not the MoE trainer |

The MoE trainer uses `--backend cuda` to select PyTorch and `--device` to select the
actual device. It learns attention LoRA, routers, and a KP BIOES classifier; expert tensors
remain frozen. It saves resumable PEFT checkpoints and a merged model loadable by the
EuroPriv-Bench `kp-model` adapter. No upload occurs. See the
[RES-100 runbook](docs/res100-gpu-runbook.md) for data preparation, CPU proof, exact CUDA
commands, capacity estimates, and the Ubuntu bootstrap script.

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

# MoE continue-finetune (prepared local KP + TAB-train corpus; see RES-100 runbook).
.venv-moe/bin/python scripts/train.py moe-finetune --base openai/privacy-filter \
    --dataset runs/res100-data/train.jsonl --out klusai/kp-deid-moe-ro \
    --backend cuda --device cpu --batch-size 4 --max-steps 4 --max-train 128

# Score on EuroPriv-Bench (defers entirely to the harness via its `kp-model` adapter).
python scripts/evaluate.py --model klusai/kp-deid-mdeberta-280m \
    --suite ../europriv-bench/evaluations --only ro-realskeleton
```

Evaluation always defers to the **europriv-bench** harness (single source of truth for
scoring) so results match the public leaderboard. SOTA is claimed only with head-to-head
wins vs Piiranha (baseline-only, CC-BY-NC-ND), tabularisai, MAPA, OpenMed, GLiNER-PII,
and `openai/privacy-filter`.

The [RES-125 frozen candidate scorecard](docs/res125-candidate-scorecard.md) compares the shipped
280M, three legal-560M seeds, three TAB-trained-560M seeds, stock EuroPriv Presidio and pinned Shield
on identical benchmark rows. It reports legacy leakage separately from exact-character residual v2,
utility and paired confidence intervals. The shipped 280M remains a single checkpoint; gateway
acceptance is unverified. Historical any-overlap results must not be read as full-character protection.

### Pull request checks and merging

The `test` CI job runs `make SHELL=/bin/bash check` on Python 3.13 for every PR and
push to `main`. It installs the base package and dev extra, pins `europriv-bench`
to a Git commit in `.github/workflows/ci.yml`, and adds CPU PyTorch and `datasets`
for the existing unit tests. HF offline flags prevent model downloads; the SDK
model integration test and optional MLX tests use their existing skip conditions.

Auto-merge is ported from `klusai/rb-cli`. A repository admin must enable squash
and auto-merge, protect `main` with the required `test` check, and configure the
team's Codex review requirement. The workflow honors required reviews; it does
not request a Codex review itself. If it cannot confirm that `test` is actively
required, it leaves auto-merge untouched. Protection exposed only through a
ruleset without that branch API summary also fails closed.

Drafts and `needs-review` or `hold` labels disable auto-merge. Forks, `review/` or
`hold/` branches, and Dependabot major updates require an `automerge` label;
that override never bypasses a draft or either hard hold label. The privileged
workflow reads PR metadata only and never checks out or executes PR code.

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
