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

## Usage

```bash
make install && source .venv/bin/activate
make check
python scripts/train.py moe-finetune --base openai/privacy-filter \
    --dataset klusai/ds-kp-legal-ro-50k --out klusai/kp-deid-moe-ro --backend mlx
```

Evaluation always defers to the **europriv-bench** harness (single source of truth for
scoring) so results match the public leaderboard. SOTA is claimed only with head-to-head
wins vs Piiranha (baseline-only, CC-BY-NC-ND), tabularisai, MAPA, OpenMed, GLiNER-PII,
and `openai/privacy-filter`.
