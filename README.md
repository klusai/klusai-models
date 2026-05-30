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
