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
