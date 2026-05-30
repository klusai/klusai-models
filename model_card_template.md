---
license: apache-2.0
language:
- ro
library_name: transformers
pipeline_tag: token-classification
tags:
- pii
- privacy
- de-identification
- bioes
- kp
base_model: openai/privacy-filter
datasets:
- klusai/ds-kp-legal-ro-50k
---

# {MODEL_NAME}

A KlusAI Privacy (KP) de-identification model — {ONE_LINE_DESCRIPTION}. Part of the
[EuroPriv-Bench](https://huggingface.co/datasets/klusai/europriv-bench) program.

## Model Details

| Property | Value |
|----------|-------|
| Task | Token classification (PII/PHI detection), BIOES |
| Base model | {BASE_MODEL} |
| Languages | {LANGS} |
| Domain | {DOMAIN} |
| Taxonomy | Harmonized KP (GDPR-aligned crosswalk) |
| Backend | {mlx \| cuda} |

## Evaluation

Scored on **EuroPriv-Bench** — entity F1 / recall-weighted F2 **plus** re-identification-risk
and privacy-utility. See the leaderboard for head-to-head vs `openai/privacy-filter`,
`OpenMed/privacy-filter-multilingual`, `tabularisai/eu-pii-safeguard`, GLiNER, Presidio.

| Split | Entity F1 | F2 (recall-weighted) | Re-id risk ↓ |
|-------|-----------|----------------------|--------------|
| {SPLIT} | {F1} | {F2} | {REID} |

## Intended Use & Limitations

Research + production de-identification for {DOMAIN} text in {LANGS}. Use behind a governance
layer (human review / deterministic pre-filters). Not a substitute for legal compliance review.

## Citation

```bibtex
@misc{klusai_europriv_2026,
  title  = {EuroPriv-Bench: A Unified Pan-European De-identification Benchmark},
  author = {KlusAI},
  year   = {2026}
}
```

## Related Artifacts

| Artifact | HF ID |
|----------|-------|
| Benchmark | `klusai/europriv-bench` |
| Training data | `{DATASET}` |
| SDK | `klusai-privacy` (extract_pii / deidentify / pseudonymize) |
