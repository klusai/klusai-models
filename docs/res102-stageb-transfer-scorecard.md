# RES-102 — Stage-B structural-diversity transfer to REAL legal gold (TAB)

*2026-06-07T21:43:03.161190+00:00* · config_status: **dev** (synthetic training; TAB is real external gold)

## Question

Does training kp-deid-mdeberta-280m on structurally-diverse stage-B synthetic EN data transfer better to REAL legal gold (TAB ECHR) than template-splice stage-A data, at matched size + matched PII (only document STRUCTURE differs)?

## Result (the delta is the headline)

| Arm | Train data | Structure | TAB entity-F1 |
|---|---|---|---|
| A (stage-A) | `en_stagea_matched_v1.jsonl` (2000) | template-splice | **0.0261** |
| B (stage-B) | `en_stageb_v1.jsonl` (2000) | LLM-narrated | **0.1006** |

**Delta (stage-B − stage-A) = +0.0745** · effect: **positive_directional** (noise floor 0.01)

## Reading

POSITIVE (directional). Stage-B transferred better to REAL legal gold: TAB entity-F1 0.1006 vs stage-A 0.0261 (delta +0.0745). The arms share identical PII and size; only document STRUCTURE differs, so the gap is attributable to structural diversity. Single seed per arm — directional, not a CI-backed claim. Absolute levels are low (2000-doc models); the delta is the result.

## Setup (matched pair — only STRUCTURE differs)

- Base model: `microsoft/mdeberta-v3-base` (the shipped kp-deid-mdeberta-280m architecture)
- Hyperparams (identical across arms): `{"epochs": 3, "lr": 0.0003, "lora_rank": 16, "batch_size": 16, "max_length": 256, "seed": 0}`
- Identical hyperparams + seed across arms: **True**
- Eval: TAB ECHR legal English (REAL peer-reviewed gold) — 127 docs, labels ['ADDRESS', 'CASE_NUMBER', 'DATE', 'ORG_PARTY', 'PERSON']
- Scoring: entity_f1 via scorecard_klu106 (_predict/_mask/_entity_f1_on_rows) — EXACT RES-97 path; apples-to-apples with the mdeberta-280m TAB baseline.
- Single seed per arm (seed 0); no multi-seed CI band.

## Dataset integrity

- **stage-A**: n=2000, invalid NINO=0, invalid IBAN=0, all rows byte-equal + BIOES-valid=True; matched-pair PII: {"docs_compared": 2000, "docs_all_pii_values_in_stageb_pair": 2000}
- **stage-B**: n=2000, invalid NINO=0, invalid IBAN=0, all rows byte-equal + BIOES-valid=True

## Context prior

- RES-97: mdeberta-280m on the full 8-lang synthetic corpus scored TAB entity-F1 = 0.0489 (the known collapse). Context only; not part of this matched pair.

## Guards

- config_status: dev (synthetic training; TAB is real external gold)
- single_seed_per_arm: True
- no_citable_or_sota_claim: True
- negative_or_null_reported_truthfully: True
