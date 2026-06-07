# RES-104 — Stage-B structural-diversity transfer to REAL legal gold (TAB), xlmr-560m

*2026-06-07T22:53:17.642072+00:00* · config_status: **dev** (synthetic training; TAB is real external gold)

## Question

Does training the BEST detector architecture (xlmr-560m, base FacebookAI/xlm-roberta-large) on structurally-diverse stage-B synthetic EN data transfer better to REAL legal gold (TAB ECHR) than template-splice stage-A data, at matched size + matched PII (only document STRUCTURE differs)? [xlmr-560m analog of RES-102]

## Result (the within-pair delta is the headline)

| Arm | Train data | Structure | TAB entity-F1 |
|---|---|---|---|
| A (stage-A) | `en_stagea_matched_v1.jsonl` (2000) | template-splice | **0.2961** |
| B (stage-B) | `en_stageb_v1.jsonl` (2000) | LLM-narrated | **0.2297** |

**Delta (stage-B − stage-A) = -0.0664** · effect: **negative** (noise floor 0.01)

## Reading

NEGATIVE. Stage-B did NOT transfer better for xlmr-560m: TAB entity-F1 0.2297 vs stage-A 0.2961 (delta -0.0664). On this matched 2000-doc setup, template-splice data matched or beat structurally-diverse data on REAL-legal transfer. Reported as-is; step 2b (scale-up) is NOT justified by this run.

## Comparison to baselines on the board

| Baseline | Setup | TAB entity-F1 | Matched-pair delta |
|---|---|---|---|
| **RES-104 xlmr-560m** (this) | 2000-doc matched pair | A 0.2961 / B 0.2297 | **-0.0664** |
| RES-102 mdeberta-280m | 2000-doc matched pair | A 0.0261 / B 0.1006 | +0.0745 |
| Full-corpus xlmr-560m | 40k template-splice | 0.2439 | n/a (single arm) |

**Does the bigger base amplify or dampen the stage-B benefit?** REVERSED. The 280M showed a positive matched-pair delta (+0.0745); xlmr-560m's is -0.0664 (opposite sign). The bigger base did not reproduce — and here reversed — the 280M stage-B benefit (single seed; directional).

> **Comparability caveat.** These 2000-doc arms are NOT directly comparable in absolute terms to the 40k full-corpus xlmr-560m board entry (0.2439, template-splice). The within-pair DELTA is the result of this experiment.

## Setup (matched pair — only STRUCTURE differs)

- Base model: `FacebookAI/xlm-roberta-large` (the BEST detector architecture, xlmr-560m)
- Hyperparams (identical across arms): `{"epochs": 3, "lr": 0.0003, "lora_rank": 16, "batch_size": 16, "max_length": 256, "seed": 0}`
- Identical hyperparams + seed across arms: **True**
- Eval: TAB ECHR legal English (REAL peer-reviewed gold) — 127 docs, labels ['ADDRESS', 'CASE_NUMBER', 'DATE', 'ORG_PARTY', 'PERSON']
- Scoring: entity_f1 via scorecard_klu106 (_predict/_mask/_entity_f1_on_rows) — EXACT RES-97 path; apples-to-apples with the xlmr-560m TAB baseline.
- Single seed per arm (seed 0); no multi-seed CI band — directional.

## Dataset integrity

- **stage-A**: n=2000, invalid NINO=0, invalid IBAN=0, all rows byte-equal + BIOES-valid=True; matched-pair PII: {"docs_compared": 2000, "docs_all_pii_values_in_stageb_pair": 2000}
- **stage-B**: n=2000, invalid NINO=0, invalid IBAN=0, all rows byte-equal + BIOES-valid=True

## Guards

- config_status: dev (synthetic training; TAB is real external gold)
- single_seed_per_arm: True
- no_citable_or_sota_claim: True
- negative_or_null_reported_truthfully: True
- absolute_not_comparable_to_40k_board_entry: True
