# RES-72 / RES-104 — CJEU real-structure -> TAB zero-shot transfer scorecard (xlmr-560m)

*2026-06-08* · config_status: **dev** (synthetic training; TAB is real external gold) · **NOT citable / NOT SOTA**

Machine-readable companion: [`res72-cjeu-tab-transfer-scorecard.json`](res72-cjeu-tab-transfer-scorecard.json).

## Question (the decisive gate)

Does training **xlmr-560m** on a **REAL legal-STRUCTURE** corpus — CJEU real-skeleton (mined CJEU
layout signatures + authored connective prose + checksum-valid **synthetic** PII) — transfer to the
**TAB real-legal board ZERO-SHOT** *better* than generic synthetic (which caps ~0.30)? This is the
go/no-go for whether to scale the full multilingual RES-72 corpus.

## Result — CLEARS the generic ceiling

| Model / data | TAB entity-F1 | P | R |
|---|---|---|---|
| **CJEU real-skeleton, 20k (this)** | **0.3399** | 0.3924 | 0.2997 |
| Generic synthetic xlmr-560m, 40k template-splice (board) | 0.2439 | — | — |
| Generic-synthetic ceiling (RES-97/104.2a regime) | ~0.24–0.30 | — | — |
| Presidio (target) | 0.589 | — | — |

**Deltas:** +0.096 vs generic 40k (0.2439); **+0.0399 vs the ~0.30 generic ceiling**; −0.2491 vs
Presidio (0.589). The CJEU-trained model closes ~28% of the generic-40k → Presidio gap.

## Reading

**POSITIVE — clears the ceiling.** Training xlmr-560m on the CJEU real-STRUCTURE corpus reaches TAB
entity-F1 **0.3399** zero-shot, above the generic-synthetic 40k board entry (0.2439) and above the
top of the generic ceiling (~0.30). Real legal **structure** is a working transfer lever for the
real-legal board: the structurally-diverse-by-construction CJEU scaffold (unique-skeleton ratio
0.846 vs generic ~0.004) transfers to held-out ECHR legal prose better than the generic synthetic
regime does.

This is a **single seed → directional** result, not a CI-backed claim. It is a **real-structure /
synthetic-PII** corpus (NOT real PII): the layout signatures are mined from real CJEU judgments
(structure only — no source prose), the prose is authored boilerplate, and every PII value is
checksum-/format-valid synthetic.

## Decision — GO

**GO: scale to the full multilingual RES-72 build.** The bounded test cleared its gate (TAB-F1
0.3399 > ~0.30). Real legal structure is justified as a corpus lever; the recommendation is to
proceed to the multilingual RES-72 scale-up. (Lexical realism / real legalese remains a separate,
complementary lever toward Presidio 0.589 — not contradicted by this result, but not what this test
isolated.)

## Training setup (EXACT RES-104 hyperparams)

- Base model: `FacebookAI/xlm-roberta-large` (xlmr-560m); LoRA r=16, targets `query/key/value`
  (2.43M / 561M trainable, 0.43%).
- Hyperparams: `{epochs: 3, lr: 3e-4, lora_rank: 16, batch_size: 16, max_length: 256, seed: 0}` —
  identical to RES-104.
- Device: MPS (M3 Ultra), single process (one training job, one log — RES-97/102 lesson). Train
  2057.7s (29.2 samples/s); total wall 2083.3s.
- Eval: TAB ECHR legal English (REAL gold), n=127, labels [ADDRESS, CASE_NUMBER, DATE, ORG_PARTY,
  PERSON]; entity_f1 via `scorecard_klu106` (`_predict`/`_mask`/`_entity_f1_on_rows`) — the EXACT
  RES-97 path, apples-to-apples with every board number. Deterministic CPU scoring.
- Checkpoint: `runs/res72-cjeu-tab-seed0`.

## Corpus integrity + diversity

- `en_cjeu_skeleton_20k.jsonl` — n=20000, seed 20260608, built in 19.0s from 91 mined CJEU layout
  signatures (structure only). Build script: `klusai-datasets/scripts/generate_en_cjeu_skeleton.py`
  (reuses the merged, validated `en_skeletons_legal.generate_dataset`). Artifact gitignored (PoC; not
  pushed to HF).
- **Gold integrity:** 45,972 IBAN spans, **0 invalid IBAN** (mod-97); **0 misaligned spans**; all
  20,000 rows byte-equal + strict-BIOES valid (fail-loud validator). 244,099 total spans; labels
  [ADDRESS, CASE_NUMBER, DATE, IBAN, ORG_PARTY, PERSON].
- **Diversity (RES-94 method, `template_repetition`):** unique-skeleton ratio **0.846** (16,920 /
  20,000 unique; top-skeleton share 0.0879) — vs the generic-synthetic reference ~0.004. Structurally
  diverse by construction.

## Contamination assert (the load-bearing zero-shot guard)

**PASS.** No TAB test doc text appears in the CJEU training corpus. TAB is ECHR/HUDOC (Council of
Europe); the corpus is CJEU-structure (a different court / collection) + synthetic PII, so there is
no document overlap by construction. Verified: across 127 TAB docs vs 20,000 corpus docs —
**0 full-doc hits** and **0 / 4578 TAB-sentence (≥40 chars) verbatim hits** in the corpus.

## Guards

- config_status: dev (synthetic training; TAB is real external gold).
- single_seed: true (directional, not CI-backed).
- no_citable_or_sota_claim: true.
- result_reported_truthfully: true.
- real_structure_synthetic_pii_not_real_pii: true.
