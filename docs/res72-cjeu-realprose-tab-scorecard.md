# RES-72/104 — lexical-realism lever: real CJEU *prose* → TAB zero-shot (negative)

**Status: `config_status=dev` (synthetic training) / `real-external-gold` (TAB). Single seed → directional.
NOT SOTA/citable.** Machine-readable companion: `runs/res72-realprose-result.json`,
build report `runs/res72-realprose-build-report.json`.

## Question

The CJEU real-*skeleton* (real legal STRUCTURE + synthetic gold PII) lifted xlmr-560m to TAB zero-shot
**0.3399**, clearing the generic-synthetic ceiling. Does adding **lexical realism** — training on the
REAL CJEU *prose* itself (synthetic gold PII spliced in by entity-replacement) — close more of the gap
to Presidio (0.589)?

## Method

Real EN CJEU bodies (`davidwickerhf/cjeu-opendata`, Apache-2.0 / EU-2011/833 reusable) are kept as the
document text; spaCy `en_core_web_lg` + a case-number regex detect entity mentions, each replaced with
a checksum-/format-valid synthetic value of the same KP type (offset-spliced → the replacement span IS
the gold span). Institutional references ("Court", "Commission", …) are deliberately left as real
prose (not parties). 20,026 docs, seed 20260608; xlmr-560m, identical RES-104 hyperparams; scored on
TAB via the exact RES-97/`scorecard_klu106` path.

## Result — lexical realism does NOT help; it regresses vs structure-only

| Training data | TAB entity-F1 | precision | recall |
|---|---:|---:|---:|
| generic synthetic (40k template-splice) | 0.244 | — | — |
| **real legal STRUCTURE** (CJEU skeleton, 20k) | **0.340** | 0.392 | 0.300 |
| **real legal PROSE** (entity-replacement, 20k) | **0.265** | **0.789** | **0.159** |
| Presidio (zero-shot, board leader) | 0.589 | — | — |

Real-prose lands **below** real-structure (−0.075) and only marginally above generic synthetic.

## Reading (honest)

The diagnostic is the precision/recall split: **P 0.79 / R 0.16** — the real-prose-trained model is
*precise but severely under-recalling*. The cause is **label inconsistency inherent to
entity-replacement**: institutional ORGs are intentionally unlabeled and there is DATE-boundary
detection noise, so the model learns to under-predict entity-like tokens. That recall cost outweighs
the lexical-realism benefit. Corpus integrity was clean (build report: 400,366 gold spans across
20,026 docs; 0 misaligned/BIOES-invalid; real-PERSON leakage ~0.03/doc — effectively none).

**This characterises the on-device synthesis ceiling for the TAB real-legal board:**
- generic synthetic ≈ 0.24–0.30,
- real **structure** ≈ **0.34** (the best synthesis lever),
- real **prose** (entity-replacement) ≈ 0.26 (worse — label-consistency cost),
- all far below Presidio's zero-shot **0.589**.

The surest remaining lever to actually top TAB is *supervised training on real annotated legal data*
(e.g. TAB's own train split), which was deliberately deferred. Within synthesis/generalisation,
**structure — not lexical realism — is the lever**, and it caps ~0.34.

## Decision

Synthesis-only, zero-shot is **exhausted as a path to #1 on TAB** on-device. Per the agreed plan: bank
this finding and **redirect the win-track to where we are uncontested** — the re-identification-risk /
privacy-utility metric (flagship) and **multilingual legal breadth via the proven *structure* recipe**
(RES-72 acceptance: legal config in ≥3 languages). Topping TAB is reopened only if we adopt real
annotated legal supervision (a separate decision) or commit GPU-burst for a different model class.
