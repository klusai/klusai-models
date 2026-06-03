# KLU-109 — kp-anon held-out privacy-utility scorecard

**Status: `config_status=dev`. Contamination-controlled. NOT citable / NOT SOTA / NOT a "best
anonymizer" claim.** Validation gated on KLU-27 (native-speaker + IAA). This artifact discharges the
KLU-109 acceptance — *"produce the rigorous held-out privacy-utility scorecard"* — **not** "train a
great model." Machine-readable companion: [`klu-109-scorecard.json`](klu-109-scorecard.json);
frontier figure: [`klu-109-kp-anon-frontier.svg`](klu-109-kp-anon-frontier.svg).

## Setup

| | |
|---|---|
| Model | `kp-anon-mdeberta-280m` — span-replacement anonymizer (LoRA mDeBERTa detector + pseudonymization policy), seeds 0 & 1 |
| Control | **redaction baseline (KLU-104)**: the **SAME** trained detector used as a plain redactor (blanket-mask every detected span) |
| Privacy metric | `redaction_leakage.leak_rate` — per distinct subject `(doc, country, normalized value)`, leaks iff a re-identifying fragment of the gold value **survives verbatim in the output** (value-search, robust to length changes; never re-detected), 95% Wilson CI ↓ |
| Utility metrics | `information_retention` ↑ and `1 − mask_token_ratio` ↑ (unmasking utility); paired 2000-iter doc-bootstrap Δ(kp-anon − baseline) |
| Tracks | Track-C frontier on `ro-realskeleton-v1` (CNP, 1123 subj) + `pl-realskeleton-v1` (PESEL, 1096 subj), `clean_held_out` |
| Training | on-device MPS, 2 seeds, 18k balanced samples, 3 epochs, **15.4 min** wall (not stopped early), 93% mean / 94% peak GPU util |

## Result — per track (both seeds)

| Track | seed | leak: baseline → kp-anon (Wilson UB) | info-retention Δ (CI) | unmasking-utility Δ (CI excl 0) | bijection in/cross-doc | dominates? |
|---|---|---|---|---|---|---|
| ro-realskeleton-v1 | 0 | 0.0% → **2.32%** (UB 3.37%) | 0.0 (—) | **+0.251** (excl 0) | 0.999 / 0.961 | no |
| ro-realskeleton-v1 | 1 | 0.0% → **2.76%** (UB 3.89%) | 0.0 (—) | **+0.224** (excl 0) | 0.999 / 0.957 | no |
| pl-realskeleton-v1 | 0 | 0.0% → **2.74%** (UB 3.88%) | 0.0 (—) | **+0.236** (excl 0) | 1.000 / 0.963 | no |
| pl-realskeleton-v1 | 1 | 0.0% → **2.65%** (UB 3.77%) | 0.0 (—) | **+0.226** (excl 0) | 0.999 / 0.948 | no |

## Honest reading (a finding, not a failure)

kp-anon (pseudonymization) **does not dominate** blanket redaction: it leaks **2.3–2.8%** of subjects
(Wilson UB ≤ 3.9% across both languages and seeds) where the blanket-mask baseline — same detector —
leaks **0%**. In exchange it delivers a **large, seed-noise-clearing readability/utility gain**: it
removes essentially all mask tokens (`1 − mask_token_ratio` improves by **+0.22 to +0.25**, CI
excludes 0 in every cell) at **identical information retention** (Δ = 0), with a near-perfect
in-document surrogate bijection (≈0.999) and a strong cross-document bijection (≈0.95–0.96). So
kp-anon occupies a real point on the privacy-utility frontier — **trade a small, bounded privacy cost
for a much more usable document** — rather than a free win. Honestly modest, on `dev` data, gated on
KLU-27.

## ⚠ Flag for root-cause before any public use

The scorecard's stated premise — *"both anonymizers share the detector, so the leak should be the
same at equal recall; substitution introduces zero leak by construction"* — **is contradicted by the
data** (0% baseline vs ~2.3–2.8% kp-anon). The robust value-search metric rules out a length/offset
artifact, so the substitution policy **does** leave small re-identifying fragments the blanket-mask
covers — most likely **partial-fragment survival at span boundaries** (the substitution replaces the
exact predicted character span while blanket-masking covers a slightly wider region). This must be
**root-caused** (genuine boundary property vs a substitution-coverage bug) before these numbers inform
any public claim; tracked as a follow-up. It does not change the qualitative finding (kp-anon trades a
small leak for large utility), only its precise privacy figure.

## Guards

`config_status=dev`; no SOTA / "best" / validated claim (gated KLU-27); privacy axis attributable to
detection recall + substitution boundary behaviour (see flag above); per-subject `(doc, country,
value)` dedup; ≥2-seed variance reported.
