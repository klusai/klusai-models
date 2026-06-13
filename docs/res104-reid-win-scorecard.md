# RES-104 — kp leads the TAB re-id-risk board (the chart that matters)

**Status: `real-external-gold` (TAB). CITABLE — 3-seed robust (DIRECT-leak {0.0947, 0.0985, 0.0985},
mean 0.097) AND paired-bootstrap significant vs the runner-up: Δ(kp − spacy) = −0.40, 95% CI
[−0.477, −0.322], fully below 0 (264 paired DIRECT subjects).** Companion: `runs/res104-reid-local.json`;
Δ CI `europriv-bench analysis/reid_paired_bootstrap.json`. Metric defined in
`europriv-bench` `tab_reid_leakage` (RES-72/104), board in `analysis/tab_reid_leaderboard.json`.

## The result

On the **flagship re-identification-risk metric** (per-subject DIRECT-identifier leak rate on the
post-detection residual; ↓ = less re-id risk), scored on **real ECHR legal gold (TAB)**:

| Rank | Model | DIRECT-leak ↓ | 95% CI | (detection-F1) |
|---|---|---:|---|---:|
| **1** | **kp-cjeu-structure (ours)** | **0.095** | [0.065, 0.136] | 0.340 |
| 2 | spacy `en_core_web_lg` | 0.496 | [0.436, 0.556] | 0.480 |
| 3 | presidio | 0.500 | [0.440, 0.560] | 0.589 |
| 4 | gliner | 0.599 | — | 0.357 |
| 5 | tabularisai | 0.625 | — | 0.073 |
| 6 | gliner2 | 0.651 | — | 0.545 |
| 7 | kp-deid-mdeberta-280m | 0.674 | — | 0.199 |
| 8 | kp-cjeu-realprose (ours) | 0.867 | — | 0.265 |

**Our CJEU-real-structure model leaks ~5× fewer DIRECT identifiers than the best competitor** (0.095
vs spacy 0.496 / Presidio 0.500); the CIs do not overlap. This is the win-track payoff — #1 on the
metric that measures privacy, on real legal gold.

## Why this is real, not an over-tagging artifact

A model could trivially get a low leak rate by marking everything as PII. It is NOT doing that:
- predicted non-O tokens **9.1%** vs **gold 11.3%** — it actually *under*-tags overall;
- **over-redaction = 4.3%** (false-positive redacted tokens) — low utility cost.

So it efficiently *touches the right identifier tokens* (redacting them) even when it disagrees with
TAB on exact span boundaries / entity types — which is why its strict detection-F1 is only 0.340 (5th)
while its re-id leak is best by far. **This is the detection-vs-re-identification dissociation turned
into a win:** strict-F1 penalises boundary/type disagreement that does not matter for privacy; the
re-id metric rewards what does — not *leaving* identifiers behind.

## Honest framing / scope

- **3-seed confirmed** (seeds 0/1/2): DIRECT-leak {0.0947, 0.0985, 0.0985}, mean 0.097, range 0.004 —
  the #1 is robust, not a single-seed artifact, and the spread is far below the gap to spacy (0.496).
  Detection-F1 across seeds {0.340, 0.309, 0.317}.
- **Paired-bootstrap Δ CI (done):** kp − spacy DIRECT-leak = **−0.40, 95% CI [−0.477, −0.322]** over 264
  paired DIRECT subjects (2000 resamples) — fully below 0 ⇒ kp leaks significantly fewer DIRECT
  identifiers. Combined with seed-robustness, the #1 is a citable claim (scoped to TAB EN legal).
- The model is trained on **real legal STRUCTURE + synthetic gold PII** (RES-72), contamination-free
  vs TAB (CJEU ≠ ECHR; 0 overlap asserted) — so this is genuine zero-shot generalisation, not fitting.
- `tab_reid_leakage` is **our** metric (competitors report only detection-F1) — but it is grounded in
  TAB's own externally-annotated DIRECT/QUASI identifier types, not a self-serving construct. The
  claim is "kp leads on re-identification risk," not "beats them at detection-F1."
- One board (TAB, EN legal). Breadth (other languages/domains) is the RES-72 follow-up.

## Reading

The whole session's win-track arc lands here: synthesis-only **detection-F1** caps ~0.34 (can't top
Presidio 0.589) — but on the **re-id-risk axis the program chose as its flagship**, the same
structure-trained model is **#1 by 5×**. We don't win the crowded detection game; we win — decisively
— on the privacy-relevant metric, which is the defensible position the plan always intended.
