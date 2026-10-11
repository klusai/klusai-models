# RES-104 — TAB re-id win and real-supervision follow-up

**TAB-supervised follow-up complete (seeds 0/1/2): detection improves, but does not reach #1.**
On the document-disjoint TAB test (127 judgments), TAB-train-only XLM-R has mean F1 **0.5292**,
range **[0.5283, 0.5303]**, versus CJEU's mean **0.3222**. All three same-seed F1 delta CIs are
above zero versus CJEU and below zero versus Presidio (**0.5888**). DIRECT leak is **0.0947–0.0985**
(264 subjects); same-seed delta CIs versus CJEU include zero, so no new re-id improvement is claimed.
Full per-seed 95% intervals, protocol, and limitations appear in the
[TAB supervision results](#tab-train-supervision-res-104-climb) below. Existing CJEU checkpoints
were rescored for all three seeds; the optional mixed-data arm was not run.

**Prior zero-shot CJEU status: `real-external-gold` (TAB). CITABLE — 3-seed robust (DIRECT-leak {0.0947, 0.0985, 0.0985},
mean 0.097) AND paired-bootstrap significant vs the runner-up: Δ(kp − spacy) = −0.40, 95% CI
[−0.477, −0.322], fully below 0 (264 paired DIRECT subjects).** Companion: `runs/res104-reid-local.json`;
Δ CI `europriv-bench analysis/reid_paired_bootstrap.json`. Metric defined in
`europriv-bench` `tab_reid_leakage` (RES-72/104), board in `analysis/tab_reid_leaderboard.json`.

## The result

On the **flagship re-identification-risk metric** (per-subject DIRECT-identifier leak rate on the
post-detection residual; ↓ = less re-id risk), scored on **real ECHR legal gold (TAB)**:

| Rank | Model | DIRECT-leak ↓ | 95% CI | (detection-F1) |
|---|---|---:|---|---:|
| **1** | **kp-cjeu-structure (ours)** ¹ | **0.095** | [0.065, 0.136] | 0.340 |
| 2 | spacy `en_core_web_lg` | 0.496 | [0.436, 0.556] | 0.480 |
| 3 | presidio | 0.500 | [0.440, 0.560] | 0.589 |
| 4 | gliner | 0.599 | — | 0.357 |
| 5 | tabularisai | 0.625 | — | 0.073 |
| 6 | gliner2 | 0.651 | — | 0.545 |
| 7 | kp-deid-mdeberta-280m | 0.674 | — | 0.199 |

¹ `kp-cjeu-structure` = local checkpoint `runs/res72-cjeu-tab-seed0`, **published** as
[`klusai/kp-deid-xlmr-560m-legal`](https://huggingface.co/klusai/kp-deid-xlmr-560m-legal) (identical
weights; the board is scored from the local checkpoint). The separately-shipped
`kp-deid-mdeberta-280m` is a *different* model (ranks 7th here).
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

<!-- RES104-TAB-SUPERVISED -->
## TAB-train supervision (RES-104 climb)

Fixed TAB-only arm; checkpoint selection uses TAB dev only. Data licence, hashes, split checks and alignment exclusions: [manifest](res104-tab-manifest.json).
Train 1,010 / dev 126 documents after the board curator's exclusions; test 127 documents, 4,821 spans, 264 DIRECT subjects. No CJEU mix arm was run.

Three epochs, LoRA r16, LR 3e-4, batch 16 fp32 MPS; full-document 256-token windows. Minimum dev-loss checkpoint; loss is only a selection signal. Test never selects an arm/epoch.

Harness CPU scoring (`score_reid_local.score_predictions`); F1 95% CIs and paired deltas use 2,000 document-bootstrap draws (seed 104); DIRECT marginal CIs are Wilson. Delta leak CIs cluster subjects by document. Machine-readable [results](res104-tab-results.json).

| Arm | Seed | F1 [95% CI] | DIRECT leak [95% CI] | ΔF1 vs Presidio [95% CI] |
|---|---:|---|---|---|
| cjeu-seed0 | 0 | 0.3399 [0.3126, 0.3656] | 0.0947 [0.0650, 0.1361] | -0.2489 [-0.2756, -0.2203] |
| cjeu-seed1 | 1 | 0.3093 [0.2850, 0.3341] | 0.0985 [0.0681, 0.1404] | -0.2794 [-0.3057, -0.2529] |
| cjeu-seed2 | 2 | 0.3173 [0.2886, 0.3450] | 0.0985 [0.0681, 0.1404] | -0.2714 [-0.2981, -0.2430] |
| presidio | — | 0.5888 [0.5626, 0.6149] | 0.5000 [0.4401, 0.5599] | — |
| spacy | — | 0.4798 [0.4576, 0.5017] | 0.4962 [0.4364, 0.5561] | -0.1090 [-0.1231, -0.0931] |
| tab-seed0 | 0 | 0.5290 [0.4882, 0.5693] | 0.0985 [0.0681, 0.1404] | -0.0597 [-0.0982, -0.0194] |
| tab-seed1 | 1 | 0.5283 [0.4882, 0.5690] | 0.0947 [0.0650, 0.1361] | -0.0604 [-0.0982, -0.0207] |
| tab-seed2 | 2 | 0.5303 [0.4907, 0.5704] | 0.0947 [0.0650, 0.1361] | -0.0584 [-0.0970, -0.0183] |

Seed spread (training seeds; each row above is one seed):

- tab: n_seeds=3 [0, 1, 2]; F1 mean 0.5292, range [0.5283, 0.5303]; DIRECT mean 0.0960, range [0.0947, 0.0985].
- cjeu: n_seeds=3 [0, 1, 2]; F1 mean 0.3222, range [0.3093, 0.3399]; DIRECT mean 0.0972, range [0.0947, 0.0985].

| Arm | ΔF1 vs same-seed CJEU [95% CI] | ΔDIRECT vs spaCy [95% CI] | ΔDIRECT vs same-seed CJEU [95% CI] |
|---|---|---|---|
| cjeu-seed0 | — | -0.4015 [-0.4382, -0.3626] | — |
| cjeu-seed1 | — | -0.3977 [-0.4351, -0.3577] | — |
| cjeu-seed2 | — | -0.3977 [-0.4351, -0.3577] | — |
| tab-seed0 | 0.1892 [0.1697, 0.2108] | -0.3977 [-0.4349, -0.3579] | 0.0038 [0.0000, 0.0115] |
| tab-seed1 | 0.2190 [0.1970, 0.2403] | -0.4015 [-0.4382, -0.3626] | -0.0038 [-0.0115, 0.0000] |
| tab-seed2 | 0.2130 [0.1932, 0.2337] | -0.4015 [-0.4382, -0.3626] | -0.0038 [-0.0115, 0.0000] |

Scope: supervised held-out TAB EN legal, not zero-shot. Subject identity and near-duplicate passage disjointness are not established. CIs condition on the trained seed; seed ranges are reported separately. Other languages/domains and a mixed-data arm were not tested. Scoring preserves the existing encoder adapter's 512-token truncation (no stride) and its any-token-overlap per-mention re-id proxy; complete removal of every identifier character is not verified.
