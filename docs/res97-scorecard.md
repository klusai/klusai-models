# RES-97 — kp-deid-xlmr-560m held-out scorecard (vs kp-deid-mdeberta-280m)

**Status: `config_status=dev`** (synthetic) / **`real-external-gold`** (TAB). **NOT citable / NOT SOTA** —
no headline claim; citable gated on RES-77 (native-speaker/IAA). Machine-readable companion:
[`res97-scorecard.json`](res97-scorecard.json). Trained on-device (M3 Ultra, MPS/LoRA); proves
XLM-R-560m needs **no GPU burst** (RES-53/RES-63 assumption retired).

## Question

Does the 560M XLM-R encoder **beat** the 280M mDeBERTa on the **discriminating** evals (not the
saturating general split, which pins at F1≈1.0)? A real win requires the paired-bootstrap Δ CI to
exclude 0 **across both seeds**.

## Setup

| | |
|---|---|
| Model | `kp-deid-xlmr-560m` (base `FacebookAI/xlm-roberta-large`, LoRA r=16, targets `query/key/value`), **seeds 0 & 1** |
| Comparand | `kp-deid-mdeberta-280m` (KLU-106 v2 seed0), scored on the **same** held-out rows |
| Data | 8-lang balanced 40k (carved from an 80k pool), template- + subject-disjoint held-out (KLU-106 carve; invariants asserted) |
| Device | MPS (M3 Ultra), **~92 samples/s**, ~20 min/seed (no GPU burst) |
| Δ | paired 2000-iter document-bootstrap, F1(xlmr seed) − F1(mdeberta-280m) on identical rows |

## Result — the bigger encoder loses on synthetic, **wins on real data**

| Track | mdeberta-280m | xlmr-560m (seeds) | Winner |
|---|---|---|---|
| **hard-general** (8 langs, synthetic, de-saturated) | 0.96–1.00 | 0.85–0.92 | **280M** (both seeds below; Δ CI excludes 0, negative) |
| **real-skeleton** (it + others, synthetic) | 0.70–0.78 | 0.50–0.74 | **280M** |
| **TAB — ECHR legal EN (REAL gold)** | **0.0489** | **0.244 / 0.368** (mean 0.306) | **xlmr-560m** (both seeds beat; Δ CI excludes 0) |
| Ai4Privacy openpii (external) | — | — | **skipped** (config-name mismatch — follow-up) |

## Reading (honest)

On **our synthetic**, the 280M wins everywhere — it is **over-fit to our ~6 generator templates**
(RES-94: our synthetic has a unique-skeleton ratio ~0.004 vs Ai4Privacy ~1.0), so a small model
memorises the structure and a larger encoder offers no headroom on data that easy.

On the **one REAL out-of-distribution corpus (TAB legal English)**, the picture **inverts**: the 280M
collapses (entity-F1 **0.049** — it does not transfer to real legal prose), while **xlmr-560m
generalises ~6× better (0.306)**, and **both seeds beat it with the Δ CI excluding 0**. The bigger
multilingual encoder's advantage is **invisible on templated synthetic and only revealed by real
data**.

This is the load-bearing, program-level finding — not a leaderboard win: **synthetic saturation
hides real-world capability; real gold (TAB, RES-89) is what exposes generalisation; our synthetic is
templated (RES-94).** It is the empirical case for prioritising real-data anchors and the
generator upgrade (RES-95). No model is promoted; `kp-deid-mdeberta-280m` stays the shipped detector.

## Guards

`config_status=dev` (synthetic) / `real-external-gold` (TAB); no SOTA/citable claim (gated RES-77);
contamination-labelled; Δ from real runs (paired bootstrap); the carve's contamination invariants
(template- + subject-disjoint, EMPTY train∩heldout NATIONAL_ID intersection) asserted before training.

## Follow-ups (non-blocking)

- **Ai4Privacy track skipped** — the scorecard looks up `ai4privacy-openpii-{lang}-v1` but the HF
  configs are published under bare language names; re-align the config naming (or publish the
  `-v1`-suffixed configs) so the external-eval comparison runs. (RES-93 follow-up.)
- **Carve perf** — `carve_heldout_general`'s per-row Python pass is ~4 min even on 80k; parallelise
  `template_skeleton` for the full-corpus path (noted under RES-95/RES-63).
