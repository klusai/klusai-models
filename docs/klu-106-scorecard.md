# KLU-106 — kp-deid v2 held-out scorecard

**Status: `config_status=dev`. Contamination-controlled. NOT citable / NOT SOTA / NOT a "best
protector" claim.** Validation is gated on KLU-27 (native-speaker review + IAA) and ≥2 template
families per language. This artifact discharges the KLU-106 acceptance — *"produce the rigorous
scorecard on the FIXED held-out eval set"* — **not** "train a good model."

- Machine-readable companion: [`klu-106-scorecard.json`](klu-106-scorecard.json)
- Carve-out method: [`klu-106-v2-contamination-carveout.md`](klu-106-v2-contamination-carveout.md)
- Training manifest: [`../runs/klu106-train-manifest.json`](../runs/klu106-train-manifest.json)

## Setup

| | |
|---|---|
| v2 seeds scored | `runs/kp-deid-mdeberta-280m-v2-seed0`, `…-seed1` (both completed; seeds 0 & 1) |
| Control | the existing KLU-51 `kp-deid` (`runs/kp-deid-mdeberta-280m`), scored on the **same** held-out set — **NOT** the published 0.46–0.52 |
| Base | `microsoft/mdeberta-v3-base` (LoRA r=16), 8 langs ro/en/pl/de/fr/es/it/nl |
| Scoring device | PyTorch-MPS (M3 Ultra; GPU free post-training) |
| Bootstrap | 95% percentile, **paired by document**, 2000 iters, seed 12345 |
| Held-out general | `runs/klu106-heldout-general` — carved BEFORE training, template- AND subject-disjoint (subject key = `NATIONAL_ID`); subject intersection asserted 0 per language pre- and post-down-sample |

The control is *trained* on ro/en/pl general data (in-distribution for those langs) and is *zero-shot*
on the T1 langs de/fr/es/it/nl. v2 is trained on all 8. The Δ below is `v2 − control` on the
identical fixed held-out rows.

## 1. Per-language entity-F1 on the clean held-out general split (Δ vs control)

Headline gain requires the **Δ CI to exclude 0** AND be positive, **for all seeds**.

| Lang | n | Control F1 | v2 F1 (mean / min / max) | Δ mean | Δ min | Δ max | Δ CI excl 0 (both seeds) | Verdict |
|------|---|-----------|--------------------------|--------|-------|-------|--------------------------|---------|
| de | 8384 | 1.000 | 1.000 / 1.000 / 1.000 | 0.000 | 0.000 | 0.000 | no | parity — control already saturated |
| en | 8343 | 1.000 | 0.810 / 0.809 / 0.811 | **−0.190** | −0.191 | −0.189 | excl 0 but **negative** | **regression** — v2 worse than control |
| es | 8310 | 0.999 | 1.000 / 1.000 / 1.000 | +0.001 | +0.001 | +0.001 | yes | nonzero but **negligible** (ctrl 0.9992) |
| fr | 8467 | 1.000 | 1.000 / 1.000 / 1.000 | 0.000 | 0.000 | 0.000 | no | parity — control already saturated |
| **it** | 8304 | 0.865 | 1.000 / 1.000 / 1.000 | **+0.135** | +0.134 | +0.135 | **yes** | **material gain** (only clear positive) |
| nl | 8229 | 1.000 | 1.000 / 1.000 / 1.000 | 0.000 | 0.000 | 0.000 | no | parity — control already saturated |
| pl | 8481 | 1.000 | 1.000 / 1.000 / 1.000 | 0.000 | 0.000 | 0.000 | no | parity — control already saturated |
| ro | 8281 | 1.000 | 1.000 / 1.000 / 1.000 | 0.000 | 0.000 | 0.000 | no | parity — control already saturated |

Identifier-surface-form holdout: n=0 rows (the carve produced no held-out row whose every PII surface
string is absent from train, because national-IDs are near-unique and the held-out templates reuse
train-seen low-cardinality fillers); the stricter subset is therefore empty and not separately scored.

### Honest reading (this is a finding, not a failure)

The clean held-out general split **does not isolate a v2 gain for most languages**: the single
held-out template per language is a fixed sentence skeleton with only the PII slots varying, and the
PII surface cues (capitalised name after a fixed lead-in, `@`-email, IBAN prefix, fixed-position
fields) are unambiguous and largely language-independent. **The control already scores F1 = 1.000 on
de/fr/nl/pl/ro and 0.999 on es**, so Δ is structurally pinned at ~0 and **no headline gain can clear
0** there.

- **it** is the one language with a material, CI-clearing positive Δ (+0.135): the control sat at
  0.865, v2 reaches 1.000.
- **es** clears 0 but the gain is negligible (+0.0008 over a 0.9992 control) — not a meaningful gain.
- **en regresses by −0.19** (CI excludes 0 in the negative direction). The held-out EN template
  annotates `ORG_PARTY`, which v2 under-detects on this surface. This is surfaced, not hidden.
- de/fr/nl/pl/ro are parity at the F1 ceiling.

This is exactly the limitation KLU-106 and the carve-out doc pre-register: a single authored template
family per language is **not** validated generalization. A second, harder template family per language
(KLU-27) is required before any held-out F1-gain claim. The substantive, contamination-clean signal in
this run is the re-id leak on the real-skeleton tracks (§2), not the saturated general-split F1.

## 2. Re-id leak on the real-skeleton tracks (per-subject, Wilson upper bound)

Pre-registered Wilson-UB threshold: **0.01** (the v1 board sustained 0.000 with UB ≈ 0.0034). Leak is
counted per distinct national-ID subject. Detection-F1 and re-id-leak are reported side-by-side;
a **dissociation break** = F1 rises while leak *also* rises (none observed).

| Track | metric | n / subjects | Control F1 | Control leak (UB) | v2 F1 (s0 / s1) | v2 F1 gain | v2 leak rate | v2 leak Wilson UB | UB ≤ 0.01 | Dissociation break |
|-------|--------|--------------|-----------|-------------------|-----------------|-----------|--------------|-------------------|-----------|--------------------|
| ro-realskeleton-v1 | CNP | 1500 / 1123 | 0.741 | 0.000 (0.00341) | 0.702 / 0.745 | −0.039 / +0.004 | 0.000 / 0.000 | 0.00341 / 0.00341 | **yes** | no |
| pl-realskeleton-v1 | PESEL | 1500 / 1096 | 0.763 | 0.000 (0.00349) | 0.781 / 0.780 | +0.019 / +0.017 | 0.000 / 0.000 | 0.00349 / 0.00349 | **yes** | no |
| it-realskeleton-v1 | CF | — | — | — | — | — | — | — | — | skip-and-reported (not published on hub) |

### Reading

- **The 0%-protection result HOLDS.** Both real-skeleton tracks (ro/pl) show **leak_rate = 0.000**
  for the control and both v2 seeds, **Wilson UB ≈ 0.0034 ≤ 0.01** in every cell. v2 does not
  re-introduce a re-id leak. **No dissociation break** — F1 never rises together with leak.
- Detection-F1 on the real (harder) surface is mixed and inside seed-noise: **ro F1 gain straddles 0
  across seeds** (−0.039 seed0, +0.004 seed1) → not a gain; **pl shows a small consistent +0.017–
  0.019** gain, but with no per-track bootstrap CI here it is not a validated gain.
- **it-realskeleton-v1** is correctly skip-and-reported: `it-realskeleton-v1` is not yet a published
  config of `klusai/europriv-bench`.

## Guards (schema-3)

- Every cell labelled `contamination` (`clean_held_out` throughout — the in-distribution general
  configs are deliberately not scored here) and `config_status=dev`.
- No SOTA / "best protector" / validated claim. Validation gated on **KLU-27** (native-speaker / IAA)
  + ≥2 template families per language.

## Utilization (from training manifest)

Both seeds completed within bounds: batch-16 single-process fp32 (KLU-48 optimum), sustained GPU
`Device Utilization` mean **92.9%** / peak 94% (ioreg, no root), peak unified memory 9.3 GB, train
throughput ~178 samples/s, total wall **51.7 min** (well under the 3 h stop). Sustained power (W) not
captured (needs passwordless `sudo powermetrics`); the exact command is recorded in the manifest — no
fabricated power number.

## Reproduce

```bash
EUROPRIV_DEVICE=mps PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python scripts/scorecard_klu106.py \
    --manifest runs/klu106-train-manifest.json \
    --control runs/kp-deid-mdeberta-280m \
    --suite ../europriv-bench/evaluations \
    --out docs/klu-106-scorecard.json
```
