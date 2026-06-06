# RES-19 — discriminating held-out general eval (re-score of kp-deid v2)

**Status: `config_status=dev`. Contamination-controlled. NOT citable / NOT SOTA / NOT a "best
protector" claim.** This is a **detection-eval-quality fix**, not a re-id claim: the real-skeleton
tracks remain the headline re-id signal. This artifact discharges the RES-19 acceptance — *"build a
held-out general set that can discriminate a v2 gain, and re-run the KLU-106 scorecard against it"* —
**not** "train a good model." No retraining: the EXISTING trained v2 seeds + the zero-shot control are
re-scored on a harder held-out surface.

- Machine-readable companion: [`res-19-hard-general-scorecard.json`](res-19-hard-general-scorecard.json)
- Carve manifest: [`res-19-carve-manifest.json`](res-19-carve-manifest.json)
- Problem statement: [`klu-106-scorecard.md`](klu-106-scorecard.md) §"Honest reading"

## The problem (from the KLU-106 scorecard)

The KLU-106 held-out *general* split used a **single** template per language. Its PII cues
(capitalised name after a fixed lead-in, `@`-email, IBAN prefix, fixed-position fields) were so
unambiguous that the zero-shot control already scored entity-F1 **≈ 1.000 on de/fr/nl/pl/ro** (0.999
es). With the control pinned at the ceiling, the `v2 − control` Δ was structurally pinned at ≈ 0 — the
eval **could not discriminate** a breadth/finetune gain. Only `it` (control 0.865) cleared a CI;
`en` *regressed* −0.19 on the single EN template's `ORG_PARTY` surface.

## What changed (the harder eval)

For every language, **two independent template families** (mirroring the KLU-101 RO Family-A/B
approach), built in `klusai-datasets` (`klusai/privacy/datasets/data/hardgeneral.py`):

- **Family A — bureaucratic record card**: fielded layout, terse labels, administrative register.
- **Family B — narrative correspondence / case note**: running prose, mixed register, PII embedded
  inside sentences with no fixed cue word.

Harder PII surface: **non-fixed positions, varied/absent lead-ins, mixed registers** — the surface a
saturated single-template eval never tests. Both families reuse each locale's existing `_fields`
builder, so the PII generators, checksum validity, and KP labels are byte-identical to the trained
`*-synthetic` tracks; only the skeleton wording / PII position differs.

**Independence + held-out-ness are hard-gated** (token 5-gram Jaccard on PII-masked skeletons, all
**≤ 0.10**, asserted in `tests/test_hardgeneral_families.py` and re-checked at carve time):

| gate | result (all 8 languages) |
|------|--------------------------|
| Family A vs Family B 5-gram Jaccard | **0.0000** (≤ 0.10) |
| each family vs **training** `*_documents.TEMPLATES` 5-gram Jaccard | **0.0000** (≤ 0.10) |

The second gate is what makes the families **template-disjoint from training** → inherently held-out:
the trained v2 seeds AND the zero-shot control never saw these skeletons. The carve
(`scripts/carve_hard_general_res19.py`) additionally **asserts an empty `NATIONAL_ID`-subject
intersection** with the actual training corpora (`klusai/ds-kp-general-*-50k`), keying on the same
near-unique subject id the KLU-106 carve and the re-id leak metric use. **1** Italian row was dropped
for a low-entropy codice-fiscale collision with training; intersection is **0** for every language
post-filter. **Eval set: 11,999 rows** (≈1,500/language, A/B balanced).

## Setup

| | |
|---|---|
| v2 seeds scored | `runs/kp-deid-mdeberta-280m-v2-seed0`, `…-seed1` (the existing KLU-106 seeds — no retraining) |
| Control | the existing KLU-51 `kp-deid` (`runs/kp-deid-mdeberta-280m`), scored on the **same** harder set |
| Held-out general | `runs/res19-heldout-hard-general` — 2 independent families/language, template- AND subject-disjoint from training |
| Scoring device | CPU (deterministic-ish); Mac-only, no GPU burst |
| Bootstrap | 95% percentile, **paired by document**, 2000 iters, seed 12345 |

## Per-language entity-F1 on the harder held-out general split (Δ vs control)

A discriminating gain requires the **Δ CI to exclude 0** AND be positive, **for all seeds**.

| Lang | n | Control F1 | v2 F1 (mean) | Δ mean | Δ min | Δ max | Δ CI excl 0 (both seeds) | Verdict |
|------|---|-----------|--------------|--------|-------|-------|--------------------------|---------|
| de | 1500 | 0.9589 | 0.9967 | **+0.038** | +0.035 | +0.041 | **yes** | discriminating gain (was Δ=0 / saturated) |
| en | 1500 | 0.9340 | 0.9742 | **+0.040** | +0.015 | +0.066 | **yes** | gain — the old eval showed a −0.19 regression artifact |
| es | 1500 | 0.9550 | 0.9852 | **+0.030** | +0.015 | +0.045 | **yes** | discriminating gain (was +0.0008 / negligible) |
| fr | 1500 | 0.9540 | 0.9921 | **+0.038** | +0.031 | +0.046 | **yes** | discriminating gain (was Δ=0 / saturated) |
| it | 1499 | 0.8490 | 0.9737 | **+0.125** | +0.111 | +0.139 | **yes** | strongest gain (consistent with the old +0.135) |
| nl | 1500 | 0.9671 | 0.9947 | **+0.028** | +0.022 | +0.033 | **yes** | discriminating gain (was Δ=0 / saturated) |
| pl | 1500 | 0.8839 | 0.9898 | **+0.106** | +0.104 | +0.108 | **yes** | discriminating gain (was Δ=0 / saturated) |
| ro | 1500 | 0.9373 | 0.9844 | **+0.047** | +0.032 | +0.063 | **yes** | discriminating gain (was Δ=0 / saturated) |

Per-seed Δ with 95% bootstrap CIs (paired by document):

| Lang | seed 0 Δ [lo, hi] | seed 1 Δ [lo, hi] |
|------|-------------------|-------------------|
| de | +0.0346 [0.0294, 0.0399] | +0.0410 [0.0364, 0.0458] |
| en | +0.0655 [0.0605, 0.0704] | +0.0148 [0.0059, 0.0235] |
| es | +0.0450 [0.0403, 0.0494] | +0.0153 [0.0125, 0.0181] |
| fr | +0.0457 [0.0411, 0.0505] | +0.0305 [0.0266, 0.0345] |
| it | +0.1107 [0.1041, 0.1172] | +0.1387 [0.1327, 0.1445] |
| nl | +0.0224 [0.0175, 0.0275] | +0.0329 [0.0289, 0.0371] |
| pl | +0.1075 [0.1016, 0.1132] | +0.1044 [0.0987, 0.1100] |
| ro | +0.0627 [0.0575, 0.0678] | +0.0315 [0.0273, 0.0358] |

### Honest reading — the harder eval now discriminates

The control comes off the F1 ≈ 1.0 ceiling (now **0.849–0.967**) on the harder families, so there is
room for a gain to register. **All 8 languages now show a positive, CI-clearing `v2 − control` Δ in
both seeds** — where the KLU-106 single-template eval pinned de/fr/nl/pl/ro at Δ ≈ 0 and surfaced an
`en` regression. The eval-design limitation is removed:

- **de/fr/nl/pl/ro**, previously parity at the ceiling, now show clear gains (pl +0.106, ro +0.047,
  de +0.038, fr +0.038, nl +0.028) — the v2 breadth fine-tune *does* help on a non-saturated surface.
- **en flips from −0.19 to +0.040**: the old result was an artifact of one EN template's `ORG_PARTY`
  surface; across two broader families v2 is *better* than the control, not worse.
- **it** remains the strongest gain (+0.125), consistent with the old +0.135 (its control was already
  off the ceiling) — a useful sanity anchor that the harder eval did not distort the one language that
  previously discriminated.
- **es** moves from a negligible +0.0008 to a real +0.030.

This is a **finding about eval quality**: the harder held-out general split measures a real v2
generalization gain that the saturated single-template split structurally could not. It is **not** a
validated SOTA / best-protector claim, and it does **not** touch the re-id signal — the real-skeleton
tracks (KLU-106 §2: leak_rate 0.000, Wilson UB ≈ 0.0034 ≤ 0.01, no dissociation break) remain the
headline re-id result. Native-speaker review / IAA (KLU-27) is still the gate before any external
F1-gain claim.

## Guards (schema-3)

- Every general-heldout cell labelled `contamination = clean_held_out` and `config_status = dev`.
- Held-out set is **template-disjoint** (5-gram Jaccard ≤ 0.10 vs training, gated) **and
  subject-disjoint** (empty `NATIONAL_ID` intersection with the training corpora, asserted per
  language) from the data v2 trained on.
- Fully synthetic, cleanly-licensed (the families reuse the existing synthetic locale generators).
- No SOTA / "best protector" / validated claim. Re-id headline unchanged (real-skeleton tracks).
- Mac-only, CPU scoring, no DO GPU burst.

## Reproduce

```bash
# 1. Carve the harder held-out set (klusai-datasets hardgeneral families; offline cache only).
HF_HUB_OFFLINE=1 python scripts/carve_hard_general_res19.py \
    --n-per-family 750 \
    --out runs/res19-heldout-hard-general \
    --manifest runs/res19-carve-manifest.json

# 2. Re-score the EXISTING v2 seeds + control against it (no retraining).
EUROPRIV_DEVICE=cpu PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \
  python scripts/scorecard_klu106.py \
    --manifest runs/klu106-train-manifest.json \
    --control runs/kp-deid-mdeberta-280m \
    --heldout-general runs/res19-heldout-hard-general \
    --skip-realskeleton \
    --out runs/res19-hard-general-scorecard.json
```
