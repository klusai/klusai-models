# KLU-106 — kp-deid v2: multilingual protector + the contamination carve-out

## What this is

`kp-deid` **v2** is a multilingual PII/PHI token classifier (mDeBERTa-280m, LoRA) trained on **8
languages** — the v1 languages **ro/en/pl** plus the 5 **T1** LocalePacks **de/fr/es/it/nl**
(`klusai/ds-kp-general-{lang}-50k`, published in KLU-107) — **on-device** (PyTorch-MPS, M3 Ultra)
under hard runaway bounds, using the corrected template-disjoint split (KLU-54) extended to all
languages.

Acceptance for KLU-106 is **"produce the scorecard on the fixed held-out eval set,"** not "train a
good model." There is **NO SOTA / "best protector" / validated claim** — that is gated on KLU-27
(native-speaker / IAA sign-off) and ≥2 template families per language. The model card stays `dev`.

## The load-bearing bit: the contamination carve-out

v2 trains on the `ds-kp-general-*` corpora, so scoring v2 on those same general configs is
`in_distribution` and is **excluded from the headline gain**. The "material F1 gain" must be shown
on a **clean surface v2 never trained on**. Two such surfaces:

1. **A per-language held-out general split, carved out BEFORE training** — template- AND
   subject-disjoint (this doc).
2. **The real-skeleton tracks** (`ro-realskeleton-v1`, `pl-realskeleton-v1`,
   `it-realskeleton-v1`) — authored real-structure documents no model trained on
   (`it-realskeleton-v1` is scored when published on the hub; else skip-and-reported).

### Why "subject-disjoint" needs the right subject key

The bug that bit the program twice (KLU-51/54) was a held-out split that shared structure (and could
re-absorb held-out rows) with train, so "held-out" F1 measured memorization. Template-disjointness
alone (KLU-54) is necessary but **not sufficient**: a random balanced draw from the full 50k could
re-absorb a held-out *subject* via a different template.

So v2 also enforces **subject-level disjointness** — but the subject key matters. Measured on the
corpora:

| label | distinct values | nature |
|-------|-----------------|--------|
| `NATIONAL_ID` | ≈33,205 over 33,206 RO rows | **near-unique per row** — the re-id-bearing identifier |
| `PERSON` | ≈128 names | a tiny faker pool, each repeated hundreds of times |

Keying the "subject" on `PERSON` would make subject-disjointness **impossible** (every name appears
on both sides). Keying it on the near-unique, re-id-bearing **`NATIONAL_ID`** — exactly the value
whose leak the program measures — makes disjointness both **meaningful** and **achievable without
annihilating the train pool**. (Rows with no national-ID carry no re-id subject and are partitioned
by template alone; the re-id-contamination concern is vacuous for them.)

### The carve (`carve_heldout_general`, in `training/token_classification.py`)

Per language, on the merged 8-language corpus, BEFORE the ≤40k down-sample:

1. Hold out whole generator template(s) (template-disjoint, KLU-54 extended per language; ≥1 train
   template always retained).
2. Reserve the held-out rows' `NATIONAL_ID` values as the held-out subject pool.
3. **Drop from the train pool any row sharing a held-out `NATIONAL_ID`** (closes the re-absorption
   hole). Because national-IDs are near-unique, this drops ≈0 rows yet guarantees disjointness.
4. **HARD-assert** an empty `train ∩ heldout` subject intersection **per language**, and assert
   per-language template disjointness. The job refuses to proceed on any violation.

Measured on RO 50k (1 held-out template): train pool **41,718** rows, held-out **8,281** rows, **1**
row dropped for a duplicate national-ID, subject intersection **0**.

The held-out split is then **re-asserted empty after the balanced down-sample** (the exact step the
bug bit), saved to disk as a **fixed** eval set, and scored identically for v2 and the control.

### Identifier-surface-form holdout

A second, stricter memorization tripwire: held-out rows whose **every** PII surface string (any
label) is absent from the training set. F1 reported on this subset alongside the full held-out
catches surface-form memorization.

## Rigorous claims (what the scorecard reports)

* **F1 gain = bootstrap-CI'd Δ(v2 − control)** on the **same** held-out set, per language. The
  control is the **zero-shot KLU-51 `kp-deid`** scored on that same held-out set (NOT the published
  0.46–0.52). The headline requires the **Δ CI to exclude 0** per language (95% paired-by-document
  percentile bootstrap).
* **≥2 seeds** for the headline delta (bf16/MPS nondeterminism — a gain inside seed-noise is not a
  gain): mean/min/max F1 + Δ over completed seeds.
* **re-id leak ~0** on clean_held_out real-skeleton, per-distinct-subject, **Wilson upper bound** ≤ a
  pre-registered threshold (not point-0).
* **detection-F1 gain AND re-id-leak Δ side-by-side per track**, surfacing (not averaging away) any
  config where F1 rises but leak ALSO rises — the **dissociation breaking** — with an explicit flag.
* Every scorecard cell carries schema-3 labels (`contamination`, `config_status=dev`).

## Hard bounds (runaway guard)

* ≤ ~40k samples total, **balanced across the 8 languages** (NOT 50k each).
* ≤ 3 epochs (hard-capped in the CLI).
* **wall-clock stop ~3 h with stop-and-report** — no seed is started that cannot plausibly finish in
  the remaining budget; a partial scorecard is produced from whatever finished.
* Fixed eval set (carved once, saved to disk). Fits MPS → not blocked by KLU-14 (no GPU burst).

## Saturation / utilization

Per KLU-48 the measured throughput optimum for this 280M encoder on MPS is **batch-16
single-process fp32** (it is memory-bandwidth-bound and already near-saturated; scaling the batch /
adding workers regresses it). `--max-util` / `--bf16` remain opt-in. The run reports sustained GPU
`Device Utilization %` (ioreg, no root), peak unified memory, and train throughput; sustained power
draw (W) is captured only when passwordless `sudo powermetrics` is available, else the exact command
is recorded (we never fabricate a power number).

## Scripts

* `scripts/train_v2_klu106.py` — bounded 8-language training + carve-out + assertions; writes the
  per-seed model artifacts, the fixed held-out split, and `runs/klu106-train-manifest.json`.
* `scripts/scorecard_klu106.py` — scores v2 seeds + the zero-shot control on the fixed held-out set
  and the real-skeleton tracks; writes `runs/klu106-scorecard.json`.

> Numbers (per-language Δ CIs, seed variance, leak Wilson UBs, utilization) are filled into this doc
> and the model card from the produced manifest + scorecard after the run.
