# KLU-109 — kp-anon held-out privacy-utility scorecard

**Status: `config_status=dev`. Contamination-controlled. NOT citable / NOT SOTA / NOT a "best
anonymizer" claim.** Validation gated on KLU-27 (native-speaker + IAA). This artifact discharges the
KLU-109 acceptance — *"produce the rigorous held-out privacy-utility scorecard + a committed
Pareto-frontier figure"* — **not** "train a great model." Machine-readable companion:
[`klu-109-scorecard.json`](klu-109-scorecard.json); frontier figure:
[`klu-109-kp-anon-frontier.svg`](klu-109-kp-anon-frontier.svg).

## Setup

| | |
|---|---|
| Model | `kp-anon-mdeberta-280m` — span-replacement anonymizer (LoRA mDeBERTa detector + pseudonymization policy), seeds 0 & 1 |
| Control | **redaction baseline (KLU-104)**: the **SAME** trained detector used as a plain redactor (blanket-mask every detected span with `█`) |
| Privacy metric | `redaction_leakage.leak_rate` — per distinct subject `(doc, country, normalized value)`, leaks iff a re-identifying fragment (≥4-char run) of the gold value **survives verbatim in the output** (value-search from gold offsets, never re-detected), 95% Wilson CI ↓ |
| Utility metrics | `information_retention` ↑ and `1 − mask_token_ratio` ↑ (unmasking utility); paired 2000-iter doc-bootstrap Δ(kp-anon − baseline) |
| Tracks | Track-C frontier on `ro-realskeleton-v1` (CNP, 1123 subj) + `pl-realskeleton-v1` (PESEL, 1096 subj), both `clean_held_out` |
| Training | on-device MPS, 2 seeds, 18k balanced samples, 3 epochs, **15.4 min** wall (not stopped early), 93% mean / 94% peak GPU util |

## Result — per track (both seeds)

| Track | seed | leak: baseline → kp-anon (Wilson UB) | info-retention Δ | unmasking-utility Δ (CI excl 0) | bijection in/cross-doc | dominates? |
|---|---|---|---|---|---|---|
| ro-realskeleton-v1 | 0 | 0.0% → **0.0%** (UB 0.34%) | 0.0 | **+0.251** [0.249, 0.253] | 0.998 / 0.955 | **yes** |
| ro-realskeleton-v1 | 1 | 0.0% → **0.0%** (UB 0.34%) | 0.0 | **+0.224** [0.222, 0.226] | 1.000 / 0.952 | **yes** |
| pl-realskeleton-v1 | 0 | 0.0% → **0.0%** (UB 0.35%) | 0.0 | **+0.236** [0.234, 0.237] | 1.000 / 0.955 | **yes** |
| pl-realskeleton-v1 | 1 | 0.0% → **0.0%** (UB 0.35%) | 0.0 | **+0.226** [0.224, 0.227] | 1.000 / 0.937 | **yes** |

**≥2-seed aggregate:** unmasking-utility Δ across seeds — ro `[+0.224, +0.251]`, pl `[+0.226,
+0.236]`; the per-seed bootstrap CI excludes 0 in **every** cell and the point deltas agree in sign,
so the utility gain **clears seed-noise** on both languages. Privacy is **unchanged by the policy**
(`leak_unchanged_by_policy = true`): kp-anon leaks 0.0% where the blanket-mask baseline leaks 0.0%.

## Reading

On the held-out ro/pl real-skeleton carve-out, kp-anon **dominates** the redaction baseline on the
privacy-utility plane, for **both frontier languages and both seeds**:

* **Privacy axis — no cost.** At equal detector recall, kp-anon's re-identification leak (0.0%, Wilson
  UB ≤ 0.35%) is identical to the blanket-mask baseline's (0.0%). Pseudonymization does **not** trade
  away privacy.
* **Utility axis — large gain.** kp-anon removes essentially all mask glyphs (`1 − mask_token_ratio`
  improves by **+0.22 to +0.25**, CI excludes 0 everywhere) at **identical information retention**
  (Δ = 0 — the surrogate substitution touches only PII spans, leaving non-PII context verbatim).
* **Joinability.** Near-perfect in-document surrogate bijection (≈0.998–1.000) and a strong
  cross-document bijection (≈0.94–0.96) — the pseudonymized corpus stays linkable.

This is the Track-C frontier claim — *"naive redaction destroys utility; a trained pseudonymizing
anonymizer doesn't, at the same privacy"* — demonstrated on `dev` data for ≥2 languages. **Honestly
labelled `dev`, not citable**, gated on KLU-27 before any public/SOTA claim.

### Caveat — surrogate fluency, not measured here

The utility axis measures *structural* preservation (mask-token ratio, token retention), **not**
native fluency. kp-anon's surrogates are realistic structure-preserving fillers from small
multi-locale pools (a fake name / a CNP-shaped number / a plausible date), deliberately *not*
locale-perfect. Whether a RO/PL reader finds the pseudonymized text natural is a KLU-27 IAA question,
out of scope here.

## Surrogate leak-safety fix (root-caused during this close-out)

An earlier draft of the pseudonymizer generated each surrogate independently of its source's surface
digits but did **not** reject coincidental fragment collisions. On this eval that surfaced a spurious
~2–4% leak: the leak metric flags a subject if **any** ≥4-char run of its gold value survives anywhere
in the document, and random 13-digit CNP/PESEL-shaped surrogates occasionally (a) shared a 4-digit run
with their **own** source value, and (b) — the larger effect — shared a 4-digit run with a **different**
subject's gold ID elsewhere in the same document (cross-field collision). These are coincidental
n-gram echoes, not detector misses (detector recall was ~100%, so the baseline leaked 0%), but the
metric correctly scores them as leaks.

Fix (`klusai/privacy/models/anon.py`): the surrogate generator now **rejection-samples** — it redraws
(deterministically, via a salted `bump`) until the candidate shares no ≥4-char run with its own source
value *nor* with any other detected value co-occurring in documents where it appears
(`_shares_fragment` mirrors the metric's `_value_survives` with the same `_MIN_LEAK_FRAGMENT = 4`).
This restores the load-bearing invariant — substitution contributes **zero** leak; only a detector
miss can leak, exactly as for the blanket-mask baseline — driving the measured leak to 0.0% on both
languages. The fix is determinism-preserving (surrogates stay corpus-stable, bijection intact) and is
covered by `tests/test_anon.py`.

## Guards

`config_status=dev`; no SOTA / "best" / validated claim (gated KLU-27); privacy axis attributable to
detection recall only (substitution is rejection-checked to introduce zero leak); per-subject
`(doc, country, value)` dedup; ≥2-seed variance reported (min/max + per-seed CIs).
