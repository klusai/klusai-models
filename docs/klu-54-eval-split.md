# KLU-54 — fixing the leaky/trivial training eval split

## TL;DR

The KLU-44 training run reported `final_eval_loss` ~**7.2e-10** (sweep ~2e-5) — implausibly low
for token classification on real text. KLU-51 corroborated it: scoring `kp-deid` on the *same*
generator family it trained on (`ro-synthetic-v1`) gave entity-F1 **1.000**. Both are symptoms of
one root cause: **the eval split was not held out** — it shared the training corpus's generator
templates, so eval-loss measured **memorization of a handful of fixed sentence skeletons**, not
generalization.

This is now fixed: the eval split holds out **whole generator templates**, so train and eval share
no template+content. A short before/after confirmation run (same budget, same hyperparameters,
only the split changed) moves eval-loss from **0.0004 (leaky)** to **0.233 (disjoint)** — a
plausible, clearly non-trivial band.

> **The loss curve is NOT evidence of model quality.** Eval-loss here is a *split-sanity* signal
> and an input to early-stopping/checkpoint selection — nothing more. The only trustworthy quality
> numbers are the **EuroPriv-Bench harness leaderboard** scores on the contamination-free
> `ro-realskeleton-v1` track (entity F1 / F2, CNP leak-rate with Wilson CIs). Those are unaffected
> by this change.

## Root cause: a 6-template generator + a shuffled-head split

The `klusai/ds-kp-general-{ro,en,pl}-50k` corpora are produced by a small LocalePack generator.
Reducing each row to its *template skeleton* (replace every gold PII span surface with `<LABEL>`)
shows each 50k corpus is built from **exactly 6 distinct templates**:

```
'<PERSON> (<EMAIL>, tel. <PHONE>) a solicitat o programare pentru <DATE> la adresa <ADDRESS>.'
'Pacient: <PERSON>, CNP <NATIONAL_ID>, internat la data de <DATE>. Telefon de contact: <PHONE>.'
'Subsemnatul <PERSON>, CNP <NATIONAL_ID>, domiciliat în <ADDRESS>, telefon <PHONE>, declar ...'
... (6 total, ~2 per domain: legal / clinical / general / admin)
```

The old split (both `training/token_classification.py` and `scripts/full_run_klu44.py`) was:

```python
raw = raw.shuffle(seed=seed)
n_eval = int(n * eval_fraction)
eval_rows  = raw.select(range(n_eval))      # head of shuffle
train_rows = raw.select(range(n_eval, n))   # tail
```

A uniform shuffle puts **all 6 templates in both splits**. So at eval time the model sees the same
6 skeletons it trained on, with only the PII fillers swapped. Learning "these 6 sentences + what
each label type looks like" drives eval-loss to ~0 — exactly the ~7.2e-10 observed. Early-stopping
and the LR×LoRA-r sweep were therefore selecting on a near-constant, meaningless number.

## The fix: template-disjoint held-out split

`template_disjoint_split` (in `klusai/privacy/models/training/token_classification.py`) groups
rows by `template_skeleton`, deterministically orders the *templates* by a seeded hash, and peels
whole templates into eval until it reaches `eval_fraction` of the rows — keeping at least one
template in train. It asserts disjointness (raises if any template lands in both) and returns a
small `info` dict for honest logging:

```
{'total_templates': 6, 'eval_templates': 2, 'train_templates': 4,
 'eval_rows': ..., 'train_rows': ..., 'disjoint': True}
```

Every eval row's structure is therefore **absent from train** — a provably disjoint, non-trivial
held-out set. Both training entry points (`train_token_classification` and `full_run_klu44.py`)
now use it. For the multilingual full run, templates are language-specific text, so a single
merged skeleton partition holds out structures across the RO+EN+PL mix.

> **Caveat — this is an upper bound on "held-out-ness" given current data.** With only 6 templates
> per corpus we can guarantee *template* disjointness, but eval still comes from the same generator
> *family* (same faker-style PII distributions, same label taxonomy). It is a genuine
> generalization test against unseen sentence structures, not against a different domain or a real
> corpus. The stronger guarantee — eval on a *different* generator family or a small real dev set —
> is a data task (more templates / a real-corpus dev slice), tracked separately; the harness
> `ro-realskeleton-v1` track already provides the contamination-free *quality* measure.

## Before/after (short confirmation run)

`scripts/confirm_klu54_split.py` runs two identical bounded finetunes that differ **only** in the
split. RO corpus, mDeBERTa-v3-base, LoRA r=16, lr=3e-4, 2 epochs, 4000 train / 800 eval, batch 16,
seed 0, MPS (Mac Studio M3 Ultra).

| split | how eval is formed | `eval_loss` | train samples/s |
|-------|--------------------|-------------|-----------------|
| **leaky** (old) | shuffled head of corpus — shares all 6 templates with train | **0.00043** | 187.5 |
| **disjoint** (new, KLU-54) | 2 held-out templates, 0 overlap with train's 4 | **0.233** | 190.4 |

The leaky split reproduces the implausible near-zero pattern (≈540× lower). The disjoint split
lands in a plausible band where eval-loss differences between configs are meaningful again — so
the sweep / early-stopping / checkpoint selection now carry signal.

## Hardware utilization (M3 Ultra, MPS)

The confirmation run is a **short, bounded** finetune. During the active training window, GPU
`Device Utilization %` (sampled via `ioreg -c IOAccelerator`, no root) sustained a **busy-sample
mean of ~91%, peak 94%**, at **~199 train samples/s** — consistent with the KLU-48 finding that
batch-16 single-process fp32 already near-saturates this memory-bandwidth-bound 280M encoder
(scaling the batch / adding workers regresses throughput; see `docs/klu-48-max-util.md`).

**Sustained package power draw was not captured:** `sudo powermetrics` requires root and the run
environment had no passwordless sudo, and `macmon`/`asitop` are not installed on this machine.
The GPU-utilization + throughput figures above are the available saturation evidence; a power
sample (`sudo powermetrics --samplers gpu_power`) should be taken on the next interactive full run.

## Where the old loss was cited as quality (flagged)

- **`runs/kp-deid-mdeberta-280m/README.md` (model card)** cited the leaky sweep eval-losses
  (`0.000020 (best)`, etc.) in the "Hyperparameter sweep" table and called the best the "best
  held-out eval-loss". This PR adds a note to that table marking those numbers as products of the
  pre-KLU-54 leaky split (not a quality signal); the card's **quality** claims already (correctly)
  rest only on the harness `ro-realskeleton-v1` scores (F1 0.741 / 0% CNP leak), which are
  unaffected. The model artifact itself is not retrained here (this is a split + methodology fix);
  a re-run of the published model under the disjoint split is a follow-up.
- The eval-losses in `docs/klu-45-mps-vs-cpu.md` and `docs/klu-48-max-util.md` (0.0005–0.028) come
  from the same leaky split, but those docs measure **throughput / numerical parity**, never model
  quality, so their conclusions stand. They are left as-is (re-running those benchmarks under the
  disjoint split is optional and out of scope for KLU-54).
