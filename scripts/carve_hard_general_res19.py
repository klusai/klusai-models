#!/usr/bin/env python3
"""RES-19 — carve the harder, *discriminating* held-out general eval set.

The KLU-106 held-out general split used a single template per language; the zero-shot control already
scored entity-F1 ≈ 1.000 on de/fr/nl/pl/ro, so the v2−control Δ was structurally pinned at ≈ 0 and the
eval could not discriminate a breadth gain (only ``it`` cleared a CI). This script builds the
replacement eval set from the RES-19 hard-general template families (``klusai-datasets``
``hardgeneral``): per language, two independent families (A = bureaucratic record card, B = narrative
correspondence) with harder PII surface forms (non-fixed positions, varied lead-ins, mixed register).

The produced set is **inherently held-out**: the families' skeletons are template-disjoint from the
training ``*_documents.TEMPLATES`` (5-gram Jaccard ≤ 0.10, gated in ``klusai-datasets``), so the
trained v2 seeds AND the zero-shot control never saw these surfaces. This script additionally ASSERTS,
per language, an **empty NATIONAL_ID-subject intersection** with the actual training corpora
(``klusai/ds-kp-general-*-50k`` — the data v2 trained on), keying on the same near-unique subject id
the KLU-106 carve and the re-id leak metric use. So the carve is BOTH template- and subject-disjoint
from training.

Output: a saved HF dataset dir (rows ``{text, spans, language, domain, family, genre}``) consumable by
``scorecard_klu106.py`` as ``--heldout-general``, plus a manifest recording the families, per-language
overlap-gate results, and the asserted-empty subject intersection.

Mac-only, no GPU, no network beyond the locally-cached training corpora (load_from offline cache).
Run synchronously:

    python scripts/carve_hard_general_res19.py \
        --n-per-family 750 \
        --out runs/res19-heldout-hard-general \
        --manifest runs/res19-carve-manifest.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click

from klusai.privacy.models.logger import get_logger

logger = get_logger("carve_hard_general_res19")

LANGUAGES = ("ro", "en", "pl", "de", "fr", "es", "it", "nl")
TRAIN_DATASET_TMPL = "klusai/ds-kp-general-{lang}-50k"
JACCARD_MAX = 0.10
SUBJECT_LABEL = "NATIONAL_ID"


def _national_ids(text: str, spans: list[dict]) -> set[str]:
    out: set[str] = set()
    for sp in spans:
        if sp["label"] == SUBJECT_LABEL:
            v = text[sp["start"]:sp["end"]].strip()
            if v:
                out.add(v)
    return out


def _train_subject_ids(lang: str) -> set[str]:
    """All NATIONAL_ID surface forms in the *training* corpus for ``lang`` (the data v2 trained on)."""
    from datasets import load_dataset

    d = load_dataset(TRAIN_DATASET_TMPL.format(lang=lang), split="train")
    ids: set[str] = set()
    for i in range(d.num_rows):
        ids |= _national_ids(d[i]["text"], d[i]["spans"])
    return ids


@click.command()
@click.option("--n-per-family", type=int, default=750,
              help="Rows per (language, family). Total eval rows = n_per_family * 2 families * 8 langs.")
@click.option("--seed", type=int, default=20260606, help="Carve seed (FIXED held-out set).")
@click.option("--out", default="runs/res19-heldout-hard-general", help="Saved HF dataset dir.")
@click.option("--manifest", default="runs/res19-carve-manifest.json")
def main(n_per_family, seed, out, manifest):
    from datasets import Dataset

    from klusai.privacy.datasets.data import hardgeneral as hg

    # --- 1. Independence / template-disjoint-from-training gate (deterministic, falsifiable). ---
    gate: dict[str, dict] = {}
    for lang in LANGUAGES:
        ab = hg.family_5gram_jaccard(lang)
        a_t = hg.vs_training_5gram_jaccard(lang, "A")
        b_t = hg.vs_training_5gram_jaccard(lang, "B")
        gate[lang] = {"family_a_vs_b_jaccard": round(ab, 4),
                      "family_a_vs_train_jaccard": round(a_t, 4),
                      "family_b_vs_train_jaccard": round(b_t, 4)}
        for name, j in (("A/B", ab), ("A/train", a_t), ("B/train", b_t)):
            if j > JACCARD_MAX:
                raise click.ClickException(f"{lang}: 5-gram Jaccard {name}={j:.4f} exceeds {JACCARD_MAX}")
    logger.info("overlap gate PASS (all <= %.2f): %s", JACCARD_MAX, json.dumps(gate))

    # --- 2. Generate the eval rows, then DROP any row that shares a NATIONAL_ID subject with the
    #        training corpus, so the carve is subject-disjoint from training (KLU-106 subject key).
    #        Low-entropy ids (e.g. Italian codice fiscale, derived from name+DOB) can occasionally
    #        collide with a training value; those rows are removed (not the training data), and the
    #        empty intersection is then ASSERTED post-filter per language. ---
    all_rows: list[dict] = []
    per_lang_counts: dict[str, dict] = {}
    subject_intersection: dict[str, int] = {}
    dropped_for_shared_subject: dict[str, int] = {}
    for lang in LANGUAGES:
        train_ids = _train_subject_ids(lang)
        rows = list(hg.generate_hard_general(lang, n_per_family, seed=seed))
        kept: list[dict] = []
        dropped = 0
        for r in rows:
            row_ids = _national_ids(r["text"], r["spans"])
            if row_ids & train_ids:
                dropped += 1
                continue
            kept.append(r)
        fam_counts = {"A": 0, "B": 0}
        ids: set[str] = set()
        for r in kept:
            fam_counts[r["family"]] += 1
            ids |= _national_ids(r["text"], r["spans"])
        # Assert disjointness now holds after the drop.
        inter = ids & train_ids
        if inter:
            raise click.ClickException(
                f"{lang}: {len(inter)} NATIONAL_ID subjects STILL shared with training post-filter "
                f"(e.g. {sorted(inter)[:3]})"
            )
        subject_intersection[lang] = 0
        dropped_for_shared_subject[lang] = dropped
        per_lang_counts[lang] = {"rows": len(kept), "families": fam_counts,
                                 "distinct_national_id_subjects": len(ids),
                                 "dropped_for_shared_train_subject": dropped}
        all_rows.extend(kept)
        logger.info("%s: train has %d NATIONAL_ID subjects; kept %d rows (A=%d B=%d), dropped %d for "
                    "shared subject; eval∩train = 0",
                    lang, len(train_ids), len(kept), fam_counts["A"], fam_counts["B"], dropped)
    logger.info("subject-disjoint from training: PASS (all intersections 0 post-filter)")

    # --- 4. Save the held-out set (uniform schema) + manifest. ---
    ds = Dataset.from_list([
        {"text": r["text"], "spans": r["spans"], "language": r["language"],
         "domain": r["domain"], "family": r["family"], "genre": r["genre"]}
        for r in all_rows
    ])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(out)
    logger.info("saved RES-19 hard held-out general split (%d rows) -> %s", ds.num_rows, out)

    man = {
        "issue": "RES-19",
        "purpose": "discriminating held-out general eval — 2 independent template families/language",
        "source": "klusai-datasets hardgeneral (template-disjoint from training, gated)",
        "languages": list(LANGUAGES),
        "families_per_language": 2,
        "n_per_family": n_per_family,
        "carve_seed": seed,
        "total_rows": ds.num_rows,
        "heldout_general_dir": out,
        "overlap_gate_jaccard_max": JACCARD_MAX,
        "overlap_gate_per_language": gate,
        "per_language_counts": per_lang_counts,
        "subject_label": SUBJECT_LABEL,
        "train_corpora": [TRAIN_DATASET_TMPL.format(lang=lang) for lang in LANGUAGES],
        "subject_intersection_with_train_per_language": subject_intersection,
        "rows_dropped_for_shared_train_subject_per_language": dropped_for_shared_subject,
        "subject_disjoint_from_train": all(v == 0 for v in subject_intersection.values()),
        "template_disjoint_from_train": True,
        "contamination": "clean_held_out",
        "config_status": "dev",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path(manifest).write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote carve manifest -> %s", manifest)
    click.echo(json.dumps({"total_rows": ds.num_rows, "overlap_gate_per_language": gate,
                           "subject_intersection_with_train_per_language": subject_intersection},
                          indent=2))


if __name__ == "__main__":
    main()
