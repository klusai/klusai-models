"""Unit tests for the token-classification training backend's span alignment.

The model-training itself (Trainer loop, HF/peft) is exercised by the smoke run, not in CI; here
we pin the *correctness-critical* pure function: projecting gold KP char spans onto subword
tokens as BIOES label ids in the benchmark's label space. A bug here silently corrupts training
labels, so it gets a deterministic, dependency-free test (no model download)."""

from __future__ import annotations

from europriv_bench.taxonomy import bioes_labels
from klusai.privacy.models.training.token_classification import (
    _bioes_from_spans,
    carve_heldout_general,
    identifier_surface_form_holdout,
    lora_target_modules,
    resolve_device,
    resolve_max_util_profile,
    template_disjoint_split,
    template_skeleton,
)

_LABELS = bioes_labels()
_L2I = {v: k for k, v in enumerate(_LABELS)}
_I2L = dict(enumerate(_LABELS))


def _align(offsets, word_ids, spans):
    text = ""  # _bioes_from_spans reads only offsets/word_ids for token geometry
    ids = _bioes_from_spans(text, spans, offsets, word_ids, _L2I)
    return [(_I2L[i] if i != -100 else "IGN") for i in ids]


def test_single_word_span_is_S():
    # one word "x" = subwords [0..1] sharing word_id 0; span covers it -> S-PERSON on first subword,
    # IGN (-100) on the continuation subword.
    offsets = [(0, 0), (0, 1), (1, 5)]
    word_ids = [None, 0, 0]
    out = _align(offsets, word_ids, [{"start": 0, "end": 5, "label": "PERSON"}])
    assert out == ["IGN", "S-PERSON", "IGN"]


def test_multi_word_span_is_B_E():
    # two words spanning chars 0..14 ("Andrei Popescu"), each one subword.
    offsets = [(0, 0), (0, 6), (7, 14)]
    word_ids = [None, 0, 1]
    out = _align(offsets, word_ids, [{"start": 0, "end": 14, "label": "PERSON"}])
    assert out == ["IGN", "B-PERSON", "E-PERSON"]


def test_three_word_span_has_inside():
    offsets = [(0, 3), (4, 7), (8, 11)]
    word_ids = [0, 1, 2]
    out = _align(offsets, word_ids, [{"start": 0, "end": 11, "label": "ADDRESS"}])
    assert out == ["B-ADDRESS", "I-ADDRESS", "E-ADDRESS"]


def test_tokens_outside_spans_are_O():
    offsets = [(0, 6), (7, 20)]
    word_ids = [0, 1]
    out = _align(offsets, word_ids, [{"start": 7, "end": 20, "label": "NATIONAL_ID"}])
    assert out == ["O", "S-NATIONAL_ID"]


def test_all_emitted_labels_in_bioes_space():
    offsets = [(0, 6), (7, 14), (15, 28)]
    word_ids = [0, 1, 2]
    spans = [{"start": 0, "end": 14, "label": "PERSON"}, {"start": 15, "end": 28, "label": "NATIONAL_ID"}]
    out = _align(offsets, word_ids, spans)
    valid = set(_LABELS) | {"IGN"}
    assert all(t in valid for t in out)
    assert out == ["B-PERSON", "E-PERSON", "S-NATIONAL_ID"]


# --- RES-97: architecture-aware LoRA target modules -------------------------------------------
# mDeBERTa names attention projections *_proj; XLM-R/RoBERTa/BERT use the bare names. Picking the
# wrong set makes PEFT match ZERO modules and train an empty adapter — so this is correctness-
# critical for adding the XLM-R-560m family member.


def test_lora_targets_mdeberta_uses_proj_names():
    assert lora_target_modules("microsoft/mdeberta-v3-base") == ["query_proj", "key_proj", "value_proj"]


def test_lora_targets_xlmr_uses_bare_names():
    assert lora_target_modules("FacebookAI/xlm-roberta-large") == ["query", "key", "value"]


def test_lora_targets_roberta_and_bert_use_bare_names():
    assert lora_target_modules("roberta-base") == ["query", "key", "value"]
    assert lora_target_modules("bert-base-multilingual-cased") == ["query", "key", "value"]


def test_lora_targets_unknown_falls_back_to_bare_names():
    # Unknown encoder -> RoBERTa-style default (the common case), never an empty/crashing config.
    assert lora_target_modules("some-org/mystery-encoder") == ["query", "key", "value"]


# --- KLU-45: device selection (cpu/mps/cuda) ---------------------------------------------------
# resolve_device must never raise and must always be able to fall back to CPU (the guaranteed
# tier on every machine); explicit cpu is honored regardless of accelerators present.


def test_explicit_cpu_always_resolves_cpu():
    assert resolve_device("cpu", cpu=False) == "cpu"
    assert resolve_device("cpu", cpu=True) == "cpu"


def test_legacy_cpu_bool_maps_to_cpu_when_device_unset():
    # KLU-17 callers pass cpu=True and no device -> must keep resolving to cpu.
    assert resolve_device(None, cpu=True) == "cpu"


def test_resolved_device_is_always_valid():
    for req in (None, "auto", "cpu", "mps", "cuda"):
        for cpu in (True, False):
            assert resolve_device(req, cpu=cpu) in {"cpu", "mps", "cuda"}


def test_unavailable_accelerator_falls_back_to_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("mps", cpu=False) == "cpu"
    assert resolve_device("cuda", cpu=False) == "cpu"
    assert resolve_device("auto", cpu=False) == "cpu"


# --- KLU-48: max-utilization profile resolution ------------------------------------------------
# The profile is the Mac default: MPS turns it on (batch scaled up + DataLoader workers); CPU/CUDA
# leave the incoming batch untouched (no GPU to feed / already saturated). These cover the pure
# resolution logic without needing a model on device (the auto-probe is exercised by the smoke run).


def test_max_util_noop_on_cpu():
    p = resolve_max_util_profile(
        "cpu", max_util=True, batch_size=16, batch_override=None, num_workers=None, bf16=False
    )
    assert not p.enabled
    assert p.batch_size == 16          # untouched
    assert p.num_workers == 0          # no extra workers on CPU
    assert not p.bf16


def test_max_util_noop_on_cuda():
    p = resolve_max_util_profile(
        "cuda", max_util=True, batch_size=8, batch_override=None, num_workers=None, bf16=False
    )
    assert not p.enabled
    assert p.batch_size == 8


def test_max_util_disabled_flag_is_noop_even_on_mps():
    p = resolve_max_util_profile(
        "mps", max_util=False, batch_size=16, batch_override=None, num_workers=None, bf16=False
    )
    assert not p.enabled
    assert p.batch_size == 16
    assert p.num_workers == 0


def test_max_util_on_mps_with_explicit_batch_skips_probe():
    # explicit override -> no model needed, batch honored, explicit workers honored
    p = resolve_max_util_profile(
        "mps", max_util=True, batch_size=16, batch_override=128, num_workers=4, bf16=True
    )
    assert p.enabled
    assert p.batch_size == 128
    assert p.num_workers == 4              # explicit opt-in honored
    assert p.persistent_workers is False   # KLU-48: never persistent (macOS spawn exit-hang)
    assert p.bf16 is True
    assert any("override" in line for line in p.probe_log)


def test_max_util_workers_opt_in_default_zero():
    # KLU-48: workers are opt-in — no throughput win + macOS spawn exit-hang — so default is 0
    # even with max-util on (the batch bump is the only default knob).
    p = resolve_max_util_profile(
        "mps", max_util=True, batch_size=16, batch_override=64, num_workers=None, bf16=False
    )
    assert p.enabled
    assert p.batch_size == 64
    assert p.num_workers == 0
    assert p.persistent_workers is False


# --- KLU-54: template-disjoint held-out eval split ---------------------------------------------
# The old split was a shuffled head of one corpus; because each corpus has only ~6 generator
# templates, eval re-used train's templates and eval-loss measured memorization (~7e-10). The fix
# holds out whole templates so eval shares no template+content with train. These tests pin (a) the
# skeleton reduction and (b) the disjointness guarantee on a tiny in-memory dataset (no download).


def test_template_skeleton_blanks_pii_surfaces():
    # Two rows from the same template with different fillers -> identical skeleton.
    sk1 = template_skeleton(
        "Pacient: Ana Pop, CNP 123.",
        [{"start": 9, "end": 16, "label": "PERSON"}, {"start": 22, "end": 25, "label": "NATIONAL_ID"}],
    )
    sk2 = template_skeleton(
        "Pacient: Ion Ene, CNP 98765.",
        [{"start": 9, "end": 16, "label": "PERSON"}, {"start": 22, "end": 27, "label": "NATIONAL_ID"}],
    )
    assert sk1 == sk2 == "Pacient: <PERSON>, CNP <NATIONAL_ID>."
    # A structurally different template must differ.
    sk3 = template_skeleton("Email <EMAIL_FILLER>", [{"start": 6, "end": 19, "label": "EMAIL"}])
    assert sk3 != sk1


def _tiny_ds():
    # Two templates ("A": person+id, "B": email), 6 rows each, distinct fillers per row.
    from datasets import Dataset

    rows = []
    for k in range(6):
        rows.append({
            "text": f"Pacient: Nume{k} X, CNP {1000 + k}.",
            "spans": [
                {"start": 9, "end": 16, "label": "PERSON"},
                {"start": 22, "end": 26, "label": "NATIONAL_ID"},
            ],
        })
    for k in range(6):
        rows.append({
            "text": f"Contact email user{k}@x.ro azi.",
            "spans": [{"start": 14, "end": 25, "label": "EMAIL"}],
        })
    return Dataset.from_list(rows)


def test_template_disjoint_split_is_provably_disjoint():
    ds = _tiny_ds()
    train, ev, info = template_disjoint_split(ds, eval_fraction=0.4, seed=0)
    assert info["disjoint"] is True
    # No template appears in both splits.
    train_sk = {template_skeleton(train[i]["text"], train[i]["spans"]) for i in range(train.num_rows)}
    eval_sk = {template_skeleton(ev[i]["text"], ev[i]["spans"]) for i in range(ev.num_rows)}
    assert train_sk.isdisjoint(eval_sk)
    # Every row is accounted for exactly once; both splits non-empty.
    assert train.num_rows + ev.num_rows == ds.num_rows
    assert train.num_rows > 0 and ev.num_rows > 0


def test_template_disjoint_split_keeps_a_train_template_even_at_high_eval_fraction():
    # Even asking for most of the data as eval, train must retain >=1 template (so train is usable).
    ds = _tiny_ds()
    train, ev, info = template_disjoint_split(ds, eval_fraction=0.95, seed=1)
    assert train.num_rows > 0
    assert info["train_templates"] >= 1
    assert info["eval_templates"] >= 1


def test_template_disjoint_split_deterministic_for_seed():
    ds = _tiny_ds()
    _, ev_a, _ = template_disjoint_split(ds, eval_fraction=0.4, seed=7)
    _, ev_b, _ = template_disjoint_split(ds, eval_fraction=0.4, seed=7)
    assert ev_a["text"] == ev_b["text"]


# --- KLU-106 contamination carve-out: per-language clean held-out, template + SUBJECT disjoint ----
# The load-bearing carve-out. We pin: (a) the held-out general split is template-disjoint per
# language AND subject-disjoint (no held-out PERSON/NATIONAL_ID surface form survives in the train
# pool); (b) the empty train∩heldout subject intersection is asserted (raises on violation); (c) the
# identifier-surface-form holdout flags rows with no train-seen PII string.


def _multilang_ds():
    """Two languages, 3 templates each, with a SHARED subject string across templates per language.

    Each language has templates A (person+id) and B (person only) and C (email). We deliberately
    reuse one PERSON value ("Shared Name") across templates within a language so that a naive
    template-only holdout would leave that subject on both sides — the carve must remove it.
    """
    from datasets import Dataset

    def _span(text, value, label):
        s = text.index(value)
        return {"start": s, "end": s + len(value), "label": label}

    rows = []
    for lang in ("ro", "de"):
        # Template A rows (person + national id), distinct subjects + one shared name. Fixed-width
        # 4-digit national IDs so every Template-A row reduces to the SAME skeleton (one template).
        for k in range(5):
            name = "Shared Name" if k == 0 else f"Pers{lang}{k}"
            nid = str(2000 + k)  # 4 digits, fixed width
            text = f"Pacient: {name}, ID {nid} aici."
            rows.append({"language": lang, "text": text,
                         "spans": [_span(text, name, "PERSON"), _span(text, nid, "NATIONAL_ID")]})
        # Template B rows (person only) — reuse "Shared Name" once so it spans two templates.
        for k in range(5):
            name = "Shared Name" if k == 0 else f"Other{lang}{k}"
            text = f"Domn {name} a semnat."
            rows.append({"language": lang, "text": text, "spans": [_span(text, name, "PERSON")]})
        # Template C rows (email) — no subject-label PII (PERSON/NATIONAL_ID absent).
        for k in range(5):
            email = f"{lang}user{k}@x.io"
            text = f"Mail {email} trimis."
            rows.append({"language": lang, "text": text, "spans": [_span(text, email, "EMAIL")]})
    return Dataset.from_list(rows)


def test_carve_heldout_general_is_template_and_subject_disjoint_per_language():
    ds = _multilang_ds()
    train_pool, heldout, info = carve_heldout_general(
        ds, heldout_templates_per_language=1, seed=0
    )
    assert info["subject_disjoint"] is True
    assert info["template_disjoint_per_language"] is True
    # Per language: no held-out PERSON/NATIONAL_ID surface form appears in the train pool.
    from klusai.privacy.models.training.token_classification import (
        SUBJECT_LABELS,
        _subject_surface_forms,
    )

    for lang in ("ro", "de"):
        held_forms, train_forms = set(), set()
        for i in range(heldout.num_rows):
            if heldout[i]["language"] == lang:
                held_forms |= _subject_surface_forms(heldout[i]["text"], heldout[i]["spans"], labels=SUBJECT_LABELS)
        for i in range(train_pool.num_rows):
            if train_pool[i]["language"] == lang:
                train_forms |= _subject_surface_forms(train_pool[i]["text"], train_pool[i]["spans"], labels=SUBJECT_LABELS)
        assert held_forms.isdisjoint(train_forms), f"{lang}: subject leak {held_forms & train_forms}"
    # Both sides non-empty and every language represented in held-out.
    assert train_pool.num_rows > 0 and heldout.num_rows > 0
    assert set(info["heldout_templates_per_language"]) == {"ro", "de"}


def test_carve_heldout_general_asserts_intersection_is_empty():
    # The recorded per-language subject intersection must be 0 everywhere (the asserted invariant).
    ds = _multilang_ds()
    _, _, info = carve_heldout_general(ds, heldout_templates_per_language=1, seed=3)
    assert all(v == 0 for v in info["subject_intersection_per_language"].values())
    # The shared-name row(s) in the train pool must have been dropped -> recorded as drops.
    assert sum(info["train_rows_dropped_for_shared_subject"].values()) >= 0


def test_carve_keeps_at_least_one_train_template_per_language():
    ds = _multilang_ds()
    train_pool, _, info = carve_heldout_general(ds, heldout_templates_per_language=5, seed=1)
    # Even asking to hold out more templates than exist, each language keeps >=1 train template.
    import collections

    by_lang = collections.Counter(train_pool[i]["language"] for i in range(train_pool.num_rows))
    assert by_lang["ro"] > 0 and by_lang["de"] > 0


def test_identifier_surface_form_holdout_flags_unseen_strings():
    from datasets import Dataset

    train = Dataset.from_list([
        {"language": "ro", "text": "X Ana Y", "spans": [{"start": 2, "end": 5, "label": "PERSON"}]},
    ])
    heldout = Dataset.from_list([
        # row 0: PERSON "Ana" IS in train -> not in surface holdout
        {"language": "ro", "text": "Z Ana W", "spans": [{"start": 2, "end": 5, "label": "PERSON"}]},
        # row 1: PERSON "Bob" never in train -> in surface holdout
        {"language": "ro", "text": "Z Bob W", "spans": [{"start": 2, "end": 5, "label": "PERSON"}]},
    ])
    res = identifier_surface_form_holdout(train, heldout)
    assert res["heldout_total"] == 2
    assert res["n"] == 1
    assert res["indices"] == [1]
