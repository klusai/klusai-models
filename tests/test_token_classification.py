"""Unit tests for the token-classification training backend's span alignment.

The model-training itself (Trainer loop, HF/peft) is exercised by the smoke run, not in CI; here
we pin the *correctness-critical* pure function: projecting gold KP char spans onto subword
tokens as BIOES label ids in the benchmark's label space. A bug here silently corrupts training
labels, so it gets a deterministic, dependency-free test (no model download)."""

from __future__ import annotations

from europriv_bench.taxonomy import bioes_labels
from klusai.privacy.models.training.token_classification import _bioes_from_spans

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
