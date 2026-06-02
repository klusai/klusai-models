"""Tests for klusai.privacy.sdk — the adoption-layer SDK (KLU-20).

Two layers:
  * pure-logic tests for `_bioes_to_spans` + the default-model wiring (no model download); and
  * an integration test that the live `extract_pii()` reproduces the EuroPriv-Bench
    `KpModelAdapter` predictions (skipped if the HF model/weights aren't available offline).
"""

from __future__ import annotations

import os

import pytest

from klusai.privacy.sdk import DEFAULT_MODEL, Span, _bioes_to_spans, extract_pii


def test_default_model_is_the_shipped_h1_weight():
    # kp-deid-moe does not exist yet (H2 maybe); the default must resolve to a real H1 weight.
    assert DEFAULT_MODEL == "klusai/kp-deid-mdeberta-280m"


def test_bioes_to_spans_offsets_index_back_to_surface_text():
    text = "Ion Popescu lives in Bucharest"
    # B-/E-PERSON over the first two tokens, S-ADDRESS over the last.
    tags = ["B-PERSON", "E-PERSON", "O", "O", "S-ADDRESS"]
    spans = _bioes_to_spans(text, tags)
    assert spans == [(0, 11, "PERSON"), (21, 30, "ADDRESS")]
    # The defining invariant: char offsets slice back to the surface text.
    assert text[0:11] == "Ion Popescu"
    assert text[21:30] == "Bucharest"


def test_bioes_to_spans_single_token_entity():
    text = "Contact ion@example.com now"
    tags = ["O", "S-EMAIL", "O"]
    assert _bioes_to_spans(text, tags) == [(8, 23, "EMAIL")]
    assert text[8:23] == "ion@example.com"


def _model_available() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    # If we're offline, the snapshot must already be cached for the pipeline to load.
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
        cached = try_to_load_from_cache(DEFAULT_MODEL, "config.json")
        return isinstance(cached, str)
    return True


@pytest.mark.skipif(not _model_available(), reason="kp-deid-mdeberta-280m weights not available offline")
def test_extract_pii_reproduces_leaderboard_adapter():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from europriv_bench.adapters import KpModelAdapter
    from europriv_bench.spans import Span as EBSpan
    from europriv_bench.spans import char_spans_to_bioes

    text = "Ion Popescu lives in Bucharest and his email is ion@example.com"
    spans = extract_pii(text)

    # Typed spans whose char offsets index back to the surface text.
    assert spans, "expected at least one PII span"
    for s in spans:
        assert isinstance(s, Span)
        assert text[s.start : s.end] == s.text
        assert 0.0 <= s.score <= 1.0

    # Same model + same decoding path => SDK spans reproduce the leaderboard BIOES predictions.
    adapter_tags = KpModelAdapter(model_id=DEFAULT_MODEL).predict_tags([text])[0]
    sdk_tags = char_spans_to_bioes(text, [EBSpan(s.start, s.end, s.label) for s in spans])
    assert sdk_tags == adapter_tags
