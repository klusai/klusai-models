"""Tests for klusai.privacy.sdk — the adoption-layer SDK (KLU-20).

  * the default-model wiring (no model download); and
  * an integration test that the live `extract_pii()` reproduces the EuroPriv-Bench
    `KpModelAdapter` predictions (skipped if the HF model/weights aren't available offline).

`extract_pii` delegates span reconstruction to `KpModelAdapter.predict_spans` (KLU-59), which
owns + tests the BIOES->char-span logic in europriv_bench — so the SDK no longer carries (or
unit-tests) its own copy.
"""

from __future__ import annotations

import os

import pytest

from klusai.privacy.sdk import DEFAULT_MODEL, Span, extract_pii


def test_default_model_is_the_shipped_h1_weight():
    # kp-deid-moe does not exist yet (H2 maybe); the default must resolve to a real H1 weight.
    assert DEFAULT_MODEL == "klusai/kp-deid-mdeberta-280m"


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
