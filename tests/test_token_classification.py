"""Unit tests for the token-classification training backend's span alignment.

The model-training itself (Trainer loop, HF/peft) is exercised by the smoke run, not in CI; here
we pin the *correctness-critical* pure function: projecting gold KP char spans onto subword
tokens as BIOES label ids in the benchmark's label space. A bug here silently corrupts training
labels, so it gets a deterministic, dependency-free test (no model download)."""

from __future__ import annotations

from europriv_bench.taxonomy import bioes_labels
from klusai.privacy.models.training.token_classification import (
    _bioes_from_spans,
    resolve_device,
    resolve_max_util_profile,
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
