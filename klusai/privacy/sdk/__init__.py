"""klusai.privacy.sdk — the adoption-layer SDK for KP privacy models.

Mirrors the ergonomics of Presidio / Private AI / Tonic (`extract_pii` / `deidentify` /
`pseudonymize`) so a privacy workflow is a function call, not a model-loading chore. Backed by
KP models on HF; entity labels are the harmonized KP taxonomy from `europriv_bench`.

`extract_pii()` is wired (H1). It reuses ``europriv_bench``'s ``KpModelAdapter`` — the *same*
model-loading + decoding path the EuroPriv-Bench leaderboard scores — so SDK spans reproduce the
leaderboard row's predictions for the default model. `deidentify()` / `pseudonymize()` follow in
H2 with the `kp-anon` models.
"""

from __future__ import annotations

from dataclasses import dataclass

# The taxonomy / BIOES label space and the leaderboard adapter are the single source of truth in
# europriv_bench — never copy them here.
from europriv_bench.adapters import KpModelAdapter

__version__ = "0.1.0"

# The H1 shipped weight: a microsoft/mdeberta-v3-base (280M) token classifier trained directly on
# the harmonized KP taxonomy. `klusai/kp-deid-moe` is an H2 *maybe* (does not exist yet), so the
# default must point at the model that actually ships — anything else is a load-time failure.
DEFAULT_MODEL = "klusai/kp-deid-mdeberta-280m"


@dataclass
class Span:
    """A detected PII/PHI span in the harmonized KP taxonomy.

    ``start``/``end`` are character offsets into the input text (``end`` exclusive), so
    ``text[span.start:span.end] == span.text``. ``label`` is a KP entity type (e.g. ``PERSON``,
    ``NATIONAL_ID``; see ``europriv_bench.taxonomy``). ``score`` is the model's mean confidence
    over the subword pieces backing the span.
    """

    start: int          # char offset, inclusive
    end: int            # char offset, exclusive
    label: str          # harmonized KP type, e.g. "PERSON"
    text: str           # surface text == input[start:end]
    score: float        # mean model confidence over the span's pieces


# Adapter instances are cached per model id so repeated calls reuse one loaded pipeline.
_ADAPTERS: dict[str, KpModelAdapter] = {}


def _adapter(model: str) -> KpModelAdapter:
    if model not in _ADAPTERS:
        _ADAPTERS[model] = KpModelAdapter(model_id=model)
    return _ADAPTERS[model]


def extract_pii(text: str, model: str = DEFAULT_MODEL) -> list[Span]:
    """Detect PII/PHI spans in ``text``, typed in the harmonized KP taxonomy.

    Returns a list of :class:`Span` (entity type, char start/end, surface text, score). Uses the
    same model load + token-classification decoding path as the EuroPriv-Bench ``KpModelAdapter``,
    so for the default model the spans reproduce the leaderboard row's predictions.

    >>> spans = extract_pii("Ion Popescu lives in Bucharest.")
    >>> spans[0].label, text[spans[0].start:spans[0].end]  # doctest: +SKIP
    ('PERSON', 'Ion Popescu')
    """
    adapter = _adapter(model)
    # Use the adapter's PUBLIC predict_spans accessor — the same model-load + decoding path the
    # leaderboard scores, returning char-offset spans + score. (Previously this reached into the
    # private adapter._pipeline() and re-derived the spans here; predict_spans now encapsulates that
    # exact reconstruction in europriv_bench, so an adapter-internals refactor can't break us.)
    return [
        Span(start=p.start, end=p.end, label=p.label, text=p.text, score=p.score)
        for p in adapter.predict_spans(text)
    ]


def deidentify(text: str, method: str = "mask", model: str = DEFAULT_MODEL) -> str:
    """Return `text` with detected PII removed. method: mask | redact | pseudonymize."""
    raise NotImplementedError("deidentify: H2 — apply detection then the chosen redaction strategy (kp-anon)")


def pseudonymize(text: str, model: str = DEFAULT_MODEL) -> str:
    """Replace PII with consistent synthetic surrogates (reversible mapping kept out-of-band)."""
    raise NotImplementedError("pseudonymize: H2 — consistent surrogate generation + utility preservation (kp-anon)")
