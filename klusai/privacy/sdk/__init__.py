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
from europriv_bench.crosswalk import kp_entities_to_bioes
from europriv_bench.spans import whitespace_tokens

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


def _bioes_to_spans(text: str, tags: list[str]) -> list[tuple[int, int, str]]:
    """Reconstruct ``(start, end, label)`` char spans from BIOES tags over whitespace tokens.

    This walks the *exact* token grid (``whitespace_tokens``) the leaderboard BIOES sequence is
    defined on, so a span's char extent is the span the harness scores — start at the first
    token's char-start, end at the last token's char-end.
    """
    toks = whitespace_tokens(text)
    spans: list[tuple[int, int, str]] = []
    i = 0
    n = len(tags)
    while i < n:
        tag = tags[i]
        if tag == "O":
            i += 1
            continue
        kind, _, etype = tag.partition("-")
        if kind == "S":
            _, ts, te = toks[i]
            spans.append((ts, te, etype))
            i += 1
            continue
        if kind == "B":
            start_tok = i
            j = i
            # Consume I-* then the closing E-* of the same type (labels_to_bioes guarantees this
            # shape; be tolerant of a missing E by stopping at the run of same-type tokens).
            while j + 1 < n:
                nk, _, ne = tags[j + 1].partition("-")
                if nk in {"I", "E"} and ne == etype:
                    j += 1
                    if nk == "E":
                        break
                else:
                    break
            _, ts, _ = toks[start_tok]
            _, _, te = toks[j]
            spans.append((ts, te, etype))
            i = j + 1
            continue
        # Stray I-/E- without a B- (malformed): treat as a lone token to stay total.
        _, ts, te = toks[i]
        spans.append((ts, te, etype))
        i += 1
    return spans


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
    pipe = adapter._pipeline()
    # Raw pipeline entities (entity_group=KP type, char start/end, score) — identical to what the
    # adapter consumes. Aggregation is "simple", so we get subword-grouped pieces.
    raw = pipe(text)
    kp_ents = [
        {"start": int(e["start"]), "end": int(e["end"]), "label": e["entity_group"], "score": float(e["score"])}
        for e in raw
    ]
    # Re-derive the BIOES tag sequence exactly as the leaderboard does, then read spans back off the
    # same whitespace-token grid the harness scores on.
    tags = kp_entities_to_bioes(text, kp_ents)
    spans: list[Span] = []
    for start, end, label in _bioes_to_spans(text, tags):
        # Mean confidence over the pipeline pieces overlapping this span.
        pieces = [e["score"] for e in kp_ents if e["start"] < end and e["end"] > start and e["label"] == label]
        score = sum(pieces) / len(pieces) if pieces else 0.0
        spans.append(Span(start=start, end=end, label=label, text=text[start:end], score=score))
    return spans


def deidentify(text: str, method: str = "mask", model: str = DEFAULT_MODEL) -> str:
    """Return `text` with detected PII removed. method: mask | redact | pseudonymize."""
    raise NotImplementedError("deidentify: H2 — apply detection then the chosen redaction strategy (kp-anon)")


def pseudonymize(text: str, model: str = DEFAULT_MODEL) -> str:
    """Replace PII with consistent synthetic surrogates (reversible mapping kept out-of-band)."""
    raise NotImplementedError("pseudonymize: H2 — consistent surrogate generation + utility preservation (kp-anon)")
