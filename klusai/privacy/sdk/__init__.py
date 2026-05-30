"""klusai.privacy.sdk — the adoption-layer SDK for KP privacy models.

Mirrors OpenMed's ergonomics (`extract_pii` / `deidentify` / `pseudonymize`) so a privacy
workflow is three function calls, not a model-loading chore. Backed by KP models on HF; entity
labels are the harmonized KP taxonomy from `europriv_bench`. Implementations land in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

__version__ = "0.1.0"

DEFAULT_MODEL = "klusai/kp-deid-moe"


@dataclass
class Entity:
    start: int
    end: int
    label: str       # harmonized KP type, e.g. "PERSON" (see europriv_bench.taxonomy)
    text: str
    score: float


def extract_pii(text: str, model_name: str = DEFAULT_MODEL) -> list[Entity]:
    """Detect PII/PHI spans in `text`, returned in the harmonized KP taxonomy."""
    raise NotImplementedError("extract_pii: Phase 3 — load KP model, run token classification, map to KP labels")


def deidentify(text: str, method: str = "mask", model_name: str = DEFAULT_MODEL) -> str:
    """Return `text` with detected PII removed. method: mask | redact | pseudonymize."""
    raise NotImplementedError("deidentify: Phase 3/4 — apply detection then the chosen redaction strategy")


def pseudonymize(text: str, model_name: str = DEFAULT_MODEL) -> str:
    """Replace PII with consistent synthetic surrogates (reversible mapping kept out-of-band)."""
    raise NotImplementedError("pseudonymize: Phase 4 — consistent surrogate generation + utility preservation")
