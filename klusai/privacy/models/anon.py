"""kp-anon — the span-replacement anonymizer's pseudonymization policy + adapter (KLU-109).

kp-anon is a *span-replacement* anonymizer (the KLU-109 pre-committed minimum-viable config): a
trained PII **detector** (the LoRA mDeBERTa-280m token classifier from
``scripts/train_kp_anon_klu109.py``) whose detected spans are **pseudonymized** — replaced with
deterministic, type-consistent *surrogates* — rather than blanket-masked. This is the difference the
Track-C privacy-utility frontier (KLU-104 metrics) is designed to surface:

  * The **redaction baseline** (``europriv_bench.adapters.BaseAdapter.anonymize``) masks every
    detected span with the single placeholder glyph ``█``. At equal detection recall it leaks no
    more PII, but it shreds the document: a large ``structural_disruption.mask_token_ratio`` and a
    document a downstream reader/model can no longer use.
  * **kp-anon** swaps each detected span for a realistic same-type surrogate (a fake but plausible
    name / date / national-ID / e-mail / …), reused consistently for the same source value. At the
    SAME detection recall it has the SAME re-identification leak (``redaction_leakage`` is read from
    the gold offsets, and a *missed* span survives verbatim either way), but a near-zero mask-token
    ratio — it stays on the utility-preserving side of the frontier.

CLAIM-LANGUAGE / HONESTY (KLU-109 guards): config_status=dev; NO "SOTA"/"best"/"validated" claim
(validation is gated on KLU-27). The substitution policy is engineered so a surrogate **cannot**
reintroduce a re-identifying fragment of its source value: each surrogate is generated independently
of the source's surface digits/letters AND is explicitly *rejection-checked* against the leak metric's
fragment rule (``_shares_fragment`` mirrors ``europriv_bench.metrics._value_survives`` with the same
``_MIN_LEAK_FRAGMENT``), redrawing until it shares no such run with the source. So the substitution
itself contributes zero leak — only a *detector miss* leaks, exactly as for the redaction baseline.
This keeps the privacy axis attributable to detection recall, not to the anonymizer "cheating".

NB: an earlier draft generated surrogates from an independent hash but did NOT reject coincidental
fragment collisions; on the ro/pl real-skeleton eval that surfaced a spurious ~3-4% leak (random
13-digit CNP-shaped surrogates occasionally sharing a 4-digit run with the gold CNP). The rejection
check above closes that gap so kp-anon's leak matches the redaction baseline's at equal recall.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from europriv_bench.adapters import KpModelAdapter, _bioes_to_kp_spans

# Small, license-clean surrogate pools (synthetic — no real individuals). Kept deliberately generic
# and multi-locale so RO/PL/EN documents read plausibly after substitution. These are NOT meant to be
# locale-perfect; they are realistic, structure-preserving fillers (the utility proxy measures
# structural disruption, not native fluency — KLU-104's cross-lingual caveat).
_FIRST_NAMES = (
    "Andrei", "Maria", "Ioan", "Elena", "Petru", "Ana", "Mihai", "Sofia", "Pavel", "Irina",
    "Jan", "Anna", "Piotr", "Zofia", "Marek", "Ewa", "Tomasz", "Katarzyna", "James", "Laura",
    "David", "Sarah", "Michael", "Emma", "Robert", "Olivia", "Daniel", "Hannah", "Lukas", "Nora",
)
_LAST_NAMES = (
    "Popescu", "Ionescu", "Dumitru", "Stan", "Munteanu", "Georgescu", "Radu", "Marin",
    "Kowalski", "Nowak", "Wisniewski", "Wojcik", "Kaminski", "Lewandowski",
    "Smith", "Johnson", "Brown", "Taylor", "Wilson", "Davies", "Clark", "Walker",
)
_STREETS = (
    "Strada Florilor", "Aleea Teilor", "Bulevardul Unirii", "ulica Lipowa", "ulica Polna",
    "Maple Street", "Oak Avenue", "Elm Road", "Park Lane", "High Street",
)
_CITIES = (
    "Cluj", "Iasi", "Timisoara", "Brasov", "Warszawa", "Krakow", "Gdansk",
    "London", "Manchester", "Bristol", "Leeds",
)
_DOMAINS = ("example.org", "example.net", "mail.example.com", "post.example.eu")
_ORGS = (
    "Acme Holding SRL", "Northwind Sp. z o.o.", "Globex Ltd", "Initech Group",
    "Meridian Servicii SA", "Vertex Partners",
)


# Minimum re-identifying fragment length, kept in lock-step with the leak metric's
# ``europriv_bench.metrics._MIN_LEAK_FRAGMENT``: a surrogate that shares a run this long with its
# source value would be flagged a leak by ``redaction_leakage``. The substitution policy below
# rejects any such surrogate so the privacy axis stays attributable to detection recall only.
_MIN_LEAK_FRAGMENT = 4


def _h(value: str, salt: str, bump: int = 0) -> int:
    """Deterministic non-negative int from a (value, salt, bump) triple — stable across runs/processes.

    Used to pick a surrogate from a fixed pool so the SAME source value always maps to the SAME
    surrogate (the bijection the ``pseudonymization_consistency`` metric measures), with no shared
    mutable state. ``salt`` separates the streams used for different surrogate fields; ``bump`` lets
    the caller deterministically draw an alternative when a candidate must be rejected (collision).
    """
    key = f"{salt}:{bump}:{value}" if bump else f"{salt}:{value}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def _pick(pool: Sequence[str], value: str, salt: str, bump: int = 0) -> str:
    return pool[_h(value, salt, bump) % len(pool)]


def _surrogate_cnp(value: str, bump: int = 0) -> str:
    """A FRESH, checksum-valid-format 13-digit Romanian-CNP-shaped surrogate, derived from a hash.

    Deliberately generated from a hash of the source value, NOT from its digits, so it shares no
    re-identifying fragment with the real CNP (a real CNP encodes DOB+sex+county; this surrogate's
    digits are pseudo-random, disclosing nothing about the subject). Reuses the harness's check-digit
    so the surrogate is *shaped* like a CNP (utility: the document still parses) without being the
    real one. NB: this surrogate is by construction NOT equal to the source, so substituting it can
    never re-disclose the gold value. ``bump`` redraws when a candidate shares a fragment-run with
    the source (see ``Pseudonymizer.surrogate``).
    """
    from europriv_bench.national_id import check_digit

    h = _h(value, "cnp", bump)
    first12 = f"{h % 10**12:012d}"
    # Avoid the degenerate all-equal-to-source case (astronomically unlikely, but be safe).
    if first12 == value[:12]:
        first12 = f"{(h + 1) % 10**12:012d}"
    return first12 + str(check_digit(first12))


def _digits_like(value: str, salt: str, bump: int = 0) -> str:
    """A hashed digit string the same length as ``value``'s digits (for generic numeric IDs)."""
    digits = [c for c in value if c.isdigit()] or ["0"] * 6
    h = _h(value, salt, bump)
    rep = f"{h:0{len(digits)}d}"[: len(digits)]
    return rep


def _shares_fragment(surrogate: str, value: str, min_fragment: int = _MIN_LEAK_FRAGMENT) -> bool:
    """True iff ``surrogate`` and source ``value`` share a contiguous non-space run >= ``min_fragment``.

    Mirrors ``europriv_bench.metrics._value_survives``: whitespace is stripped from both, then every
    length-``min_fragment`` window of the source value is checked against the surrogate. A surrogate
    that shares such a run would be (correctly) counted as a re-identification leak by the metric — so
    the policy rejects it. For values shorter than ``min_fragment`` the whole value must collide.
    """
    v = "".join(value.split())
    s = "".join(surrogate.split())
    if not v or not s:
        return False
    k = min(min_fragment, len(v))
    return any(v[i : i + k] in s for i in range(0, len(v) - k + 1))


class Pseudonymizer:
    """Deterministic, type-consistent surrogate generator (the kp-anon substitution policy).

    Maps a ``(KP label, normalized source value)`` to a stable surrogate string, reused for every
    occurrence of that value (so the bijection rate is 1.0 by construction). Surrogates are realistic
    same-type fillers — a fake name for PERSON, a fresh CNP-shaped number for NATIONAL_ID, etc. — and
    are generated independently of the source's surface form, so a surrogate never re-discloses a
    fragment of the value it replaces. The policy is pure/stateless given the (label, value) inputs:
    the same corpus yields the same surrogates regardless of document order.
    """

    # Bound the rejection-resampling: numeric surrogates with the SAME digit-length as a long source
    # value can, in rare cases, be hard to make collision-free; after this many redraws we fall back
    # to a typed synthetic token that carries no digits of the source (guaranteed leak-free).
    _MAX_BUMPS = 64

    def _raw_surrogate(self, label: str, v: str, bump: int) -> str:
        """One deterministic candidate surrogate for ``(label, v)`` at draw index ``bump``."""
        if label == "PERSON":
            return f"{_pick(_FIRST_NAMES, v, 'fn', bump)} {_pick(_LAST_NAMES, v, 'ln', bump)}"
        if label == "NATIONAL_ID":
            return _surrogate_cnp(v, bump)
        if label == "EMAIL":
            return f"user{_h(v, 'email', bump) % 10000:04d}@{_pick(_DOMAINS, v, 'dom', bump)}"
        if label == "PHONE":
            return (f"+40 7{_digits_like(v, 'phone', bump)[:2]} "
                    f"{_h(v, 'ph2', bump) % 1000:03d} {_h(v, 'ph3', bump) % 1000:03d}")
        if label == "ADDRESS":
            return f"{_pick(_STREETS, v, 'st', bump)} {_h(v, 'no', bump) % 200 + 1}, {_pick(_CITIES, v, 'city', bump)}"
        if label == "DATE":
            # A plausible date that is NOT the source (day/month/year from independent hash streams).
            d = _h(v, "day", bump) % 28 + 1
            m = _h(v, "mon", bump) % 12 + 1
            y = 1950 + _h(v, "yr", bump) % 70
            return f"{d:02d}.{m:02d}.{y:04d}"
        if label in ("ORG_PARTY", "PROVIDER", "FACILITY", "COURT"):
            return _pick(_ORGS, v, "org", bump)
        if label in ("ACCOUNT_ID", "MRN", "CASE_NUMBER", "COMPANY_ID", "SECRET", "URL", "STATUTE_REF",
                     "HEALTH_CONDITION"):
            # Structured/opaque identifiers → a typed, clearly-synthetic token carrying the digits'
            # length but none of their value. Keeps the slot filled (utility) without disclosure.
            return f"{label}-{_digits_like(v, label, bump)}"
        # Any unforeseen type: a typed synthetic token (never the source, never a mask glyph).
        return f"{label}-{_h(v, label, bump) % 10**6:06d}"

    def surrogate(self, label: str, value: str, avoid: Sequence[str] = ()) -> str:
        """A deterministic, type-consistent surrogate that shares NO re-identifying fragment with ``value``.

        Draws candidates deterministically (incrementing ``bump``) and returns the first that does not
        share a ``_MIN_LEAK_FRAGMENT``-length run with the source ``value`` *nor* with any value in
        ``avoid`` — so the substitution itself contributes zero ``redaction_leakage`` (the load-bearing
        invariant the frontier comparison rests on), including the *cross-field* case where a surrogate
        for one field would otherwise echo a fragment of a DIFFERENT subject value in the same document.
        If no clean same-type candidate is found within ``_MAX_BUMPS`` draws (rare, only for long numeric
        IDs against many constraints), falls back to a typed synthetic token, then an opaque token,
        each still rejection-checked — guaranteed leak-free against ``value`` (best-effort against ``avoid``).
        """
        v = " ".join(value.split())
        targets = (v, *avoid)

        def clean(cand: str) -> bool:
            return not any(_shares_fragment(cand, t) for t in targets)

        for bump in range(self._MAX_BUMPS):
            cand = self._raw_surrogate(label, v, bump)
            if clean(cand):
                return cand
        # Fallback: a typed token whose digits come from a draw proven not to collide. Search a fresh
        # bump range on the generic numeric form (length-independent of the source) so we always succeed.
        for bump in range(self._MAX_BUMPS, self._MAX_BUMPS * 4):
            cand = f"{label}-{_h(v, label, bump) % 10**6:06d}"
            if clean(cand):
                return cand
        # Best-effort last resorts: keep the source-value invariant even if an ``avoid`` clash is unavoidable.
        for bump in range(self._MAX_BUMPS):
            cand = f"{label}-X{_h(v, label, bump) % 10**4:04d}"
            if clean(cand) or not _shares_fragment(cand, v):
                return cand
        return f"{label}-X{_h(v, label) % 10**4:04d}"


class KpAnonAdapter(KpModelAdapter):
    """kp-anon: the trained kp-deid-style detector wrapped with the pseudonymization policy (KLU-109).

    Inherits ``predict_tags``/``predict_spans`` from :class:`KpModelAdapter` (so its detection recall
    — what drives the post-redaction leak — is identical to the same model used as a plain detector),
    and overrides ``anonymize``/``pseudonymize`` to SUBSTITUTE type-consistent surrogates instead of
    masking. ``anonymize`` and ``pseudonymize`` share the SAME surrogate map, so the redacted text and
    the reported bijection map are consistent.

    The surrogate for a given ``(label, value)`` is stable across the whole corpus (cross-doc
    consistent), giving a measurable bijection rate of 1.0 and joinable pseudonymized documents.
    """

    name = "kp-anon"

    def __init__(self, model_id: str = "klusai/kp-anon-mdeberta-280m") -> None:
        super().__init__(model_id=model_id)
        self._pseudo = Pseudonymizer()

    def _corpus_surrogates(self, texts: Sequence[str]):
        """Build the per-doc detected spans + a corpus-stable ``(label, norm value) -> surrogate`` map.

        One detector pass shared by ``anonymize`` and ``pseudonymize`` so they agree. The map is keyed
        by ``(label, whitespace-normalized value)`` so the same value gets one surrogate corpus-wide.

        Each surrogate is generated to share no re-identifying fragment with its own source value AND
        with the OTHER detected values co-occurring in any document where that value appears (the
        cross-field constraint): otherwise a surrogate for one field could coincidentally echo a 4-gram
        of a *different* subject's national ID in the same document, which ``redaction_leakage`` would
        (correctly) score as a leak. Collecting the co-occurring values first keeps the surrogate
        corpus-stable (one surrogate per value) while satisfying every document it lands in.
        """
        per_doc_spans: list[list[tuple[int, int, str]]] = []
        norm_values_per_doc: list[list[str]] = []
        for text, tags in zip(texts, self.predict_tags(list(texts))):
            spans = sorted(_bioes_to_kp_spans(text, tags))
            per_doc_spans.append(spans)
            norm_values_per_doc.append(["".join(text[s:e].split()) for s, e, _ in spans])

        # value -> set of OTHER normalized values it ever co-occurs with (cross-field avoid set).
        co_occur: dict[str, set[str]] = {}
        for vals in norm_values_per_doc:
            uniq = set(vals)
            for v in uniq:
                co_occur.setdefault(v, set()).update(uniq - {v})

        surrogate_map: dict[tuple[str, str], str] = {}
        for spans, vals in zip(per_doc_spans, norm_values_per_doc):
            for (s, e, label), nv in zip(spans, vals):
                key = (label, nv)
                if key not in surrogate_map:
                    avoid = sorted(co_occur.get(nv, ()))  # sorted → deterministic across runs
                    surrogate_map[key] = self._pseudo.surrogate(label, nv, avoid=avoid)
        return per_doc_spans, surrogate_map

    def anonymize(self, texts: Sequence[str]) -> list[str]:
        """Replace each detected span with its stable, type-consistent surrogate (NOT a mask glyph).

        A *missed* span (detector recall failure) survives verbatim and is counted as a leak by
        ``redaction_leakage`` — exactly as for the redaction baseline — so the privacy axis stays
        attributable to detection recall, while the utility axis (low mask-token ratio) reflects the
        surrogate-substitution policy.
        """
        texts = list(texts)
        per_doc_spans, surrogate_map = self._corpus_surrogates(texts)
        out: list[str] = []
        for text, spans in zip(texts, per_doc_spans):
            pieces: list[str] = []
            cursor = 0
            last_end = -1
            for s, e, label in spans:
                if s < cursor or s < last_end:  # overlap/contained → skip (left-to-right, non-overlap)
                    continue
                pieces.append(text[cursor:s])
                pieces.append(surrogate_map[(label, "".join(text[s:e].split()))])
                cursor = e
                last_end = e
            pieces.append(text[cursor:])
            out.append("".join(pieces))
        return out

    def pseudonymize(self, texts: Sequence[str]) -> list[dict[str, str]]:
        """Per-doc ``{source value -> surrogate}`` map, consistent with :meth:`anonymize`.

        Keyed by the surface value as it appears in the document (so the harness can resolve entities
        by normalized value); the surrogate is the SAME corpus-stable one ``anonymize`` substitutes,
        so the bijection rate the metric reports matches the text the leak metric scored.
        """
        texts = list(texts)
        per_doc_spans, surrogate_map = self._corpus_surrogates(texts)
        maps: list[dict[str, str]] = []
        for text, spans in zip(texts, per_doc_spans):
            m: dict[str, str] = {}
            for s, e, label in spans:
                value = text[s:e]
                m[value] = surrogate_map[(label, "".join(value.split()))]
            maps.append(m)
        return maps
