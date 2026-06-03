"""KLU-109 — kp-anon span-replacement anonymizer: pseudonymization policy + adapter.

Offline (no model backend, no HF): the detector is stubbed so these test ONLY the substitution
policy and the anonymize/pseudonymize seam against the real EuroPriv-Bench Track-C metrics. The
load-bearing properties:

  * a DETECTED span is replaced by a type-consistent surrogate (no mask glyph) → low structural
    disruption + a measurable bijection,
  * a MISSED span survives verbatim and is counted as a leak (privacy attributable to detection
    recall, exactly like the redaction baseline),
  * the surrogate NEVER re-discloses a re-identifying fragment of the value it replaces.
"""

from __future__ import annotations

from europriv_bench.metrics import (
    information_retention,
    pseudonymization_consistency,
    redaction_leakage,
    structural_disruption,
)
from europriv_bench.national_id import check_digit
from klusai.privacy.models.anon import KpAnonAdapter, Pseudonymizer, _surrogate_cnp

CNP = "185071540001" + str(check_digit("185071540001"))
CNP2 = "605031120007" + str(check_digit("605031120007"))


class _StubAnon(KpAnonAdapter):
    """kp-anon with a stubbed detector: tags 'Ion Popescu' as PERSON and any exact CNP as NATIONAL_ID.

    Bypasses __init__'s model load — we only exercise the policy + the anonymize/pseudonymize seam.
    """

    def __init__(self, detect_cnp: bool = True):
        self._pseudo = Pseudonymizer()
        self.model_id = "stub"
        self._detect_cnp = detect_cnp

    def predict_tags(self, texts):
        out = []
        for t in texts:
            toks = t.split()
            tags = ["O"] * len(toks)
            for i, tok in enumerate(toks):
                if tok == "Ion":
                    tags[i] = "B-PERSON"
                elif tok == "Popescu":
                    tags[i] = "E-PERSON"
                elif self._detect_cnp and tok in (CNP, CNP2):
                    tags[i] = "S-NATIONAL_ID"
            out.append(tags)
        return out


def _ro_row(text, spans):
    return {"text": text, "country": "RO", "spans": spans}


# --------------------------------------------------------------------------- #
# Pseudonymizer policy
# --------------------------------------------------------------------------- #
def test_surrogate_is_stable_for_same_value():
    p = Pseudonymizer()
    assert p.surrogate("PERSON", "Ion Popescu") == p.surrogate("PERSON", "Ion  Popescu")  # ws-normalized
    assert p.surrogate("NATIONAL_ID", CNP) == p.surrogate("NATIONAL_ID", CNP)


def test_surrogate_differs_for_distinct_values():
    p = Pseudonymizer()
    assert p.surrogate("PERSON", "Ion Popescu") != p.surrogate("PERSON", "Maria Stan")
    assert p.surrogate("NATIONAL_ID", CNP) != p.surrogate("NATIONAL_ID", CNP2)


def test_surrogate_cnp_is_valid_format_and_not_the_source():
    sur = _surrogate_cnp(CNP)
    assert sur != CNP
    assert len(sur) == 13 and sur.isdigit()
    assert sur[-1] == str(check_digit(sur[:12]))  # checksum-valid shape → document still parses


def test_surrogate_shares_no_reidentifying_fragment_with_source():
    # The surrogate must not contain a 4-char run of the source value (else it would re-leak).
    p = Pseudonymizer()
    for label, value in [("NATIONAL_ID", CNP), ("PERSON", "Ion Popescu"), ("EMAIL", "ion@spital.ro")]:
        sur = "".join(p.surrogate(label, value).split())
        v = "".join(value.split())
        assert not any(v[i:i + 4] in sur for i in range(len(v) - 3)), (label, value, sur)


def test_surrogate_is_never_a_mask_glyph():
    p = Pseudonymizer()
    for label in ("PERSON", "NATIONAL_ID", "EMAIL", "PHONE", "ADDRESS", "DATE", "ACCOUNT_ID", "ORG_PARTY"):
        sur = p.surrogate(label, "some value 123")
        assert "█" not in sur and sur.strip() != ""


# --------------------------------------------------------------------------- #
# Adapter: anonymize / pseudonymize against the real metrics
# --------------------------------------------------------------------------- #
def test_detected_spans_become_surrogates_zero_leak_zero_masks():
    text = f"Pacientul Ion Popescu CNP {CNP} are febra"
    rows = [_ro_row(text, [
        {"start": text.index("Ion"), "end": text.index("Ion") + len("Ion Popescu"), "label": "PERSON"},
        {"start": text.index(CNP), "end": text.index(CNP) + len(CNP), "label": "NATIONAL_ID"},
    ])]
    a = _StubAnon()
    red = a.anonymize([text])
    # Detected gold values masked-by-substitution → no leak; no mask glyphs → utility preserved.
    assert redaction_leakage(rows, red)["leak_rate"] == 0.0
    assert structural_disruption(rows, red)["mask_token_ratio"] == 0.0
    assert information_retention(rows, red)["information_retention"] == 1.0
    assert "█" not in red[0]


def test_missed_span_survives_and_leaks():
    # Detector misses the CNP entirely → it survives verbatim → leak (recall-attributable privacy).
    text = f"CNP {CNP} aici"
    rows = [_ro_row(text, [{"start": 4, "end": 4 + len(CNP), "label": "NATIONAL_ID"}])]
    a = _StubAnon(detect_cnp=False)
    red = a.anonymize([text])
    assert redaction_leakage(rows, red)["leak_rate"] == 1.0


def test_bijection_is_perfect_in_and_cross_doc():
    a = _StubAnon()
    maps = a.pseudonymize(["Ion Popescu vine", "Ion Popescu pleaca"])
    res = pseudonymization_consistency([{}, {}], maps)
    assert res["in_doc_bijection_rate"] == 1.0
    assert res["cross_doc_bijection_rate"] == 1.0
    assert maps[0]["Ion Popescu"] == maps[1]["Ion Popescu"]  # cross-doc stable surrogate


def test_anonymize_and_pseudonymize_agree_on_surrogate():
    text = "Ion Popescu vine"
    a = _StubAnon()
    red = a.anonymize([text])[0]
    surrogate = a.pseudonymize([text])[0]["Ion Popescu"]
    assert surrogate in red  # the text substitution uses the SAME surrogate the map reports


def test_non_pii_context_is_preserved_verbatim():
    text = "Pacientul Ion Popescu are febra mare"
    red = _StubAnon().anonymize([text])[0]
    for tok in ("Pacientul", "are", "febra", "mare"):
        assert tok in red
