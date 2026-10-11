"""Guard against invented full-character coverage, zero-floor wins and unpaired CIs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_res125 import paired, privacy_verdict, utility_verdict  # noqa: E402
from scorecard_res125 import literal_mask, utility_counts  # noqa: E402

from europriv_bench.metrics import residual_leakage_v2, tab_reid_leakage  # noqa: E402


def test_partial_native_name_removal_does_not_inherit_legacy_zero():
    row = {"text": "John Smith called", "spans": [
        {"start": 0, "end": 10, "label": "PERSON", "entity_id": "p1", "identifier_type": "DIRECT"},
    ]}
    assert literal_mask(row["text"], [(0, 4)]) == "████ Smith called"
    assert tab_reid_leakage([row], [["S-PERSON", "O", "O"]])["direct_leak_rate"] == 0
    assert residual_leakage_v2([row], [[(0, 4)]])["direct_leak_rate"] == 1
    assert residual_leakage_v2([row], [[(0, 4), (5, 10)]])["direct_leak_rate"] == 0


def test_utility_unions_overlap_and_excludes_unicode_whitespace():
    row = {"text": "Éva\u2003called!", "spans": [
        {"start": 0, "end": 3, "label": "PERSON"}, {"start": 1, "end": 3, "label": "PERSON"},
    ]}
    assert utility_counts(row, [(0, 6), (4, 7)]) == [3, 7]
    assert utility_counts({"text": "Éva", "spans": [{"start": 0, "end": 3}]}, [(0, 3)]) == [0, 0]
    with pytest.raises(ValueError, match="Invalid"):
        literal_mask("a", [(0, 2)])


def test_win_requires_disjoint_wilson_and_paired_ci_and_available_support():
    a = {"all_available": 1, "all_leak_rate_ci_low": 0.0, "all_leak_rate_ci_high": 0.1}
    b = {"all_available": 1, "all_leak_rate_ci_low": 0.3, "all_leak_rate_ci_high": 0.5}
    assert privacy_verdict(a, b, [-0.4, -0.2]) == "win"
    assert privacy_verdict(a, b, [-0.4, 0.01]) == "tie"
    assert privacy_verdict(a, a, [0, 0]) == "tie"
    assert privacy_verdict({**a, "all_available": 0}, b, [-0.4, -0.2]) == "unverified"
    assert utility_verdict([-0.1, 0.0]) == "win"
    assert utility_verdict([-0.1, 0.001]) == "unverified"
    assert utility_verdict([0.001, 0.1]) == "loss"


def test_pairing_is_by_document_and_rejects_mismatched_gold_or_denominators():
    a = {"gold_sha256": "same", "n_documents": 2, "counts": {"v2_all": [[0, 10], [1, 1]]}}
    assert paired(a, a)["v2_all"]["ci95"] == [0, 0]
    b = {**a, "counts": {"v2_all": [[10, 10], [0, 1]]}}
    # Two whole-document clusters: the bootstrap can draw either document twice.
    # Independent-subject sampling would incorrectly yield a much narrower interval.
    assert paired(a, b)["v2_all"]["ci95"] == [-1, 1]
    with pytest.raises(ValueError, match="ordered gold"):
        paired(a, {**b, "gold_sha256": "different"})
    with pytest.raises(ValueError, match="denominators"):
        paired(a, {**b, "counts": {"v2_all": [[10, 11], [0, 1]]}})
