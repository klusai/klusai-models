"""Gateway token collisions must never remove conversations from privacy scoring."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
if importlib.util.find_spec("europriv_bench.gateway") is None:
    pytest.skip("RES-126 gateway harness required", allow_module_level=True)

from score_res127_metrics import score_one  # noqa: E402
from scorecard_res125 import digest, literal_mask, write  # noqa: E402


def test_collision_keeps_full_privacy_denominator(tmp_path):
    rows = [
        {"id": "csv", "text": "Ana;123", "spans": [
            {"start": 0, "end": 3, "label": "PERSON", "entity_id": "p", "identifier_type": "DIRECT"},
            {"start": 4, "end": 7, "label": "NATIONAL_ID", "entity_id": "id", "identifier_type": "DIRECT"},
        ]},
        {"id": "prose", "text": "Ana", "spans": [
            {"start": 0, "end": 3, "label": "PERSON", "entity_id": "p", "identifier_type": "DIRECT"},
        ]},
    ]
    path = tmp_path / "gold/gateway-ro.json"
    write(path, {"rows": rows, "language": "ro", "dataset": {}})
    receipts = [{"intervals": [[0, 3]], "tags": ["S-PERSON"],
                 "output": literal_mask(row["text"], [[0, 3]])} for row in rows]
    write(tmp_path / "predictions/candidate--gateway-ro.json",
          {"identity": {"kind": "kp", "gold_sha256": digest(path)}, "receipts": receipts})
    score_one(tmp_path, "candidate", "gateway-ro")
    result = json.loads((tmp_path / "results/candidate--gateway-ro.json").read_text())
    assert result["n_documents"] == 2
    assert result["f1_scope"]["scored_documents"] == 1
    assert result["f1_scope"]["excluded_documents"] == 1
    assert result["f1_scope"]["full_split_available"] is False
    assert result["residual_leakage_v2"]["all_subjects_total"] == 3
    assert result["residual_leakage_v2"]["all_subjects_leaked"] == 1
    assert result["entity_f1"]["f1"] == 1
    assert len(result["counts"]["over_redaction"]) == 2
