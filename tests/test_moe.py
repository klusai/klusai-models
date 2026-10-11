"""MoE label-boundary, data validation, and CLI isolation regressions (no model downloads)."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from klusai.privacy.models.training.moe import align_spans, load_rows

LABELS = {label: i for i, label in enumerate(["O", "B-PERSON", "I-PERSON", "E-PERSON", "S-PERSON"])}


def test_adjacent_same_type_entities_and_specials():
    spans = [{"start": 0, "end": 4, "label": "PERSON"}, {"start": 5, "end": 9, "label": "PERSON"}]
    assert align_spans(spans, [(0, 0), (0, 2), (2, 4), (5, 9), (10, 11), (0, 0)], LABELS) == [-100, 1, 3, 4, 0, -100]


def test_chunk_boundaries_keep_supervised_tokens():
    spans = [{"start": 0, "end": 10, "label": "PERSON"}]
    assert align_spans(spans, [(4, 6), (6, 8), (8, 10)], LABELS) == [1, 2, 3]


@pytest.mark.parametrize(
    "spans",
    [
        [{"start": 0, "end": 2, "label": "UNRECOGNIZED"}],
        [{"start": 0, "end": 20, "label": "PERSON"}],
        [{"start": 0, "end": 2, "label": "PERSON"}, {"start": 1, "end": 3, "label": "PERSON"}],
    ],
)
def test_invalid_gold_rejected(tmp_path, spans):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"text": "Alice", "spans": spans}))
    with pytest.raises(ValueError):
        load_rows(path)


def test_cli_dispatch_is_local_and_separate():
    spec = importlib.util.spec_from_file_location("train_cli", Path(__file__).parents[1] / "scripts/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = [
        "moe-finetune",
        "--base",
        "openai/privacy-filter",
        "--dataset",
        "train.jsonl",
        "--out",
        "klusai/local",
        "--backend",
        "cuda",
        "--device",
        "cpu",
        "--max-steps",
        "2",
    ]
    with patch(
        "klusai.privacy.models.training.moe.train_moe", return_value={"checkpoint": "local", "completed_steps": 2}
    ) as run:
        result = CliRunner().invoke(module.cli, args)
        assert result.exit_code == 0, result.output
        assert run.call_args.kwargs["max_steps"] == 2
        assert run.call_args.kwargs["device"] == "cpu"
        assert CliRunner().invoke(module.cli, args + ["--push"]).exit_code != 0
        assert run.call_count == 1
