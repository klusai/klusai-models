"""Leakage gates and uncertainty checks for the supervised TAB experiment."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("prepare_tab_res104", SCRIPTS / "prepare_tab_res104.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


@pytest.mark.parametrize(
    "test_row",
    [
        {"doc_id": "train-id", "text": "different judgment"},
        {"doc_id": "test-id", "text": "same  judgment\ntext"},
    ],
)
def test_rejects_document_or_normalized_text_contamination(test_row):
    with pytest.raises(ValueError, match="Contamination"):
        prep.audit_splits({"train": [{"doc_id": "train-id", "text": "same judgment text"}], "test": [test_row]})


def test_bootstrap_counts_match_harness_and_pair_identical_models():
    pytest.importorskip("numpy")
    pytest.importorskip("datasets")
    pytest.importorskip("seqeval")
    sys.path.insert(0, str(SCRIPTS))
    try:
        from score_reid_local import score_predictions
        from score_tab_res104 import bootstrap_samples, ci
    finally:
        sys.path.remove(str(SCRIPTS))
    rows = [
        {
            "text": "Alice here",
            "spans": [{"start": 0, "end": 5, "label": "PERSON", "identifier_type": "DIRECT", "entity_id": "1"}],
        }
        for _ in range(3)
    ]
    gold = [["S-PERSON", "O"] for _ in rows]
    perfect = score_predictions(rows, gold)
    assert perfect["entity_f1"]["f1"] == 1
    assert perfect["tab_reid_leakage"]["direct_leak_rate"] == 0
    f1, leak = bootstrap_samples(gold, gold, rows)
    assert ci(f1) == [1, 1]
    assert ci(leak) == [0, 0]
    assert ci(f1 - f1) == [0, 0]
    with pytest.raises(ValueError, match="align"):
        score_predictions(rows, gold[:-1])


def test_training_windows_retain_identifiers_beyond_first_window():
    pytest.importorskip("datasets")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    sys.path.insert(0, str(SCRIPTS))
    try:
        from train_tab_res104 import encode_windows
    finally:
        sys.path.remove(str(SCRIPTS))
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "word": 1, "Alice": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    text = "word " * 600 + "Alice"
    rows = [{"text": text, "spans": [{"start": 3000, "end": 3005, "label": "PERSON"}]}]
    ds = encode_windows(rows, tokenizer, {"O": 0, "S-PERSON": 1})
    assert len(ds) == 3
    assert sum(len(r["input_ids"]) for r in ds) == 601
    assert sum(label == 1 for r in ds for label in r["labels"]) == 1
    assert ds[-1]["labels"][-1] == 1
