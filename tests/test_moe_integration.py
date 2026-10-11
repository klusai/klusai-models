"""Real Transformers 5 / PEFT resume and adapter round-trip, tiny local architecture, no download."""

import json

import pytest

transformers = pytest.importorskip("transformers")
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip("MoE integration runs in .venv-moe; encoders retain Transformers 4.x", allow_module_level=True)
torch = pytest.importorskip("torch")
pytest.importorskip("peft")


def test_checkpoint_resume_and_harness_adapter(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        AutoModelForTokenClassification,
        OpenAIPrivacyFilterConfig,
        PreTrainedTokenizerFast,
    )

    from europriv_bench.adapters import KpModelAdapter
    from europriv_bench.taxonomy import bioes_labels
    from klusai.privacy.models.training.moe import train_moe

    base = tmp_path / "base"
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "Alice": 2, "met": 3, "Bob": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]").save_pretrained(base)
    config = OpenAIPrivacyFilterConfig(
        vocab_size=5,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        pad_token_id=0,
    )
    AutoModelForTokenClassification.from_config(config).save_pretrained(base)
    data = tmp_path / "train.jsonl"
    row = {"text": "Alice met Bob", "spans": [{"start": 0, "end": 5, "label": "PERSON"}]}
    data.write_text("\n".join(json.dumps(row) for _ in range(8)))
    output = tmp_path / "run"
    kwargs = dict(
        base_model=str(base),
        dataset=str(data),
        output_dir=str(output),
        batch_size=2,
        max_steps=4,
        save_steps=2,
        max_length=16,
        epochs=3,
        device="cpu",
    )
    result = train_moe(**kwargs)
    assert result["completed_steps"] == 4
    assert (output / "checkpoint-2/optimizer.pt").exists()
    assert (output / "checkpoint-4/rng_state.pth").exists()
    trained = AutoModelForTokenClassification.from_pretrained(output / "merged")
    state = {key: value.clone() for key, value in trained.state_dict().items()}
    assert list(trained.config.id2label.values()) == bioes_labels()
    assert any(".router." in key for key in result["trainable_parameters"])
    assert not any(".experts." in key for key in result["trainable_parameters"])
    train_moe(**kwargs, resume=str(output / "checkpoint-2"))
    resumed = AutoModelForTokenClassification.from_pretrained(output / "merged")
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    adapter = KpModelAdapter(str(output / "merged"))
    assert len(adapter.predict_tags(["Alice met Bob"])[0]) == 3
    with pytest.raises(ValueError, match="Resume configuration changed: seed"):
        train_moe(**{**kwargs, "seed": 1}, resume="latest")
