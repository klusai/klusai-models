"""Tests for the KLU-46 MLX DeBERTa-v2 encoder SPIKE (proof-of-concept).

`mlx` is an optional extra; every test skips gracefully when it (or the network/model
cache) is absent, so the suite stays green on the MPS-only / CUDA / CI environments that
never install the `mlx` extra.
"""

from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx.core", reason="mlx extra not installed")


def _small_config():
    from klusai.privacy.models.mlx_encoder import DebertaV2Config

    # tiny but architecturally faithful (share_att_key + p2c/c2p like mdeberta-v3)
    return DebertaV2Config(
        vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=64, position_buckets=16,
        pos_att_type=("p2c", "c2p"), share_att_key=True, position_biased_input=False,
        type_vocab_size=0,
    )


def test_forward_shapes_and_finite():
    import mlx.core as mx

    from klusai.privacy.models.mlx_encoder import DebertaV2ForTokenClassification

    cfg = _small_config()
    model = DebertaV2ForTokenClassification(cfg, num_labels=7)
    ids = mx.array([[1, 5, 9, 3, 0, 0]], dtype=mx.int32)
    mask = mx.array([[1, 1, 1, 1, 0, 0]], dtype=mx.int32)
    logits = model(ids, mask)
    mx.eval(logits)
    assert tuple(logits.shape) == (1, 6, 7)
    assert bool(mx.all(mx.isfinite(logits)))


def test_relative_position_buckets_are_symmetric():
    import numpy as np

    from klusai.privacy.models.mlx_encoder import build_relative_position

    rp = build_relative_position(8, 8, bucket_size=16, max_position=64)
    rp = np.array(rp)[0]
    # diagonal (rel pos 0) is 0; matrix is antisymmetric in the near field
    assert np.all(np.diag(rp) == 0)
    assert rp[0, 1] == -rp[1, 0]


def test_disentangled_bias_contributes():
    """Zeroing the relative-position embeddings must change the output — proves the
    c2p/p2c disentangled-bias path is actually wired into attention (not dead code)."""
    import mlx.core as mx

    from klusai.privacy.models.mlx_encoder import DebertaV2ForTokenClassification

    ids = mx.array([[1, 5, 9, 3]], dtype=mx.int32)
    mask = mx.array([[1, 1, 1, 1]], dtype=mx.int32)

    cfg = _small_config()
    model = DebertaV2ForTokenClassification(cfg, num_labels=7)
    out_with = model.deberta(ids, mask)

    # zero the rel-embedding table -> c2p/p2c bias collapses to ~0; content path unchanged
    rel = model.deberta.rel_embeddings
    rel.weight = mx.zeros_like(rel.weight)
    out_without = model.deberta(ids, mask)

    mx.eval(out_with, out_without)
    assert float(mx.max(mx.abs(out_with - out_without))) > 1e-4


@pytest.mark.parametrize("model_id", ["microsoft/mdeberta-v3-base"])
def test_forward_parity_with_transformers(model_id):
    """Opt-in real-weight parity; skips offline or without the hf extra."""
    pytest.importorskip("transformers", reason="hf extra not installed")
    pytest.importorskip("torch", reason="hf extra not installed")
    import numpy as np
    import torch

    try:
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        hf = AutoModel.from_pretrained(model_id).eval()
    except Exception as e:  # offline / model not cached
        pytest.skip(f"model {model_id} unavailable offline: {type(e).__name__}")

    import mlx.core as mx

    from klusai.privacy.models.mlx_encoder import DebertaV2Config, DebertaV2ForTokenClassification
    from klusai.privacy.models.mlx_encoder_loader import load_hf_weights

    enc = tok(["Ion Popescu locuiește în București."], padding="max_length",
              truncation=True, max_length=32, return_tensors="pt")
    with torch.no_grad():
        ref = hf(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state.numpy()

    cfg = DebertaV2Config.from_hf(hf.config)
    m = DebertaV2ForTokenClassification(cfg, num_labels=2)
    load_hf_weights(m, hf.state_dict())
    out = m.deberta(mx.array(enc["input_ids"].numpy()), mx.array(enc["attention_mask"].numpy()))
    mx.eval(out)

    valid = enc["attention_mask"].numpy().astype(bool)
    max_abs = float(np.abs(ref[valid] - np.array(out)[valid]).max())
    assert max_abs < 2e-3, f"forward parity drift too large: {max_abs:.3e}"
