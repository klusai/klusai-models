"""Map a `transformers` DeBERTa-v2 state dict onto the MLX POC module — KLU-46 spike.

HF lays the bare model out as siblings `deberta.embeddings.*` and `deberta.encoder.*`.
The MLX POC (`mlx_encoder.DebertaV2Encoder`) folds both under one `deberta` module:
embeddings + the `layer` list + `rel_embeddings` + the rel-embedding `LayerNorm`. This
loader rewrites the key prefixes and hands the arrays to `Module.load_weights`.

`nn.Linear` in MLX uses the same `(out, in)` weight layout as `torch.nn.Linear`, so weights
copy across with no transpose.
"""

from __future__ import annotations

import mlx.core as mx

from .mlx_encoder import DebertaV2Config, DebertaV2ForTokenClassification


def _remap_key(k: str) -> str | None:
    """HF key -> MLX-module key. Returns None to drop a key the POC does not model."""
    # token-classification head (HF: classifier.weight / .bias) stays as-is
    if k.startswith("classifier."):
        return k
    # strip a leading "deberta." (present on the ForTokenClassification model)
    if k.startswith("deberta."):
        k = k[len("deberta.") :]
    if k.startswith("embeddings."):
        return "deberta." + k
    if k.startswith("encoder.layer."):
        return "deberta.layer." + k[len("encoder.layer.") :]
    if k == "encoder.rel_embeddings.weight":
        return "deberta.rel_embeddings.weight"
    if k.startswith("encoder.LayerNorm."):
        return "deberta.LayerNorm." + k[len("encoder.LayerNorm.") :]
    # pooler / unmodeled tensors -> drop
    return None


def load_hf_weights(model: DebertaV2ForTokenClassification, hf_state_dict) -> list[str]:
    """Load a torch state-dict (or dict of np/torch arrays) into the MLX model.

    Returns the list of HF keys that were dropped (not modeled), for transparency.
    """
    mlx_weights: dict[str, mx.array] = {}
    dropped: list[str] = []
    for k, v in hf_state_dict.items():
        nk = _remap_key(k)
        if nk is None:
            dropped.append(k)
            continue
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else v
        mlx_weights[nk] = mx.array(arr)
    # strict=False: the bare-encoder parity path leaves the classifier head un-loaded
    # (random); a real token-classification checkpoint supplies classifier.{weight,bias}.
    model.load_weights(list(mlx_weights.items()), strict=False)
    return dropped


def from_pretrained(model_id: str, num_labels: int | None = None):
    """Build the MLX POC and load real `transformers` weights for `model_id`.

    If `num_labels` is None, loads the bare encoder (random classifier head); otherwise
    expects a token-classification checkpoint with a matching head.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification

    hf_cfg = AutoConfig.from_pretrained(model_id)
    cfg = DebertaV2Config.from_hf(hf_cfg)

    if num_labels is None:
        hf_model = AutoModel.from_pretrained(model_id)
        n_labels = 2  # dummy head; not loaded
    else:
        hf_model = AutoModelForTokenClassification.from_pretrained(model_id, num_labels=num_labels)
        n_labels = num_labels

    model = DebertaV2ForTokenClassification(cfg, num_labels=n_labels)
    dropped = load_hf_weights(model, hf_model.state_dict())
    return model, cfg, dropped
