"""MLX-native DeBERTa-v2 encoder — KLU-46 feasibility SPIKE (proof-of-concept).

This is **spike code**, not a production path. It exists to answer one question:
*can `mlx.core` + `mlx.nn` express DeBERTa-v2's disentangled attention cleanly, and
does a forward pass match the `transformers` reference?* See `docs/klu-46-mlx-encoder-spike.md`
for the GO/NO-GO finding.

Scope / what this faithfully ports (enough for `microsoft/mdeberta-v3-base`):
* token + (optional) position + (optional) token-type embeddings, embed-proj, LayerNorm;
* disentangled self-attention: content->content, content->position (c2p),
  position->content (p2c), with the log-bucketed relative-position index;
* the `share_att_key=True` path (mdeberta-v3 uses it — positions reuse query_proj/key_proj);
* `norm_rel_ebd="layer_norm"` on the relative-position embedding table;
* a transformer FFN block (intermediate + output, GELU);
* a thin token-classification head (Linear) on top.

Deliberately NOT ported (out of scope; mdeberta-v3-base does not use them):
* the `ConvLayer` (only when `conv_kernel_size>0`);
* the separate `pos_key_proj`/`pos_query_proj` (only when `share_att_key=False`);
* dropout (eval-only POC), gradient checkpointing, the MLM/QA/seq-cls heads;
* any training loop / optimizer (a forward-parity POC is the bound of this spike).

`mlx` is an optional extra (`pip install '.[mlx]'`), so this module imports `mlx`
lazily-at-top and is never pulled in by the core SDK or the MPS training path.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DebertaV2Config:
    """The subset of `transformers` DebertaV2Config this POC reads."""

    vocab_size: int = 251000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    max_position_embeddings: int = 512
    relative_attention: bool = True
    position_buckets: int = 256
    max_relative_positions: int = -1
    pos_att_type: tuple[str, ...] = ("p2c", "c2p")
    share_att_key: bool = True
    position_biased_input: bool = False
    norm_rel_ebd: str = "layer_norm"
    type_vocab_size: int = 0
    embedding_size: int | None = None
    layer_norm_eps: float = 1e-7

    @classmethod
    def from_hf(cls, hf_config) -> "DebertaV2Config":
        pat = hf_config.pos_att_type or []
        if isinstance(pat, str):
            pat = [pat]
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            hidden_act=hf_config.hidden_act,
            max_position_embeddings=hf_config.max_position_embeddings,
            relative_attention=getattr(hf_config, "relative_attention", False),
            position_buckets=getattr(hf_config, "position_buckets", -1),
            max_relative_positions=getattr(hf_config, "max_relative_positions", -1),
            pos_att_type=tuple(pat),
            share_att_key=getattr(hf_config, "share_att_key", False),
            position_biased_input=getattr(hf_config, "position_biased_input", True),
            norm_rel_ebd=getattr(hf_config, "norm_rel_ebd", "none"),
            type_vocab_size=getattr(hf_config, "type_vocab_size", 0),
            embedding_size=getattr(hf_config, "embedding_size", None),
            layer_norm_eps=getattr(hf_config, "layer_norm_eps", 1e-7),
        )


# --- relative-position helpers (faithful ports of the transformers fns) -----------------


def make_log_bucket_position(relative_pos: mx.array, bucket_size: int, max_position: int) -> mx.array:
    sign = mx.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = mx.where(
        (relative_pos < mid) & (relative_pos > -mid),
        mx.array(mid - 1, dtype=relative_pos.dtype),
        mx.abs(relative_pos),
    )
    log_pos = (
        mx.ceil(mx.log(abs_pos / mid) / mx.log(mx.array((max_position - 1) / mid)) * (mid - 1)) + mid
    )
    bucket_pos = mx.where(abs_pos <= mid, relative_pos.astype(log_pos.dtype), log_pos * sign)
    return bucket_pos


def build_relative_position(query_size: int, key_size: int, bucket_size: int = -1, max_position: int = -1) -> mx.array:
    q_ids = mx.arange(query_size)
    k_ids = mx.arange(key_size)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.astype(mx.int32)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    return rel_pos_ids[None, :, :]  # [1, q, k]


# --- modules ----------------------------------------------------------------------------


class DisentangledSelfAttention(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.share_att_key = config.share_att_key
        self.pos_att_type = config.pos_att_type
        self.relative_attention = config.relative_attention
        self.position_buckets = config.position_buckets
        self.max_relative_positions = config.max_relative_positions
        if self.max_relative_positions < 1:
            self.max_relative_positions = config.max_position_embeddings
        self.pos_ebd_size = self.max_relative_positions
        if self.position_buckets > 0:
            self.pos_ebd_size = self.position_buckets

        if self.relative_attention and not self.share_att_key:
            # share_att_key=True for mdeberta-v3; the separate projections are out-of-scope.
            if "c2p" in self.pos_att_type:
                self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size)
            if "p2c" in self.pos_att_type:
                self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

    def _transpose_for_scores(self, x: mx.array, heads: int) -> mx.array:
        # x: [B, N, all_head] -> [B*heads, N, head_size]
        b, n, _ = x.shape
        x = x.reshape(b, n, heads, -1)
        x = x.transpose(0, 2, 1, 3)  # [B, heads, N, head_size]
        return x.reshape(b * heads, n, -1)

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array, rel_embeddings: mx.array | None,
                 relative_pos: mx.array | None) -> mx.array:
        q = self._transpose_for_scores(self.query_proj(hidden_states), self.num_attention_heads)
        k = self._transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        v = self._transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

        scale_factor = 1 + sum(t in self.pos_att_type for t in ("c2p", "p2c"))
        scale = mx.sqrt(mx.array(q.shape[-1] * scale_factor, dtype=mx.float32))
        scores = mx.matmul(q, k.transpose(0, 2, 1) / scale)

        if self.relative_attention and rel_embeddings is not None:
            rel_att = self._disentangled_bias(q, k, relative_pos, rel_embeddings, scale_factor)
            scores = scores + rel_att

        bh, n, _ = scores.shape
        b = bh // self.num_attention_heads
        scores = scores.reshape(b, self.num_attention_heads, n, n)

        # attention_mask: [B, N] (1=keep). Build [B,1,N,N] additive mask.
        m = attention_mask.astype(mx.bool_)  # [B, N]
        mask2d = (m[:, None, :] & m[:, :, None])[:, None, :, :]  # [B,1,N,N]
        neg = mx.finfo(mx.float32).min
        scores = mx.where(mask2d, scores, mx.array(neg, dtype=scores.dtype))

        probs = mx.softmax(scores, axis=-1)
        probs = probs.reshape(bh, n, n)
        ctx = mx.matmul(probs, v)  # [B*heads, N, head_size]
        ctx = ctx.reshape(b, self.num_attention_heads, n, -1).transpose(0, 2, 1, 3)
        return ctx.reshape(b, n, -1)

    def _disentangled_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        n = query_layer.shape[-2]
        if relative_pos is None:
            relative_pos = build_relative_position(n, n, self.position_buckets, self.max_relative_positions)
        # relative_pos: [1, q, k] -> [1, 1, q, k]
        relative_pos = relative_pos[:, None, :, :].astype(mx.int32)

        att_span = self.pos_ebd_size
        rel_embeddings = rel_embeddings[0 : att_span * 2, :][None, :, :]

        bh = query_layer.shape[0]
        repeat = bh // self.num_attention_heads
        if self.share_att_key:
            pos_query = mx.tile(
                self._transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads),
                (repeat, 1, 1),
            )
            pos_key = mx.tile(
                self._transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads),
                (repeat, 1, 1),
            )
        else:
            pos_key = mx.tile(
                self._transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads),
                (repeat, 1, 1),
            )
            pos_query = mx.tile(
                self._transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads),
                (repeat, 1, 1),
            )

        score = mx.zeros((bh, n, n), dtype=query_layer.dtype)

        if "c2p" in self.pos_att_type:
            scale = mx.sqrt(mx.array(pos_key.shape[-1] * scale_factor, dtype=mx.float32))
            c2p_att = mx.matmul(query_layer, pos_key.transpose(0, 2, 1))  # [bh, n, 2*att_span]
            c2p_pos = mx.clip(relative_pos + att_span, 0, att_span * 2 - 1)  # [1,1,q,k]
            idx = mx.broadcast_to(c2p_pos[0], (bh, n, n)).astype(mx.int32)
            c2p_att = mx.take_along_axis(c2p_att, idx, axis=-1)
            score = score + c2p_att / scale

        if "p2c" in self.pos_att_type:
            scale = mx.sqrt(mx.array(pos_query.shape[-1] * scale_factor, dtype=mx.float32))
            r_pos = relative_pos  # q==k here
            p2c_pos = mx.clip(-r_pos + att_span, 0, att_span * 2 - 1)  # [1,1,q,k]
            p2c_att = mx.matmul(key_layer, pos_query.transpose(0, 2, 1))  # [bh, n, 2*att_span]
            idx = mx.broadcast_to(p2c_pos[0], (bh, n, n)).astype(mx.int32)
            p2c_att = mx.take_along_axis(p2c_att, idx, axis=-1).transpose(0, 2, 1)
            score = score + p2c_att / scale

        return score


class SelfOutput(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class Attention(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = SelfOutput(config)

    def __call__(self, hidden_states, attention_mask, rel_embeddings, relative_pos):
        self_out = self.self(hidden_states, attention_mask, rel_embeddings, relative_pos)
        return self.output(self_out, hidden_states)


class Intermediate(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.gelu  # mdeberta-v3 uses "gelu"

    def __call__(self, x):
        return self.act(self.dense(x))


class Output(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class Layer(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def __call__(self, hidden_states, attention_mask, rel_embeddings, relative_pos):
        att = self.attention(hidden_states, attention_mask, rel_embeddings, relative_pos)
        inter = self.intermediate(att)
        return self.output(inter, att)


class Embeddings(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.embedding_size = config.embedding_size or config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size)
        self.position_biased_input = config.position_biased_input
        if self.position_biased_input:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)
        else:
            self.token_type_embeddings = None
        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        else:
            self.embed_proj = None
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids, attention_mask):
        seq_len = input_ids.shape[1]
        emb = self.word_embeddings(input_ids)
        if self.position_biased_input:
            pos_ids = mx.arange(seq_len)[None, :]
            emb = emb + self.position_embeddings(pos_ids)
        if self.token_type_embeddings is not None:
            tt = mx.zeros(input_ids.shape, dtype=mx.int32)
            emb = emb + self.token_type_embeddings(tt)
        if self.embed_proj is not None:
            emb = self.embed_proj(emb)
        emb = self.LayerNorm(emb)
        # mask the embeddings (transformers multiplies by the input mask)
        mask = attention_mask.astype(emb.dtype)[:, :, None]
        return emb * mask


class DebertaV2Encoder(nn.Module):
    """The bare encoder (`deberta` prefix in the HF state dict)."""

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.layer = [Layer(config) for _ in range(config.num_hidden_layers)]
        self.relative_attention = config.relative_attention
        self.norm_rel_ebd = [x.strip() for x in config.norm_rel_ebd.lower().split("|")]
        if self.relative_attention:
            self.max_relative_positions = config.max_relative_positions
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.position_buckets = config.position_buckets
            pos_ebd_size = self.max_relative_positions * 2
            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2
            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)
            if "layer_norm" in self.norm_rel_ebd:
                self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _get_rel_embedding(self):
        if not self.relative_attention:
            return None
        rel = self.rel_embeddings.weight
        if "layer_norm" in self.norm_rel_ebd:
            rel = self.LayerNorm(rel)
        return rel

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        hidden = self.embeddings(input_ids, attention_mask)
        n = input_ids.shape[1]
        rel_embeddings = self._get_rel_embedding()
        relative_pos = None
        if self.relative_attention:
            relative_pos = build_relative_position(n, n, self.position_buckets, self.max_relative_positions)
        for layer in self.layer:
            hidden = layer(hidden, attention_mask, rel_embeddings, relative_pos)
        return hidden


class DebertaV2ForTokenClassification(nn.Module):
    """Encoder + a thin linear token-classification head (matches the HF layout)."""

    def __init__(self, config: DebertaV2Config, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.deberta = DebertaV2Encoder(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        sequence_output = self.deberta(input_ids, attention_mask)
        return self.classifier(sequence_output)
