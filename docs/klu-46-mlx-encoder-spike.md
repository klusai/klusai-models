# KLU-46 — MLX-native DeBERTa-v2 encoder spike (feasibility)

**Status:** Done (bounded feasibility spike). **Recommendation: GO for a feasibility-proven
MLX *inference* encoder, DEFER the full retrainable production path.** MLX-core (`mlx.core` +
`mlx.nn`) **can** express DeBERTa-v2 disentangled attention cleanly, and a from-scratch MLX
port loads the real `microsoft/mdeberta-v3-base` weights and reproduces the `transformers`
forward pass to fp32 numerical-noise tolerance. The remaining gap to "production" is a
training path (autograd loss/optimizer loop, dropout, SentencePiece tokenizer, checkpoint
round-trip, the MPS-parity test matrix) — multi-day work that is **not justified now** given
the MPS path already trains this family well (KLU-45/48).

Reproduce:

```bash
pip install '.[mlx]' '.[hf]'          # POC needs mlx-core (mlx.core/mlx.nn) + the hf reference
python scripts/spike_klu46_mlx_encoder.py            # forward-pass parity vs transformers
python scripts/spike_klu46_mlx_encoder.py --bench    # + MLX vs torch-MPS forward throughput
```

Spike code: `klusai/privacy/models/mlx_encoder.py` (the module),
`klusai/privacy/models/mlx_encoder_loader.py` (HF→MLX weight map),
`scripts/spike_klu46_mlx_encoder.py` (parity + bench), `tests/test_mlx_encoder.py`.

## The crux: can MLX-core express disentangled attention? — **YES**

DeBERTa-v2's disentangled attention adds two relative-position bias terms to the standard
content→content scores: content→position (**c2p**) and position→content (**p2c**), indexed by
a **log-bucketed relative position** matrix. Every primitive it needs exists in MLX-core:

| transformers op (`modeling_deberta_v2.py`) | MLX-core equivalent |
| --- | --- |
| `nn.Linear` (`query/key/value_proj`), `(out,in)` layout | `mlx.nn.Linear` — **same layout, weights copy with no transpose** |
| `nn.Embedding`, `nn.LayerNorm`, `gelu`, `softmax` | `mlx.nn.{Embedding,LayerNorm}`, `mlx.nn.gelu`, `mlx.softmax` |
| `torch.bmm` / batched matmul | `mx.matmul` |
| log-bucket index (`sign/abs/ceil/log/where`) | `mx.sign/abs/ceil/log/where` — 1:1 |
| `torch.gather(..., dim=-1)` for the c2p/p2c lookup | `mx.take_along_axis(..., axis=-1)` |
| `torch.clamp` | `mx.clip` |

The `make_log_bucket_position` / `build_relative_position` helpers and the c2p/p2c gather port
**line-for-line**. mdeberta-v3-base's config makes the port even smaller: `share_att_key=True`
(positions reuse `query_proj`/`key_proj` — no separate pos projections), `type_vocab_size=0`
(no token-type embeddings), `position_biased_input=False` (no absolute-position add),
`embedding_size==hidden_size` (no embed-proj), and **no ConvLayer** (`conv_kernel_size` unset).
The MLX module is ~330 lines and reads like the reference. **The hard part is not hard in
MLX-core** — this is the headline finding.

A second, practical finding: the POC depends only on the **`mlx`** extra (mlx-core), **not
`mlx-lm`**. The KLU-11 blocker ("`mlx-lm` has no encoder arch") and the README's `hf`/`mlx`
non-combinable pin conflict are both **`mlx-lm`** problems (it caps `transformers`). Plain
`mlx` carries no transformers constraint, so it **coexists cleanly with the `hf` stack** —
this spike ran both in one environment.

## Forward-pass parity vs `transformers` — **PASS**

Same machine (Mac Studio M3 Ultra), real cached weights, eval mode, fp32 throughout.

**Bare encoder** (last hidden state), 3 multilingual sentences, seq 64:

| metric | value |
| --- | --- |
| `max | Δ |` | **7.9e-4** |
| `mean | Δ |` | **1.2e-5** |
| reference scale (`mean | x |`) | 4.9e-1 |
| relative mean error | **2.5e-5** |

**Full token-classification model** on the fine-tuned `klusai/kp-deid-mdeberta-280m`
(73-label head loaded too), logits:

| metric | value |
| --- | --- |
| `max | Δ |` (logits) | **6.9e-4** |
| `mean | Δ |` (logits) | 3.2e-5 |
| **argmax label agreement** | **100.00%** |

The residual ~1e-3 max diff is fp32 reduction-order noise (the M3 Ultra MPS↔CPU spread in
KLU-45 was the same order). The disentangled-attention bias, the rel-embedding `LayerNorm`,
the embedding mask, and the classifier head all reproduce correctly: **the MLX encoder is
numerically equivalent to the transformers reference, and produces identical NER predictions.**

## Forward throughput vs PyTorch-MPS — **MLX modestly ahead**

`--bench`, batch 16, eval forward only, 50 iters, MLX(GPU) vs `transformers` on torch-MPS:

| seq len | MLX forward | torch-MPS forward | MLX / MPS |
| --- | --- | --- | --- |
| 64 | 0.0234 s/it (683 samp/s) | 0.0261 s/it (613 samp/s) | **1.11x** |
| 256 | 0.0795 s/it (201 samp/s) | 0.1102 s/it (145 samp/s) | **1.39x** |

MLX forward is ~1.1–1.4x the torch-MPS forward, the edge widening at the longer (training-
representative) seq 256. This is **forward-only**; it does not include a backward pass and so
is **not** a training-throughput claim. Consistent with KLU-48's finding that this 280M
encoder is bandwidth-bound on the M3 Ultra, the win is modest, not transformational.

**Power:** not measured. Per KLU-48, the only power reader on this box is `sudo powermetrics`
and `sudo` is unavailable non-interactively here. To read watts during a run:
`sudo powermetrics --samplers gpu_power -i 1000 -n 10` in a second terminal.

## What I did NOT do (scope honesty)

- **No training.** No autograd loss/optimizer loop, no LoRA in MLX, no dropout (eval-only
  POC). `mlx` has `mlx.nn.value_and_grad` + `mlx.optimizers`, so this is feasible — but
  building + validating a token-classification trainer (CE with `-100` masking), LoRA
  adapters, and a checkpoint round-trip is the bulk of the remaining effort.
- **No SentencePiece-in-MLX.** The POC reuses the HF tokenizer to make tensors; a standalone
  MLX path would still lean on `tokenizers`/`sentencepiece` (fine — tokenization is not the
  GPU path).
- **Only the mdeberta-v3-base config shape.** The `share_att_key=False`, ConvLayer, token-type,
  and embed-proj branches are stubbed/asserted, not exercised. A general DeBERTa-v2 port would
  need them.
- **No MPS-parity test matrix / no backward parity.** Forward parity on a handful of inputs is
  the bound of the spike.

### Effort estimate for a full retrainable production path

~**3–5 focused days**: (1) MLX training loop + CE/`-100` loss + AdamW — ~1 day; (2) LoRA in
MLX matching the `query_proj/key_proj/value_proj` targets + merge/save — ~1 day; (3) dropout,
the un-ported config branches, checkpoint round-trip (MLX↔HF safetensors) — ~1 day;
(4) backward-parity + convergence validation vs the MPS run, plus wiring `--backend mlx` for
`xlmr-ner` and the test matrix — ~1–2 days. Plus ongoing maintenance: this is a hand-port that
must track upstream DeBERTa-v2 changes, which the `transformers` path gets for free.

## Recommendation — **GO (inference POC, proven) / DEFER (full production path)**

- **GO** on the narrow finding: MLX-core **can** do DeBERTa-v2 disentangled attention, the
  port is small and clean, forward parity is exact, and `mlx` (not `mlx-lm`) sidesteps the
  KLU-11/pin-conflict blockers. The POC in this PR is the proof. This **revises KLU-11's
  "MLX not viable" specifically for `mlx-lm`** — mlx-core is a different story.
- **DEFER** the full retrainable production encoder. The MPS path already trains this family
  at ~115–189 samples/s, numerically matched to CPU, with zero hand-ported code to maintain
  (KLU-45/48). The MLX forward edge (~1.1–1.4x, bandwidth-bound) does not justify owning a
  3–5 day hand-port + its maintenance tax **for a comparison/baseline model**. Revisit if/when
  (a) MLX-native *inference* serving on Apple Silicon becomes a product requirement (the POC is
  a strong starting point), or (b) the MoE/causal-LM track — which *does* saturate the GPU —
  needs an encoder companion in the same MLX runtime.

This is a **doc + POC** PR, not a production wiring change: the MLX module ships behind the
`mlx` extra, is never imported by the SDK or the MPS training path, and every test skips
gracefully where `mlx`/the model is absent. **Do not merge as a production path** — it is the
recorded feasibility finding.
