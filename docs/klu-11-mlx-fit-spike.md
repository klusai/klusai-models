# KLU-11 — MLX-fit spike: mDeBERTa-v3 LoRA token-classification on Mac Studio

**Status:** Done. **Decision: CPU fallback** (`transformers` + `peft` on CPU, threads
capped to 4). MLX is **not viable** for this model family today.

Reproduce: `python scripts/spike_klu11_mdeberta_lora.py` (full proof, downloads ~280M once)
or `--mlx-only` (instant; just shows the MLX blocker). Full rationale lives in that
script's module docstring.

## What we tried

Per the repo's MLX-first compute model we tried MLX (`mlx-lm`) first, then fell back to
the device-agnostic CPU/CUDA path (`transformers` + `peft`).

## Evidence — MLX BLOCKED

`mlx-lm` (0.29.1) is a **causal/decoder-LM** toolkit and has **no encoder architectures**:

```
$ ls .venv/.../mlx_lm/models/ | grep -iE 'deberta|bert|roberta|xlm|encoder'   # empty
```

`mdeberta-v3-base` has `model_type: "deberta-v2"`. `mlx_lm.load` resolves the arch by
importing `mlx_lm.models.<model_type>`, which for this model produces:

```
ValueError: Model type deberta-v2 not supported.
```

Additionally the `mlx_lm` LoRA tuner implements only a causal-LM (next-token cross-entropy)
loss — there is **no token-classification head/loss**. So even a hand-ported encoder could
not be NER-finetuned through that trainer. The classic mDeBERTa frictions (tied embeddings,
custom SentencePiece tokenizer) are downstream of this; we never reach them because the
architecture is unsupported up front. A DeBERTa-v2 + token-class-head + NER-trainer port to
MLX is multi-day work — not justified for a *comparison/baseline* model.

## Evidence — CPU fallback WORKS

`scripts/spike_klu11_mdeberta_lora.py` runs a real tiny LoRA step on the actual 280M model:

| metric | value |
|---|---|
| base load (cold, incl. slow SP tokenizer convert) | ~41 s |
| LoRA trainable params | 498,505 / 278,773,394 (0.179%) |
| loss over 3 steps | 4.61 → 4.56 → 4.43 (decreasing — grad path OK) |
| warm throughput | ~0.48 s/step, ~2.1 it/s @ bs=8, seq=64, CPU/4 threads |

Machine: Mac Studio, macOS 15.7 arm64, Python 3.13, torch 2.12, transformers 4.57.6,
peft 0.19.1, mlx-lm 0.29.1. Consistent with the repo perf note (MoE ~3x slower on MPS than
CPU): for a 280M encoder, CPU with threads capped to ~4 is the cheap, reliable Mac path.

## Hand-off to KLU-17 (the real mDeBERTa-280m finetune)

- Stack: `transformers` + `peft`, `AutoModelForTokenClassification`,
  `TaskType.TOKEN_CLS`. **LoRA `target_modules = ["query_proj","key_proj","value_proj"]`**
  (DeBERTa-v2 disentangled-attention names — *not* the BERT `query/key/value`).
- Backend: `cpu` for Mac smoke tests; **`cuda` (DO droplet) for the real run**. `mlx` should
  be rejected at config time for the `xlmr-ner` / `classifier` families; the `-mlx` publish
  variant does not apply to them.
- Tokenizer: mDeBERTa uses a SentencePiece tokenizer (fast-convert warns about byte-fallback —
  harmless for training; mind `model_max_length` / truncation).
- Labels: shared `europriv_bench.taxonomy.bioes_labels()` (73 labels), not a local copy.
- The classifier head is newly initialized (expected) — must be trained downstream.
