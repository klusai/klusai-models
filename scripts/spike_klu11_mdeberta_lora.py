#!/usr/bin/env python3
"""KLU-11 spike — validate mDeBERTa-v3 LoRA token-classification on Mac Studio.

DECISION: **CPU fallback (transformers + peft on CPU, threads capped to 4).**
MLX is NOT viable for this model family today — see EVIDENCE below.

Why this spike exists
---------------------
The H1 plan's first comparison model, ``kp-deid-mdeberta-280m`` (KLU-17), assumes
``microsoft/mdeberta-v3-base`` LoRA token-classification is trainable on this Mac Studio.
Per the repo's "MLX-first" compute model we tried MLX first, then fell back.

EVIDENCE — why MLX does not work (reproducible, no download required)
---------------------------------------------------------------------
``mlx-lm`` is a *causal/decoder-LM* toolkit. It has **no encoder architectures at all**:

    $ ls .venv/.../mlx_lm/models/ | grep -iE 'deberta|bert|roberta|xlm|encoder'
    # (empty — only llama/qwen/gemma/mixtral/… decoder families)

``mdeberta-v3-base`` reports ``model_type: "deberta-v2"``. ``mlx_lm.load`` resolves the
architecture by importing ``mlx_lm.models.<model_type>``; for this model it raises:

    ValueError: Model type deberta-v2 not supported.

On top of that, the ``mlx_lm`` LoRA tuner only implements a **causal-LM (next-token
cross-entropy) loss** — there is no token-classification head/loss, so even a hand-ported
encoder could not be fine-tuned for NER through that trainer. The well-known mDeBERTa
frictions (tied embeddings, custom SentencePiece tokenizer) are downstream of this; we
never reach them because the architecture is unsupported up front. Porting DeBERTa-v2 +
a TokenClassification head + a NER trainer to MLX is a multi-day effort, out of scope for
a comparison/baseline model.

DECISION RATIONALE
------------------
* MLX path: blocked (no encoder arch, no token-class loss). Not worth a port for a
  *comparison* model.
* CPU fallback: works end-to-end here (this script proves a real LoRA step on the actual
  280M model). Consistent with the repo's perf note that the privacy-filter MoE runs ~3x
  slower on MPS than CPU — for a 280M encoder, CPU (threads capped ~4) is the cheap,
  reliable path on this box; burst to a CUDA droplet (``--backend cuda``) for the full run.

This script runs the fallback for real and reports throughput. Run from repo root:

    python scripts/spike_klu11_mdeberta_lora.py            # tiny CPU proof (downloads ~280M once)
    python scripts/spike_klu11_mdeberta_lora.py --mlx-only # just demonstrate the MLX blocker

KLU-17 take-aways
-----------------
* Train mDeBERTa-v3 / XLM-R token-classification via ``transformers + peft`` with a
  ``TokenClassification`` head; LoRA targets ``query_proj,key_proj,value_proj`` (DeBERTa-v2
  disentangled-attention proj names — *not* the BERT ``query/key/value``).
* Backend for this family is effectively ``cuda`` (DO droplet) for a real run, ``cpu`` for
  smoke tests on the Mac. ``mlx`` should be rejected at config time for ``xlmr-ner`` /
  ``classifier`` families. The ``-mlx`` publish variant does not apply here.
* mDeBERTa needs ``slow``/SentencePiece tokenizer + ``model_max_length`` care; label set is
  the shared ``europriv_bench.taxonomy.bioes_labels()`` (73 labels).
"""

from __future__ import annotations

import os
import time

import click

# Cap threads to ~4 per the repo's perf notes (CPU beats MPS for these small models).
# Must be set before torch / numpy import their native libs.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

from klusai.privacy.models.logger import get_logger  # noqa: E402

logger = get_logger("spike-klu11")

BASE_MODEL = "microsoft/mdeberta-v3-base"


def probe_mlx() -> str:
    """Reproduce the MLX blocker; return a one-line evidence string."""
    try:
        import importlib

        from mlx_lm import models  # noqa: F401  (ensures mlx_lm is importable)
    except Exception as e:  # pragma: no cover - mlx_lm not installed
        return f"mlx_lm import failed: {e!r}"

    # mdeberta-v3-base's HF config model_type is 'deberta-v2'.
    model_type = "deberta-v2"
    module = f"mlx_lm.models.{model_type.replace('-', '_')}"
    try:
        importlib.import_module(module)
        return f"UNEXPECTED: {module} exists — re-evaluate MLX viability."
    except ModuleNotFoundError:
        return (
            f"mlx_lm has no '{module}' (model_type={model_type!r}); mlx_lm.load would raise "
            f"'ValueError: Model type {model_type} not supported.' — no encoder arch & no "
            f"token-classification loss in the mlx_lm tuner. MLX path BLOCKED."
        )


def run_cpu_fallback(n_examples: int, steps: int, seq_len: int) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from europriv_bench.taxonomy import bioes_labels

    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device("cpu")

    labels = bioes_labels()
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}
    logger.info("CPU fallback: %d labels, base=%s, threads=%d", len(labels), BASE_MODEL, torch.get_num_threads())

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)  # SentencePiece slow tokenizer for mDeBERTa
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(labels), id2label=id2label, label2id=label2id
    )
    logger.info("loaded base in %.1fs", time.perf_counter() - t0)

    # DeBERTa-v2 disentangled-attention projections (NOT bert-style query/key/value).
    lora = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["query_proj", "key_proj", "value_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA attached: %d trainable / %d total (%.3f%%)", trainable, total, 100 * trainable / total)

    # Tiny synthetic token-classification slice — enough to prove the gradient path.
    rng = torch.Generator().manual_seed(0)
    sentences = [f"Contact person number {i} lives in city {i} with id {i}." for i in range(n_examples)]
    enc = tok(sentences, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    enc["labels"] = torch.randint(0, len(labels), enc["input_ids"].shape, generator=rng)
    enc["labels"][enc["attention_mask"] == 0] = -100  # ignore padding in CE loss

    model.train().to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)

    bs = min(8, n_examples)
    step_times: list[float] = []
    for step in range(steps):
        sl = slice((step * bs) % n_examples, (step * bs) % n_examples + bs)
        batch = {k: v[sl].to(device) for k, v in enc.items()}
        s = time.perf_counter()
        opt.zero_grad()
        out = model(**batch)
        out.loss.backward()
        opt.step()
        dt = time.perf_counter() - s
        step_times.append(dt)
        logger.info("step %d/%d  loss=%.4f  %.3fs/step (bs=%d, seq=%d)", step + 1, steps, out.loss.item(), dt, bs, seq_len)

    warm = step_times[1:] or step_times  # drop first (lazy-init) step
    s_per_step = sum(warm) / len(warm)
    logger.info(
        "DONE — warm throughput ~%.3f s/step (~%.2f it/s) at bs=%d seq=%d on CPU/4-threads",
        s_per_step,
        1.0 / s_per_step,
        bs,
        seq_len,
    )


@click.command()
@click.option("--mlx-only", is_flag=True, help="Only demonstrate the MLX blocker; skip the CPU run/download.")
@click.option("--n-examples", default=256, show_default=True)
@click.option("--steps", default=3, show_default=True)
@click.option("--seq-len", default=64, show_default=True)
def main(mlx_only: bool, n_examples: int, steps: int, seq_len: int) -> None:
    logger.info("MLX probe: %s", probe_mlx())
    if mlx_only:
        return
    run_cpu_fallback(n_examples=n_examples, steps=steps, seq_len=seq_len)


if __name__ == "__main__":
    main()
