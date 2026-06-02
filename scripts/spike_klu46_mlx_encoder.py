#!/usr/bin/env python3
"""KLU-46 spike — MLX-native DeBERTa-v2 encoder: forward-pass parity vs transformers.

Loads `microsoft/mdeberta-v3-base` (cached ~280M), runs the bare encoder forward through
both the `transformers` reference and the MLX POC (`klusai.privacy.models.mlx_encoder`),
and reports the max/mean absolute difference of the last hidden state on a few inputs.

This is SPIKE code (forward parity only — no training). See docs/klu-46-mlx-encoder-spike.md.

    python scripts/spike_klu46_mlx_encoder.py                 # parity (needs '.[mlx]' + '.[hf]')
    python scripts/spike_klu46_mlx_encoder.py --bench          # + a small forward throughput bench
    python scripts/spike_klu46_mlx_encoder.py --model <hf-id>  # try another deberta-v2 checkpoint
"""

from __future__ import annotations

import time

import click

from klusai.privacy.models.logger import get_logger

logger = get_logger("spike-klu46")

DEFAULT_MODEL = "microsoft/mdeberta-v3-base"
SAMPLES = [
    "Ion Popescu locuiește în București, pe strada Florilor 12.",
    "Contact me at jane.doe@example.com or +40 721 234 567.",
    "The patient, born 1985-03-02, was admitted to the clinic.",
]


def _run_parity(model_id: str, seq_len: int) -> float:
    import mlx.core as mx
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    from klusai.privacy.models.mlx_encoder import DebertaV2Config, DebertaV2ForTokenClassification
    from klusai.privacy.models.mlx_encoder_loader import load_hf_weights

    tok = AutoTokenizer.from_pretrained(model_id)
    enc = tok(SAMPLES, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")

    logger.info("loading transformers reference %s ...", model_id)
    t0 = time.perf_counter()
    hf = AutoModel.from_pretrained(model_id).eval()
    logger.info("reference loaded in %.1fs", time.perf_counter() - t0)
    with torch.no_grad():
        ref = hf(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
    ref_np = ref.numpy()

    cfg = DebertaV2Config.from_hf(hf.config)
    mlx_model = DebertaV2ForTokenClassification(cfg, num_labels=2)
    dropped = load_hf_weights(mlx_model, hf.state_dict())
    logger.info("MLX POC loaded; dropped (unmodeled) HF keys: %s", dropped or "<none>")

    ids = mx.array(enc["input_ids"].numpy())
    mask = mx.array(enc["attention_mask"].numpy())
    out = mlx_model.deberta(ids, mask)  # bare encoder last hidden state
    mx.eval(out)
    mlx_np = np.array(out)

    # compare only non-padding positions
    valid = enc["attention_mask"].numpy().astype(bool)
    diff = np.abs(ref_np[valid] - mlx_np[valid])
    max_abs, mean_abs = float(diff.max()), float(diff.mean())
    ref_scale = float(np.abs(ref_np[valid]).mean())
    logger.info(
        "PARITY  max|Δ|=%.3e  mean|Δ|=%.3e  (ref mean|x|=%.3e, rel mean=%.2e)  shape=%s",
        max_abs, mean_abs, ref_scale, mean_abs / ref_scale, tuple(mlx_np.shape),
    )
    tol = 2e-3
    verdict = "PASS" if max_abs < tol else "FAIL"
    logger.info("PARITY VERDICT: %s (tol max|Δ| < %.0e)", verdict, tol)
    return max_abs


def _run_bench(model_id: str, seq_len: int, iters: int) -> None:
    import mlx.core as mx
    import torch
    from transformers import AutoModel, AutoTokenizer

    from klusai.privacy.models.mlx_encoder import DebertaV2Config, DebertaV2ForTokenClassification
    from klusai.privacy.models.mlx_encoder_loader import load_hf_weights

    tok = AutoTokenizer.from_pretrained(model_id)
    batch = [SAMPLES[i % len(SAMPLES)] for i in range(16)]
    enc = tok(batch, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")

    hf = AutoModel.from_pretrained(model_id).eval()
    cfg = DebertaV2Config.from_hf(hf.config)
    mlx_model = DebertaV2ForTokenClassification(cfg, num_labels=2)
    load_hf_weights(mlx_model, hf.state_dict())

    ids = mx.array(enc["input_ids"].numpy())
    mask = mx.array(enc["attention_mask"].numpy())

    # MLX (GPU by default on Apple Silicon)
    mx.eval(mlx_model.deberta(ids, mask))  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(mlx_model.deberta(ids, mask))
    mlx_dt = (time.perf_counter() - t0) / iters

    # transformers MPS
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    hf_mps = hf.to(dev)
    ii, mm = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    with torch.no_grad():
        _ = hf_mps(input_ids=ii, attention_mask=mm)  # warmup
        if dev == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = hf_mps(input_ids=ii, attention_mask=mm)
        if dev == "mps":
            torch.mps.synchronize()
        mps_dt = (time.perf_counter() - t0) / iters

    bs = len(batch)
    logger.info("BENCH (batch=%d, seq=%d, %d iters)", bs, seq_len, iters)
    logger.info("  MLX  forward: %.4fs/it  (%.1f samples/s)", mlx_dt, bs / mlx_dt)
    logger.info("  torch-%s fwd: %.4fs/it  (%.1f samples/s)", dev, mps_dt, bs / mps_dt)
    logger.info("  MLX / torch-%s throughput ratio: %.2fx", dev, mps_dt / mlx_dt)


@click.command()
@click.option("--model", "model_id", default=DEFAULT_MODEL, show_default=True)
@click.option("--seq-len", default=64, show_default=True)
@click.option("--bench", is_flag=True, help="Run a small forward-throughput bench (MLX vs torch-MPS).")
@click.option("--iters", default=30, show_default=True)
def main(model_id: str, seq_len: int, bench: bool, iters: int) -> None:
    _run_parity(model_id, seq_len)
    if bench:
        _run_bench(model_id, seq_len, iters)


if __name__ == "__main__":
    main()
