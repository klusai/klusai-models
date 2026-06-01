#!/usr/bin/env python3
"""Unified training CLI for all KP model families (mirrors diacritics-finetuning train.py).

Run from the repo root after `make install`:

    # token classification (xlmr-ner family) — transformers+peft LoRA, CPU-feasible smoke run
    python scripts/train.py xlmr-ner --base microsoft/mdeberta-v3-base \\
        --dataset klusai/ds-kp-general-ro-50k --out klusai/kp-deid-mdeberta-280m \\
        --backend cuda --epochs 1 --max-train 800 --push

    # MoE continue-finetune (primary track) — lands later
    python scripts/train.py moe-finetune --base openai/privacy-filter \\
        --dataset klusai/ds-kp-legal-ro-50k --out klusai/kp-deid-moe-ro --backend mlx

Backend is device-agnostic: `--backend cuda` (CPU/DO GPU droplet via transformers) or
`--backend mlx` (Mac Studio, MoE family only). Datasets/checkpoints sync via HF Hub so a
droplet stays stateless. Per KLU-11 the encoder token-classification family never uses MLX.
"""

from __future__ import annotations

import click

from klusai.privacy.models.logger import get_logger
from klusai.privacy.models.training.config import Backend, Family, TrainingConfig

logger = get_logger("train")


def _run(cfg: TrainingConfig) -> None:
    logger.info("family=%s base=%s backend=%s → %s", cfg.family.value, cfg.base_model, cfg.backend.value, cfg.publish_id())

    if cfg.family is Family.XLMR_NER:
        # KLU-11: encoder token classification is transformers+peft (never MLX).
        if cfg.backend is Backend.MLX:
            raise click.UsageError(
                "xlmr-ner is a transformers+peft family (KLU-11 rejected MLX for it); use --backend cuda "
                "(runs on CPU when no GPU is present)."
            )
        from klusai.privacy.models.training.token_classification import train_token_classification

        opts = cfg.extra
        # KLU-45: on the Mac tier this family defaults to the GPU (MPS) — materially faster than
        # CPU and numerically stable (docs/klu-45-mps-vs-cpu.md). Device resolution lives in the
        # backend's resolve_device(): with no --device, --gpu (the default) → "auto" (MPS if
        # present, else CUDA droplet, else CPU); CPU stays the guaranteed fallback, selectable with
        # --device cpu or the legacy --cpu.
        result = train_token_classification(
            base_model=cfg.base_model,
            dataset=cfg.dataset,
            publish_id=cfg.publish_id(),
            output_dir=opts.get("output_dir") or f"runs/{cfg.output_repo.split('/')[-1]}",
            epochs=cfg.epochs,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            lora_rank=cfg.lora_rank or 8,
            max_train=opts.get("max_train"),
            max_eval=opts.get("max_eval"),
            push=opts.get("push", False),
            cpu=opts.get("cpu", False),
            threads=opts.get("threads", 4),
            device=opts.get("device"),
        )
        logger.info(
            "done: %s (%d train / %d eval, %d epochs, eval_loss=%s, pushed=%s) → %s",
            result.publish_id, result.train_examples, result.eval_examples,
            result.epochs, result.eval_loss, result.pushed, result.output_dir,
        )
        return

    raise NotImplementedError(
        f"training for {cfg.family.value} lands later — dispatch to the {cfg.backend.value} backend "
        f"(mlx-lm for the MoE family), then push_to_hub({cfg.publish_id()})"
    )


@click.group()
def cli() -> None:
    """Train/finetune KP privacy models."""


def _common(f):
    f = click.option("--base", "base_model", required=True, help="Base model HF id.")(f)
    f = click.option("--dataset", required=True, help="Training dataset HF id.")(f)
    f = click.option("--out", "output_repo", required=True, help="Output HF repo id.")(f)
    f = click.option("--backend", type=click.Choice([b.value for b in Backend]), default="mlx")(f)
    f = click.option("--epochs", type=int, default=5)(f)
    f = click.option("--lr", type=float, default=3e-4, help="Learning rate.")(f)
    f = click.option("--batch-size", type=int, default=8)(f)
    f = click.option("--lora-rank", type=int, default=8, help="LoRA rank (parameter-efficient families).")(f)
    f = click.option("--max-train", type=int, default=None, help="Cap training examples (bounded/smoke run).")(f)
    f = click.option("--max-eval", type=int, default=None, help="Cap eval examples.")(f)
    f = click.option("--output-dir", default=None, help="Local output dir (default runs/<name>).")(f)
    f = click.option("--push/--no-push", default=False, help="push_to_hub the trained model.")(f)
    f = click.option("--cpu/--gpu", default=False, help="Legacy: force CPU (--cpu) or allow accelerator (--gpu, default). Superseded by --device; KLU-45 made the Mac GPU/MPS the default for xlmr-ner.")(f)
    f = click.option(
        "--device",
        type=click.Choice(["auto", "cpu", "mps", "cuda"]),
        default=None,
        help="Training device (KLU-45). 'mps' = Mac GPU (Metal), 'auto' = best Mac-tier device. "
        "Overrides --cpu/--gpu; CPU is the guaranteed fallback when the requested device is absent.",
    )(f)
    f = click.option("--threads", type=int, default=4, help="CPU BLAS threads when device=cpu.")(f)
    return f


def _register(fam: Family) -> None:
    @cli.command(name=fam.value)
    @_common
    def _cmd(base_model, dataset, output_repo, backend, epochs, lr, batch_size, lora_rank,
             max_train, max_eval, output_dir, push, cpu, device, threads):
        _run(TrainingConfig(
            family=fam, base_model=base_model, dataset=dataset, output_repo=output_repo,
            backend=Backend(backend), epochs=epochs, lr=lr, batch_size=batch_size,
            lora_rank=lora_rank,
            extra={
                "max_train": max_train, "max_eval": max_eval, "output_dir": output_dir,
                "push": push, "cpu": cpu, "device": device, "threads": threads,
            },
        ))


for _fam in Family:
    _register(_fam)


if __name__ == "__main__":
    cli()
