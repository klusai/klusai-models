#!/usr/bin/env python3
"""Unified training CLI for all KP model families (mirrors diacritics-finetuning train.py).

Run from the repo root after `make install`:

    python scripts/train.py moe-finetune --base openai/privacy-filter \\
        --dataset klusai/ds-kp-legal-ro-50k --out klusai/kp-deid-moe-ro --backend mlx

Backend is device-agnostic: `--backend mlx` (Mac Studio) or `--backend cuda` (DO GPU droplet).
Datasets/checkpoints sync via HF Hub so a droplet stays stateless.
"""

from __future__ import annotations

import click

from klusai.privacy.models.logger import get_logger
from klusai.privacy.models.training.config import Backend, Family, TrainingConfig

logger = get_logger("train")


def _run(cfg: TrainingConfig) -> None:
    logger.info("family=%s base=%s backend=%s → %s", cfg.family.value, cfg.base_model, cfg.backend.value, cfg.publish_id())
    raise NotImplementedError(
        f"training for {cfg.family.value} lands in Phase 3 — dispatch to the {cfg.backend.value} backend "
        f"(transformers+peft/trl for cuda, mlx-lm for mlx), then push_to_hub({cfg.publish_id()})"
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
    return f


def _register(fam: Family) -> None:
    @cli.command(name=fam.value)
    @_common
    def _cmd(base_model, dataset, output_repo, backend, epochs):
        _run(TrainingConfig(
            family=fam, base_model=base_model, dataset=dataset, output_repo=output_repo,
            backend=Backend(backend), epochs=epochs,
        ))


for _fam in Family:
    _register(_fam)


if __name__ == "__main__":
    cli()
