"""Training configuration: model families + device-agnostic backend.

The compute model is **MLX-first, burst to DigitalOcean GPU**. Training code must be
device-agnostic: a single `backend` flag selects MLX (Mac Studio) vs CUDA (DO droplet);
datasets/checkpoints sync via HF Hub so a droplet is stateless and disposable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml


class Backend(str, Enum):
    MLX = "mlx"      # Apple Silicon / Mac Studio — the default
    CUDA = "cuda"    # DigitalOcean GPU droplet — burst for heavy jobs


class Family(str, Enum):
    MOE_FINETUNE = "moe-finetune"  # continue-finetune openai/privacy-filter (primary)
    XLMR_NER = "xlmr-ner"          # XLM-R / mDeBERTa token classification
    GLINER = "gliner"              # GLiNER-style extensible/zero-shot
    ANON_LORA = "anon-lora"        # LoRA-tuned small LLM anonymizer/pseudonymizer
    CLASSIFIER = "classifier"      # document-level privacy/sensitivity classifier


@dataclass
class TrainingConfig:
    family: Family
    base_model: str                       # HF id of the base, e.g. openai/privacy-filter
    dataset: str                          # HF id, e.g. klusai/ds-kp-legal-ro-50k
    output_repo: str                      # HF target, e.g. klusai/kp-deid-moe-ro
    backend: Backend = Backend.MLX
    epochs: int = 5
    lr: float = 3e-4
    batch_size: int = 16
    seed: int = 0
    lora_rank: int | None = None          # set for anon-lora / parameter-efficient runs
    extra: dict = field(default_factory=dict)

    def publish_id(self) -> str:
        """MLX runs publish an extra `-mlx` variant (mirrors tf3-50m-base-mlx)."""
        return f"{self.output_repo}-mlx" if self.backend is Backend.MLX else self.output_repo


def load_registry(path: str | Path = "conf/models.yaml") -> dict:
    """Load the model-family registry (families → variants)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
