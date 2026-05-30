"""Tests for training config + model registry + the shared-taxonomy contract."""

from europriv_bench.taxonomy import bioes_labels
from klusai.privacy.models.training.config import Backend, Family, TrainingConfig, load_registry


def test_mlx_publish_id_gets_mlx_suffix():
    cfg = TrainingConfig(
        family=Family.MOE_FINETUNE, base_model="openai/privacy-filter",
        dataset="klusai/ds-kp-legal-ro-50k", output_repo="klusai/kp-deid-moe-ro",
        backend=Backend.MLX,
    )
    assert cfg.publish_id() == "klusai/kp-deid-moe-ro-mlx"


def test_cuda_publish_id_unchanged():
    cfg = TrainingConfig(
        family=Family.XLMR_NER, base_model="FacebookAI/xlm-roberta-large",
        dataset="klusai/ds-kp-legal-ro-50k", output_repo="klusai/kp-deid-xlmr-560m",
        backend=Backend.CUDA,
    )
    assert cfg.publish_id() == "klusai/kp-deid-xlmr-560m"


def test_registry_has_primary_moe_track_and_baselines():
    reg = load_registry("conf/models.yaml")
    assert reg["moe-finetune"]["variants"][0]["base_model"] == "openai/privacy-filter"
    piiranha = next(b for b in reg["baselines"] if "piiranha" in b["name"])
    assert piiranha["role"] == "baseline-only"


def test_every_family_enum_has_registry_entry():
    reg = load_registry("conf/models.yaml")
    for fam in Family:
        assert fam.value in reg, f"{fam.value} missing from conf/models.yaml"


def test_shared_taxonomy_contract():
    # Model label maps must come from the benchmark's taxonomy, not a local copy.
    labels = bioes_labels()
    assert "O" in labels and "S-PERSON" in labels
