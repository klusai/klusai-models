"""klusai.privacy.models — finetuning + evaluation for KP privacy models.

Primary track: continue-finetune `openai/privacy-filter` (proven by OpenMed, MLX-friendly).
Comparison tracks: XLM-R/mDeBERTa NER, GLiNER, an anonymizer LLM (LoRA), a sensitivity
classifier. Models publish to `klusai/kp-{task}-{base}-{size}[-variant]`. Label maps come from
`europriv_bench.taxonomy` (shared source of truth) — never hardcoded here.
"""

__version__ = "0.1.0"
