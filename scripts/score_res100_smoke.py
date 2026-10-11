#!/usr/bin/env python3
"""Load a merged checkpoint via the actual europriv adapter and score held-out KP rows."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click


@click.command()
@click.option("--model", required=True)
@click.option("--rows", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--limit", type=click.IntRange(min=1), default=16)
def main(model, rows, out, limit):
    os.environ["EUROPRIV_DEVICE"] = "cpu"
    os.environ["EUROPRIV_BATCH_SIZE"] = "4"
    import numpy as np
    import torch

    from europriv_bench.adapters import KpModelAdapter
    from europriv_bench.metrics import entity_f1
    from europriv_bench.runner import _rows_to_gold, run_spec
    from europriv_bench.spec import EvalSpec

    torch.set_num_threads(4)
    data = [json.loads(line) for line in rows.read_text().splitlines()][:limit]
    spec = EvalSpec.model_validate(
        {
            "name": "RES-100 held-out KP smoke (not a quality evaluation)",
            "version": 1,
            "task": "detection",
            "domain": "general",
            "languages": sorted({r["language"] for r in data}),
            "dataset": {
                "hf_id": "local/res100-smoke",
                "config": "res100-heldout-smoke",
                "split": "test",
                "license": "CC-BY-4.0",
            },
            "metrics": ["entity_f1", "national_id_leakage"],
        }
    )

    class RecordedAdapter(KpModelAdapter):
        def predict_tags(self, texts):
            self.predictions = super().predict_tags(texts)
            return self.predictions

    adapter = RecordedAdapter(model)
    result = run_spec(spec, adapter, rows=data, limit=limit)
    _, gold = _rows_to_gold(data)
    eval_labels = set(result["eval_labels"])
    predictions = [
        [tag if tag == "O" or tag.split("-", 1)[1] in eval_labels else "O" for tag in seq]
        for seq in adapter.predictions
    ]
    rng = np.random.default_rng(0)
    bootstrap = []
    for _ in range(1000):
        indices = rng.integers(0, len(data), len(data))
        bootstrap.append(entity_f1([gold[i] for i in indices], [predictions[i] for i in indices])["f1"])
    result["entity_f1_bootstrap"] = {
        "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "resamples": 1000,
        "seed": 0,
        "unit": "document",
    }
    result.update(
        {
            "seed_count": 1,
            "training_seed": 0,
            "quality_claim": None,
            "scope": "checkpoint/adapter smoke only; tiny held-out KP carve, no TAB score",
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
