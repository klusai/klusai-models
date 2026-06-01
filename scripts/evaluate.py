#!/usr/bin/env python3
"""Evaluate a KP model on EuroPriv-Bench.

Run from the repo root after `make install`. Thin wrapper: defers to the `europriv-bench`
harness (the single source of truth for scoring) via its ``kp-model`` adapter, so results are
directly comparable to the public leaderboard. Metrics are NEVER re-implemented here.

    python scripts/evaluate.py --model klusai/kp-deid-mdeberta-280m \\
        --suite ../europriv-bench/evaluations --only pii-detection-ro-realskeleton \\
        --out runs/leaderboard-kp.json
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click

from klusai.privacy.models.logger import get_logger

logger = get_logger("evaluate")


@click.command()
@click.option("--model", required=True, help="HF id (or local dir) of the KP model to score.")
@click.option("--suite", default="../europriv-bench/evaluations", help="europriv-bench eval suite directory.")
@click.option("--only", default=None, help="Substring filter on the spec yaml filename (e.g. ro-realskeleton).")
@click.option("--out", default="runs/leaderboard-kp.json", help="Leaderboard output path.")
@click.option("--limit", type=int, default=None, help="Cap examples per spec (fast iteration).")
@click.option("--threads", type=int, default=4, help="BLAS threads (4 is fastest for these models on CPU).")
def main(model: str, suite: str, only: str | None, out: str, limit: int | None, threads: int) -> None:
    # Single source of truth: the harness owns specs, scoring, provenance, and the leaderboard schema.
    from europriv_bench.adapters import KpModelAdapter
    from europriv_bench.leaderboard import format_leaderboard, write_leaderboard
    from europriv_bench.runner import run_spec
    from europriv_bench.spec import EvalSpec

    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass

    specs_dir = Path(suite)
    spec_paths = sorted(specs_dir.glob("*.yaml"))
    if only:
        spec_paths = [p for p in spec_paths if only in p.name]
    if not spec_paths:
        raise click.UsageError(f"no specs matched in {suite!r} (filter={only!r})")

    adapter = KpModelAdapter(model_id=model)
    ts = datetime.now(timezone.utc).isoformat()
    results = []
    for path in spec_paths:
        spec = EvalSpec.from_yaml(path)
        logger.info("scoring %s on %s", model, spec.name)
        results.append(run_spec(spec, adapter, timestamp=ts, limit=limit))

    written = write_leaderboard(results, out)
    click.echo(f"wrote {written}")
    click.echo(format_leaderboard({"schema": 2, "entries": {f"kp-model::{model}": results}}))


if __name__ == "__main__":
    main()
