#!/usr/bin/env python3
"""Evaluate a KP model on EuroPriv-Bench.

Run from the repo root after `make install`. Thin wrapper: defers to the `europriv-bench`
harness (the single source of truth for scoring) via its adapter layer, so model results are
directly comparable to the public leaderboard. Never re-implement metrics here.
"""

import click

from klusai.privacy.models.logger import get_logger

logger = get_logger("evaluate")


@click.command()
@click.option("--model", required=True, help="HF id of the model/adapter to score.")
@click.option("--suite", default="evaluations", help="europriv-bench eval suite directory.")
def main(model: str, suite: str) -> None:
    raise NotImplementedError(
        "evaluate: Phase 1/3 — register `model` as a europriv-bench adapter and run "
        "`europriv run --adapter <name> --suite <suite>`; never re-implement metrics here."
    )


if __name__ == "__main__":
    main()
