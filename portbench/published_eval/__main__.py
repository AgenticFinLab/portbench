"""Replay the published evaluation from frozen decisions.

Examples:
    python -m portbench.published_eval replay --source-root .
    python -m portbench.published_eval classical --source-root .
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .classical import run_classical_baselines
from .replay import run_saved_allocations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=("replay", "classical"),
        help="Saved-decision replay, or the five classical baselines on that replay",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Checkout that holds datasets/ and the frozen EXPERIMENTS_SA_UPGRADE index",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Table directory (default: <source-root>/outputs/portfolio_evaluation)",
    )
    args = parser.parse_args(argv)
    output = args.output_dir or (args.source_root / "outputs" / "portfolio_evaluation")
    if args.step == "replay":
        run_saved_allocations(args.source_root, output)
    else:
        run_classical_baselines(args.source_root, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
