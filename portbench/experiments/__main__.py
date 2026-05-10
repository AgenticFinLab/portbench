"""CLI entry: python -m portbench.experiments --config <path.yaml>"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ExperimentConfig
from .runner import BatchRunner


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="PortBench batch experiment runner")
    p.add_argument("--config", default=None, help="Path to YAML experiment config")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the (provider, model, profile, scenario) matrix without running",
    )
    p.add_argument(
        "--analyze",
        action="store_true",
        help="Run post-run analysis on a rebalance directory (requires --rebalance)",
    )
    p.add_argument(
        "--rebalance",
        default="monthly",
        help="Rebalance frequency directory to analyze (default: monthly)",
    )
    p.add_argument(
        "--output-root",
        default="EXPERIMENTS",
        help="Root directory for experiment outputs (default: EXPERIMENTS)",
    )
    args = p.parse_args(argv)

    if args.analyze:
        from .analysis import analyze_runs
        report = analyze_runs(args.rebalance, output_root=args.output_root)
        print(f"Analysis report written to: {report}")
        return 0

    if not args.config:
        p.error("--config is required unless --analyze is set")

    cfg_path = Path(args.config)
    raw_yaml = cfg_path.read_text(encoding="utf-8")
    cfg = ExperimentConfig.from_yaml(cfg_path)

    runner = BatchRunner(cfg, raw_yaml=raw_yaml)

    if args.dry_run:
        matrix = runner.dry_run()
        print(json.dumps(matrix, indent=2))
        print(f"\nTotal experiments: {len(matrix)}")
        return 0

    summary = runner.run()
    print("\n" + "=" * 60)
    print(f"Batch complete [{cfg.rebalance}]")
    print(f"  Completed: {summary['n_completed']}")
    print(f"  Reused:    {summary['n_reused']}")
    print(f"  Failed:    {summary['n_failed']}")
    print(f"  Output:    {cfg.output_root}/{cfg.rebalance}/")
    return 0 if summary["n_failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
