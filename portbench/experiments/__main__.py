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
        "--rescore",
        action="store_true",
        help=(
            "Recompute CEPS scores (no LLM calls), then regenerate all comparison "
            "figures and analysis report. Use after changing S3 GT or evaluation logic."
        ),
    )
    p.add_argument(
        "--sigma-ablation",
        action="store_true",
        help=(
            "Run σ ablation: rescore S3 for σ ∈ sigma_ablation_values from config "
            "(default [0.0, 0.25, 0.5, 0.75, 1.0]). Writes results.json + sigma_ablation.png "
            "to EXPERIMENTS/{rebalance}/sigma_ablation/. No LLM re-calls required."
        ),
    )
    p.add_argument(
        "--rebalance",
        default="monthly",
        help="Rebalance frequency directory (default: monthly)",
    )
    p.add_argument(
        "--output-root",
        default="EXPERIMENTS",
        help="Root directory for experiment outputs (default: EXPERIMENTS)",
    )
    args = p.parse_args(argv)

    if args.rescore:
        from .rescore import rescore_ceps

        result = rescore_ceps(
            rebalance=args.rebalance,
            output_root=args.output_root,
            config_path=args.config,
        )
        print(f"Rescore + figure regeneration complete: {result}")
        return 0

    if getattr(args, "sigma_ablation", False):
        from .rescore import rescore_sigma_ablation

        sigma_values = None
        if args.config:
            cfg = ExperimentConfig.from_yaml(args.config)
            sigma_values = cfg.sigma_ablation_values
        results_path = rescore_sigma_ablation(
            sigma_values=sigma_values,
            rebalance=args.rebalance,
            output_root=args.output_root,
            config_path=args.config,
        )
        print(f"σ ablation complete: {results_path}")
        return 0

    if not args.config:
        p.error(
            "--config is required (or use --rescore / --sigma-ablation to recompute without LLM calls)"
        )

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
    print(f"  Resumed:   {summary['n_resumed']}")
    print(f"  Failed:    {summary['n_failed']}")
    print(f"  Output:    {cfg.output_root}/{cfg.rebalance}/")
    return 0 if summary["n_failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
