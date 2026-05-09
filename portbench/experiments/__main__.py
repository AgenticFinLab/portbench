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
        help="Print the (model,profile,scenario) matrix without running",
    )
    p.add_argument("--batch-id", default=None, help="Override batch_id from YAML")
    p.add_argument(
        "--analyze",
        action="store_true",
        help="Run post-batch analysis on an existing batch_id (requires --batch-id)",
    )
    p.add_argument(
        "--analyze-qa",
        action="store_true",
        help="Run QA evaluation analysis on an existing batch_id (requires --batch-id)",
    )
    p.add_argument(
        "--output-root",
        default="EXPERIMENTS",
        help="Root directory for experiment outputs (default: EXPERIMENTS)",
    )
    args = p.parse_args(argv)

    if args.analyze:
        if not args.batch_id:
            p.error("--analyze requires --batch-id")
        from .analysis import analyze_batch
        report = analyze_batch(args.batch_id, output_root=args.output_root)
        print(f"Analysis report written to: {report}")
        return 0

    if args.analyze_qa:
        if not args.batch_id:
            p.error("--analyze-qa requires --batch-id")
        from ..qa_eval.analysis import analyze_qa_batch
        report = analyze_qa_batch(args.batch_id, output_root=args.output_root)
        print(f"QA analysis report written to: {report}")
        return 0

    if not args.config:
        p.error("--config is required unless --analyze or --analyze-qa is set")

    cfg_path = Path(args.config)
    raw_yaml = cfg_path.read_text(encoding="utf-8")
    cfg = ExperimentConfig.from_yaml(cfg_path)
    if args.batch_id:
        cfg.batch_id = args.batch_id

    runner = BatchRunner(cfg, raw_yaml=raw_yaml)

    if args.dry_run:
        matrix = runner.dry_run()
        print(json.dumps(matrix, indent=2))
        print(f"\nTotal experiments: {len(matrix)}")
        return 0

    summary = runner.run()
    print("\n" + "=" * 60)
    print(f"Batch '{cfg.batch_id}' complete")
    print(f"  Completed: {summary['n_completed']}")
    print(f"  Failed:    {summary['n_failed']}")
    print(f"  Elapsed:   {summary['elapsed_seconds']}s")
    print(f"  Output:    {cfg.output_root}/{cfg.batch_id}/")
    return 0 if summary["n_failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
