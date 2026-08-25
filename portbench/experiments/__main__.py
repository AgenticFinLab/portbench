"""CLI entry: python -m portbench.experiments --config <path.yaml>"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from .config import ExperimentConfig
from .runner import BatchRunner


def _resolve_output_root(args) -> str:
    """Return the effective output_root, respecting experiment_tag from config."""
    if args.config:
        try:
            cfg = ExperimentConfig.from_yaml(args.config)
            return cfg.output_root  # already includes experiment_tag via __post_init__
        except Exception:
            pass  # fall through to CLI default
    return args.output_root


def _filter_models_by_provider(cfg: ExperimentConfig, providers: str | None) -> tuple[str, ...]:
    """Restrict one batch invocation to configured LLM providers."""
    if not providers:
        return ()
    selected = tuple(part.strip().lower() for part in providers.split(",") if part.strip())
    if not selected:
        raise ValueError("--providers must name at least one provider")
    available = {spec.provider for spec in cfg.models if spec.provider}
    unknown = sorted(set(selected) - available)
    if unknown:
        raise ValueError(f"--providers contains unconfigured providers: {', '.join(unknown)}")
    cfg.models = [spec for spec in cfg.models if spec.provider in selected]
    return selected


def _filtered_config_snapshot(raw_yaml: str, cfg: ExperimentConfig, providers: tuple[str, ...]) -> str:
    """Record the effective provider-filtered model list for one batch run."""
    if not providers:
        return raw_yaml
    snapshot = yaml.safe_load(raw_yaml)
    snapshot["models"] = [asdict(spec) for spec in cfg.models]
    snapshot["selected_providers"] = list(providers)
    return yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="PortBench batch experiment runner")
    p.add_argument("--config", default=None, help="Path to YAML experiment config")
    p.add_argument(
        "--providers",
        default=None,
        help="Comma-separated configured providers to run in this invocation",
    )
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
        "--analyze-qa",
        action="store_true",
        help="Regenerate QA analysis figures and report from existing qa_summary.json.",
    )
    p.add_argument(
        "--analyze-qa-info-level",
        action="store_true",
        help=(
            "Compare full-info vs restricted-info QA accuracy for T4/T5. "
            "Reads T4/T4_restricted and T5/T5_restricted results from EXPERIMENTS/qa_eval/. "
            "Generates comparison figures and info_level_comparison.json. No LLM calls required."
        ),
    )
    p.add_argument(
        "--lambda-sweep",
        action="store_true",
        help=(
            "Run CEPS λ sensitivity analysis: recompute CEPS at multiple "
            "propagation_weight values (0, 0.05, 0.1, 0.2, 0.5) from existing "
            "pipeline_logs. Reports ranking stability. No LLM calls."
        ),
    )
    p.add_argument(
        "--analyze-causal",
        action="store_true",
        help="Aggregate completed SA v4 online repair logs without making LLM calls.",
    )
    p.add_argument(
        "--causal-input",
        default=None,
        help="Rebalance directory containing causal pipeline logs.",
    )
    p.add_argument(
        "--causal-output",
        default=None,
        help="Directory for causal attribution JSON and heatmap artifacts.",
    )
    p.add_argument(
        "--causal-bootstrap",
        type=int,
        default=10_000,
        help="Moving-block bootstrap draws for --analyze-causal.",
    )
    p.add_argument(
        "--analyze-qa-v2",
        action="store_true",
        help="Aggregate locked constraint-v2 QA results with item-bootstrap intervals.",
    )
    p.add_argument(
        "--qa-v2-input",
        default=None,
        help="Experiment output root containing qa_eval results for --analyze-qa-v2.",
    )
    p.add_argument(
        "--qa-v2-output",
        default=None,
        help="Directory for constraint-v2 QA statistics and figures.",
    )
    p.add_argument(
        "--qa-v2-bootstrap",
        type=int,
        default=10_000,
        help="Item-bootstrap draws for --analyze-qa-v2.",
    )
    p.add_argument(
        "--export-paper-artifacts",
        action="store_true",
        help="Export LaTeX rows and a numeric manifest from frozen causal and QA summaries.",
    )
    p.add_argument("--paper-causal-summary", default=None)
    p.add_argument("--paper-qa-summary", default=None)
    p.add_argument("--paper-output", default=None)
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
            output_root=_resolve_output_root(args),
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
            output_root=_resolve_output_root(args),
            config_path=args.config,
        )
        print(f"σ ablation complete: {results_path}")
        return 0

    if getattr(args, "analyze_qa", False):
        from ..qa_eval.analysis import analyze_qa_results

        report = analyze_qa_results(output_root=_resolve_output_root(args))
        print(f"QA analysis complete: {report}")
        return 0

    if getattr(args, "analyze_qa_info_level", False):
        from ..qa_eval.analysis import analyze_qa_info_level_comparison

        report = analyze_qa_info_level_comparison(output_root=_resolve_output_root(args))
        print(f"QA info-level comparison complete: {report}")
        return 0

    if getattr(args, "lambda_sweep", False):
        from .rescore import rescore_lambda_sweep

        result = rescore_lambda_sweep(
            rebalance=args.rebalance,
            output_root=_resolve_output_root(args),
            config_path=args.config,
        )
        print(f"λ sweep complete: {result}")
        return 0

    if getattr(args, "analyze_causal", False):
        from .causal_analysis import (
            load_online_repair_records,
            summarize_causal_attribution,
            write_causal_artifacts,
        )

        if args.causal_input:
            input_root = Path(args.causal_input)
        elif args.config:
            config = ExperimentConfig.from_yaml(args.config)
            input_root = Path(config.output_root) / config.rebalance
        else:
            p.error("--analyze-causal requires --causal-input or --config")
        output_root = (
            Path(args.causal_output)
            if args.causal_output
            else input_root / "causal_attribution"
        )
        records = load_online_repair_records(input_root)
        summary = summarize_causal_attribution(
            records,
            n_bootstrap=args.causal_bootstrap,
            seed=42,
            block_size=3,
        )
        result = write_causal_artifacts(summary, output_root)
        print(f"Causal attribution complete: {result}")
        return 0

    if getattr(args, "analyze_qa_v2", False):
        from ..qa_eval.constraint_analysis import (
            load_constraint_v2_records,
            summarize_constraint_v2,
            write_constraint_v2_artifacts,
        )

        if args.qa_v2_input:
            input_root = Path(args.qa_v2_input)
        elif args.config:
            input_root = Path(ExperimentConfig.from_yaml(args.config).output_root)
        else:
            p.error("--analyze-qa-v2 requires --qa-v2-input or --config")
        output_root = Path(args.qa_v2_output) if args.qa_v2_output else input_root / "qa_eval" / "constraint_v2_analysis"
        result = write_constraint_v2_artifacts(
            summarize_constraint_v2(
                load_constraint_v2_records(input_root),
                n_bootstrap=args.qa_v2_bootstrap,
                seed=42,
            ),
            output_root,
        )
        print(f"Constraint-v2 QA analysis complete: {result}")
        return 0

    if getattr(args, "export_paper_artifacts", False):
        from .paper_export import export_paper_artifacts

        if not (args.paper_causal_summary and args.paper_qa_summary and args.paper_output):
            p.error(
                "--export-paper-artifacts requires --paper-causal-summary, "
                "--paper-qa-summary, and --paper-output"
            )
        result = export_paper_artifacts(
            causal_summary_path=args.paper_causal_summary,
            qa_summary_path=args.paper_qa_summary,
            output_dir=args.paper_output,
        )
        print(f"Paper artifacts exported: {result}")
        return 0

    if not args.config:
        p.error(
            "--config is required (or use --rescore / --sigma-ablation / "
            "--analyze-qa / --analyze-qa-info-level to recompute without LLM calls)"
        )

    cfg_path = Path(args.config)
    raw_yaml = cfg_path.read_text(encoding="utf-8")
    cfg = ExperimentConfig.from_yaml(cfg_path)
    try:
        selected_providers = _filter_models_by_provider(cfg, args.providers)
    except ValueError as exc:
        p.error(str(exc))
    raw_yaml = _filtered_config_snapshot(raw_yaml, cfg, selected_providers)

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
