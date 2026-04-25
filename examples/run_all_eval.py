"""
Unified PortBench evaluation entry point.

Two evaluation tiers:
  qa      — Tier 1: static QA dataset accuracy (T1-T7 templates)
  sandbox — Tier 2: stateful Sandbox × 3 Profiles (stress gate + normal backtest)

Usage:
    # Both tiers, mock agent (no API keys needed)
    python examples/run_all_eval.py

    # Specific tier(s)
    python examples/run_all_eval.py --eval qa
    python examples/run_all_eval.py --eval sandbox

    # Real LLM
    python examples/run_all_eval.py --model qwen:qwen-plus

    # Baseline
    python examples/run_all_eval.py --baseline equal_weight

Results:
    outputs/qa/{model}/{timestamp}/qa_results.json
    outputs/sandbox/{model}/{timestamp}/{profile}/normal/backtest_result.json
    outputs/sandbox/{model}/{timestamp}/profile_comparison.json
"""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.agent_eval.run_qa_eval import run_qa_evaluation
from examples.sandbox.run_backtest import run_backtest


def parse_args():
    parser = argparse.ArgumentParser(description="PortBench unified evaluation runner")

    parser.add_argument("--eval", nargs="+",
                        choices=["qa", "sandbox"],
                        default=["qa", "sandbox"],
                        help="Which evaluation tiers to run (default: both)")

    # Model / adapter selection
    parser.add_argument("--model", type=str, default=None,
                        help="'<provider>:<model_id>' e.g. 'anthropic:claude-opus-4-7'")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["equal_weight", "sixty_forty", "risk_parity"])
    parser.add_argument("--local-model", type=str, default=None)
    parser.add_argument("--local-url", type=str, default=None)

    # Common settings
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-provider", type=str, default="processed",
                        choices=["mock", "processed"])
    parser.add_argument("--data-dir", type=str, default="datasets/processed")
    parser.add_argument("--sec-dir", type=str, default="datasets/sec")

    # QA-specific
    parser.add_argument("--qa-path", type=str, default="datasets/qa_dataset/test.jsonl")
    parser.add_argument("--templates", nargs="+", default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    # Sandbox-specific
    parser.add_argument("--profiles", nargs="+",
                        choices=["conservative", "balanced", "aggressive"],
                        default=["conservative", "balanced", "aggressive"])
    parser.add_argument("--rebalance", type=str, default="monthly",
                        choices=["weekly", "monthly", "quarterly"])
    parser.add_argument("--initial-nav", type=float, default=1_000_000.0)
    parser.add_argument("--no-pipeline", action="store_true")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel threads: QA uses this for LLM calls, Sandbox for profiles+stress")

    return parser.parse_args()


def main():
    args = parse_args()
    evals = args.eval

    print("PortBench Unified Evaluation")
    print(f"  Tiers  : {evals}")
    model_str = args.model or (f"baseline:{args.baseline}" if args.baseline else "mock_agent")
    print(f"  Model  : {model_str}")
    print("=" * 60)

    if "qa" in evals:
        print("\n[Tier 1] QA evaluation…")
        qa_args = Namespace(
            model=args.model,
            noise=args.noise,
            seed=args.seed,
            qa_path=args.qa_path,
            templates=args.templates,
            max_samples=args.max_samples,
            workers=args.workers,
        )
        run_qa_evaluation(qa_args)

    if "sandbox" in evals:
        print("\n[Tier 2] Sandbox × Profile evaluation…")
        sandbox_args = Namespace(
            model=args.model,
            baseline=args.baseline,
            no_pipeline=args.no_pipeline,
            profiles=args.profiles,
            rebalance=args.rebalance,
            initial_nav=args.initial_nav,
            noise=args.noise,
            seed=args.seed,
            data_provider=args.data_provider,
            data_dir=args.data_dir,
            sec_dir=args.sec_dir,
            workers=args.workers,
        )
        run_backtest(sandbox_args)

    print("\n" + "=" * 60)
    print("All evaluations complete.")


if __name__ == "__main__":
    main()
