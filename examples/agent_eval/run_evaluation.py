"""
Run end-to-end agent evaluation on the PortBench pipeline.

Usage:
    # Evaluate the mock agent (no API keys needed)
    python examples/agent_eval/run_evaluation.py

    # Evaluate with a specific noise level
    python examples/agent_eval/run_evaluation.py --noise 0.3

    # Run stress tests only
    python examples/agent_eval/run_evaluation.py --stress-only

Output:
    outputs/eval_results/{model_name}/
        per_stage_scores.json    — per-stage scores across all episodes
        ceps_scores.json         — CEPS metric with propagation penalty
        stress_test_results.json — pass/fail for each stress scenario
        risk_first_ranking.json  — final ranking (stress-gated performance)
        summary.txt              — human-readable summary

The evaluation workflow:
  1. Stress gate:  Agent must pass all three stress scenarios (2008, 2020, 2022).
  2. Performance:  CEPS computed over normal market episodes (2024 test set).
  3. Risk-first ranking: Models that failed stress gate are shown but marked FAILED.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.agent_eval import build_default_pipeline, STRESS_SCENARIOS
from portbench.agent_eval.base import MarketSnapshot, StageID
from portbench.agent_eval.mock_agent import MockAgentAdapter
from portbench.agent_eval.stress_scenarios import ScenarioInjector
from portbench.baselines import EqualWeightBaseline, SixtyFortyBaseline, RiskParityBaseline
from portbench.metrics.ceps import CEPS
from portbench.qa_builder.mock_data import MockDataProvider


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="PortBench agent evaluation")
    parser.add_argument("--noise", type=float, default=0.2,
                        help="Mock agent noise level (0=perfect, 1=random)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--n-episodes", type=int, default=20,
                        help="Number of normal-market evaluation episodes")
    parser.add_argument("--stress-only", action="store_true",
                        help="Run only stress tests, skip normal-market evaluation")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["equal_weight", "sixty_forty", "risk_parity"],
                        help="Evaluate a baseline instead of the mock LLM agent")

    # Local model options
    local = parser.add_argument_group("Local model options")
    local.add_argument("--local-model", type=str, default=None,
                       help="Use a local model instead of the mock agent. "
                            "Format: '<backend>:<model>', e.g. "
                            "'vllm:meta-llama/Llama-3.1-8B-Instruct', "
                            "'ollama:llama3.1', "
                            "'hf:microsoft/Phi-3-mini-4k-instruct'")
    local.add_argument("--local-url", type=str, default=None,
                       help="Base URL for vLLM server (default: http://localhost:8000/v1)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Snapshot generation from mock data
# ---------------------------------------------------------------------------

def build_snapshots(provider: MockDataProvider, n: int, seed: int = 0) -> list[MarketSnapshot]:
    """Generate n MarketSnapshot objects from test-period dates (2024)."""
    test_dates = pd.bdate_range("2024-01-01", "2024-12-31").date.tolist()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(test_dates, size=min(n, len(test_dates)), replace=False)
    sampled = sorted(sampled.tolist())

    assets = provider.list_assets()
    snapshots = []

    for d in sampled:
        from datetime import timedelta
        lookback_start = d - timedelta(days=90)
        price_data, return_data = {}, {}

        for asset in assets:
            try:
                prices = provider.get_price_series(asset, lookback_start, d)
                returns = provider.get_return_series(asset, lookback_start, d)
                price_data[asset] = prices
                return_data[asset] = returns
            except Exception:
                continue

        macro = provider.get_macro(d)
        regime = provider.get_regime(d).value
        n_assets = len(assets)
        current_weights = {a: round(1.0 / n_assets, 4) for a in assets}

        snapshots.append(MarketSnapshot(
            decision_date=d,
            price_data=price_data,
            return_data=return_data,
            macro_data=macro,
            current_weights=current_weights,
            market_regime=regime,
        ))

    return snapshots


# ---------------------------------------------------------------------------
# Local adapter factory
# ---------------------------------------------------------------------------

def _build_local_adapter(spec: str, url: Optional[str] = None):
    """
    Parse a '<backend>:<model>' string and return the appropriate local adapter.

    Examples:
        "vllm:meta-llama/Llama-3.1-8B-Instruct"
        "ollama:llama3.1"
        "hf:microsoft/Phi-3-mini-4k-instruct"
    """
    from portbench.agent_eval.local_adapter import VLLMAdapter, OllamaAdapter, HuggingFaceAdapter

    if ":" not in spec:
        raise ValueError(
            f"--local-model must be '<backend>:<model>', got: {spec!r}\n"
            "Examples: 'vllm:meta-llama/Llama-3.1-8B-Instruct', 'ollama:llama3.1', "
            "'hf:microsoft/Phi-3-mini-4k-instruct'"
        )

    backend, model = spec.split(":", 1)
    backend = backend.lower()

    if backend == "vllm":
        kwargs = {}
        if url:
            kwargs["base_url"] = url
        return VLLMAdapter(model=model, **kwargs)
    elif backend == "ollama":
        kwargs = {}
        if url:
            kwargs["host"] = url
        return OllamaAdapter(model=model, **kwargs)
    elif backend in ("hf", "huggingface", "transformers"):
        return HuggingFaceAdapter(model_name=model)
    else:
        raise ValueError(
            f"Unknown local backend: {backend!r}. Supported: vllm, ollama, hf"
        )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args):
    # Select adapter
    if args.baseline == "equal_weight":
        adapter = EqualWeightBaseline()
    elif args.baseline == "sixty_forty":
        adapter = SixtyFortyBaseline()
    elif args.baseline == "risk_parity":
        adapter = RiskParityBaseline()
    elif args.local_model:
        adapter = _build_local_adapter(args.local_model, args.local_url)
    else:
        adapter = MockAgentAdapter(noise_level=args.noise, seed=args.seed)

    model_name = adapter.model_name
    output_dir = Path(f"outputs/eval_results/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PortBench Evaluation — Model: {model_name}")
    print("=" * 60)

    pipeline = build_default_pipeline(adapter)
    provider = MockDataProvider(seed=args.seed)

    # -----------------------------------------------------------------------
    # Phase 1: Stress tests (risk gate)
    # -----------------------------------------------------------------------
    print("\n[Phase 1] Stress Testing (Risk Gate)")
    print("-" * 40)

    injector = ScenarioInjector(provider=provider, assets=provider.list_assets(), lookback_days=60)
    stress_results = []
    all_passed = True

    for scenario in STRESS_SCENARIOS:
        result = injector.run_stress_test(scenario, pipeline, step_days=10)
        stress_results.append(result)
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"  {scenario.name}: {status} (CEPS={result['mean_ceps']:.3f}, threshold={result['min_pass_score']:.2f})")
        if not result["passed"]:
            all_passed = False

    # Save stress results
    with open(output_dir / "stress_test_results.json", "w") as f:
        json.dump(stress_results, f, indent=2, default=str)

    if not all_passed:
        print("\n  WARNING: Model failed one or more stress scenarios.")
        print("  Model will appear in ranking but marked as RISK_GATE_FAILED.")

    if args.stress_only:
        print(f"\nResults saved to {output_dir}")
        return

    # -----------------------------------------------------------------------
    # Phase 2: Normal-market evaluation
    # -----------------------------------------------------------------------
    print(f"\n[Phase 2] Normal-Market Evaluation ({args.n_episodes} episodes)")
    print("-" * 40)

    snapshots = build_snapshots(provider, n=args.n_episodes, seed=args.seed)
    episode_results = []
    stage_score_lists = []

    for snap in snapshots:
        result = pipeline.run_episode(snap)
        episode_results.append(result)
        stage_score_lists.append(result.to_stage_score_list())

    # Per-stage scores
    ordered_ids = [
        StageID.S1_MARKET_INTERPRETATION,
        StageID.S2_SIGNAL_GENERATION,
        StageID.S3_WEIGHT_OPTIMIZATION,
        StageID.S4_EXECUTION_SIMULATION,
        StageID.S5_RISK_MONITORING,
    ]
    per_stage = {
        sid.value: [r.stage_scores.get(sid, 0.0) for r in episode_results]
        for sid in ordered_ids
    }
    per_stage_means = {k: float(np.mean(v)) for k, v in per_stage.items()}

    with open(output_dir / "per_stage_scores.json", "w") as f:
        json.dump({"per_episode": per_stage, "mean": per_stage_means}, f, indent=2)

    # CEPS
    ceps_obj = CEPS()
    ceps_result = ceps_obj.compute_batch(stage_score_lists)

    with open(output_dir / "ceps_scores.json", "w") as f:
        json.dump(ceps_result, f, indent=2, default=str)

    print(f"  Mean CEPS: {ceps_result['mean_ceps']:.4f}")
    print(f"  Std  CEPS: {ceps_result['std_ceps']:.4f}")
    for sid_val, mean in ceps_result["per_stage_mean"].items():
        print(f"    {sid_val}: {mean:.4f}")

    # -----------------------------------------------------------------------
    # Phase 3: Risk-first ranking record
    # -----------------------------------------------------------------------
    ranking_entry = {
        "model_name": model_name,
        "risk_gate_passed": all_passed,
        "mean_ceps": ceps_result["mean_ceps"],
        "std_ceps": ceps_result["std_ceps"],
        "per_stage_mean": ceps_result["per_stage_mean"],
        "stress_results": stress_results,
        "n_episodes": ceps_result["n_episodes"],
    }

    with open(output_dir / "risk_first_ranking.json", "w") as f:
        json.dump(ranking_entry, f, indent=2, default=str)

    # -----------------------------------------------------------------------
    # Human-readable summary
    # -----------------------------------------------------------------------
    summary_lines = [
        f"PortBench Evaluation Summary",
        f"Model:          {model_name}",
        f"Risk Gate:      {'PASSED' if all_passed else 'FAILED'}",
        f"Mean CEPS:      {ceps_result['mean_ceps']:.4f}",
        f"Std  CEPS:      {ceps_result['std_ceps']:.4f}",
        f"Episodes:       {ceps_result['n_episodes']}",
        "",
        "Per-stage means:",
    ]
    for sid_val, mean in ceps_result["per_stage_mean"].items():
        summary_lines.append(f"  {sid_val}: {mean:.4f}")
    summary_lines += [
        "",
        "Stress test results:",
    ]
    for sr in stress_results:
        status = "PASSED" if sr["passed"] else "FAILED"
        summary_lines.append(f"  {sr['scenario_name']}: {status} (CEPS={sr['mean_ceps']:.3f})")

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary + "\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
