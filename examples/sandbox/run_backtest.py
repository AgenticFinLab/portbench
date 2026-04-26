"""
Run PortBench Sandbox backtest: stateful portfolio simulation with real PnL.

Evaluates a model (or baseline) across three investor risk profiles
(conservative / balanced / aggressive), each with:
  - Phase A: stress gate (3 scenarios; pass condition: max_drawdown ≤ profile tolerance)
  - Phase B: normal-market backtest (2024, only if stress gate passed)

Usage:
    # MockAgent (no API keys needed)
    python examples/sandbox/run_backtest.py

    # LLM model
    python examples/sandbox/run_backtest.py --model qwen:qwen-plus

    # Baseline strategy
    python examples/sandbox/run_backtest.py --baseline equal_weight

    # Single profile
    python examples/sandbox/run_backtest.py --profiles conservative

    # Custom rebalance frequency
    python examples/sandbox/run_backtest.py --model qwen:qwen-plus --rebalance monthly

Output:
    outputs/sandbox/{model_name}/{timestamp}/{profile}/
        stress_{scenario_name}/
            backtest_result.json    — PnL metrics + stress_passed flag
            summary.txt
        normal/                     — only written if stress gate passed
            nav_curve.csv
            weight_history.csv
            backtest_result.json    — PnL + mean_ceps + mean_profile_score
            summary.txt
    outputs/sandbox/{model_name}/{timestamp}/
        profile_comparison.json     — three profiles side-by-side + adaptation_score
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.agent_eval.llm_adapters import (
    AnthropicAdapter,
    OpenAIAdapter,
    LiteLLMAdapter,
)
from portbench.agent_eval.mock_agent import MockAgentAdapter
from portbench.agent_eval.investor_profiles import PROFILES
from portbench.agent_eval.stress_scenarios import STRESS_SCENARIOS
from portbench.baselines import (
    CovarianceRiskParityBaseline,
    EqualWeightBaseline,
    SixtyFortyBaseline,
    RiskParityBaseline,
)
from portbench.qa_builder.mock_data import MockDataProvider
from portbench.qa_builder.processed_data import ProcessedDataProvider
from portbench.sandbox import BacktestEngine


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="PortBench Sandbox backtest (Profile × Stress + Normal)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Cloud/API model: '<provider>:<model_id>', e.g. 'qwen:qwen-plus'",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=["equal_weight", "sixty_forty", "risk_parity", "cov_risk_parity"],
    )
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Skip S1-S5 pipeline; call allocate() directly (baselines)",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["conservative", "balanced", "aggressive"],
        default=["conservative", "balanced", "aggressive"],
    )
    parser.add_argument(
        "--rebalance",
        type=str,
        default="monthly",
        choices=["weekly", "monthly", "quarterly"],
    )
    parser.add_argument("--initial-nav", type=float, default=1_000_000.0)
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-provider", type=str, default="processed", choices=["mock", "processed"]
    )
    parser.add_argument("--data-dir", type=str, default="datasets/processed")
    parser.add_argument("--sec-dir", type=str, default="datasets/sec")
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Parallel threads across profiles and stress scenarios (default: 3)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

_OPENAI_COMPAT = {
    "qwen": ("DASHSCOPE_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "kimi": ("MOONSHOT_API_KEY", "https://api.moonshot.cn/v1"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    "tencent": ("TENCENT_API_KEY", "https://tokenhub.tencentmaas.com/v1"),
}


def _build_adapter(args):
    if args.baseline == "equal_weight":
        return EqualWeightBaseline()
    elif args.baseline == "sixty_forty":
        return SixtyFortyBaseline()
    elif args.baseline == "risk_parity":
        return RiskParityBaseline()
    elif args.baseline == "cov_risk_parity":
        return CovarianceRiskParityBaseline()
    elif args.model:
        if ":" not in args.model:
            raise ValueError(
                f"--model must be '<provider>:<model_id>', got: {args.model!r}"
            )
        provider, model = args.model.split(":", 1)
        provider = provider.lower()
        if provider == "anthropic":
            return AnthropicAdapter(model=model)
        elif provider == "openai":
            return OpenAIAdapter(model=model)
        elif provider in _OPENAI_COMPAT:
            key_env, base_url = _OPENAI_COMPAT[provider]
            if not os.environ.get(key_env):
                print(f"  WARNING: {key_env} is not set.")
            return OpenAIAdapter(model=model, base_url=base_url, api_key_env=key_env)
        else:
            from portbench.agent_eval.llm_adapters import LiteLLMAdapter

            return LiteLLMAdapter(model=f"{provider}/{model}")
    else:
        return MockAgentAdapter(noise_level=args.noise, seed=args.seed)


def _build_asset_class_map(provider) -> dict[str, str]:
    all_classes = [
        "equities",
        "bonds",
        "commodities",
        "real_estate",
        "cryptocurrency",
        "cash",
    ]
    result = {}
    for cls in all_classes:
        try:
            for a in provider.list_assets(cls):
                result[a] = cls
        except Exception:
            pass
    return result


def _save_result(result, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dict = result.to_dict()
    with open(out_dir / "backtest_result.json", "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(result.summary() + "\n")
    if (
        hasattr(result, "nav_curve")
        and result.nav_curve is not None
        and len(result.nav_curve)
    ):
        nav_df = result.nav_curve.reset_index()
        nav_df.columns = ["date", "nav"]
        nav_df.to_csv(out_dir / "nav_curve.csv", index=False)
        result.weight_history.reset_index().rename(columns={"index": "date"}).to_csv(
            out_dir / "weight_history.csv", index=False
        )
    if result.trade_history:
        with open(out_dir / "trade_history.json", "w") as f:
            json.dump(result.trade_history, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_backtest(args):
    adapter = _build_adapter(args)
    model_name = adapter.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/sandbox/{model_name}/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _processed_ready = (
        args.data_provider == "processed"
        and Path(args.data_dir).exists()
        and (Path(args.data_dir) / "equities.csv").exists()
    )
    if _processed_ready:
        provider = ProcessedDataProvider(data_dir=args.data_dir, sec_dir=args.sec_dir)
        print(f"Data provider: ProcessedDataProvider({args.data_dir})")
    else:
        if args.data_provider == "processed":
            print(
                f"  WARNING: Processed data not found. Falling back to MockDataProvider."
            )
        provider = MockDataProvider(seed=args.seed)
        print(f"Data provider: MockDataProvider(seed={args.seed})")

    use_pipeline = not args.no_pipeline
    asset_class_map = _build_asset_class_map(provider)

    print(f"PortBench Sandbox — Model: {model_name}")
    print(f"Timestamp:  {timestamp}")
    print(f"Profiles:   {args.profiles}")
    print(f"Pipeline:   {'S1→S5' if use_pipeline else 'direct allocate()'}")
    print(f"Workers:    {args.workers}")
    print("=" * 60)

    def _run_stress(profile_name, scenario):
        """Run one stress scenario; returns (scenario_name, result, passed)."""
        profile = PROFILES[profile_name]
        engine = BacktestEngine(
            strategy=adapter,
            provider=provider,
            start_date=scenario.start,
            end_date=scenario.end,
            rebalance_freq="weekly",
            initial_nav=args.initial_nav,
            use_pipeline=use_pipeline,
            profile=profile,
            asset_class_map=asset_class_map,
        )
        result = engine.run()
        passed = abs(result.max_drawdown) <= profile.max_drawdown_tolerance
        result.stress_passed = passed
        return scenario.name, result, passed

    def _run_profile(profile_name):
        """Run Phase A (stress, parallel) + Phase B (normal) for one profile."""
        profile = PROFILES[profile_name]
        profile_out = output_dir / profile_name
        print(f"\n[Profile: {profile_name.upper()}]")
        print(
            f"  Constraints: max_equity={profile.max_equity_weight}, "
            f"min_bond_cash={profile.min_bond_cash_weight}, "
            f"max_drawdown={profile.max_drawdown_tolerance}"
        )
        print(f"  Phase A: Stress gate ({len(STRESS_SCENARIOS)} scenarios, parallel)")

        # Stress scenarios in parallel
        stress_summaries = []
        all_stress_passed = True
        with ThreadPoolExecutor(max_workers=len(STRESS_SCENARIOS)) as ex:
            futures = {
                ex.submit(_run_stress, profile_name, sc): sc for sc in STRESS_SCENARIOS
            }
            stress_results = {}
            for fut in as_completed(futures):
                sc_name, result, passed = fut.result()
                stress_results[sc_name] = (result, passed)

        for scenario in STRESS_SCENARIOS:  # print in deterministic order
            result, passed = stress_results[scenario.name]
            status = "PASSED" if passed else "FAILED"
            print(
                f"    {scenario.name}: {status} "
                f"(drawdown={result.max_drawdown:.1%}, "
                f"tolerance={profile.max_drawdown_tolerance:.0%})"
            )
            _save_result(result, profile_out / f"stress_{scenario.name}")
            stress_summaries.append(
                {
                    "scenario": scenario.name,
                    "passed": passed,
                    "max_drawdown": round(result.max_drawdown, 4),
                    "tolerance": profile.max_drawdown_tolerance,
                    "total_return": round(result.total_return, 4),
                }
            )
            if not passed:
                all_stress_passed = False

        # Normal backtest (only if stress passed)
        normal_result = None
        if all_stress_passed:
            print(f"  Phase B: Normal market backtest (2024-01-01 → 2024-12-31)")
            engine = BacktestEngine(
                strategy=adapter,
                provider=provider,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                rebalance_freq=args.rebalance,
                initial_nav=args.initial_nav,
                use_pipeline=use_pipeline,
                profile=profile,
                asset_class_map=asset_class_map,
            )
            normal_result = engine.run()
            _save_result(normal_result, profile_out / "normal")
            print(
                f"    Return={normal_result.total_return:+.2%}  "
                f"Sharpe={normal_result.sharpe_ratio:.3f}  "
                f"CEPS={normal_result.mean_ceps:.4f}  "
                f"Alignment={normal_result.mean_profile_score:.4f}"
            )
        else:
            print(f"  Phase B: Skipped (stress gate FAILED)")

        return profile_name, {
            "stress_gate_passed": all_stress_passed,
            "stress_results": stress_summaries,
            "normal": normal_result.to_dict() if normal_result else None,
        }

    # Run all profiles in parallel
    comparison: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_profile, p): p for p in args.profiles}
        for fut in as_completed(futures):
            profile_name, profile_data = fut.result()
            comparison[profile_name] = profile_data

    # ── Profile comparison JSON ───────────────────────────────────────────
    normal_returns = [
        v["normal"]["total_return"]
        for v in comparison.values()
        if v["normal"] is not None
    ]
    adaptation_score = float(np.std(normal_returns)) if len(normal_returns) > 1 else 0.0

    comparison_out = {
        "model_name": model_name,
        "timestamp": timestamp,
        "profiles": comparison,
        "adaptation_score": round(adaptation_score, 4),
    }
    with open(output_dir / "profile_comparison.json", "w") as f:
        json.dump(comparison_out, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Adaptation score (std of per-profile returns): {adaptation_score:.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_backtest(args)
