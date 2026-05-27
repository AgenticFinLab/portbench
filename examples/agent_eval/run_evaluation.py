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
    outputs/evaluation_results/eval_results/{model_name}/
        per_stage_scores.json    — per-stage scores across all episodes
        ceps_scores.json         — CEPS metric with propagation penalty
        stress_test_results.json — pass/fail for each stress scenario
        risk_first_ranking.json  — final ranking (stress-gated performance)
        summary.txt              — human-readable summary

The evaluation workflow:
  1. Stress gate:  Agent must pass all three stress scenarios (2015, 2020, 2022).
  2. Performance:  CEPS computed over normal market episodes (2024 test set).
  3. Risk-first ranking: Models that failed stress gate are shown but marked FAILED.
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import os

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.agent_eval import build_default_pipeline, STRESS_SCENARIOS
from portbench.agent_eval.llm_adapters import AnthropicAdapter, OpenAIAdapter, LiteLLMAdapter
from portbench.agent_eval.base import MarketSnapshot, StageID
from portbench.agent_eval.mock_agent import MockAgentAdapter
from portbench.agent_eval.stress_scenarios import ScenarioInjector
from portbench.baselines import EqualWeightBaseline, SixtyFortyBaseline, RiskParityBaseline
from portbench.metrics.ceps import CEPS
from portbench.qa_builder.mock_data import MockDataProvider
from portbench.qa_builder.processed_data import ProcessedDataProvider


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
    parser.add_argument("--use-tools", action="store_true",
                        help="Enable tool-calling for S1/S2/S3 stages (calculator, correlation, "
                             "volatility, mean_return). Only effective with cloud/LLM adapters. "
                             "Add --web-search to also enable the web search tool.")
    parser.add_argument("--web-search", action="store_true",
                        help="Enable web search tool (requires SERPER_API_KEY). "
                             "Only active when --use-tools is also set.")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["equal_weight", "sixty_forty", "risk_parity"],
                        help="Evaluate a baseline instead of the mock LLM agent")
    parser.add_argument("--data-provider", type=str, default="processed",
                        choices=["mock", "processed"],
                        help="Data source: 'processed' (real data from datasets/processed/, "
                             "default) or 'mock' (synthetic GBM, no API keys needed). "
                             "Falls back to mock automatically if processed data is missing.")
    parser.add_argument("--data-dir", type=str, default="datasets/processed",
                        help="Path to processed data directory (for --data-provider=processed)")
    parser.add_argument("--sec-dir", type=str, default="datasets/sec",
                        help="Path to SEC filings directory (for --data-provider=processed)")

    parser.add_argument("--model", type=str, default=None,
                        help="Evaluate a cloud/API model. "
                             "Format: '<provider>:<model_id>', e.g. "
                             "'anthropic:claude-opus-4-7', "
                             "'openai:gpt-4o', "
                             "'litellm:together_ai/mistralai/Mixtral-8x7B'")

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

def build_snapshots(provider, n: int, seed: int = 0) -> list[MarketSnapshot]:
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

        # Compute pairwise correlation matrix from return data
        if len(return_data) >= 2:
            ret_df = pd.DataFrame(return_data)
            corr = ret_df.corr()
        else:
            corr = None

        # Pull news text for the first asset that has any (text-bearing classes only)
        news_text = ""
        for asset in assets:
            try:
                txt = provider.get_news(asset, d)
                if txt:
                    news_text = txt
                    break
            except Exception:
                continue

        snapshots.append(MarketSnapshot(
            decision_date=d,
            price_data=price_data,
            return_data=return_data,
            macro_data=macro,
            current_weights=current_weights,
            market_regime=regime,
            news_text=news_text,
            correlation_matrix=corr,
        ))

    return snapshots


# ---------------------------------------------------------------------------
# Cloud adapter factory
# ---------------------------------------------------------------------------

def _build_cloud_adapter(spec: str):
    """
    Parse a '<provider>:<model_id>' string and return the appropriate cloud adapter.

    Supported providers:
        anthropic   — Claude models (ANTHROPIC_API_KEY)
        openai      — GPT models (OPENAI_API_KEY)
        qwen        — Qwen models via DashScope (DASHSCOPE_API_KEY)
                      e.g. "qwen:qwen-plus", "qwen:qwen-max", "qwen:qwen-long"
        kimi        — Moonshot AI models (MOONSHOT_API_KEY)
                      e.g. "kimi:moonshot-v1-8k", "kimi:moonshot-v1-32k"
        deepseek    — DeepSeek models (DEEPSEEK_API_KEY)
                      e.g. "deepseek:deepseek-chat", "deepseek:deepseek-reasoner"
        litellm     — Any model via LiteLLM unified interface
    """
    if ":" not in spec:
        raise ValueError(
            f"--model must be '<provider>:<model_id>', got: {spec!r}\n"
            "Examples: 'anthropic:claude-opus-4-7', 'openai:gpt-4o', "
            "'qwen:qwen-plus', 'kimi:moonshot-v1-8k', 'deepseek:deepseek-chat'"
        )

    provider, model = spec.split(":", 1)
    provider = provider.lower()

    # Providers that use OpenAI-compatible endpoints
    _OPENAI_COMPAT = {
        "qwen": (
            "DASHSCOPE_API_KEY",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "kimi": (
            "MOONSHOT_API_KEY",
            "https://api.moonshot.cn/v1",
        ),
        "deepseek": (
            "DEEPSEEK_API_KEY",
            "https://api.deepseek.com/v1",
        ),
    }

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("  WARNING: ANTHROPIC_API_KEY is not set.")
        return AnthropicAdapter(model=model)
    elif provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("  WARNING: OPENAI_API_KEY is not set.")
        return OpenAIAdapter(model=model)
    elif provider in _OPENAI_COMPAT:
        key_env, base_url = _OPENAI_COMPAT[provider]
        if not os.environ.get(key_env):
            print(f"  WARNING: {key_env} is not set.")
        return OpenAIAdapter(model=model, base_url=base_url, api_key_env=key_env)
    elif provider == "litellm":
        return LiteLLMAdapter(model=model)
    else:
        print(f"  INFO: Unknown provider '{provider}', routing through LiteLLM as '{provider}/{model}'.")
        return LiteLLMAdapter(model=f"{provider}/{model}")


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
# CEPS back-fill helpers
# ---------------------------------------------------------------------------

def _compute_per_episode_ceps(stage_score_lists: list) -> list[float]:
    """Compute per-episode CEPS score from stage score lists."""
    from portbench.metrics.ceps import CEPS
    ceps_obj = CEPS()
    scores = []
    for ssl in stage_score_lists:
        vals = [s.score for s in ssl]
        avg = sum(vals) / len(vals) if vals else 0.0
        drops = sum(max(vals[i] - vals[i + 1], 0) for i in range(len(vals) - 1))
        scores.append(max(0.0, min(1.0, avg - 0.1 * drops)))
    return scores


def _backfill_episode_ceps(logger, ep_ceps_list: list[float]) -> None:
    """Update the last N episode JSON files with their computed CEPS scores."""
    ep_files = sorted(logger.episodes_dir.glob("*.json"))
    # Normal-market episodes are always written last — take the tail
    target_files = ep_files[-len(ep_ceps_list):] if ep_ceps_list else []
    for ep_file, ceps_score in zip(target_files, ep_ceps_list):
        try:
            data = json.loads(ep_file.read_text(encoding="utf-8"))
            data["ceps_score"] = round(ceps_score, 6)
            ep_file.write_text(
                json.dumps(data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass


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
    elif args.model:
        adapter = _build_cloud_adapter(args.model)
    elif args.local_model:
        adapter = _build_local_adapter(args.local_model, args.local_url)
    else:
        adapter = MockAgentAdapter(noise_level=args.noise, seed=args.seed)

    model_name = adapter.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/ceps/{model_name}/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PortBench Evaluation — Model: {model_name}")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    use_tools = getattr(args, "use_tools", False)
    include_web_search = getattr(args, "web_search", False)
    if use_tools:
        from portbench.agent_eval.tools import get_tools
        active_tools = get_tools(include_web_search=include_web_search)
        tool_names = [t.name for t in active_tools]
        print(f"Tool-calling enabled: {tool_names}")

    pipeline = build_default_pipeline(adapter, use_tools=use_tools)

    _processed_ready = (
        args.data_provider == "processed"
        and Path(args.data_dir).exists()
        and (Path(args.data_dir) / "equities.csv").exists()
    )
    if _processed_ready:
        provider = ProcessedDataProvider(data_dir=args.data_dir, sec_dir=args.sec_dir)
        actual_data_provider = f"processed:{args.data_dir}"
        print(f"Data provider: ProcessedDataProvider({args.data_dir})")
    else:
        if args.data_provider == "processed":
            print(f"  WARNING: Processed data not found at {args.data_dir}/equities.csv")
            print(f"  Run examples/data_collect/get_all.py first to download real data.")
            print(f"  Falling back to MockDataProvider.")
        provider = MockDataProvider(seed=args.seed)
        actual_data_provider = f"mock(seed={args.seed})"
        print(f"Data provider: MockDataProvider(seed={args.seed})")

    log_dir = output_dir / "logs"
    pipeline.enable_logging(
        output_dir=str(log_dir),
        model_name=model_name,
        run_id=timestamp,
        config={
            "run_id": timestamp,
            "n_episodes": args.n_episodes,
            "data_provider": actual_data_provider,
            "seed": args.seed,
            "use_tools": use_tools,
        },
    )
    print(f"Interaction logging → {log_dir}/")

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

    for i, snap in enumerate(snapshots):
        result = pipeline.run_episode(snap)
        episode_results.append(result)
        stage_score_lists.append(result.to_stage_score_list())
        print(f"  Episode {i+1}/{len(snapshots)}: {snap.decision_date} "
              f"(regime={snap.market_regime})")

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

    # Back-fill per-episode CEPS scores into the episode log files
    if pipeline._logger is not None:
        ep_ceps_list = _compute_per_episode_ceps(stage_score_lists)
        _backfill_episode_ceps(pipeline._logger, ep_ceps_list)
        log_path = pipeline.finalize_logging()
        print(f"  Interaction logs → {log_path}")

    # -----------------------------------------------------------------------
    # Phase 3: Risk-first ranking record
    # -----------------------------------------------------------------------
    ranking_entry = {
        "run_id": timestamp,
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
