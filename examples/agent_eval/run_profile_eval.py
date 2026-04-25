"""
Run personalized investor profile evaluation on the PortBench pipeline.

Evaluates an agent separately for three investor risk profiles
(conservative / balanced / aggressive) and measures how well its
weight allocations satisfy each profile's constraints.

Usage:
    python examples/agent_eval/run_profile_eval.py                          # mock agent
    python examples/agent_eval/run_profile_eval.py --model anthropic:claude-opus-4-7
    python examples/agent_eval/run_profile_eval.py --baseline equal_weight
    python examples/agent_eval/run_profile_eval.py --profiles conservative balanced

Output:
    outputs/profile/{model_name}/{timestamp}/
        conservative/
            profile_scores.json    — per-episode alignment scores + mean
            ceps_scores.json       — CEPS for this profile's run
            summary.txt
        balanced/  ...
        aggressive/  ...
        profile_comparison.json   — all three side-by-side + adaptation_score
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import os
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.agent_eval import build_default_pipeline
from portbench.agent_eval.llm_adapters import AnthropicAdapter, OpenAIAdapter, LiteLLMAdapter
from portbench.agent_eval.base import MarketSnapshot, StageID
from portbench.agent_eval.mock_agent import MockAgentAdapter
from portbench.agent_eval.investor_profiles import PROFILES, ProfiledPipeline
from portbench.baselines import EqualWeightBaseline, SixtyFortyBaseline, RiskParityBaseline
from portbench.metrics.ceps import CEPS
from portbench.qa_builder.mock_data import MockDataProvider
from portbench.qa_builder.processed_data import ProcessedDataProvider


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="PortBench personalized profile evaluation")
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--profiles", nargs="+",
                        choices=["conservative", "balanced", "aggressive"],
                        default=["conservative", "balanced", "aggressive"],
                        help="Which investor profiles to evaluate")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["equal_weight", "sixty_forty", "risk_parity"])
    parser.add_argument("--model", type=str, default=None,
                        help="'<provider>:<model_id>' e.g. 'anthropic:claude-opus-4-7'")
    parser.add_argument("--data-provider", type=str, default="processed",
                        choices=["mock", "processed"])
    parser.add_argument("--data-dir", type=str, default="datasets/processed")
    parser.add_argument("--sec-dir", type=str, default="datasets/sec")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers copied from run_evaluation.py
# ---------------------------------------------------------------------------

def _build_cloud_adapter(spec: str):
    if ":" not in spec:
        raise ValueError(f"--model must be '<provider>:<model_id>', got: {spec!r}")
    provider, model = spec.split(":", 1)
    provider = provider.lower()
    _COMPAT = {
        "qwen":     ("DASHSCOPE_API_KEY",  "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "kimi":     ("MOONSHOT_API_KEY",   "https://api.moonshot.cn/v1"),
        "deepseek": ("DEEPSEEK_API_KEY",   "https://api.deepseek.com/v1"),
    }
    if provider == "anthropic":
        return AnthropicAdapter(model=model)
    elif provider == "openai":
        return OpenAIAdapter(model=model)
    elif provider in _COMPAT:
        key_env, base_url = _COMPAT[provider]
        return OpenAIAdapter(model=model, base_url=base_url, api_key_env=key_env)
    else:
        return LiteLLMAdapter(model=f"{provider}/{model}")


def build_snapshots(provider, n: int, seed: int = 0) -> list[MarketSnapshot]:
    test_dates = pd.bdate_range("2024-01-01", "2024-12-31").date.tolist()
    rng = np.random.default_rng(seed)
    sampled = sorted(rng.choice(test_dates, size=min(n, len(test_dates)), replace=False).tolist())

    assets = provider.list_assets()
    snapshots = []
    for d in sampled:
        lookback_start = d - timedelta(days=90)
        price_data, return_data = {}, {}
        for asset in assets:
            try:
                price_data[asset]  = provider.get_price_series(asset, lookback_start, d)
                return_data[asset] = provider.get_return_series(asset, lookback_start, d)
            except Exception:
                continue

        corr = pd.DataFrame(return_data).corr() if len(return_data) >= 2 else None
        news_text = ""
        for asset in assets:
            try:
                txt = provider.get_news(asset, d)
                if txt:
                    news_text = txt
                    break
            except Exception:
                continue

        n_assets = len(assets)
        snapshots.append(MarketSnapshot(
            decision_date=d,
            price_data=price_data,
            return_data=return_data,
            macro_data=provider.get_macro(d),
            current_weights={a: round(1.0 / n_assets, 4) for a in assets},
            market_regime=provider.get_regime(d).value,
            news_text=news_text,
            correlation_matrix=corr,
        ))
    return snapshots


def _build_asset_class_map(provider) -> dict[str, str]:
    all_classes = ["equities", "bonds", "commodities", "real_estate", "cryptocurrency", "cash"]
    result = {}
    for cls in all_classes:
        try:
            for a in provider.list_assets(cls):
                result[a] = cls
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_profile_evaluation(args):
    # --- Adapter ---
    if args.baseline == "equal_weight":
        adapter = EqualWeightBaseline()
    elif args.baseline == "sixty_forty":
        adapter = SixtyFortyBaseline()
    elif args.baseline == "risk_parity":
        adapter = RiskParityBaseline()
    elif args.model:
        adapter = _build_cloud_adapter(args.model)
    else:
        adapter = MockAgentAdapter(noise_level=args.noise, seed=args.seed)

    model_name = adapter.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profiles_dir = Path(f"outputs/profile/{model_name}/{timestamp}")
    profiles_dir.mkdir(parents=True, exist_ok=True)

    print(f"PortBench Profile Evaluation — Model: {model_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Profiles: {args.profiles}")
    print("=" * 60)

    # --- Provider ---
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
            print(f"  WARNING: Processed data not found. Falling back to MockDataProvider.")
        provider = MockDataProvider(seed=args.seed)
        print(f"Data provider: MockDataProvider(seed={args.seed})")

    asset_class_map = _build_asset_class_map(provider)
    pipeline = build_default_pipeline(adapter)
    profiled = ProfiledPipeline(pipeline, asset_class_map)
    ceps_obj = CEPS()

    ordered_ids = [
        StageID.S1_MARKET_INTERPRETATION,
        StageID.S2_SIGNAL_GENERATION,
        StageID.S3_WEIGHT_OPTIMIZATION,
        StageID.S4_EXECUTION_SIMULATION,
        StageID.S5_RISK_MONITORING,
    ]

    # Build snapshots once (same episodes for all profiles for fair comparison)
    print(f"\nBuilding {args.n_episodes} snapshots...")
    snapshots = build_snapshots(provider, n=args.n_episodes, seed=args.seed)
    print(f"  Got {len(snapshots)} snapshots.")

    # --- Per-profile evaluation ---
    comparison: dict[str, dict] = {}

    for profile_name in args.profiles:
        profile = PROFILES[profile_name]
        print(f"\n[Profile: {profile_name.upper()}]")
        profile_out_dir = profiles_dir / profile_name
        profile_out_dir.mkdir(parents=True, exist_ok=True)

        alignment_scores: list[float] = []
        stage_score_lists = []
        episode_results = []

        for i, snap in enumerate(snapshots):
            episode, align_score = profiled.run_episode(snap, profile)
            episode_results.append(episode)
            alignment_scores.append(align_score)
            stage_score_lists.append(episode.to_stage_score_list())
            print(f"  Episode {i+1}/{len(snapshots)}: {snap.decision_date}  "
                  f"alignment={align_score:.3f}")

        mean_align = float(np.mean(alignment_scores)) if alignment_scores else 0.0
        std_align  = float(np.std(alignment_scores))  if alignment_scores else 0.0
        print(f"  Mean alignment score: {mean_align:.4f}")

        # Per-stage CEPS
        ceps_result = ceps_obj.compute_batch(stage_score_lists)
        print(f"  Mean CEPS: {ceps_result['mean_ceps']:.4f}")

        # Profile scores JSON
        profile_scores = {
            "profile": profile_name,
            "mean_profile_score": mean_align,
            "std_profile_score":  std_align,
            "per_episode": alignment_scores,
        }
        with open(profile_out_dir / "profile_scores.json", "w") as f:
            json.dump(profile_scores, f, indent=2)

        # CEPS JSON
        with open(profile_out_dir / "ceps_scores.json", "w") as f:
            json.dump(ceps_result, f, indent=2, default=str)

        # Summary
        per_stage = {
            sid.value: [r.stage_scores.get(sid, 0.0) for r in episode_results]
            for sid in ordered_ids
        }
        per_stage_means = {k: float(np.mean(v)) for k, v in per_stage.items()}
        summary_lines = [
            f"Profile:        {profile_name.upper()}",
            f"Model:          {model_name}",
            f"Mean alignment: {mean_align:.4f}",
            f"Std  alignment: {std_align:.4f}",
            f"Mean CEPS:      {ceps_result['mean_ceps']:.4f}",
            f"Episodes:       {len(alignment_scores)}",
            "",
            "Per-stage means:",
        ] + [f"  {k}: {v:.4f}" for k, v in per_stage_means.items()]
        summary = "\n".join(summary_lines)
        with open(profile_out_dir / "summary.txt", "w") as f:
            f.write(summary + "\n")

        comparison[profile_name] = {
            "mean_ceps":          ceps_result["mean_ceps"],
            "mean_profile_score": mean_align,
            "std_profile_score":  std_align,
            "n_episodes":         len(alignment_scores),
        }

    # --- Profile comparison + adaptation score ---
    align_means = [v["mean_profile_score"] for v in comparison.values()]
    adaptation_score = float(np.std(align_means)) if len(align_means) > 1 else 0.0

    comparison_out = {
        "model_name":       model_name,
        "run_id":           timestamp,
        "profiles":         comparison,
        "adaptation_score": adaptation_score,
    }
    with open(profiles_dir / "profile_comparison.json", "w") as f:
        json.dump(comparison_out, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Adaptation score: {adaptation_score:.4f}  "
          f"({'adapts' if adaptation_score > 0.05 else 'uniform'})")
    print(f"Results saved to {profiles_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_profile_evaluation(args)
