"""
Thin CLI for PortBench live / same-day evaluation.

Core logic lives in portbench.live.LiveEvalRunner.

Usage:
    # Mock (no API keys)
    python examples/live/run_live_eval.py --mock

    # Real model (uses PROVIDER_REGISTRY + .env)
    python examples/live/run_live_eval.py --provider dashscope --model qwen3.7-max

    # Optional: refresh Yahoo/FRED + preprocess first
    python examples/live/run_live_eval.py --mock --force-refresh
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.live import LiveEvalRunner


def _parse_date(value):
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> int:
    p = argparse.ArgumentParser(description="PortBench live eval (yesterday → today)")
    p.add_argument("--provider", default="mock")
    p.add_argument("--model", default=None)
    p.add_argument("--profile", default="balanced")
    p.add_argument("--mock", action="store_true", help="Use MockAgentAdapter")
    p.add_argument("--decision-date", default=None, help="YYYY-MM-DD (yesterday)")
    p.add_argument("--as-of-today", default=None, help="YYYY-MM-DD (today / GT day)")
    p.add_argument("--force-refresh", action="store_true", help="Re-download Yahoo+FRED")
    p.add_argument(
        "--run-preprocess",
        action="store_true",
        help="Run preprocess_all after refresh / before eval",
    )
    p.add_argument("--data-dir", default="datasets/processed")
    p.add_argument("--sec-dir", default="datasets/sec")
    p.add_argument("--output-root", default="outputs/live")
    args = p.parse_args()

    runner = LiveEvalRunner(
        data_dir=args.data_dir,
        sec_dir=args.sec_dir,
        output_root=args.output_root,
    )
    result = runner.run(
        provider=args.provider,
        model=args.model,
        profile=args.profile,
        decision_date=_parse_date(args.decision_date),
        as_of_today=_parse_date(args.as_of_today),
        mock=args.mock or args.provider == "mock",
        force_refresh=args.force_refresh,
        skip_refresh=not args.force_refresh,
        skip_preprocess=not args.run_preprocess,
    )

    print(f"decision_date (yesterday): {result.decision_date}")
    print(f"today (ex-post GT):        {result.today}")
    print(f"output: {result.output_dir}")
    print("CEPS lookback:", result.scores["lookback"]["ceps"])
    print("CEPS ex_post: ", result.scores["ex_post"]["ceps"])
    print("recommended weights (top 8):")
    top = sorted(
        result.recommended_weights.items(), key=lambda kv: -kv[1]
    )[:8]
    print(json.dumps(dict(top), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
