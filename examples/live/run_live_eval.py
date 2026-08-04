"""
Thin CLI for PortBench live / rolling evaluation.

Core logic: portbench.live.LiveEvalRunner

Examples:
    # Single step (latest available decision → next day)
    python examples/live/run_live_eval.py --mock

    # July 2025 last two weeks, daily rebalance (simulated live capability demo)
    python examples/live/run_live_eval.py --mock \\
        --start 2025-07-16 --end 2025-07-31 --rebalance daily

    # Same window, real model
    python examples/live/run_live_eval.py --provider dashscope --model qwen3.7-max \\
        --start 2025-07-16 --end 2025-07-31 --rebalance daily --profile balanced

    # Monthly rebalance over a quarter (closer to main-paper frequency)
    python examples/live/run_live_eval.py --mock \\
        --start 2025-01-01 --end 2025-03-31 --rebalance monthly
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from portbench.live import SUPPORTED_FREQUENCIES, LiveEvalRunner


def _parse_date(value):
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> int:
    p = argparse.ArgumentParser(
        description="PortBench live / rolling eval (dual oracle: lookback + ex_post)"
    )
    p.add_argument("--provider", default="mock")
    p.add_argument("--model", default=None)
    p.add_argument("--profile", default="balanced")
    p.add_argument("--mock", action="store_true", help="Use MockAgentAdapter")
    p.add_argument(
        "--rebalance",
        default="daily",
        choices=list(SUPPORTED_FREQUENCIES),
        help="Rebalance frequency for range mode (default: daily)",
    )
    p.add_argument(
        "--start",
        default=None,
        help="Range start YYYY-MM-DD (with --end: rolling live window)",
    )
    p.add_argument(
        "--end",
        default=None,
        help="Range end YYYY-MM-DD (decision dates in [start, end])",
    )
    p.add_argument("--decision-date", default=None, help="Single-step decision date")
    p.add_argument("--as-of-today", default=None, help="Single-step realization date")
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Always re-download Yahoo+FRED + preprocess before eval",
    )
    p.add_argument(
        "--no-auto-refresh",
        action="store_true",
        help="Do not auto-refresh when requested dates exceed local data "
        "(default: auto-refresh + preprocess when coverage is missing)",
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
    common = dict(
        provider=args.provider,
        model=args.model,
        profile=args.profile,
        mock=args.mock or args.provider == "mock",
        force_refresh=args.force_refresh,
        auto_refresh=not args.no_auto_refresh,
        skip_refresh=True,
        skip_preprocess=True,
    )

    if args.start or args.end:
        if not (args.start and args.end):
            p.error("Range mode requires both --start and --end")
        result = runner.run_range(
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            rebalance=args.rebalance,
            **common,
        )
        print(f"rebalance: {result.rebalance}")
        print(f"window:    {result.start} → {result.end}")
        print(f"episodes:  {len(result.episodes)}")
        print(f"summary:   {result.summary_path}")
        summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
        print("mean CEPS lookback:", summary.get("mean_ceps_lookback"))
        print("mean CEPS ex_post: ", summary.get("mean_ceps_ex_post"))
        for row in summary.get("episodes", []):
            print(
                f"  {row['decision_date']} → {row['realization_date']}: "
                f"lookback={row['ceps_lookback']}, ex_post={row['ceps_ex_post']}"
            )
        return 0

    result = runner.run(
        decision_date=_parse_date(args.decision_date),
        as_of_today=_parse_date(args.as_of_today),
        rebalance=args.rebalance,
        **common,
    )
    print(f"decision_date:    {result.decision_date}")
    print(f"realization_date: {result.today}")
    print(f"output: {result.output_dir}")
    print("CEPS lookback:", result.scores["lookback"]["ceps"])
    print("CEPS ex_post: ", result.scores["ex_post"]["ceps"])
    top = sorted(result.recommended_weights.items(), key=lambda kv: -kv[1])[:8]
    print("recommended weights (top 8):")
    print(json.dumps(dict(top), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
