"""Resource-cost aggregation for architecture comparison tables."""

from datetime import date

import pandas as pd

from portbench.sandbox.result import BacktestResult


def test_backtest_result_aggregates_resource_costs():
    result = BacktestResult(
        model_name="MA",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        initial_nav=100.0,
        nav_curve=pd.Series([100.0, 101.0]),
        weight_history=pd.DataFrame([{"SPY": 1.0}]),
        resource_audit=[
            {
                "usage": {
                    "token_exact": 100,
                    "token_est": 0,
                    "request_count": 8,
                    "tool_call_count": 2,
                    "cache_hit_count": 0,
                    "logical_call_count": 8,
                    "latency_ms": 800.0,
                }
            },
            {
                "usage": {
                    "token_exact": 120,
                    "token_est": 0,
                    "request_count": 8,
                    "tool_call_count": 1,
                    "cache_hit_count": 0,
                    "logical_call_count": 8,
                    "latency_ms": 1000.0,
                }
            },
        ],
    )
    totals = result.to_dict()["resource_totals"]
    assert totals["episodes"] == 2
    assert totals["token_exact"] == 220
    assert totals["request_count"] == 16
    assert totals["logical_call_count"] == 16
    assert totals["mean_latency_ms_per_episode"] == 900.0
