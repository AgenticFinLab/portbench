# Live / rolling evaluation

Simulate a live track: on each rebalance date the model sees only PiT data, then
we score the same decision under **lookback** and **ex-post** (returns until the
next rebalance).

Core: `portbench/live/`. This folder is only a CLI.

## Why daily (not only monthly)?

Main paper pipeline is **monthly**. A true wall-clock monthly live wait is slow.
For proving the benchmark has live-eval capability, use **daily** (or weekly)
rebalance over a short historical window — e.g. the last two weeks of July —
as if you had tested the model every day. Frequencies supported:

`daily | weekly | monthly | quarterly | yearly`

## July last two weeks (daily) — recommended demo

Processed data currently covers **2025-07** (not 2026-07 unless you refresh).

```powershell
cd D:\GitHub\portbench

# 1) Smoke with mock (no API keys)
python examples/live/run_live_eval.py --mock `
  --start 2025-07-16 --end 2025-07-31 --rebalance daily

# 2) Real model (example)
python examples/live/run_live_eval.py `
  --provider dashscope --model qwen3.7-max --profile balanced `
  --start 2025-07-16 --end 2025-07-31 --rebalance daily
```

Outputs:

```text
outputs/live/daily_2025-07-16_2025-07-31/{provider}/{model}/{profile}/
  range_summary.json          # mean CEPS + per-day rows
  YYYY-MM-DD/                 # one folder per decision date
    scores_lookback.json
    scores_ex_post.json
    recommended_weights.json
    episode_trace.json
```

Each episode: decision on day \(D\), realization / ex-post GT on the next
trading day after \(D\).

## Other frequencies

```powershell
# Weekly over the same July window
python examples/live/run_live_eval.py --mock --start 2025-07-16 --end 2025-07-31 --rebalance weekly

# Monthly over Q1 2025 (closer to main-paper cadence)
python examples/live/run_live_eval.py --mock --start 2025-01-01 --end 2025-03-31 --rebalance monthly

# Single latest step only
python examples/live/run_live_eval.py --mock
```

## Refresh calendar-current data (e.g. July 2026)

```powershell
python examples/live/run_live_eval.py --mock --force-refresh --run-preprocess `
  --start 2026-07-16 --end 2026-07-31 --rebalance daily
```

Needs `FRED_API_KEY`. Slow (full Yahoo+FRED re-download + preprocess).

## Limits

- Without refresh, dates must exist in `datasets/processed/`
- EOD only; FRED/news lag
- Rolling historical window ≠ waiting wall-clock for a live month
- Dual scores are per decision; this is not a full NAV backtest report
