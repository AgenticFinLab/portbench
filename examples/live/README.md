# Live / same-day eval

Thin demo that refreshes public market data, lets a model trade on **yesterday's**
information only, and scores the same decision under two oracles:

1. **lookback** — yesterday-optimal (trailing max-Sharpe)
2. **ex_post** — future-optimal using **today's** realized returns

Core code: `portbench/live/`. This folder is only a CLI wrapper.

## Quick start

```powershell
cd D:\GitHub\portbench

# Mock (no API keys). Uses existing datasets/processed.
python examples/live/run_live_eval.py --mock

# Real model
python examples/live/run_live_eval.py --provider dashscope --model qwen3.7-max --profile balanced
```

Optional refresh (slow; needs `FRED_API_KEY`):

```powershell
python examples/live/run_live_eval.py --mock --force-refresh --run-preprocess
```

## Defaults

| Role | Date | Use |
|------|------|-----|
| Decision | previous trading day in **processed data** | LLM prompt / PiT snapshot |
| Realization | latest trading day in **processed data** | `future_return_data` for ex-post GT only |

Without `--force-refresh --run-preprocess`, “today” means the newest date already
in `datasets/processed/` (not necessarily the calendar today).

Outputs land in `outputs/live/{provider}/{model}/{decision_date}/`:

- `episode_trace.json`
- `scores_lookback.json`
- `scores_ex_post.json`
- `recommended_weights.json`
- `run_meta.json`

## Limits

- EOD data; before US close, “today” may be the previous session
- FRED macro series lag; news/SEC are not same-day
- Dual scores measure **one decision**, not a multi-day NAV backtest
- This does **not** remove all training-data leakage risk
