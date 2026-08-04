# Live / rolling evaluation

Simulate a live track: on each rebalance date the model sees only PiT data, then
we score the same decision under **lookback** and **ex-post** (returns until the
next rebalance).

Core: `portbench/live/`. This folder is only a CLI.

## Config YAML + CLI

Default config: [`configs/live/default.yaml`](../../configs/live/default.yaml)

`models` / `profiles` match [`configs/experiments/default.yaml`](../../configs/experiments/default.yaml)
(10 LLMs + 6 baselines × 3 profiles). Running with no flags sweeps that full grid.

```powershell
python examples/live/run_live_eval.py
# or:
python examples/live/run_live_eval.py --config configs/live/default.yaml
```

CLI overrides / filters, e.g. one model:

```powershell
python examples/live/run_live_eval.py --provider dashscope --model qwen3.7-max --profile balanced
```

## Data coverage (auto-refresh)

Batch experiments keep using frozen `datasets/processed/` (never call Yahoo).

Live uses **`datasets/processed_live/`**: a copy of `processed` plus an overlay
from `datasets/akshare_ext/` (AKShare gap-fill when Yahoo is rate-limited).
`datasets/yahoo/` and `datasets/processed/` are not rewritten by the AKShare
fallback. Auto-refresh: incremental Yahoo attempt → AKShare ext → rebuild
`processed_live`.

Do **not** use `--force-refresh` unless necessary.

Needs `FRED_API_KEY`. Opt out with `--no-auto-refresh`.

**Resume:** by default `skip_existing: true` — if a decision-day folder already has
complete scores (`scores_lookback.json` + `scores_ex_post.json` + …), that day is
reused (no LLM re-call). Force re-run with `--no-skip-existing`.

```powershell
# Today / latest session (auto-pulls if local data is behind)
python examples/live/run_live_eval.py --mock
python examples/live/run_live_eval.py --provider dashscope --model qwen3.7-max
```

## Why daily (not only monthly)?

Main paper pipeline is **monthly**. A true wall-clock monthly live wait is slow.
For proving live-eval capability, use **daily** (or weekly) over a short window
— e.g. the last two weeks of July. Frequencies:

`daily | weekly | monthly | quarterly | yearly`

## July last two weeks (daily)

```powershell
cd D:\GitHub\portbench

# Mock
python examples/live/run_live_eval.py --mock `
  --start 2025-07-16 --end 2025-07-31 --rebalance daily

# Real model
python examples/live/run_live_eval.py `
  --provider dashscope --model qwen3.7-max --profile balanced `
  --start 2025-07-16 --end 2025-07-31 --rebalance daily
```

If you ask for a window newer than local data (e.g. 2026-07), refresh runs
automatically before the episodes.

Outputs:

```text
outputs/live/daily_2025-07-16_2025-07-31/{provider}/{model}/{profile}/
  range_summary.json          # mean CEPS + per-day rows (+ data_coverage)
  YYYY-MM-DD/                 # one folder per decision date
    scores_lookback.json
    scores_ex_post.json
    recommended_weights.json
    episode_trace.json
    prompts.json              # full S1–S3 prompts (no truncation)
    llm_responses.json        # full raw LLM responses (no truncation)
```


## Other frequencies

```powershell
python examples/live/run_live_eval.py --mock --start 2025-07-16 --end 2025-07-31 --rebalance weekly
python examples/live/run_live_eval.py --mock --start 2025-01-01 --end 2025-03-31 --rebalance monthly
```

## Limits

- Auto-refresh is EOD Yahoo/FRED, not intraday streaming
- Before US close, “today” may still be the previous session after refresh
- FRED/news lag; dual scores are per decision, not a full NAV report
