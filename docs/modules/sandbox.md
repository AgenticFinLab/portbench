# Sandbox Backtest (`portbench/sandbox/`)

## Overview

The Sandbox provides a **stateful, closed-loop backtest environment** that unifies profile-aware evaluation, stress testing, and PnL measurement in a single run:

| Mechanism | Measures | Stateful |
|-----------|---------|---------|
| QA Dataset | Knowledge understanding and reasoning accuracy | No |
| **Sandbox Backtest** | **Realized PnL + CEPS + Profile Alignment across 3 investor profiles** | **Yes** |

The key distinction from a simple backtest: the Sandbox maintains portfolio state (NAV, weights, cash) across time steps, applies transaction costs on each rebalance, injects the investor profile description into each LLM prompt, and collects per-step CEPS and profile alignment scores as byproducts.

---

## Evaluation Architecture

Each model is evaluated across **three investor profiles** (conservative, balanced, aggressive). For each profile, evaluation runs in two phases:

```
Phase A — Stress Gate (3 crisis windows, profile-sensitive pass/fail)
  2015_china_shock:    2015-08-01 → 2016-02-29
  2020_covid:          2020-02-01 → 2020-05-31
  2022_crypto:         2022-05-01 → 2022-12-31
  Pass condition: abs(max_drawdown) ≤ profile.max_drawdown_tolerance

Phase B — Normal Market Backtest (2024, always runs regardless of Phase A result)
  Outputs: CEPS per step + profile alignment score + NAV curve + PnL metrics
```

Profile drawdown tolerances:

| Profile | `max_drawdown_tolerance` | `max_equity_weight` | `var_limit` |
|---------|--------------------------|---------------------|-------------|
| Conservative | 10% | 40% | 5% |
| Balanced | 20% | 65% | 10% |
| Aggressive | 35% | 90% | 20% |

---

## Architecture

```
DataProvider (MockDataProvider or ProcessedDataProvider)
      │
      ▼
SnapshotBuilder.build(decision_date, current_weights, nav, forward_days)
      │  ↑ live state + forward_days from rebalance_freq
      ▼                    │
MarketSnapshot (patched: "[INVESTOR PROFILE] ..." prepended to news_text)
      │
      ▼
EvalPipeline.run_episode()  (use_pipeline=True, LLM)
or
BaselineStrategy.allocate() (use_pipeline=False)
      │
      ▼ target_weights + EpisodeResult
PortfolioState.rebalance(target_weights, prices, date)
      │  (transaction costs applied)
      │  (CEPS + ProfileAlignment scores collected per step)
      ▼
PortfolioState.mark_to_market(daily_returns, date)  ← non-rebalance days
      │
      ▼
BacktestResult (NAV curve, weights history, PnL metrics, mean_ceps, mean_profile_score)
```

---

## Modules

### `portfolio.py` — PortfolioState

Tracks NAV, current weights, and full trade history across all time steps.

```python
from portbench.sandbox import PortfolioState

state = PortfolioState(nav=1_000_000.0, weights={"SPY": 0.5, "TLT": 0.5})

# On rebalance dates
trade = state.rebalance(target_weights={"SPY": 0.6, "TLT": 0.4}, prices=prices, d=date.today())

# On non-rebalance dates
state.mark_to_market(returns={"SPY": 0.005, "TLT": -0.002}, d=date.today())
```

Transaction cost model (mirrors S4 `ExecutionSimulation`):
- Slippage: **10 bps** linear market impact (buy: +10 bps, sell: −10 bps)
- Commission: **5 bps** per trade value
- Cost is deducted from NAV at rebalance time

Weight drift (mark-to-market) updates weights proportionally to per-asset returns between rebalances, so the next rebalance snapshot reflects the actual drifted portfolio.

### `snapshot_builder.py` — SnapshotBuilder

Constructs a `MarketSnapshot` for any decision date, injecting the real current portfolio state rather than the equal-weight assumption used in the static pipeline. When `forward_days > 0`, also fetches realized future returns and populates `snapshot.future_return_data` for S3 ground-truth computation. `BacktestEngine` derives `forward_days` automatically from `rebalance_freq` (weekly=5, monthly=21, quarterly=63).

```python
from portbench.sandbox import SnapshotBuilder

builder = SnapshotBuilder(provider=provider, assets=assets, lookback_days=60)
# forward_days is set automatically by BacktestEngine; pass manually for one-off use:
snapshot = builder.build(decision_date, current_weights=state.weights, nav=state.nav, forward_days=21)
```

### `engine.py` — BacktestEngine

Main entry point. Drives the full backtest loop and returns a `BacktestResult`.

```python
from portbench.sandbox import BacktestEngine
from portbench.agent_eval.investor_profiles import PROFILES
from portbench.agent_eval.llm_adapters import AnthropicAdapter
from portbench.qa_builder.mock_data import MockDataProvider

adapter = AnthropicAdapter(model="claude-opus-4-7")
provider = MockDataProvider(seed=42)
profile = PROFILES["conservative"]

engine = BacktestEngine(
    strategy=adapter,
    provider=provider,
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
    rebalance_freq="monthly",   # "weekly" | "monthly" | "quarterly"
    initial_nav=1_000_000.0,
    use_pipeline=True,          # True = S1→S5; False = direct allocate()
    profile=profile,            # InvestorProfile — injected into each prompt
    asset_class_map=asset_class_map,  # dict[ticker, asset_class] for alignment scoring
)

result = engine.run()
print(result.summary())
```

**`use_pipeline` parameter:**

| Value | Path | API calls per rebalance | When to use |
|-------|------|------------------------|-------------|
| `True` | Full S1→S5 EvalPipeline | 3 (S1, S2, S3) | LLM agent evaluation |
| `False` | `BaselineStrategy.allocate()` | 0 | Baseline comparison, sanity checks |

When `profile` is provided, the engine prepends `[INVESTOR PROFILE] {profile.description}` to `snapshot.news_text` before each LLM call, and collects per-step CEPS and profile alignment scores.

### `result.py` — BacktestResult

Immutable result container with all performance metrics computed from the NAV curve via `portbench/metrics/` functions.

```python
result.total_return    # float, e.g. 0.1502 = +15.02%
result.cagr            # Compound Annual Growth Rate
result.sharpe_ratio    # Annualized Sharpe
result.sortino_ratio   # Annualized Sortino
result.max_drawdown    # Maximum drawdown (negative, e.g. -0.12)
result.calmar_ratio    # CAGR / |max_drawdown|
result.volatility      # Annualized volatility
result.n_rebalances    # Number of rebalance events
result.total_transaction_cost  # Cumulative cost in dollars

# Profile evaluation fields (populated when profile= is passed to BacktestEngine)
result.profile_name         # str, e.g. "conservative"
result.mean_ceps            # float [0,1], averaged over all rebalance steps
result.mean_profile_score   # float [0,1], averaged profile alignment score
result.stress_passed        # bool | None — set externally by run_backtest.py

result.nav_curve       # pd.Series (index=date, values=NAV)
result.weight_history  # pd.DataFrame (index=date, columns=assets)
result.trade_history   # list[dict] with per-order cost breakdown

result.to_dict()       # JSON-safe dict
result.summary()       # human-readable string
```

---

## Rebalance Frequencies

| `rebalance_freq` | pandas offset | Typical use |
|-----------------|---------------|-------------|
| `"weekly"` | `W-FRI` | Active strategies, higher API cost |
| `"monthly"` | `BMS` | Standard evaluation setting |
| `"quarterly"` | `QS-JAN` | Long-term / passive strategies |

---

## Running the Sandbox CLI

```bash
# MockAgent smoke test — all 3 profiles, stress + normal (no API keys needed)
python examples/sandbox/run_backtest.py --data-provider mock

# Specific profiles only
python examples/sandbox/run_backtest.py --data-provider mock --profiles conservative balanced

# LLM model (any provider supported by the pipeline)
python examples/sandbox/run_backtest.py --model qwen:qwen-plus
python examples/sandbox/run_backtest.py --model anthropic:claude-opus-4-7
python examples/sandbox/run_backtest.py --model openai:gpt-4o

# Baseline comparison (direct allocate(), no API)
python examples/sandbox/run_backtest.py --baseline equal_weight
python examples/sandbox/run_backtest.py --baseline sixty_forty
python examples/sandbox/run_backtest.py --baseline risk_parity

# Processed real data (requires datasets/processed/ to exist)
python examples/sandbox/run_backtest.py --model qwen:qwen-plus --data-provider processed
```

### Output structure

```
outputs/sandbox/{model}/{timestamp}/
  {profile}/                         # conservative | balanced | aggressive
    stress_2015_china_shock/
      backtest_result.json
    stress_2020_covid_flash_crash/
      backtest_result.json
    stress_2022_crypto_collapse/
      backtest_result.json
    normal/                          # always written when run_normal=true
      nav_curve.csv
      weight_history.csv
      backtest_result.json
      summary.txt
  profile_comparison.json            # cross-profile summary + adaptation_score
```

`profile_comparison.json` fields:

| Field | Description |
|-------|-------------|
| `model_name` | Model identifier |
| `profiles` | Per-profile summary dict: `stress_passed`, `total_return`, `mean_ceps`, `mean_profile_score` |
| `adaptation_score` | `std(per-profile total_return)` — how differently the model performs across profiles |

`backtest_result.json` fields (profile runs, subset):

| Field | Description |
|-------|-------------|
| `profile_name` | `"conservative"` / `"balanced"` / `"aggressive"` |
| `mean_ceps` | Average CEPS score across all rebalance steps |
| `mean_profile_score` | Average profile alignment score across all rebalance steps |
| `stress_passed` | Whether this run's max_drawdown ≤ profile tolerance |
| `total_return`, `sharpe_ratio`, `max_drawdown`, ... | Standard PnL metrics |

---

## Relationship to CEPS

CEPS is no longer a separate evaluation mode — it is collected as a **byproduct of each rebalance step** within the Sandbox when `use_pipeline=True`. This means CEPS scores and realized PnL are always available from the same run, enabling direct validation of whether decision quality correlates with returns.

---

## Data Provider Compatibility

| Provider | Sandbox coverage | Notes |
|---------|-----------------|-------|
| `MockDataProvider` | 2015-01-01 → present (synthetic GBM) | No API keys, fully reproducible |
| `ProcessedDataProvider` | 2015-01-02 → 2025-12-31 | Real prices; requires `get_all.py` pipeline |

All three stress windows (2015, 2020, 2022) are fully covered by both providers.
