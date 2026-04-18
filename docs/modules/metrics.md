# Metrics Library (`portbench/metrics/`)

## Overview

Shared financial metrics used by both the QA dataset builder (ground-truth computation) and the end-to-end agent evaluation pipeline (scoring). All functions are pure (no side effects) and operate on `pd.Series` of daily returns.

## Configuration

```python
from portbench.metrics import MetricsConfig

config = MetricsConfig(
    risk_free_rate=0.04,        # Annual risk-free rate for Sharpe/Sortino
    annualization_factor=252,   # Trading days per year
    var_confidence=0.95,        # VaR confidence level
    benchmark_returns=None,     # pd.Series for information ratio
)
```

## Functions by File

### `return_metrics.py`

| Function | Description | Formula |
|----------|-------------|---------|
| `total_return(r)` | Cumulative return | `∏(1+rᵢ) - 1` |
| `cagr(r, config)` | Compound Annual Growth Rate | `(1+total_return)^(252/n) - 1` |

### `risk_metrics.py`

| Function | Description | Formula |
|----------|-------------|---------|
| `volatility(r, config)` | Annualized volatility | `std(r) × √252` |
| `max_drawdown(r)` | Maximum peak-to-trough decline | `min(Pₜ/max(Pᵢ, i≤t) - 1)` |
| `var(r, config)` | Value at Risk (historical simulation) | `quantile(r, 1-confidence)` |
| `cvar(r, config)` | Conditional VaR / Expected Shortfall | `mean(r[r ≤ VaR])` |

### `risk_adjusted.py`

| Function | Description | Formula |
|----------|-------------|---------|
| `sharpe_ratio(r, config)` | Sharpe Ratio | `(CAGR - rf) / vol` |
| `sortino_ratio(r, config)` | Sortino Ratio | `(CAGR - rf) / downside_vol` |
| `calmar_ratio(r, config)` | Calmar Ratio | `CAGR / |max_drawdown|` |
| `information_ratio(r, config)` | Information Ratio | `active_return / tracking_error` |

### `allocation_metrics.py`

| Function | Description |
|----------|-------------|
| `weight_mae(actual, gt)` | Mean absolute error between two weight dicts |
| `portfolio_return_gap(actual_w, gt_w, returns)` | Return gap attributable to weight difference |

### `ceps.py` — Cross-Stage Error Propagation Score

The CEPS metric quantifies how errors compound through the five-stage pipeline.

```python
from portbench.metrics.ceps import CEPS, StageScore

ceps = CEPS(propagation_weight=0.3)

# Single episode
stage_scores = [StageScore("S1", "S1_MARKET_INTERPRETATION", score=0.85), ...]
result = ceps.compute(stage_scores)
print(result.ceps_score)          # e.g. 0.74
print(result.propagation_penalty) # cascade amplification penalty
print(result.isolated_avg)        # simple average of stage scores

# Batch (multiple episodes)
batch_result = ceps.compute_batch([episode1_scores, episode2_scores, ...])
print(batch_result["mean_ceps"])
print(batch_result["per_stage_mean"])   # {"S1": 0.85, "S2": 0.80, ...}
```

**CEPS Formula:**

```
isolated_avg = mean(stage_scores)
cascade_drops = [max(s[i] - s[i+1], 0) for i in range(len-1)]
propagation_penalty = propagation_weight × sum(cascade_drops)
CEPS = clip(isolated_avg - propagation_penalty, 0, 1)
```

The cascade penalty specifically penalizes the pattern where a good stage is followed by a worse stage — the intuition is that errors propagate and amplify downstream. Uniform mediocrity is penalized less than a sharp mid-pipeline drop.

## Convenience Function

```python
from portbench.metrics import compute_all

metrics = compute_all(returns, config=MetricsConfig(), benchmark_returns=spy_returns)
# Returns PortfolioMetrics dataclass with all metrics computed
print(metrics.sharpe_ratio, metrics.max_drawdown, metrics.var_95)
```

## Design Notes

- All functions return `float` (or `Optional[float]` for edge cases like too-short series).
- Edge cases (empty series, zero volatility) return `None` rather than raising exceptions.
- `weight_mae` handles missing keys in either dict by treating them as zero weight.
