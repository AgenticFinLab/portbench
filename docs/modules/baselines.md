# Baseline Strategies (`portbench/baselines/`)

## Overview

Non-LLM portfolio allocation strategies that implement the same `AgentAdapter` interface as LLM agents. Drop any baseline into `build_default_pipeline()` to run a controlled comparison. Baselines are evaluated through the same five-stage pipeline and stress tests as LLM agents.

## Interface

All baselines extend `BaselineStrategy` which itself extends `AgentAdapter`:

```python
class BaselineStrategy(AgentAdapter):
    def allocate(self, snapshot: MarketSnapshot) -> dict[str, float]:
        """Return weight dict summing to 1.0."""
        ...
```

The `complete()` method (required by `AgentAdapter`) is implemented to return a JSON stub — stages that need LLM text bypass it and use `compute_ground_truth()` directly.

## Available Baselines

### `EqualWeightBaseline` — 1/N

```python
from portbench.baselines import EqualWeightBaseline
baseline = EqualWeightBaseline()
# or with explicit universe:
baseline = EqualWeightBaseline(asset_universe=["SPY", "TLT", "GLD", "BTC-USD"])
```

Divides weight equally across all assets in the snapshot. Despite its simplicity, consistently competitive in empirical studies (DeMiguel et al., 2009). Used as the minimum bar in PortBench rankings.

### `SixtyFortyBaseline` — 60/40

```python
from portbench.baselines import SixtyFortyBaseline
baseline = SixtyFortyBaseline(equity_fraction=0.60, bond_fraction=0.40)
# With alternatives (reduces eq/bond to make room):
baseline = SixtyFortyBaseline(include_alternatives=True, alt_fraction=0.10)
```

Traditional institutional allocation. Assets are classified by ticker keyword matching (`_EQUITY_KEYWORDS`, `_BOND_KEYWORDS`, etc.) so it works with any standard ticker universe. Unrecognized assets receive 0 weight.

### `RiskParityBaseline` — Inverse Volatility

```python
from portbench.baselines import RiskParityBaseline
baseline = RiskParityBaseline(min_periods=20, vol_window=60)
```

Weights inversely proportional to each asset's annualized volatility:

```
w_i = (1/σ_i) / Σ(1/σ_j)
```

Uses trailing `vol_window`-day return standard deviation from `snapshot.return_data`. Assets with fewer than `min_periods` observations receive the cross-sectional median volatility as a conservative fallback.

Reference: Maillard, Roncalli & Teïletche (2010), *Journal of Portfolio Management*.

### `CovarianceRiskParityBaseline` — Equal Risk Contribution (ERC)

```python
from portbench.baselines import CovarianceRiskParityBaseline
baseline = CovarianceRiskParityBaseline(min_periods=20, max_iter=500, tol=1e-8)
```

Solves the Equal-Risk-Contribution problem using the **full empirical covariance matrix** (not just diagonal volatilities). Each asset's marginal contribution to portfolio variance is driven to `1/N`:

```
RC_i = w_i * (Σ w)_i / (w' Σ w)   →   target RC_i = 1/N
```

Optimized via cyclical coordinate descent on Spinu (2013)'s log-loss
`L(w) = 0.5 w'Σw - (1/N) Σ log(w_i)`, which has a unique long-only minimizer. Weights of assets without enough return history fall back to zero (then re-normalized over the rest).

This is the strong correlation-aware counterpart to `RiskParityBaseline`. Use it whenever `correlation_matrix.csv` / `covariance_matrix.csv` exist in `datasets/processed/` to get a baseline that actually exploits the cross-asset structure the LLM agents are scored on.

Reference: Spinu, F. (2013). *An Algorithm for Computing Risk Parity Weights*.

### `MinVarianceBaseline` — Minimum Variance

```python
from portbench.baselines import MinVarianceBaseline
baseline = MinVarianceBaseline(min_periods=20, ridge=1e-4)
```

Solves the long-only minimum variance problem using the full empirical covariance matrix:

```text
min  w' Σ w
s.t. sum(w) = 1,  w_i >= 0
```

Optimized via SLSQP (scipy). This is the defensive endpoint of the Markowitz efficient frontier — it concentrates weight on the globally least-volatile combination rather than equalizing risk contributions (ERC). Produces more concentrated allocations than ERC when some assets have very low correlation.

Reference: Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance, 7(1), 77–91.

## Usage in EvalPipeline

```python
from portbench.baselines import RiskParityBaseline
from portbench.agent_eval import build_default_pipeline

baseline = RiskParityBaseline()
pipeline = build_default_pipeline(baseline)
result = pipeline.run_episode(snapshot)
```

Or via the evaluation script:
```bash
python examples/agent_eval/run_evaluation.py --baseline risk_parity
python examples/agent_eval/run_evaluation.py --baseline cov_risk_parity
python examples/agent_eval/run_evaluation.py --baseline min_variance
python examples/agent_eval/run_evaluation.py --baseline equal_weight
python examples/agent_eval/run_evaluation.py --baseline sixty_forty
```

## Risk-First Ranking

Baselines are subject to the same stress gate as LLM agents: they must achieve minimum CEPS scores on all three stress scenarios before entering the performance ranking. This ensures that the ranking compares risk-aware strategies on a level playing field.

Expected baseline ordering in normal markets:

```text
MinVariance ≥ CovarianceRiskParity ≥ RiskParity ≥ 60/40 ≥ EqualWeight  (CEPS score)
```
(This is a hypothesis to be validated by the benchmark results.)
