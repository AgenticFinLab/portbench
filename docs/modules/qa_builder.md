# QA Dataset Builder (`portbench/qa_builder/`)

## Overview

Generates the PortBench QA dataset: structured question-answer pairs derived from historical (or synthetic) market data. Seven question templates (T1–T7) cover four complexity levels and produce `QAPair` objects stratified by market regime and train/val/test split.

## Data Providers

Two implementations of the `DataProvider` ABC:

### `MockDataProvider` (development, no API keys needed)
```python
from portbench.qa_builder import MockDataProvider
provider = MockDataProvider(seed=42)
```
Generates synthetic price paths via Geometric Brownian Motion for 12 predefined assets (SPY, QQQ, EEM, TLT, IEF, LQD, GLD, USO, VNQ, BTC, ETH, BIL). Regime is assigned from 6-month trailing SPY returns. Macro indicators are deterministic sinusoidal functions of time.

### `ProcessedDataProvider` (production, requires `datasets/processed/`)
```python
from portbench.qa_builder import ProcessedDataProvider
provider = ProcessedDataProvider(data_dir="datasets/processed")
```
Reads real `datasets/processed/*.csv` files. Drop-in replacement for `MockDataProvider`. Column lookup searches for `yahoo_<TICKER>_close` → `kaggle_<TICKER>_close` → any matching column. Supports `market_regimes.csv` for pre-computed regime labels.

## Dataset Splits

| Split | Date Range | Purpose |
|-------|-----------|---------|
| `train` | first 70% of data timeline | Model fine-tuning |
| `val` | next 15% | Hyperparameter selection |
| `test` | remaining 15% | Final benchmark evaluation |

Splits are computed **dynamically** from the actual data timeline via `QAConfig.from_date_range()`, rather than being hardcoded. Boundaries snap to year-ends for clean cutoffs. The computed boundaries are saved to `outputs/qa_dataset/stats.json` under `_meta` for reproducibility.

```python
from portbench.qa_builder.base import QAConfig
from datetime import date

# Auto-compute splits from actual data range (recommended)
config = QAConfig.from_date_range(
    data_start=date(2015, 1, 1),
    data_end=date(2025, 12, 31),
    train_frac=0.70,
    val_frac=0.15,
)
print(config.describe())
# → "Train: 2015-01-01 – 2022-12-31  |  Val: 2023-01-01 – 2024-12-31  |  Test: 2025-01-01 – 2025-12-31"

# Or override with explicit dates (backwards-compatible)
config = QAConfig(train_start="2015-01-01", train_end="2022-12-31", ...)
```

## Question Templates

| Template | Task | Complexity | Assets |
|----------|------|-----------|--------|
| T1 `T1ReturnPrediction` | Predict return direction (up/down/flat) for next N days | 1 | Single |
| T2 `T2RiskAssessment` | Estimate VaR and CVaR at given confidence | 1 | Single |
| T3 `T3PositionSizing` | Compute max position size given drawdown limit | 1 | Single |
| T4 `T4PairwiseAllocation` | Minimum-variance allocation across 2 assets | 2 | Pair |
| T5 `T5MultiAssetOptimization` | Maximum-Sharpe allocation across 3+ assets (SLSQP) | 3 | Multi |
| T6 `T6RebalancingDecision` | Decide whether to rebalance given current vs. target weights | 3 | Multi |
| T7 `T7RegimeDetection` | Detect regime and recommend allocation direction | 4 | Full portfolio |

### Ground Truth Computation

Each template computes ground truth from observable data strictly before `decision_date` (PiT constraint):

- **T1**: Future return from `decision_date` to `decision_date + horizon_days` (uses future prices — only the answer uses future data, not the context)
- **T2**: Historical simulation VaR/CVaR from 252-day return window
- **T3**: Fixed-fractional: `f* = max_drawdown_limit / |VaR(99%)|`, capped at 1.0
- **T4**: Analytic min-variance: `w1* = (σ₂² - σ₁₂) / (σ₁² + σ₂² - 2σ₁₂)`
- **T5**: SLSQP optimizer: `max E[r]ᵀw / sqrt(wᵀΣw)` subject to `Σwᵢ=1, wᵢ≥0`
- **T6**: `|w_current - w_target| > threshold` → "rebalance"; else → "hold"
- **T7**: Regime from trailing returns; allocation direction from `_REGIME_ALLOCATION` lookup

## ContextWindow and Cross-Asset Correlation

`ContextWindow` is the central data object passed to every template. It now includes a `correlation_matrix` field:

```python
@dataclass
class ContextWindow:
    decision_date: date
    assets: list[str]
    price_history: dict[str, pd.Series]
    returns_history: dict[str, pd.Series]
    macro_context: dict[str, float]
    market_regime: Optional[MarketRegime]
    correlation_matrix: Optional[pd.DataFrame]   # assets × assets, Pearson
```

`DataProvider.build_context()` automatically computes the correlation matrix from `returns_history` for any multi-asset context window. Templates T4, T5, and T7 use this matrix in their ground truth computation (min-variance and Sharpe-maximization require the full covariance structure). The correlation matrix is also included in the evaluation context provided to LLMs, enabling the model to reason about cross-asset diversification.

```python
ctx = provider.build_context(date(2020, 3, 15), ["SPY", "TLT", "GLD"], lookback_days=60)
print(ctx.correlation_matrix)
#          SPY       TLT       GLD
# SPY  1.000000 -0.623451  0.012344
# TLT -0.623451  1.000000  0.234567
# GLD  0.012344  0.234567  1.000000
```

## QAPair Format

```json
{
  "qa_id": "T1_equities_20200315_001",
  "template_id": "T1",
  "complexity": 1,
  "split": "train",
  "market_regime": "crisis",
  "asset_class": "equities",
  "assets": ["SPY"],
  "decision_date": "2020-03-15",
  "context_summary": "...",
  "question": "Given the 60-day market context for SPY ending 2020-03-15, predict the return direction over the next 21 trading days.",
  "answer": "down",
  "answer_numeric": -0.342,
  "explanation": "S&P 500 fell 34% in 33 days during COVID-19 crash...",
  "metadata": {}
}
```

## Point-in-Time (PiT) Constraint

`ContextWindow.validate_pit()` raises `ValueError` if any data in `price_history` or `returns_history` has an index >= `decision_date`. All `DataProvider.build_context()` calls invoke this automatically.

## Building the Dataset

```bash
python examples/qa_builder/build_qa_dataset.py
```

Output:
```
outputs/qa_dataset/
├── all_pairs.jsonl    # Complete dataset
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── stats.json         # Template × regime × split distribution + split boundary metadata
```

## Programmatic Usage

```python
from portbench.qa_builder import MockDataProvider, QAConfig, get_all_builders
from datetime import date

provider = MockDataProvider(seed=42)
config = QAConfig.from_date_range(
    data_start=date(2015, 1, 1),
    data_end=date(2025, 12, 31),
    samples_per_template=100,
)
builders = get_all_builders(provider, config)

import pandas as pd, random
dates = pd.bdate_range(config.train_start, config.test_end).date.tolist()[::5]
random.shuffle(dates)

all_pairs = []
for builder in builders:
    pairs = builder.build(n=100, decision_dates=dates)
    all_pairs.extend(pairs)
```
