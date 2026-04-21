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
provider = ProcessedDataProvider(data_dir="datasets/processed", sec_dir="datasets/sec")
```
Reads real `datasets/processed/*.csv` files. Drop-in replacement for `MockDataProvider`.

**Price column lookup** (`_find_column`): tries `yahoo_<TICKER>_close` → `kaggle_<TICKER>_close` → fuzzy match on ticker name. Handles `BTC-USD` ↔ `BTC_USD` naming variants. Falls back to `_open` columns when `_close` is missing (some preprocessed CSVs drop `_close` after de-duplication).

**Duplicate-date handling**: `_load_csv` collapses duplicate date rows to first occurrence per date (`equities.csv` may contain duplicates from the SEC text-record explosion).

**Text retrieval** (`get_news`, `has_text`): see [Text Context Pipeline](#text-context-pipeline) below.

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

## Text Context Pipeline

Because PortBench evaluates **agents**, not time-series predictors, news/filing text is treated as a first-class context input alongside numeric features. Text is injected into the QA `question` field whenever available.

### Text source priority (`ProcessedDataProvider.get_news`)

For an asset and decision date, `get_news()` returns the most recent text strictly before the decision date, falling back through four sources:

1. **Preprocessed `text_json` column** in `equities.csv` / `cryptocurrency.csv` — JSON-aggregated SEC + Kaggle records, produced by the preprocessing pipeline (canonical, fastest)
2. **Raw SEC 10-K / 10-Q htm files** under `datasets/sec/equities/<TICKER>/` — for equities not yet covered by preprocessing
3. **Kaggle `stock-data-with-news`** per-ticker CSVs — daily news column for 99 equity tickers
4. **Kaggle `crypto-news`** — 2021–2023 crypto news headlines (cryptocurrency assets only)

Returns empty string for assets in non-text classes (bonds, commodities, real estate, cash).

### Cheap probe: `has_text(asset, before_date) -> bool`

Used by the build script to rank candidate dates without running the full text-retrieval pipeline. Returns True iff the loaded `text_json` frame for the asset's class has any non-null record before `before_date`. The default implementation on `DataProvider` returns False; `ProcessedDataProvider` overrides it.

### Pipeline integration

`DataProvider.build_context()` automatically calls `get_news(assets[0], decision_date)` and stores the result in `ContextWindow.news_text`. Each template appends this to the question:

```python
question = (
    f"Asset: {asset}\n..."
    + (f"Recent filing/news:\n{context.news_text}\n" if context.news_text else "")
    + "\nPredict whether the return..."
)
```

`QABuilder.build()` injects `metadata["has_text"]` and `metadata["text_chars"]` on every QAPair so `stats.json` can report text coverage.

## QA Generation Strategy

The build pipeline is designed to **maximize text coverage** (since the benchmark is for agent evaluation, not time-series prediction) while remaining **adaptive to data feasibility** (templates that need 3+ aligned assets generate as many pairs as the data supports, up to a per-template cap).

### 1. Text-first date ordering

`build_qa_dataset.py` partitions each split's business-day candidate list into two buckets via `provider.has_text("SPY", d) or provider.has_text("BTC-USD", d)`:
- **text-bearing dates** (most dates after 2015)
- **plain dates** (early dates before any news data starts)

Each bucket is shuffled deterministically (seeded by `config.random_seed`), then concatenated text-first. The three splits are then **round-robin interleaved** so that hitting the per-template cap early still yields balanced train/val/test coverage.

### 2. Text-bearing asset preference per template

`_select_assets()` in each template biases class selection toward text-bearing classes (`equities`, `cryptocurrency`):

| Template | Asset selection rule |
|----------|---------------------|
| T1 / T2 / T3 (single asset) | 80% pick from {equities, crypto}, 20% from {bonds, commodities, real_estate, cash} |
| T4 (2 assets) | First asset always from {equities, crypto}; second is 50/50 text-class vs other-class |
| T5 (3+ assets) | Always include both equities + crypto, fill remaining slots from other classes |
| T6 (4 assets) | Same as T5 |
| T7 (regime detection) | Always uses equities (regime signal) — no change needed |

Per-date determinism is preserved via `random.Random(hash(decision_date) + offset)`.

### 3. Adaptive per-template cap

`samples_per_template` (default 1000) is now a **cap, not a target**. `QABuilder.build(n, decision_dates)` iterates candidate dates and:
- Skips dates outside any configured split
- Skips dates where `_select_assets()` or `build_context()` raises (e.g., insufficient price history)
- Stops at `n` pairs OR when all candidates are exhausted

Templates with high feasibility (T1–T4, single or pair-asset) typically reach the cap. Templates needing 3+ aligned assets (T5/T6) generate fewer — currently ~500 each on the real dataset.

### 4. Text statistics in `stats.json`

Every template entry includes a `text` block, plus a global `_meta.text_overall`:

```json
"T1": {
  "n_total": 1000,
  "by_split": {"train": 700, "val": 200, "test": 100},
  "by_regime": {...},
  "text": {
    "n_with_text": 693,
    "pct_with_text": 69.3,
    "avg_chars": 1024.5,
    "max_chars": 1528,
    "min_chars": 20
  }
}
```

### Observed coverage

On the current real dataset (2015–2025):

| Template | Pairs | With text | Coverage |
|----------|-------|-----------|----------|
| T1 / T2 / T3 | 1000 each | ~700 | ~70% |
| T4 | 1000 | 884 | 88% |
| T5 | 509 | 509 | 100% |
| T6 | 464 | 464 | 100% |
| T7 | 491 | 491 | 100% |
| **Total** | **5464** | **4436** | **81%** |

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
  "question": "Given the 60-day market context for SPY ending 2020-03-15, predict the return direction over the next 21 trading days.\nRecent filing/news:\n[SEC 10-K AAPL 2020-01-30] ...",
  "answer": "down",
  "answer_numeric": -0.342,
  "explanation": "S&P 500 fell 34% in 33 days during COVID-19 crash...",
  "metadata": {
    "has_text": true,
    "text_chars": 1487,
    "future_return": -0.342,
    "horizon_days": 21
  }
}
```

## Point-in-Time (PiT) Constraint

`ContextWindow.validate_pit()` raises `ValueError` if any data in `price_history` or `returns_history` has an index >= `decision_date`. All `DataProvider.build_context()` calls invoke this automatically.

## Building the Dataset

```bash
python examples/qa_builder/build_qa_dataset.py
```

Requires `datasets/processed/` populated by `examples/data_preprocess/preprocess_all.py` first.

Output:
```
outputs/qa_dataset/
├── all_pairs.jsonl    # Complete dataset
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── stats.json         # Template × regime × split + text coverage + split boundary metadata
```

The build script:
1. Loads `ProcessedDataProvider`, derives split boundaries from actual data range
2. Ranks candidate dates text-first per split, round-robin interleaves splits
3. For each template, calls `builder.build(n=samples_per_template, decision_dates=...)`
4. Writes per-split JSONL files and `stats.json` with text statistics

## Programmatic Usage

```python
from portbench.qa_builder import ProcessedDataProvider, QAConfig, get_all_builders
from datetime import date

provider = ProcessedDataProvider(data_dir="datasets/processed", sec_dir="datasets/sec")
config = QAConfig.from_date_range(
    data_start=date(2015, 1, 1),
    data_end=date(2025, 12, 31),
    samples_per_template=1000,  # cap, not target — actual count adapts to data feasibility
)
builders = get_all_builders(provider, config)

import pandas as pd, random
dates = pd.bdate_range(config.train_start, config.test_end).date.tolist()
# Optional: rank text-first to maximize text coverage
text_dates = [d for d in dates if provider.has_text("SPY", d) or provider.has_text("BTC-USD", d)]
plain_dates = [d for d in dates if d not in set(text_dates)]
random.Random(42).shuffle(text_dates)
random.Random(43).shuffle(plain_dates)
ranked = text_dates + plain_dates

all_pairs = []
for builder in builders:
    pairs = builder.build(n=1000, decision_dates=ranked)
    all_pairs.extend(pairs)
```
