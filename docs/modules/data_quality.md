# Data Quality (`portbench/data_quality/`)

## Overview

Assesses collected and processed data quality before using it for QA generation or evaluation. Produces a structured quality report with PASS/WARN/FAIL ratings and actionable diagnostics.

## Quality Levels

| Level | Meaning |
|-------|---------|
| `PASS` | Meets all quality thresholds |
| `WARN` | Below optimal but usable; review recommended |
| `FAIL` | Below minimum threshold; data should not be used |

Aggregate level = worst individual check level.

## Checkers

### `NumericQualityChecker`

Evaluates numeric (price/yield) CSV files. Five checks:

| Check | What it measures | WARN threshold | FAIL threshold |
|-------|-----------------|---------------|---------------|
| `temporal_coverage` | Fraction of expected trading days present | < 95% | < 85% |
| `stress_coverage` | Coverage in each of 3 stress periods | < 90% | < 70% |
| `split_completeness` | Data present in train/val/test splits | < 95% | < 80% |
| `missing_gap_length` | Max consecutive missing days | > 5 days | > 10 days |
| `price_sanity` | Negative prices / extreme returns (> 50% daily) | any extreme | > 1% of rows |

### `TextQualityChecker`

Evaluates text (news, SEC filings) data. Six checks:

| Check | What it measures |
|-------|-----------------|
| `document_count` | Total number of text documents |
| `avg_text_length` | Average characters per document |
| `empty_doc_rate` | Fraction of empty/near-empty documents |
| `encoding_issues` | Fraction of documents with encoding problems |
| `duplicate_rate` | Fraction of duplicate documents |
| `text_temporal_coverage` | Years with at least one document |

### `CrossAssetQualityChecker`

Orchestrates the full assessment across all six asset classes plus cross-asset checks:

- Runs `NumericQualityChecker` and `TextQualityChecker` on each asset class
- Checks that all asset classes have overlapping date coverage
- Verifies stress period coverage across all assets simultaneously
- Calls `label_market_regimes()` and checks regime label distribution
- `cross_class_correlation_structure` — builds one daily-return series per
  asset class (median across price columns), computes the cross-class
  correlation matrix, and reports the off-diagonal NaN ratio + min/mean/max.
  FAIL when all off-diagonals are NaN, WARN when > 50% are NaN.
  Catches the failure mode where the correlation channel is dead but
  per-asset numeric checks still pass.

### `label_market_regimes()`

```python
from portbench.data_quality import label_market_regimes
regimes = label_market_regimes(spy_prices, lookback=252)
# Returns pd.Series with values "bull" | "bear" | "sideways" | "crisis"
```

Rule-based regime labeling using SPY price series:
- **Crisis**: 6-month drawdown > 20% *or* current drawdown from peak > 20%
- **Bear**: 6-month return < -5%
- **Bull**: 50-day MA > 200-day MA *and* 6-month return > 10%
- **Sideways**: otherwise

Output is saved to `datasets/processed/market_regimes.csv` and used by both QA builders and stress scenario evaluation.

## Configuration

```python
from portbench.data_quality import QualityConfig

config = QualityConfig(
    min_coverage=0.85,          # FAIL below this
    warn_coverage=0.95,         # WARN below this
    stress_periods=[            # Must have good coverage in all three
        ("2015-08-01", "2016-02-29"),
        ("2020-02-01", "2020-05-31"),
        ("2022-05-01", "2022-12-31"),
    ],
    max_gap_days=10,
    max_extreme_return=0.50,
)
```

## Running Quality Assessment

```bash
python examples/data_quality/run_quality_check.py
```

Output: `outputs/quality_reports/report.json`

```json
{
  "overall_level": "WARN",
  "asset_classes": {
    "equities": { "level": "PASS", "checks": [...] },
    "bonds":    { "level": "WARN", "checks": [...] },
    ...
  },
  "generated_at": "2026-04-16T10:00:00"
}
```

## Design Notes

- Modeled after `portbench/data_collect/base.py` — each checker follows the same `DataQualityChecker` ABC pattern.
- The three stress periods are the canonical PortBench evaluation scenarios and receive extra scrutiny.
- `label_market_regimes()` output feeds directly into QA dataset stratification and stress test injection.
