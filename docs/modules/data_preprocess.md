# Data Preprocessing (`portbench/data_preprocess/`)

## Overview

Transforms raw collected data into standardized, analysis-ready CSVs. One preprocessor per asset class, all implementing the abstract `AssetPreprocessor` base. The top-level `process_all_assets()` function runs the full pipeline.

## Output Format

Each preprocessor writes `datasets/processed/<asset_class>.csv` with:

- DatetimeIndex (`date` column)
- Columns named `<source>_<ticker>_<feature>` (e.g., `yahoo_SPY_close`, `fred_DGS10`)
- Computed daily returns: `<source>_<ticker>_return`
- Text data merged as JSON string in `text_json` column (where applicable)

After all per-asset CSVs are written, `examples/data_preprocess/preprocess_all.py` also emits three cross-asset artifacts in the same directory:

- `correlation_matrix.csv` — asset × asset Pearson correlation of daily returns across the common window
- `covariance_matrix.csv` — asset × asset annualized covariance (`daily_cov × 252`)
- `asset_class_map.json` — `{ ticker: asset_class }` mapping derived from the per-asset prefixes; consumed by `MarketSnapshot.asset_class_map` so downstream code can reason about intra-class vs inter-class correlation separately

## Preprocessing Steps

All numeric preprocessors apply these steps in order:

1. **Load & merge** — read all raw CSVs for the asset class, outer-join on date
2. **Business-day reindex** (`TimeAligner.to_business_days()`) — reindex merged DataFrame to `pd.bdate_range` (Mon–Fri) to drop phantom weekend rows introduced by crypto data; skipped in `CryptocurrencyPreprocessor`
3. **Forward-fill** (`NumericPreprocessor.fill_missing()`) — two limits:
   - Daily series (Yahoo, Kaggle): `max_ffill_days = 3` (handles weekends, holidays)
   - Monthly FRED series: `monthly_ffill_days = 31` (handles ~30-day publication gaps); callers pass `freq="M"` when filling FRED DataFrames before the outer-join merge
4. **Winsorization** — clip returns at 1st/99th percentile to remove data errors
5. **Log returns** — compute `log(P_t / P_{t-1})` for each price series
6. **Rolling z-score** — optional normalization over 252-day rolling window
7. **Split labeling** — add `split` column: `train` (2015–2022), `val` (2023), `test` (2024–2025)

## Per-Asset-Class Preprocessors

### `EquitiesPreprocessor`
- Sources: Yahoo OHLCV + Kaggle NASDAQ-100 + SEC filings (text)
- Key computed features: daily returns, 20/50/200-day moving averages, rolling volatility
- Text: SEC 10-K/10-Q filings merged by company and date

### `BondsPreprocessor`
- Sources: Yahoo bond ETF prices + FRED treasury yields and spreads
- Key computed features: yield changes, spread changes, duration-adjusted returns
- Note: FRED yield series are level data (not prices), treated separately from ETF price series; fill with `freq="M"` for monthly FRED series before merge

### `CommoditiesPreprocessor`
- Sources: Yahoo commodity ETF prices + Kaggle spot prices + FRED spot prices
- Key computed features: daily returns, rolling correlations with SPY and TLT
- Note: FRED monthly commodity price series (wheat, corn, copper) use `freq="M"` fill

### `RealEstatePreprocessor`
- Sources: Yahoo REIT ETF prices + FRED Case-Shiller and housing activity series
- Note: Monthly FRED series (CSUSHPINSA, MSPUS, HOUST) are forward-filled with `freq="M"` before being merged with daily Yahoo data

### `CryptocurrencyPreprocessor`
- Sources: Yahoo crypto prices + Kaggle OHLCV datasets
- Key computed features: daily returns, rolling volatility (annualized)
- Text: Kaggle crypto news with sentiment labels
- Note: crypto trades 24/7; this preprocessor sets `resample_to_business_days=False` and retains all calendar days in its own output

### `CashPreprocessor`
- Sources: Yahoo T-bill ETFs + FRED monetary policy and macro series
- Output is primarily used as the macro context input to the QA builder and agent eval pipeline
- Note: Most FRED macro series are monthly; use `freq="M"` fill for FEDFUNDS, CPI, GDP, etc.

## Shared Utilities

### `TimeAligner`

```python
aligner = TimeAligner(config)

# Find overlapping date range across all scanned datasets
# Raises ValueError if no overlap exists (previously silent)
start, end = aligner.find_common_range(datasets)

# Reindex to Mon–Fri business-day calendar after outer-join merge
# Leaves weekday gaps as NaN; subsequent fill_missing() fills them
df = aligner.to_business_days(df, date_col="date")
```

`find_common_range()` raises `ValueError` if `common_start >= common_end`, so empty intersection is never silently propagated as an empty DataFrame.

`to_business_days()` uses `pd.bdate_range` (no holiday calendar — US federal holidays still appear as NaN and are filled by `fill_missing()`).

### `NumericPreprocessor`

```python
processor = NumericPreprocessor(config)

# Daily data (Yahoo, Kaggle) — limit=3
df = processor.fill_missing(df, freq="D")

# Monthly FRED data — limit=31
df_fred = processor.fill_missing(df_fred, freq="M")
```

### `TextPreprocessor`
```python
TextPreprocessor().clean_text(text)  # strips HTML, normalizes whitespace
```

### `PreprocessConfig`

Key time-alignment fields:

| Field | Default | Purpose |
|---|---|---|
| `max_ffill_days` | `3` | Forward-fill limit for daily series |
| `monthly_ffill_days` | `31` | Forward-fill limit for monthly FRED series |
| `resample_to_business_days` | `True` | Drop weekend rows after outer-join merge |
| `train_start / train_end` | `2015-01-01 / 2022-12-31` | Train split boundaries |
| `val_start / val_end` | `2023-01-01 / 2023-12-31` | Val split boundaries |
| `test_start / test_end` | `2024-01-01 / 2025-12-31` | Test split boundaries |

## Running Preprocessing

```bash
python examples/data_preprocess/preprocess_all.py
```

Requires that `examples/data_collect/get_all.py` has been run first.

## Known Behaviors

- **Timezone stripping**: both `_extract_time_range()` and `align_to_dates()` emit a `WARNING` log via Python's `logging` module when timezone information is discarded. Source data should be in UTC or already converted to local time before ingestion to avoid one-day boundary shifts.
- **Crypto calendar**: cryptocurrency outputs retain 365-day calendars. Any cross-asset merge in downstream code (QA builder, agent eval) must call `TimeAligner.to_business_days()` before combining with equity/bond data.
- **Empty overlap**: `find_common_range()` raises `ValueError` (not a silent empty DataFrame) if the computed common range has `start >= end`.
- **Monthly FRED gaps**: series like `CSUSHPINSA`, `FEDFUNDS`, `UNRATE` are published monthly. They must be filled with `fill_missing(freq="M")` (limit=31) before the outer-join merge with daily data; using the default `freq="D"` (limit=3) leaves them almost entirely NaN.

## Design Notes

- **PiT safety**: preprocessing never uses information beyond the current date; rolling windows are computed with `.shift(1)` to avoid look-ahead.
- **Deterministic**: given the same raw data, preprocessing always produces identical outputs.
- **Extensible**: add a new asset class by subclassing `AssetPreprocessor` and registering it in `process_all_assets()`.
