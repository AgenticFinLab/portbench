# Data Collection (`portbench/data_collect/`)

## Overview

Collects raw financial data from four sources across all six asset classes. Each collector implements the abstract `DataCollector` base class, ensuring a consistent interface.

## Asset Class Coverage

| Asset Class | Yahoo Finance | FRED | Kaggle | SEC |
|-------------|:---:|:---:|:---:|:---:|
| Equities | ✅ 29 tickers | — | ✅ NASDAQ-100 history, news | ✅ 20 companies |
| Bonds | ✅ 15 tickers | ✅ 22 series | — | — |
| Commodities | ✅ 15 tickers | ✅ 9 series | ✅ Gold, crude oil, multi-commodity | — |
| Real Estate | ✅ 10 tickers | ✅ 9 series | ✅ Redfin housing data | — |
| Cryptocurrency | ✅ 12 tickers | — | ✅ OHLCV top-50 + 5000 coins | — |
| Cash / Macro | ✅ 4 tickers | ✅ 20 series | — | — |

## Collectors

### `YahooCollector` — 72 tickers

Downloads OHLCV daily price data via `yfinance`. Tickers are defined in `YAHOO_TICKERS` and cover:

- **Equities**: US broad market (SPY/QQQ/IWM/VTI), international (EFA/EEM/FXI/EWJ/ACWI), 10 SPDR sector ETFs, 4 factor ETFs, VIX
- **Bonds**: Full maturity spectrum (SHV→TLT), TIPS inflation-protected (TIP/SCHP/STIP), corporate (LQD/HYG/JNK), international (BNDX/EMB/IGOV)
- **Commodities**: Precious metals (GLD/SLV/PPLT/PALL), energy (USO/UNG/BNO), agriculture (DBA/CORN/WEAT/SOYB), broad index + base metals (DBC/PDBC/DBB/CPER)
- **Real Estate**: Broad REITs (VNQ/IYR/SCHH), sub-sector (REZ/MORT/INDS), international (VNQI/HAUZ), homebuilders (XHB/ITB)
- **Crypto**: BTC, ETH, BNB, SOL, XRP, ADA, AVAX, DOT, LINK, MATIC, UNI, AAVE
- **Cash**: BIL, SGOV, CSHI, ICSH

```python
from portbench.data_collect import YahooCollector, AssetClass
collector = YahooCollector(base_dir="datasets", start_date="2015-01-01")
collector.download_all()
# or by asset class:
collector.download_by_asset_class(AssetClass.BONDS)
```

### `FREDCollector` — 60 series

Downloads economic time series from the Federal Reserve Economic Data API. Requires `FRED_API_KEY` in `.env`.

Series are organized into four groups:

| Group | Series count | Key series |
|-------|-------------|-----------|
| Bonds — Yield curve | 12 | DGS1MO → DGS30, T10Y2Y, T10Y3M |
| Bonds — TIPS & breakeven | 6 | DFII5/10/30, T5YIE/T10YIE, T5YIFR |
| Bonds — Credit & mortgage | 7 | BAMLH0A0HYM2, TEDRATE, MORTGAGE30/15US |
| Real Estate — Prices | 4 | CSUSHPINSA, CSUSHPISA, SPCS20RSA, MSPUS |
| Real Estate — Activity | 5 | HOUST, PERMIT, HSN1F, EXHOSLUSM495S, MSACSR |
| Commodities — Spot prices | 9 | DCOILWTICO, DCOILBRENTEU, GOLDPMGBD228NLBM, wheat/corn/soybean/copper |
| Cash — Monetary policy | 5 | DFF, FEDFUNDS, SOFR, IOER, WALCL (Fed balance sheet) |
| Cash — Inflation | 4 | CPIAUCSL, CPILFESL, PCEPILFE, PCEPI |
| Cash — Growth & labour | 7 | GDP, GDPC1, INDPRO, TCU, UNRATE, PAYEMS, ICSA |
| Cash — Sentiment & credit | 6 | UMCSENT, USSLIND, M2SL, M2V, DPSACBW027SBOG, TOTCI |

```python
from portbench.data_collect import FREDCollector
collector = FREDCollector(base_dir="datasets")  # reads FRED_API_KEY from .env
collector.download_all()
```

### `KaggleCollector` — 10 datasets

Downloads curated Kaggle datasets via `kagglehub`. Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`.

| Dataset | Asset Class | Content |
|---------|------------|---------|
| Crypto OHLCV top-50 | CRYPTOCURRENCY | Daily prices 2013–present |
| Crypto 5000 coins | CRYPTOCURRENCY | Broader altcoin coverage |
| Crypto news | CRYPTOCURRENCY | Text news with sentiment labels |
| Gold prices | COMMODITIES | Spot gold price history |
| WTI crude oil | COMMODITIES | Daily WTI price |
| Multi-commodity | COMMODITIES | 7 commodities (gold, silver, platinum, palladium, brent, natgas, WTI) |
| NASDAQ-100 history | EQUITIES | 514K rows OHLCV data |
| Stock + news | EQUITIES | Combined price and news text |
| Google finance news | EQUITIES | Text news corpus |
| US housing (Redfin) | REAL_ESTATE | Monthly housing market metrics 2012–present |

### `SECCollector` — 20 companies

Downloads SEC filings (10-K annual, 10-Q quarterly) as raw HTML from the EDGAR API. Requires `SEC_API_KEY` in `.env`.

Companies covered: AAPL, MSFT, GOOGL, AMZN, NVDA, META (Tech); JPM, BAC, GS, BLK (Finance); JNJ, UNH, PFE (Healthcare); PG, KO, WMT (Consumer); XOM, CVX (Energy); CAT, BA (Industrial).

## Output Format

```
datasets/
├── yahoo/<asset_class>/<TICKER>.csv          # OHLCV: date,open,high,low,close,volume
├── fred/<asset_class>/<SERIES_ID>.csv        # date,value
├── kaggle/<asset_class>/<dataset>/           # raw files
└── sec/equities/<TICKER>/<filing_type>/      # HTML files
```

## Running Data Collection

```bash
# All sources
python examples/data_collect/get_all.py

# Individual source
python -c "
from portbench.data_collect import YahooCollector
YahooCollector('datasets').download_all()
"
```

## Design Notes

- **Point-in-time safety**: All data is indexed by `date`. The preprocessing layer enforces that no future data leaks into training windows.
- **Retry logic**: All collectors retry up to 3 times with exponential backoff on transient API errors.
- **Idempotent**: Re-running a collector skips already-downloaded files unless `force=True`.
- **Rate limiting**: 1-second sleep between API calls to respect rate limits.
