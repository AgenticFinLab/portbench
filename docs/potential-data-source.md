## Potential Data Sources for PortBench

### 1. Websites

#### Yahoo Finance
- **URL**: https://finance.yahoo.com/
- **Description**: Free financial data provider with historical OHLCV data for stocks, ETFs, indices, commodities, and cryptocurrencies. Accessible via `yfinance` Python library.
- **Relevant Assets**: Equities (SPY, CSI 300 ETFs), Bonds (TLT, SHV), Commodities (GLD, DBC), Real Estate (VNQ), Crypto (BTC-USD, ETH-USD)

#### FRED (Federal Reserve Economic Data)
- **URL**: https://fred.stlouisfed.org/
- **Description**: St. Louis Fed's database with 800,000+ economic time series. Covers GDP, CPI, unemployment, interest rates, yield curves, and monetary policy data since 1991.
- **Relevant Assets**: Bonds (Treasury yields, interest rates), Macro indicators (GDP, CPI, unemployment rate, Fed funds rate)

#### Portfolio Visualizer
- **URL**: https://www.portfoliovisualizer.com/
- **Description**: Portfolio analysis and backtesting platform with historical asset class returns, correlation matrices, and factor analysis tools.
- **Relevant Assets**: Multi-asset allocation benchmarks, historical performance data

#### Crypto News 
- **URL**: https://cryptonews.com; https://cryptopotato.com; https://cointelegraph.com
- **Description**: Crypto news and analysis from multiple sources
- **Relevant Assets**: Crypto news and analysis

---

### 2. Kaggle Datasets

#### Cryptocurrency

- **Crypto Data 2014-2026**
  - URL: https://www.kaggle.com/datasets/kaushalnandania/crypto-data-2014-2026/data
  - Data Type: Numeric
  - Period: September 2014 - February 2026
  - Description: Daily OHLCV for top 50 cryptocurrencies by market cap. Covers: 2017 ICO boom, 2018 crash, 2020-2021 DeFi/NFT bull run, 2022 crypto winter, 2024-2025 recovery

- **Crypto Quants Dataset**
  - URL: https://www.kaggle.com/datasets/emranalbiek/crypto-quants-dataset
  - Data Type: Numeric
  - Source: CoinMarketCap API
  - Description: 95,100+ samples covering ~5,000 unique cryptocurrencies

- **Crypto News Dataset**
  - URL: https://www.kaggle.com/datasets/oliviervha/crypto-news/data
  - Data Type: Text
  - Source: cryptonews.com; cryptopotato.com; cointelegraph.com
  - Description: Crypto news data from over a year (2021-10-12 / 2023-12-19) in a structured format including title, text, source, subject, and sentiment analysis. 

#### Commodities

- **Gold Price Dynamics**
  - URL: https://www.kaggle.com/datasets/ayeshaimran1619/gold-price-dynamics-and-market-behavior
  - Data Type: Numeric
  - Period: 2016-01-29 - 2026-01-23
  - Description: Daily gold price data from 2016 to 2026. Features: OHLCV, moving averages, daily returns, volatility

- **Crude Oil Price**
  - URL: https://www.kaggle.com/datasets/sc231997/crude-oil-price
  - Data Type: Numeric
  - Period: 1983-03-01 - 2026-02-01
  - Description: Crude Oil WTI (USD/Bbl) historical data

- **7 Commodities — Multi-Timeframe Market Data**
  - URL: https://www.kaggle.com/datasets/anthonygocmen/8-commodities-multi-timeframe-market-data
  - Data Type: Numeric
  - Source: IC Markets using Python and the MetaTrader 5 API
  - Period: 2016-10-12 17:00 - 2025-12-08 10:00
  - Description: This dataset contains historical market data for 7 major commodities: XAUUSD (Gold); XAGUSD (Silver); XPDUSD (Palladium); XPTUSD (Platinum); XBRUSD (Brent Crude); XNGUSD (Natural Gas); XTIUSD (WTI Crude)

#### Equities

- **NASDAQ 100 Historical**
  - URL: https://www.kaggle.com/datasets/jacksaleeby/nasdaq100-historical-data-2000-2026-upvote
  - Data Type: Numeric
  - Period: January 2000 - February 2026
  - Description: All 100 NASDAQ-100 constituents. 514,000+ rows of split-adjusted price and volume data

- **Stock Data with News**
  - URL: https://www.kaggle.com/datasets/ekalabyaghosh/stock-data-with-news
  - Data Type: Text
  - Period: 1980-12-12 - 2026-02-19
  - Description: 99 stock OHLCV data and news data

- **Google Stock Financial News: 2000–Today**
  - URL: https://www.kaggle.com/datasets/emrekaany/google-googl-financial-news-from-2000-to-today
  - Data Type: Text
  - Period: 2000-01-01 - 2026-02-24
  - Source: Yahoo Finance
  - Description: daily news articles about Alphabet Inc. (NASDAQ: GOOGL) from January 1, 2000

#### Real Estate

- **US Cities Housing Market**
  - URL: https://www.kaggle.com/datasets/vincentvaseghi/us-cities-housing-market-data
  - Data Type: Numeric
  - Source: Redfin
  - Period: February 2012 - present (monthly updates)

---

### 3. Academic Paper Datasets

#### Paper 1: MASS (Multi-Agent Simulation Scaling)

- **Paper**: https://arxiv.org/abs/2505.10278
- **Code**: https://github.com/gta0804/MASS
- **Market**: Chinese A-share (SSE 50, CSI 300, ChiNext 100)
- **Period**: Full year 2023 (high volatility)
- **Data Types**: News, financial reports, price-volume features, fundamentals, macro indicators
- **Status**: Open-sourced

#### Paper 2: SmartFolio (IJCAI 2025)

- **Code**: https://github.com/ChloeWenyiZhang/SmartFolio
- **Markets**: CSI 300, CSI 500, NASDAQ 100, S&P 500
- **Period**: 2018-2024 (Train: 2018-2022, Val: 2023, Test: 2024)
- **Data Types**:
  - Daily OHLCV + previous close
  - Rolling-window normalized features (20-day)
  - K-means clustered groups
  - Monthly Pearson correlation matrices
  - Positive/negative correlation graphs (threshold: ±0.2)
- **Note**: ~5% stocks discarded due to incomplete data

#### Paper 3: InvestorBench (ACL 2025)

- **Paper**: https://arxiv.org/abs/2412.18174
- **Code**: https://github.com/felis33/INVESTOR-BENCH
- **Environments**:
  - **Stock Market** (2020-07-01 to 2021-05-06): Yahoo Finance OHLCV, SEC EDGAR (10-Q, 10-K), News for MSFT/JNJ/UVV/HON/TSLA/AAPL/NIO, GPT-3.5 sentiment
  - **Crypto Market** (2023-02-13 to 2023-11-05): CoinMarketCap OHLCV, News from cryptonews/cryptopotato/cointelegraph, GPT-3.5 sentiment
  - **ETF Market** (2019-07-29 to 2020-09-21): NIFTY dataset (news headlines + sentiment)

---

### 4. Coverage Summary

- **Equities**: Yahoo Finance, SmartFolio, InvestorBench | Text: SEC filings, News, Fundamentals (MASS)
- **Bonds**: FRED (Treasury yields) | Text: Central bank statements
- **Commodities**: Kaggle (Gold, Oil) | Text: -
- **Real Estate**: Kaggle (US housing) | Text: -
- **Cryptocurrency**: Kaggle, CoinMarketCap, InvestorBench | Text: News + sentiment
- **Macro**: FRED | Text: GDP, CPI, unemployment, interest rates
