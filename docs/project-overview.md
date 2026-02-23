## PortBench: Multi-Asset Portfolio Management Benchmark

### 1. Overview

**Problem**: Existing portfolio benchmarks are limited to equity markets only.

**Solution**: PortBench provides a multi-asset benchmark spanning six asset classes for AI-driven portfolio management evaluation.

---

### 2. Six Core Asset Classes

| Asset Class | Representative Assets | Key Drivers | Risk Level |
|-------------|----------------------|-------------|------------|
| **Equities** | S&P 500, CSI 300, MSCI EM | Earnings, monetary policy, economic cycle | High ($\sigma \approx$ 15-20%) |
| **Bonds** | US 10Y/2Y Treasury, IG/HY Corporate | Interest rates, inflation, credit cycle | Low-Medium ($\sigma \approx$ 3-8%) |
| **Commodities** | Gold, Crude Oil, Copper | Supply-demand, geopolitics, USD | High ($\sigma \approx$ 15-30%) |
| **Real Estate** | VNQ (REIT ETF) | Interest rates, employment, demographics | Medium ($\sigma \approx$ 10-15%) |
| **Cryptocurrency** | BTC, ETH | Regulation, adoption, liquidity | Very High ($\sigma \approx$ 60-100%) |
| **Cash** | T-Bills, Money Market | Central bank policy | Near Zero |

*$\sigma$ denotes annualized volatility (standard deviation of returns).*

**Cross-Asset Correlation**: Computed from historical return series to quantify diversification benefits across asset classes.

---

### 3. Data Specification

#### 3.1 Temporal Scope

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2015-2022 | Model learning |
| Validation | 2023 | Hyperparameter tuning |
| Test | 2024-2025 | Out-of-sample evaluation |

**Frequency**: Daily OHLCV (hourly for crypto)

#### 3.2 Data Sources

| Asset Class | Primary Source | Format |
|-------------|----------------|--------|
| Equities | Yahoo Finance | CSV |
| Bonds | FRED | CSV |
| Commodities | Yahoo Finance | CSV |
| Real Estate | Yahoo Finance | CSV |
| Crypto | CoinGecko API | JSON |

#### 3.3 Auxiliary Data

Beyond price series, each asset class includes contextual information:

| Asset Class | Auxiliary Data |
|-------------|----------------|
| Equities | News, earnings reports, financial statements, analyst ratings, sector indices |
| Bonds | Yield curves, credit ratings, central bank statements, inflation reports |
| Commodities | Inventory reports, weather data, geopolitical news, supply-demand forecasts |
| Real Estate | Housing indices, mortgage rates, employment data, construction permits |
| Crypto | On-chain metrics, social sentiment, regulatory news, protocol updates |
| Macro | GDP, CPI, unemployment rate, interest rate decisions, PMI |

#### 3.4 Preprocessing

- **Missing values**: Forward-fill (≤3 days), exclude if longer
- **Outliers**: Winsorize at 1st/99th percentile
- **Normalization**: Log returns, rolling z-score (252-day)
- **Alignment**: UTC timezone, US equity calendar as base

---

### 4. QA Dataset Construction

#### 4.1 Question Templates

**T1 - Return Prediction**:
```
Asset: {asset}, Prices (past {window} days): {data}
News: {news}, Macro indicators: {macro}
Predict return direction for next {horizon} days.
```

**T2 - Risk Assessment**:
```
Asset: {asset}, Prices: {data}
Recent events: {news}, Volatility regime: {vol_context}
Compute VaR at {confidence}% confidence level.
```

**T3 - Position Sizing**:
```
Asset: {asset}, Prices: {data}
Fundamentals: {fundamentals}, Market sentiment: {sentiment}
Determine position size with max drawdown ≤ {threshold}%.
```

**T4 - Pairwise Allocation**:
```
Assets: {asset_a}, {asset_b}, Prices: {data_a}, {data_b}
Sector info: {sector}, Correlation context: {corr_news}
Allocate weights to minimize portfolio variance.
```

**T5 - Multi-Asset Optimization**:
```
Assets: {asset_list}, Prices: {data_matrix}
Macro context: {macro}, Constraints: {constraints}
Compute optimal weights maximizing Sharpe ratio.
```

**T6 - Rebalancing Decision**:
```
Current weights: {w_current}, Target weights: {w_target}
Transaction cost: {tc}%, Recent market news: {news}
Determine whether to rebalance given threshold {rebal_threshold}%.
```

**T7 - Regime Detection**:
```
Assets: {asset_list}, Prices: {data_matrix}
Economic indicators: {macro}, Central bank signals: {cb_news}
Identify market regime (bull/bear/sideways) and adjust allocation.
```

#### 4.2 Complexity Levels

| Level | Assets | Templates  | Task                                             |
| ----- | ------ | ---------- | ------------------------------------------------ |
| 1     | 1      | T1, T2, T3 | Single-asset analysis (prediction, risk, sizing) |
| 2     | 2      | T4         | Pairwise allocation                              |
| 3     | 3-4    | T5, T6     | Multi-asset optimization, rebalancing            |
| 4     | All    | T5, T6, T7 | Full portfolio with regime detection             |

#### 4.3 Evaluation Metrics

**Return Metrics**:

| Metric       | Formula                                                 |
| ------------ | ------------------------------------------------------- |
| Total Return | $R_{total} = \frac{P_T - P_0}{P_0}$                     |
| CAGR         | $CAGR = \left(\frac{P_T}{P_0}\right)^{\frac{1}{T}} - 1$ |

**Risk Metrics**:

| Metric       | Formula                                                                          |
| ------------ | -------------------------------------------------------------------------------- |
| Volatility   | $\sigma = \sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(r_t - \bar{r})^2} \times \sqrt{252}$ |
| Max Drawdown | $MDD = \max_{t \in [0,T]} \frac{P_{peak} - P_t}{P_{peak}}$                       |
| VaR (95%)    | $VaR_{0.95} = -\text{Quantile}_{0.05}(r_1, ..., r_T)$                            |
| CVaR (95%)   | $CVaR_{0.95} = -\mathbb{E}[r \mid r \leq -VaR_{0.95}]$                           |

**Risk-Adjusted Metrics**:

| Metric            | Formula                                                |
| ----------------- | ------------------------------------------------------ |
| Sharpe Ratio      | $SR = \frac{\bar{r}_p - r_f}{\sigma_p}$                |
| Sortino Ratio     | $Sortino = \frac{\bar{r}_p - r_f}{\sigma_{downside}}$  |
| Calmar Ratio      | $Calmar = \frac{CAGR}{MDD}$                            |
| Information Ratio | $IR = \frac{\bar{r}_p - \bar{r}_b}{\sigma_{tracking}}$ |

**Allocation Accuracy**:

| Metric               | Formula                                                        |
| -------------------- | -------------------------------------------------------------- |
| Weight MAE           | $MAE_w = \frac{1}{n}\sum_{i=1}^{n}\|w_i^{pred} - w_i^{true}\|$ |
| Portfolio Return Gap | $\Delta R = R_{pred} - R_{optimal}$                            |

---

### 5. AI Training Framework

#### 5.1 RL Formulation

- **State**: $s_t = \{P_{t-n:t}, w_t, {NAV}_t\}$
- **Action**: $a_t = w_{t+1} \in [0,1]^n, \sum_{i=1}^{n} w_i = 1$
- **Reward**: $r_t = R_p - \lambda \cdot \sigma_{downside} - c \cdot \|w_{t+1} - w_t\|_1$

*Notation*: $P_{t-n:t}$ = price history over window $n$; $w_t$ = current weights; ${NAV}_t$ = net asset value; $R_p$ = portfolio return; $\lambda$ = risk penalty coefficient; $\sigma_{downside}$ = downside deviation; $c$ = transaction cost rate.

#### 5.2 LLM Fine-tuning

```json
{
  "input": {"market_data": "...", "constraints": "...", "text_input": "..."},
  "output": {"weights": {...}, "reasoning": "..."}
}
```

Methods: SFT -> GRPO

#### 5.3 Baseline Strategies

| Strategy     | Description                      |
| ------------ | -------------------------------- |
| Equal Weight | $w_i = \frac{1}{n}$              |
| 60/40        | $w_{equity}=0.6, w_{bond}=0.4$   |
| Risk Parity  | $w_i \propto \frac{1}{\sigma_i}$ |

#### 5.4 Tool Calling Support

LLMs can invoke auxiliary tools during evaluation. Tools provide computational support only, not final solutions.

**Available Tools**:

| Tool            | Function Signature                      | Description                     |
| --------------- | --------------------------------------- | ------------------------------- |
| `calculator`    | `calc(expression)`                      | Evaluate arithmetic expressions |
| `correlation`   | `correlation(asset_a, asset_b, window)` | Compute Pearson correlation     |
| `covariance`    | `cov(asset_a, asset_b, window)`         | Compute covariance              |
| `volatility`    | `volatility(asset, window)`             | Compute annualized volatility   |
| `mean_return`   | `mean_return(asset, window)`            | Compute mean return             |


**Evaluation Modes**:

| Mode          | Tool Access | Purpose               |
| ------------- | ----------- | --------------------- |
| No-Tool       | Disabled    | Test pure reasoning   |
| Tool-Assisted | Enabled     | Test tool utilization |

---

