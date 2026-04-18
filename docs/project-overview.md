## PortBench: Multi-Asset Portfolio Management Benchmark

### 1. Research Problem

Existing financial LLM benchmarks share three systematic gaps that PortBench is designed to close:

**Gap 1 — Evaluation object mismatch**: All existing benchmarks (FinBen, InvestorBench, QuantBench) evaluate single-asset buy/sell decisions. Real portfolio management requires joint weight allocation across heterogeneous asset classes, correlation-aware diversification, and risk budgeting — none of which are captured by per-asset decision tasks.

**Gap 2 — Return-first evaluation paradigm**: Every existing benchmark ranks models by return metrics (Sharpe ratio, CAGR). Banerjee et al. (Standard Benchmarks Fail, 2025) show this creates a false reliability illusion — models with high returns can be highly vulnerable to hallucinated facts, stale data, and adversarial prompts. Portfolio management's first principle is capital preservation, not return maximization.

**Gap 3 — Isolated subtask evaluation**: Portfolio management is a multi-stage sequential pipeline. Liu et al. (FinMaster, 2025) demonstrate that accuracy above 90% on isolated tasks drops to 40% in complex multi-step pipelines. No existing benchmark measures cross-stage error propagation in end-to-end portfolio workflows.

---

### 2. PortBench Design

PortBench addresses the three gaps through a unified benchmark with two evaluation components.

#### 2.1 Six Asset Classes

| Asset Class | Representative Instruments | Risk Level |
|-------------|---------------------------|------------|
| Equities | S&P 500, NASDAQ-100, MSCI EM | High (σ ≈ 15–20%) |
| Bonds | US 10Y/2Y Treasury, IG/HY Corporate | Low–Medium (σ ≈ 3–8%) |
| Commodities | Gold, Crude Oil, Silver | High (σ ≈ 15–30%) |
| Real Estate | VNQ, IYR (REIT ETFs) | Medium (σ ≈ 10–15%) |
| Cryptocurrency | BTC, ETH | Very High (σ ≈ 60–100%) |
| Cash | T-Bills, Fed Funds Rate | Near Zero |

#### 2.2 Component 1 — QA Dataset

Structured question-answer pairs generated from historical market data. Each QA pair contains: a point-in-time context window (price history + news + macro), a question, a ground-truth answer, and an explanation.

**Seven Question Templates**:

| Template | Task | Complexity |
|----------|------|------------|
| T1 | Return prediction — direction for next N days | Level 1 |
| T2 | Risk assessment — VaR at given confidence level | Level 1 |
| T3 | Position sizing — given max drawdown constraint | Level 1 |
| T4 | Pairwise allocation — minimize variance for 2 assets | Level 2 |
| T5 | Multi-asset optimization — maximize Sharpe for 3+ assets | Level 3 |
| T6 | Rebalancing decision — threshold-based trigger | Level 3 |
| T7 | Regime detection — identify bull/bear/sideways + adjust allocation | Level 4 |

The dataset is stratified by train/val/test split and by market state (bull/bear/sideways/crisis).

**Evaluation Metrics for QA**:

*Return*: Total Return, CAGR

*Risk*: Volatility (σ), Max Drawdown, VaR(95%), CVaR(95%)

*Risk-Adjusted*: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Information Ratio

*Allocation Accuracy*: Weight MAE = (1/n) Σ|wᵢ_pred − wᵢ_true|, Portfolio Return Gap = R_pred − R_optimal

#### 2.3 Component 2 — End-to-End Agent Evaluation

LLM agents are evaluated across the full five-stage investment pipeline in a MarS-based simulation sandbox:

| Stage | Task |
|-------|------|
| S1 Market Interpretation | Parse price data + news into structured market views |
| S2 Signal Generation | Produce directional signals and confidence scores per asset |
| S3 Weight Optimization | Translate signals into portfolio weight allocations |
| S4 Execution Simulation | Simulate trades with transaction costs and slippage |
| S5 Risk Monitoring | Monitor live portfolio risk; trigger rebalancing if thresholds breached |

**CEPS (Cross-stage Error Propagation Score)**: Quantifies how errors at each stage amplify through the pipeline. Measured as the deviation between each stage's output and the ground-truth optimal output for that stage.

#### 2.4 Risk-First Evaluation Protocol

Models must pass stress tests before entering performance rankings:

| Stress Period | Date Range | Event |
|---------------|-----------|-------|
| 2008 Crisis | 2008-09 – 2009-03 | Global Financial Crisis |
| 2020 COVID | 2020-02 – 2020-05 | COVID-19 Flash Crash |
| 2022 Crypto | 2022-05 – 2022-12 | Crypto Collapse |

A secondary layer evaluates personalized advice for three investor profiles: **conservative**, **balanced**, **aggressive** — assessing whether models adjust recommendations to match individual risk tolerance and investment horizon.

---

### 3. Data Specification

#### 3.1 Temporal Scope

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2015–2022 | Model learning |
| Validation | 2023 | Hyperparameter tuning |
| Test | 2024–2025 | Out-of-sample evaluation |

Frequency: daily OHLCV (primary); hourly for cryptocurrency.

#### 3.2 Data Sources

| Source | Content | Asset Classes |
|--------|---------|---------------|
| Yahoo Finance | OHLCV price series | Equities, Bonds, Commodities, Real Estate, Crypto, Cash |
| FRED | Macroeconomic indicators, yield curves | Bonds, Cash |
| Kaggle | Historical datasets, news corpora | Equities, Commodities, Crypto, Real Estate |
| SEC EDGAR | 10-K / 10-Q filings (text) | Equities |

#### 3.3 Key Design Constraints

- **Point-in-Time (PiT)**: Any feature at decision timestamp t must use only information available at t. Validated via alpha decay analysis (Look-Ahead-Bench methodology).
- **Market state partitioning**: Test data is labeled bull/bear/sideways/crisis to enable per-state performance decomposition.
- **Framework variable control**: Agent architecture choices are held constant across model comparisons; only the LLM backbone varies.

---

### 4. Baseline Strategies

| Strategy | Allocation Rule |
|----------|----------------|
| Equal Weight | wᵢ = 1/n |
| 60/40 | w_equity = 0.6, w_bond = 0.4 |
| Risk Parity | wᵢ ∝ 1/σᵢ |
| SmartFolio | IJCAI 2025 SOTA (non-LLM upper bound) |

---

### 5. Tool Calling Support

LLMs may invoke auxiliary tools during evaluation. Two modes are tested:

| Mode | Tool Access | Purpose |
|------|-------------|---------|
| No-Tool | Disabled | Test pure reasoning |
| Tool-Assisted | Enabled | Test tool utilization |

Available tools:

| Tool | Signature | Description |
|------|-----------|-------------|
| `calculator` | `calc(expression)` | Evaluate arithmetic expressions |
| `correlation` | `correlation(a, b, window)` | Compute Pearson correlation |
| `covariance` | `cov(a, b, window)` | Compute covariance |
| `volatility` | `volatility(asset, window)` | Compute annualized volatility |
| `mean_return` | `mean_return(asset, window)` | Compute mean return |

---

### 6. Sandbox Environment

The evaluation environment is built on top of **MarS** (ICLR 2025), a generative market simulation engine that models order flow as token sequences using a Large Market Model. MarS is extended to support:

- Multi-asset joint simulation across all six asset classes
- Stress scenario injection (historical events + synthetic shocks)
- Market state labeling for per-state performance decomposition

Interface conventions follow **FinRL-Meta** (NeurIPS 2022).
