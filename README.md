# PortBench: A Correlation-Aware, Full-Pipeline Benchmark for LLM-Driven Portfolio Management

[![Market Base Dataset](https://img.shields.io/badge/🤗_HuggingFace-Market_Base_Dataset-yellow)](https://huggingface.co/datasets/AgenticFinLab/PortBench-Market)
[![QA Dataset](https://img.shields.io/badge/🤗_HuggingFace-QA_Dataset-yellow)](https://huggingface.co/datasets/AgenticFinLab/PortBench-QA)

Existing financial benchmarks are limited because they either focus on single assets or evaluate multi-assets in isolation, thereby ignoring asset correlations. Furthermore, they lack a full-pipeline assessment that mirrors real-world portfolio management workflows. To address these gaps, our contributions include:

1. **Multi-asset Market Base Dataset**: we collect and release a ten-year (Jan 2015–Dec 2025) dataset covering 183 financial instruments across six heterogeneous asset classes (Equities, Bonds, Commodities, Real Estate, Cryptocurrency, Cash), with associated news text, macroeconomic indicators, and cross-asset correlation structures. Both evaluation layers are built on top of this dataset. [[PortBench-Market](https://huggingface.co/datasets/AgenticFinLab/PortBench-Market)]
2. **Dual-layer evaluation**: a static QA layer (6,269 pairs across 7 templates T1–T7) probes correlation-based financial reasoning, paired with a dynamic five-stage sandbox pipeline (market interpretation -> signal generation -> weight optimization -> execution -> risk monitoring) that evaluates the full sequential decision cycle under realistic market replay. [[PortBench-QA](https://huggingface.co/datasets/AgenticFinLab/PortBench-QA)]
3. **Novel evaluation metrics**: a two-layer correlation scoring criterion that penalizes intra-class concentration and rewards inter-class hedging in portfolio weights, together with CEPS (Cross-stage Error Propagation Score) that quantifies how errors compound across pipeline stages.
4. **Stress regime and investor profile evaluation**: models are evaluated under three historical stress regimes (2015 China Shock, 2020 COVID Crash, 2022 Crypto Collapse) and three investor profiles (conservative, balanced, aggressive), testing both robustness under correlation shocks and alignment with investor-specific risk constraints.

<p align="center">
  <img src="figures/intro_overview.png" width="100%" alt="PortBench Overview"/>
</p>
<p align="center"><b>Figure 1.</b> Overview of PortBench: Market Base Dataset across six asset classes (Jan 2015–Dec 2025), dual evaluation layer (static QA + dynamic pipeline), robustness evaluation under three stress regimes, and investor task profiles with distinct risk constraints.</p>

## Evaluation Architecture

PortBench evaluates LLMs through two complementary layers:

| Layer | Component | Stateful | Measures |
|-------|-----------|----------|----------|
| Static | **QA Dataset** (6,269 pairs, 7 templates T1–T7) | No | Correlation-based financial reasoning across four difficulty levels |
| Dynamic | **Five-Stage Sandbox Pipeline** | Yes | Realized PnL + CEPS + profile alignment, under 3 profiles × 3 stress regimes |

The **QA layer** probes isolated financial reasoning — from single-asset return prediction (T1) to regime-driven portfolio rebalancing (T7). The **dynamic layer** replays market data point-in-time and measures how well models sustain decision quality across the full sequential pipeline.

<p align="center">
  <img src="figures/method_framework.png" width="100%" alt="Evaluation Framework"/>
</p>
<p align="center"><b>Figure 2.</b> Evaluation framework. <b>Top:</b> Static QA evaluation with representative QA pairs from seven task templates (T1–T7). <b>Bottom:</b> Dynamic five-stage pipeline evaluation under three investor profiles and three historical stress regimes.</p>

For additional visualizations, including Market Base Dataset time-series slices, QA template examples, pipeline evaluation input snapshots, and per-model result breakdowns, please refer to the paper appendix.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

API keys in `.env`:
```
KAGGLE_USERNAME=...
KAGGLE_KEY=...
FRED_API_KEY=...
SEC_API_KEY=...
ANTHROPIC_API_KEY=...   # for LLM evaluation

# Batch-experiment providers (each provider needs all three)
DASHSCOPE_API_KEY=...   ; DASHSCOPE_BASE_URL=...   ; DASHSCOPE_MODEL=qwen-plus
TENCENT_API_KEY=...     ; TENCENT_BASE_URL=...     ; TENCENT_MODEL=hunyuan-pro
ARK_API_KEY=...         ; ARK_BASE_URL=...         ; ARK_MODEL=doubao-pro-32k
# Add new providers by adding {PREFIX}_API_KEY/_BASE_URL/_MODEL here
# AND one line to PROVIDER_REGISTRY in portbench/experiments/providers.py.
```

## Quick Start

```bash
# 1. Collect & preprocess data (requires API keys)
python examples/data_collect/get_all.py
python examples/data_preprocess/preprocess_all.py

# 2. Build QA dataset
python examples/qa_builder/build_qa_dataset.py

# 3. Static QA evaluation
python examples/agent_eval/run_qa_eval.py

# 4. Sandbox backtest (3 profiles × stress + normal)
python examples/sandbox/run_backtest.py --model anthropic:claude-opus-4-7

# Mock-only mode (no API keys, no data)
python examples/sandbox/run_backtest.py --data-provider mock

# 5. Batch sweep across providers / profiles / stress scenarios
python -m portbench.experiments --config configs/experiments/default.yaml --dry-run
python -m portbench.experiments --config configs/experiments/default.yaml
```

## Module Documentation

Detailed module docs live in [`docs/modules/`](docs/modules/):

| Module | Path | Docs |
|--------|------|------|
| Data Collection | `portbench/data_collect/` | [data_collect.md](docs/modules/data_collect.md) |
| Data Preprocessing | `portbench/data_preprocess/` | [data_preprocess.md](docs/modules/data_preprocess.md) |
| Data Quality | `portbench/data_quality/` | [data_quality.md](docs/modules/data_quality.md) |
| Metrics Library | `portbench/metrics/` | [metrics.md](docs/modules/metrics.md) |
| QA Builder | `portbench/qa_builder/` | [qa_builder.md](docs/modules/qa_builder.md) |
| Agent Evaluation | `portbench/agent_eval/` | [agent_eval.md](docs/modules/agent_eval.md) |
| Sandbox Backtest | `portbench/sandbox/` | [sandbox.md](docs/modules/sandbox.md) |
| Baselines | `portbench/baselines/` | [baselines.md](docs/modules/baselines.md) |
| Batch Experiments | `portbench/experiments/` | [experiments.md](docs/modules/experiments.md) |
| Correlation | `portbench/` (cross-cutting) | [correlation.md](docs/modules/correlation.md) |

## Module Overview

**`portbench/data_collect/`** — Four collectors covering all six asset classes: Yahoo Finance (83 tickers), FRED (60 macro/yield series), Kaggle (10 datasets including news + crypto), SEC EDGAR (23 companies, 10-K/10-Q filings). After preprocessing, the final dataset contains **183 distinct financial instruments** spanning six asset classes (equities 126, commodities 16, bonds 15, cryptocurrency 12, real estate 10, cash 4).

**`portbench/data_preprocess/`** — Per-asset-class preprocessors with shared time-alignment, daily/monthly forward-fill, winsorization, log returns, rolling z-score, and dynamic train/val/test labeling. Outputs `datasets/processed/<asset_class>.csv`.

**`portbench/data_quality/`** — Numeric / text / cross-asset checkers with PASS/WARN/FAIL ratings; flags coverage gaps in stress windows. `label_market_regimes()` produces bull/bear/sideways/crisis labels used by both QA stratification and stress evaluation.

**`portbench/metrics/`** — Pure-function library: returns, risk (vol, max drawdown, VaR, CVaR), risk-adjusted (Sharpe, Sortino, Calmar, IR), allocation MAE, and the **CEPS** metric with cascade propagation penalty.

**`portbench/qa_builder/`** — Seven question templates (T1–T7) generating QA pairs from a `DataProvider` (`MockDataProvider` for synthetic GBM, `ProcessedDataProvider` for real data). Splits are computed dynamically from the data range. `ContextWindow` carries cross-asset correlations and SEC/Kaggle text.

**`portbench/baselines/`** — `EqualWeightBaseline`, `SixtyFortyBaseline`, `RiskParityBaseline` (naive inverse-vol), `CovarianceRiskParityBaseline` (Equal-Risk-Contribution using full covariance), `MinVarianceBaseline` (long-only minimum variance) — all implement the same `AgentAdapter` interface as LLM agents and pass through the identical evaluation pipeline.

**`portbench/visualization/`** — Matplotlib helpers for generating dataset, regime, ranking, CEPS, stress, and QA-sample figures.

**`portbench/experiments/`** — Batch experiment framework: YAML-driven sweeps over `(provider × model × profile × stress scenario)`, provider registry that reads `{PREFIX}_API_KEY/_BASE_URL/_MODEL` from `.env` (one line to add a new provider), per-`(model, profile)` failure isolation. Built-in providers: `dashscope`, `tencent`, `deepseek`, `glm`, `kimi`, `minimax`, `ark`, `openai`, `google`, `anthropic`. Built-in baselines: `equal_weight`, `sixty_forty`, `risk_parity`, `cov_risk_parity`, `min_variance`.

---

## Data Layout

```
datasets/                 # Raw + preprocessed data (gitignored)
├── yahoo/                # Raw OHLCV CSVs
├── fred/                 # Raw macro/yield CSVs
├── kaggle/               # Raw Kaggle datasets
├── sec/                  # SEC filing HTMLs
├── processed/            # Per-asset-class preprocessed CSVs +
│                         #   correlation_matrix.csv, covariance_matrix.csv,
│                         #   asset_class_map.json (cross-asset artifacts)
└── qa_dataset/           # Built QA dataset (all_pairs.jsonl, train/val/test, stats.json)

outputs/                  # All generated artifacts (gitignored)
├── qa/{model}/...        # QA static eval results
├── sandbox/{model}/...   # Sandbox backtest results
├── evaluation_results/   # Per-run pipeline logs
└── quality_reports/      # Data quality assessment reports
```

## Key Design Constraints

- **Point-in-Time (PiT) safety**: enforced by `ContextWindow.validate_pit()`; no feature may use information from on-or-after the decision date.
- **Market state partitioning**: test data is labeled bull/bear/sideways/crisis to enable per-state performance decomposition.
- **Profile-sensitive stress thresholds**: stress gates are tied to each investor profile's drawdown tolerance, not fixed CEPS cutoffs.
- **Same pipeline for everyone**: baselines and LLM agents flow through identical S1–S5 stages, ensuring a controlled comparison.
