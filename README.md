# PortBench

**PortBench** is a benchmark for evaluating LLMs on multi-asset portfolio management. It addresses three gaps in existing financial benchmarks:

1. **Multi-asset evaluation** — the unit is a portfolio weight allocation across six asset classes (Equities, Bonds, Commodities, Real Estate, Cryptocurrency, Cash), not single-asset buy/sell decisions.
2. **Risk-first paradigm** — models must pass stress tests (2008 crisis, 2020 COVID crash, 2022 crypto collapse) before entering performance rankings.
3. **End-to-end pipeline evaluation** — five sequential stages from market interpretation to risk monitoring, quantified by the CEPS (Cross-Stage Error Propagation Score) metric.

## Evaluation Components

| Component | Description |
|-----------|-------------|
| **QA Dataset** | 7 question templates (T1–T7) × 4 complexity levels, generated from historical market data |
| **Agent Evaluation** | 5-stage pipeline (S1→S5), CEPS score, stress gate, risk-first ranking |

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
ANTHROPIC_API_KEY=...   # for real LLM evaluation
```

## Quick Start

```bash
# 1. Collect data (requires API keys)
python examples/data_collect/get_all.py
python examples/data_preprocess/preprocess_all.py

# 2. Build QA dataset (mock data, no keys needed)
python examples/qa_builder/build_qa_dataset.py

# 3. Run evaluation (mock agent, no keys needed)
python examples/agent_eval/run_evaluation.py

# 4. Run with real LLM
ANTHROPIC_API_KEY=... python examples/agent_eval/run_evaluation.py
```

## Module Documentation

Detailed documentation for each module is in [`docs/modules/`](docs/modules/):

| Module | Path | Docs |
|--------|------|------|
| Data Collection | `portbench/data_collect/` | [data_collect.md](docs/modules/data_collect.md) |
| Data Preprocessing | `portbench/data_preprocess/` | [data_preprocess.md](docs/modules/data_preprocess.md) |
| Data Quality | `portbench/data_quality/` | [data_quality.md](docs/modules/data_quality.md) |
| Metrics Library | `portbench/metrics/` | [metrics.md](docs/modules/metrics.md) |
| QA Builder | `portbench/qa_builder/` | [qa_builder.md](docs/modules/qa_builder.md) |
| Agent Evaluation | `portbench/agent_eval/` | [agent_eval.md](docs/modules/agent_eval.md) |
| Baselines | `portbench/baselines/` | [baselines.md](docs/modules/baselines.md) |

## Module Summary

**`portbench/data_collect/`** — Four collectors (Yahoo: 72 tickers, FRED: 60 series, Kaggle: 10 datasets, SEC: 20 companies). Full coverage of all six asset classes including sector ETFs, TIPS, Case-Shiller housing, and commodity spot prices.

**`portbench/data_preprocess/`** — Per-asset-class preprocessors (time alignment, forward-fill ≤3 days, winsorization, log returns). Outputs `datasets/processed/<asset_class>.csv`.

**`portbench/data_quality/`** — Three checkers (numeric, text, cross-asset). Flags coverage gaps in stress periods. Includes `label_market_regimes()` for bull/bear/sideways/crisis labeling.

**`portbench/metrics/`** — Shared metrics: returns, risk, risk-adjusted, allocation MAE. CEPS metric with cascade propagation penalty.

**`portbench/qa_builder/`** — T1–T7 templates with `MockDataProvider` (GBM synthesis) and `ProcessedDataProvider` (real data). Strict PiT enforcement via `ContextWindow.validate_pit()`. Split boundaries computed dynamically from the data timeline via `QAConfig.from_date_range()`. `ContextWindow` includes a `correlation_matrix` for cross-asset correlation-aware evaluation.

**`portbench/agent_eval/`** — Five pipeline stages (S1–S5). Cloud adapters: `AnthropicAdapter`, `OpenAIAdapter`, `LiteLLMAdapter`. Local adapters: `VLLMAdapter`, `OllamaAdapter`, `HuggingFaceAdapter`. S3 scoring includes a correlation-awareness component (70% weight accuracy + 30% portfolio variance reduction vs. ground truth). `EvalLogger` persists every prompt, response, score, and latency to `outputs/eval_logs/{run_id}/`.

**`portbench/baselines/`** — `EqualWeightBaseline`, `SixtyFortyBaseline`, `RiskParityBaseline`, `SmartFolioBaseline` — all implement `AgentAdapter` for direct pipeline integration.

## Data Layout

```
datasets/            # Raw and preprocessed data (inputs to the pipeline)
├── yahoo/           # Raw Yahoo Finance OHLCV CSVs
├── fred/            # Raw FRED time-series CSVs
├── kaggle/          # Raw Kaggle datasets
├── sec/             # SEC filing HTML files
└── processed/       # Preprocessed per-asset-class CSVs

outputs/             # All generated artifacts (gitignored)
├── qa_dataset/      # Built QA dataset (all_pairs.jsonl, train/val/test splits, stats.json)
├── eval_results/    # Agent evaluation outputs by model name
├── eval_logs/       # Full interaction logs (prompts, responses, scores, latencies)
└── quality_reports/ # Data quality assessment reports
```

## Tests

```bash
pytest
```
