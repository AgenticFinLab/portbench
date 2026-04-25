# PortBench

**PortBench** is a benchmark for evaluating LLMs on multi-asset portfolio management. It addresses three gaps in existing financial benchmarks:

1. **Multi-asset evaluation** — the unit is a portfolio weight allocation across six asset classes (Equities, Bonds, Commodities, Real Estate, Cryptocurrency, Cash), not single-asset buy/sell decisions.
2. **Risk-first paradigm** — models must pass profile-sensitive stress gates on three crisis windows (2015 China shock, 2020 COVID, 2022 crypto collapse) before entering the performance ranking.
3. **Two-tier evaluation** — a static QA dataset measuring knowledge, plus a stateful Sandbox backtest measuring realized PnL + CEPS + profile alignment across three investor profiles.

## Evaluation Architecture

PortBench uses a two-tier structure:

| Tier | Component           | Stateful | Measures                                                            |
|------|---------------------|----------|---------------------------------------------------------------------|
| 1    | **QA Dataset**      | No       | Knowledge & financial reasoning (per-template accuracy, T1–T7)      |
| 2    | **Sandbox Backtest**| **Yes**  | Realized PnL + CEPS + profile alignment, across 3 investor profiles |

Unified entry: `python examples/run_all_eval.py --eval qa sandbox`

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

## Module Overview

**`portbench/data_collect/`** — Four collectors covering all six asset classes: Yahoo Finance (72 tickers), FRED (60 macro/yield series), Kaggle (10 datasets including news + crypto), SEC EDGAR (20 companies, 10-K/10-Q filings).

**`portbench/data_preprocess/`** — Per-asset-class preprocessors with shared time-alignment, daily/monthly forward-fill, winsorization, log returns, rolling z-score, and dynamic train/val/test labeling. Outputs `datasets/processed/<asset_class>.csv`.

**`portbench/data_quality/`** — Numeric / text / cross-asset checkers with PASS/WARN/FAIL ratings; flags coverage gaps in stress windows. `label_market_regimes()` produces bull/bear/sideways/crisis labels used by both QA stratification and stress evaluation.

**`portbench/metrics/`** — Pure-function library: returns, risk (vol, max drawdown, VaR, CVaR), risk-adjusted (Sharpe, Sortino, Calmar, IR), allocation MAE, and the **CEPS** metric with cascade propagation penalty.

**`portbench/qa_builder/`** — Seven question templates (T1–T7) generating QA pairs from a `DataProvider` (`MockDataProvider` for synthetic GBM, `ProcessedDataProvider` for real data). Splits are computed dynamically from the data range. `ContextWindow` carries cross-asset correlations and SEC/Kaggle text. Build pipeline ranks dates text-first to maximize text coverage (~81% on the current dataset).

**`portbench/baselines/`** — `EqualWeightBaseline`, `SixtyFortyBaseline`, `RiskParityBaseline`, `SmartFolioBaseline` — all implement the same `AgentAdapter` interface as LLM agents and pass through the identical evaluation pipeline.

**`portbench/visualization/`** — Matplotlib helpers for generating dataset, regime, ranking, CEPS, stress, and QA-sample figures.

---

## Agent Evaluation (Detailed)

The agent evaluation framework (`portbench/agent_eval/` + `portbench/sandbox/`) is the centerpiece of PortBench. It evaluates an LLM agent across the full investment decision pipeline and quantifies how decision errors compound and translate into realized PnL.

### Five-Stage Pipeline

Every rebalance step routes a `MarketSnapshot` through five sequential stages:

```
MarketSnapshot (price/return data, macro context, news_text, correlation_matrix)
      │
      ▼
 S1: Market Interpretation   ── asset_views ──►   (LLM, sentiment per asset in [-1, +1])
 S2: Signal Generation       ── signals ─────►    (LLM, buy/hold/sell per asset)
 S3: Weight Optimization     ── weights ─────►    (LLM, target weights summing to 1)
 S4: Execution Simulation    ── executed_w ──►    (deterministic, slippage 10bps + commission 5bps)
 S5: Risk Monitoring         ── alerts ──────►    (deterministic, VaR/drawdown/rebalance flag)
      │
      ▼
 EpisodeResult → CEPS score (collected per step inside the Sandbox)
```

| Stage | Type          | Scoring                                                                                   |
|-------|---------------|-------------------------------------------------------------------------------------------|
| S1    | LLM           | `1 − MAE(views, gt) / 2`                                                                  |
| S2    | LLM           | Fraction of assets with correct signal direction                                          |
| S3    | LLM           | **70% weight accuracy + 30% correlation awareness** (variance-reduction vs. ground truth) |
| S4    | Deterministic | not LLM-scored                                                                            |
| S5    | Deterministic | 50% rebalance decision + 50% VaR/drawdown accuracy                                        |

Three stages call the LLM (S1, S2, S3) per rebalance; S4 and S5 are deterministic numerical layers.

### CEPS — Cross-Stage Error Propagation Score

```
isolated_avg          = mean(stage_scores)
cascade_drops         = [max(s[i] − s[i+1], 0) for adjacent stage pairs]
propagation_penalty   = propagation_weight × Σ cascade_drops
CEPS                  = clip(isolated_avg − propagation_penalty, 0, 1)
```

CEPS specifically penalizes the pattern where a strong stage is followed by a weaker stage — the intuition being that errors propagate and amplify downstream. Uniform mediocrity is penalized less than a sharp mid-pipeline drop. CEPS is no longer a standalone evaluation — it is collected as a **per-step byproduct of the Sandbox rebalance loop**, so CEPS and realized PnL always come from the same run.

### Sandbox: Stateful Backtest × 3 Investor Profiles

The Sandbox (`portbench/sandbox/`) is a closed-loop, stateful backtest that maintains NAV, weights, and trade history across time, applies transaction costs at each rebalance, and injects an investor profile description into every LLM prompt.

Each model is evaluated across **three investor profiles**, each with its own pass/fail threshold:

| Profile | `max_drawdown_tolerance` | `max_equity_weight` | `var_limit` |
|---------|--------------------------|---------------------|-------------|
| Conservative | 10% | 40% | 5% |
| Balanced | 20% | 65% | 10% |
| Aggressive | 35% | 90% | 20% |

For each profile, evaluation runs in two phases:

**Phase A — Stress Gate** (3 historical crisis windows):
| Scenario | Period |
|---|---|
| `2015_china_shock` | 2015-08-01 → 2016-02-29 |
| `2020_covid` | 2020-02-01 → 2020-05-31 |
| `2022_crypto` | 2022-05-01 → 2022-12-31 |

Pass condition: `abs(max_drawdown) ≤ profile.max_drawdown_tolerance`. If any scenario fails, the profile is marked `stress_failed` and the normal phase is skipped.

**Phase B — Normal Market Backtest (2024 full year)** — runs only if all 3 stress scenarios pass. Outputs CEPS per step + profile alignment + NAV curve + Sharpe / CAGR / max drawdown / Calmar / etc.

The cross-profile summary computes an `adaptation_score = std(per-profile total_return)` measuring how differently the model behaves across profiles.

### LLM Adapters

**Cloud adapters** (`llm_adapters.py`):

| Adapter | Examples | Env var |
|---|---|---|
| `AnthropicAdapter` | `claude-opus-4-7`, `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `OpenAIAdapter` | `gpt-4o`, `gpt-4-turbo` | `OPENAI_API_KEY` |
| `LiteLLMAdapter` | any litellm string | depends on model |

**Local adapters** (`local_adapter.py`):

| Adapter | Backend | Best for |
|---|---|---|
| `VLLMAdapter` | vLLM OpenAI-compatible server | Batch evaluation |
| `OllamaAdapter` | Ollama HTTP API | Desktop experiments |
| `HuggingFaceAdapter` | Direct transformers inference | No server, any HF model |

All adapters retry up to 3× with exponential backoff and accept configurable `system_prompt` / `temperature`.

### Logging & Reproducibility

Every evaluation run can be fully persisted via `pipeline.enable_logging()`:

```
outputs/evaluation_results/eval_logs/{run_id}/
├── run_meta.json          # model, timestamps, config
├── run_summary.json       # totals (on close)
├── errors.jsonl           # rolling stage-level errors
└── episodes/<date>_<n>.json   # full prompt + raw response + parsed output + ground truth + score per stage
```

This enables replay, prompt-engineering iteration, and post-hoc audit of any individual decision.

### Oracle Mode (Ablation)

Inject ground truth at any stage to isolate the contribution of downstream stages:

```python
result = pipeline.run_episode(snapshot, inject_gt_at=StageID.S3_WEIGHT_OPTIMIZATION)
# Measures impact of S4 + S5 only (S1–S3 are perfect)
```

### Running Evaluation

```bash
# QA static evaluation
python examples/agent_eval/run_qa_eval.py

# Sandbox: all 3 profiles, stress + normal
python examples/sandbox/run_backtest.py --model anthropic:claude-opus-4-7

# Specific profiles
python examples/sandbox/run_backtest.py --profiles conservative balanced

# Baselines (no API)
python examples/sandbox/run_backtest.py --baseline risk_parity

# Local LLM
python examples/sandbox/run_backtest.py --model vllm:meta-llama/Llama-3.1-8B-Instruct
python examples/sandbox/run_backtest.py --model ollama:llama3.1
```

### Sandbox Output Layout

```
outputs/sandbox/{model}/{timestamp}/
  conservative/
    stress_2015_china_shock/backtest_result.json
    stress_2020_covid/backtest_result.json
    stress_2022_crypto/backtest_result.json
    normal/                          # only if all stress scenarios pass
      nav_curve.csv
      weight_history.csv
      backtest_result.json
      summary.txt
  balanced/...
  aggressive/...
  profile_comparison.json            # cross-profile summary + adaptation_score
```

---

## Data Layout

```
datasets/                 # Raw + preprocessed data (gitignored)
├── yahoo/                # Raw OHLCV CSVs
├── fred/                 # Raw macro/yield CSVs
├── kaggle/               # Raw Kaggle datasets
├── sec/                  # SEC filing HTMLs
├── processed/            # Per-asset-class preprocessed CSVs
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
