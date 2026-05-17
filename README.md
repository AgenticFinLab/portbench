# PortBench

**PortBench** is a benchmark for evaluating LLMs on multi-asset portfolio management. It addresses three gaps in existing financial benchmarks:

1. **Multi-asset evaluation** — the unit is a portfolio weight allocation across six asset classes (Equities, Bonds, Commodities, Real Estate, Cryptocurrency, Cash), not single-asset buy/sell decisions.
2. **Risk-first paradigm** — models are evaluated on three crisis windows (2015 China shock, 2020 COVID, 2022 crypto collapse) with profile-sensitive drawdown thresholds, enabling direct comparison of risk management and return generation across all models.
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

## Module Overview

**`portbench/data_collect/`** — Four collectors covering all six asset classes: Yahoo Finance (72 tickers), FRED (60 macro/yield series), Kaggle (10 datasets including news + crypto), SEC EDGAR (20 companies, 10-K/10-Q filings).

**`portbench/data_preprocess/`** — Per-asset-class preprocessors with shared time-alignment, daily/monthly forward-fill, winsorization, log returns, rolling z-score, and dynamic train/val/test labeling. Outputs `datasets/processed/<asset_class>.csv`.

**`portbench/data_quality/`** — Numeric / text / cross-asset checkers with PASS/WARN/FAIL ratings; flags coverage gaps in stress windows. `label_market_regimes()` produces bull/bear/sideways/crisis labels used by both QA stratification and stress evaluation.

**`portbench/metrics/`** — Pure-function library: returns, risk (vol, max drawdown, VaR, CVaR), risk-adjusted (Sharpe, Sortino, Calmar, IR), allocation MAE, and the **CEPS** metric with cascade propagation penalty.

**`portbench/qa_builder/`** — Seven question templates (T1–T7) generating QA pairs from a `DataProvider` (`MockDataProvider` for synthetic GBM, `ProcessedDataProvider` for real data). Splits are computed dynamically from the data range. `ContextWindow` carries cross-asset correlations and SEC/Kaggle text. Build pipeline ranks dates text-first to maximize text coverage (~81% on the current dataset).

**`portbench/baselines/`** — `EqualWeightBaseline`, `SixtyFortyBaseline`, `RiskParityBaseline` (naive inverse-vol), `CovarianceRiskParityBaseline` (Equal-Risk-Contribution using full covariance), `MinVarianceBaseline` (long-only minimum variance) — all implement the same `AgentAdapter` interface as LLM agents and pass through the identical evaluation pipeline.

**`portbench/visualization/`** — Matplotlib helpers for generating dataset, regime, ranking, CEPS, stress, and QA-sample figures.

**`portbench/experiments/`** — Batch experiment framework: YAML-driven sweeps over `(provider × model × profile × stress scenario)`, provider registry that reads `{PREFIX}_API_KEY/_BASE_URL/_MODEL` from `.env` (one line to add a new provider), per-`(model, profile)` failure isolation, full intermediate-artifact capture (per-stage prompt/response, per-rebalance snapshots, NAV/weight/trade CSVs, figures). See [docs/modules/experiments.md](docs/modules/experiments.md).

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

| Stage | Type          | Ground Truth | Scoring                                                                                   |
|-------|---------------|--------------|-------------------------------------------------------------------------------------------|
| S1    | LLM           | `clip(trailing_return / 0.10, -1, 1)` per asset | `1 − MAE(views, gt) / 2`                          |
| S2    | LLM           | `view > 0.2 → buy`, `< -0.2 → sell`, else hold | Fraction of assets with correct signal direction |
| S3    | LLM           | **max-Sharpe weights** over buy-signal assets (computed from realized future returns; never shown to LLM) | **σ × weight accuracy + (1−σ) × correlation awareness** (default σ=0.5). Correlation awareness = 50% intra-class concentration penalty + 50% inter-class hedging credit when `asset_class_map` is available; variance-ratio fallback otherwise. σ is configurable: σ=1 → pure return optimality, σ=0 → pure risk diversification. |
| S4    | Deterministic | Execute S3 GT weights at zero slippage | Weight MAE vs GT executed weights                |
| S5    | Deterministic | Risk metrics from S4 GT weights | 50% rebalance decision + 50% VaR/drawdown accuracy |

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

Pass condition: `abs(max_drawdown) ≤ profile.max_drawdown_tolerance`.

**Phase B — Normal Market Backtest (2024 full year)** — always runs regardless of Phase A result. Outputs CEPS per step + profile alignment + NAV curve + Sharpe / CAGR / max drawdown / Calmar / etc.

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
    normal/                          # always written when run_normal=true
      nav_curve.csv
      weight_history.csv
      backtest_result.json
      summary.txt
  balanced/...
  aggressive/...
  profile_comparison.json            # cross-profile summary + adaptation_score
```

---

## Batch Experiments

For large-scale sweeps across multiple providers, profiles, and stress scenarios, use the `portbench/experiments/` framework. It is a thin orchestrator over the same `BacktestEngine` used by `examples/sandbox/run_backtest.py`, with three additions:

1. **YAML-driven sweeps** — declare every `(provider, model, profile, scenario)` combination once.
2. **Provider registry from `.env`** — each provider's `{PREFIX}_API_KEY/_BASE_URL/_MODEL` is read from environment. Adding a new provider = one line in `PROVIDER_REGISTRY` plus three env vars.
3. **Per-model failure isolation** — one model crashing does not abort the batch; failures are recorded in `errors.jsonl` with full tracebacks.

Built-in providers: `dashscope`, `tencent`, `deepseek`, `glm`, `kimi`, `minimax`, `ark`, `openai`, `google`, `anthropic`. Built-in baselines: `equal_weight`, `sixty_forty`, `risk_parity`, `cov_risk_parity`. Plus `mock: true` for harness smoke testing.

```bash
python -m portbench.experiments --config configs/experiments/default.yaml --dry-run
python -m portbench.experiments --config configs/experiments/default.yaml
python -m portbench.experiments --rescore --rebalance monthly --config configs/experiments/default.yaml  # recompute CEPS + regenerate all figures + report
```

Output layout — results are keyed by `(rebalance, provider, model, timestamp)` and reusable across batches:

```
EXPERIMENTS/
├── _dataset_figures/                   # shared dataset-level correlation figures
└── {rebalance}/                        # monthly | weekly | quarterly
    ├── comparison_figures/             # cross-model comparison figures (auto-generated)
    ├── {provider}/                     # ark | tencent | baseline | ...
    │   └── {model}/                    # doubao-seed-2-0-pro-260215 | equal_weight | ...
    │       └── {timestamp}/            # one complete run (all profiles)
    │           ├── run_summary.json    # aggregated results for this run
    │           ├── checkpoint.json     # completed profiles
    │           └── {profile}/
    │               ├── experiment.log / figures/
    │               ├── stress_{scenario}/
    │               │   └── backtest_result.json + nav_curve.csv + snapshots/ + ...
    │               └── normal/         # always written when run_normal=true
    └── analysis_report.md             # generated by --rescore
```

Set `reuse_latest: true` in the YAML to skip models that already have results in the directory (picks the most complete, then latest run per model). See [docs/modules/experiments.md](docs/modules/experiments.md) for the full YAML schema and registry details.

---

## Cross-Asset Correlation Modeling

Multi-asset portfolios face two distinct correlation layers and PortBench surfaces both:

- **Intra-class** (e.g. several tickers inside `equities`) — high mutual correlation means concentration risk is hidden by naive position counts.
- **Inter-class** (e.g. `equities` vs `bonds` vs `commodities`) — low / negative cross-class correlation is what drives real diversification.

Where this shows up in code:

- `MarketSnapshot` carries `correlation_matrix` plus an optional `asset_class_map` and exposes `get_intra_class_correlation()` (per-class sub-matrices) and `get_inter_class_correlation()` (asset-class × asset-class matrix from averaged cross-class pairwise correlations).
- The S1/S3 prompt builder (`portbench/agent_eval/stages.py::_format_correlation`) emits three blocks when the map is available: the pairwise table, an intra-class average-correlation summary (concentration risk per class), and the inter-class matrix (hedging structure).
- The S3 score uses **σ × weight_accuracy + (1−σ) × correlation_awareness** (default σ=0.5). Correlation awareness = 50% intra-class concentration penalty + 50% inter-class hedging credit when `asset_class_map` is present (variance-ratio fallback otherwise). σ controls the return-vs-risk evaluation balance.
- `CrossAssetQualityChecker` adds a `cross_class_correlation_structure` check: builds one daily-return series per class and reports the off-diagonal NaN ratio + min/mean/max so a degenerate cross-class matrix is flagged early.
- `examples/data_preprocess/preprocess_all.py` writes `datasets/processed/correlation_matrix.csv`, `covariance_matrix.csv` (annualized), and `asset_class_map.json` after the per-asset CSVs are built — frozen, PiT-correct artifacts that downstream consumers can read instead of recomputing.
- A new covariance-aware baseline `CovarianceRiskParityBaseline` (`portbench/baselines/covariance_risk_parity.py`) solves Spinu's Equal-Risk-Contribution objective via cyclical coordinate descent, providing a strong baseline that actually uses the covariance matrix (the original `RiskParityBaseline` is naive inverse-volatility). CLI: `--baseline cov_risk_parity`. Experiments registry key: `cov_risk_parity`.

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

- **Point-in-Time (PiT) safety**: enforced by `ContextWindow.validate_pit()`; no feature may use information from on-or-after the decision date. `MarketSnapshot.future_return_data` is the sole intentional exception — it carries realized forward returns used only to compute the S3 ground truth (max-Sharpe weights) and is never included in any LLM prompt.
- **Market state partitioning**: test data is labeled bull/bear/sideways/crisis to enable per-state performance decomposition.
- **Profile-sensitive stress thresholds**: stress gates are tied to each investor profile's drawdown tolerance, not fixed CEPS cutoffs.
- **Same pipeline for everyone**: baselines and LLM agents flow through identical S1–S5 stages, ensuring a controlled comparison.
