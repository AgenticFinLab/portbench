# Agent Evaluation Framework (`portbench/agent_eval/`)

## Overview

End-to-end evaluation of LLM agents across a five-stage portfolio management pipeline. The pipeline models the full investment decision process from market interpretation to risk monitoring, and quantifies cross-stage error propagation via the CEPS metric.

## Evaluation Architecture

PortBench uses a two-tier evaluation structure:

```
Tier 1: QA Static Evaluation
  Script:  examples/agent_eval/run_qa_eval.py
  Input:   datasets/qa_dataset/test.jsonl
  Output:  outputs/qa/{model}/{timestamp}/qa_results.json
  Metric:  per-template accuracy (T1–T7), exact-match comparison
  Purpose: Measures knowledge and financial reasoning without market simulation

Tier 2: Sandbox × 3 Investor Profiles
  Script:  examples/sandbox/run_backtest.py
  Output:  outputs/sandbox/{model}/{timestamp}/{profile}/
  For each profile (conservative / balanced / aggressive):
    Phase A — Stress gate (3 historical crisis windows)
      2015_china_shock:   2015-08-01 → 2016-02-29
      2020_covid:         2020-02-01 → 2020-05-31
      2022_crypto:        2022-05-01 → 2022-12-31
      Pass condition:     max_drawdown ≤ profile.max_drawdown_tolerance
                          (profile-sensitive: conservative=10%, balanced=20%, aggressive=35%)
    Phase B — Normal market backtest (2024, only if Phase A passed)
      Outputs per rebalance: CEPS score + profile alignment score + NAV
      Final outputs:         Sharpe, CAGR, max_drawdown (realized PnL)
  Aggregated: profile_comparison.json with adaptation_score = std(per-profile returns)

Unified entry: examples/run_all_eval.py --eval qa sandbox
```

**Key design principles:**
- CEPS is not a standalone evaluation; it is a per-step byproduct of the Sandbox rebalance loop
- Profile constraints are injected into the LLM prompt context at every rebalance step (not post-scored on static outputs)
- Stress thresholds are portfolio drawdown limits tied to each profile's `max_drawdown_tolerance`, not fixed CEPS cutoffs
- All three stress scenarios use real data from `datasets/processed/` (2015-01-02 to 2025-12-31)

## Stress Test Scenarios

| Scenario | Period | Event |
|----------|--------|-------|
| `2015_china_shock` | Aug 2015 – Feb 2016 | China currency devaluation + oil price collapse |
| `2020_covid_flash_crash` | Feb 2020 – May 2020 | COVID-19 pandemic shock, 34% drawdown in 33 days |
| `2022_crypto_collapse` | May 2022 – Dec 2022 | Terra/LUNA + FTX collapse, Fed rate hike cycle |

Pass condition per profile:

| Profile | max_drawdown_tolerance |
|---------|----------------------|
| Conservative | 10% |
| Balanced | 20% |
| Aggressive | 35% |

## Pipeline Architecture

```
MarketSnapshot (includes correlation_matrix)
      │
      ▼
 S1: Market Interpretation   ─── asset_views ──►
 S2: Signal Generation       ─── signals    ──►
 S3: Weight Optimization     ─── weights    ──►  (scored with correlation-awareness)
 S4: Execution Simulation    ─── executed_w ──►
 S5: Risk Monitoring         ─── alerts     ──►
      │
      ▼
 EpisodeResult → CEPS score (collected as Sandbox per-step byproduct)
```


## MarketSnapshot

`MarketSnapshot` is the single external input to the pipeline. It carries numeric market data, a cross-asset correlation matrix, and (when available) recent news/filing text:

```python
@dataclass
class MarketSnapshot:
    decision_date: date
    price_data: dict[str, pd.Series]
    return_data: dict[str, pd.Series]
    macro_data: dict[str, float]
    news_text: str = ""                          # SEC filing / news excerpt, lifted from DataProvider.get_news()
    current_weights: dict[str, float]
    portfolio_value: float
    market_regime: Optional[str]
    correlation_matrix: Optional[pd.DataFrame]   # assets × assets, lazy-computed
    asset_class_map: Optional[dict[str, str]] = None  # ticker -> asset class

    def get_correlation(self) -> pd.DataFrame:
        """Compute (and cache) Pearson correlation matrix from return_data."""

    def get_intra_class_correlation(self) -> dict[str, pd.DataFrame]:
        """Per-asset-class sub-matrices (concentration risk inside each class)."""

    def get_inter_class_correlation(self) -> pd.DataFrame:
        """Asset-class × asset-class matrix from averaged cross-class pairs."""
```

`news_text` is populated by the snapshot builders (`build_snapshots()` in the example, and `ScenarioInjector.generate_snapshots()`) by walking the asset list and calling `provider.get_news(asset, decision_date)` until a non-empty result is found. Empty when no asset in the snapshot has text data (e.g., MockDataProvider, or a snapshot containing only bonds/commodities/cash).

The `correlation_matrix` captures the flat asset × asset structure. When `asset_class_map` is supplied (the `BacktestEngine` forwards the same map it uses for profile alignment), `get_intra_class_correlation()` and `get_inter_class_correlation()` expose the *intra-class* layer (concentration risk inside e.g. equities) and the *inter-class* layer (hedging dynamics across e.g. equities vs bonds) separately. Both layers feed the S1/S3 prompt blocks and the S3 score.

## Pipeline Stages

### S1 — Market Interpretation (`S1MarketInterpretation`)

**Input**: `MarketSnapshot` (price data, return data, macro context, optional `news_text`)
**Output**: `S1Output` — `asset_views: dict[str, float]` in [-1, +1]

Ground truth: `view = clip(trailing_return / 0.10, -1, 1)` — normalizes a ±10% trailing return to ±1.0.

LLM prompt asks the model to analyze price/return data and output a JSON dict of sentiment scores per asset. When `snapshot.news_text` is non-empty, a `RECENT NEWS / FILINGS:` block is appended to the prompt so the model can incorporate qualitative context.

**Scoring**: `1 - MAE(actual_views, gt_views) / 2`

### S2 — Signal Generation (`S2SignalGeneration`)

**Input**: `S1Output` (and `snapshot.news_text` from the carried-through `MarketSnapshot`)
**Output**: `S2Output` — `signals: dict[str, "buy"|"hold"|"sell"]`

Ground truth: `view > 0.2 → buy`, `view < -0.2 → sell`, else `hold`.

LLM prompt receives S1 views and must convert them to directional signals with strength scores. News/filing text is included when present, allowing the model to override pure-numeric views (e.g., bearish view but bullish-rated earnings filing).

**Scoring**: Fraction of assets with correct signal direction.

### S3 — Weight Optimization (`S3WeightOptimization`)

**Input**: `S2Output`  
**Output**: `S3Output` — `weights: dict[str, float]` summing to 1.0

Ground truth: equal-weight among "buy" assets; 0 for "sell" assets; equal-weight all if no buys.

LLM prompt receives signals and must output a weight allocation with sum=1, all values in [0,1].

**Scoring**: Composite score — **70% weight accuracy + 30% correlation awareness**:

- *Weight accuracy*: `1 - weight_MAE / 2`
- *Correlation awareness* (30%): when `snapshot.asset_class_map` is set, the 30% is split into:
  - *15% intra-class concentration penalty* — for each class, `class_weight × max(avg_intra_corr, 0)` is summed and subtracted from 1; piling weight inside a class with high mutual correlation is penalized.
  - *15% inter-class hedging credit* — weighted average of off-diagonal inter-class correlations (`Σ w_i w_j corr(class_i, class_j)`), mapped via `(1 − avg)/2`; spreading weight across weakly / negatively correlated classes is rewarded.
  - When no `asset_class_map` is available, the full 30% falls back to the variance-ratio score `clip(2 − var(actual)/var(gt), 0, 1)`.

### S4 — Execution Simulation (`S4ExecutionSimulation`)

**Deterministic** — no LLM call. Simulates execution costs:
- Slippage: 10 bps linear market impact
- Commission: 5 bps per trade value
- Adjusts executed weights for cost drag

### S5 — Risk Monitoring (`S5RiskMonitoring`)

**Deterministic** — computes portfolio risk metrics numerically:
- 1-day VaR at 95% confidence (historical simulation)
- Maximum drawdown from peak
- Weight drift vs equal-weight target
- Triggers alerts and `rebalance_needed` flag when limits are breached

**Scoring**: 50% correct `rebalance_needed` decision + 50% VaR/drawdown accuracy.

## LLM Adapters

### Cloud Adapters (`llm_adapters.py`)

| Adapter | Model examples | Env var needed |
|---------|---------------|---------------|
| `AnthropicAdapter` | `claude-opus-4-6`, `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `OpenAIAdapter` | `gpt-4o`, `gpt-4-turbo` | `OPENAI_API_KEY` |
| `LiteLLMAdapter` | any litellm string: `anthropic/...`, `ollama/llama3` | depends on model |

### Local Adapters (`local_adapter.py`)

For evaluation without external API calls:

| Adapter | Backend | Best for |
|---------|---------|---------|
| `VLLMAdapter` | vLLM OpenAI-compatible server | Batch evaluation, high throughput |
| `OllamaAdapter` | Ollama HTTP API | Desktop experiments, easy setup |
| `HuggingFaceAdapter` | Direct transformers inference | No server needed, any HF model |

```python
# vLLM (start server first: python -m vllm.entrypoints.openai.api_server --model ...)
from portbench.agent_eval import VLLMAdapter, build_default_pipeline
adapter = VLLMAdapter(model="meta-llama/Llama-3.1-8B-Instruct", base_url="http://localhost:8000/v1")

# Ollama (start server: ollama serve; ollama pull llama3.1)
from portbench.agent_eval import OllamaAdapter
adapter = OllamaAdapter(model="llama3.1")

# HuggingFace (no server; loads model into GPU/CPU memory)
from portbench.agent_eval import HuggingFaceAdapter
adapter = HuggingFaceAdapter(model_name="microsoft/Phi-3-mini-4k-instruct", load_in_4bit=True)

pipeline = build_default_pipeline(adapter)
```

All adapters (cloud and local):
- Retry up to 3 times with exponential backoff on transient errors
- Accept a configurable `system_prompt` and `temperature`

```python
from portbench.agent_eval import AnthropicAdapter, build_default_pipeline

adapter = AnthropicAdapter(model="claude-opus-4-6", temperature=0.0)
pipeline = build_default_pipeline(adapter)
```

## Evaluation Logging

Every evaluation run can be fully persisted to disk for replay and analysis.

```python
pipeline = build_default_pipeline(adapter)
pipeline.enable_logging(
    output_dir="outputs/evaluation_results/eval_logs",
    model_name="claude-opus-4-6",
    config={"dataset": "test_2024", "n_episodes": 50},
)

for snapshot in snapshots:
    result = pipeline.run_episode(snapshot)

log_dir = pipeline.finalize_logging()
print(f"Logs saved to: {log_dir}")
```

### Log Directory Structure

```
outputs/evaluation_results/eval_logs/{run_id}/
├── run_meta.json          # Model name, timestamps, config
├── run_summary.json       # Total episodes, duration (written on close)
├── errors.jsonl           # Rolling log of stage-level errors
└── episodes/
    ├── 2024-01-05_0001.json
    ├── 2024-01-10_0002.json
    └── ...
```

### Episode Log Format

Each episode file contains:
```json
{
  "episode_id": "2024-01-05_0001",
  "decision_date": "2024-01-05",
  "model_name": "anthropic/claude-opus-4-6",
  "ceps_score": 0.74,
  "duration_ms": 3420.5,
  "stages": [
    {
      "stage_id": "S1",
      "score": 0.85,
      "latency_ms": 1200.0,
      "prompt": "You are a portfolio manager analyzing...",
      "raw_response": "{\"asset_views\": {...}}",
      "parsed_output": {"asset_views": {...}, "detected_regime": "bear"},
      "ground_truth": {"asset_views": {...}, "detected_regime": "bear"},
      "error": ""
    },
    ...
  ]
}
```

## Stress Testing

```python
from portbench.agent_eval import ScenarioInjector, STRESS_SCENARIOS

injector = ScenarioInjector(provider=provider, assets=assets, lookback_days=60)

for scenario in STRESS_SCENARIOS:
    result = injector.run_stress_test(scenario, pipeline, step_days=10)
    print(f"{scenario.name}: {'PASSED' if result['passed'] else 'FAILED'} "
          f"(CEPS={result['mean_ceps']:.3f}, threshold={scenario.min_pass_score})")
```

Predefined scenarios:

| Scenario | Period | Min pass score |
|----------|--------|---------------|
| 2008 Global Financial Crisis | 2008-09 – 2009-03 | 0.40 |
| 2020 COVID Flash Crash | 2020-02 – 2020-05 | 0.45 |
| 2022 Crypto Collapse | 2022-05 – 2022-12 | 0.50 |

## Oracle Mode (Ablation)

Inject ground truth at any stage to isolate the contribution of downstream stages:

```python
# Measure impact of S4+S5 errors only (S1–S3 are perfect)
result = pipeline.run_episode(snapshot, inject_gt_at=StageID.S3_WEIGHT_OPTIMIZATION)
```

## Running Evaluation

```bash
# Mock agent + mock data (no API keys, tests pipeline end-to-end)
python examples/agent_eval/run_evaluation.py

# Real cloud LLM (uses MockDataProvider unless --data-provider=processed is set)
ANTHROPIC_API_KEY=... python examples/agent_eval/run_evaluation.py

# Real data from datasets/processed/ — populates news_text into snapshots
python examples/agent_eval/run_evaluation.py --data-provider processed

# Custom data dirs
python examples/agent_eval/run_evaluation.py --data-provider processed \
    --data-dir datasets/processed --sec-dir datasets/sec

# Local model — vLLM
python examples/agent_eval/run_evaluation.py --local-model vllm:meta-llama/Llama-3.1-8B-Instruct

# Local model — Ollama
python examples/agent_eval/run_evaluation.py --local-model ollama:llama3.1

# Local model — HuggingFace
python examples/agent_eval/run_evaluation.py --local-model hf:microsoft/Phi-3-mini-4k-instruct

# Baselines
python examples/agent_eval/run_evaluation.py --baseline risk_parity

# Stress tests only
python examples/agent_eval/run_evaluation.py --stress-only
```

### Data Provider Selection

The `--data-provider` flag chooses between:

| Provider | Source | News text | Use case |
|----------|--------|-----------|----------|
| `mock` (default) | Synthetic GBM, deterministic | Always empty | CI, agent debugging, no setup |
| `processed` | `datasets/processed/*.csv` | SEC + Kaggle (where available) | Real-world benchmark runs |

`processed` requires `examples/data_collect/get_all.py` and `examples/data_preprocess/preprocess_all.py` to have been run. It also requires `datasets/processed/` to span the stress-test windows (2008/2020/2022); if it doesn't, those stress tests will return zero snapshots and fail the gate.

Output: `outputs/evaluation_results/eval_results/{run_id}/`
- `per_stage_scores.json` — per-episode and mean stage scores
- `ceps_scores.json` — CEPS with propagation penalty breakdown
- `stress_test_results.json` — per-scenario pass/fail
- `risk_first_ranking.json` — final ranking entry (stress-gated)
- `summary.txt` — human-readable summary
