# Batch Experiments (`portbench/experiments/`)

## Overview

`portbench/experiments/` is the unified runner for **large-scale sweeps** across `(provider × model × profile × stress scenario)`. It wraps the Sandbox `BacktestEngine` with three additions that the original `examples/sandbox/run_backtest.py` lacked:

1. **YAML-driven sweeps** — declare all model/profile/scenario combinations in one config file; no per-script CLI surgery.
2. **Provider registry from `.env`** — every provider's `API_KEY` / `BASE_URL` / `MODEL` is read from environment variables. Adding a new provider requires one line in `PROVIDER_REGISTRY` plus three `.env` vars; no adapter code changes.
3. **Per-model failure isolation** — one model crashing does not abort the batch. All errors land in `errors.jsonl` with full tracebacks.

Every intermediate artifact is persisted: stage-level prompt/response/parsed-output, per-rebalance market snapshots, full `BacktestResult` (NAV/weights/trades), and rendered figures.

One **run** = one model across all its profiles, stored in a single timestamp directory. Results are keyed by `(rebalance, provider, model, timestamp)` and reusable across different batch invocations.

---

## Module Layout

```text
portbench/experiments/
  __init__.py        — public exports (BatchRunner, ExperimentConfig, build_adapter, ...)
  providers.py       — PROVIDER_REGISTRY + build_adapter() / build_baseline() / build_mock()
                       + spec_provider_name() / spec_model_name() (directory-name helpers)
  config.py          — ExperimentConfig + ModelSpec + LoggingConfig + YAML loader
  paths.py           — four-level directory helpers + find_best_run() + save_backtest_result()
  runner.py          — BatchRunner: sweep + reuse_latest + failure isolation + summary
  figures.py         — per-experiment and cross-model plot rendering
  analysis.py        — analyze_runs(): post-run figures + analysis_report.md
  __main__.py        — CLI entry: python -m portbench.experiments --config X.yaml
```

Supporting files:

- `configs/experiments/default.yaml` — production sweep template
- `configs/experiments/smoke.yaml` — mock-data smoke test (no API keys required)
- `examples/experiments/run_batch.py` — pure-Python entry equivalent to the CLI

---

## Provider Registry

`portbench/experiments/providers.py` defines `PROVIDER_REGISTRY` as the single source of truth for every provider PortBench can call:

```python
PROVIDER_REGISTRY = {
    "dashscope": ProviderSpec("DASHSCOPE", "openai_compat"),
    "tencent":   ProviderSpec("TENCENT",   "openai_compat"),
    "deepseek":  ProviderSpec("DEEPSEEK",  "openai_compat"),
    "glm":       ProviderSpec("GLM",       "openai_compat"),
    "kimi":      ProviderSpec("KIMI",      "openai_compat"),
    "minimax":   ProviderSpec("MINIMAX",   "openai_compat"),
    "ark":       ProviderSpec("ARK",       "openai_compat"),
    "openai":    ProviderSpec("OPENAI",    "openai_compat"),
    "google":    ProviderSpec("GOOGLE",    "openai_compat"),
    "anthropic": ProviderSpec("ANTHROPIC", "anthropic"),
}
```

Built-in baseline keys (used in YAML as `- baseline: <key>`):
`equal_weight`, `sixty_forty`, `risk_parity` (naive inverse-vol),
`cov_risk_parity` (Equal-Risk-Contribution using full covariance).

For each provider key, `build_adapter()` reads three env vars:

| Variable | Required | Purpose |
| -------- | -------- | ------- |
| `{PREFIX}_API_KEY` | Yes | Auth credential. Missing → `RuntimeError`, **no silent fallback**. |
| `{PREFIX}_BASE_URL` | Optional (anthropic ignores) | OpenAI-compatible endpoint URL. |
| `{PREFIX}_MODEL` | Optional | Default model when YAML omits `model:`. |

Example `.env` snippet:

```bash
DASHSCOPE_API_KEY=sk-...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen-plus

TENCENT_API_KEY=...
TENCENT_BASE_URL=https://tokenhub.tencentmaas.com/v1
TENCENT_MODEL=hunyuan-pro

ARK_API_KEY=...
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=doubao-pro-32k
```

### Adding a New Provider

```python
# providers.py
PROVIDER_REGISTRY["myprov"] = ProviderSpec("MYPROV", "openai_compat")
```

```bash
# .env
MYPROV_API_KEY=...
MYPROV_BASE_URL=https://...
MYPROV_MODEL=my-flagship
```

That is the entire change. Adapter code stays untouched.

---

## YAML Schema

```yaml
# Optional human-readable label stored in run metadata; not used for directory naming.
# Supports {models}, {rebalance}, {date}, {time} placeholders.
batch_id: "{models}_{rebalance}_{date}"

data_provider: processed               # processed | mock
data_dir: datasets/processed
sec_dir: datasets/sec
rebalance: monthly                     # weekly | monthly | quarterly  → Level-1 directory
initial_nav: 1000000
workers_per_experiment: 3              # parallel stress scenarios per profile
parallel_experiments: 1               # concurrent model runs
seed: 42
noise: 0.2                             # MockAgentAdapter only

models:
  - provider: dashscope                # uses DASHSCOPE_MODEL from .env
  - provider: tencent
    model: hunyuan-pro                 # explicit model override
  - provider: ark
  - baseline: equal_weight             # built-in baseline (no API)
  - mock: true                         # MockAgentAdapter, for harness tests

profiles: [conservative, balanced, aggressive]
stress_scenarios: all                  # "all" | [name1, name2]
run_normal: true
normal_period:
  start: 2024-01-01
  end: 2024-12-31

logging:
  save_pipeline_logs: true             # per-stage prompt + raw_response + parsed_output
  save_snapshots: true                 # per-rebalance MarketSnapshot dump
  save_figures: true                   # NAV / metrics / stress_drawdown PNGs

on_error: isolate                      # isolate | fail_fast
use_tools: false                       # true = S1/S2/S3 stages call complete_with_tools()
output_root: EXPERIMENTS
propagation_weight: 0.1                # CEPS cascade penalty weight (default 0.1)
reuse_latest: false                    # true = skip models with existing results in the directory hierarchy
```

### Notes

- `batch_id` is a metadata label only — it is stored in run files but **not** used for directory naming.
- A `ModelSpec` element must have exactly one of `provider`, `baseline`, `mock`. Mixing them raises in `ModelSpec.kind()`.
- `stress_scenarios: all` resolves at runtime via `STRESS_SCENARIOS` (currently `2015_china_shock`, `2020_covid_flash_crash`, `2022_crypto_collapse`).
- When `data_provider: processed` is requested but `datasets/processed/equities.csv` is missing, the runner refuses to silently fall back to mock — set `data_provider: mock` explicitly if that is what you want.
- `use_tools: true` enables native tool-calling for S1/S2/S3 stages. Has no effect on baseline or mock models.
- `propagation_weight` controls the CEPS cascade penalty: `ceps = mean_stage_scores - weight × Σmax(score[i]-score[i+1], 0)`.
- `reuse_latest: true` checks `EXPERIMENTS/{rebalance}/{provider}/{model}/` for existing timestamp directories. Picks the most complete (most profiles finished), then the latest. A warning is printed if the reused run is partial.

---

## Output Layout

Results are stored in a **four-level hierarchy** that makes them reusable across different batch invocations:

```text
EXPERIMENTS/
├── _dataset_figures/                     # dataset-level correlation figures (shared)
└── {rebalance}/                          # monthly | weekly | quarterly
    ├── _last_run_config.yaml             # config snapshot of the most recent run
    ├── _env_meta.json                    # git hash, python version
    ├── comparison_figures/               # cross-model comparison figures (auto-generated)
    │   ├── nav_comparison_{profile}.png
    │   ├── metrics_comparison_{profile}.png
    │   └── stress_drawdown_{provider}_{model}.png
    ├── analysis_figures/                 # generated by --analyze
    │   ├── rankings.png
    │   ├── stress_gate.png
    │   └── ceps_breakdown.png
    ├── analysis_report.md                # generated by --analyze
    └── {provider}/                       # ark | tencent | baseline | mock
        └── {model}/                      # doubao-seed-2-0-pro-260215 | equal_weight
            └── {timestamp}/             # 20260510_042407  (one complete run)
                ├── run_summary.json     # aggregated results across all profiles
                ├── run_config.yaml      # config snapshot for this run
                ├── checkpoint.json      # list of completed profile names
                ├── runner.log           # model-level runner log
                ├── errors.jsonl         # per-profile errors with full tracebacks
                └── {profile}/           # conservative | balanced | aggressive
                    ├── experiment.log   # per-profile logging.Logger output
                    ├── error.json       # only if this profile failed
                    ├── figures/
                    │   ├── nav.png
                    │   ├── metrics.png
                    │   ├── stress_drawdown.png
                    │   └── correlation_evolution_{phase}.png
                    ├── stress_{scenario}/
                    │   ├── backtest_result.json   # full BacktestResult.to_dict()
                    │   ├── summary.txt
                    │   ├── nav_curve.csv
                    │   ├── weight_history.csv
                    │   ├── trade_history.json
                    │   ├── snapshots/{YYYY-MM-DD}.json
                    │   └── pipeline_logs/{run_id}/episodes/{date}_{n}.json
                    └── normal/           # only when stress gate passes AND run_normal=true
                        └── (same structure as stress_*)
```

### run_summary.json structure

```json
{
  "provider": "ark",
  "model_name": "doubao-seed-2-0-pro-260215",
  "rebalance": "monthly",
  "run_id": "20260510_042407",
  "n_completed": 3,
  "elapsed_seconds": 1234.5,
  "profiles": {
    "aggressive": {
      "stress_gate_passed": true,
      "stress_results": [
        {"scenario": "2015_china_shock", "passed": true, "max_drawdown": -0.065, ...}
      ],
      "normal": {"total_return": 0.1418, "sharpe_ratio": 0.575, "mean_ceps": 0.693, ...}
    }
  }
}
```

---

## Failure Isolation

Granularity is **per model** (all profiles for one model share a timestamp directory). Inside one model run, profiles execute sequentially; if a profile raises, it is logged and the runner continues to the next profile. A model-level failure (e.g. missing API key during adapter construction) is recorded and the model is skipped entirely.

Two kinds of failures are recorded:

| Trigger | Where recorded |
| ------- | -------------- |
| Adapter build fails (e.g. missing API key) | `{run_dir}/errors.jsonl` at model level |
| Profile execution fails (e.g. S1 JSON parse failure, network) | `{profile_dir}/error.json` + appended to `{run_dir}/errors.jsonl` |

Set `on_error: fail_fast` in YAML to abort the batch on the first model failure.

---

## Intermediate Artifacts (What Gets Saved)

| Artifact | When | Path |
| -------- | ---- | ---- |
| Per-stage prompt + raw response + parsed output + ground truth + score | Every rebalance, S1–S5 | `pipeline_logs/{run_id}/episodes/<date>_<n>.json` |
| Per-rebalance `MarketSnapshot` dump | Every rebalance | `snapshots/<date>.json` |
| Full `BacktestResult` (metrics, stress flag) | End of each scenario/normal run | `backtest_result.json` |
| `nav_curve.csv` + `weight_history.csv` + `trade_history.json` | Same as above | sibling files |
| Run-level aggregate (all profiles) | After model run completes | `{timestamp}/run_summary.json` |
| Per-profile figures (NAV, metrics, stress drawdown, correlation) | After each profile | `{profile}/figures/*.png` |
| Cross-model comparison figures | After all models complete | `{rebalance}/comparison_figures/*.png` |

Toggle the heavy ones via `logging:` in YAML if disk space matters.

---

## CLI Usage

```bash
# Print the (provider, model, profile, scenario) matrix without running
python -m portbench.experiments --config configs/experiments/default.yaml --dry-run

# Run the batch
python -m portbench.experiments --config configs/experiments/default.yaml

# Smoke test (no API keys, mock data, ~10s)
python -m portbench.experiments --config configs/experiments/smoke.yaml

# Post-run analysis: generate rankings, CEPS breakdown, stress gate figures
python -m portbench.experiments --analyze --rebalance monthly

# Specify a non-default output root
python -m portbench.experiments --analyze --rebalance weekly --output-root /data/experiments
```

Exit code: `0` if all experiments completed, `2` if any were recorded in `errors.jsonl`.

## Python Usage

```python
from portbench.experiments import BatchRunner, ExperimentConfig

cfg = ExperimentConfig.from_yaml("configs/experiments/default.yaml")
summary = BatchRunner(cfg).run()
print(summary["n_completed"], "ok |", summary["n_reused"], "reused |", summary["n_failed"], "failed")
# summary["run_timestamps"] → {"ark/doubao-seed-2-0-pro-260215": "20260510_042407", ...}
```

---

## Reused Components

| Component | File | Purpose |
| --------- | ---- | ------- |
| `EvalPipeline.enable_logging()` | `portbench/agent_eval/base.py` | Per-stage logging plumbing. |
| `EvalLogger` | `portbench/agent_eval/eval_logger.py` | Episode-level JSON writer. |
| `BacktestEngine` | `portbench/sandbox/engine.py` | Stateful backtest loop (optional `snapshot_dump_dir`). |
| `BacktestResult` | `portbench/sandbox/result.py` | Metrics container + `to_dict()` / `summary()`. |
| `OpenAIAdapter` / `AnthropicAdapter` | `portbench/agent_eval/llm_adapters.py` | Unmodified — provider registry passes `base_url` + `api_key_env`. |
| `MockAgentAdapter` | `portbench/agent_eval/mock_agent.py` | For harness smoke tests. |
| `STRESS_SCENARIOS` / `PROFILES` | `portbench/agent_eval/{stress_scenarios,investor_profiles}.py` | Canonical scenario + profile definitions. |
| `sandbox_plots` | `portbench/visualization/sandbox_plots.py` | Plot functions wrapped by `figures.py`. |
