# Batch Experiments (`portbench/experiments/`)

## Overview

`portbench/experiments/` is the unified runner for **large-scale sweeps** across `(provider × model × profile × stress scenario)`. It wraps the Sandbox `BacktestEngine` with three additions that the original `examples/sandbox/run_backtest.py` lacked:

1. **YAML-driven sweeps** — declare all model/profile/scenario combinations in one config file; no per-script CLI surgery.
2. **Provider registry from `.env`** — every provider's `API_KEY` / `BASE_URL` / `MODEL` is read from environment variables. Adding a new provider requires one line in `PROVIDER_REGISTRY` plus three `.env` vars; no adapter code changes.
3. **Per-`(model, profile)` failure isolation** — one combination crashing does not abort the batch. All errors land in `errors.jsonl` with full tracebacks.

Every intermediate artifact is persisted: stage-level prompt/response/parsed-output, per-rebalance market snapshots, full `BacktestResult` (NAV/weights/trades), and rendered figures.

---

## Module Layout

```
portbench/experiments/
  __init__.py        — public exports (BatchRunner, ExperimentConfig, build_adapter, ...)
  providers.py       — PROVIDER_REGISTRY + build_adapter() / build_baseline() / build_mock()
  config.py          — ExperimentConfig + ModelSpec + LoggingConfig + YAML loader
  paths.py           — directory naming + save_backtest_result()
  runner.py          — BatchRunner: sweep + failure isolation + summary
  figures.py         — per-experiment plot rendering (NAV, metrics, stress drawdown)
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
|----------|----------|---------|
| `{PREFIX}_API_KEY`  | Yes | Auth credential. Missing → `RuntimeError`, **no silent fallback**. |
| `{PREFIX}_BASE_URL` | Optional (anthropic ignores) | OpenAI-compatible endpoint URL. |
| `{PREFIX}_MODEL`    | Optional | Default model when YAML omits `model:`. |

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
batch_id: default_apr25                # optional; defaults to batch_<timestamp>
data_provider: processed               # processed | mock
data_dir: datasets/processed
sec_dir: datasets/sec
rebalance: monthly                     # weekly | monthly | quarterly
initial_nav: 1000000
workers_per_experiment: 3              # parallel stress scenarios per (model, profile)
parallel_experiments: 1                # reserved; current runner is sequential per (model)
seed: 42
noise: 0.2                             # MockAgentAdapter only

models:
  - provider: dashscope                # uses DASHSCOPE_MODEL from .env
  - provider: tencent
    model: hunyuan-pro                 # explicit override
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
resume: false                          # true = skip already-completed (model, profile) pairs
```

**Notes**
- A `ModelSpec` element must have exactly one of `provider`, `baseline`, `mock`. Mixing them raises in `ModelSpec.kind()`.
- `stress_scenarios: all` resolves at runtime via `STRESS_SCENARIOS` (currently `2015_china_shock`, `2020_covid_flash_crash`, `2022_crypto_collapse`).
- When `data_provider: processed` is requested but `datasets/processed/equities.csv` is missing, the runner refuses to silently fall back to mock — set `data_provider: mock` explicitly if that is what you want.
- `use_tools: true` enables native tool-calling (calculator, correlation, statistical helpers) for S1/S2/S3 stages via `complete_with_tools()`. Has no effect on baseline or mock models — tool use is silently skipped for those.
- `propagation_weight` controls the CEPS cascade penalty: `ceps = mean_stage_scores - weight × Σmax(score[i]-score[i+1], 0)`. Changing this value enables sensitivity analysis of the ranking metric.
- `resume: true` reads `checkpoint.json` in the batch directory and skips any `(model, profile)` that already completed successfully. Useful when resuming after a crash or network interruption.

---

## Output Layout

```
EXPERIMENTS/{batch_id}/
├── batch_config.yaml                  # exact copy of the YAML used for the run
├── batch_summary.json                 # rows: one per (model, profile, phase, scenario); normal rows include std_ceps
├── errors.jsonl                       # one JSON per failed experiment (model, profile, stage, traceback)
├── checkpoint.json                    # completed (model, profile) pairs; used by resume: true
├── env_meta.json                      # git commit hash, Python version, created_at
├── analysis_figures/                  # generated by --analyze: rankings.png, stress_gate.png, ceps_breakdown.png
├── analysis_report.md                 # generated by --analyze: markdown table + figure links
└── {model_label}/                     # e.g. "tencent-hunyuan-pro", "baseline-equal_weight", "mock"
    ├── profile_comparison.json        # cross-profile summary + adaptation_score
    └── {profile}/
        ├── experiment.log             # per-experiment logging.Logger output
        ├── error.json                 # only if this (model, profile) failed
        ├── figures/
        │   ├── nav.png                # normal NAV curve
        │   ├── metrics.png            # normal performance metrics bar chart
        │   └── stress_drawdown.png    # stress scenarios heatmap
        ├── stress_{scenario}/
        │   ├── backtest_result.json   # full BacktestResult.to_dict() incl. stress_passed
        │   ├── summary.txt            # human-readable
        │   ├── nav_curve.csv          # date, nav
        │   ├── weight_history.csv     # date, <asset_columns>
        │   ├── trade_history.json     # all rebalances
        │   ├── snapshots/             # per-rebalance MarketSnapshot snapshots
        │   │   └── {YYYY-MM-DD}.json  #   decision_date, weights, regime, macro, trailing_ret, news preview
        │   └── pipeline_logs/         # only when use_pipeline=True (i.e. not baseline)
        │       └── {run_id}/          # EvalLogger output
        │           ├── run_meta.json
        │           ├── run_summary.json
        │           ├── errors.jsonl
        │           └── episodes/
        │               └── {date}_{seq:04d}.json   # per-stage prompt + raw_response + parsed + score
        └── normal/                    # only when stress gate passes AND run_normal=true
            └── (same structure as stress_*)
```

The pipeline-logs JSON inside `episodes/` is what enables prompt iteration and post-mortem audits: every S1/S2/S3 call's full prompt, model raw output, parsed dict, ground-truth comparison, and per-stage score is preserved.

---

## Failure Isolation

Granularity is **(model, profile)**. Inside one experiment, stress scenarios run in parallel; if any scenario raises, the whole `(model, profile)` is marked failed, recorded, and the batch moves to the next combination.

Two kinds of failures are recorded:

| Trigger | Where | Notes |
|---------|-------|-------|
| `build_adapter` raises (e.g. missing API key) | All profiles for that model fail at once | The model directory gets a top-level `error.json`. |
| `_run_profile_experiment` raises (e.g. JSON parse failure across all retries, network) | Just that `(model, profile)` | The profile directory gets `error.json`; same record appended to batch-level `errors.jsonl`. |

Set `on_error: fail_fast` in YAML to abort the batch on the first failure.

---

## Intermediate Artifacts (What Gets Saved)

| Artifact | When | Path |
|----------|------|------|
| Per-stage prompt + raw response + parsed output + ground truth + score | Every rebalance, S1–S5 | `pipeline_logs/{run_id}/episodes/<date>_<n>.json` (via `EvalLogger`) |
| Per-rebalance `MarketSnapshot` dump | Every rebalance | `snapshots/<date>.json` (via `BacktestEngine.snapshot_dump_dir`) |
| Full `BacktestResult` (metrics, profile fields, stress flag) | End of each scenario / normal run | `backtest_result.json` (`paths.save_backtest_result`) |
| `nav_curve.csv` + `weight_history.csv` + `trade_history.json` | Same as above | sibling files |
| Cross-profile aggregate (adaptation score) | After each model finishes | `{model_label}/profile_comparison.json` |
| Batch-level summary (one row per phase per scenario) | At the end | `batch_summary.json` |
| Figures (NAV, metrics, stress drawdown) | After each `(model, profile)` | `figures/*.png` (via `figures.render_experiment_figures`) |

Toggle the heavy ones via `logging:` in YAML if disk space matters.

---

## CLI Usage

```bash
# Print the (model, profile, scenario) matrix without running anything
python -m portbench.experiments --config configs/experiments/default.yaml --dry-run

# Run the batch
python -m portbench.experiments --config configs/experiments/default.yaml

# Override the batch_id from the command line
python -m portbench.experiments --config configs/experiments/default.yaml --batch-id ablation_run_3

# Resume a partially completed batch (skip already-completed (model, profile) pairs)
python -m portbench.experiments --config configs/experiments/default.yaml --batch-id ablation_run_3
# (set resume: true in YAML, or it will warn and overwrite)

# Post-batch analysis: generate rankings, CEPS breakdown, stress gate figures + analysis_report.md
python -m portbench.experiments --analyze --batch-id ablation_run_3

# Smoke test (no API keys, mock data, ~10s)
python -m portbench.experiments --config configs/experiments/smoke.yaml
```

Exit code: `0` if all experiments completed, `2` if any were recorded in `errors.jsonl`.

## Python Usage

```python
from portbench.experiments import BatchRunner, ExperimentConfig

cfg = ExperimentConfig.from_yaml("configs/experiments/default.yaml")
summary = BatchRunner(cfg).run()
print(summary["n_completed"], "ok |", summary["n_failed"], "failed")
```

---

## Reused Components

| Component | File | Purpose |
|-----------|------|---------|
| `EvalPipeline.enable_logging()` | `portbench/agent_eval/base.py` | Per-stage logging plumbing. |
| `EvalLogger` | `portbench/agent_eval/eval_logger.py` | Episode-level JSON writer. |
| `BacktestEngine` | `portbench/sandbox/engine.py` | Stateful backtest loop (now with optional `snapshot_dump_dir`). |
| `BacktestResult` | `portbench/sandbox/result.py` | Metrics container + `to_dict()` / `summary()`. |
| `OpenAIAdapter` / `AnthropicAdapter` | `portbench/agent_eval/llm_adapters.py` | Unmodified — provider registry passes them `base_url` + `api_key_env`. |
| `MockAgentAdapter` | `portbench/agent_eval/mock_agent.py` | For harness smoke tests. |
| `STRESS_SCENARIOS` / `PROFILES` | `portbench/agent_eval/{stress_scenarios,investor_profiles}.py` | Canonical scenario + profile definitions. |
| `sandbox_plots` | `portbench/visualization/sandbox_plots.py` | Plot functions wrapped by `figures.py`. |
