# Visualization (`portbench/visualization/`)

## Overview

`portbench/visualization/` is the publication-quality plotting library for PortBench. It covers six thematic areas: CEPS metrics, stress tests, model rankings, dataset overview, QA samples, regime analysis, sandbox backtest output, investor profiles, and cross-asset correlation. All functions target `matplotlib` and are designed for use in papers, slides, and experiment figure pipelines.

---

## Module Layout

```
portbench/visualization/
  __init__.py           — full public re-export (all functions below)
  style.py              — apply_paper_style(), save_figure()
  ceps_plots.py         — CEPS radar, heatmap, violin
  stress_plots.py       — stress gate bar chart
  ranking_plots.py      — risk-first model ranking
  dataset_plots.py      — dataset overview, regime heatmap
  qa_sample_plots.py    — QA card display (T1–T7)
  regime_plots.py       — regime distribution, mock data helper
  sandbox_plots.py      — NAV curves, performance metrics, stress drawdown, profile overlay
  profile_plots.py      — investor profile alignment, radar
  correlation_plots.py  — heatmap (intra-class blocks), inter-class panel, rolling evolution
  correlation_graph.py  — Mantegna MST, threshold-filtered network
```

---

## Shared Utilities

**`style.py`**
- `apply_paper_style()` — sets matplotlib rcParams for publication quality (font, DPI, tight layout).
- `save_figure(fig, path, ...)` — saves to PNG/PDF, calls `tight_layout`, closes the figure.

---

## Thematic Submodules

### CEPS (`ceps_plots.py`)
- `plot_ceps_radar(...)` — radar chart comparing per-stage CEPS scores across models.
- `plot_ceps_heatmap(...)` — model × stage heatmap of CEPS scores.
- `plot_ceps_violin(...)` — distribution of CEPS scores across runs.

### Stress Gate (`stress_plots.py`)
- `plot_stress_gate(...)` — bar chart showing max drawdown per scenario vs threshold; pass/fail coloring.

### Risk-First Ranking (`ranking_plots.py`)
- `plot_risk_ranking(...)` — ranked table/bar chart of models, with those failing the stress gate visually excluded from performance rankings.

### Dataset Overview (`dataset_plots.py`)
- `plot_dataset_overview(...)` — per-asset-class data availability timeline and sample counts.
- `plot_regime_heatmap(...)` — date × asset-class heatmap of bull/bear/sideways/crisis labels.

### QA Sample Cards (`qa_sample_plots.py`)
- `plot_qa_sample_cards(...)` — renders multiple QA pairs as card layout.
- `plot_single_card(...)` — single QA card showing context window, question, answer, explanation.

### Regime Distributions (`regime_plots.py`)
- `plot_regime_distributions(...)` — stacked bar of regime proportions per asset class in train/val/test splits.
- `build_regime_data_from_mock(...)` — helper that generates synthetic regime label data for testing.

### Sandbox Backtest (`sandbox_plots.py`)
- `plot_sandbox_nav(...)` — NAV curve over time.
- `plot_sandbox_metrics(...)` — bar chart of Sharpe, Sortino, max drawdown, Calmar ratio.
- `plot_ceps_vs_pnl(...)` — scatter of CEPS score vs P&L across runs.
- `plot_stress_drawdown(...)` — heatmap of stress scenario max drawdowns.
- `plot_profile_nav(...)` — multi-profile NAV overlay on one axis.
- `load_sandbox_results(path)` — loads a single `BacktestResult` from JSON.
- `load_sandbox_results_full(model_dir)` — loads all `BacktestResult`s under a model directory.

Used by `portbench/experiments/figures.py` to render per-experiment PNGs.

### Investor Profiles (`profile_plots.py`)
- `plot_profile_alignment(...)` — bar/scatter showing how closely model allocations match the target profile (conservative / balanced / aggressive).
- `plot_profile_radar(...)` — radar chart of risk-return characteristics per investor profile.

---

## Cross-Asset Correlation

### Heatmap and Panel Views (`correlation_plots.py`)

These functions mirror the two-layer correlation model used in the S3 score (15% intra-class concentration penalty + 15% inter-class hedging credit).

| Function | Description |
|----------|-------------|
| `plot_correlation_heatmap(corr, asset_class_map, ...)` | Asset × asset heatmap, rows/cols grouped by class with colored block outlines. `annotate=True` prints numeric values for small matrices. |
| `plot_inter_class_correlation(corr, asset_class_map, ...)` | Two-panel figure: left = class × class heatmap; right = bar chart of per-class intra-class average. Explicitly mirrors the S3 hedging/concentration terms. |
| `plot_correlation_evolution(snapshot_dir, asset_class_map, window=6, ...)` | Rolling cross-asset average and per-class intra-class correlation over a backtest run, read from `BacktestEngine` snapshot dumps. |
| `load_processed_correlation(processed_dir)` | Loads `correlation_matrix.csv` + `asset_class_map.json` from `datasets/processed/`, returns `(corr_df, asset_class_map)`. |

**Internal helpers:**
- `_group_by_class(corr, asset_class_map)` — reorders rows/cols so intra-class assets are contiguous; returns reordered DataFrame + class span boundaries.
- `_intra_class_avgs(corr, asset_class_map)` — mean off-diagonal within-class correlation per class (feeds concentration penalty).
- `_inter_class_matrix(corr, asset_class_map)` — class × class matrix of cross-class pairwise averages (feeds hedging credit).
- `_load_snapshot_returns(snapshot_dir)` — loads trailing returns from per-rebalance BacktestEngine JSON snapshots.

### Graph Views (`correlation_graph.py`)

| Function | Description |
|----------|-------------|
| `plot_correlation_mst(corr, asset_class_map, ...)` | Mantegna MST (minimum spanning tree) of cross-asset relations. Edge weight = `sqrt(2(1−ρ))`; node color = asset class; edge color = positive (red) / negative (blue); edge width = `|ρ|`. |
| `plot_correlation_threshold(corr, asset_class_map, threshold=0.5, ...)` | Force-directed network keeping only edges with `|ρ| ≥ threshold`. Node size encodes filtered-graph degree; isolated nodes still drawn. |

**Internal helper:**
- `_build_full_graph(corr)` — constructs `networkx.Graph` with Mantegna distance weights and raw `rho` edge attributes.

---

## Integration with Experiment Pipeline

`portbench/experiments/figures.py` calls three visualization entry points:

| Call | When | Output |
|------|------|--------|
| `render_experiment_figures(profile_dir, ...)` | After each `(model, profile)` completes | `figures/nav.png`, `figures/metrics.png`, `figures/stress_drawdown.png`, `figures/correlation_evolution_*.png` |
| `render_dataset_correlation_figures(output_dir, processed_dir, ...)` | Once per batch | `figures/correlation_heatmap.png`, `figures/inter_class_correlation.png` |
| `render_model_summary_figures(model_dir, ...)` | After all profiles for one model complete | `figures/profile_nav.png` |

`render_dataset_correlation_figures` is a safe no-op if `datasets/processed/` artifacts are missing.

---

## Usage Examples

```python
from portbench.visualization import (
    load_processed_correlation,
    plot_correlation_heatmap,
    plot_inter_class_correlation,
    plot_correlation_mst,
    apply_paper_style,
    save_figure,
)

apply_paper_style()
corr, asset_class_map = load_processed_correlation("datasets/processed")

fig = plot_correlation_heatmap(corr, asset_class_map, annotate=False)
save_figure(fig, "outputs/correlation_heatmap.png")

fig = plot_inter_class_correlation(corr, asset_class_map)
save_figure(fig, "outputs/inter_class_correlation.png")

fig = plot_correlation_mst(corr, asset_class_map)
save_figure(fig, "outputs/correlation_mst.png")
```
