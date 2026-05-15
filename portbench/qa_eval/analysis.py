"""
analyze_qa_results: post-run QA analysis across all models.

Usage:
    python -m portbench.experiments --analyze-qa --output-root EXPERIMENTS
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def analyze_qa_results(
    output_root: str = "EXPERIMENTS",
    logger=None,
) -> Path:
    """
    Read qa_summary.json, generate all cross-model figures, write report.

    Figures written to EXPERIMENTS/qa_eval/comparison_figures/.
    Per-model figures written to EXPERIMENTS/qa_eval/{provider}/{model}/figures/.
    """
    from ..visualization.qa_accuracy_plots import (
        plot_qa_accuracy_heatmap,
        plot_qa_per_template_comparison,
        plot_qa_template_radar,
        plot_qa_accuracy_by_regime,
        plot_qa_score_distribution,
        plot_qa_model_comparison,
    )
    from ..visualization.style import save_figure
    from . import paths as qpaths

    log = logger.info if logger else print

    root = qpaths.qa_root(output_root)
    summary_path = root / "qa_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"qa_summary.json not found in {root}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    models_data = summary.get("models", {})
    if not models_data:
        raise FileNotFoundError("No model data in qa_summary.json")

    comp_dir = root / "comparison_figures"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # ── Build cross-model data structures ────────────────────────────────────
    acc_data: dict[str, dict[str, float]] = {}
    regime_data: dict[str, dict[str, dict[str, float]]] = {}
    dist_data: dict[str, dict[str, list[float]]] = {}
    mean_acc: dict[str, float] = {}

    for label, mdata in models_data.items():
        provider = mdata.get("provider", "")
        model_name = mdata.get("model", "")
        mean_acc[label] = mdata.get("mean_accuracy", 0.0)
        templates = mdata.get("templates", {})

        acc_data[label] = {}
        regime_data[label] = {}
        dist_data[label] = {}

        for tid, tdata in templates.items():
            acc_data[label][tid] = tdata.get("accuracy", 0.0)
            regime_data[label][tid] = tdata.get("by_regime", {})

            results_file = (
                qpaths.qa_template_dir(output_root, provider, model_name, tid)
                / "results.jsonl"
            )
            scores: list[float] = []
            if results_file.exists():
                for line in results_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        scores.append(json.loads(line).get("score", 0.0))
            dist_data[label][tid] = scores

    figures_written: list[str] = []

    def _save(fig, name: str) -> None:
        save_figure(fig, str(comp_dir / name), formats=("png",))
        figures_written.append(name)
        log(f"qa analysis: {name}")

    # Fig 1: accuracy heatmap (model × template)
    try:
        _save(plot_qa_accuracy_heatmap(acc_data), "accuracy_heatmap.png")
    except Exception as exc:
        log(f"qa analysis: accuracy_heatmap.png skipped ({exc})")

    # Fig 2: per-template grouped bar (models side by side per template)
    try:
        _save(plot_qa_per_template_comparison(acc_data), "per_template_comparison.png")
    except Exception as exc:
        log(f"qa analysis: per_template_comparison.png skipped ({exc})")

    # Fig 3: T1-T7 radar per model
    try:
        _save(plot_qa_template_radar(acc_data), "template_radar.png")
    except Exception as exc:
        log(f"qa analysis: template_radar.png skipped ({exc})")

    # Fig 4: accuracy by market regime
    try:
        _save(plot_qa_accuracy_by_regime(regime_data), "accuracy_by_regime.png")
    except Exception as exc:
        log(f"qa analysis: accuracy_by_regime.png skipped ({exc})")

    # Fig 5: score distribution (box + jitter)
    try:
        _save(plot_qa_score_distribution(dist_data), "score_distribution.png")
    except Exception as exc:
        log(f"qa analysis: score_distribution.png skipped ({exc})")

    # Fig 6: model comparison (mean accuracy ranking)
    try:
        _save(plot_qa_model_comparison(mean_acc), "model_comparison.png")
    except Exception as exc:
        log(f"qa analysis: model_comparison.png skipped ({exc})")

    # ── Figure index ──────────────────────────────────────────────────────────
    _write_qa_figure_index(comp_dir, figures_written)

    # ── Markdown report ───────────────────────────────────────────────────────
    report_path = _write_qa_report(
        root, models_data, acc_data, regime_data, mean_acc, figures_written
    )
    log(f"QA analysis report: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMPLATE_NAMES = {
    "T1": "Return Prediction",
    "T2": "VaR Assessment",
    "T3": "Position Sizing",
    "T4": "Pairwise Allocation",
    "T5": "Multi-Asset Optimization",
    "T6": "Rebalancing Decision",
    "T7": "Regime Detection",
}
_TEMPLATE_ORDER = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]

_FIGURE_DESCRIPTIONS = {
    "accuracy_heatmap.png": (
        "**Accuracy Heatmap** — Model × Template grid showing per-cell accuracy (0-1). "
        "RdBu colormap: blue = high, red = low. Use to spot which model-template pairs "
        "are strong or weak."
    ),
    "per_template_comparison.png": (
        "**Per-Template Comparison** — Grouped bar chart: each group = one template "
        "(T1-T7), each bar = one model. Directly compares all models at each task level."
    ),
    "template_radar.png": (
        "**Capability Radar** — Spider chart with T1-T7 on 7 axes. Each polygon = one "
        "model. Reveals overall capability shape: balanced vs. specialized."
    ),
    "accuracy_by_regime.png": (
        "**Accuracy by Market Regime** — Grouped bars: templates on x-axis, bar color = "
        "market regime (bull/bear/sideways/crisis). Averaged across all models. "
        "Highlights regime-specific difficulty per task."
    ),
    "score_distribution.png": (
        "**Score Distribution** — Box + jitter plot of individual question scores "
        "per template (pooled across all models). Shows variance and whether scores "
        "cluster at 0/1 (binary task) or spread continuously."
    ),
    "model_comparison.png": (
        "**Model Ranking** — Horizontal bar chart of mean accuracy across all templates, "
        "sorted descending. Quick overall performance overview."
    ),
}


def _write_qa_figure_index(comp_dir: Path, figures_written: list[str]) -> None:
    lines = [
        "# QA Evaluation — Figure Index",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for fname in figures_written:
        desc = _FIGURE_DESCRIPTIONS.get(fname, "")
        lines += [f"## `{fname}`", "", desc, ""]
    (comp_dir / "figure_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_qa_report(
    root: Path,
    models_data: dict,
    acc_data: dict,
    regime_data: dict,
    mean_acc: dict,
    figures_written: list[str],
) -> Path:
    active_templates = [
        t for t in _TEMPLATE_ORDER
        if any(t in acc_data.get(m, {}) for m in acc_data)
    ]

    lines = [
        "# QA Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Models: {len(models_data)}",
        "",
        "## Model Comparison",
        "",
        "| model | " + " | ".join(active_templates) + " | mean |",
        "|-------|" + "|".join("------" for _ in active_templates) + "|------|",
    ]

    for label in sorted(models_data.keys(), key=lambda m: -mean_acc.get(m, 0.0)):
        vals = [
            f"{acc_data.get(label, {}).get(t, 0.0):.3f}"
            for t in active_templates
        ]
        lines.append(
            f"| {label} | " + " | ".join(vals) + f" | {mean_acc.get(label, 0.0):.3f} |"
        )

    for tid in active_templates:
        tname = _TEMPLATE_NAMES.get(tid, tid)
        regimes = sorted({
            r for m in regime_data for r in regime_data[m].get(tid, {})
        })
        lines += [
            "",
            f"### {tid} — {tname}",
            "",
            "| model | accuracy | n_total | " + " | ".join(regimes) + " |",
            "|-------|----------|---------|" + "|".join("------" for _ in regimes) + "|",
        ]
        for label in sorted(models_data.keys(), key=lambda m: -mean_acc.get(m, 0.0)):
            tdata = models_data[label].get("templates", {}).get(tid, {})
            regime_vals = [
                f"{regime_data.get(label, {}).get(tid, {}).get(r, 0.0):.3f}"
                for r in regimes
            ]
            lines.append(
                f"| {label} | {tdata.get('accuracy', 0.0):.3f} | "
                f"{tdata.get('n_total', 0)} | " + " | ".join(regime_vals) + " |"
            )

    if figures_written:
        lines += ["", "## Figures", ""]
        for fname in figures_written:
            lines.append(f"- [{fname}](comparison_figures/{fname})")

    report_path = root / "qa_analysis_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# Deprecated alias
def analyze_qa_batch(batch_id: str, output_root: str = "EXPERIMENTS", logger=None) -> Path:
    return analyze_qa_results(output_root=output_root, logger=logger)
