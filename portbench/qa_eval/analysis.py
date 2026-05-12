"""
analyze_qa_results: post-run analysis across all models in EXPERIMENTS/qa_eval/.

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
    Read qa_summary.json, generate cross-model figures, and write qa_analysis_report.md.

    Scans EXPERIMENTS/qa_eval/ for results written by QAEvaluator.
    """
    from ..visualization.qa_accuracy_plots import (
        plot_qa_accuracy_heatmap,
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

    fig_dir = root / "analysis_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    figures_written: list[str] = []

    # Build cross-model data structures
    # model key in models_data is "{provider}/{model_name}" label
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
            scores = []
            if results_file.exists():
                for line in results_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        scores.append(rec.get("score", 0.0))
            dist_data[label][tid] = scores

    # Figure 1: accuracy heatmap
    if acc_data:
        try:
            fig = plot_qa_accuracy_heatmap(acc_data, title="QA Accuracy")
            save_figure(fig, str(fig_dir / "accuracy_heatmap.png"), formats=("png",))
            figures_written.append("accuracy_heatmap.png")
            log("analysis: accuracy_heatmap.png")
        except Exception as exc:
            log(f"analysis: accuracy_heatmap.png skipped ({exc})")

    # Figure 2: by regime
    if regime_data:
        try:
            fig = plot_qa_accuracy_by_regime(regime_data, title="QA Accuracy by Regime")
            save_figure(fig, str(fig_dir / "accuracy_by_regime.png"), formats=("png",))
            figures_written.append("accuracy_by_regime.png")
            log("analysis: accuracy_by_regime.png")
        except Exception as exc:
            log(f"analysis: accuracy_by_regime.png skipped ({exc})")

    # Figure 3: score distribution
    if dist_data:
        try:
            fig = plot_qa_score_distribution(dist_data, title="QA Score Distribution")
            save_figure(fig, str(fig_dir / "score_distribution.png"), formats=("png",))
            figures_written.append("score_distribution.png")
            log("analysis: score_distribution.png")
        except Exception as exc:
            log(f"analysis: score_distribution.png skipped ({exc})")

    # Figure 4: model comparison
    if mean_acc:
        try:
            fig = plot_qa_model_comparison(mean_acc, title="QA Model Comparison")
            save_figure(fig, str(fig_dir / "model_comparison.png"), formats=("png",))
            figures_written.append("model_comparison.png")
            log("analysis: model_comparison.png")
        except Exception as exc:
            log(f"analysis: model_comparison.png skipped ({exc})")

    # --- Write qa_analysis_report.md ---
    template_order = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    template_names = {
        "T1": "Return Prediction",
        "T2": "VaR Assessment",
        "T3": "Position Sizing",
        "T4": "Pairwise Allocation",
        "T5": "Multi-Asset Optimization",
        "T6": "Rebalancing Decision",
        "T7": "Regime Detection",
    }

    active_templates = [
        t for t in template_order
        if any(t in acc_data.get(m, {}) for m in acc_data)
    ]

    report_path = root / "qa_analysis_report.md"
    lines = [
        "# QA Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Models: {len(models_data)} | Elapsed: {summary.get('elapsed_seconds', '?')}s",
        "",
        "## Model Comparison",
        "",
        "| model | " + " | ".join(active_templates) + " | mean |",
        "|-------|" + "|".join("------" for _ in active_templates) + "|------|",
    ]

    for label in sorted(models_data.keys()):
        vals = [f"{acc_data.get(label, {}).get(t, 0.0):.3f}" for t in active_templates]
        mean_val = f"{mean_acc.get(label, 0.0):.3f}"
        lines.append(f"| {label} | " + " | ".join(vals) + f" | {mean_val} |")

    for tid in active_templates:
        tname = template_names.get(tid, tid)
        regimes_for_t = sorted({r for m in regime_data for r in regime_data[m].get(tid, {})})
        lines += [
            "",
            f"### {tid} — {tname}",
            "",
            "| model | accuracy | n_total | " + " | ".join(regimes_for_t) + " |",
            "|-------|----------|---------|" + "|".join("------" for _ in regimes_for_t) + "|",
        ]
        for label in sorted(models_data.keys()):
            tdata = models_data[label].get("templates", {}).get(tid, {})
            acc = f"{tdata.get('accuracy', 0.0):.3f}"
            n = str(tdata.get("n_total", 0))
            regime_vals = [
                f"{regime_data.get(label, {}).get(tid, {}).get(r, 0.0):.3f}"
                for r in regimes_for_t
            ]
            lines.append(f"| {label} | {acc} | {n} | " + " | ".join(regime_vals) + " |")

    if figures_written:
        lines += ["", "## Figures", ""]
        for fname in figures_written:
            lines.append(f"- [{fname}](analysis_figures/{fname})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"QA analysis report: {report_path}")
    return report_path


# Deprecated alias
def analyze_qa_batch(batch_id: str, output_root: str = "EXPERIMENTS", logger=None) -> Path:
    return analyze_qa_results(output_root=output_root, logger=logger)
