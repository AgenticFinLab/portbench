"""
analyze_qa_batch: one-command post-run analysis for a completed QA evaluation.

Usage:
    python -m portbench.experiments --analyze-qa --batch-id my_batch_id
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def analyze_qa_batch(
    batch_id: str,
    output_root: str = "EXPERIMENTS",
    logger=None,
) -> Path:
    """
    Read qa_summary.json, generate cross-model figures, and write qa_analysis_report.md.
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

    root = qpaths.qa_root(output_root, batch_id)
    summary_path = root / "qa_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"qa_summary.json not found in {root}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    models_data = summary.get("models", {})

    fig_dir = root / "analysis_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    figures_written: list[str] = []

    # Build cross-model data structures
    acc_data: dict[str, dict[str, float]] = {}
    regime_data: dict[str, dict[str, dict[str, float]]] = {}
    dist_data: dict[str, dict[str, list[float]]] = {}
    mean_acc: dict[str, float] = {}

    for model, mdata in models_data.items():
        mean_acc[model] = mdata.get("mean_accuracy", 0.0)
        templates = mdata.get("templates", {})

        acc_data[model] = {}
        regime_data[model] = {}
        dist_data[model] = {}

        for tid, tdata in templates.items():
            acc_data[model][tid] = tdata.get("accuracy", 0.0)
            regime_data[model][tid] = tdata.get("by_regime", {})

            # Load per-question scores from results.jsonl
            results_file = qpaths.qa_template_dir(
                output_root, batch_id, model, tid
            ) / "results.jsonl"
            scores = []
            if results_file.exists():
                for line in results_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        scores.append(rec.get("score", 0.0))
            dist_data[model][tid] = scores

    # Figure 1: accuracy heatmap
    if acc_data:
        try:
            fig = plot_qa_accuracy_heatmap(acc_data, title=f"QA Accuracy — {batch_id}")
            save_figure(fig, str(fig_dir / "accuracy_heatmap.png"), formats=("png",))
            figures_written.append("accuracy_heatmap.png")
            log("analysis: accuracy_heatmap.png")
        except Exception as exc:
            log(f"analysis: accuracy_heatmap.png skipped ({exc})")

    # Figure 2: by regime
    if regime_data:
        try:
            fig = plot_qa_accuracy_by_regime(regime_data, title=f"QA by Regime — {batch_id}")
            save_figure(fig, str(fig_dir / "accuracy_by_regime.png"), formats=("png",))
            figures_written.append("accuracy_by_regime.png")
            log("analysis: accuracy_by_regime.png")
        except Exception as exc:
            log(f"analysis: accuracy_by_regime.png skipped ({exc})")

    # Figure 3: score distribution
    if dist_data:
        try:
            fig = plot_qa_score_distribution(dist_data, title=f"QA Scores — {batch_id}")
            save_figure(fig, str(fig_dir / "score_distribution.png"), formats=("png",))
            figures_written.append("score_distribution.png")
            log("analysis: score_distribution.png")
        except Exception as exc:
            log(f"analysis: score_distribution.png skipped ({exc})")

    # Figure 4: model comparison
    if mean_acc:
        try:
            fig = plot_qa_model_comparison(mean_acc, title=f"QA Model Comparison — {batch_id}")
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

    report_path = root / "qa_analysis_report.md"
    lines = [
        f"# QA Evaluation Report: {batch_id}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Models: {len(models_data)} | "
        f"Elapsed: {summary.get('elapsed_seconds', '?')}s",
        "",
        "## Model Comparison",
        "",
        "| model | " + " | ".join(t for t in template_order if any(t in acc_data.get(m, {}) for m in acc_data)) + " | mean |",
        "|-------|" + "|".join("------" for t in template_order if any(t in acc_data.get(m, {}) for m in acc_data)) + "|------|",
    ]

    active_templates = [t for t in template_order if any(t in acc_data.get(m, {}) for m in acc_data)]
    for model in sorted(models_data.keys()):
        vals = [f"{acc_data.get(model, {}).get(t, 0.0):.3f}" for t in active_templates]
        mean_val = f"{mean_acc.get(model, 0.0):.3f}"
        lines.append(f"| {model} | " + " | ".join(vals) + f" | {mean_val} |")

    # Per-template detail with regime breakdown
    for tid in active_templates:
        tname = template_names.get(tid, tid)
        lines += [
            "",
            f"### {tid} — {tname}",
            "",
            "| model | accuracy | n_total | " + " | ".join(
                sorted({r for m in regime_data for r in regime_data[m].get(tid, {})})
            ) + " |",
            "|-------|----------|---------|" + "|".join(
                "------" for _ in sorted({r for m in regime_data for r in regime_data[m].get(tid, {})})
            ) + "|",
        ]
        regimes_for_t = sorted({r for m in regime_data for r in regime_data[m].get(tid, {})})
        for model in sorted(models_data.keys()):
            tdata = models_data[model].get("templates", {}).get(tid, {})
            acc = f"{tdata.get('accuracy', 0.0):.3f}"
            n = str(tdata.get("n_total", 0))
            regime_vals = [
                f"{regime_data.get(model, {}).get(tid, {}).get(r, 0.0):.3f}"
                for r in regimes_for_t
            ]
            lines.append(f"| {model} | {acc} | {n} | " + " | ".join(regime_vals) + " |")

    if figures_written:
        lines += ["", "## Figures", ""]
        for fname in figures_written:
            lines.append(f"- [{fname}](analysis_figures/{fname})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"QA analysis report: {report_path}")
    return report_path
