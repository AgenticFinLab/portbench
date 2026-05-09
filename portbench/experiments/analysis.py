"""
analyze_batch: one-command post-run analysis for a completed batch experiment.

Usage:
    from portbench.experiments.analysis import analyze_batch
    report_path = analyze_batch("my_batch_id")

    # Or via CLI:
    python -m portbench.experiments --analyze --batch-id my_batch_id
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def analyze_batch(
    batch_id: str,
    output_root: str = "EXPERIMENTS",
    logger=None,
) -> Path:
    """
    Read batch_summary.json, generate figures, and write analysis_report.md.

    Returns the path to the generated report.
    """
    from . import paths
    from ..visualization.ranking_plots import plot_risk_ranking
    from ..visualization.stress_plots import plot_stress_gate
    from ..visualization.ceps_plots import plot_ceps_heatmap
    from ..visualization.style import save_figure

    log = logger.info if logger else print

    bd = paths.batch_dir(output_root, batch_id)
    summary_path = bd / "batch_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"batch_summary.json not found in {bd}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("rows", [])

    fig_dir = bd / "analysis_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Build aggregated per-model metrics from normal rows ---
    model_normal: dict[str, list[dict]] = {}
    model_stress: dict[str, dict[str, dict]] = {}

    for row in rows:
        model = row["model"]
        if row.get("phase") == "normal":
            model_normal.setdefault(model, []).append(row)
        elif row.get("phase") == "stress":
            model_stress.setdefault(model, {})
            sc = row.get("scenario", "unknown")
            model_stress[model].setdefault(sc, {
                "scenario_name": sc,
                "mean_ceps": row.get("mean_ceps", 0.0),
                "min_pass_score": 0.4,
                "passed": row.get("stress_gate_passed", False),
            })

    # --- Figure 1: Risk-first model ranking ---
    ranking_data = []
    for model, normal_rows in model_normal.items():
        ceps_vals = [r.get("mean_ceps", 0.0) for r in normal_rows]
        std_vals = [r.get("std_ceps", 0.0) for r in normal_rows]
        gate_passed = all(r.get("stress_gate_passed", False) for r in normal_rows)
        ranking_data.append({
            "model_name": model,
            "mean_ceps": sum(ceps_vals) / len(ceps_vals) if ceps_vals else 0.0,
            "std_ceps": sum(std_vals) / len(std_vals) if std_vals else 0.0,
            "risk_gate_passed": gate_passed,
        })

    figures_written: list[str] = []
    if ranking_data:
        try:
            fig = plot_risk_ranking(ranking_data, title=f"Risk-First Ranking — {batch_id}")
            save_figure(fig, str(fig_dir / "rankings.png"), formats=("png",))
            figures_written.append("rankings.png")
            log(f"analysis: rankings.png")
        except Exception as exc:
            log(f"analysis: rankings.png skipped ({exc})")

    # --- Figure 2: Stress gate grouped bars ---
    if model_stress:
        try:
            fig = plot_stress_gate(
                model_stress,  # type: ignore[arg-type]
                title=f"Stress Gate — {batch_id}",
            )
            save_figure(fig, str(fig_dir / "stress_gate.png"), formats=("png",))
            figures_written.append("stress_gate.png")
            log("analysis: stress_gate.png")
        except Exception as exc:
            log(f"analysis: stress_gate.png skipped ({exc})")

    # --- Figure 3: CEPS heatmap (per-model mean_ceps total) ---
    # Requires per-stage data; fall back gracefully when absent.
    if model_normal:
        try:
            ceps_results: dict[str, dict[str, float]] = {}
            ceps_totals: dict[str, float] = {}
            for model, normal_rows in model_normal.items():
                ceps_totals[model] = (
                    sum(r.get("mean_ceps", 0.0) for r in normal_rows) / len(normal_rows)
                )
                # per-stage means are not stored in batch_summary (they're in pipeline logs)
                # so we leave stage scores empty — heatmap will show only totals column
                ceps_results[model] = {}

            fig = plot_ceps_heatmap(
                ceps_results,
                ceps_totals=ceps_totals,
                title=f"CEPS Breakdown — {batch_id}",
            )
            save_figure(fig, str(fig_dir / "ceps_breakdown.png"), formats=("png",))
            figures_written.append("ceps_breakdown.png")
            log("analysis: ceps_breakdown.png")
        except Exception as exc:
            log(f"analysis: ceps_breakdown.png skipped ({exc})")

    # --- Build summary table rows ---
    table_rows: list[dict] = []
    for row in rows:
        if row.get("phase") != "normal":
            continue
        table_rows.append({
            "model": row.get("model", ""),
            "profile": row.get("profile", ""),
            "return": f"{row.get('total_return', 0.0) * 100:+.2f}%",
            "sharpe": f"{row.get('sharpe_ratio', 0.0):.3f}",
            "mean_ceps": f"{row.get('mean_ceps', 0.0):.4f}",
            "std_ceps": f"{row.get('std_ceps', 0.0):.4f}",
            "stress": "PASS" if row.get("stress_gate_passed") else "FAIL",
        })
    table_rows.sort(key=lambda r: (r["model"], r["profile"]))

    # --- Write analysis_report.md ---
    report_path = bd / "analysis_report.md"
    lines = [
        f"# Batch Analysis: {batch_id}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Completed: {summary.get('n_completed', '?')} | "
        f"Failed: {summary.get('n_failed', '?')} | "
        f"Elapsed: {summary.get('elapsed_seconds', '?')}s",
        "",
        "## Summary Table",
        "",
        "| model | profile | return | sharpe | mean_ceps | std_ceps | stress |",
        "|-------|---------|--------|--------|-----------|----------|--------|",
    ]
    for r in table_rows:
        lines.append(
            f"| {r['model']} | {r['profile']} | {r['return']} | "
            f"{r['sharpe']} | {r['mean_ceps']} | {r['std_ceps']} | {r['stress']} |"
        )

    if figures_written:
        lines += ["", "## Figures", ""]
        for fname in figures_written:
            lines.append(f"- [{fname}](analysis_figures/{fname})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"analysis report: {report_path}")
    return report_path
