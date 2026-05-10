"""
analyze_runs: post-run analysis for all models in a rebalance directory.

Usage:
    from portbench.experiments.analysis import analyze_runs
    report_path = analyze_runs("monthly")

    # Or via CLI:
    python -m portbench.experiments --analyze --rebalance monthly
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import paths


def _load_run_summaries(
    output_root: str, rebalance: str
) -> list[dict]:
    """
    Scan EXPERIMENTS/{rebalance}/ and return all run_summary.json contents,
    using the best (most complete, then latest) run per model.
    """
    rebal_dir = paths.rebalance_dir(output_root, rebalance)
    if not rebal_dir.exists():
        return []

    summaries = []
    for prov_dir in sorted(rebal_dir.iterdir()):
        if not prov_dir.is_dir() or prov_dir.name.startswith("_") or prov_dir.name == "comparison_figures":
            continue
        for m_dir in sorted(prov_dir.iterdir()):
            if not m_dir.is_dir():
                continue
            # Find best run (most profiles complete, then latest)
            ts = paths.find_best_run(output_root, rebalance, prov_dir.name, m_dir.name, [])
            if not ts:
                continue
            summary_file = paths.run_dir(output_root, rebalance, prov_dir.name, m_dir.name, ts) / "run_summary.json"
            if summary_file.exists():
                data = json.loads(summary_file.read_text(encoding="utf-8"))
                summaries.append(data)
    return summaries


def _flatten_rows(summaries: list[dict]) -> list[dict]:
    """Flatten run_summary records into per-row dicts compatible with the analysis tables."""
    rows = []
    for s in summaries:
        model_key = f"{s['provider']}/{s['model_name']}"
        for profile_name, payload in s.get("profiles", {}).items():
            base = {
                "model": model_key,
                "profile": profile_name,
                "stress_gate_passed": payload.get("stress_gate_passed", False),
            }
            for sr in payload.get("stress_results", []):
                rows.append({**base, "phase": "stress", **sr})
            if payload.get("normal") is not None:
                import numpy as np
                normal = dict(payload["normal"])
                per_step = normal.pop("per_step_ceps", [])
                normal["std_ceps"] = round(float(np.std(per_step)), 6) if per_step else 0.0
                rows.append({**base, "phase": "normal", **normal})
    return rows


def analyze_runs(
    rebalance: str = "monthly",
    output_root: str = "EXPERIMENTS",
    logger=None,
) -> Path:
    """
    Scan EXPERIMENTS/{rebalance}/, generate figures, and write analysis_report.md.
    Returns path to the generated report.
    """
    from ..visualization.ranking_plots import plot_risk_ranking
    from ..visualization.stress_plots import plot_stress_gate
    from ..visualization.ceps_plots import plot_ceps_heatmap
    from ..visualization.style import save_figure

    log = logger.info if logger else print

    rebal_dir = paths.rebalance_dir(output_root, rebalance)
    if not rebal_dir.exists():
        raise FileNotFoundError(f"No results found at {rebal_dir}")

    summaries = _load_run_summaries(output_root, rebalance)
    if not summaries:
        raise FileNotFoundError(f"No run_summary.json files found under {rebal_dir}")

    rows = _flatten_rows(summaries)

    fig_dir = rebal_dir / "analysis_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate per-model metrics
    import numpy as np

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

    figures_written: list[str] = []

    # Fig 1: Risk-first model ranking
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

    if ranking_data:
        try:
            fig = plot_risk_ranking(ranking_data, title=f"Risk-First Ranking — {rebalance}")
            save_figure(fig, str(fig_dir / "rankings.png"), formats=("png",))
            figures_written.append("rankings.png")
        except Exception as exc:
            log(f"analysis: rankings.png skipped ({exc})")

    # Fig 2: Stress gate bars
    if model_stress:
        try:
            fig = plot_stress_gate(model_stress, title=f"Stress Gate — {rebalance}")  # type: ignore
            save_figure(fig, str(fig_dir / "stress_gate.png"), formats=("png",))
            figures_written.append("stress_gate.png")
        except Exception as exc:
            log(f"analysis: stress_gate.png skipped ({exc})")

    # Fig 3: CEPS heatmap
    if model_normal:
        try:
            ceps_totals: dict[str, float] = {}
            for model, normal_rows in model_normal.items():
                ceps_totals[model] = (
                    sum(r.get("mean_ceps", 0.0) for r in normal_rows) / len(normal_rows)
                )
            fig = plot_ceps_heatmap({m: {} for m in ceps_totals}, ceps_totals=ceps_totals,
                                    title=f"CEPS Breakdown — {rebalance}")
            save_figure(fig, str(fig_dir / "ceps_breakdown.png"), formats=("png",))
            figures_written.append("ceps_breakdown.png")
        except Exception as exc:
            log(f"analysis: ceps_breakdown.png skipped ({exc})")

    # Summary table
    table_rows = [
        {
            "model": r.get("model", ""),
            "profile": r.get("profile", ""),
            "return": f"{r.get('total_return', 0.0) * 100:+.2f}%",
            "sharpe": f"{r.get('sharpe_ratio', 0.0):.3f}",
            "mean_ceps": f"{r.get('mean_ceps', 0.0):.4f}",
            "std_ceps": f"{r.get('std_ceps', 0.0):.4f}",
            "stress": "PASS" if r.get("stress_gate_passed") else "FAIL",
        }
        for r in rows if r.get("phase") == "normal"
    ]
    table_rows.sort(key=lambda r: (r["model"], r["profile"]))

    report_path = rebal_dir / "analysis_report.md"
    lines = [
        f"# Analysis Report: {rebalance}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Models found: {len(summaries)}",
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


# Keep old name as alias for backward compat
def analyze_batch(batch_id: str, output_root: str = "EXPERIMENTS", logger=None) -> Path:
    """Deprecated: use analyze_runs(rebalance, output_root) instead."""
    import warnings
    warnings.warn("analyze_batch is deprecated; use analyze_runs(rebalance)", DeprecationWarning)
    return analyze_runs(rebalance=batch_id, output_root=output_root, logger=logger)
