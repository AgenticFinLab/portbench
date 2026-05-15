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


def _load_stage_scores(output_root: str, rebalance: str) -> dict[str, dict[str, float]]:
    """
    Read pipeline_logs episode files for each LLM model and return
    averaged per-stage scores: {model_key: {"S1": float, ..., "S5": float}}.
    Baselines are skipped (no pipeline logs).
    """
    import numpy as np

    rebal_dir = paths.rebalance_dir(output_root, rebalance)
    results: dict[str, dict[str, list[float]]] = {}

    for prov_dir in sorted(rebal_dir.iterdir()):
        if not prov_dir.is_dir() or prov_dir.name.startswith("_") or prov_dir.name == "comparison_figures":
            continue
        if prov_dir.name == "baseline":
            continue
        for m_dir in sorted(prov_dir.iterdir()):
            if not m_dir.is_dir():
                continue
            model_key = f"{prov_dir.name}/{m_dir.name}"
            ts = paths.find_best_run(output_root, rebalance, prov_dir.name, m_dir.name, [])
            if not ts:
                continue
            r_dir = paths.run_dir(output_root, rebalance, prov_dir.name, m_dir.name, ts)

            stage_accum: dict[str, list[float]] = {}
            for profile_dir in sorted(r_dir.iterdir()):
                if not profile_dir.is_dir() or profile_dir.name not in (
                    "conservative", "balanced", "aggressive"
                ):
                    continue
                logs_root = profile_dir / "normal" / "pipeline_logs"
                if not logs_root.exists():
                    continue
                for ep_file in sorted(logs_root.glob("*/episodes/*.json")):
                    try:
                        ep = json.loads(ep_file.read_text(encoding="utf-8"))
                        for sl in ep.get("stages", []):
                            sid = sl.get("stage_id", "")
                            sc = float(sl.get("score", 0.0))
                            stage_accum.setdefault(sid, []).append(sc)
                    except Exception:
                        continue

            if stage_accum:
                results[model_key] = {
                    sid: round(float(np.mean(scores)), 4)
                    for sid, scores in stage_accum.items()
                }

    return results


def _write_figure_index(comp_dir: Path, rebalance: str) -> None:
    """Write figure_index.md describing every figure in comparison_figures/."""
    _DESCRIPTIONS: dict[str, str] = {
        # Analysis figures (copied from analysis_figures/)
        "rankings.png": (
            "**Risk-First Ranking** — Horizontal bar chart ranking models by mean CEPS score. "
            "Models that passed all stress gates are color-coded; failed models are grayed/hatched. "
            "Use this to compare overall pipeline quality across models."
        ),
        "stress_gate.png": (
            "**Stress Gate Results** — Grouped bar chart showing mean CEPS per model per stress scenario "
            "(2008 Financial Crisis, 2020 COVID Crash, 2022 Crypto Collapse). "
            "Bars below the pass threshold are hatched in red. "
            "Use this to identify which models fail under which market regime."
        ),
        "ceps_breakdown.png": (
            "**CEPS Stage Breakdown** — Heatmap of per-stage scores (S1–S5) and CEPS total per model. "
            "Red cells indicate stages with high error propagation. "
            "Use this to diagnose where in the pipeline each model loses accuracy."
        ),
        "risk_return_scatter.png": (
            "**Risk-Return Scatter** — Each point is a (model, investor profile) pair. "
            "X-axis: worst stress-test max drawdown (lower = riskier). "
            "Y-axis: normal-period Sharpe ratio. "
            "Green = passed stress gate; red = failed. "
            "Use this to identify models with high return but poor risk management (top-left quadrant)."
        ),
        # Comparison figures (generated by render_batch_comparison_figures)
        "nav_comparison_conservative.png": (
            "**NAV Curves — Conservative** — Normalized NAV trajectories for all models under the "
            "conservative investor profile (strict drawdown tolerance). "
            "Use this to compare capital preservation across models."
        ),
        "nav_comparison_balanced.png": (
            "**NAV Curves — Balanced** — NAV trajectories for all models under the balanced profile. "
            "Represents the baseline risk/return trade-off."
        ),
        "nav_comparison_aggressive.png": (
            "**NAV Curves — Aggressive** — NAV trajectories under the aggressive profile "
            "(relaxed drawdown tolerance). Highlights return-seeking behavior."
        ),
        "metrics_comparison_conservative.png": (
            "**Metrics Bar Chart — Conservative** — Side-by-side comparison of total return, Sharpe ratio, "
            "max drawdown, and mean CEPS for all models under the conservative profile."
        ),
        "metrics_comparison_balanced.png": (
            "**Metrics Bar Chart — Balanced** — Same as above for the balanced investor profile."
        ),
        "metrics_comparison_aggressive.png": (
            "**Metrics Bar Chart — Aggressive** — Same as above for the aggressive investor profile."
        ),
    }

    # Collect all files present
    present = sorted(p.name for p in comp_dir.glob("*.png"))

    lines = [
        f"# Figure Index — {rebalance}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Figures in this directory are produced by two pipelines:",
        "- `analysis_figures/` (from `--analyze`): rankings, stress gate, CEPS breakdown, risk-return scatter",
        "- `render_batch_comparison_figures`: NAV curves and metrics comparisons per investor profile",
        "",
    ]

    for fname in present:
        desc = _DESCRIPTIONS.get(fname)
        if desc:
            lines.append(f"## `{fname}`")
            lines.append("")
            lines.append(desc)
        else:
            # stress_drawdown_{model}.png pattern
            lines.append(f"## `{fname}`")
            lines.append("")
            if fname.startswith("stress_drawdown_"):
                model = fname[len("stress_drawdown_"):-4].replace("_", "/", 1)
                lines.append(
                    f"**Stress Drawdown Heatmap — {model}** — Rows = stress scenarios, "
                    "columns = investor profiles. Cell color = magnitude of max drawdown; "
                    "✓/✗ indicates pass/fail against the profile's drawdown tolerance."
                )
        lines.append("")

    (comp_dir / "figure_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    from ..visualization.risk_return_plots import plot_risk_return_scatter
    from ..visualization.style import save_figure

    log = logger.info if logger else print

    rebal_dir = paths.rebalance_dir(output_root, rebalance)
    if not rebal_dir.exists():
        raise FileNotFoundError(f"No results found at {rebal_dir}")

    summaries = _load_run_summaries(output_root, rebalance)
    if not summaries:
        raise FileNotFoundError(f"No run_summary.json files found under {rebal_dir}")

    rows = _flatten_rows(summaries)

    fig_dir = rebal_dir / "comparison_figures"
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
            abs_dd = abs(row.get("max_drawdown", 0.0))
            passed = row.get("passed", False)
            tol = row.get("tolerance", 0.2)
            if sc not in model_stress[model]:
                model_stress[model][sc] = {
                    "scenario_name": sc,
                    "mean_ceps": abs_dd,
                    "min_pass_score": tol,
                    "passed": passed,
                }
            else:
                # keep worst drawdown across profiles for this model/scenario
                if abs_dd > model_stress[model][sc]["mean_ceps"]:
                    model_stress[model][sc]["mean_ceps"] = abs_dd
                    model_stress[model][sc]["passed"] = passed

    figures_written: list[str] = []

    # Fig 1: Risk-first model ranking  (LLM models only — baselines have CEPS=0)
    ranking_data = []
    for model, normal_rows in model_normal.items():
        if model.startswith("baseline/"):
            continue
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
            # plot_stress_gate expects {model: [list of scenario dicts]}
            stress_list = {m: list(sc.values()) for m, sc in model_stress.items()}
            fig = plot_stress_gate(stress_list, title=f"Stress Gate — {rebalance}")  # type: ignore
            save_figure(fig, str(fig_dir / "stress_gate.png"), formats=("png",))
            figures_written.append("stress_gate.png")
        except Exception as exc:
            log(f"analysis: stress_gate.png skipped ({exc})")

    # Fig 3: CEPS heatmap  (LLM models only, with per-stage scores from pipeline_logs)
    if model_normal:
        try:
            ceps_totals: dict[str, float] = {}
            for model, normal_rows in model_normal.items():
                if model.startswith("baseline/"):
                    continue
                ceps_totals[model] = (
                    sum(r.get("mean_ceps", 0.0) for r in normal_rows) / len(normal_rows)
                )
            if ceps_totals:
                stage_scores = _load_stage_scores(output_root, rebalance)
                # Fill zeros for models without pipeline_logs
                results_map = {m: stage_scores.get(m, {}) for m in ceps_totals}
                fig = plot_ceps_heatmap(results_map, ceps_totals=ceps_totals,
                                        title=f"CEPS Breakdown — {rebalance}")
                save_figure(fig, str(fig_dir / "ceps_breakdown.png"), formats=("png",))
                figures_written.append("ceps_breakdown.png")
        except Exception as exc:
            log(f"analysis: ceps_breakdown.png skipped ({exc})")

    # Fig 4: Risk-return scatter  (LLM models only — baselines have no CEPS)
    try:
        llm_rows = [r for r in rows if not r.get("model", "").startswith("baseline/")]
        fig = plot_risk_return_scatter(
            llm_rows,
            title=f"Risk vs. Return — {rebalance}",
        )
        save_figure(fig, str(fig_dir / "risk_return_scatter.png"), formats=("png",))
        figures_written.append("risk_return_scatter.png")
    except Exception as exc:
        log(f"analysis: risk_return_scatter.png skipped ({exc})")

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
            lines.append(f"- [{fname}](comparison_figures/{fname})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"analysis report: {report_path}")

    _write_figure_index(fig_dir, rebalance)
    log(f"figure index: {fig_dir / 'figure_index.md'}")

    return report_path


# Keep old name as alias for backward compat
def analyze_batch(batch_id: str, output_root: str = "EXPERIMENTS", logger=None) -> Path:
    """Deprecated: use analyze_runs(rebalance, output_root) instead."""
    import warnings
    warnings.warn("analyze_batch is deprecated; use analyze_runs(rebalance)", DeprecationWarning)
    return analyze_runs(rebalance=batch_id, output_root=output_root, logger=logger)
