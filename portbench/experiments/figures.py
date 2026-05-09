"""
Per-experiment figure rendering: NAV, metrics, profile NAV, stress drawdown.

Reads the artifacts written by paths.save_backtest_result and uses
portbench.visualization.sandbox_plots to produce PNGs in figures/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ..visualization.sandbox_plots import (
    plot_profile_nav,
    plot_sandbox_metrics,
    plot_sandbox_nav,
    plot_stress_drawdown,
)
from ..visualization.correlation_plots import (
    load_processed_correlation,
    plot_correlation_evolution,
    plot_correlation_heatmap,
    plot_inter_class_correlation,
)
from ..visualization.style import save_figure
from ..agent_eval.investor_profiles import PROFILES


def _load_normal_nav(p_dir: Path) -> Optional[pd.Series]:
    nav_csv = p_dir / "normal" / "nav_curve.csv"
    if not nav_csv.exists():
        return None
    df = pd.read_csv(nav_csv, parse_dates=["date"], index_col="date")
    return df["nav"]


def _load_normal_result(p_dir: Path) -> Optional[dict]:
    rj = p_dir / "normal" / "backtest_result.json"
    if not rj.exists():
        return None
    return json.loads(rj.read_text(encoding="utf-8"))


def _load_stress_results(p_dir: Path) -> dict[str, dict]:
    """Return {scenario_name: backtest_result_dict} for all stress_* dirs in p_dir."""
    out = {}
    for child in sorted(p_dir.iterdir()):
        if child.is_dir() and child.name.startswith("stress_"):
            rj = child / "backtest_result.json"
            if rj.exists():
                out[child.name[len("stress_") :]] = json.loads(
                    rj.read_text(encoding="utf-8")
                )
    return out


def render_experiment_figures(
    profile_dir: Path,
    model_label: str,
    profile_name: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Render figures for one (model, profile) experiment into profile_dir/figures/."""
    fig_dir = profile_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log = logger.info if logger else (lambda *a, **k: None)

    # 1. Normal NAV curve
    nav = _load_normal_nav(profile_dir)
    if nav is not None and len(nav) > 1:
        fig = plot_sandbox_nav(
            {f"{model_label}/{profile_name}": nav},
            title=f"Normal NAV — {model_label} / {profile_name}",
        )
        save_figure(fig, str(fig_dir / "nav.png"), formats=("png",))
        log("figure: nav.png")

    # 2. Metrics bar chart (normal only)
    normal = _load_normal_result(profile_dir)
    if normal:
        fig = plot_sandbox_metrics(
            {profile_name: normal},
            title=f"Normal Metrics — {model_label} / {profile_name}",
        )
        save_figure(fig, str(fig_dir / "metrics.png"), formats=("png",))
        log("figure: metrics.png")

    # 3. Stress drawdown heatmap (one column for this profile)
    stress = _load_stress_results(profile_dir)
    if stress:
        # plot_stress_drawdown expects {profile: {scenario: {max_drawdown,tolerance,stress_passed}}}

        tol = PROFILES[profile_name].max_drawdown_tolerance
        formatted = {
            profile_name: {
                sc: {
                    "max_drawdown": payload.get("max_drawdown", 0.0),
                    "tolerance": tol,
                    "stress_passed": payload.get("stress_passed", False),
                }
                for sc, payload in stress.items()
            }
        }
        fig = plot_stress_drawdown(
            formatted, title=f"Stress Drawdown — {model_label} / {profile_name}"
        )
        save_figure(fig, str(fig_dir / "stress_drawdown.png"), formats=("png",))
        log("figure: stress_drawdown.png")


    # 4. Cross-asset correlation evolution (one figure per phase that has snapshots)
    for phase_dir in [profile_dir / "normal"] + sorted(
        c for c in profile_dir.iterdir() if c.is_dir() and c.name.startswith("stress_")
    ):
        snap_dir = phase_dir / "snapshots"
        if not snap_dir.exists():
            continue
        _, amap = load_processed_correlation(Path("datasets/processed"))
        fig = plot_correlation_evolution(
            snap_dir,
            asset_class_map=amap,
            title=f"Correlation Evolution — {model_label} / {profile_name} / {phase_dir.name}",
        )
        if fig is not None:
            save_figure(
                fig,
                str(fig_dir / f"correlation_evolution_{phase_dir.name}.png"),
                formats=("png",),
            )
            log(f"figure: correlation_evolution_{phase_dir.name}.png")


def render_dataset_correlation_figures(
    output_dir: Path,
    processed_dir: Path = Path("datasets/processed"),
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Render the two dataset-level correlation figures (heatmap + inter/intra)
    from the frozen artifacts in datasets/processed/. Safe no-op when those
    files are missing.
    """
    log = logger.info if logger else (lambda *a, **k: None)
    corr, amap = load_processed_correlation(processed_dir)
    if corr is None:
        log("dataset correlation skipped: correlation_matrix.csv not found")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_correlation_heatmap(corr, amap)
    save_figure(fig, str(output_dir / "correlation_heatmap.png"), formats=("png",))
    log("figure: correlation_heatmap.png")

    if amap:
        fig = plot_inter_class_correlation(corr, amap)
        save_figure(fig, str(output_dir / "inter_class_correlation.png"), formats=("png",))
        log("figure: inter_class_correlation.png")


def render_model_summary_figures(
    model_dir: Path,
    model_label: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """After all profiles for a model finish, render a cross-profile NAV figure."""
    log = logger.info if logger else (lambda *a, **k: None)
    profile_navs: dict[str, pd.Series] = {}
    for child in sorted(model_dir.iterdir()):
        if not child.is_dir():
            continue
        nav = _load_normal_nav(child)
        if nav is not None and len(nav) > 1:
            profile_navs[child.name] = nav
    if len(profile_navs) >= 2:
        fig = plot_profile_nav(profile_navs, model_name=model_label)
        save_figure(fig, str(model_dir / "profile_nav.png"), formats=("png",))
        log("figure: profile_nav.png")
