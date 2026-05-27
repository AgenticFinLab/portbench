"""PortBench visualization module."""

from .ceps_plots import plot_ceps_radar, plot_ceps_heatmap, plot_ceps_violin
from .stress_plots import plot_stress_gate
from .ranking_plots import plot_risk_ranking
from .dataset_plots import (
    plot_dataset_overview,
    plot_regime_heatmap,
    plot_per_asset_class_overview,
    plot_all_asset_class_overviews,
)
from .qa_sample_plots import plot_qa_sample_cards, plot_single_card
from .regime_plots import plot_regime_distributions, build_regime_data_from_mock
from .sandbox_plots import (
    plot_sandbox_nav,
    plot_sandbox_metrics,
    plot_ceps_vs_pnl,
    plot_stress_drawdown,
    plot_profile_nav,
    load_sandbox_results,
    load_sandbox_results_full,
)
from .profile_plots import plot_profile_alignment, plot_profile_radar
from .correlation_plots import (
    plot_correlation_heatmap,
    plot_inter_class_correlation,
    plot_correlation_evolution,
    load_processed_correlation,
)
from .correlation_graph import (
    plot_correlation_mst,
    plot_correlation_threshold,
)
from .style import apply_paper_style, save_figure

__all__ = [
    "plot_ceps_radar",
    "plot_ceps_heatmap",
    "plot_ceps_violin",
    "plot_stress_gate",
    "plot_risk_ranking",
    "plot_dataset_overview",
    "plot_regime_heatmap",
    "plot_per_asset_class_overview",
    "plot_all_asset_class_overviews",
    "plot_qa_sample_cards",
    "plot_single_card",
    "plot_regime_distributions",
    "build_regime_data_from_mock",
    # Sandbox
    "plot_sandbox_nav",
    "plot_sandbox_metrics",
    "plot_ceps_vs_pnl",
    "plot_stress_drawdown",
    "plot_profile_nav",
    "load_sandbox_results",
    "load_sandbox_results_full",
    # Investor profiles
    "plot_profile_alignment",
    "plot_profile_radar",
    # Cross-asset correlation
    "plot_correlation_heatmap",
    "plot_inter_class_correlation",
    "plot_correlation_evolution",
    "load_processed_correlation",
    "plot_correlation_mst",
    "plot_correlation_threshold",
    # Shared
    "apply_paper_style",
    "save_figure",
]
