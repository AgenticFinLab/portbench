"""
Shared matplotlib style for PortBench publication figures.

Themes available:
  - "initial"   : original serif publication style
  - "minimalist": Modern Minimalist  (theme-factory skill)
  - "galaxy"    : Midnight Galaxy    (theme-factory skill)
  - "frost"     : Arctic Frost       (theme-factory skill)

Call apply_style(theme) or the legacy apply_paper_style() at the start of any plotting script.

Color rule: each skill theme uses exactly the 4 hex values from its .md file.
All palette entries are either those 4 values or direct interpolations between them.
No external hex codes are introduced.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------------------------------------------------
# Theme-agnostic constants
# -----------------------------------------------------------------------

STAGE_LABELS = [
    "S1\nInterpretation",
    "S2\nSignal Gen",
    "S3\nOptimization",
    "S4\nExecution",
    "S5\nRisk Monitor",
]
STAGE_IDS = ["S1", "S2", "S3", "S4", "S5"]

# -----------------------------------------------------------------------
# Initial theme (original paper style, unchanged)
# -----------------------------------------------------------------------

_INITIAL_COLORS = {
    "passed": "#2ecc71",
    "failed": "#e74c3c",
    "neutral": "#95a7b5",
    "s1": "#3498db",
    "s2": "#9b59b6",
    "s3": "#e67e22",
    "s4": "#1abc9c",
    "s5": "#e74c3c",
}
_INITIAL_STAGE_COLORS = ["#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]
_INITIAL_REGIME_COLORS = {
    "bull": "#2c7bb6",
    "bear": "#e74c3c",
    "crisis": "#e67e22",
    "sideways": "#95a7b5",
}
_INITIAL_MODEL_PALETTE = [
    "#2c7bb6",
    "#d7191c",
    "#fdae61",
    "#1a9641",
    "#abd9e9",
    "#a6d96a",
    "#762a83",
    "#f46d43",
]

# -----------------------------------------------------------------------
# Modern Minimalist  (theme-factory skill)
# 4 source colors:
#   Charcoal    #36454f  (darkest)
#   Slate Gray  #708090
#   Light Gray  #d3d3d3
#   White       #ffffff  (lightest)
# Interpolated midpoints are computed as simple hex blends of the above.
# -----------------------------------------------------------------------

_MM1 = "#36454f"  # charcoal
_MM2 = "#708090"  # slate gray
_MM3 = "#d3d3d3"  # light gray
_MM4 = "#ffffff"  # white

MINIMALIST_COLORS = {
    "passed": _MM1,
    "failed": _MM2,
    "neutral": _MM3,
    "s1": _MM1,
    "s2": _MM2,
    "s3": "#546878",
    "s4": "#a8b8c0",
    "s5": _MM3,
}
MINIMALIST_STAGE_COLORS = [_MM1, "#546878", _MM2, "#a8b8c0", _MM3]
MINIMALIST_REGIME_COLORS = {
    "bull": _MM1,
    "bear": _MM3,
    "crisis": _MM2,
    "sideways": "#b8c4ca",
}
# palette: alternate darkest↔lightest to maximise series contrast
MINIMALIST_MODEL_PALETTE = [
    _MM1,  # charcoal        (darkest)
    _MM3,  # light gray      (big jump)
    _MM2,  # slate gray
    _MM4,  # white
    "#546878",  # charcoal→slate midpoint
    "#a8b8c0",  # slate→light midpoint
    "#1e2d35",  # charcoal darkened
    "#b8c4ca",  # slate→light ¾
]

# -----------------------------------------------------------------------
# Midnight Galaxy  (theme-factory skill)
# 4 source colors:
#   Deep Purple  #2b1e3e  (darkest)
#   Cosmic Blue  #4a4e8f
#   Lavender     #a490c2
#   Silver       #e6e6fa  (lightest)
# -----------------------------------------------------------------------

_MG1 = "#2b1e3e"  # deep purple
_MG2 = "#4a4e8f"  # cosmic blue
_MG3 = "#a490c2"  # lavender
_MG4 = "#e6e6fa"  # silver

GALAXY_COLORS = {
    "passed": _MG2,
    "failed": _MG3,
    "neutral": _MG4,
    "s1": _MG1,
    "s2": _MG2,
    "s3": "#6d6aaa",
    "s4": _MG3,
    "s5": _MG4,
}
GALAXY_STAGE_COLORS = [_MG1, _MG2, "#6d6aaa", _MG3, _MG4]
GALAXY_REGIME_COLORS = {
    "bull": _MG2,
    "bear": _MG1,
    "crisis": _MG3,
    "sideways": _MG4,
}
GALAXY_MODEL_PALETTE = [
    _MG1,  # deep purple     (darkest)
    _MG4,  # silver          (big jump)
    _MG2,  # cosmic blue
    _MG3,  # lavender
    "#1a1228",  # deep purple darkened
    "#6d6aaa",  # MG1→MG2 midpoint
    "#c8c0dc",  # MG3→MG4 midpoint
    "#3a3070",  # MG1→MG2 ¾
]

# -----------------------------------------------------------------------
# Arctic Frost  (theme-factory skill)
# 4 source colors:
#   Steel Blue   #4a6fa5  (primary saturated)
#   Ice Blue     #d4e4f7
#   Silver       #c0c0c0
#   Crisp White  #fafafa  (lightest)
# -----------------------------------------------------------------------

_AF1 = "#4a6fa5"  # steel blue
_AF2 = "#d4e4f7"  # ice blue
_AF3 = "#c0c0c0"  # silver
_AF4 = "#fafafa"  # crisp white

FROST_COLORS = {
    "passed": _AF1,
    "failed": _AF3,
    "neutral": _AF2,
    "s1": "#1e3d6e",
    "s2": _AF1,
    "s3": "#7a9fc5",
    "s4": _AF2,
    "s5": _AF3,
}
FROST_STAGE_COLORS = ["#1e3d6e", _AF1, "#7a9fc5", _AF2, _AF3]
FROST_REGIME_COLORS = {
    "bull": _AF1,
    "bear": "#1e3d6e",
    "crisis": "#7a9fc5",
    "sideways": _AF3,
}
FROST_MODEL_PALETTE = [
    "#1e3d6e",  # steel blue darkened  (darkest)
    _AF2,  # ice blue             (big jump)
    _AF1,  # steel blue
    _AF3,  # silver
    "#0d2040",  # steel blue very dark
    "#7a9fc5",  # steel→ice midpoint
    "#e8f2fc",  # ice→white midpoint
    "#8a8a8a",  # silver darkened
]

# -----------------------------------------------------------------------
# rcParams per theme
# -----------------------------------------------------------------------

_THEME_RCPARAMS = {
    "initial": {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.5,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "0.8",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    },
    "minimalist": {
        # Modern Minimalist — DejaVu Sans, white, charcoal/slate palette
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": _MM3,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.color": _MM3,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
        "legend.framealpha": 0.0,
        "legend.edgecolor": "none",
        "legend.borderpad": 0.6,
        "figure.facecolor": _MM4,
        "axes.facecolor": _MM4,
        "text.color": _MM1,
        "axes.labelcolor": _MM2,
        "xtick.color": _MM2,
        "ytick.color": _MM2,
        "axes.titlecolor": _MM1,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
    },
    "galaxy": {
        # Midnight Galaxy — sans-serif, white, deep purple/cosmic blue palette
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": _MG3,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.color": _MG4,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
        "legend.framealpha": 0.85,
        "legend.edgecolor": _MG3,
        "legend.facecolor": "#ffffff",
        "legend.borderpad": 0.6,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "text.color": _MG1,
        "axes.labelcolor": _MG2,
        "xtick.color": _MG2,
        "ytick.color": _MG2,
        "axes.titlecolor": _MG1,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
    },
    "frost": {
        # Arctic Frost — DejaVu Sans, crisp white, steel blue palette
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": _AF2,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.color": _AF2,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
        "legend.framealpha": 0.85,
        "legend.edgecolor": _AF2,
        "legend.facecolor": _AF4,
        "legend.borderpad": 0.6,
        "figure.facecolor": _AF4,
        "axes.facecolor": _AF4,
        "text.color": "#1e3d6e",
        "axes.labelcolor": _AF1,
        "xtick.color": _AF1,
        "ytick.color": _AF1,
        "axes.titlecolor": "#1e3d6e",
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
    },
}

_THEME_PALETTES = {
    "initial": (
        _INITIAL_COLORS,
        _INITIAL_STAGE_COLORS,
        _INITIAL_REGIME_COLORS,
        _INITIAL_MODEL_PALETTE,
    ),
    "minimalist": (
        MINIMALIST_COLORS,
        MINIMALIST_STAGE_COLORS,
        MINIMALIST_REGIME_COLORS,
        MINIMALIST_MODEL_PALETTE,
    ),
    "galaxy": (
        GALAXY_COLORS,
        GALAXY_STAGE_COLORS,
        GALAXY_REGIME_COLORS,
        GALAXY_MODEL_PALETTE,
    ),
    "frost": (
        FROST_COLORS,
        FROST_STAGE_COLORS,
        FROST_REGIME_COLORS,
        FROST_MODEL_PALETTE,
    ),
}

_active_theme: str = "initial"


def apply_style(theme: str = "galaxy") -> tuple:
    """
    Apply a named theme and return (colors, stage_colors, regime_colors, model_palette).

    Args:
        theme: "initial" | "minimalist" | "galaxy" | "frost"
    """
    global _active_theme
    if theme not in _THEME_RCPARAMS:
        raise ValueError(
            f"Unknown theme {theme!r}. Choose from: {list(_THEME_RCPARAMS)}"
        )
    _active_theme = theme
    mpl.rcParams.update(_THEME_RCPARAMS[theme])
    return _THEME_PALETTES[theme]


def get_active_palettes() -> tuple:
    """Return (colors, stage_colors, regime_colors, model_palette) for the current theme."""
    return _THEME_PALETTES[_active_theme]


def apply_paper_style(font_family: str = "serif", base_size: int = 10) -> None:
    """Legacy entry point — applies the Arctic Frost theme."""
    apply_style("frost")


# ---------------------------------------------------------------------------
# Legacy name aliases — all existing modules import these names directly.
# Pointing them at the Frost palette requires zero changes in callers.
# ---------------------------------------------------------------------------

PAPER_COLORS = FROST_COLORS
STAGE_COLORS = FROST_STAGE_COLORS
REGIME_COLORS = FROST_REGIME_COLORS
MODEL_PALETTE = FROST_MODEL_PALETTE


# ---------------------------------------------------------------------------
# Multi-series line style / marker cycling
# ---------------------------------------------------------------------------

LINE_STYLES = ["-", "--", "-.", ":"]
LINE_MARKERS = ["o", "s", "^", "D", "v", "P", "h", "*"]

# 20-color categorical palette (tab20) for figures with many series
CATEGORICAL_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#1a9850",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#9edae5", "#393b79", "#637939",
]

# Extended palette for NAV LLM lines (vivid, distinguishable)
NAV_LLM_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#1a9850",
]
# Gray shades for NAV baseline dashed lines
NAV_BASELINE_PALETTE = ["#555555", "#888888", "#aaaaaa", "#cccccc"]


# ---------------------------------------------------------------------------
# Model name abbreviation
# ---------------------------------------------------------------------------

_MODEL_ABBREVS = {
    "hy3-preview":          "HY3",
    "deepseek-v4-flash":    "DS-Flash",
    "deepseek-v4-pro":      "DS-Pro",
    "kimi-k2.6":            "Kimi-K2",
    "doubao-seed-2-0-pro":  "Doubao-Pro",
    "doubao-seed-2-0-lite": "Doubao-Lite",
    "minimax-m2.7":         "MiniMax",
    "equal_weight":         "EqWt",
    "risk_parity":          "RiskPar",
    "sixty_forty":          "60/40",
    "cov_risk_parity":      "CovRiskPar",
    "min_variance":         "MinVar",
}


def abbrev_model_name(model_key: str) -> str:
    """Shorten 'provider/model-name-YYYYMMDD' to a compact display label."""
    import re
    name = model_key.split("/")[-1] if "/" in model_key else model_key
    name = re.sub(r"-\d{6,8}$", "", name)
    return _MODEL_ABBREVS.get(name, name)


def save_figure(fig: plt.Figure, path: str, formats=("pdf", "png")) -> None:
    """Save figure in one or more formats."""
    import pathlib

    base = pathlib.Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(base.with_suffix(f".{fmt}"))
    plt.close(fig)
