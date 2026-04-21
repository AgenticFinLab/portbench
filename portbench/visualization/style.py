"""
Shared matplotlib style for PortBench publication figures.

Call apply_paper_style() at the start of any plotting script.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------------------------------------------------
# Color palette
# -----------------------------------------------------------------------

PAPER_COLORS = {
    "passed":  "#2ecc71",
    "failed":  "#e74c3c",
    "neutral": "#95a7b5",
    "s1":      "#3498db",
    "s2":      "#9b59b6",
    "s3":      "#e67e22",
    "s4":      "#1abc9c",
    "s5":      "#e74c3c",
}

STAGE_COLORS = [
    PAPER_COLORS["s1"],
    PAPER_COLORS["s2"],
    PAPER_COLORS["s3"],
    PAPER_COLORS["s4"],
    PAPER_COLORS["s5"],
]

REGIME_COLORS = {
    "bull":     "#2c7bb6",
    "bear":     "#e74c3c",
    "crisis":   "#e67e22",
    "sideways": "#95a7b5",
}

# Qualitative palette for up to ~8 models
MODEL_PALETTE = [
    "#2c7bb6", "#d7191c", "#fdae61", "#1a9641",
    "#abd9e9", "#a6d96a", "#762a83", "#f46d43",
]

STAGE_LABELS = ["S1\nInterpretation", "S2\nSignal Gen", "S3\nOptimization",
                "S4\nExecution", "S5\nRisk Monitor"]
STAGE_IDS    = ["S1", "S2", "S3", "S4", "S5"]


# -----------------------------------------------------------------------
# Style application
# -----------------------------------------------------------------------

def apply_paper_style(font_family: str = "serif", base_size: int = 10) -> None:
    """
    Apply publication-quality rcParams.

    Args:
        font_family: "serif" (Times-like, for IEEE/ACL) or "sans-serif" (Arial-like).
        base_size:   Base font size in pt.
    """
    mpl.rcParams.update({
        # Font
        "font.family":        font_family,
        "font.size":          base_size,
        "axes.titlesize":     base_size + 1,
        "axes.labelsize":     base_size,
        "xtick.labelsize":    base_size - 1,
        "ytick.labelsize":    base_size - 1,
        "legend.fontsize":    base_size - 1,

        # Figure
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,

        # Axes
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",

        # Lines
        "lines.linewidth":    1.5,

        # Legend
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "0.8",
    })


def save_figure(fig: plt.Figure, path: str, formats=("pdf", "png")) -> None:
    """Save figure in one or more formats."""
    import pathlib
    base = pathlib.Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(base.with_suffix(f".{fmt}"))
