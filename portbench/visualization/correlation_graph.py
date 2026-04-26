"""
Cross-asset correlation as a graph.

Where the matrix views (`correlation_plots.py`) prove "we computed correlation,"
the graph views surface the *structure* of those relations: which assets cluster
inside an asset class, which assets bridge across classes, and which form
isolated peripheral components. This is the multi-asset narrative — heterogeneity
as topology rather than as cell color.

Two figures:

  plot_correlation_mst        — Mantegna-style minimum spanning tree.
                                Edge weight d_ij = sqrt(2 (1 - rho_ij)) is the
                                ultrametric distance from Pearson correlation;
                                the MST is the sparsest skeleton that connects
                                every asset, exposing the true backbone of
                                cross-asset relations.

  plot_correlation_threshold  — threshold-filtered force-directed graph.
                                Keep only edges with |rho| >= threshold; node
                                color = asset class, node size = degree (within
                                the filtered graph), edge color split into
                                positive (red) / negative (blue) correlations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import networkx as nx

from .style import apply_paper_style


_CLASS_PALETTE = [
    "#1e3d6e", "#4a6fa5", "#e67e22", "#1abc9c",
    "#9b59b6", "#e74c3c", "#7a9fc5", "#2ecc71",
]


def _class_colors(asset_class_map: Optional[dict[str, str]]) -> dict[str, str]:
    if not asset_class_map:
        return {}
    classes = sorted(set(asset_class_map.values()))
    return {c: _CLASS_PALETTE[i % len(_CLASS_PALETTE)] for i, c in enumerate(classes)}


def _build_full_graph(corr: pd.DataFrame) -> nx.Graph:
    """Complete weighted graph; edge weight = MST distance sqrt(2(1-rho))."""
    g = nx.Graph()
    assets = list(corr.columns)
    g.add_nodes_from(assets)
    for i, a in enumerate(assets):
        for j in range(i + 1, len(assets)):
            b = assets[j]
            r = float(corr.loc[a, b])
            if np.isnan(r):
                continue
            r = max(min(r, 1.0), -1.0)
            dist = float(np.sqrt(2.0 * (1.0 - r)))
            g.add_edge(a, b, weight=dist, corr=r)
    return g


def plot_correlation_mst(
    correlation_matrix: pd.DataFrame,
    asset_class_map: Optional[dict[str, str]] = None,
    *,
    title: str = "Cross-Asset Correlation MST",
    figsize: tuple[float, float] = (10.0, 8.0),
    seed: int = 0,
) -> Figure:
    """
    Mantegna minimum spanning tree of the asset correlation graph.

    The MST is the unique tree that connects all N assets with N-1 edges of
    minimum total ultrametric distance. It exposes the *backbone* of the
    correlation structure: each asset attaches to its single nearest neighbor
    in correlation space, so cross-class bridges become visually obvious.
    """
    apply_paper_style()

    if correlation_matrix.empty or len(correlation_matrix) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Not enough assets for MST", ha="center", va="center")
        ax.axis("off")
        return fig

    full = _build_full_graph(correlation_matrix)
    mst = nx.minimum_spanning_tree(full, weight="weight")

    color_map = _class_colors(asset_class_map)
    node_colors = [
        color_map.get(asset_class_map.get(n, ""), "#bbbbbb") if asset_class_map else "#4a6fa5"
        for n in mst.nodes
    ]
    degrees = dict(mst.degree())
    node_sizes = [220 + 70 * degrees[n] for n in mst.nodes]

    pos = nx.spring_layout(mst, weight="weight", seed=seed, k=1.2 / np.sqrt(len(mst)))

    fig, ax = plt.subplots(figsize=figsize)
    edge_corrs = [mst[u][v]["corr"] for u, v in mst.edges]
    edge_colors = ["#c0392b" if r >= 0 else "#2980b9" for r in edge_corrs]
    edge_widths = [0.6 + 1.4 * abs(r) for r in edge_corrs]

    nx.draw_networkx_edges(
        mst, pos, ax=ax,
        edge_color=edge_colors, width=edge_widths, alpha=0.75,
    )
    nx.draw_networkx_nodes(
        mst, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        edgecolors="#1e3d6e", linewidths=0.8,
    )
    nx.draw_networkx_labels(mst, pos, ax=ax, font_size=7)

    if color_map:
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=v,
                   markeredgecolor="#1e3d6e", markersize=9, label=c)
            for c, v in color_map.items()
        ]
        legend_handles += [
            Line2D([0], [0], color="#c0392b", lw=2.0, label="positive corr"),
            Line2D([0], [0], color="#2980b9", lw=2.0, label="negative corr"),
        ]
        ax.legend(
            handles=legend_handles, fontsize=8, loc="upper left",
            frameon=True, framealpha=0.9, ncol=1,
        )

    ax.set_title(
        f"{title}  ·  N={mst.number_of_nodes()} assets, {mst.number_of_edges()} MST edges",
        fontsize=11, fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_correlation_threshold(
    correlation_matrix: pd.DataFrame,
    asset_class_map: Optional[dict[str, str]] = None,
    *,
    threshold: float = 0.5,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10.0, 8.0),
    seed: int = 0,
) -> Figure:
    """
    Threshold-filtered correlation network.

    Keeps only edges with |rho| >= threshold so the strong-association clusters
    pop visually. Node color encodes asset class; node size encodes degree
    inside the filtered graph (i.e. how many strong correlation partners each
    asset has). Singleton nodes (no qualifying edge) are still drawn so the
    user can see *which* assets fall outside the strong-association regime.
    """
    apply_paper_style()

    if correlation_matrix.empty or len(correlation_matrix) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Not enough assets", ha="center", va="center")
        ax.axis("off")
        return fig

    g = nx.Graph()
    assets = list(correlation_matrix.columns)
    g.add_nodes_from(assets)
    for i, a in enumerate(assets):
        for j in range(i + 1, len(assets)):
            b = assets[j]
            r = float(correlation_matrix.loc[a, b])
            if np.isnan(r):
                continue
            if abs(r) >= threshold:
                g.add_edge(a, b, corr=r, weight=abs(r))

    color_map = _class_colors(asset_class_map)
    node_colors = [
        color_map.get(asset_class_map.get(n, ""), "#bbbbbb") if asset_class_map else "#4a6fa5"
        for n in g.nodes
    ]
    degrees = dict(g.degree())
    node_sizes = [180 + 60 * degrees[n] for n in g.nodes]

    pos = nx.spring_layout(g, weight="weight", seed=seed, k=1.4 / np.sqrt(max(len(g), 2)))

    fig, ax = plt.subplots(figsize=figsize)
    edge_corrs = [g[u][v]["corr"] for u, v in g.edges]
    edge_colors = ["#c0392b" if r >= 0 else "#2980b9" for r in edge_corrs]
    edge_widths = [0.4 + 2.4 * (abs(r) - threshold) / max(1.0 - threshold, 1e-6) for r in edge_corrs]

    if g.number_of_edges() > 0:
        nx.draw_networkx_edges(
            g, pos, ax=ax,
            edge_color=edge_colors, width=edge_widths, alpha=0.65,
        )
    nx.draw_networkx_nodes(
        g, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        edgecolors="#1e3d6e", linewidths=0.8,
    )
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=7)

    n_isolated = sum(1 for n, d in degrees.items() if d == 0)
    sub_title = (
        f"|ρ| ≥ {threshold:.2f}  ·  {g.number_of_edges()} edges, "
        f"{n_isolated} isolated of {g.number_of_nodes()} assets"
    )
    ax.set_title(
        f"{title or 'Cross-Asset Correlation Network'}\n{sub_title}",
        fontsize=11, fontweight="bold",
    )
    ax.axis("off")

    if color_map:
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=v,
                   markeredgecolor="#1e3d6e", markersize=9, label=c)
            for c, v in color_map.items()
        ]
        legend_handles += [
            Line2D([0], [0], color="#c0392b", lw=2.0, label="positive corr"),
            Line2D([0], [0], color="#2980b9", lw=2.0, label="negative corr"),
        ]
        ax.legend(
            handles=legend_handles, fontsize=8, loc="upper left",
            frameon=True, framealpha=0.9, ncol=1,
        )

    fig.tight_layout()
    return fig
