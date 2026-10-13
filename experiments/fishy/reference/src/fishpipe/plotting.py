"""Morphospace plotting: scatter of an embedding coloured by each ground-truth factor."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import GroundTruth

_FACTOR_TITLES = {
    "belly": "Belly hue cluster (2)",
    "tail": "Tail hue cluster (2)",
    "stripe": "Stripe count (4 vs 5)",
    "cheeks": "Rosy cheeks (5% minority)",
}


def morphospace_grid(
    scores: np.ndarray,
    gt: GroundTruth,
    title: str,
    out_path: str | Path,
    dims: tuple[int, int] = (0, 1),
    axis_label: str = "PC",
    factors=("belly", "tail", "stripe", "cheeks"),
):
    """One panel per factor; points are specimens colored by that factor's label.

    Minority points (cheeks==1) are drawn larger and on top so they don't hide.
    """
    x, y = scores[:, dims[0]], scores[:, dims[1]]
    n = len(factors)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0))
    if n == 1:
        axes = [axes]
    for ax, f in zip(axes, factors):
        lab = gt.labels[f]
        for cls in np.unique(lab):
            m = lab == cls
            minority = (f == "cheeks" and cls == 1)
            ax.scatter(
                x[m], y[m],
                s=70 if minority else 22,
                alpha=0.95 if minority else 0.6,
                edgecolors="k" if minority else "none",
                linewidths=0.6 if minority else 0,
                zorder=3 if minority else 2,
                label=f"{cls}" + (" (present)" if minority else ""),
            )
        ax.set_title(_FACTOR_TITLES.get(f, f), fontsize=10)
        ax.set_xlabel(f"{axis_label}{dims[0] + 1}")
        ax.set_ylabel(f"{axis_label}{dims[1] + 1}")
        ax.legend(fontsize=7, markerscale=0.9, loc="best", framealpha=0.7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def pairwise_pc_grid(
    scores: np.ndarray, gt: GroundTruth, factor: str, title: str, out_path: str | Path,
    max_pcs: int = 4, axis_label: str = "PC",
):
    """Lower-triangular grid of PC_i vs PC_j, colored by a single factor.

    Useful to see which *pair* of components separates a given factor.
    """
    k = min(max_pcs, scores.shape[1])
    lab = gt.labels[factor]
    fig, axes = plt.subplots(k, k, figsize=(2.6 * k, 2.6 * k))
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if i == j:
                ax.text(0.5, 0.5, f"{axis_label}{i + 1}", ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            for cls in np.unique(lab):
                m = lab == cls
                minority = (factor == "cheeks" and cls == 1)
                ax.scatter(scores[m, j], scores[m, i],
                           s=44 if minority else 10,
                           alpha=0.9 if minority else 0.5,
                           edgecolors="k" if minority else "none",
                           linewidths=0.5 if minority else 0,
                           zorder=3 if minority else 2)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
