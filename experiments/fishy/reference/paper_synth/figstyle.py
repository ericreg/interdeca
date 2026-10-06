"""Publication figure style for the ColorAtlas synthetic-validation figures.

One visual language for every panel: a color-blind-safe categorical palette (Okabe-Ito),
one semantic color per planted factor, hairline axes, direct labelling instead of legends
where possible, and vector-quality text at 300+ dpi.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch  # noqa: F401  (re-exported for panels)

# --------------------------------------------------------------------------- palette
OKABE = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
PANEL_BG = "#f4f7fb"          # matches the light-blue panel boxes used elsewhere in the paper

# one semantic color per planted factor (used consistently in every figure)
FACTOR_COLOR = {
    "belly": OKABE["blue"],
    "tail": OKABE["green"],
    "stripe": OKABE["vermillion"],
    "cheeks": OKABE["purple"],
}
FACTOR_NAME = {
    "belly": "belly hue",
    "tail": "tail hue",
    "stripe": "stripe count",
    "cheeks": "cheek patch",
}
# pre/post-module contrast used in the preservation figures
PRE_COLOR = "#9aa5b1"
POST_COLOR = OKABE["blue"]


def use_style(base: float = 8.0):
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#5a5a5a",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titlepad": 5,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.frameon": False,
        "legend.fontsize": base - 1,
        "legend.handlelength": 1.2,
        "lines.linewidth": 1.4,
        "lines.solid_capstyle": "round",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, letter: str, dx: float = -0.085, dy: float = 1.06, size: float = 11):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size, fontweight="bold",
            va="top", ha="left", color=INK)


def despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def hairline_grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)


def chance_line(ax, y=0.5, label="chance", color=MUTED):
    ax.axhline(y, color=color, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.text(0.995, y, f" {label}", transform=ax.get_yaxis_transform(), va="bottom",
            ha="right", fontsize=6.5, color=color)


def place_image(ax, img, xy, zoom=1.0, frame=None, frame_lw=1.1, zorder=5):
    """Place an RGB(A) thumbnail at DATA coordinates xy on ax."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    oi = OffsetImage(img, zoom=zoom, interpolation="antialiased")
    kw = dict(frameon=frame is not None, pad=0.12, xycoords="data", zorder=zorder)
    ab = AnnotationBbox(oi, xy, **kw)
    if frame is not None:
        ab.patch.set_edgecolor(frame)
        ab.patch.set_linewidth(frame_lw)
        ab.patch.set_facecolor("white")
        ab.patch.set_boxstyle("round,pad=0.12,rounding_size=0.06")
    ax.add_artist(ab)
    return ab


def farthest_point_sample(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Indices of k well-spread points (for exemplar placement without overlap)."""
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(X)))]
    d = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
    return np.array(idx)


def savefig(fig, path, also_pdf: bool = True):
    from pathlib import Path
    path = Path(path)
    fig.savefig(path, dpi=400)
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path
