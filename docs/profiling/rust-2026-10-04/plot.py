"""Recreate the comparison chart from the adjacent measured baseline values."""

import json
import statistics
from pathlib import Path
from typing import NamedTuple

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")


class Metric(NamedTuple):
    key: str
    title: str
    unit: str
    use_maximum: bool


def main():
    root = Path(__file__).resolve().parent
    values = json.loads((root / "baseline-overview.json").read_text())
    figures = (
        Metric("wall_seconds", "Elapsed session time", "Seconds", False),
        Metric("process_cpu_seconds", "Application CPU time (excludes Blender)", "CPU seconds", False),
        Metric("rss_gib", "Peak resident memory", "GiB", True),
    )
    colors = {"python": "#87929C", "rust": "#177F86"}
    labels = {"python": "Python IO · previous capture", "rust": "Rust IO · new capture"}
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), layout="constrained")
    for ax, (metric, title, unit, use_maximum) in zip(axes, figures, strict=True):
        maximum = max(
            max(values[version][backend]["samples"][metric])
            for version in colors for backend in ("native", "morphoweave")
        )
        for row, backend in enumerate(("native", "morphoweave")):
            for offset, version in enumerate(colors):
                samples = values[version][backend]["samples"][metric]
                middle = max(samples) if use_maximum else statistics.median(samples)
                position = row * 1.15 + offset * 0.38
                # Whiskers show the observed range, not a confidence interval.
                # Resident-memory comparisons use the maximum of the three runs.
                ax.barh(
                    position, middle, height=0.30, color=colors[version],
                    label=labels[version] if row == 0 else None,
                    xerr=[[middle - min(samples)], [max(samples) - middle]],
                    error_kw={"ecolor": "#34414B", "capsize": 3, "linewidth": 1},
                )
                ax.text(
                    max(samples) + maximum * 0.018, position,
                    f"{middle:.2f}" if use_maximum else f"{middle:.1f}",
                    va="center", fontsize=10, color="#24333E",
                )
        ax.set(
            title=title, xlabel=unit, xlim=(0, maximum * 1.16),
            yticks=(0.19, 1.34), yticklabels=("Native Atlas", "MorphoWeave"),
        )
        ax.invert_yaxis()
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("47-clam pipeline: Python IO → Rust IO", fontsize=16)
    fig.supxlabel(
        "Three fresh-process baselines per backend. Whiskers show run ranges; "
        "time/CPU bars show medians, RSS bars show maxima.",
        fontsize=9,
    )
    fig.savefig(root / "performance.png", dpi=170)
    fig.savefig(root / "performance.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
