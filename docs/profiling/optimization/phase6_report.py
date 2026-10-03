"""Regenerate the consolidated tables and figure from accepted comparison files.

Run from the repository root with:
uv run --group profiling python -m docs.profiling.optimization.phase6_report
The surrounding interpretation in RESULTS.md is maintained separately.
"""

import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

from tools.profiling.serde import Comparison, Run, Span

matplotlib.use("Agg")
ROOT = Path(__file__).parent
SOURCES = {
    "B1": "phase1-comparison.json",
    "B2": "phase2-vs-B1.json",
    "B3": "phase3-vs-B2.json",
    "B4a": "phase4a-vs-B3.json",
    "B4b": "phase4b-vs-B4a.json",
    "B5": "phase5-vs-B4b.json",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path.home() /
                        "Library/Application Support/InterDeCA/optimization")
    args = parser.parse_args()
    # Validate the evidence schema and the comparison chain before deriving any
    # table. A later phase must use the previously accepted values as reference.
    reports = {
        baseline: Comparison.model_validate_json((ROOT / filename).read_text())
        for baseline, filename in SOURCES.items()
    }
    values = {"B0": {
        backend: {metric: change.reference for metric, change in metrics.items()}
        for backend, metrics in reports["B1"].metrics.items()
    }}
    for baseline, report in reports.items():
        assert report.compatible_environment and report.identical_outputs and not report.issues
        previous = next(reversed(values))
        assert report.reference.name == previous
        values[baseline] = {}
        for backend, metrics in report.metrics.items():
            for metric, change in metrics.items():
                assert change.reference == values[previous][backend][metric]
            values[baseline][backend] = {
                metric: change.candidate for metric, change in metrics.items()
            }

    # Values and percentage changes remain separate so the report shows absolute
    # costs as well as reductions against B0 and the immediate predecessor.
    lines = ["# Accepted measurements", "",
             "Generated from the linked comparison records. GiB = 2³⁰ bytes; MiB = 2²⁰ bytes.", "",
             "| Backend | Baseline | Session median (s) | Application CPU median (s) | Application RSS maximum (GiB) | Tracked heap peak (GiB) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for backend in ("native", "morphoweave"):
        for baseline, measured in values.items():
            metrics = measured[backend]
            lines.append(f"| {backend} | {baseline} | {metrics['session_wall_s']:.3f} | "
                         f"{metrics['application_cpu_s']:.3f} | {metrics['peak_rss_bytes'] / 2**30:.3f} | "
                         f"{metrics['tracked_heap_peak_bytes'] / 2**30:.3f} |")
    lines += ["", "## Reductions against both references", "",
              "Each cell is **previous accepted baseline / B0**, in percent. Negative means an increase.", "",
              "| Backend | Candidate | Session time | Application CPU | Application RSS | Tracked heap | Evidence |",
              "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for backend in ("native", "morphoweave"):
        for baseline, report in reports.items():
            cells = []
            for metric in ("session_wall_s", "application_cpu_s", "peak_rss_bytes", "tracked_heap_peak_bytes"):
                previous = report.metrics[backend][metric].reduction_percent
                original = values["B0"][backend][metric]
                cumulative = 100 * (original - values[baseline][backend][metric]) / original
                cells.append(f"{previous:.2f}% / {cumulative:.2f}%")
            lines.append(f"| {backend} | {baseline} | " + " | ".join(cells) +
                         f" | [comparison]({SOURCES[baseline]}) |")
    lines += ["", "B4a's 10.85% MorphoWeave heap increase was investigated and accepted as a bounded cache tradeoff; see [the controls](phase4a-memory-investigation.json). Small timing/RSS fluctuations are not speedup claims.", "",
              "## Absolute reductions against both references", "",
              "Each cell is **previous accepted baseline / B0**. Time differences are seconds; memory differences are GiB. Negative means an increase; the JSON records retain unrounded bytes and seconds.", "",
              "| Backend | Candidate | Session time | Application CPU | Application RSS | Tracked heap |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for backend in ("native", "morphoweave"):
        for baseline, report in reports.items():
            cells = []
            for metric, scale in (("session_wall_s", 1), ("application_cpu_s", 1),
                                  ("peak_rss_bytes", 2**30), ("tracked_heap_peak_bytes", 2**30)):
                previous = report.metrics[backend][metric].reduction / scale
                cumulative = (values["B0"][backend][metric] - values[baseline][backend][metric]) / scale
                cells.append(f"{previous:.3f} / {cumulative:.3f}")
            lines.append(f"| {backend} | {baseline} | " + " | ".join(cells) + " |")
    lines += ["",
              "## Mechanism measurements", "",
              "OBJ traffic and array-copy volume come from separate timer runs. Retained arrays are the maximum in the manager's lookup, not total live memory.", "",
              "| Backend | Baseline | OBJ reads / writes | Logical OBJ read / write (GiB) | Array-copy volume (GiB) | Manager arrays (MiB) | Cache array peak / budget (MiB) |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for backend in ("native", "morphoweave"):
        for baseline, measured in values.items():
            m = measured[backend]
            cache = "—"
            if backend == "morphoweave" and baseline == "B4a":
                cache = "91.71 / 96"
            elif backend == "morphoweave" and baseline in ("B4b", "B5"):
                cache = "183.41 / 192"
            lines.append(f"| {backend} | {baseline} | {m['obj_read_calls']:.0f} / {m['obj_write_calls']:.0f} | "
                         f"{m['obj_read_bytes'] / 2**30:.3f} / {m['obj_write_bytes'] / 2**30:.3f} | "
                         f"{m['array_copy_bytes'] / 2**30:.3f} | {m['retained_array_bytes'] / 2**20:.2f} | {cache} |")
    lines += ["", "Cache values are validated in the accepted [B4a](../../../optimization.md#acceptance-history), [B4b](phase4b-budget-comparison.json), and [B5](phase5-ownership-review.json) records. Cache metadata, parsing transients, VTK copies, and caller-owned buffers are outside the cache's array budget.", ""]

    # Read raw baseline captures for ranges, child CPU, and OS high-water RSS.
    # Capture paths can precede archival renames; resolve each file inside its
    # accepted directory. macOS reports ru_maxrss in bytes, unlike Linux.
    lines += ["## Ranges and process accounting", "",
              "Ranges are observed minimum–maximum, not confidence intervals. Child CPU is separate from application CPU. Process-tree RSS sums pages and can double-count shared memory. OS high-water RSS below is application-only; unavailable USS is not treated as zero.", "",
              "| Backend | Baseline | Repeats | Session range (s) | Application CPU range (s) | Blender CPU median (s) | Tree RSS max (GiB) | OS application RSS max (GiB) |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for backend in ("native", "morphoweave"):
        for baseline in values:
            archive = args.archive_root / baseline
            environment = json.loads((archive / "environment.json").read_text())
            directories = sorted(archive.glob(f"{backend}-baseline-*"))
            runs = [Run.model_validate_json((directory / "run.json").read_text()) for directory in directories]
            assert len(runs) == (2 if baseline == "B0" else 3)
            assert all(run.exit_code == 0 for run in runs)
            sessions = [next(span for span in map(Span.model_validate_json,
                        (directory / "spans.jsonl").read_text().splitlines()) if span.kind == "session")
                        for directory in directories]
            wall = [span.wall_seconds for span in sessions]
            cpu = [span.process_cpu_seconds for span in sessions]
            high_water = [json.loads((directory / "rusage.json").read_text())["maxrss_platform_units"]
                          for directory in directories]
            factor = 1 if environment["platform"].startswith("macOS") else 1024
            m = values[baseline][backend]
            lines.append(f"| {backend} | {baseline} | {len(runs)} | {min(wall):.3f}–{max(wall):.3f} | "
                         f"{min(cpu):.3f}–{max(cpu):.3f} | {m['blender_cpu_s']:.3f} | "
                         f"{m['tree_peak_rss_bytes'] / 2**30:.3f} | {max(high_water) * factor / 2**30:.3f} |")
    lines.append("")
    (ROOT / "phase6-tables.md").write_text("\n".join(lines))

    # Plot accepted measurements without inventing confidence intervals. Baseline
    # repeat counts and ranges are documented in RESULTS.md and the source logs.
    phases = tuple(values)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")
    for ax, metric, scale, title, unit in zip(
        axes,
        ("session_wall_s", "peak_rss_bytes", "tracked_heap_peak_bytes"),
        (1, 2**30, 2**30),
        ("Median session time", "Maximum application RSS", "Tracked live heap peak"),
        ("Seconds", "GiB", "GiB"),
        strict=True,
    ):
        for backend, label, color in (("native", "Native Atlas", "#2563a6"),
                                      ("morphoweave", "MorphoWeave", "#ae4b20")):
            ax.plot(phases, [values[phase][backend][metric] / scale for phase in phases],
                    marker="o", color=color, linewidth=2, label=label)
        ax.set(title=title, ylabel=unit, xlabel="Accepted baseline", ylim=(0, None))
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Three small clams: accepted CPU/memory optimization results", fontsize=14)
    fig.savefig(ROOT / "phase6-trends.png", dpi=150)
    fig.savefig(ROOT / "phase6-trends.svg")
    plt.close(fig)
    # Matplotlib emits trailing spaces in SVG path data. Newlines already
    # separate those numbers, so normalize whitespace without changing geometry.
    svg = ROOT / "phase6-trends.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print("Wrote phase6-tables.md and phase6-trends.png/.svg from validated comparisons.")


if __name__ == "__main__":
    main()
