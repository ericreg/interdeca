"""Render measured CPU, allocation, and resident-memory evidence without rerunning work."""

import argparse
import html
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import memray
from matplotlib import pyplot as plt

from tools.profiling.serde import (
    AllocationSite,
    FunctionCost,
    Run,
    Sample,
    Span,
    Summary,
    TaskSummary,
)

matplotlib.use("Agg")


def read_lines(path, schema):
    return [schema.model_validate_json(line) for line in path.read_text().splitlines()]


def task_summaries(spans, samples):
    tasks = []
    for span in spans:
        if span.kind != "task":
            continue
        during = [s for s in samples if span.start <= s.time <= span.end]
        parent = [p.rss_bytes for s in during for p in s.processes if p.parent]
        # Endpoint samples cover tasks shorter than the external sampling interval.
        parent += [v for v in (span.rss_start, span.rss_end) if v is not None]
        trees = [sum(p.rss_bytes for p in s.processes) for s in during]
        tasks.append(
            TaskSummary(
                pipeline=span.pipeline,
                task=span.name,
                wall_seconds=span.wall_seconds,
                parent_cpu_seconds=span.process_cpu_seconds,
                children_cpu_seconds=span.children_cpu_seconds,
                parent_peak_rss_bytes=max(parent, default=0),
                tree_peak_rss_bytes=max([*trees, *parent], default=0),
                retained_array_bytes=span.retained_array_bytes or 0,
            )
        )
    return tasks


def function_costs(spans):
    costs = defaultdict(FunctionCost)
    for span in spans:
        if span.kind != "function":
            continue
        cost = costs[span.name]
        cost.calls += 1
        cost.inclusive_wall_seconds += span.wall_seconds
        cost.exclusive_wall_seconds += span.exclusive_wall_seconds
        cost.exclusive_thread_cpu_seconds += span.exclusive_thread_cpu_seconds
        cost.array_copy_bytes += span.array_copy_bytes or 0
        if span.file_direction == "read":
            cost.logical_read_bytes += span.file_bytes or 0
        if span.file_direction == "write":
            cost.logical_write_bytes += span.file_bytes or 0
    return dict(costs)


def allocation_sites(path):
    # Aggregated captures preserve the allocations simultaneously live at peak,
    # rather than summing allocations from different stages or all allocation churn.
    sites = []
    with memray.FileReader(path) as reader:
        for record in reader.get_high_watermark_allocation_records():
            frames = [
                f"{name} ({file}:{line})" for name, file, line in record.stack_trace()
            ]
            sites.append(
                AllocationSite(
                    bytes=record.size,
                    allocations=record.n_allocations,
                    python_stack=frames,
                )
            )
    return sorted(sites, key=lambda value: value.bytes, reverse=True)


def memory_plot(root, runs):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharey=True)
    fig.subplots_adjust(top=0.90, bottom=0.075, left=0.085, right=0.99, hspace=0.40)
    maximum = 0
    for ax, backend in zip(axes, ("native", "morphoweave"), strict=True):
        run = next(
            r
            for r in runs
            if r.config.backend == backend and r.config.mode == "baseline"
        )
        samples = read_lines(run.config.output / "resources.jsonl", Sample)
        spans = read_lines(run.config.output / "spans.jsonl", Span)
        session = next(s for s in spans if s.kind == "session")
        samples = [s for s in samples if session.start <= s.time <= session.end]
        times = [s.time - session.start for s in samples]
        parent = [
            sum(p.rss_bytes for p in s.processes if p.parent) / 2**20 for s in samples
        ]
        tree = [sum(p.rss_bytes for p in s.processes) / 2**20 for s in samples]
        ax.plot(times, tree, label="Application + Blender RSS", color="#b76520", lw=1.3)
        ax.plot(times, parent, label="Application RSS", color="#226ca8", lw=1.3)
        maximum = max(maximum, max(tree, default=0))
        # A pipeline's result lookup is released on return. Reset its measured
        # ownership instead of drawing the previous run's buffers through the
        # next pipeline's first operation; application-owned data is separate.
        retained = [
            (s.end - session.start, (s.retained_array_bytes or 0) / 2**20)
            for s in spans
            if s.kind == "task"
        ]
        retained += [
            (s.start - session.start, 0) for s in spans if s.kind == "pipeline"
        ]
        retained += [(s.end - session.start, 0) for s in spans if s.kind == "pipeline"]
        retained.sort()
        ax.step(
            [t for t, _ in retained],
            [v for _, v in retained],
            where="post",
            label="Manager-retained NumPy buffers",
            color="#3b8748",
            lw=1.2,
        )
        for s in spans:
            if s.kind == "pipeline" and s.name in (
                "ATLAS and Texture Transfer",
                "Color processing",
            ):
                ax.axvline(
                    s.end - session.start,
                    color="#999999",
                    linestyle="--",
                    linewidth=0.7,
                )
        ax.set(
            title=f"{backend}: first baseline run",
            xlabel="Seconds since CLI session started",
            ylabel="MiB",
        )
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
    axes[0].set_ylim(0, maximum * 1.06)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=9
    )
    fig.savefig(root / "memory.png", dpi=150)
    fig.savefig(root / "memory.svg")
    plt.close(fig)


def table(headers, rows):
    heading = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    runs = [
        Run.model_validate_json(path.read_text())
        for path in sorted(root.glob("*/run.json"))
    ]
    if not runs or any(run.exit_code for run in runs):
        raise RuntimeError("Complete successful runs are required before reporting.")
    if any(
        not any(
            r.config.backend == backend and r.config.mode == "baseline" for r in runs
        )
        for backend in ("native", "morphoweave")
    ):
        raise RuntimeError(
            "Record a baseline for each backend before generating the comparison report."
        )
    summary = Summary(
        runs=runs, tasks={}, functions={}, peak_allocations={}, tracked_heap_peaks={}
    )
    for run in runs:
        directory = run.config.output
        spans = read_lines(directory / "spans.jsonl", Span)
        samples = read_lines(directory / "resources.jsonl", Sample)
        summary.tasks[directory.name] = task_summaries(spans, samples)
        summary.functions[directory.name] = function_costs(spans)
        if run.config.mode == "memory":
            sites = allocation_sites(directory / "allocations.bin")
            summary.tracked_heap_peaks[directory.name] = sum(s.bytes for s in sites)
            summary.peak_allocations[directory.name] = sites[:40]
            # Native flame graphs preserve allocation call chains for later review.
            destination = directory / "allocations.html"
            if not destination.exists():
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "memray",
                        "flamegraph",
                        "--output",
                        str(destination),
                        str(directory / "allocations.bin"),
                    ],
                    check=True,
                )
    (root / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
    memory_plot(root, runs)
    environment = json.loads((root / "environment.json").read_text())
    count = len(environment.get("specimens", [])) or 3
    sections = [
        f"<h1>{count}-clam CPU and memory profile</h1>",
        f"<p>Fresh CLI processes using the shared application core; {count} prepared specimens, 512 × 512 baking, 1,000 sampled faces, PCA and presentation actions. Slicer startup and rendering are excluded.</p>",
    ]
    if any((run.config.output / "omitted-artifacts.json").exists() for run in runs):
        sections.append("<p>Generated meshes, images, and arrays were hashed and removed after verification to bound disk use. Logs, profiles, metadata, and hashes are retained.</p>")
    # Keep the low-overhead comparison ahead of the instrumented tables. With
    # only two repeats these are descriptive medians, not confidence intervals.
    baseline_rows = []
    for backend in ("native", "morphoweave"):
        selected = [
            r
            for r in runs
            if r.config.backend == backend and r.config.mode == "baseline"
        ]
        sessions = [
            next(
                s
                for s in read_lines(r.config.output / "spans.jsonl", Span)
                if s.kind == "session"
            )
            for r in selected
        ]
        atlas_times = [
            sum(
                t.wall_seconds
                for t in summary.tasks[r.config.output.name]
                if t.pipeline == "ATLAS and Texture Transfer"
            )
            for r in selected
        ]
        peaks = [r.parent_peak_rss_bytes / 2**30 for r in selected]
        retained = (
            max(
                t.retained_array_bytes
                for r in selected
                for t in summary.tasks[r.config.output.name]
            )
            / 2**20
        )
        baseline_rows.append(
            (
                backend,
                len(selected),
                f"{statistics.median(s.wall_seconds for s in sessions):.2f}",
                f"{statistics.median(s.process_cpu_seconds for s in sessions):.2f}",
                f"{statistics.median(atlas_times):.2f}",
                f"{min(peaks):.2f}–{max(peaks):.2f}",
                f"{retained:.1f}",
            )
        )
    sections += [
        "<h2>Baseline comparison</h2>",
        table(
            (
                "Backend",
                "Repeats",
                "Median session wall s",
                "Median application CPU s",
                "Median atlas task time s",
                "Sampled application peak RSS GiB",
                "Peak retained NumPy MiB",
            ),
            baseline_rows,
        ),
    ]
    environment = json.loads((root / "environment.json").read_text())
    sections.append(
        f"<p>Environment: {html.escape(environment['platform'])}; {environment['ram_bytes'] / 2**30:.0f} GiB RAM; {environment['logical_cpus']} logical CPUs. Core revision {html.escape(environment['git_head'])}. Exact versions and input hashes are in environment.json.</p>"
    )
    rows = []
    for run in runs:
        spans = read_lines(run.config.output / "spans.jsonl", Span)
        session = next(s for s in spans if s.kind == "session")
        rows.append(
            (
                run.config.output.name,
                f"{session.wall_seconds:.2f}",
                f"{session.process_cpu_seconds:.2f}",
                f"{session.children_cpu_seconds:.2f}",
                f"{run.parent_peak_rss_bytes / 2**20:.1f}",
                f"{run.tree_peak_rss_bytes / 2**20:.1f}",
                f"{run.tree_peak_uss_bytes / 2**20:.1f}"
                if run.tree_peak_uss_bytes
                else "unavailable",
            )
        )
    sections += [
        "<h2>Run measurements</h2>",
        table(
            (
                "Run",
                "Session wall s",
                "Parent CPU s",
                "Child CPU s",
                "Parent peak RSS MiB",
                "Tree peak RSS MiB",
                "Tree peak USS MiB",
            ),
            rows,
        ),
        '<img src="memory.svg" alt="Memory timeline for the first baseline run of each backend">',
    ]
    sections.append(
        "<p>Use baseline runs for performance conclusions. CPU and allocation runs have separate instrumentation overhead. CPU seconds sum user and kernel time; parallel native threads can exceed wall time. RSS includes allocator retention and shared pages. Summed process RSS can double-count shared pages; USS is a less frequent private-page measurement. Sampled peaks can miss brief spikes.</p>"
    )
    for backend in ("native", "morphoweave"):
        baseline = next(
            r
            for r in runs
            if r.config.backend == backend and r.config.mode == "baseline"
        )
        sections += [
            f"<h2>{backend}: first baseline tasks</h2>",
            table(
                (
                    "Pipeline",
                    "Task",
                    "Wall s",
                    "Parent CPU s",
                    "Child CPU s",
                    "Parent peak MiB",
                    "Tree peak MiB",
                    "Retained NumPy MiB",
                ),
                (
                    (
                        t.pipeline,
                        t.task,
                        f"{t.wall_seconds:.3f}",
                        f"{t.parent_cpu_seconds:.3f}",
                        f"{t.children_cpu_seconds:.3f}",
                        f"{t.parent_peak_rss_bytes / 2**20:.1f}",
                        f"{t.tree_peak_rss_bytes / 2**20:.1f}",
                        f"{t.retained_array_bytes / 2**20:.1f}",
                    )
                    for t in summary.tasks[baseline.config.output.name]
                ),
            ),
        ]
        for run in runs:
            if run.config.backend != backend or run.config.mode == "baseline":
                continue
            name = run.config.output.name
            costs = sorted(
                summary.functions[name].items(),
                key=lambda pair: pair[1].exclusive_wall_seconds,
                reverse=True,
            )
            sections += [
                f"<h2>{name}: instrumented functions</h2>",
                table(
                    (
                        "Function",
                        "Calls",
                        "Inclusive wall s",
                        "Exclusive wall s",
                        "Exclusive thread CPU s",
                        "Array copies MiB",
                        "OBJ reads MiB",
                        "OBJ writes MiB",
                    ),
                    (
                        (
                            key,
                            v.calls,
                            f"{v.inclusive_wall_seconds:.3f}",
                            f"{v.exclusive_wall_seconds:.3f}",
                            f"{v.exclusive_thread_cpu_seconds:.3f}",
                            f"{v.array_copy_bytes / 2**20:.1f}",
                            f"{v.logical_read_bytes / 2**20:.1f}",
                            f"{v.logical_write_bytes / 2**20:.1f}",
                        )
                        for key, v in costs
                    ),
                ),
            ]
            if run.config.mode == "memory":
                sections += [
                    f'<p><a href="{name}/allocations.html">Interactive native allocation flame graph</a>. Tracked heap at peak: {summary.tracked_heap_peaks[name] / 2**20:.1f} MiB.</p>',
                    table(
                        (
                            "Live MiB at peak",
                            "Allocations",
                            "Python call chain (leaf first)",
                        ),
                        (
                            (
                                f"{s.bytes / 2**20:.2f}",
                                s.allocations,
                                " → ".join(s.python_stack[:7]),
                            )
                            for s in summary.peak_allocations[name][:15]
                        ),
                    ),
                ]
            elif run.config.mode == "cpu":
                sections.append(
                    f'<p><a href="{name}/cpu-01.txt">Atlas Yappi CPU report</a>; additional per-pipeline .pstats and text reports are in this run directory.</p>'
                )
    sections.append(
        "<h2>Interpretation limits</h2><p>Nested inclusive times must not be summed. Exclusive spans subtract only other explicitly instrumented spans; they are not pure kernel time. Yappi uses per-thread CPU clocks and exports thread totals plus combined Python-thread function tables. Task process CPU includes all application threads; waited-for child CPU is separate. Logical OBJ bytes count codec traffic, not physical storage transfers. Memray starts after module imports, tracks allocations in the application process including native libraries, and does not track Blender after exec. Its aggregated format records live allocations at peak, not total allocation churn; native symbols may be incomplete on macOS. No fifty-specimen peak memory is inferred by multiplying a three-specimen measurement.</p>"
    )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Small-clams profile</title><style>body{font:15px system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#17212b}table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13px}th,td{border-bottom:1px solid #d7dee5;text-align:left;padding:8px;overflow-wrap:anywhere}th{background:#eef3f7}img{max-width:100%}p{line-height:1.6;max-width:1100px}h2{margin-top:40px}</style>'
        + "".join(sections)
        + "</html>"
    )
    (root / "report.html").write_text(document)
    print(root / "report.html")


if __name__ == "__main__":
    main()
