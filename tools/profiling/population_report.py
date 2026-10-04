"""Render the full-population evidence after all requested measurements finish."""

import argparse
import pstats
import shutil
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from tools.profiling.serde import (
    CacheCapture,
    Comparison,
    HostMemorySample,
    PopulationAccuracy,
    PopulationDataset,
    PopulationEnvironment,
    ResourceUsage,
    Run,
    Span,
    Summary,
)
from tools.profiling.summarize import function_costs

matplotlib.use("Agg")


def number(value):
    return "unavailable" if value is None else f"{value:.6g}"


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def source_geometry_table(accuracy: PopulationAccuracy) -> str:
    """Describe workload density using the aligned originals used for fidelity."""
    rows = []
    for backend, measurements in accuracy.backends.items():
        specimens = measurements.specimens
        largest_id = max(specimens, key=lambda name: specimens[name].aligned.triangles)
        largest = specimens[largest_id].aligned
        rows.append(
            (
                backend,
                f"{sum(s.aligned.vertices for s in specimens.values()):,}",
                f"{sum(s.aligned.triangles for s in specimens.values()):,}",
                f"`{largest_id}`",
                f"{largest.vertices:,}",
                f"{largest.triangles:,}",
            )
        )
    # Compare counts by specimen ID; matching totals alone could hide a changed
    # source mesh if another specimen changed in the opposite direction.
    reference = accuracy.backends["native"].specimens
    same_counts = all(
        specimen.aligned.vertices == reference[name].aligned.vertices
        and specimen.aligned.triangles == reference[name].aligned.triangles
        for measurements in accuracy.backends.values()
        for name, specimen in measurements.specimens.items()
    )
    # Mesh records use float64 XYZ vertices, int64 triangle indices, and six
    # float64 UV coordinates per triangle. This estimates one copy, not RSS.
    geometry_bytes = sum(
        specimen.aligned.vertices * 3 * 8
        + specimen.aligned.triangles * (3 * 8 + 6 * 8)
        for specimen in reference.values()
    )
    return (
        table(
            (
                "Aligned originals",
                "Total vertices",
                "Total triangles",
                "Largest specimen",
                "Its vertices",
                "Its triangles",
            ),
            rows,
        )
        + f"\nThese counts describe the full aligned source meshes, before resampling to the atlas topology. Matching per-specimen source counts across backends: **{same_counts}**. Specimen count alone does not describe this workload's memory and codec costs.\n"
        + f"\nBoth workflows inspect and retain all source meshes before atlas construction. One copy of this population's vertex, triangle-index, and per-corner UV arrays alone occupies about **{geometry_bytes / 2**30:.3f} GiB** under the mesh record's 64-bit representation. This calculated array footprint excludes Python objects, parser temporaries, aligned/resampled geometry, and VTK copies; it is not a measured RSS peak. Source texture inspection retains paths after decoding each image, not all full-resolution decoded images. MorphoWeave's separate 192 MiB cache does not bound caller-owned population geometry.\n"
    )


def host_memory_context(root: Path) -> str:
    """Describe whole-host memory and page counters without attributing them."""
    path = root / "host-memory.jsonl"
    if not path.exists():
        return "Whole-host memory and swap context was not recorded.\n"
    samples = [
        HostMemorySample.model_validate_json(line)
        for line in path.read_text().splitlines()
    ]
    rows = []
    for version in ("B0", "current"):
        for directory in sorted((root / version).glob("*-baseline-*")):
            spans = [
                Span.model_validate_json(line)
                for line in (directory / "spans.jsonl").read_text().splitlines()
            ]
            session = next(span for span in spans if span.kind == "session")
            observed = [
                sample
                for sample in samples
                if session.start <= sample.time <= session.end
            ]
            if len(observed) < 2:
                rows.append((version, directory.name, "unavailable", "—", "—", "—"))
                continue
            # The monitor began after the first run started. Mark incomplete
            # intervals instead of implying their minima or counter deltas are totals.
            complete = (
                observed[0].time - session.start <= 5
                and session.end - observed[-1].time <= 5
                and max(b.time - a.time for a, b in pairwise(observed)) <= 6
            )
            rows.append(
                (
                    version,
                    directory.name,
                    "complete" if complete else "partial",
                    f"{min(sample.available_bytes for sample in observed) / 2**30:.2f}",
                    f"{(observed[-1].swap_in_bytes - observed[0].swap_in_bytes) / 2**30:.2f}",
                    f"{(observed[-1].swap_out_bytes - observed[0].swap_out_bytes) / 2**30:.2f}",
                )
            )
    return (
        "### Whole-host memory context\n\n"
        + table(
            (
                "Version",
                "Capture",
                "Monitor coverage",
                "Minimum available RAM (GiB)",
                "psutil sin delta (GiB)",
                "psutil sout delta (GiB)",
            ),
            rows,
        )
        + "\nThese two-second samples describe the whole computer, including other applications. "
        "Existing swap usage is not a measure of this application's allocations. "
        "On this macOS host, psutil 7.2.2's sin/sout fields are broader page-in/page-out counters, not dedicated swap-in/swap-out counters; "
        "see its [platform implementation](https://github.com/giampaolo/psutil/blob/release-7.2.2/psutil/arch/osx/mem.c#L100-L101). "
        "Their deltas cover only the observed interval, cannot be attributed entirely to InterDeCA, and do not by themselves establish swap thrashing. "
        "The first run has partial monitoring; captures were fresh processes, with operating-system file caches left intact.\n"
    )


def task_timing_table(root: Path) -> str:
    """Aggregate serial per-specimen tasks before taking the median over runs."""
    captures = {}
    families = set()
    for version in ("B0", "current"):
        for backend in ("native", "morphoweave"):
            groups = []
            for directory in sorted((root / version).glob(f"{backend}-baseline-*")):
                totals = defaultdict(float)
                for line in (directory / "spans.jsonl").read_text().splitlines():
                    span = Span.model_validate_json(line)
                    if span.kind != "task":
                        continue
                    # Only trailing specimen indices are normalized. For example,
                    # align, aligned, resample, and resampled remain distinct work.
                    prefix, _, suffix = span.name.rpartition("_")
                    family = prefix if suffix.isdigit() else span.name
                    totals[family] += span.wall_seconds
                groups.append(totals)
                families.update(totals)
            captures[version, backend] = groups
    medians = {
        family: {
            key: statistics.median(group.get(family, 0) for group in groups)
            for key, groups in captures.items()
        }
        for family in families
    }
    rows = []
    for family in sorted(
        families, key=lambda name: max(medians[name].values()), reverse=True
    ):
        values = medians[family]
        if max(values.values()) < 5:
            continue
        rows.append(
            (
                f"`{family}`",
                *(
                    f"{values['B0', backend]:.2f} → {values['current', backend]:.2f}"
                    for backend in ("native", "morphoweave")
                ),
            )
        )
    # Shared staging moved out of the two high-level MorphoWeave tasks. Sum
    # those related stages within each run before comparing their medians.
    combined = [
        statistics.median(
            sum(
                group.get(family, 0)
                for family in ("morphoweave_inputs", "atlas", "morphoweave_aligned")
            )
            for group in captures[version, "morphoweave"]
        )
        for version in ("B0", "current")
    ]
    return (
        "### Where the elapsed time goes\n\n"
        + f"MorphoWeave preparation + atlas + alignment takes a median **{combined[0]:.2f} → {combined[1]:.2f} s** across each version's three normal runs. This groups the stages whose boundaries changed during optimization.\n\n"
        + table(
            (
                "Task family",
                "Native Atlas: B0 → current (s)",
                "MorphoWeave: B0 → current (s)",
            ),
            rows,
        )
        + "\nEach entry is the median sum of that task family's durations over a complete run. "
        "Families taking at least five seconds in any implementation are shown; zero means the family is absent or negligible. "
        "Native atlas construction includes the per-specimen `contribute` tasks and the final `atlas` task. "
        "`inspect` validates source inputs, `aligned` and `resampled` write output files, and `resample` constructs correspondence. "
        "`bake` includes waiting for Blender; `analysis` prepares color analysis and `sample` evaluates specimen colors. "
        "B0 stages MorphoWeave inputs inside its high-level atlas/alignment tasks; current code exposes shared staging as `morphoweave_inputs`, so those three families must be considered together. "
        "Task medians need not sum to the median application session, which also includes work outside tasks.\n"
    )


def fidelity_plot(accuracy: PopulationAccuracy, destination: Path):
    """Show every specimen alongside the median without hiding outliers."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    metrics = (
        (
            "Collapsed resampled triangles",
            "% of triangles",
            lambda s: 100 * s.resampled.collapsed_fraction,
        ),
        (
            "Resampled interiors to original surface",
            "RMS / landmark centroid size (%)",
            lambda s: 100 * s.interiors_to_input.rms / s.aligned_centroid_size,
        ),
        (
            "Transferred landmarks (in-sample)",
            "RMS / landmark centroid size (%)",
            lambda s: 100 * s.transferred_landmarks_error_over_centroid_size.rms,
        ),
        (
            "Baked versus source texture color",
            "RGB channel MAE (0–255)",
            lambda s: s.baked_to_source_color.mean_absolute_channel_0_255,
        ),
    )
    for ax, (title, units, getter) in zip(axes.flat, metrics, strict=True):
        for position, (backend, color) in enumerate(
            (("native", "#007f83"), ("morphoweave", "#b76520"))
        ):
            values = [
                getter(record)
                for record in accuracy.backends[backend].specimens.values()
            ]
            values = np.array([value for value in values if value is not None])
            # A fixed horizontal jitter separates nearby points without changing
            # their measured vertical values or allowing large outliers to vanish.
            jitter = np.random.default_rng(413).uniform(-0.12, 0.12, len(values))
            ax.scatter(position + jitter, values, s=16, color=color, alpha=0.65)
            if len(values):
                ax.hlines(
                    np.median(values),
                    position - 0.22,
                    position + 0.22,
                    color=color,
                    lw=2.5,
                )
        ax.set_xticks((0, 1), ("Native Atlas", "MorphoWeave"))
        ax.set(title=title, ylabel=units, xlim=(-0.5, 1.5))
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    count = len(accuracy.cross_backend.specimens)
    figure.suptitle(f"{count} clams · output fidelity to supplied geometry and colors")
    figure.text(
        0.5,
        0.015,
        "Each dot is one specimen; horizontal bars are medians. These are not independent anatomical accuracy scores.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.98))
    figure.savefig(destination / "fidelity.png", dpi=180)
    figure.savefig(destination / "fidelity.svg")
    plt.close(figure)


def codec_cpu_table(root: Path) -> str:
    """Report measured codec CPU without calling logical file bytes disk traffic."""
    rows = []
    for version in ("B0", "current"):
        for backend in ("native", "morphoweave"):
            # Use this same span capture's process CPU denominator. Normal-run
            # CPU comes from different executions and cannot form this fraction.
            spans = [
                Span.model_validate_json(line)
                for line in (root / version / f"{backend}-spans-1/spans.jsonl")
                .read_text()
                .splitlines()
            ]
            functions = function_costs(spans)
            read = functions["MeshIO._read_obj"].exclusive_thread_cpu_seconds
            write = functions["MeshIO.write_obj"].exclusive_thread_cpu_seconds
            session_cpu = next(
                span.process_cpu_seconds for span in spans if span.kind == "session"
            )
            rows.append(
                (
                    version,
                    backend,
                    f"{read:.2f}",
                    f"{write:.2f}",
                    f"{read + write:.2f}",
                    f"{session_cpu:.2f}",
                    f"{100 * (read + write) / session_cpu:.1f}",
                )
            )
    return (
        table(
            (
                "Version",
                "Backend",
                "OBJ parsing CPU (s)",
                "OBJ writing CPU (s)",
                "Combined codec CPU (s)",
                "Same-capture session CPU (s)",
                "Codec share (%)",
            ),
            rows,
        )
        + "\nThese are exclusive thread-CPU measurements from the separate span pass. The share divides their sum by application process CPU from that same session; executed Blender processes are excluded from both. They expose serialization/parsing work, not elapsed disk waiting, and are not a partition of baseline CPU time.\n"
    )


def cache_evidence(root: Path) -> str:
    """Keep bounded cache storage distinct from caller-owned population data."""
    path = root / "current/morphoweave-spans-1/mesh-cache.jsonl"
    captures = [
        CacheCapture.model_validate_json(line) for line in path.read_text().splitlines()
    ]
    cache = captures[-1]
    return (
        table(
            ("Current MorphoWeave span capture", "Value"),
            (
                (
                    "Array budget / observed peak (MiB)",
                    f"{cache.budget_bytes / 2**20:.2f} / {cache.peak_array_bytes / 2**20:.2f}",
                ),
                (
                    "Hits / misses / evictions",
                    f"{cache.hits} / {cache.misses} / {cache.evictions}",
                ),
                ("Peak entries", cache.peak_entries),
                (
                    "Entries / array bytes after close",
                    f"{cache.entries} / {cache.array_bytes}",
                ),
            ),
        )
        + "\nThe bounded cache is much smaller than this working set. Its limited reuse explains why the small-clams reduction in repeated OBJ reads does not carry over to 47 specimens. Closing this cache does not release geometry still owned by the caller or upstream algorithm.\n"
    )


def compact_source(value: str, root: Path) -> str:
    """Keep profiler source locations readable without losing file or line identity."""
    value = value.replace(str(root / "sources") + "/", "")
    value = value.replace(str(Path(__file__).resolve().parents[2]) + "/", "")
    return value.replace("|", "\\|")


def instrumented_cpu_tables(root: Path) -> str:
    """Attribute profiled thread CPU, separately from uninstrumented speed claims."""
    sections = [
        "### Function-level CPU attribution\n",
        (
            "Yappi records Python-thread CPU, including built-in calls, in a separate capture. "
            "The ten largest self-time entries below combine that capture's sequential application pipelines. "
            "Profiling is enabled during execution-manager pipelines; work between those pipelines is outside these call tables. "
            "Self time excludes called functions; inclusive time includes them and must not be summed across rows. "
            "Deterministic profiling substantially slows Python-heavy codecs, so these instrumented values identify call sites but do not estimate normal runtime. "
            "Executed Blender processes are excluded.\n"
        ),
    ]
    for version in ("B0", "current"):
        for backend in ("native", "morphoweave"):
            directory = root / version / f"{backend}-cpu-1"
            paths = sorted(directory.glob("*.pstats"))
            if not paths:
                raise ValueError(f"Missing CPU profiles: {version}/{backend}")
            durations = []
            # Compare like session boundaries, excluding interpreter startup in
            # both modes, to make instrumentation overhead visible to the reader.
            for capture in [
                directory,
                *sorted((root / version).glob(f"{backend}-baseline-*")),
            ]:
                spans = (
                    Span.model_validate_json(line)
                    for line in (capture / "spans.jsonl").read_text().splitlines()
                )
                durations.append(
                    next(span.wall_seconds for span in spans if span.kind == "session")
                )
            baseline = statistics.median(durations[1:])
            # Pipelines execute serially and each file owns one profiling interval;
            # adding them does not double-count overlapping profile captures.
            stats = pstats.Stats(*(str(path) for path in paths))
            rows = []
            for (file, line, name), values in sorted(
                stats.stats.items(), key=lambda item: item[1][2], reverse=True
            )[:10]:
                _, calls, self_seconds, inclusive_seconds, _ = values
                label = name.replace("|", "\\|")
                rows.append(
                    (
                        f"`{label}`",
                        f"`{compact_source(file, root)}:{line}`",
                        f"{calls:,}",
                        f"{self_seconds:.2f}",
                        f"{inclusive_seconds:.2f}",
                    )
                )
            sections += [
                f"**{version} / {backend}**\n",
                f"Profiled session: {durations[0]:.2f} s; normal baseline median: {baseline:.2f} s. This instrumented capture took {durations[0] / baseline:.2f} times as long; the ratio also includes host and run-to-run variation.\n",
                table(
                    ("Function", "Source", "Calls", "Self CPU (s)", "Inclusive CPU (s)"),
                    rows,
                ),
            ]
    return "\n".join(sections)


def allocation_tables(root: Path) -> str:
    """Show allocations simultaneously live at each measured heap high-water mark."""
    sections = [
        "### What occupies the peak live heap\n",
        (
            "Each table shows the eight largest allocation stacks simultaneously live at that capture's Memray high-water mark. "
            "These are allocation origins, not necessarily the objects' current owners. "
            "The total uses every stack, while the table shows only the largest. "
            "Excerpts omit recorder-wrapper frames; the accompanying summaries retain complete stacks. "
            "Native allocations are tracked, but executed Blender processes and memory outside the tracked allocator are excluded. "
            "Tracking covers the CLI session, excluding earlier interpreter/import allocations. "
            "The recorder does not enable individual Python-allocator tracing: Python arenas may contain free slots, so allocation counts and sizes are not an exact census of live Python objects. "
            "Tracked allocation peaks are distinct from cumulative allocation/copy traffic and process RSS.\n"
        ),
    ]
    for version in ("B0", "current"):
        summary = Summary.model_validate_json(
            (root / version / "summary.json").read_text()
        )
        for backend in ("native", "morphoweave"):
            capture = f"{backend}-memory-1"
            rows = []
            for site in summary.peak_allocations[capture][:8]:
                # Hide recorder wrappers so excerpts expose actual application
                # callers. Complete stacks remain in the accompanying summary.
                frames = [
                    frame
                    for frame in site.python_stack
                    if "/tools/profiling/capture.py:" not in frame
                ]
                origin = "<br>← ".join(
                    f"`{compact_source(frame, root)}`"
                    for frame in frames[:5]
                ) or "No Python frame recorded"
                rows.append(
                    (
                        f"{site.bytes / 2**20:.2f}",
                        f"{site.allocations:,}",
                        origin,
                    )
                )
            sections += [
                f"**{version} / {backend}: {summary.tracked_heap_peaks[capture] / 2**30:.3f} GiB tracked live peak**\n",
                table(("Live MiB", "Tracked allocations", "Allocation origin"), rows),
            ]
    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, destination = args.root.resolve(), args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    comparison = Comparison.model_validate_json((root / "comparison.json").read_text())
    accuracy = PopulationAccuracy.model_validate_json(
        (root / "accuracy.json").read_text()
    )
    manifest = PopulationDataset.model_validate_json(
        (root / "dataset.json").read_text()
    )
    count = len(manifest.specimens)
    if any(
        set(backend.specimens) != set(manifest.specimens)
        for backend in accuracy.backends.values()
    ):
        raise ValueError("Incomplete accuracy measurements")
    if set(accuracy.cross_backend.specimens) != set(manifest.specimens):
        raise ValueError("Incomplete cross-backend accuracy measurements")
    # Publish compact evidence in Git; raw captures remain in the durable archive.
    for name in ("comparison.json", "accuracy.json", "dataset.json"):
        shutil.copyfile(root / name, destination / name)
    for version in ("B0", "current"):
        shutil.copyfile(
            root / version / "environment.json",
            destination / f"{version}-environment.json",
        )
        # Summaries and first-repeat memory curves make allocation attribution
        # reviewable without adding multi-gigabyte raw profiler files to Git.
        for name in ("summary.json", "memory.png", "memory.svg"):
            shutil.copyfile(root / version / name, destination / f"{version}-{name}")
    provenance = root / "import-provenance.json"
    if provenance.exists():
        shutil.copyfile(provenance, destination / provenance.name)
    # Validation runs after capture finalization. Retain its exact output when
    # the report is regenerated after the audit, without inventing test results.
    for name in ("final-pytest.log", "final-integrity-audit.log"):
        if (root / name).exists():
            shutil.copyfile(root / name, destination / name)
    backends = ("native", "morphoweave")
    labels = ("Native Atlas", "MorphoWeave")
    metrics = [
        ("Application session (s)", "session_wall_s", 1),
        ("Atlas + texture transfer (s)", "ATLAS and Texture Transfer_wall_s", 1),
        ("Color processing (s)", "Color processing_wall_s", 1),
        ("Application CPU (s)", "application_cpu_s", 1),
        ("Blender CPU (s)", "blender_cpu_s", 1),
        ("Peak application RSS (GiB)", "peak_rss_bytes", 2**30),
        ("Peak application + Blender RSS (GiB)", "tree_peak_rss_bytes", 2**30),
        ("Peak tracked allocations (GiB)", "tracked_heap_peak_bytes", 2**30),
    ]
    rows = []
    for label, key, divisor in metrics:
        values = []
        for backend in backends:
            change = comparison.metrics[backend].get(key)
            values.append(
                "unavailable"
                if change is None
                else f"{change.reference / divisor:.3f} → {change.candidate / divisor:.3f} ({change.reduction_percent:.2f}% reduction)"
            )
        rows.append((label, *values))
    ownership_rows = []
    for label, key, divisor in (
        ("Manager-retained arrays (GiB)", "retained_array_bytes", 2**30),
        ("OBJ reads (calls)", "obj_read_calls", 1),
        ("OBJ input volume (GiB)", "obj_read_bytes", 2**30),
        ("OBJ writes (calls)", "obj_write_calls", 1),
        ("OBJ output volume (GiB)", "obj_write_bytes", 2**30),
        ("Instrumented array copies (GiB)", "array_copy_bytes", 2**30),
    ):
        values = []
        for backend in backends:
            change = comparison.metrics[backend].get(key)
            values.append(
                "unavailable"
                if change is None
                else f"{change.reference / divisor:.3f} → {change.candidate / divisor:.3f}"
            )
        ownership_rows.append((label, *values))
    sections = [
        f"# Full-population comparison: {count} clams\n",
        "This compares pre-optimization InterDeCA B0 with the current application, for both Native Atlas and MorphoWeave. It does not rerun the older `3d_color_package` implementation. The directory is named `mussels_52`, but its reviewed manifest and complete inputs contain 47 specimens.\n",
        f"Original capture revision: `{comparison.reference_revision}` (the `129c2e2` numerical implementation plus its profiling harness). Current capture revision: `{comparison.candidate_revision}`. Both use the same uv environment and pinned, unmodified MorphoWeave source.\n",
        "## Input preparation\n",
        "Source files were read-only and hashed. Dense meshes and landmarks were converted from LPS to RAS without decimation or image resizing. The raw `UF_IZ_63733` mesh contains one exactly zero-area triangle (zero-based face 177014; two coincident corners). Both revisions reject it under strict source validation. That one face was removed only from the private common input copy, preserving all vertices and the order and corner UVs of all remaining faces. Thus these are measurements of the identically prepared 47-specimen dataset; the unmodified raw dataset fails validation.\n",
        "Source images comprise 27 at 4096², 18 at 8192², and two at 2048². Each specimen has 48 landmarks. Baking is 512² and color analysis samples 1,000 faces with neighbor averaging, followed by two-component PCA, swapped axes, reconstruction, display, selection, export, and clear-selection.\n",
        source_geometry_table(accuracy),
        "## Performance\n",
        table(
            ("Metric", "Native Atlas: B0 → current", "MorphoWeave: B0 → current"), rows
        ),
        "Timing/CPU values are medians of three fresh-process baseline captures per implementation/backend. Resident memory is the maximum sampled peak across those repeats. Separate span, CPU, and allocation captures are excluded from timing claims, giving 24 captures total. Session wall time and application CPU exclude interpreter startup and Slicer rendering. CPU is processor time, not elapsed time. Output fingerprinting, storage compaction, accuracy diagnostics, and report generation happen outside the measured child. RSS includes allocator retention; simultaneous process-tree RSS can double-count shared pages. Memray excludes executed Blender processes.\n",
        "![Performance comparison](performance.png)\n",
    ]
    ranges = []
    for version in ("B0", "current"):
        for backend, label in zip(backends, labels, strict=True):
            directories = sorted((root / version).glob(f"{backend}-baseline-*"))
            sessions = []
            peaks = []
            high_water = []
            platform = PopulationEnvironment.model_validate_json(
                (root / version / "environment.json").read_text()
            ).platform
            for directory in directories:
                run = Run.model_validate_json((directory / "run.json").read_text())
                spans = [
                    Span.model_validate_json(line)
                    for line in (directory / "spans.jsonl").read_text().splitlines()
                ]
                sessions.append(
                    next(span.wall_seconds for span in spans if span.kind == "session")
                )
                peaks.append(run.parent_peak_rss_bytes / 2**30)
                # The captured macOS getrusage value is measured in bytes. Do not
                # guess the native units if this report is reused on another OS.
                if platform.startswith(("macOS", "Darwin")):
                    usage = ResourceUsage.model_validate_json(
                        (directory / "rusage.json").read_text()
                    )
                    high_water.append(usage.maxrss_platform_units / 2**30)
            ranges.append(
                (
                    version,
                    label,
                    len(sessions),
                    f"{min(sessions):.3f}–{max(sessions):.3f}",
                    f"{min(peaks):.3f}–{max(peaks):.3f}",
                    f"{min(high_water):.3f}–{max(high_water):.3f}"
                    if high_water
                    else "unavailable",
                )
            )
    sections += [
        table(
            (
                "Version",
                "Backend",
                "Repeats",
                "Session range (s)",
                "RSS peak range (GiB)",
                "OS high-water RSS range (GiB)",
            ),
            ranges,
        ),
        host_memory_context(root),
        task_timing_table(root),
        "### Retention and serialization\n",
        table(
            ("Metric", "Native Atlas: B0 → current", "MorphoWeave: B0 → current"),
            ownership_rows,
        ),
        "Manager retention is the maximum unique NumPy-buffer ownership reported at task boundaries in baseline captures; it excludes objects held elsewhere. Codec calls, logical file volume, and cumulative instrumented array-copy volume come from the separate span capture. Logical bytes are not physical disk traffic, and cumulative copy bytes are not peak live memory. The bounded cache does not limit total population geometry held by the algorithms.\n",
        codec_cpu_table(root),
        cache_evidence(root),
        instrumented_cpu_tables(root),
        allocation_tables(root),
        "[B0 memory timeline](B0-memory.png) and [current memory timeline](current-memory.png) show each backend's first baseline repeat. The peak-RSS table above uses all three repeats. Full task/function and allocation-stack summaries are in [B0-summary.json](B0-summary.json) and [current-summary.json](current-summary.json).\n",
        "## Preservation of numerical output\n",
        f"Compatible environment: **{comparison.compatible_environment}**. Identical numerical outputs and checked metadata across versions and repeated/instrumented runs: **{comparison.identical_outputs}**. These checks preserve vertex order, triangle order, corner UVs, decoded texture results, population and presentation artifacts, and completed actions. They establish agreement within each backend, not equivalence between Atlas and MorphoWeave.\n",
    ]
    if comparison.issues or comparison.investigation_alerts:
        sections += [
            "Issues/alerts:\n",
            "\n".join(
                f"- {item}"
                for item in comparison.issues + comparison.investigation_alerts
            )
            + "\n",
        ]
    sections += [
        "## Geometry and landmark measurements\n",
        "![Per-specimen fidelity measurements](fidelity.png)\n",
    ]
    rows = []
    for name, label in zip(backends, labels, strict=True):
        backend = accuracy.backends[name]
        rows.append(
            (
                label,
                backend.atlas.vertices,
                backend.atlas.triangles,
                number(backend.atlas_centroid_size),
                number(backend.mean_input_centroid_size),
                number(backend.atlas_landmark_to_surface.rms),
            )
        )
    sections += [
        table(
            (
                "Backend",
                "Atlas vertices",
                "Atlas triangles",
                "Atlas centroid size",
                "Mean input centroid size",
                "Atlas landmark-to-surface RMS",
            ),
            rows,
        )
    ]
    cross = accuracy.cross_backend
    sections += [
        f"MorphoWeave-to-Native fitted atlas scale is **{cross.morphoweave_to_native_scale:.9g}**. Atlas landmark RMS after rigid alignment is **{cross.atlas_landmark_rigid_residual.rms:.6g}**; after separately fitting scale it is **{cross.atlas_landmark_similarity_residual.rms:.6g}**. Distances use input coordinate units; no millimeter assumption is made.\n"
    ]
    rows = []
    for name, surface in (
        ("Rigid", cross.atlas_surface_rigid),
        ("Similarity (fitted scale)", cross.atlas_surface_similarity),
    ):
        for direction, values in (
            ("Native → MorphoWeave", surface.first_to_second),
            ("MorphoWeave → Native", surface.second_to_first),
        ):
            rows.append(
                (
                    name,
                    direction,
                    number(values.rms),
                    number(values.p95),
                    number(values.maximum),
                )
            )
    sections += [
        table(("Atlas alignment", "Direction", "RMS", "P95", "Sampled maximum"), rows),
        f"Each surface direction uses {accuracy.samples_per_surface:,} uniformly area-weighted samples (seed {accuracy.seed}). These sampled maxima are not exact Hausdorff bounds. Proper rotations exclude reflections.\n",
        "The fitted scale minimizes labeled-landmark residuals. It does not minimize surface distances, so surface RMS need not decrease after fitting it.\n",
    ]
    rows = []
    for name, label in zip(backends, labels, strict=True):
        records = list(accuracy.backends[name].specimens.values())
        for metric, getter in (
            (
                "Collapsed resampled faces (%)",
                lambda s: 100 * s.resampled.collapsed_fraction,
            ),
            ("Vertex-to-original RMS", lambda s: s.vertex_to_input.rms),
            ("Triangle-interior-to-original RMS", lambda s: s.interiors_to_input.rms),
            ("Original-to-resampled RMS", lambda s: s.input_to_resampled.rms),
            (
                "Triangle-interior RMS / centroid size (%)",
                lambda s: 100 * s.interiors_to_input.rms / s.aligned_centroid_size,
            ),
            (
                "Original-to-resampled RMS / centroid size (%)",
                lambda s: 100 * s.input_to_resampled.rms / s.aligned_centroid_size,
            ),
            (
                "Transferred landmark RMS / centroid size (%)",
                lambda s: 100 * s.transferred_landmarks_error_over_centroid_size.rms,
            ),
            ("Aligned / input centroid size", lambda s: s.size_ratio),
            (
                "Resampled / original surface area",
                lambda s: s.resampled.area / s.aligned.area,
            ),
        ):
            values = [getter(record) for record in records]
            rows.append(
                (
                    label,
                    metric,
                    number(statistics.median(values)),
                    f"{number(min(values))}–{number(max(values))}",
                )
            )
    sections += [
        table(
            (
                "Backend",
                "Per-specimen metric",
                f"Median over {count}",
                f"Range over {count}",
            ),
            rows,
        ),
        f"Projection fidelity is measured against each backend's own aligned original mesh. The vertex diagnostic samples up to {accuracy.samples_per_surface:,} vertex indices per specimen. Vertex fidelity alone does not establish triangle-interior fidelity, surface coverage, or anatomical correspondence. Landmark transfer projects atlas landmarks to atlas triangles and carries their barycentric coordinates to the specimen. Because the landmarks also guide fitting, this is an in-sample check and includes atlas landmark-to-surface offset; it is not an independent anatomical accuracy score.\n",
    ]
    rows = []
    for label, getter in (
        (
            "Aligned-landmark rigid-fit RMS",
            lambda s: s.aligned_landmark_rigid_residual.rms,
        ),
        (
            "Resampled Native → MorphoWeave surface RMS",
            lambda s: s.resampled_surface_after_specimen_rigid_fit.first_to_second.rms,
        ),
        (
            "Resampled MorphoWeave → Native surface RMS",
            lambda s: s.resampled_surface_after_specimen_rigid_fit.second_to_first.rms,
        ),
    ):
        # Fit each specimen's pose with its own matching landmarks, without a
        # fitted scale. Normalize with Native's input-preserving centroid size.
        values = [
            100
            * getter(record)
            / accuracy.backends["native"].specimens[specimen].aligned_centroid_size
            for specimen, record in cross.specimens.items()
        ]
        rows.append(
            (
                label,
                number(statistics.median(values)),
                f"{number(min(values))}–{number(max(values))}",
            )
        )
    sections += [
        table(
            (
                "Per-specimen cross-backend difference",
                "Median / centroid size (%)",
                "Range / centroid size (%)",
            ),
            rows,
        ),
        "These specimen comparisons remove pose using each specimen's labeled landmarks, without fitting its scale. Surface-area ratios above one can reflect bridging, folds, or overlapping triangles; they do not establish useful additional detail.\n",
        "## Surface-color measurements\n",
    ]
    rows = []
    for name, label in zip(backends, labels, strict=True):
        colors = [
            s.baked_to_source_color for s in accuracy.backends[name].specimens.values()
        ]
        for metric, getter in (
            ("RGB channel MAE (0–255)", lambda c: c.mean_absolute_channel_0_255),
            ("RGB channel RMS (0–255)", lambda c: c.rms_channel_0_255),
            ("Mean Delta E 76", lambda c: c.delta_e_76.mean),
            ("Baked valid coverage (%)", lambda c: 100 * c.first_covered / c.samples),
            ("Source valid coverage (%)", lambda c: 100 * c.second_covered / c.samples),
            ("Joint valid coverage (%)", lambda c: 100 * c.jointly_covered / c.samples),
        ):
            values = [getter(c) for c in colors if getter(c) is not None]
            rows.append(
                (
                    label,
                    metric,
                    number(statistics.median(values)) if values else "unavailable",
                    f"{number(min(values))}–{number(max(values))}"
                    if values
                    else "unavailable",
                )
            )
    sections += [
        table(
            (
                "Baked versus original source",
                "Metric",
                "Median over specimens",
                "Range",
            ),
            rows,
        ),
        "Source comparison samples resampled triangle interiors, finds their nearest point on the aligned original surface, and interpolates each texture at that surface location. This combines projection and baking error. Bilinear sampling uses premultiplied alpha and Blender pixel centers. Samples require alpha ≥ 0.99 in both images; missing coverage is counted separately. Source resolution is unchanged, but baked textures are deliberately 512².\n",
    ]
    colors = [s.color_at_common_atlas_locations for s in cross.specimens.values()]
    errors = [
        c.mean_absolute_channel_0_255
        for c in colors
        if c.mean_absolute_channel_0_255 is not None
    ]
    common_color_median = number(statistics.median(errors)) if errors else "unavailable"
    sections += [
        f"Across backends, the median specimen RGB channel MAE at common atlas surface locations is **{common_color_median}/255**. The average atlas texture has MAE **{number(cross.average_color_at_common_atlas_locations.mean_absolute_channel_0_255)}/255**. Matching uses nearest spatial points after a shared rigid atlas fit; different UV layouts are never compared at identical pixel indices. These differences do not establish which backend has better anatomical color correspondence.\n",
        f"PCA pairwise specimen-distance correlation is **{cross.pca_pairwise_distance_correlation:.6g}**; relative RMS distance difference is **{100 * cross.pca_pairwise_distance_relative_rms:.4g}%**. Different atlas sampling creates different feature spaces, so this is a descriptive population difference, not model accuracy.\n",
        "## Evidence\n",
        f"Full inputs, sources, logs, numerical outputs, resource samples, CPU profiles, and allocation captures: `{root}`. Compact [performance comparison](comparison.json), [all per-specimen accuracy measurements](accuracy.json), [input/repair manifest](dataset.json), [B0 environment](B0-environment.json), and [current environment](current-environment.json) accompany this report. The tests run in the uv environment; nothing is registered with Slicer/CMake.\n",
    ]
    if (destination / "BASELINE_RESULTS.md").exists():
        sections.append(
            "[Interpretation of the performance and numerical findings](BASELINE_RESULTS.md).\n"
        )
    if all(
        (destination / name).exists()
        for name in ("final-pytest.log", "final-integrity-audit.log")
    ):
        sections.append(
            "Final verification evidence: [pytest output](final-pytest.log) and [source/input/runtime integrity audit](final-integrity-audit.log).\n"
        )
    (destination / "RESULTS.md").write_text("\n".join(sections))
    # Fixed units and a zero baseline make version/backend comparisons readable.
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (title, key, divisor) in zip(
        axes,
        (
            ("Application session (s)", "session_wall_s", 1),
            ("Application CPU (s)", "application_cpu_s", 1),
            ("Peak application RSS (GiB)", "peak_rss_bytes", 2**30),
        ),
        strict=True,
    ):
        x = np.arange(2)
        before = [comparison.metrics[b][key].reference / divisor for b in backends]
        after = [comparison.metrics[b][key].candidate / divisor for b in backends]
        ax.bar(x - 0.18, before, 0.36, label="Original B0", color="#969eac")
        ax.bar(x + 0.18, after, 0.36, label="Current", color="#007f83")
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.suptitle(f"{count} specimens · same prepared inputs and numerical settings")
    figure.tight_layout()
    figure.savefig(destination / "performance.png", dpi=180)
    figure.savefig(destination / "performance.svg")
    plt.close(figure)
    fidelity_plot(accuracy, destination)


if __name__ == "__main__":
    main()
