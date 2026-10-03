"""Compare a candidate with an accepted profile, including relocated archives."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from interdeca.io import MeshIO
from tools.profiling.serde import Comparison, MetricChange, Run, Span, Summary

REQUIRED_ACTIONS = {
    "atlas",
    "colors",
    "population",
    "plot",
    "morph_controls",
    "morphospace",
    "display",
    "selection",
    "export",
    "clear_selection",
}
IDENTITY_FIELDS = {"request_id", "run_id", "dataset_id", "model_id"}


class Outputs(NamedTuple):
    run: Run
    files: dict[str, Path]
    hashes: dict[str, str]
    presentations: list[dict]
    completions: list[dict]
    atlas_metadata: dict


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relocated(path: str, original: Path, current: Path) -> Path:
    """Resolve archived run paths locally; never silently read the old directory."""
    return current / Path(path).relative_to(original)


def outputs(directory: Path) -> Outputs:
    run = Run.model_validate_json((directory / "run.json").read_text())
    if run.exit_code:
        raise ValueError(f"{directory.name}: failed run ({run.exit_code})")
    events = [
        json.loads(line)
        for line in (directory / "cli/events.jsonl").read_text().splitlines()
    ]
    if any(event["type"] in ("failed", "cancelled") for event in events):
        raise ValueError(f"{directory.name}: failed/cancelled application action")
    results = [event for event in events if event["type"] == "result"]
    actions = {event["action"] for event in results}
    if not REQUIRED_ACTIONS <= actions:
        raise ValueError(
            f"{directory.name}: missing actions {REQUIRED_ACTIONS - actions}"
        )
    last = {event["action"]: event["payload"] for event in results}
    atlas = relocated(last["atlas"]["directory"], run.config.output, directory)
    if not atlas.is_dir():
        raise ValueError(f"{directory.name}: missing atlas directory")
    files = {
        "atlas/" + str(path.relative_to(atlas)): path
        for path in atlas.rglob("*")
        if path.is_file() and path.name != "atlas_run.json"
    }

    def collect(value, key):
        if isinstance(value, dict):
            if {"path", "dtype", "shape"} <= value.keys():
                path = directory / "cli" / value["path"]
                if not path.resolve().is_relative_to((directory / "cli").resolve()):
                    raise ValueError(f"Artifact escapes its run: {key}")
                array = np.load(path, allow_pickle=False, mmap_mode="r")
                if (
                    list(array.shape) != value["shape"]
                    or array.dtype.str != value["dtype"]
                ):
                    raise ValueError(
                        f"Artifact shape/dtype disagrees with metadata: {key}"
                    )
                files[key] = path
            else:
                for name, child in value.items():
                    collect(child, key + "/" + name)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect(child, key + "/" + str(index))

    collect(last, "presentations")
    hashes = {key: digest(path) for key, path in files.items()}
    if hashes != run.output_hashes:
        changed = sorted(
            key
            for key in hashes.keys() | run.output_hashes.keys()
            if hashes.get(key) != run.output_hashes.get(key)
        )
        raise ValueError(
            f"{directory.name}: stored output hashes disagree with files: {changed}"
        )

    # Only explicit session identities and output locations are normalized. Keep
    # all numerical fields, actions, revisions, and even earlier plot presentations.
    identities = {key: {} for key in IDENTITY_FIELDS}

    def normalize(value, key=""):
        if key in IDENTITY_FIELDS and value is not None:
            names = identities[key]
            return names.setdefault(value, f"{key}-{len(names)}")
        if isinstance(value, dict):
            if {"path", "dtype", "shape"} <= value.keys():
                path = directory / "cli" / value["path"]
                return {**value, "path": digest(path)}
            return {name: normalize(child, name) for name, child in value.items()}
        if isinstance(value, list):
            return [normalize(child) for child in value]
        if key in ("directory", "average") and isinstance(value, str):
            return str(Path(value).relative_to(atlas_original))
        return value

    atlas_original = Path(last["atlas"]["directory"])
    presentations = [normalize(event) for event in results]
    # Progress task counts may change with an optimization. Terminal actions must
    # still complete; normalize request identities consistently with result events.
    completions = [normalize(event) for event in events if event["type"] == "completed"]
    accepted = [event for event in events if event["type"] == "accepted"]
    completed_ids = [
        event["request_id"] for event in events if event["type"] == "completed"
    ]
    if len(set(completed_ids)) != len(completed_ids) or any(
        event["request_id"] not in completed_ids for event in accepted
    ):
        raise ValueError(f"{directory.name}: missing or duplicate action completion")
    metadata = json.loads((atlas / "atlas_run.json").read_text())
    for name in ("models", "landmarks", "textures"):
        if metadata["options"].get(name) is not None:
            metadata["options"][name] = str(
                Path(metadata["options"][name]).relative_to(run.config.inputs)
            )
    metadata["options"]["output"] = "<run output>"
    # Blender identity is checked in the environment, not by its installation path.
    metadata["options"]["blender"] = "<environment Blender>"
    return Outputs(run, files, hashes, presentations, completions, metadata)


def differences(left, right, prefix="") -> list[str]:
    """Report the precise field without hiding changes behind a tolerance."""
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            location = f"{prefix}/{key}"
            if key not in left or key not in right:
                result.append(f"{location}: missing field")
            else:
                result.extend(differences(left[key], right[key], location))
        return result
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            issue
            for index, (a, b) in enumerate(zip(left, right, strict=True))
            for issue in differences(a, b, f"{prefix}/{index}")
        ]
    return [] if left == right else [f"{prefix}: {left!r} -> {right!r}"]


def array_difference(left, right) -> str:
    if left.shape != right.shape or left.dtype != right.dtype:
        return f"shape/dtype {left.shape} {left.dtype} -> {right.shape} {right.dtype}"
    changed = np.count_nonzero(left != right)
    maximum = (
        float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        if left.size
        else 0
    )
    return f"shape={left.shape}, dtype={left.dtype}, changed={changed}, max_abs={maximum:.17g}"


def file_difference(left: Path, right: Path) -> str:
    if left.suffix == ".npy":
        return array_difference(
            np.load(left, allow_pickle=False), np.load(right, allow_pickle=False)
        )
    if left.suffix.lower() in (".png", ".tiff"):
        with Image.open(left) as a, Image.open(right) as b:
            return "decoded pixels: " + array_difference(np.asarray(a), np.asarray(b))
    if left.suffix == ".obj":
        a, b = (MeshIO.read(path, allow_degenerate=True) for path in (left, right))
        return "; ".join(
            name
            + ": "
            + ("presence changed" if x is None or y is None else array_difference(x, y))
            for name, x, y in zip(("vertices", "faces", "uv"), a, b, strict=True)
            if x is not None or y is not None
        )
    if left.suffix == ".json":
        return "; ".join(
            differences(json.loads(left.read_text()), json.loads(right.read_text()))
        )
    return "file bytes changed"


def compare_outputs(left: Outputs, right: Outputs) -> list[str]:
    issues = []
    for key in sorted(left.hashes.keys() | right.hashes.keys()):
        if key not in left.files or key not in right.files:
            issues.append(f"{key}: missing output")
        elif left.hashes[key] != right.hashes[key]:
            issues.append(
                f"{key}: {file_difference(left.files[key], right.files[key])}"
            )
    issues += differences(left.atlas_metadata, right.atlas_metadata, "atlas_run")
    issues += differences(left.presentations, right.presentations, "presentations")
    issues += differences(left.completions, right.completions, "completions")
    return issues


def environment(root: Path) -> dict:
    data = json.loads((root / "environment.json").read_text())
    resumed = root / "environment-resume.json"
    if resumed.exists():
        newer = json.loads(resumed.read_text())
        # A resume may add profiler provenance, but may not change the experiment.
        for name in ("inputs_sha256", "platform", "python", "thread_environment"):
            if data[name] != newer[name]:
                raise ValueError(f"Resume changed environment field {name}")
        data.update(newer)
    supplement = root / "provenance.json"
    if supplement.exists():
        additions = json.loads(supplement.read_text())
        if set(additions) - {"blender", "lock_sha256"}:
            raise ValueError(
                "Provenance may only supplement Blender and lockfile evidence"
            )
        for name, value in additions.items():
            if name in data and data[name] != value:
                raise ValueError(f"Conflicting provenance for {name}")
        data.update(additions)
    required = (
        "platform",
        "python",
        "logical_cpus",
        "physical_cpus",
        "ram_bytes",
        "versions",
        "thread_environment",
        "inputs_sha256",
        "blender",
        "lock_sha256",
    )
    if any(key not in data for key in required):
        raise ValueError(f"Missing environment evidence: {set(required) - data.keys()}")
    actual = {
        str(p.relative_to(root / "inputs")): digest(p)
        for p in (root / "inputs").rglob("*")
        if p.is_file()
    }
    if actual != data["inputs_sha256"]:
        raise ValueError("Prepared input hashes disagree with environment record")
    return {**{key: data[key] for key in required}, "git_head": data["git_head"]}


def measurements(root: Path, runs: list[Outputs], backend: str) -> dict[str, float]:
    selected = [
        run
        for run in runs
        if run.run.config.backend == backend and run.run.config.mode == "baseline"
    ]
    if not selected:
        raise ValueError(f"No baseline runs for {backend}")
    spans = []
    for item in selected:
        path = root / item.run.config.output.name / "spans.jsonl"
        spans.append(
            [Span.model_validate_json(line) for line in path.read_text().splitlines()]
        )
    sessions = [
        next(span for span in group if span.kind == "session") for group in spans
    ]
    metrics = {
        "session_wall_s": statistics.median(s.wall_seconds for s in sessions),
        "application_cpu_s": statistics.median(s.process_cpu_seconds for s in sessions),
        "blender_cpu_s": statistics.median(s.children_cpu_seconds for s in sessions),
        "peak_rss_bytes": max(item.run.parent_peak_rss_bytes for item in selected),
        "tree_peak_rss_bytes": max(item.run.tree_peak_rss_bytes for item in selected),
        "retained_array_bytes": max(
            s.retained_array_bytes or 0 for group in spans for s in group
        ),
    }
    for pipeline in ("ATLAS and Texture Transfer", "Color processing"):
        metrics[pipeline + "_wall_s"] = statistics.median(
            sum(
                s.wall_seconds
                for s in group
                if s.kind == "pipeline" and s.name == pipeline
            )
            for group in spans
        )
    summary_path = root / "summary.json"
    if summary_path.exists():
        summary = Summary.model_validate_json(summary_path.read_text())
        if {run.config.output.name: run for run in summary.runs} != {
            item.run.config.output.name: item.run for item in runs
        }:
            raise ValueError("Summary is stale or belongs to different captures")
        key = f"{backend}-memory-1"
        if key in summary.tracked_heap_peaks:
            metrics["tracked_heap_peak_bytes"] = summary.tracked_heap_peaks[key]
        functions = summary.functions.get(f"{backend}-spans-1", {})
        for label, field in (
            ("obj_read_calls", "calls"),
            ("obj_read_bytes", "logical_read_bytes"),
        ):
            if "MeshIO._read_obj" in functions:
                metrics[label] = getattr(functions["MeshIO._read_obj"], field)
        for label, field in (
            ("obj_write_calls", "calls"),
            ("obj_write_bytes", "logical_write_bytes"),
        ):
            if "MeshIO.write_obj" in functions:
                metrics[label] = getattr(functions["MeshIO.write_obj"], field)
        if "Arrays.frozen" in functions:
            metrics["array_copy_bytes"] = functions["Arrays.frozen"].array_copy_bytes
    return metrics


def compare(reference: Path, candidate: Path) -> Comparison:
    reference, candidate = reference.resolve(), candidate.resolve()
    left_env, right_env = environment(reference), environment(candidate)
    left_revision, right_revision = (
        env.pop("git_head") for env in (left_env, right_env)
    )
    # Installation paths are host-specific; executable version/description matter.
    for env in (left_env, right_env):
        env["blender"].pop("executable", None)
    issues = differences(left_env, right_env, "environment")
    compatible = not issues
    suites = [
        [outputs(path.parent) for path in sorted(root.glob("*/run.json"))]
        for root in (reference, candidate)
    ]
    for root in (reference, candidate):
        if any(
            not (path.parent / "run.json").is_file()
            for path in root.glob("*/config.json")
        ):
            raise ValueError(f"Incomplete capture in {root}")
    if not all(suites):
        raise ValueError("Both profile directories must contain successful runs")
    output_issues = []
    metrics, alerts = {}, []
    for backend in ("native", "morphoweave"):
        reference_runs, candidate_runs = (
            [run for run in suite if run.run.config.backend == backend]
            for suite in suites
        )
        if not reference_runs or not candidate_runs:
            raise ValueError(f"Missing backend: {backend}")
        if {item.run.sample_interval_seconds for item in reference_runs} != {
            item.run.sample_interval_seconds for item in candidate_runs
        }:
            issues.append(f"{backend}: memory sampling intervals differ")
            compatible = False
        for item in [*reference_runs[1:], *candidate_runs]:
            output_issues.extend(
                f"{item.run.config.output.name}: {issue}"
                for issue in compare_outputs(reference_runs[0], item)
            )
        before, after = (
            measurements(root, suite, backend)
            for root, suite in zip((reference, candidate), suites, strict=True)
        )
        metrics[backend] = {}
        for name in sorted(before.keys() ^ after.keys()):
            alerts.append(f"{backend}/{name}: unavailable on one side of comparison")
        for name in sorted(before.keys() & after.keys()):
            a, b = before[name], after[name]
            percent = 100 * (a - b) / a if a else None
            metrics[backend][name] = MetricChange(
                reference=a, candidate=b, reduction=a - b, reduction_percent=percent
            )
            threshold = 10 if "bytes" in name else 5
            if percent is not None and percent < -threshold:
                alerts.append(f"{backend}/{name}: increased {-percent:.2f}%")
    return Comparison(
        reference=reference,
        candidate=candidate,
        reference_revision=left_revision,
        candidate_revision=right_revision,
        compatible_environment=compatible,
        identical_outputs=not output_issues,
        issues=issues + output_issues,
        investigation_alerts=alerts,
        metrics=metrics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.output.write_text(result.model_dump_json(indent=2) + "\n")
    print(
        f"Environment compatible: {result.compatible_environment}; identical outputs: {result.identical_outputs}"
    )
    for backend, values in result.metrics.items():
        for name in (
            "session_wall_s",
            "application_cpu_s",
            "peak_rss_bytes",
            "tracked_heap_peak_bytes",
        ):
            if name in values:
                value = values[name]
                print(
                    f"{backend} {name}: {value.reference:.3f} -> {value.candidate:.3f} ({value.reduction_percent:.2f}% reduction)"
                )
    for issue in result.issues + result.investigation_alerts:
        print(issue)
    if result.issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
