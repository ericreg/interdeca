"""Profile only the three small clams in fresh, sequential CLI processes."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import pstats
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from interdeca.blender import Blender
from interdeca.serde import BlenderInstallationData
from tools.profiling.capture import run_child
from tools.profiling.serde import Config, ProcessSample, Run, Sample

REPOSITORY = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(directory: Path) -> dict[str, str]:
    """Verify instrumentation preserved every exported atlas and presentation array.

    Session identities and manifest destination paths are deliberately omitted.
    Array keys follow presentation fields, so random artifact filenames cannot
    hide a changed result. Files are compared exactly, with no tolerance tuning.
    """
    events = [
        json.loads(line)
        for line in (directory / "cli/events.jsonl").read_text().splitlines()
    ]
    if any(event["type"] in ("failed", "cancelled") for event in events):
        raise RuntimeError(f"Application failure in {directory}")
    results = {
        event["action"]: event["payload"]
        for event in events
        if event["type"] == "result"
    }
    required = {
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
    if not required <= results.keys():
        raise RuntimeError(f"Missing presentations: {required - results.keys()}")
    atlas = Path(results["atlas"]["directory"])
    hashes = {
        "atlas/" + str(path.relative_to(atlas)): sha256(path)
        for path in atlas.rglob("*")
        if path.is_file() and path.name != "atlas_run.json"
    }

    def visit(value, key):
        if isinstance(value, dict):
            if "path" in value and "dtype" in value and "shape" in value:
                hashes[key] = sha256(directory / "cli" / value["path"])
            else:
                for name, child in value.items():
                    visit(child, key + "/" + name)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, key + "/" + str(index))

    visit(results, "presentations")
    return hashes


def snapshot(process: psutil.Process, full: bool) -> Sample:
    """Measure simultaneous RSS, not a sum of peaks from different moments.

    Summed RSS can double-count shared pages. USS is sampled less frequently
    because asking the OS to inspect private pages is more expensive.
    """
    rows = []
    try:
        processes = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        processes = []
    for item in processes:
        try:
            with item.oneshot():
                memory, cpu = item.memory_info(), item.cpu_times()
                uss = None
                if full:
                    try:
                        uss = item.memory_full_info().uss
                    except (psutil.AccessDenied, AttributeError, NotImplementedError):
                        pass
                rows.append(
                    ProcessSample(
                        pid=item.pid,
                        created=item.create_time(),
                        parent=item.pid == process.pid,
                        name=item.name(),
                        rss_bytes=memory.rss,
                        uss_bytes=uss,
                        user_seconds=cpu.user,
                        system_seconds=cpu.system,
                        threads=item.num_threads(),
                    )
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return Sample(time=time.perf_counter(), processes=rows)


def measure(config: Config, interval: float):
    config.output.mkdir()
    path = config.output / "config.json"
    path.write_text(config.model_dump_json(indent=2) + "\n")
    started = time.perf_counter()
    samples = []
    with (config.output / "console.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "tools.profiling", "--child", str(path)],
            cwd=REPOSITORY,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        monitored = psutil.Process(process.pid)
        try:
            with (config.output / "resources.jsonl").open("x") as stream:
                while process.poll() is None:
                    if time.perf_counter() - started > 2400:
                        raise TimeoutError(
                            f"Profiling exceeded 40 minutes: {config.output}"
                        )
                    before = time.perf_counter()
                    sample = snapshot(monitored, full=len(samples) % 5 == 0)
                    samples.append(sample)
                    stream.write(sample.model_dump_json() + "\n")
                    stream.flush()
                    time.sleep(max(0.001, interval - (time.perf_counter() - before)))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    elapsed = time.perf_counter() - started
    # Invalid profiler output must not leave a success marker that --resume
    # would trust. Application failures retain their ordinary failed run record.
    if process.returncode == 0:
        for path in config.output.glob("*.pstats"):
            stats = pstats.Stats(str(path))
            if any(v[2] < 0 or v[3] < 0 for v in stats.stats.values()):
                raise RuntimeError(f"Invalid negative profiler timings: {path}")
    parent_rss = [p.rss_bytes for s in samples for p in s.processes if p.parent]
    parent_uss = [
        p.uss_bytes
        for s in samples
        for p in s.processes
        if p.parent and p.uss_bytes is not None
    ]
    tree_uss = [
        sum(p.uss_bytes for p in s.processes)
        for s in samples
        if s.processes and all(p.uss_bytes is not None for p in s.processes)
    ]
    result = Run(
        config=config,
        exit_code=process.returncode,
        wall_seconds=elapsed,
        sample_interval_seconds=interval,
        parent_peak_rss_bytes=max(parent_rss, default=0),
        tree_peak_rss_bytes=max(
            (sum(p.rss_bytes for p in s.processes) for s in samples), default=0
        ),
        parent_peak_uss_bytes=max(parent_uss, default=None),
        tree_peak_uss_bytes=max(tree_uss, default=None),
        sampled_processes=len(
            {(p.pid, p.created) for s in samples for p in s.processes}
        ),
        output_hashes=fingerprint(config.output) if process.returncode == 0 else {},
    )
    (config.output / "run.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(
        f"{config.output.name}: exit={result.exit_code}, wall={elapsed:.1f}s, peak tree RSS={result.tree_peak_rss_bytes / 2**20:.0f} MiB",
        flush=True,
    )
    if process.returncode:
        raise RuntimeError(f"Profile failed; see {config.output / 'console.log'}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and preserve completed runs in an existing output directory.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Baseline repeats; order reverses on alternate repeats.",
    )
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("native", "morphoweave"),
        default=("native", "morphoweave"),
        help="Use both for acceptance; one backend is useful for budget experiments.",
    )
    parser.add_argument(
        "--cache-budget-mib",
        type=int,
        help="Experimental parsed-mesh budget; omit to measure the production default.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("baseline", "spans", "cpu", "memory"),
        default=("baseline", "spans", "cpu", "memory"),
    )
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        run_child(Config.model_validate_json(args.child.read_text()))
        return
    if args.output is None or args.repeats < 1 or not 0.01 <= args.interval <= 1:
        parser.error(
            "Supply a new --output directory, positive --repeats, and interval 0.01–1 seconds."
        )
    if args.cache_budget_mib is not None and args.cache_budget_mib < 0:
        parser.error("Cache budget must be nonnegative.")
    if len(set(args.backends)) != len(args.backends):
        parser.error("Specify each backend only once.")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    # Reuse the exact e2e fixture preparation, restricted to its three IDs. Input
    # conversion is outside every measurement and never changes reference files.
    if not args.resume:
        fixture = runpy.run_path(
            str(REPOSITORY / "tests/end_to_end/test_small_clams.py")
        )
        fixture["prepare_inputs"](root / "inputs")
    environment = {
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_bytes": psutil.virtual_memory().total,
        "blender": BlenderInstallationData.from_internal(
            Blender.check_available()
        ).model_dump(mode="json"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "vtk",
                "memray",
                "psutil",
                "yappi",
                "interdeca",
            )
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPOSITORY, text=True
        ),
        "lock_sha256": sha256(REPOSITORY / "uv.lock"),
        "source_sha256": {
            str(path.relative_to(REPOSITORY)): sha256(path)
            for folder in ("src", "client", "tools/profiling")
            for path in (REPOSITORY / folder).rglob("*.py")
        },
        "inputs_sha256": {
            str(p.relative_to(root / "inputs")): sha256(p)
            for p in (root / "inputs").rglob("*")
            if p.is_file()
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    if args.resume:
        previous = json.loads((root / "environment.json").read_text())
        if (
            previous["inputs_sha256"] != environment["inputs_sha256"]
            or previous["git_head"] != environment["git_head"]
            or previous.get("lock_sha256") != environment["lock_sha256"]
            or previous.get("source_sha256") != environment["source_sha256"]
        ):
            raise RuntimeError(
                "Cannot resume with changed inputs, sources, lockfile, or repository HEAD."
            )
        (root / "environment-resume.json").write_text(
            json.dumps(environment, indent=2) + "\n"
        )
    else:
        (root / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    expected = {}
    for path in root.glob("*/run.json"):
        completed = Run.model_validate_json(path.read_text())
        if completed.exit_code == 0:
            actual = fingerprint(completed.config.output)
            if actual != completed.output_hashes:
                raise RuntimeError(f"Previously completed outputs changed: {path}")
            if (
                completed.config.backend in expected
                and actual != expected[completed.config.backend]
            ):
                raise RuntimeError("Previously completed runs disagree.")
            expected[completed.config.backend] = actual
    for mode in args.modes:
        for repeat in range(args.repeats if mode == "baseline" else 1):
            backends = (
                args.backends if repeat % 2 == 0 else tuple(reversed(args.backends))
            )
            for backend in backends:
                config = Config(
                    backend=backend,
                    mode=mode,
                    inputs=root / "inputs",
                    output=root / f"{backend}-{mode}-{repeat + 1}",
                    cache_budget_bytes=(
                        None
                        if args.cache_budget_mib is None
                        else args.cache_budget_mib * 2**20
                    ),
                )
                if args.resume and (config.output / "run.json").exists():
                    existing = Run.model_validate_json(
                        (config.output / "run.json").read_text()
                    )
                    if existing.exit_code == 0 and existing.config == config:
                        continue
                print(f"Starting {config.output.name}", flush=True)
                result = measure(config, args.interval)
                if backend in expected and result.output_hashes != expected[backend]:
                    raise RuntimeError(
                        f"Instrumented outputs differ for {backend}; inspect {config.output}"
                    )
                expected[backend] = result.output_hashes
    print(
        f"All runs completed with identical outputs within each backend: {root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
