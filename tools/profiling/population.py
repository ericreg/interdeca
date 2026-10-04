"""Compare preserved and current implementations on an explicitly selected population.

No numerical code is edited. Each capture uses the recorder from its own reviewed
revision, in an isolated source directory and a fresh process. Both revisions use
the same uv interpreter, dependencies, converted inputs, and Blender executable.
"""

import argparse
import importlib.metadata
import io
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import psutil

from interdeca.blender import Blender
from interdeca.io import LandmarkIO, MeshIO
from interdeca.models import Mesh
from interdeca.serde import BlenderInstallationData
from tools.profiling.__main__ import REPOSITORY, measure, sha256
from tools.profiling.compare import compare, compare_outputs, outputs
from tools.profiling.serde import (
    Config,
    PopulationDataset,
    PopulationEnvironment,
    Run,
    Sample,
    Span,
    Summary,
)
from tools.profiling.summarize import (
    allocation_sites,
    function_costs,
    memory_plot,
    read_lines,
    task_summaries,
)


def clone(source: Path, destination: Path):
    """Make an independent copy, using APFS shared storage when available."""
    if sys.platform == "darwin":
        subprocess.run(["cp", "-c", str(source), str(destination)], check=True)
    else:
        shutil.copy2(source, destination)


def prepare(dataset: Path, destination: Path, expected: int, *, repair_zero_area=False):
    """Convert full-resolution inputs to RAS, with optional recorded zero-area repair.

    Repair affects only the private copy and never changes the vertex order or
    the order and corner UVs of remaining faces. Default validation stays strict.
    """
    ids = sorted(path.stem for path in (dataset / "models").glob("*.obj"))
    if len(ids) != expected:
        raise ValueError(f"Expected {expected} specimens; found {len(ids)}")
    # Every mesh must have exactly one landmark file and one texture. No silently
    # incomplete specimens, downsampling, or texture resizing enter this benchmark.
    sources = []
    for specimen in ids:
        textures = [
            path
            for path in (dataset / "textures").glob(specimen + ".*")
            if path.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg")
        ]
        landmark = dataset / "landmarks" / f"{specimen}.mrk.json"
        if len(textures) != 1 or not landmark.is_file():
            raise ValueError(f"Missing or ambiguous inputs for {specimen}")
        sources.extend([dataset / "models" / f"{specimen}.obj", landmark, textures[0]])
    for name in ("models", "landmarks", "textures"):
        (destination / name).mkdir(parents=True)
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    source_hashes, repairs = {}, {}
    for index, specimen in enumerate(ids):
        model, landmark, texture = sources[index * 3 : index * 3 + 3]
        for path in (model, landmark, texture):
            source_hashes[str(path.relative_to(dataset))] = sha256(path)
        mesh = MeshIO.read(
            model, coordinate_system="LPS", allow_degenerate=repair_zero_area
        )
        if repair_zero_area:
            # Repair only exactly zero-area raw faces in the private input copy.
            # Retain all vertices and the order/UVs of every remaining triangle;
            # numerically thin nonzero faces must still pass ordinary validation.
            triangles = mesh.vertices[mesh.faces]
            doubled = np.linalg.norm(
                np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                ),
                axis=1,
            )
            removed = np.flatnonzero(doubled == 0)
            if len(removed):
                keep = doubled != 0
                repairs[specimen] = removed.tolist()
                mesh = Mesh(
                    mesh.vertices,
                    mesh.faces[keep],
                    None if mesh.uv is None else mesh.uv[keep],
                ).validated()
                print(
                    f"Removed exactly zero-area input faces: {specimen} {removed.tolist()}",
                    flush=True,
                )
            else:
                mesh = mesh.validated()
        points = LandmarkIO.read(landmark)
        MeshIO.write_obj(mesh, destination / "models" / model.name)
        LandmarkIO.write_json(points, destination / "landmarks" / landmark.name)
        clone(texture, destination / "textures" / texture.name)
        low = np.minimum(low, points.points.min(axis=0))
        high = np.maximum(high, points.points.max(axis=0))
        print(f"Prepared {index + 1}/{len(ids)}: {specimen}", flush=True)
    # Use the same anatomical rectangle rule as the small-clams host parity test.
    first, second = np.argsort(high - low)[-2:]
    curve = np.tile((low + high) / 2, (4, 1))
    curve[:, first] = [
        low[first],
        (low[first] + high[first]) / 2,
        (low[first] + high[first]) / 2,
        low[first],
    ]
    curve[:, second] = [low[second], low[second], high[second], high[second]]
    np.save(destination / "curve.npy", curve, allow_pickle=False)
    return PopulationDataset(
        source=dataset,
        specimens=tuple(ids),
        source_sha256=source_hashes,
        removed_zero_area_faces=repairs,
    )


def snapshot_sources(revision: str, destination: Path):
    """Use immutable Git sources rather than checking out over the user's work."""
    content = subprocess.check_output(
        ["git", "archive", revision, "src", "client", "tools/profiling"],
        cwd=REPOSITORY,
    )
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(content)) as archive:
        archive.extractall(destination, filter="data")
    # The upstream pin is identical at both revisions and the adapter verifies its
    # source hash at import. Reuse its read-only files without duplicating the tree.
    vendor = destination / "src/interdeca/_vendor/morphoweave"
    vendor.rmdir()
    vendor.symlink_to(
        REPOSITORY / "src/interdeca/_vendor/morphoweave", target_is_directory=True
    )
    return {
        str(path.relative_to(destination)): sha256(path)
        for folder in ("src", "client", "tools/profiling")
        for path in (destination / folder).rglob("*.py")
    }


def environment(
    inputs: Path, revision: str, source_hashes: dict[str, str]
) -> PopulationEnvironment:
    """Record the actual common runtime, distinct from the application revision."""
    return PopulationEnvironment(
        platform=platform.platform(),
        python=sys.version,
        logical_cpus=psutil.cpu_count(),
        physical_cpus=psutil.cpu_count(logical=False),
        ram_bytes=psutil.virtual_memory().total,
        blender=BlenderInstallationData.from_internal(Blender.check_available()),
        versions={
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
        git_head=revision,
        lock_sha256=sha256(REPOSITORY / "uv.lock"),
        source_sha256=source_hashes,
        inputs_sha256={
            str(p.relative_to(inputs)): sha256(p)
            for p in inputs.rglob("*")
            if p.is_file()
        },
        thread_environment={
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    )


def compact(directory: Path, known: dict[str, Path]):
    """Share storage only after verifying identical bytes, outside timed captures.

    APFS clones have independent writable inodes. Captures still contain complete
    files, so rehashing and numerical inspection never depend on a deleted run.
    """
    for path in directory.rglob("*"):
        if not path.is_file() or path.stat().st_size < 1024**2:
            continue
        digest = sha256(path)
        previous = known.get(digest)
        if previous is None:
            known[digest] = path
        elif previous != path:
            temporary = path.with_name(path.name + ".clone")
            clone(previous, temporary)
            if sha256(temporary) != digest:
                raise ValueError(f"Clone changed bytes: {path}")
            temporary.replace(path)


def summarize(root: Path):
    """Reuse measured-data accounting without the small-clams report's fixed prose."""
    runs = [
        Run.model_validate_json(p.read_text()) for p in sorted(root.glob("*/run.json"))
    ]
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
            summary.tracked_heap_peaks[directory.name] = sum(
                site.bytes for site in sites
            )
            summary.peak_allocations[directory.name] = sites[:40]
    (root / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
    memory_plot(root, runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-specimens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        default="31fb5d9",
        help="B0 numerical code plus its reviewed recorder",
    )
    parser.add_argument("--candidate", default="3aa93df")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("baseline", "spans", "cpu", "memory"),
        default=("baseline", "spans", "cpu", "memory"),
    )
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--repair-zero-area-inputs",
        action="store_true",
        help="Record and remove exactly zero-area faces only in the private input copy",
    )
    args = parser.parse_args()
    if args.expected_specimens < 3 or args.repeats < 1:
        parser.error("Supply at least three specimens and one repeat")
    root, dataset = args.output.resolve(), args.dataset.resolve()
    revisions = {
        name: subprocess.check_output(
            ["git", "rev-parse", revision], cwd=REPOSITORY, text=True
        ).strip()
        for name, revision in (("B0", args.reference), ("current", args.candidate))
    }
    root.mkdir(parents=True, exist_ok=args.resume)
    inputs = root / "inputs"
    if not args.resume:
        manifest = prepare(
            dataset,
            inputs,
            args.expected_specimens,
            repair_zero_area=args.repair_zero_area_inputs,
        )
        # The captured action chain deliberately uses the same two specimens for
        # reconstruction/display in both small and full populations.
        if not {"UF_IZ_507781", "UF_IZ_507782"} <= set(manifest.specimens):
            raise ValueError("This action chain requires UF_IZ_507781 and UF_IZ_507782")
        (root / "dataset.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        for name, revision in revisions.items():
            sources = snapshot_sources(revision, root / "sources" / name)
            run_root = root / name
            run_root.mkdir()
            (run_root / "inputs").symlink_to(inputs, target_is_directory=True)
            record = environment(inputs, revision, sources)
            (run_root / "environment.json").write_text(
                record.model_dump_json(indent=2) + "\n"
            )
    else:
        manifest = PopulationDataset.model_validate_json(
            (root / "dataset.json").read_text()
        )
        if (
            manifest.source != dataset
            or len(manifest.specimens) != args.expected_specimens
        ):
            raise ValueError("Resume dataset identity changed")
        for relative, expected in manifest.source_sha256.items():
            if sha256(dataset / relative) != expected:
                raise ValueError(f"Source changed: {relative}")
        for name, revision in revisions.items():
            record = PopulationEnvironment.model_validate_json(
                (root / name / "environment.json").read_text()
            )
            sources = root / "sources" / name
            current_hashes = {
                relative: sha256(sources / relative)
                for relative in record.source_sha256
            }
            # Compare the saved identity, excluding private conversion caches in
            # nested core models that are not part of the serialized boundary.
            current = environment(inputs, revision, current_hashes)
            if record.model_dump(mode="json") != current.model_dump(mode="json"):
                raise ValueError(f"Resume environment or sources changed: {name}")
    # Preserve one byte-verified reference per implementation/backend. Exact
    # cross-version comparisons happen after captures, so a difference is reported
    # rather than silently replacing the original baseline.
    references, known = {}, {}
    for path in sorted(root.glob("*/*/run.json")):
        observed = outputs(path.parent)
        references.setdefault(
            (path.parents[1].name, observed.run.config.backend), observed
        )
        compact(path.parent, known)
    combinations = [
        (version, backend)
        for version in revisions
        for backend in ("native", "morphoweave")
    ]
    for mode in args.modes:
        for repeat in range(args.repeats if mode == "baseline" else 1):
            order = combinations if repeat % 2 == 0 else list(reversed(combinations))
            for version, backend in order:
                directory = root / version / f"{backend}-{mode}-{repeat + 1}"
                if (directory / "run.json").exists():
                    observed = outputs(directory)
                else:
                    if shutil.disk_usage(root).free < 8 * 2**30:
                        raise RuntimeError(
                            "Fewer than 8 GiB free before capture; retain evidence and free scratch space"
                        )
                    source = root / "sources" / version
                    child_env = {
                        **os.environ,
                        "PYTHONPATH": os.pathsep.join(
                            [str(source / "src"), str(source / "client/src")]
                        ),
                    }
                    config = Config(
                        backend=backend, mode=mode, inputs=inputs, output=directory
                    )
                    print(f"Starting {version}/{directory.name}", flush=True)
                    measure(
                        config,
                        0.1,
                        repository=source,
                        environment=child_env,
                        timeout=args.timeout,
                    )
                    observed = outputs(directory)
                key = (version, backend)
                if key in references:
                    issues = compare_outputs(references[key], observed)
                    if issues:
                        raise ValueError(
                            f"Repeat/instrumentation changed outputs: {issues}"
                        )
                else:
                    references[key] = observed
                compact(directory, known)
                print(
                    f"Verified {version}/{directory.name}: {len(observed.hashes)} fingerprints",
                    flush=True,
                )
    for version in revisions:
        summarize(root / version)
    result = compare(root / "B0", root / "current")
    (root / "comparison.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(
        f"Completed: identical_outputs={result.identical_outputs}, environment={result.compatible_environment}",
        flush=True,
    )
    if result.issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
