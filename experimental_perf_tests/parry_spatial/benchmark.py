"""Compare VTK and Parry on preserved 47-specimen correspondence workloads.

Only construction and closest-surface queries are timed. Loading, warping,
verification, and report generation stay outside those intervals. Nothing in
the production package is patched or redirected to this prototype.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import NamedTuple

import interdeca_parry_spatial
import numpy as np
import psutil
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import reference, vtkPoints, vtkSMPTools
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkStaticCellLocator
from vtkmodules.vtkCommonTransforms import vtkThinPlateSplineTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter

from interdeca.geometry import Geometry
from interdeca.io import LandmarkIO, MeshIO
from interdeca.models import Landmarks, Mesh
from serde import Agreement, Experiment, Timing


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
METHODS = ("vtk_existing", "vtk_reuse", "parry_serial", "parry_parallel")


class Projection(NamedTuple):
    points: np.ndarray
    triangles: np.ndarray
    squared_distances: np.ndarray


class Measurement(NamedTuple):
    timing: Timing
    projection: Projection


class Workload(NamedTuple):
    mesh: Mesh
    queries: np.ndarray
    original_vertices: np.ndarray


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def vtk_points(array: np.ndarray) -> vtkPoints:
    result = vtkPoints()
    result.SetData(numpy_to_vtk(array, deep=True))
    return result


def prepare(backend: str, atlas: Mesh, landmarks: Landmarks, mesh: Mesh,
            subject: Landmarks) -> Workload:
    """Reproduce the coordinates presented to each backend's spatial search.

    Native uses the atlas landmarks as common space in the saved CLI workflow.
    MorphoWeave casts atlas queries to float32 before its VTK r-basis TPS; both
    engines then receive those same coordinates represented as float64.
    """
    subject = subject.reordered(landmarks.labels)
    if backend == "native":
        warped = Geometry._warp(mesh.vertices, subject, landmarks)
        queries = Geometry._warp(atlas.vertices, landmarks, landmarks)
        return Workload(Mesh(warped, mesh.faces).validated(), queries, mesh.vertices)
    # This is the upstream TPS sequence, with no approximation by SciPy's spline.
    transform = vtkThinPlateSplineTransform()
    transform.SetSourceLandmarks(vtk_points(landmarks.points))
    transform.SetTargetLandmarks(vtk_points(subject.points))
    transform.SetBasisToR()
    query_surface = vtkPolyData()
    query_surface.SetPoints(vtk_points(atlas.vertices.astype(np.float32)))
    warp = vtkTransformPolyDataFilter()
    warp.SetInputData(query_surface)
    warp.SetTransform(transform)
    warp.Update()
    queries = vtk_to_numpy(warp.GetOutput().GetPoints().GetData()).astype(np.float64)
    return Workload(Mesh(mesh.vertices, mesh.faces), queries, mesh.vertices)


def vtk_project(locator: vtkStaticCellLocator, queries: np.ndarray,
                *, reuse: bool) -> Projection:
    """The Native loop, with an additional scratch-reuse control for overhead.

    Both modes return the same three float64/int64 outputs as Parry. This is a
    search microbenchmark, not MorphoWeave's float32 vtkPoints output loop.
    """
    closest = np.empty_like(queries)
    cell_ids = np.empty(len(queries), dtype=np.int64)
    squared_distances = np.empty(len(queries))
    destination = [0.0, 0.0, 0.0]
    cell_id, sub_id, distance_squared = reference(0), reference(0), reference(0.0)
    for index, point in enumerate(queries):
        # Recreate the production scratch allocations, unless measuring the
        # inexpensive Python-only alternative of reusing those same containers.
        if not reuse:
            destination = [0.0, 0.0, 0.0]
            cell_id, sub_id, distance_squared = reference(0), reference(0), reference(0.0)
        locator.FindClosestPoint(point, destination, cell_id, sub_id, distance_squared)
        if int(cell_id) < 0:
            raise ValueError(f"VTK projection failed at query {index}")
        closest[index], cell_ids[index], squared_distances[index] = (
            destination, int(cell_id), float(distance_squared)
        )
    return Projection(closest, cell_ids, squared_distances)


def measure(work: Workload, method: str, threads: int, backend: str,
            specimen: str, repeat: int) -> Measurement:
    # Each observation owns a fresh index. Garbage collection and disposal are
    # outside timing; both engines include their mesh copies and output arrays.
    gc.collect()
    build_cpu = time.process_time()
    build_start = time.perf_counter()
    if method.startswith("vtk"):
        surface = MeshIO.to_polydata(work.mesh)
        index = vtkStaticCellLocator()
        index.SetDataSet(surface)
        index.BuildLocator()
    else:
        index = interdeca_parry_spatial.ParryIndex(
            work.mesh.vertices, work.mesh.faces,
            threads=threads if method == "parry_parallel" else 1,
        )
    build_seconds = time.perf_counter() - build_start
    build_cpu = time.process_time() - build_cpu
    query_cpu = time.process_time()
    query_start = time.perf_counter()
    if method.startswith("vtk"):
        projection = vtk_project(index, work.queries, reuse=method == "vtk_reuse")
    else:
        projection = Projection(*index.project(work.queries))
    query_seconds = time.perf_counter() - query_start
    query_cpu = time.process_time() - query_cpu
    return Measurement(Timing(
        backend=backend, specimen=specimen, repeat=repeat, method=method,
        vertices=len(work.mesh.vertices), triangles=len(work.mesh.faces),
        queries=len(work.queries), build_seconds=build_seconds,
        query_seconds=query_seconds, build_cpu_seconds=build_cpu,
        query_cpu_seconds=query_cpu,
    ), projection)


def recover(work: Workload, projection: Projection) -> np.ndarray:
    """Apply Native's existing barycentric recovery outside benchmark timing."""
    triangles = work.mesh.vertices[work.mesh.faces[projection.triangles]]
    first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    relative = projection.points - triangles[:, 0]
    dot00, dot01, dot11 = (np.sum(first * first, axis=1), np.sum(first * second, axis=1),
                         np.sum(second * second, axis=1))
    dot20, dot21 = np.sum(relative * first, axis=1), np.sum(relative * second, axis=1)
    determinant = dot00 * dot11 - dot01 * dot01
    if np.any(determinant <= np.finfo(float).eps * dot00 * dot11):
        raise ValueError("Native barycentric recovery rejected a very thin triangle")
    weight1 = (dot11 * dot20 - dot01 * dot21) / determinant
    weight2 = (dot00 * dot21 - dot01 * dot20) / determinant
    weights = np.clip(np.column_stack((1 - weight1 - weight2, weight1, weight2)), 0, 1)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ni,nij->nj", weights, work.original_vertices[work.mesh.faces[projection.triangles]])


def agreement(work: Workload, vtk: Projection, parry: Projection,
              parallel: Projection, backend: str, specimen: str) -> Agreement:
    # Triangle-ID differences alone are not failures: edges and vertices can
    # have several equally near incident triangles. Record them, never hide them.
    diagonal = float(np.linalg.norm(np.ptp(work.mesh.vertices, axis=0)))
    tolerance = 1e-9 * max(diagonal, 1.0)
    points = np.linalg.norm(vtk.points - parry.points, axis=1)
    distances = np.abs(np.sqrt(vtk.squared_distances) - np.sqrt(parry.squared_distances))
    assert all(np.isfinite(array).all() for array in parry)
    assert np.all((parry.triangles >= 0) & (parry.triangles < len(work.mesh.faces)))
    return Agreement(
        backend=backend, specimen=specimen, queries=len(work.queries),
        bbox_diagonal=diagonal, point_max_error=float(points.max()),
        point_rms_error=float(np.sqrt(np.mean(points**2))),
        distance_max_error=float(distances.max()),
        different_triangle_ids=int(np.count_nonzero(vtk.triangles != parry.triangles)),
        points_outside_tolerance=int(np.count_nonzero(points > tolerance)),
        distances_outside_tolerance=int(np.count_nonzero(distances > tolerance)),
        parallel_bitwise_equal=all(np.array_equal(a, b) for a, b in zip(parry, parallel)),
        recovered_native_max_error=(float(np.linalg.norm(recover(work, vtk) - recover(work, parry), axis=1).max())
                                    if backend == "native" else None),
        different_float32_output_points=int(np.count_nonzero(np.any(
            vtk.points.astype(np.float32) != parry.points.astype(np.float32), axis=1))),
    )


def verify() -> list[str]:
    """Small analytic checks guard against fast but semantically wrong queries."""
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    queries = np.array([[.2, .3, 2.], [.7, .7, 1.], [-1., -1., 0.]])
    expected = np.array([[.2, .3, 0.], [.5, .5, 0.], [0., 0., 0.]])
    for threads in (1, 2):
        index = interdeca_parry_spatial.ParryIndex(vertices, faces, threads)
        points, ids, distance2 = index.project(queries)
        np.testing.assert_allclose(points, expected, atol=1e-14)
        np.testing.assert_array_equal(ids, np.zeros(3, dtype=np.int64))
        np.testing.assert_allclose(distance2, np.sum((queries - expected)**2, axis=1))
        assert index.project(np.empty((0, 3)))[0].shape == (0, 3)
    # Interior points must project onto the boundary, not be returned unchanged
    # as they would be with Parry's solid-interior option enabled.
    tetra = np.vstack((vertices, [0., 0., 1.]))
    tetra_faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    index = interdeca_parry_spatial.ParryIndex(tetra, tetra_faces)
    point, _, distance = index.project(np.array([[.1, .2, .3]]))
    np.testing.assert_allclose(point, [[0., .2, .3]], atol=1e-14)
    np.testing.assert_allclose(distance, [.01], atol=1e-14)
    # Two disconnected triangles make an accidental BVH-leaf index distinguishable
    # from the original face index promised by the public prototype interface.
    separate = np.vstack((vertices, vertices + [0., 0., 10.]))
    index = interdeca_parry_spatial.ParryIndex(separate, np.array([[0, 1, 2], [3, 4, 5]]))
    np.testing.assert_array_equal(index.project(np.array([[.2, .3, 12.]]))[1], [1])
    for bad_vertices, bad_faces in (
        (vertices * np.nan, faces), (vertices, faces + 10),
        (vertices, np.empty((0, 3), dtype=np.int64)),
        (np.asfortranarray(vertices), faces),
        (np.ndarray((3, 3), dtype=np.float64, buffer=bytearray(73), offset=1), faces),
    ):
        try:
            interdeca_parry_spatial.ParryIndex(bad_vertices, bad_faces)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid mesh was accepted")
    return ["triangle interior, edge, vertex", "squared distance", "original triangle ID",
            "serial/parallel analytic answers", "empty query batch", "closed-mesh interior",
            "nonfinite vertices rejected", "out-of-range indices rejected", "empty mesh rejected",
            "Fortran-contiguous arrays rejected", "unaligned arrays rejected"]


def report(experiment: Experiment, output: Path) -> None:
    lines = ["# Parry spatial-search experiment", "",
             "Seconds for all specimens in each workload; median of repeat totals.", "",
             "| Workload | Method | Build | Query | Build + query | Speedup |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for backend in sorted({item.backend for item in experiment.timings}):
        totals = {}
        for method in METHODS:
            passes = [[item for item in experiment.timings if item.backend == backend
                       and item.method == method and item.repeat == repeat]
                      for repeat in range(experiment.repeats)]
            builds = [sum(item.build_seconds for item in items) for items in passes]
            queries = [sum(item.query_seconds for item in items) for items in passes]
            totals[method] = statistics.median(a + b for a, b in zip(builds, queries))
            lines.append(f"| {backend} | {method} | {statistics.median(builds):.3f} | "
                         f"{statistics.median(queries):.3f} | {totals[method]:.3f} | "
                         f"{totals['vtk_existing'] / totals[method]:.2f}× |")
    lines += ["", "Build/query medians are calculated independently; their sum can differ "
              "slightly from the median combined time.", "",
              "| Workload | Specimens | Queries per repeat | Different triangle IDs | "
              "Max point error | Max distance error | Points beyond tolerance |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for backend in sorted({item.backend for item in experiment.agreement}):
        items = [item for item in experiment.agreement if item.backend == backend]
        lines.append(f"| {backend} | {len(items)} | {sum(x.queries for x in items):,} | "
                     f"{sum(x.different_triangle_ids for x in items):,} | "
                     f"{max(x.point_max_error for x in items):.6g} | "
                     f"{max(x.distance_max_error for x in items):.6g} | "
                     f"{sum(x.points_outside_tolerance for x in items):,} |")
    lines += ["", f"Repeats: {experiment.repeats}. Parallel Parry workers: "
              f"{experiment.parallel_threads}. Serial and parallel results bitwise equal: "
              f"{all(x.parallel_bitwise_equal for x in experiment.agreement)}.", "",
              "Tolerance: 1e-9 × max(surface bounding-box diagonal, 1), in input length units. "
              "Different closest points can still have equal minimum distances at surface ties. "
              "No exact triangle-ID equivalence is assumed.", "",
              f"Maximum recovered Native-coordinate difference: "
              f"{max((x.recovered_native_max_error or 0 for x in experiment.agreement), default=0):.6g}. "
              f"MorphoWeave-workload projected points that differ after float32 rounding: "
              f"{sum(x.different_float32_output_points for x in experiment.agreement if x.backend == 'morphoweave'):,}.", "",
              "The VTK baseline reproduces the Native query loop on both query workloads. "
              "vtk_reuse reuses its scratch objects. These are not timings of complete "
              "MorphoWeave correspondence or its vtkPoints marshalling loop.", "",
              "Build includes mesh copying/conversion, bounds checks in Parry, index construction, "
              "and thread-pool creation for parallel Parry. Query includes snapshots, finite "
              "checks, searches, and NumPy result construction. OBJ reading, spline warping, "
              "verification, garbage collection, and index destruction are excluded. "
              "No barycentric reconstruction or mesh validation after projection is timed.", "",
              "VTK uses the Sequential SMP backend. Parry index construction is serial; "
              "only its explicitly labelled parallel query mode uses multiple workers. "
              "Each observation rebuilds the index. Meshes are processed sequentially, with "
              "three repeated observations by default and method order rotated by specimen "
              "and repeat. These are warm-process microbenchmarks, not fresh-process pipeline runs.", "",
              "Production InterDeCA and vendored MorphoWeave are unchanged. Parry is an "
              "experimental optional extension, not a new application dependency.", "",
              f"Platform: {experiment.platform}; {experiment.processor}; "
              f"physical/logical CPUs: {experiment.physical_cpus}/{experiment.logical_cpus}.", "",
              "See results.json for per-specimen timings, CPU time, versions, input hashes, "
              "source hashes, invocation, and agreement measurements.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=Path.home() / "Library/Application Support/InterDeCA/full-population-2026-10-03/current")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--backend", choices=("native", "morphoweave", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Counts must be positive")
    # Fix VTK's build parallelism so the serial comparison has an explicit core
    # budget; BLAS thread limits are supplied before Python starts (see README).
    vtkSMPTools.SetBackend("Sequential")
    warnings.filterwarnings("ignore", message=".*SetCells.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*vtkTransformPolyDataFilter.*", category=DeprecationWarning)
    cases = verify()
    if args.verify_only:
        print("Verified: " + "; ".join(cases))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    binary = Path(interdeca_parry_spatial.__file__)
    if binary.suffix == ".py":
        binary = next(binary.parent.glob("*.so"))
    files = [ROOT / "src/lib.rs", ROOT / "Cargo.lock", ROOT / "Cargo.toml",
             ROOT / "benchmark.py", ROOT / "serde.py", binary,
             REPOSITORY / "src/interdeca/geometry.py", REPOSITORY / "src/interdeca/io.py"]
    experiment = Experiment(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        platform=platform.platform(), python=sys.version,
        processor=platform.processor(), physical_cpus=psutil.cpu_count(logical=False),
        logical_cpus=psutil.cpu_count(),
        versions={**{name: importlib.metadata.version(name) for name in ("numpy", "vtk", "scipy", "maturin")},
                  "parry3d-f64": interdeca_parry_spatial.PARRY_VERSION,
                  "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip()},
        thread_environment={name: os.environ.get(name) for name in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")},
        implementation_sha256={str(path): sha256(path) for path in files},
        command=sys.argv, captures=str(args.captures.resolve()), repeats=args.repeats,
        parallel_threads=args.threads, verification_cases=cases,
    )
    backends = ("native", "morphoweave") if args.backend == "both" else (args.backend,)
    for backend in backends:
        runs = list((args.captures / f"{backend}-baseline-1/atlas").glob("interdeca_*"))
        if len(runs) != 1:
            raise ValueError(f"Expected one preserved atlas run for {backend}, found {runs}")
        run = runs[0]
        atlas_path, landmarks_path = run / "atlas.obj", run / "atlas.mrk.json"
        atlas = MeshIO.read(atlas_path, allow_degenerate=True)
        landmarks = LandmarkIO.read(landmarks_path)
        for path in (atlas_path, landmarks_path):
            experiment.source_sha256[str(path)] = sha256(path)
        sources = sorted((run / "aligned").glob("*.obj"))
        if len(sources) != 47:
            raise ValueError(f"Expected the full 47-specimen capture, found {len(sources)}")
        sources = sources[:args.limit]
        for number, source in enumerate(sources):
            landmark_path = source.with_suffix(".mrk.json")
            for path in (source, landmark_path):
                experiment.source_sha256[str(path)] = sha256(path)
            work = prepare(backend, atlas, landmarks, MeshIO.read(source), LandmarkIO.read(landmark_path))
            for repeat in range(args.repeats):
                shift = (number + repeat) % len(METHODS)
                projections = {}
                for method in METHODS[shift:] + METHODS[:shift]:
                    result = measure(work, method, args.threads, backend, source.stem, repeat)
                    experiment.timings.append(result.timing)
                    projections[method] = result.projection
                for a, b in zip(projections["vtk_existing"], projections["vtk_reuse"]):
                    np.testing.assert_array_equal(a, b)
                checked = agreement(work, projections["vtk_existing"], projections["parry_serial"],
                                    projections["parry_parallel"], backend, source.stem)
                if not checked.parallel_bitwise_equal:
                    raise AssertionError("Parallel queries changed the serial result")
                if repeat == 0:
                    experiment.agreement.append(checked)
            (args.output / "results.json").write_text(experiment.model_dump_json(indent=2) + "\n")
            print(f"{backend} {number + 1}/{len(sources)} {source.stem}: "
                  f"{len(work.mesh.faces):,} triangles, {len(work.queries):,} queries; "
                  f"max point error {checked.point_max_error:.3g}, "
                  f"different triangle IDs {checked.different_triangle_ids}", flush=True)
    report(experiment, args.output)
    print((args.output / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
