"""Compare source-derived cleanup and UV preparation against the captured atlas."""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from interdeca.blender import Blender
from interdeca.io import LandmarkIO
from interdeca.models import AtlasModel, Mesh
from interdeca.surface import MeshProcessing
from tools.profiling.storage import read_mesh
from serde import MeshReport


def oriented_faces(faces: np.ndarray) -> np.ndarray:
    """Canonicalize cyclic corner order without accepting reversed winding."""
    start = np.argmin(faces, axis=1)
    shifted = np.take_along_axis(faces, (start[:, None] + np.arange(3)) % 3, axis=1)
    return shifted[np.lexsort(shifted.T[::-1])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=Path("build/profiling/parry-47-2026-10-04/current"))
    parser.add_argument("--backend", default="native", choices=("native", "morphoweave"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("build/native-surface/meshcheck"))
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--reference-file", type=Path, help="Reuse a captured Blender cleanup .npz.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = next((args.captures / f"{args.backend}-baseline-1" / "atlas").iterdir())
    mesh = read_mesh(root / "atlas.obj")
    print("mesh", len(mesh.vertices), len(mesh.faces), flush=True)
    atlas = AtlasModel(mesh, LandmarkIO.read(root / "atlas.mrk.json"))
    start = time.perf_counter()
    result = MeshProcessing.clean(mesh)
    cleanup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    layout = MeshProcessing.uv_layout(mesh, resolution=(args.resolution, args.resolution))
    uv_seconds = time.perf_counter() - start
    np.savez(args.output / f"{args.backend}-{args.resolution}.npz", vertices=mesh.vertices,
             faces=mesh.faces, uv=layout.uv, charts=layout.chart_ids)
    uv = layout.uv
    area = np.abs((uv[:, 1, 0]-uv[:, 0, 0])*(uv[:, 2, 1]-uv[:, 0, 1])-(uv[:, 1, 1]-uv[:, 0, 1])*(uv[:, 2, 0]-uv[:, 0, 0])) / 2
    old = mesh.uv
    old_area = np.abs((old[:, 1, 0]-old[:, 0, 0])*(old[:, 2, 1]-old[:, 0, 1])-(old[:, 1, 1]-old[:, 0, 1])*(old[:, 2, 0]-old[:, 0, 0])) / 2
    row = MeshReport(backend=args.backend, resolution=args.resolution, vertices=len(mesh.vertices),
        triangles=len(mesh.faces), charts=int(layout.chart_ids.max() + 1),
        cleanup_seconds=cleanup_seconds, uv_seconds=uv_seconds,
        occupied_fraction=float(area.sum()), blender_occupied_fraction=float(old_area.sum()),
        subpixel_triangles=int(np.count_nonzero(area * args.resolution**2 < 1)))
    if args.reference or args.reference_file:
        if args.reference_file:
            with np.load(args.reference_file, allow_pickle=False) as stored:
                expected = Mesh(stored["vertices"], stored["faces"])
        else:
            expected = Blender.clean_atlas(atlas, Blender.check_available()).mesh
            np.savez(args.output / f"{args.backend}-blender-clean.npz", vertices=expected.vertices, faces=expected.faces)
        distance, mapping = cKDTree(expected.vertices).query(result.mesh.vertices)
        observed = oriented_faces(mapping[result.mesh.faces])
        reference = oriented_faces(expected.faces)
        row.cleanup_same_connectivity = bool(np.array_equal(observed, reference))
        row.cleanup_max_distance = float(distance.max())
        row.cleanup_coordinate_tolerance = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0))) * 1e-6
        row.cleanup_passed = (row.cleanup_same_connectivity and
            len(result.mesh.vertices) == len(expected.vertices) and
            len(set(mapping)) == len(mapping) and
            row.cleanup_max_distance <= row.cleanup_coordinate_tolerance)
    print(row.model_dump_json(), flush=True)
    (args.output / f"{args.backend}-{args.resolution}.json").write_text(row.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
