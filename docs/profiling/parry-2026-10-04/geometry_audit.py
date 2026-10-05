"""Compare every generated mesh with the preserved pre-Parry population output.

Run from the repository root with PYTHONPATH=. so the profiling helpers are
available. This is an off-timer numerical audit, not a performance measurement.
"""

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from tools.profiling.accuracy import atlas_directory
from tools.profiling.storage import read_mesh


class MeshDelta(NamedTuple):
    path: str
    vertex_count: int
    triangle_count: int
    identical_vertices: bool
    identical_faces: bool
    identical_uvs: bool
    maximum_vertex_distance: float
    rms_vertex_distance: float
    maximum_over_reference_diagonal: float
    collapse_classification_changes: int
    maximum_changed_twice_area_over_diagonal_squared: float


def collapse_mask(mesh):
    """Use the same relative collapse criterion as the independent accuracy pass."""
    triangles = mesh.vertices[mesh.faces]
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    area = np.linalg.norm(np.cross(first, second), axis=1)
    threshold = 32 * np.finfo(float).eps * np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return area <= threshold, area


def compare_meshes(reference: Path, candidate: Path, name: str) -> MeshDelta:
    """Compare corresponding indices directly, without fitting or reordering."""
    before = read_mesh(reference, allow_degenerate=True)
    after = read_mesh(candidate, allow_degenerate=True)
    if before.vertices.shape != after.vertices.shape:
        raise ValueError(f"Vertex shapes changed: {name}")
    faces_equal = np.array_equal(before.faces, after.faces)
    uv_equal = np.array_equal(before.uv, after.uv)
    if not faces_equal or not uv_equal:
        raise ValueError(f"Triangle order or per-corner UVs changed: {name}")
    # Indexwise differences expose any lost correspondence. A bounding-box
    # ratio makes the same result interpretable without assuming input units.
    distances = np.linalg.norm(after.vertices - before.vertices, axis=1)
    diagonal = float(np.linalg.norm(np.ptp(before.vertices, axis=0)))
    maximum = float(distances.max(initial=0))
    if diagonal <= 0:
        raise ValueError(f"Reference mesh has no spatial extent: {name}")
    identical_vertices = np.array_equal(before.vertices, after.vertices)
    changed_count, changed_area = 0, 0.0
    if not identical_vertices:
        # Nearly collapsed triangles can cross a floating-point classification
        # threshold. Record their physical area rather than hiding changed counts.
        old_mask, old_area = collapse_mask(before)
        new_mask, new_area = collapse_mask(after)
        changed = old_mask != new_mask
        changed_count = int(changed.sum())
        changed_area = float(max(old_area[changed].max(initial=0), new_area[changed].max(initial=0)))
    return MeshDelta(
        name, len(after.vertices), len(after.faces),
        identical_vertices, faces_equal, uv_equal,
        maximum, float(np.sqrt(np.mean(distances**2))), maximum / diagonal,
        changed_count, changed_area / diagonal**2,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads((args.reference / "environment.json").read_text())
    current = json.loads((args.candidate / "environment.json").read_text())
    if previous["inputs_sha256"] != current["inputs_sha256"]:
        raise ValueError("Prepared input hashes differ between populations.")
    specimen_ids = current["specimens"]
    if len(specimen_ids) != 47:
        raise ValueError("This audit requires the complete 47-specimen population.")
    evidence = {}
    for backend in ("native", "morphoweave"):
        before = atlas_directory(args.reference / f"{backend}-baseline-1")
        after = atlas_directory(args.candidate / f"{backend}-baseline-1")
        names = ["atlas.obj"] + [
            f"{folder}/{specimen}.obj"
            for specimen in specimen_ids for folder in ("aligned", "resampled")
        ]
        rows = []
        for number, name in enumerate(names, 1):
            rows.append(compare_meshes(before / name, after / name, name))
            if number % 10 == 0 or number == len(names):
                print(f"{backend}: audited {number}/{len(names)} meshes", flush=True)
        evidence[backend] = [row._asdict() for row in rows]
    # Only a complete audit gets the requested final filename. Restored meshes
    # are temporary; the verified capture archives remain unchanged.
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
