"""Black-box cleanup characterization using stable labeled triangle fans."""

import argparse
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from interdeca.blender import Blender
from interdeca.models import AtlasModel, Landmarks, Mesh
from interdeca.surface import MeshProcessing
from serde import CleanupFixtureResult


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/native-surface/cleanup-fixtures.json"))
    args = parser.parse_args()
    installation = Blender.check_available()
    landmarks = Landmarks(np.eye(3), ("a", "b", "c"))
    records = []
    for positions in ([0, .75, 1.5], [1.5, .75, 0], [.75, 0, 1.5], [0, 1, 2], [2, 0, 1], [0, .2, .4, .6, .8, 1.0, 1.2]):
        vertices = np.array([[x, 0, 0] for x in positions] + [[0, 5, 0], [0, 0, 5]], float)
        n = len(positions)
        faces = np.array([[i, n, n+1] for i in range(n)], np.int64)
        mesh = Mesh(vertices, faces)
        expected = Blender.clean_atlas(AtlasModel(mesh, landmarks), installation, merge_distance=1).mesh
        observed = MeshProcessing.clean(mesh, merge_distance=1).mesh
        row = CleanupFixtureResult(positions=positions, blender_vertices=expected.vertices.tolist(),
            rust_vertices=observed.vertices.tolist(), blender_faces=expected.faces.tolist(),
            rust_faces=observed.faces.tolist())
        records.append(row)
        print(row.model_dump_json(), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(TypeAdapter(list[CleanupFixtureResult]).dump_json(records, indent=2) + b"\n")



if __name__ == "__main__":
    main()
