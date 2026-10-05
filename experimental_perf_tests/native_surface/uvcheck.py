"""Compare the source-derived Smart UV implementation with Blender 5.2.1.

References and input arrays are retained. A cached reference allows repeated
diagnostics without starting Blender again; it must match the input checksum.
"""

import argparse
import hashlib
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from interdeca._blender_protocol import BlenderUVRequest
from interdeca.blender import Blender
from interdeca.models import Mesh
from interdeca.surface import MeshProcessing
from tools.profiling.storage import read_mesh
from serde import UVComparison


class Case(NamedTuple):
    name: str
    vertices: np.ndarray
    faces: np.ndarray


def fixtures() -> list[Case]:
    """Exercise axis ties, folds, disconnected surfaces and non-manifold edges."""
    v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], float)
    f = np.array([[0,1,2],[0,2,3]], np.int64)
    tetra = np.array([[0,0,0],[1.1,0,.1],[.2,1.3,0],[.3,.4,1.5]])
    tf = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]], np.int64)
    angles = np.linspace(0, np.pi, 17)
    strip = np.array([[np.cos(a), np.sin(a), z] for z in (0., .25, .5) for a in angles])
    sf = np.array([t for row in range(2) for c in range(16)
        for t in ((row*17+c, row*17+c+1, (row+1)*17+c+1),
                  (row*17+c, (row+1)*17+c+1, (row+1)*17+c))], np.int64)
    degenerate = v.copy()
    degenerate[2] = degenerate[1]
    return [Case("square",v,f), Case("triangle",v,f[:1]),
        Case("disconnected", np.vstack([v,v]), np.vstack([f,f+4])),
        Case("tetrahedron",tetra,tf), Case("strip",strip,sf),
        Case("translated_strip",strip+[97.3,-26.9,4.2],sf),
        Case("nonmanifold",np.vstack([v,[.2,-.7,0]]),np.vstack([f,[0,2,4]])),
        Case("degenerate",degenerate,f)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/native-surface/source-uv"))
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--margin", type=float, default=.002)
    parser.add_argument("--backend", choices=("native","morphoweave"))
    parser.add_argument("--captures", type=Path, default=Path("build/profiling/parry-47-2026-10-04/current"))
    args = parser.parse_args()
    cases = fixtures()
    if args.backend:
        root = next((args.captures / f"{args.backend}-baseline-1" / "atlas").iterdir())
        mesh = read_mesh(root / "atlas.obj")
        cases = [Case(args.backend,mesh.vertices,mesh.faces)]
    if args.case:
        cases = [case for case in cases if case.name == args.case]
    if not cases:
        parser.error("No matching case.")
    args.output.mkdir(parents=True, exist_ok=True)
    installation = Blender.check_available() if not args.reference_dir else None
    rows = []
    for case in cases:
        root = args.output / case.name
        root.mkdir(exist_ok=True)
        mesh = Mesh(case.vertices, case.faces)
        digest = hashlib.sha256(case.vertices.tobytes()+case.faces.tobytes()+repr(args.margin).encode()).hexdigest()
        Blender._save_mesh(mesh, root / "input.npz", allow_degenerate=True)
        (root / "input.sha256").write_text(digest+"\n")
        reference = root / "blender.npy"
        blender_seconds = None
        if installation:
            config = BlenderUVRequest(str((root/"input.npz").resolve()),str(reference.resolve()),66,args.margin)
            start = time.perf_counter()
            Blender._run(installation,root.resolve(),config,timeout=600,threads=1)
            blender_seconds = time.perf_counter()-start
        else:
            reference = args.reference_dir / case.name / "blender.npy"
            if (reference.parent/"input.sha256").read_text().strip() != digest:
                raise ValueError("Cached UV reference settings/input differ.")
        expected = np.load(reference,allow_pickle=False)
        start = time.perf_counter()
        native = MeshProcessing.uv_layout(mesh,island_margin=args.margin)
        elapsed = time.perf_counter()-start
        np.savez(root/"native.npz",vertices=mesh.vertices,faces=mesh.faces,uv=native.uv,charts=native.chart_ids)
        row = UVComparison(case=case.name,faces=len(mesh.faces),charts=len(set(native.chart_ids)),margin=args.margin,
            native_seconds=elapsed,blender_seconds=blender_seconds,
            maximum_uv_error=float(np.abs(expected-native.uv).max()),
            exact_corners=int(np.all(expected==native.uv,axis=2).sum()),total_corners=len(mesh.faces)*3,
            input_sha256=digest,native_sha256=hashlib.sha256(native.uv.tobytes()).hexdigest(),
            reference_sha256=hashlib.sha256(expected.tobytes()).hexdigest())
        rows.append(row)
        print(row.model_dump_json(),flush=True)
        (args.output/"report.json").write_text("[\n"+",\n".join(r.model_dump_json(indent=2) for r in rows)+"\n]\n")


if __name__ == "__main__":
    main()
