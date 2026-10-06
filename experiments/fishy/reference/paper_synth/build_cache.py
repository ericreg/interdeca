"""Build every cache the synthetic-cichlid paper analyses need. Run once."""
from __future__ import annotations

import json
import time

import numpy as np

import common as C


def main():
    t0 = time.time()
    print("=== 1. atlas (post-module) per-face colours ===")
    fcd, art, mesh = C.atlas_data()
    print(f"  colours {fcd.colors.shape}  faces={mesh.n_faces}  verts={mesh.vertices.shape[0]}")
    print(f"  artifact cells: {art.mean() * 100:.3f}%  "
          f"(specimens with >1% artifact: {(art.mean(1) > 0.01).sum()})")

    print("=== 2. ground truth join ===")
    df, labels = C.load_gt(fcd.names)
    for k, v in labels.items():
        print(f"  {k:8s}: {np.bincount(v)}")
    df.to_csv(C.RESULTS / "gt_ordered.csv", index=False)

    print("=== 3. face adjacency (atlas) ===")
    A = C.face_adjacency(mesh, C.ATLAS_DS.cache_dir)
    print(f"  adjacency nnz={A.nnz}")

    print("=== 4. native (pre-module) composition ===")
    nat = C.native_face_colors()
    print(f"  native hist {nat['hist'].shape}; order match: "
          f"{list(nat['names']) == list(fcd.names)}")

    print("=== 5. Laplacian eigenbasis (atlas, k=300) ===")
    from fishpipe import spectral
    w, U = spectral.laplacian_eigenbasis(300, mesh=mesh, cache_dir=C.ATLAS_DS.cache_dir)
    print(f"  eigvals[:5]={np.round(w[:5], 6)}  U={U.shape}")

    print("=== 6. per-vertex Lab (atlas) ===")
    vlab = spectral.vertex_lab(fcd, mesh=mesh, cache_dir=C.ATLAS_DS.cache_dir)
    print(f"  vertex_lab {vlab.shape}")

    meta = {
        "n_specimens": int(fcd.colors.shape[0]),
        "atlas_faces": int(mesh.n_faces),
        "atlas_verts": int(mesh.vertices.shape[0]),
        "artifact_frac": float(art.mean()),
        "seconds": round(time.time() - t0, 1),
    }
    (C.RESULTS / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
