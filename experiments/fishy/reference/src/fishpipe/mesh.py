"""Lightweight OBJ (v / vt / f) loader and texture sampling.

We deliberately avoid heavy mesh libraries: the fishy mesh is a plain triangulated
OBJ with per-corner UVs (``f v/vt/vn``). We need three things from it:

1. per-face 3D area  -> the "how much of the animal's surface" weighting the module
   uses for area-weighted color vectors (constant across specimens, same mesh).
2. per-face UVs       -> to sample each baked texture into a per-face color.
3. per-face 3D centroids -> for spatial/region-based feature engineering.

The module (`_calculateFaceAverageColors`) samples the texture at each vertex UV
(nearest pixel, V flipped) and averages the 3 corners per face. We reproduce that,
but sample each corner's *own* vt (more correct at UV seams; identical elsewhere).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray   # (Nv, 3) float64 world positions
    uvs: np.ndarray        # (Nvt, 2) float64 in [0,1]
    face_v: np.ndarray     # (Nf, 3) int, 0-based vertex indices
    face_vt: np.ndarray    # (Nf, 3) int, 0-based uv indices

    @property
    def n_faces(self) -> int:
        return self.face_v.shape[0]

    # --- derived geometry --------------------------------------------------
    def face_areas(self) -> np.ndarray:
        """3D triangle areas, shape (Nf,)."""
        p = self.vertices[self.face_v]            # (Nf, 3, 3)
        a = p[:, 1] - p[:, 0]
        b = p[:, 2] - p[:, 0]
        cross = np.cross(a, b)
        return 0.5 * np.linalg.norm(cross, axis=1)

    def face_centroids(self) -> np.ndarray:
        """3D centroids, shape (Nf, 3)."""
        return self.vertices[self.face_v].mean(axis=1)

    def face_corner_uvs(self) -> np.ndarray:
        """Per-face corner UVs, shape (Nf, 3, 2)."""
        return self.uvs[self.face_vt]


def load_obj(path: str | Path) -> Mesh:
    path = Path(path)
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    face_v: list[tuple[int, int, int]] = []
    face_vt: list[tuple[int, int, int]] = []

    with open(path, "r") as fh:
        for line in fh:
            if not line or line[0] not in "vf":
                continue
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "vt":
                uvs.append((float(parts[1]), float(parts[2])))
            elif tag == "f":
                # supports v/vt/vn ; triangulate fan if >3 corners
                vi, ti = [], []
                for token in parts[1:]:
                    chunks = token.split("/")
                    vi.append(int(chunks[0]) - 1)
                    ti.append(int(chunks[1]) - 1 if len(chunks) > 1 and chunks[1] else -1)
                for k in range(1, len(vi) - 1):
                    face_v.append((vi[0], vi[k], vi[k + 1]))
                    face_vt.append((ti[0], ti[k], ti[k + 1]))

    return Mesh(
        vertices=np.asarray(verts, dtype=np.float64),
        uvs=np.asarray(uvs, dtype=np.float64),
        face_v=np.asarray(face_v, dtype=np.int64),
        face_vt=np.asarray(face_vt, dtype=np.int64),
    )


def sample_face_colors(mesh: Mesh, texture: np.ndarray) -> np.ndarray:
    """Per-face average RGB by sampling a texture at the 3 corner UVs.

    Reproduces `InterDeCALogic._calculateFaceAverageColors` (nearest-pixel, V flipped,
    mean of corners). Returns uint8 (Nf, 3).
    """
    h, w = texture.shape[:2]
    corner_uv = mesh.face_corner_uvs()              # (Nf, 3, 2)
    u = np.clip(corner_uv[..., 0], 0.0, 1.0)
    v = np.clip(1.0 - corner_uv[..., 1], 0.0, 1.0)  # flip V
    px = np.clip((u * (w - 1)).astype(np.int64), 0, w - 1)
    py = np.clip((v * (h - 1)).astype(np.int64), 0, h - 1)
    corner_rgb = texture[py, px, :3].astype(np.float64)   # (Nf, 3, 3)
    face_rgb = corner_rgb.mean(axis=1)              # (Nf, 3)
    return np.clip(face_rgb, 0, 255).astype(np.uint8)
