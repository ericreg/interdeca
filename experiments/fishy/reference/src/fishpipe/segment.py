"""Color segmentation (quantization) on the mesh — faithful to the Slicer module.

The module quantizes each specimen's per-face colors into a small shared palette (KMeans in
Lab), optionally after **neighbor-averaging** (graph smoothing) which is what makes the
segmented regions clean enough to analyze as blobs. We reproduce that here so pattern analysis
can run on a *segmented* map (discrete colors) instead of raw continuous colors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans

from .data import FaceColorData
from .features import rgb_to_lab
from .structure import face_adjacency


def _avg_matrix(A: sp.csr_matrix) -> sp.csr_matrix:
    """Row-normalized self+neighbor averaging matrix (module's `_applyNeighborAveraging`)."""
    n = A.shape[0]
    M = (A + sp.identity(n, format="csr")).astype(np.float64)
    rs = np.asarray(M.sum(1)).ravel()
    rs[rs == 0] = 1.0
    return sp.diags(1.0 / rs) @ M


def neighbor_average(colors: np.ndarray, M: sp.csr_matrix, iters: int = 1) -> np.ndarray:
    out = colors.astype(np.float64)
    for _ in range(iters):
        out = M @ out
    return out


@dataclass
class Segmentation:
    labels: np.ndarray          # (N, Nf) int palette index per face
    centroids_lab: np.ndarray   # (K, 3) palette in Lab
    dark_order: np.ndarray      # palette indices sorted darkest->lightest (by L*)


def segment(
    fcd: FaceColorData, n_colors: int = 6, smooth_iters: int = 1, seed: int = 0,
    per_specimen: bool = False, mesh_adjacency: sp.csr_matrix | None = None,
) -> Segmentation:
    """Segment every specimen into `n_colors` palette colors (shared population palette).

    smooth_iters: graph neighbor-averaging passes before clustering (module uses 1).
    per_specimen: if True, each specimen gets its own KMeans, then palettes are matched to a
    shared reference by nearest Lab centroid (so labels are comparable across specimens).
    """
    A = mesh_adjacency if mesh_adjacency is not None else face_adjacency()
    M = _avg_matrix(A) if smooth_iters > 0 else None
    N, Nf, _ = fcd.colors.shape

    lab = rgb_to_lab(fcd.colors)                       # (N, Nf, 3)
    if smooth_iters > 0:
        rgb = fcd.colors.astype(np.float64)
        lab = np.stack([rgb_to_lab(np.clip(neighbor_average(rgb[i], M, smooth_iters), 0, 255)
                                   .astype(np.uint8)) for i in range(N)])

    rng = np.random.default_rng(seed)
    pooled = lab.reshape(N * Nf, 3)
    fit_idx = rng.choice(pooled.shape[0], size=min(200_000, pooled.shape[0]), replace=False)
    km = KMeans(n_colors, n_init=5, random_state=seed).fit(pooled[fit_idx])
    centroids = km.cluster_centers_

    labels = np.empty((N, Nf), dtype=np.int32)
    for i in range(N):
        if per_specimen:
            kmi = KMeans(n_colors, n_init=3, random_state=seed).fit(lab[i])
            # match this fish's centroids to the shared palette by nearest Lab
            d = np.linalg.norm(kmi.cluster_centers_[:, None] - centroids[None], axis=2)
            remap = d.argmin(axis=1)
            labels[i] = remap[kmi.labels_]
        else:
            labels[i] = km.predict(lab[i])

    dark_order = np.argsort(centroids[:, 0])           # darkest (low L*) first
    return Segmentation(labels=labels, centroids_lab=centroids, dark_order=dark_order)
