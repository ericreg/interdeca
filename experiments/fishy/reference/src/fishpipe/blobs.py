"""Blob (connected-component) analysis of a segmented color map on the mesh graph.

Once colors are segmented (segment.py), each palette color forms a set of connected regions
("blobs") on the face-adjacency graph. Counting / measuring these blobs turns the continuous,
jitter-blurred pattern into DISCRETE, topological features — e.g. "# of dark blobs" ≈ stripe
count (jitter-invariant), "is there a red blob" ≈ rosy cheeks. This is the biology-standard
(patternize/Endler) route the Slicer module's segmentation step sets up.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .data import FaceColorData
from .segment import Segmentation
from .structure import face_adjacency


def _blob_sizes(mask: np.ndarray, A: sp.csr_matrix, areas: np.ndarray, tot: float,
                min_area_frac: float) -> list[float]:
    if not mask.any():
        return []
    sub = A[mask][:, mask]
    ncomp, comp = connected_components(sub, directed=False)
    a = areas[mask]
    sizes = [a[comp == k].sum() / tot for k in range(ncomp)]
    return [s for s in sizes if s >= min_area_frac]


def blob_descriptor(
    fcd: FaceColorData, seg: Segmentation, min_area_frac: float = 0.001,
    region_mask: np.ndarray | None = None, mesh_adjacency: sp.csr_matrix | None = None,
) -> dict:
    """Per-specimen blob features for each palette color.

    Returns dict of arrays (each (N, K)) plus a stacked feature matrix:
      - count        : # blobs of each color (>= min_area_frac)
      - total_area   : surface fraction of each color
      - mean_size, max_size, size_entropy : blob size distribution per color
    region_mask: optional (Nf,) bool to restrict analysis (e.g. salient/dorsal region).
    """
    A = mesh_adjacency if mesh_adjacency is not None else face_adjacency()
    areas = fcd.areas
    if region_mask is not None:
        keep = np.where(region_mask)[0]
        A = A[keep][:, keep]
        areas = areas[keep]
        labels = seg.labels[:, keep]
    else:
        labels = seg.labels
    tot = areas.sum()
    N = labels.shape[0]
    K = seg.centroids_lab.shape[0]

    count = np.zeros((N, K)); tarea = np.zeros((N, K))
    msize = np.zeros((N, K)); xsize = np.zeros((N, K)); ent = np.zeros((N, K))
    for i in range(N):
        for c in range(K):
            sizes = _blob_sizes(labels[i] == c, A, areas, tot, min_area_frac)
            count[i, c] = len(sizes)
            if sizes:
                s = np.array(sizes)
                tarea[i, c] = s.sum(); msize[i, c] = s.mean(); xsize[i, c] = s.max()
                p = s / s.sum(); ent[i, c] = -(p * np.log(p + 1e-12)).sum()
    feats = np.concatenate([count, tarea, msize, xsize, ent], axis=1)
    return {"count": count, "total_area": tarea, "mean_size": msize, "max_size": xsize,
            "size_entropy": ent, "features": feats}


def endler_transitions(fcd: FaceColorData, seg: Segmentation,
                       mesh_adjacency: sp.csr_matrix | None = None,
                       area_weighted: bool = True) -> np.ndarray:
    """Per-specimen Endler color-transition matrix on the SEGMENTED map → (N, K*(K+1)/2).

    Counts how often each pair of palette colors borders across adjacent faces (boundary
    length ~ face areas). Captures pattern *structure* (e.g. dark↔base transition density)
    on a clean segmentation. Position-invariant; the biology-standard adjacency descriptor.
    """
    A = (mesh_adjacency if mesh_adjacency is not None else face_adjacency()).tocoo()
    fi, fj = A.row, A.col
    keep = fi < fj
    fi, fj = fi[keep], fj[keep]
    areas = fcd.areas
    w = np.sqrt(areas[fi] * areas[fj]) if area_weighted else np.ones(len(fi))
    K = seg.centroids_lab.shape[0]
    iu = np.triu_indices(K, 0)
    N = seg.labels.shape[0]
    out = np.zeros((N, len(iu[0])))
    for s in range(N):
        ci, cj = seg.labels[s, fi], seg.labels[s, fj]
        M = np.zeros((K, K))
        np.add.at(M, (np.minimum(ci, cj), np.maximum(ci, cj)), w)
        M = M + M.T - np.diag(np.diag(M))
        t = M[iu].sum()
        out[s] = M[iu] / t if t > 0 else M[iu]
    return out


def axial_dark_peaks(fcd: FaceColorData, seg: Segmentation, n_dark: int = 1, n_bins: int = 64,
                     axis_vec: np.ndarray | None = None, region_mask: np.ndarray | None = None,
                     return_profile: bool = False):
    """Segmented version of EXP-02: dark-fraction profile along the body axis -> peak count.

    Project faces onto the principal axis; per bin compute the area-fraction that is dark
    (in the darkest n_dark palette colors); smooth; count peaks. A *binary segmented* profile
    can give cleaner peaks than the raw-darkness profile. Uses the auto principal axis (no prior
    beyond 'a body axis exists')."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    from .structure import principal_axis

    axis = axis_vec if axis_vec is not None else principal_axis()
    t = fcd.centroids @ axis
    areas = fcd.areas
    mask = region_mask if region_mask is not None else np.ones(len(t), bool)
    edges = np.linspace(t[mask].min(), t[mask].max(), n_bins + 1)
    bid = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    abin = np.bincount(bid[mask], weights=areas[mask], minlength=n_bins); abin[abin == 0] = 1
    dark_set = seg.dark_order[:n_dark]
    N = seg.labels.shape[0]
    prof = np.zeros((N, n_bins)); counts = np.zeros(N)
    for i in range(N):
        isd = np.isin(seg.labels[i], dark_set).astype(float)
        prof[i] = np.bincount(bid[mask], weights=(areas * isd)[mask], minlength=n_bins) / abin
        sm = uniform_filter1d(prof[i], 3)
        pk, _ = find_peaks(sm, height=sm.mean() + 0.2 * sm.std(), distance=max(2, n_bins // 16))
        counts[i] = len(pk)
    if return_profile:
        return counts, prof
    return counts


def dark_blob_count(fcd: FaceColorData, seg: Segmentation, n_dark: int = 1,
                    min_area_frac: float = 0.001, region_mask: np.ndarray | None = None,
                    mesh_adjacency: sp.csr_matrix | None = None) -> np.ndarray:
    """# connected blobs in the darkest n_dark palette colors per specimen (≈ stripe count)."""
    A = mesh_adjacency if mesh_adjacency is not None else face_adjacency()
    areas = fcd.areas
    labels = seg.labels
    if region_mask is not None:
        keep = np.where(region_mask)[0]
        A = A[keep][:, keep]; areas = areas[keep]; labels = labels[:, keep]
    tot = areas.sum()
    dark = seg.dark_order[:n_dark]
    N = labels.shape[0]
    out = np.zeros(N)
    for i in range(N):
        mask = np.isin(labels[i], dark)
        out[i] = len(_blob_sizes(mask, A, areas, tot, min_area_frac))
    return out
