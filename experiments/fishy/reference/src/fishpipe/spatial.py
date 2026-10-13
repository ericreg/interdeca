"""Spatial statistics of a segmented color pattern — beyond blob counting.

Two patterns can have the same number of blobs but be arranged completely differently
(clustered vs regular vs linear, spread out vs concentrated, aligned vs isotropic). This
module describes, per palette color, *how that color is laid out on the surface*:

  - composition      : surface-area fraction
  - spread           : eigenvalues of the area-weighted 3-D covariance (extent per axis)
  - anisotropy       : elongation / directionality of the layout
  - axial layout     : mean position + spread + periodicity along the auto body axis
  - blob dispersion  : nearest-neighbour spacing of blob centroids (regular vs clustered)

These are general (no taxon prior), position/rotation-aware via the mesh geometry, and
distinguish "same count, different distribution".
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .data import FaceColorData
from .segment import Segmentation
from .structure import face_adjacency, principal_axis


def _weighted_cov(pts: np.ndarray, w: np.ndarray):
    W = w.sum()
    mu = (w[:, None] * pts).sum(0) / W
    d = pts - mu
    cov = np.einsum("i,ij,ik->jk", w, d, d) / W
    return mu, cov


def spatial_descriptor(
    fcd: FaceColorData, seg: Segmentation, region_mask: np.ndarray | None = None,
    n_axes: int = 3, blob_dispersion: bool = True, min_area_frac: float = 0.0005,
    fft_bins: int = 32, mesh_adjacency: sp.csr_matrix | None = None,
) -> np.ndarray:
    """Per-specimen spatial layout descriptor → (N, K * n_stats).

    Per palette color, concatenates: [area_frac, 3 spread eigvals, anisotropy, axial_mean,
    axial_spread, axial_fft_peakfreq, n_blobs, nn_dist_mean, nn_dist_cv].
    """
    A = mesh_adjacency if mesh_adjacency is not None else face_adjacency()
    cents = fcd.centroids
    areas = fcd.areas
    if region_mask is not None:
        keep = np.where(region_mask)[0]
        A = A[keep][:, keep]; cents = cents[keep]; areas = areas[keep]
        labels = seg.labels[:, keep]
    else:
        labels = seg.labels
    tot = areas.sum()
    axis = principal_axis()
    perp1 = _perp_axes(axis)
    t_all = cents @ axis
    N, Nf = labels.shape
    K = seg.centroids_lab.shape[0]
    rows = []
    for i in range(N):
        row = []
        for c in range(K):
            m = labels[i] == c
            a = areas[m]; W = a.sum()
            if W <= 0 or m.sum() < 2:
                row += [0.0] * 10
                continue
            pts = cents[m]
            mu, cov = _weighted_cov(pts, a)
            eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
            eig = np.sqrt(np.clip(eig, 0, None))
            aniso = float(eig[0] / (eig.sum() + 1e-12))
            tp = t_all[m]
            ax_mean = float((a * tp).sum() / W)
            ax_spread = float(np.sqrt(((a * (tp - ax_mean) ** 2).sum() / W)))
            # axial periodicity: area-weighted darkness-free histogram of this color along axis -> FFT peak
            edges = np.linspace(t_all.min(), t_all.max(), fft_bins + 1)
            h = np.histogram(tp, bins=edges, weights=a)[0]
            f = np.abs(np.fft.rfft(h - h.mean()))[1:]
            fft_peak = float(np.argmax(f) + 1) if f.size else 0.0
            n_blobs, nn_mean, nn_cv = 0.0, 0.0, 0.0
            if blob_dispersion:
                sub = A[m][:, m]
                ncomp, comp = connected_components(sub, directed=False)
                bc = []
                for k in range(ncomp):
                    bm = comp == k
                    if a[bm].sum() / tot >= min_area_frac:
                        bc.append((a[bm, None] * pts[bm]).sum(0) / a[bm].sum())
                n_blobs = float(len(bc))
                if len(bc) >= 2:
                    bc = np.array(bc)
                    D = np.linalg.norm(bc[:, None] - bc[None], axis=2)
                    np.fill_diagonal(D, np.inf)
                    nn = D.min(1)
                    nn_mean = float(nn.mean()); nn_cv = float(nn.std() / (nn.mean() + 1e-12))
            row += [W / tot, eig[0], eig[1], eig[2], aniso, ax_mean, ax_spread, fft_peak,
                    n_blobs, nn_cv]
        rows.append(row)
    return np.asarray(rows)


def _perp_axes(axis):
    a = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    p1 = ref - (ref @ a) * a; p1 /= np.linalg.norm(p1)
    return p1
