"""Structural / pattern descriptors: Endler color-adjacency + connected-component counts.

These operate on the **mesh face-adjacency graph** (UV-independent, position-invariant) using
a shared color quantization. They capture pattern STRUCTURE — how color patches border each
other and how many distinct patches there are — which is what separates e.g. 4- vs 5-stripe
fish. CPU-trivial; the adjacency graph + quantizer are built once on the shared mesh.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans

from . import config
from .data import FaceColorData, load_mesh
from .features import rgb_to_lab


@lru_cache(maxsize=2)
def face_adjacency() -> sp.csr_matrix:
    """Symmetric face-adjacency (faces sharing an edge) as a sparse 0/1 matrix. Cached."""
    cache = config.CACHE_DIR / "face_adjacency.npz"
    if cache.exists():
        return sp.load_npz(cache)
    mesh = load_mesh()
    fv = mesh.face_v                       # (Nf,3)
    nf = fv.shape[0]
    # map each undirected edge -> list of incident faces
    edges = {}
    for fi in range(nf):
        a, b, c = fv[fi]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edges.setdefault(key, []).append(fi)
    rows, cols = [], []
    for faces in edges.values():
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                rows += [faces[i], faces[j]]
                cols += [faces[j], faces[i]]
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nf, nf)).tocsr()
    A.data[:] = 1.0
    sp.save_npz(cache, A)
    return A


@lru_cache(maxsize=4)
def color_quantizer(n_colors: int = 6, seed: int = 0) -> np.ndarray:
    """KMeans centroids (Lab) over pooled population face colors → shared palette. Cached."""
    cache = config.CACHE_DIR / f"palette_lab_{n_colors}.npy"
    if cache.exists():
        return np.load(cache)
    from .data import build_face_colors

    fcd = build_face_colors()
    lab = rgb_to_lab(fcd.colors).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    idx = rng.choice(lab.shape[0], size=min(200_000, lab.shape[0]), replace=False)
    km = KMeans(n_clusters=n_colors, n_init=5, random_state=seed).fit(lab[idx])
    np.save(cache, km.cluster_centers_)
    return km.cluster_centers_


def _quantize(fcd: FaceColorData, palette: np.ndarray) -> np.ndarray:
    """Assign each face to nearest palette color → labels (N, Nf)."""
    from scipy.spatial.distance import cdist

    lab = rgb_to_lab(fcd.colors)                       # (N, Nf, 3)
    N = lab.shape[0]
    labels = np.empty((N, lab.shape[1]), dtype=np.int32)
    for s in range(N):
        labels[s] = cdist(lab[s], palette).argmin(axis=1)
    return labels


def endler_adjacency(
    fcd: FaceColorData, n_colors: int = 6, include_self: bool = False,
    area_weighted: bool = True,
) -> np.ndarray:
    """Endler color-transition-matrix descriptor → (N, n_colors*(n_colors+1)//2 [or full]).

    For each specimen build the symmetric color-class transition matrix over adjacent face
    pairs (optionally weighted by shared boundary ~ face areas), normalize, and flatten the
    upper triangle. Off-diagonal entries = how often two colors border each other; diagonal
    (if include_self) = within-patch homogeneity. Stripe count shows up as black↔base
    transition density. Position-invariant.
    """
    A = face_adjacency().tocoo()
    fi, fj = A.row, A.col
    keep = fi < fj                                     # each undirected edge once
    fi, fj = fi[keep], fj[keep]
    labels = _quantize(fcd, color_quantizer(n_colors))
    areas = fcd.areas
    w = np.sqrt(areas[fi] * areas[fj]) if area_weighted else np.ones(len(fi))

    N = labels.shape[0]
    iu = np.triu_indices(n_colors, k=0 if include_self else 1)
    out = np.zeros((N, len(iu[0])), dtype=np.float64)
    for s in range(N):
        ci, cj = labels[s, fi], labels[s, fj]
        M = np.zeros((n_colors, n_colors))
        lo = np.minimum(ci, cj)
        hi = np.maximum(ci, cj)
        np.add.at(M, (lo, hi), w)
        M = M + M.T - np.diag(np.diag(M))              # symmetric
        tot = M[iu].sum()
        if tot > 0:
            out[s] = M[iu] / tot
    return out


@lru_cache(maxsize=2)
def principal_axis() -> np.ndarray:
    """Unit vector of the mesh's longest axis (auto body axis), from face centroids."""
    from .data import build_face_colors

    cents = build_face_colors().centroids
    c = cents - cents.mean(0)
    _, _, vt = np.linalg.svd(c - c.mean(0), full_matrices=False)
    return vt[0]


def axial_banding_descriptor(
    fcd: FaceColorData, n_bins: int = 48, n_fft: int = 16, smooth: int = 2,
    dorsal_quantile: float | None = None, return_profile: bool = False,
):
    """General stripe/banding descriptor: |FFT| of darkness along the auto body axis.

    Projects face centroids onto the mesh principal axis, builds an area-weighted darkness
    (100-L*) profile, and returns its mean-removed power spectrum (shift-invariant → robust
    to longitudinal offset). Generalizes EXP-02 with NO hardcoded axis/region. Optionally
    restrict to the top `dorsal_quantile` fraction of the perpendicular axis to boost SNR.
    """
    from scipy.ndimage import uniform_filter1d

    cents = fcd.centroids
    axis = principal_axis()
    t = cents @ axis
    mask = np.ones(len(t), bool)
    if dorsal_quantile is not None:
        # use the secondary axis as "dorsal-ventral"; keep the high end
        c = cents - cents.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        perp = cents @ vt[1]
        thr = np.quantile(perp, 1 - dorsal_quantile)
        mask = perp >= thr

    edges = np.linspace(t[mask].min(), t[mask].max(), n_bins + 1)
    bid = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    areas = fcd.areas
    abin = np.bincount(bid[mask], weights=areas[mask], minlength=n_bins)
    abin[abin == 0] = 1.0

    L = rgb_to_lab(fcd.colors)[..., 0]
    dark = 100.0 - L
    N = L.shape[0]
    prof = np.zeros((N, n_bins))
    for i in range(N):
        prof[i] = np.bincount(bid[mask], weights=(areas * dark[i])[mask], minlength=n_bins) / abin
    prof = uniform_filter1d(prof, size=max(1, smooth), axis=1, mode="nearest")
    power = np.abs(np.fft.rfft(prof - prof.mean(axis=1, keepdims=True), axis=1))[:, 1:n_fft + 1]
    if return_profile:
        return power, prof
    return power


def component_counts(
    fcd: FaceColorData, n_colors: int = 6, min_area_frac: float = 0.002,
) -> np.ndarray:
    """Per-color connected-component counts on the face graph → (N, n_colors).

    Counts how many distinct surface patches each palette color forms (e.g. number of black
    stripe bands). Small components below `min_area_frac` of total area are dropped as noise.
    """
    A = face_adjacency()
    labels = _quantize(fcd, color_quantizer(n_colors))
    areas = fcd.areas
    tot_area = areas.sum()
    N, Nf = labels.shape
    out = np.zeros((N, n_colors), dtype=np.float64)
    for s in range(N):
        for c in range(n_colors):
            mask = labels[s] == c
            if not mask.any():
                continue
            sub = A[mask][:, mask]
            n_comp, comp = sp.csgraph.connected_components(sub, directed=False)
            # drop tiny components
            sub_area = areas[mask]
            good = 0
            for k in range(n_comp):
                if sub_area[comp == k].sum() / tot_area >= min_area_frac:
                    good += 1
            out[s, c] = good
    return out
