"""Decoupled texture representation: a shared 'texton' codebook over local surface patches.

The small-n insight: there are only ~25-250 specimens, but each carries tens of thousands
of mesh faces. So a texture VOCABULARY can be LEARNED from millions of pooled local patches
(huge effective sample size), and each specimen is then summarized as a *histogram / VLAD*
over that shared vocabulary. The per-specimen feature is a denoised, low-variance estimate
even though specimens are few — which is exactly what small-n clustering needs.

This is the classic textons / bag-of-visual-words / VLAD pipeline (Leung-Malik 2001;
Varma-Zisserman 2005; Jegou VLAD 2010), made mesh-native: the 'local patch' is a multi-scale
graph-diffusion jet of the per-face Lab signal (so it is shift-invariant and UV-distortion-proof,
computed on the shared mesh). It is a strict generalization of the team's color-segmentation +
Endler approach: a richer, soft-assigned, multi-scale codebook instead of a hard color KMeans.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from . import config
from .data import FaceColorData
from .features import rgb_to_lab


def diffusion_operator(cache_dir=None) -> sp.csr_matrix:
    """Row-normalized face-adjacency-with-self-loop P = D^-1 (A + I). One graph hop = local
    smoothing; P^s = scale-s smoothing. Cached adjacency is the shared mesh's face graph."""
    cdir = cache_dir or config.CACHE_DIR
    A = sp.load_npz(cdir / "face_adjacency.npz").tocsr()
    A = (A + sp.identity(A.shape[0], format="csr")).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    Dinv = sp.diags(1.0 / np.maximum(deg, 1e-12))
    return (Dinv @ A).tocsr()


def local_jet(fcd: FaceColorData, scales=(2, 4), P=None, cache_dir=None) -> np.ndarray:
    """Per-face multi-scale local descriptor → (N, Nf, D). Captures local colour + local
    contrast/edges (where stripes/spots live) at several spatial scales. CPU-cheap: a few
    sparse mat-vecs per specimen on the shared graph.

    D columns: [L, a, b, L_smooth, a_smooth, b_smooth, bandpass_L, multiscale_bandpass_L,
    local_L_std]  (smoothing at the largest scale; band-pass = signal minus smoothed = edges).
    """
    if P is None:
        P = diffusion_operator(cache_dir)
    lab = rgb_to_lab(fcd.colors).astype(np.float32)          # (N, Nf, 3)
    N, Nf, _ = lab.shape
    s_small, s_big = min(scales), max(scales)
    D = 9
    out = np.empty((N, Nf, D), dtype=np.float32)
    for i in range(N):
        s = lab[i]                                           # (Nf, 3)
        sm_small = s.copy()
        for _ in range(s_small):
            sm_small = P @ sm_small
        sm_big = sm_small.copy()
        for _ in range(s_big - s_small):
            sm_big = P @ sm_big
        L = s[:, 0]
        # local variance of L at the small scale: E[L^2]-E[L]^2 under P^s_small
        L2 = L * L
        eL = L.copy(); eL2 = L2.copy()
        for _ in range(s_small):
            eL = P @ eL; eL2 = P @ eL2
        local_std = np.sqrt(np.maximum(eL2 - eL * eL, 0.0))
        out[i, :, 0:3] = s
        out[i, :, 3:6] = sm_big
        out[i, :, 6] = L - sm_big[:, 0]                      # band-pass lightness (edges)
        out[i, :, 7] = sm_small[:, 0] - sm_big[:, 0]         # multiscale band-pass
        out[i, :, 8] = local_std
    return out


def build_codebook(jet: np.ndarray, K: int = 128, sample_per_spec: int = 4000, seed: int = 0):
    """Learn a shared texton codebook from patches pooled across ALL specimens.
    Returns (scaler, kmeans). Standardize columns first (heterogeneous units)."""
    rng = np.random.default_rng(seed)
    N, Nf, D = jet.shape
    pool = np.empty((N * sample_per_spec, D), dtype=np.float32)
    for i in range(N):
        idx = rng.choice(Nf, size=sample_per_spec, replace=False)
        pool[i * sample_per_spec:(i + 1) * sample_per_spec] = jet[i, idx]
    scaler = StandardScaler().fit(pool)
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=4096, n_init=3,
                         max_iter=200).fit(scaler.transform(pool))
    return scaler, km


def encode(fcd: FaceColorData, jet: np.ndarray, scaler, km, mode: str = "vlad") -> np.ndarray:
    """Per-specimen encoding over the shared codebook → (N, D_enc).
      mode='bow'  : area-weighted soft histogram of texton assignments (N, K), L1-normalized.
      mode='vlad' : area-weighted residual encoding (N, K*D), power+L2-normalized (richer).
    Area weighting makes the descriptor area-normalized and robust to face-count differences.
    """
    K = km.n_clusters
    N, Nf, D = jet.shape
    areas = fcd.areas.astype(np.float64)
    areas = areas / areas.sum()
    cents = km.cluster_centers_                              # (K, D) in standardized space
    if mode == "bow":
        X = np.zeros((N, K))
        for i in range(N):
            a = km.predict(scaler.transform(jet[i]))
            X[i] = np.bincount(a, weights=areas, minlength=K)
        X = X / (X.sum(1, keepdims=True) + 1e-12)
        return X
    # VLAD
    X = np.zeros((N, K * D))
    for i in range(N):
        Z = scaler.transform(jet[i])                        # (Nf, D)
        a = km.predict(Z)
        acc = np.zeros((K, D))
        for k in range(K):
            m = a == k
            if m.any():
                acc[k] = (areas[m, None] * (Z[m] - cents[k])).sum(0)
        v = acc.ravel()
        v = np.sign(v) * np.sqrt(np.abs(v))                 # power normalization
        n = np.linalg.norm(v)
        X[i] = v / n if n > 0 else v
    return X


def texton_descriptor(fcd: FaceColorData, K: int = 128, scales=(2, 4), mode: str = "vlad",
                      seed: int = 0, cache_dir=None) -> np.ndarray:
    """End-to-end: local jet → shared codebook → per-specimen VLAD/BoW. (N, D_enc)."""
    P = diffusion_operator(cache_dir)
    jet = local_jet(fcd, scales=scales, P=P, cache_dir=cache_dir)
    scaler, km = build_codebook(jet, K=K, seed=seed)
    return encode(fcd, jet, scaler, km, mode=mode)
