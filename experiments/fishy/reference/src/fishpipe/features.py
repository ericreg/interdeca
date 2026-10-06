"""Per-specimen feature builders.

Input is always a `FaceColorData` (per-face RGB colors + shared face areas/centroids).
Each builder returns a matrix X of shape (N_specimens, D) — one row per fish — that is
then fed to a dimensionality reducer (embed.py).

Two builders faithfully reproduce the Slicer module
(`InterDeCALogic.performPopulationAnalysis`):

* ``spatial_flatten`` — flattened per-face colors (the "subsampled flattened" path and,
  with subsample=None, the PCAMorphospace per-vertex path).
* ``area_hist`` — area-weighted color-cluster histogram (the "original" area-weighted
  path): quantize pooled colors into K Lab clusters, then per fish accumulate face area
  into the nearest cluster and unit-normalize.

The remaining builders are research variants tested as hypotheses.
"""
from __future__ import annotations

import numpy as np
from skimage import color as skcolor
from sklearn.cluster import KMeans

from .data import FaceColorData


# ---- color space helpers --------------------------------------------------
def rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 RGB -> (..., 3) float Lab (skimage, matches module)."""
    return skcolor.rgb2lab(rgb_u8.astype(np.float64) / 255.0)


def rgb_to_hsv_circular(rgb_u8: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 RGB -> (..., 4): [hue_cos, hue_sin, sat, val] (matches module HSV)."""
    hsv = skcolor.rgb2hsv(rgb_u8.astype(np.float64) / 255.0)
    h = hsv[..., 0] * 2 * np.pi
    return np.stack([np.cos(h), np.sin(h), hsv[..., 1], hsv[..., 2]], axis=-1)


# ---- module-faithful builders --------------------------------------------
def spatial_flatten(
    fcd: FaceColorData, color_space: str = "lab", subsample: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Flatten per-face colors into one vector per specimen.

    color_space: 'rgb' | 'lab' | 'hsv'. subsample: number of faces to keep (random,
    fixed across specimens) or None for all faces.
    """
    colors = fcd.colors  # (N, Nf, 3) uint8
    if subsample is not None and subsample < colors.shape[1]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(colors.shape[1], size=subsample, replace=False)
        idx.sort()
        colors = colors[:, idx, :]

    if color_space == "rgb":
        feat = colors.astype(np.float64) / 255.0
    elif color_space == "lab":
        feat = rgb_to_lab(colors)
    elif color_space == "hsv":
        feat = rgb_to_hsv_circular(colors)
    else:
        raise ValueError(color_space)
    return feat.reshape(colors.shape[0], -1)


def area_hist(
    fcd: FaceColorData, n_clusters: int = 16, color_space: str = "lab",
    unit_norm: bool = True, seed: int = 0, return_centroids: bool = False,
):
    """Area-weighted color-cluster histogram (module 'area-weighted' path).

    Pool all face colors across specimens, KMeans into n_clusters (in `color_space`),
    then per specimen sum face *area* into the nearest cluster; optionally unit-normalize.
    Returns (N, n_clusters).
    """
    N, Nf, _ = fcd.colors.shape
    if color_space == "lab":
        feat_all = rgb_to_lab(fcd.colors)                  # (N, Nf, 3)
    elif color_space == "rgb":
        feat_all = fcd.colors.astype(np.float64) / 255.0
    elif color_space == "hsv":
        feat_all = rgb_to_hsv_circular(fcd.colors)         # (N, Nf, 4)
    else:
        raise ValueError(color_space)

    pooled = feat_all.reshape(N * Nf, -1)
    # subsample pooled points for kmeans speed
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(pooled.shape[0], size=min(200_000, pooled.shape[0]), replace=False)
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=seed)
    km.fit(pooled[fit_idx])
    centroids = km.cluster_centers_

    areas = fcd.areas
    X = np.zeros((N, n_clusters), dtype=np.float64)
    for i in range(N):
        lab = km.predict(feat_all[i])                      # (Nf,) nearest cluster
        X[i] = np.bincount(lab, weights=areas, minlength=n_clusters)
    if unit_norm:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
    if return_centroids:
        return X, centroids
    return X


# ---- research variants ----------------------------------------------------
from functools import lru_cache

from . import config


@lru_cache(maxsize=8)
def _region_labels(n_regions: int, seed: int = 0) -> np.ndarray:
    """KMeans partition of the (shared) mesh faces by 3D centroid → region id per face.

    Cached on disk because it depends only on the mesh + n_regions (same for every fish).
    """
    cache = config.CACHE_DIR / f"region_labels_{n_regions}.npy"
    if cache.exists():
        return np.load(cache)
    cents = np.load(config.CACHE_DIR / "face_centroids.npy")
    km = KMeans(n_clusters=n_regions, n_init=4, random_state=seed)
    lab = km.fit_predict(cents).astype(np.int32)
    np.save(cache, lab)
    return lab


def region_summary(
    fcd: FaceColorData, n_regions: int = 128, stat: str = "mean",
    color_space: str = "lab",
) -> np.ndarray:
    """Per-region color summary → (N, n_regions * C).

    Partition the body into n_regions spatial regions (shared across specimens), and per
    region summarize that fish's face colors. `stat`:
      - 'mean'      : area-weighted mean color  (good for hue clusters; layout via regions)
      - 'maxchroma' : color of the highest-chroma face in the region (catches tiny vivid
                      spots like rosy cheeks — area-independent within region)
    Per-region summarization area-normalizes small regions (each region is one block of
    features regardless of its surface area), which is the lever for the minority class.
    """
    reg = _region_labels(n_regions)
    if color_space == "lab":
        feat_all = rgb_to_lab(fcd.colors)
    elif color_space == "hsv":
        feat_all = rgb_to_hsv_circular(fcd.colors)
    elif color_space == "rgb":
        feat_all = fcd.colors.astype(np.float64) / 255.0
    else:
        raise ValueError(color_space)
    N, Nf, C = feat_all.shape
    areas = fcd.areas

    out = np.zeros((N, n_regions, C), dtype=np.float64)
    if stat == "mean":
        # area-weighted mean per region
        wsum = np.bincount(reg, weights=areas, minlength=n_regions)
        wsum[wsum == 0] = 1.0
        for i in range(N):
            for c in range(C):
                num = np.bincount(reg, weights=areas * feat_all[i, :, c], minlength=n_regions)
                out[i, :, c] = num / wsum
    elif stat == "maxchroma":
        lab_all = rgb_to_lab(fcd.colors)
        chroma = np.sqrt(np.sum(lab_all[..., 1:] ** 2, axis=-1))      # (N,Nf)
        region_faces = [np.where(reg == r)[0] for r in range(n_regions)]
        for i in range(N):
            for r, fr in enumerate(region_faces):
                if fr.size:
                    out[i, r] = feat_all[i, fr[np.argmax(chroma[i, fr])]]
    else:
        raise ValueError(stat)
    return out.reshape(N, -1)


def population_novelty_map(fcd: FaceColorData) -> np.ndarray:
    """Per-face chromatic anomaly vs the per-face population, shape (N, Nf).

    Because every fish shares the mesh, each face has a population distribution of Lab
    colors across the 250 specimens. Novelty = Euclidean Lab distance from that face's
    population median. Localized rare anomalies (rosy-cheek red on 14/250 fish) score high
    on their faces; ordinary base-color noise scores low. General (no hardcoded palette).
    """
    lab = rgb_to_lab(fcd.colors)                       # (N, Nf, 3)
    med = np.median(lab, axis=0)                       # (Nf, 3) per-face population median
    return np.linalg.norm(lab - med[None], axis=2)     # (N, Nf)


def novelty_features(fcd: FaceColorData, thresholds=(15.0, 25.0, 40.0)) -> np.ndarray:
    """Compact rare-variant descriptor per specimen → (N, 2+len(thresholds)).

    Features: [max novelty, area-weighted mean novelty, area-fraction above each threshold].
    Designed so a tiny vivid anomaly (rosy cheeks) is detectable beside the morphospace.
    """
    nov = population_novelty_map(fcd)                  # (N, Nf)
    areas = fcd.areas
    tot = areas.sum()
    feats = [nov.max(axis=1), (nov * areas[None]).sum(axis=1) / tot]
    for t in thresholds:
        feats.append(((nov > t) * areas[None]).sum(axis=1) / tot)
    return np.column_stack(feats)


def stripe_profile(
    fcd: FaceColorData, axis: int = 1, n_bins: int = 64, smooth: int = 2,
) -> dict:
    """Longitudinal darkness profile and its spectrum (stripe-count estimator).

    Bin faces along a body axis (default Y), per bin take area-weighted darkness
    (100 - L*). Returns dict with:
      - 'profile'  (N, n_bins)        darkness vs position
      - 'fft'      (N, n_bins//2)     |FFT| of mean-removed profile (shift-invariant)
      - 'peakcount'(N,)               number of dark peaks (≈ stripe count)
    """
    cents = fcd.centroids[:, axis]
    edges = np.linspace(cents.min(), cents.max(), n_bins + 1)
    bin_id = np.clip(np.digitize(cents, edges) - 1, 0, n_bins - 1)
    areas = fcd.areas
    abin = np.bincount(bin_id, weights=areas, minlength=n_bins)
    abin[abin == 0] = 1.0

    L = rgb_to_lab(fcd.colors)[..., 0]                 # (N, Nf) lightness
    dark = 100.0 - L
    N = L.shape[0]
    prof = np.zeros((N, n_bins))
    for i in range(N):
        prof[i] = np.bincount(bin_id, weights=areas * dark[i], minlength=n_bins) / abin

    # smooth, count peaks
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    sm = uniform_filter1d(prof, size=max(1, smooth), axis=1, mode="nearest")
    fft = np.abs(np.fft.rfft(prof - prof.mean(axis=1, keepdims=True), axis=1))[:, 1:]
    peakcount = np.zeros(N, dtype=int)
    for i in range(N):
        thr = sm[i].mean() + 0.25 * sm[i].std()
        pk, _ = find_peaks(sm[i], height=thr, distance=max(2, n_bins // 12))
        peakcount[i] = len(pk)
    return {"profile": prof, "fft": fft, "peakcount": peakcount}
