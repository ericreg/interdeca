"""The recommended analysis: a color morphospace + a *measured* pattern view + auto-clusters.

Embodies the EXP-04 conclusion (discovery ≠ measurement):
  * COLOR factors → shown by an unsupervised color morphospace (PCA on color composition).
  * STRUCTURE factors (stripe count) → MEASURED directly with label-free detectors and shown
    as named axes (banding count / strength), because unsupervised reduction cannot surface
    low-variance structure.
  * RARE variants (rosy cheeks) → a chromatic-novelty score overlay.
  * Auto cluster coloring (GMM + BIC) so non-technical users SEE the segments.

All label-free (no ground truth used), CPU-only, general (auto body axis + auto pattern region).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .data import FaceColorData
from .features import area_hist, rgb_to_lab
from . import embed, structure


# ---------------------------------------------------------------------------
# Color morphospace (unsupervised; the existing, working path)
# ---------------------------------------------------------------------------
def color_morphospace(fcd: FaceColorData, n_clusters: int = 24, n_components: int = 4):
    """PCA on the area-weighted color-composition histogram → color morphospace scores."""
    X = area_hist(fcd, n_clusters=n_clusters, color_space="lab")
    return embed.pca(X, n_components=n_components, standardize=True)


# ---------------------------------------------------------------------------
# Label-free "salient pattern region": faces whose lightness varies most across
# the population — that is where the pattern lives (stripes shift specimen-to-
# specimen). A general, ground-truth-free replacement for a hand-drawn "dorsal" mask.
# ---------------------------------------------------------------------------
def salient_pattern_mask(fcd: FaceColorData, quantile: float = 0.85) -> np.ndarray:
    """Faces whose lightness varies most across the population — where a shifting pattern
    (stripes) lives. Label-free, general; isolates the patterned region without a hand mask.

    NOTE (EXP-06 audit): this quantile is the one real knob. Stripe recovery is rock-stable
    for quantile >= 0.80 (~0.89) but degrades to ~0.74 at 0.70 (a looser mask lets belly/tail
    region dilute the banding signal). Keep quantile >= 0.80; default 0.85 sits safely above
    the cliff."""
    L = rgb_to_lab(fcd.colors)[..., 0]            # (N, Nf)
    var = L.var(axis=0)                            # (Nf,) cross-specimen lightness variance
    return var >= np.quantile(var, quantile)


# ---------------------------------------------------------------------------
# Measured, label-free structural traits
# ---------------------------------------------------------------------------
def pattern_traits(fcd: FaceColorData, n_bins: int = 64, smooth: int = 3) -> dict:
    """Directly-measured banding traits along the auto body axis, within the salient region.

    Returns dict of per-specimen scalars (all label-free):
      - 'banding_count'    : # dark peaks (≈ stripe count)
      - 'banding_strength' : profile contrast (std of darkness profile)
      - 'dark_fraction'    : salient-region area fraction that is dark
      - 'profile'          : (N, n_bins) the darkness profiles (for plotting)
    """
    mask = salient_pattern_mask(fcd)
    axis = structure.principal_axis()
    t = fcd.centroids @ axis
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
    prof_s = uniform_filter1d(prof, size=max(1, smooth), axis=1, mode="nearest")

    count = np.zeros(N)
    for i in range(N):
        thr = prof_s[i].mean() + 0.2 * prof_s[i].std()
        pk, _ = find_peaks(prof_s[i], height=thr, distance=max(2, n_bins // 16))
        count[i] = len(pk)
    # shift-invariant banding spectrum (power in the stripe frequency band)
    fft = np.abs(np.fft.rfft(prof - prof.mean(axis=1, keepdims=True), axis=1))
    band_power = fft[:, 2:8]                        # (N, 6) stripe-scale bins
    return {
        "banding_count": count,                    # interpretable "# bands"
        "banding_strength": prof.std(axis=1),       # profile contrast
        "dark_fraction": (prof > 50).mean(axis=1),
        "band_power": band_power,
        "profile": prof_s,
        # compact pattern feature vector for the pattern morphospace (label-free, ~0.89 on stripe)
        "pattern_vector": np.column_stack(
            [prof.std(axis=1), (prof > 50).mean(axis=1), band_power]
        ),
    }


# ---------------------------------------------------------------------------
# Rare-variant chromatic novelty (cheeks-style) — label-free
# ---------------------------------------------------------------------------
def novelty_score(fcd: FaceColorData) -> np.ndarray:
    """Per-specimen chromatic novelty: max over faces of (a*,b*) distance from the per-face
    population median, area-aware. High for fish carrying a rare localized color (red cheeks)."""
    lab = rgb_to_lab(fcd.colors)
    ab = lab[..., 1:]                              # chroma only (ignore lightness/stripes)
    med = np.median(ab, axis=0)                    # (Nf, 2)
    d = np.linalg.norm(ab - med[None], axis=2)     # (N, Nf)
    # robust: 99th percentile of per-face novelty (a small vivid patch, not single-face noise)
    return np.percentile(d, 99, axis=1)


def rare_variant_descriptor(fcd: FaceColorData, k: int = 300, n_bands: int = 12) -> np.ndarray:
    """General rare-variant descriptor → (N, 3*n_bands): graph-spectral band energy of the
    chromatic channels (a*, b*, chroma). Best detector for tiny localized color anomalies
    (rosy cheeks ≈0.78 balanced acc). Use the scalar `novelty_score` for a simple overlay."""
    from . import spectral

    return spectral.spectral_descriptor(fcd, k=k, n_bands=n_bands, channels=("a", "b", "chroma"))


# ---------------------------------------------------------------------------
# Automatic cluster detection for coloring the morphospace
# ---------------------------------------------------------------------------
def auto_cluster(scores: np.ndarray, max_k: int = 6, random_state: int = 0) -> np.ndarray:
    """GMM with BIC model selection → integer cluster labels (for point coloring)."""
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(scores)
    best, best_bic = None, np.inf
    for k in range(1, max_k + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        gm.fit(Xs)
        bic = gm.bic(Xs)
        if bic < best_bic:
            best_bic, best = bic, gm
    return best.predict(Xs)
