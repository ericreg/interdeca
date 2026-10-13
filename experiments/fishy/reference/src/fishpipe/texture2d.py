r"""Orientation-aware 2D texture descriptors (Gabor filter bank) on exterior images.

The graph-spectral band-energy descriptor is magnitude-only — it captures pattern *scale*
but not *orientation/directionality*, so it under-weights a directional/wavy pattern (e.g.
507450's green chevron rays). A **Gabor filter bank** convolves the image with kernels tuned
to (frequency x orientation), so its per-orientation energy profile DOES capture directional
texture. We run it on a clean exterior crop, on both the L* channel and the a* (green-red)
channel (so chromatic rays register), masking artifact/background black to avoid fake edges.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve
from skimage.color import rgb2lab
from skimage.filters import gabor_kernel
from skimage.transform import resize


def _prep_channel(ch: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid (black bg / artifact) pixels with the valid median, then z-score."""
    ch = ch.astype(np.float64)
    if valid.any():
        ch[~valid] = np.median(ch[valid])
    return (ch - ch.mean()) / (ch.std() + 1e-6)


def gabor_features(
    img_rgb: np.ndarray, size=(192, 96), freqs=(0.08, 0.18, 0.35), n_orient: int = 6,
    channels=("L", "a"), black_thresh: int = 18, return_orient_profile: bool = False,
):
    """Gabor energy descriptor of one exterior image.

    Returns a vector of mean filter magnitudes over (channel × frequency × orientation).
    With return_orient_profile, also returns the per-orientation energy (summed over freq,
    L+a) — a compact directionality signature.
    """
    small = resize(img_rgb.astype(np.float64) / 255.0, size, anti_aliasing=True)
    lab = rgb2lab(small)
    valid = (small.max(axis=2) * 255.0) >= black_thresh        # drop background/artifact black
    chan = {"L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2]}

    feats, orient_energy = [], np.zeros(n_orient)
    for cname in channels:
        ch = _prep_channel(chan[cname], valid)
        for f in freqs:
            for o in range(n_orient):
                theta = o * np.pi / n_orient
                k = gabor_kernel(f, theta=theta)
                re = convolve(ch, np.real(k), mode="reflect")
                im = convolve(ch, np.imag(k), mode="reflect")
                mag = float(np.sqrt(re ** 2 + im ** 2).mean())
                feats.append(mag)
                orient_energy[o] += mag
    feats = np.asarray(feats)
    if return_orient_profile:
        return feats, orient_energy
    return feats


def gabor_descriptor_set(images: list[np.ndarray], **kw) -> np.ndarray:
    """Stack gabor_features over a list of exterior images → (N, D)."""
    return np.vstack([gabor_features(im, **kw) for im in images])
