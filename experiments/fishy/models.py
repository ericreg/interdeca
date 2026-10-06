"""Internal numerical records for the cichlid generator."""

from typing import NamedTuple

import numpy as np


class Parameters(NamedTuple):
    """One sampled parameter vector, in Alek's original column order."""

    base_color_hue: float
    base_color_sat: float
    base_color_val: float
    stripe_count: int
    stripe_spacing: float
    stripe_width: float
    stripe_longitudinal_offset: float
    belly_hue: float
    belly_strength: float
    belly_translation: float
    tail_hue: float
    tail_strength: float
    rosy_cheeks_present: int
    rosy_cheeks_hue: float
    rosy_cheeks_strength: float
    rosy_cheeks_translation_y: float
    rosy_cheeks_translation_z: float


class CanonicalFrame(NamedTuple):
    """Map world points to s*(P-pivot)@rotation.T + translation."""

    rotation: np.ndarray
    pivot: np.ndarray
    scale: float
    translation: np.ndarray


class Spline(NamedTuple):
    """Coefficients of the generator's r² log(r+epsilon) spline."""

    controls: np.ndarray
    weights: np.ndarray
    affine: np.ndarray


class Brush(NamedTuple):
    """Sphere when endpoints coincide, capsule otherwise; radii are world units."""

    start: np.ndarray
    end: np.ndarray
    radius: float
    sigma: float
    color: tuple[float, float, float]
    strength: float


class SampledParameters(NamedTuple):
    """True generating mixture labels retained before color rendering."""

    parameters: Parameters
    belly_group: int
    tail_group: int
