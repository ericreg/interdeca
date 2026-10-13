"""Internal numerical records for the cichlid generator."""

from typing import NamedTuple

import numpy as np

from experiments.fishy.records import (
    Parameters,
)


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
