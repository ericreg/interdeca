"""Internal numerical records for the cichlid generator."""

import math
from numbers import Real
from typing import NamedTuple, Protocol, Self

import numpy as np

from experiments.fishy.records import (
    Parameters,
)


class LandmarkInput(Protocol):
    """The generator needs only this view of a core or research landmark record."""

    labels: tuple[str, ...]
    points: np.ndarray


class GeneratorData:
    """Validate direct numerical inputs without changing precision or ownership."""

    @staticmethod
    def finite(values: np.ndarray, name: str) -> np.ndarray:
        if (
            not isinstance(values, np.ndarray)
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"{name} must be a finite real numerical array.")
        return values

    @staticmethod
    def points(values: np.ndarray, name: str) -> np.ndarray:
        GeneratorData.finite(values, name)
        if values.ndim != 2 or values.shape[1] != 3 or not len(values):
            raise ValueError(f"{name} must have nonempty shape (N, 3).")
        return values

    @staticmethod
    def scalar(value: float, name: str) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite real number.")


class CanonicalFrame(NamedTuple):
    """Map world points to s*(P-pivot)@rotation.T + translation."""

    rotation: np.ndarray
    pivot: np.ndarray
    scale: float
    translation: np.ndarray

    def validated(self) -> Self:
        """Check frame dimensions and a finite positive scale before use."""
        for name in ("rotation", "pivot", "translation"):
            GeneratorData.finite(getattr(self, name), name)
        if (
            self.rotation.shape != (3, 3)
            or self.pivot.shape != (3,)
            or self.translation.shape != (3,)
        ):
            raise ValueError("Frame needs a 3x3 rotation and length-three vectors.")
        GeneratorData.scalar(self.scale, "scale")
        if self.scale <= 0:
            raise ValueError("Frame scale must be positive.")
        return self


class Spline(NamedTuple):
    """Coefficients of the generator's r² log(r+epsilon) spline."""

    controls: np.ndarray
    weights: np.ndarray
    affine: np.ndarray

    def validated(self) -> Self:
        """Check the finite coefficient dimensions of the augmented TPS system."""
        GeneratorData.points(self.controls, "spline controls")
        GeneratorData.finite(self.weights, "spline weights")
        GeneratorData.finite(self.affine, "spline affine component")
        if (
            len(self.controls) < 4
            or self.weights.shape != self.controls.shape
            or self.affine.shape != (4, 3)
        ):
            raise ValueError("Spline coefficients do not match the control points.")
        return self


# The author's brush kernel divides by sigma + 1e-12, so this named constant is the
# only shift used by both validation and painting. Signed and zero sigma are kept as
# drawn; only the exact sigma that makes this denominator zero is a singular brush.
KERNEL_EPSILON = 1e-12


class Brush(NamedTuple):
    """Sphere when endpoints coincide, capsule otherwise; radii are world units."""

    start: np.ndarray
    end: np.ndarray
    radius: float
    sigma: float
    color: tuple[float, float, float]
    strength: float

    def validated(self) -> Self:
        """Reject malformed brushes before distances and alpha are computed."""
        for name in ("start", "end"):
            value = GeneratorData.finite(getattr(self, name), name)
            if value.shape != (3,):
                raise ValueError("Brush endpoints must have length three.")
        for name in ("radius", "sigma", "strength"):
            GeneratorData.scalar(getattr(self, name), name)
        if (
            self.radius < 0
            or self.sigma + KERNEL_EPSILON == 0
            or not 0 <= self.strength <= 1
        ):
            raise ValueError("Brush radius/width/strength is outside its valid range.")
        if len(self.color) != 3:
            raise ValueError("Brush RGB color must have length three.")
        for value in self.color:
            GeneratorData.scalar(value, "brush RGB")
            if not 0 <= value <= 1:
                raise ValueError("Brush RGB color must lie in [0, 1].")
        return self


class SampledParameters(NamedTuple):
    """True generating mixture labels retained before color rendering."""

    parameters: Parameters
    belly_group: int
    tail_group: int
