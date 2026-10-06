"""Internal records used by the frozen surface-appearance experiment."""

from pathlib import Path
from typing import NamedTuple

import numpy as np


class Specimen(NamedTuple):
    """A mesh and its native texture paired by exact filename stem."""

    identifier: str
    mesh: Path
    texture: Path


class Frame(NamedTuple):
    """A camera frame and its deterministic PCA sign-selection diagnostics."""

    direction: np.ndarray
    up: np.ndarray
    skewness: np.ndarray
    score: float
