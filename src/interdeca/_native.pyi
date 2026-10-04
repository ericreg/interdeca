"""Private native entry points; public file and schema adapters live in Python."""

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def read_obj(
    path: str | Path, *, coordinate_system: Literal["RAS", "LPS"] = "RAS",
    allow_degenerate: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64] | None]:
    """Return validated, independent, read-only vertices, faces, and corner UVs."""
    ...

def write_obj(
    path: str | Path, vertices: NDArray[np.float64], faces: NDArray[np.int64],
    uv: NDArray[np.float64] | None = None, *, overwrite: bool = False,
    allow_degenerate: bool = False,
) -> None:
    """Snapshot and validate the input, then stream an exact-round-trip OBJ."""
    ...

def texture_from_rgba8(
    pixels: NDArray[np.uint8],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize uint8 RGBA into independent, read-only float64 RGB and alpha."""
    ...

def texture_to_rgba8(
    rgb: NDArray[np.float64], alpha: NDArray[np.float64],
) -> NDArray[np.uint8]:
    """Validate unit-range channels and quantize with ties-to-even rounding."""
    ...

def texture_to_linear(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Decode the sRGB transfer curve with one final output allocation."""
    ...

def texture_from_linear(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clip to [0,1] and encode the sRGB transfer curve in one traversal."""
    ...
