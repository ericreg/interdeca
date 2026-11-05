"""Presentation records shared by hosts, containing no executable model objects."""

from pathlib import Path
from typing import NamedTuple

import numpy as np


class Mesh(NamedTuple):
    """Scene snapshot in world coordinates, retaining vertex, face, and corner order."""

    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray | None = None


class FaceColors(NamedTuple):
    """Display RGB values and a separate coverage mask in canonical face order."""

    rgb: np.ndarray
    valid: np.ndarray


class Selection(NamedTuple):
    """Canonical indices used only to draw a selection overlay."""

    vertices: np.ndarray
    faces: np.ndarray


class SelectionSummary(NamedTuple):
    """Counts for controls; the accumulated numerical selection stays in the worker."""

    vertices: int
    faces: int


class ExtractedRegion(NamedTuple):
    """Display mesh with source maps for copying MRML attributes."""

    mesh: Mesh
    original_vertices: np.ndarray
    original_faces: np.ndarray


class AtlasPresentation(NamedTuple):
    """Published atlas and persistent output paths, independent of the live session."""

    mesh: Mesh
    directory: Path
    has_textures: bool
    average: Path | None


class PopulationPresentation(NamedTuple):
    """Control metadata and an identifier for the worker-owned fit."""

    model_id: str
    method: str
    specimen_ids: tuple[str, ...]
    components: int
    representation: str
    can_reconstruct: bool


class MorphControls(NamedTuple):
    """Worker-computed slider positions and coordinates ready for display."""

    positions: tuple[float, float]
    coordinates: tuple[float, float]


class PlotData(NamedTuple):
    """Numerical plot data without estimator or feature-matrix state."""

    specimen_ids: tuple[str, ...]
    scores: np.ndarray
    method: str
    x_axis: int
    y_axis: int


class PresentationState(NamedTuple):
    """Last received control state; this is a view cache, not application authority."""

    revision: int = 0
    run_id: str | None = None
    action: str = ""
    status: str = "disconnected"
    completed: int = 0
    total: int = 0
    stage: int = 0
    stages: int = 0
    message: str = "Start the application worker."
    colors: str | None = None
    population: PopulationPresentation | None = None
    selection: SelectionSummary | None = None
    morph_active: bool = False
