"""Session records and validated user requests shared by every presentation host."""

from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np

from interdeca.models import (
    AnalysisGeometry,
    AtlasModel,
    BakedTexture,
    ClusterAlignment,
    Mesh,
    PopulationFeatures,
    PopulationModel,
    ProcessedColors,
    RegionSelection,
)


class AtlasOptions(NamedTuple):
    """Legacy defaults in model units; cleanup precedes final correspondence."""

    models: Path
    landmarks: Path
    output: Path
    textures: Path | None = None
    atlas_model: Path | None = None
    atlas_landmarks: Path | None = None
    blender: str | None = None
    merge_distance: float = 0.0001
    uv_angle: float = 66.0
    island_margin: float = 0.002
    bake_size: int = 2048
    extrusion: float = 0.001
    bake_margin: int = 2


class ColorOptions(NamedTuple):
    """Sampling and processing settings independent of any selected scene node."""

    directory: Path
    clustering: bool = True
    normalize: bool = False
    initial_clusters: int = 24
    clusters: int = 8
    sample_count: int | None = 10000
    neighbor_average: bool = False


class ModelOptions(NamedTuple):
    """A population fit request and its explicit feature representation."""

    method: Literal["pca", "ica", "umap"] = "pca"
    components: int = 2


class AtlasResult(NamedTuple):
    """Published outputs from one isolated atlas run."""

    atlas: AtlasModel
    directory: Path
    textures: tuple[BakedTexture, ...]
    average: Path | None


class ColorResult(NamedTuple):
    """Retain samples and settings so visualization does not repeat analysis."""

    analysis: AnalysisGeometry
    specimens: tuple[ProcessedColors, ...]
    alignment: ClusterAlignment | None
    options: ColorOptions


class PopulationResult(NamedTuple):
    """Live fitted model and the feature identity needed for reconstruction."""

    features: PopulationFeatures
    model: PopulationModel


class ApplicationInputs(NamedTuple):
    """Keep the last submitted inputs available to any future presentation host."""

    atlas: AtlasOptions | None = None
    colors: ColorOptions | None = None
    population: ModelOptions | None = None
    color_mesh: Mesh | None = None
    selection_mesh: Mesh | None = None
    curve_points: np.ndarray | None = None
    same_side_only: bool = True


class ApplicationState(NamedTuple):
    """Immutable session snapshot; no Qt, MRML, or worker references are stored."""

    revision: int = 0
    run_id: str | None = None
    action: str = ""
    status: str = "idle"
    completed: int = 0
    total: int = 0
    stage: int = 0
    stages: int = 0
    message: str = ""
    atlas: AtlasResult | None = None
    colors: ColorResult | None = None
    population: PopulationResult | None = None
    selection: RegionSelection | None = None
    inputs: ApplicationInputs = ApplicationInputs()


class PlotData(NamedTuple):
    """The view receives numerical plot data, never the fitted estimator."""

    specimen_ids: tuple[str, ...]
    scores: np.ndarray
    method: str
    x_axis: int
    y_axis: int
    point: np.ndarray | None = None
