"""Immutable numerical options and stage records for surface-appearance runs."""

from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np


class AppearanceOptions(NamedTuple):
    """Validated, immutable choices for one surface-appearance run."""

    profile: str
    models_dir: Path
    textures_dir: Path
    output_dir: Path
    expected_pair_count: int
    specimen_ids: tuple[str, ...]
    orientation_reference_id: str
    render_size: int
    texture_max_size: int
    render_margin: float
    crop_padding_pixels: int
    crop_alpha_tolerance: int
    patch_alpha_tolerance: int
    force_square_normalization: bool
    patch_size_px: int
    patches_per_specimen: int
    max_overlap_iou: float
    patch_sample_seed: int
    checkpoint: Literal["facebook/ijepa_vith14_1k"]
    model_revision: str
    device: Literal["cuda", "mps", "cpu"]
    allow_cpu: bool
    batch_size: int
    standardize_pca_features: bool
    umap_neighbors: int
    umap_min_dist: float
    random_seed: int
    nearest_neighbors: int


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


class InputRecord(NamedTuple):
    """Native input identity frozen before rendering."""

    specimen_id: str
    mesh: Path
    texture: Path
    mesh_sha256: str
    texture_sha256: str


class RunRecord(NamedTuple):
    """Protocol identity and provenance shared across every stage."""

    pipeline_version: str
    status: Literal["prepared", "completed"]
    inputs: tuple[InputRecord, ...]
    source_sha256: dict[str, str]
    platform: str
    packages: dict[str, str]
    scope: str


class RenderRecord(NamedTuple):
    """One normalized surface image and its camera diagnostics."""

    specimen_id: str
    path: Path
    sha256: str
    direction: tuple[float, float, float]
    up: tuple[float, float, float]
    skewness: tuple[float, float, float]
    orientation_score: float
    reference_id: str


class PatchRecord(NamedTuple):
    """One sampled crop, including its exact source rectangle."""

    specimen_id: str
    patch_index: int
    path: Path
    sha256: str
    source_render: Path
    source_render_sha256: str
    left: int
    top: int
    size: int
    background_fraction: float


class EmbeddingMetadata(NamedTuple):
    """The immutable feature protocol and saved array identities.

    Processor/model dictionaries are extensible upstream configuration maps;
    source hashes and package versions are genuine key-to-value lookup maps.
    """

    checkpoint: str
    revision: str
    device: Literal["cuda", "mps", "cpu"]
    dtype: str
    pooling: str
    aggregation: str
    processor: dict[str, object]
    model: dict[str, object]
    patch_manifest_sha256: str
    specimen_embeddings_sha256: str
    crop_embeddings_sha256: str


class Coordinate(NamedTuple):
    """One specimen's two descriptive population projections."""

    specimen_id: str
    PC1: float
    PC2: float
    UMAP1: float
    UMAP2: float


class Neighbor(NamedTuple):
    """One stable cosine-neighbor rank within the recorded cohort."""

    specimen_id: str
    rank: int
    neighbor_id: str
    cosine_similarity: float


class Completion(NamedTuple):
    """A completed analysis, with optional hashes for explicit legacy export."""

    profile: str
    specimens: int
    crops_per_specimen: int
    embedding_features: int
    PCA_explained_variance_ratio: tuple[float, float]
    UMAP_method: str
    scope: str
    supervised_performance_claim: bool
    checkpoint_revision: str
    config_sha256: str | None
    population_plot_sha256: str
    analysis_sha256: dict[str, str] | None
