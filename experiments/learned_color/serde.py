"""Strict boundary configuration for the documented frozen I-JEPA protocol."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FigureCoordinate(BaseModel):
    """One identified row in a saved population coordinate table."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    specimen_id: str = Field(min_length=1)
    PC1: float
    PC2: float
    UMAP1: float
    UMAP2: float


class FigureSummary(BaseModel):
    """The completion fields needed to identify and verify a figure export."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    profile: str
    specimens: int = Field(ge=2)
    crops_per_specimen: int = Field(ge=1)
    PCA_explained_variance_ratio: tuple[float, float]
    UMAP_method: str
    scope: str
    checkpoint_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    analysis_sha256: dict[str, str] = Field(default_factory=dict)


class RecordedFigureSpecimen(BaseModel):
    """The cohort identity retained in a prepared run's input manifest."""

    model_config = ConfigDict(extra="ignore")
    specimen_id: str = Field(min_length=1)


class FigureRunRecord(BaseModel):
    """Require a completed run, rather than accepting a copied marker alone."""

    model_config = ConfigDict(extra="ignore")
    status: Literal["completed"]
    inputs: tuple[RecordedFigureSpecimen, ...]


class Config(BaseModel):
    """Visible protocol choices; paths resolve relative to this configuration.

    An expected count is required: 47 available mussels must never silently stand
    in for the original 32-specimen experiment. The checkpoint is immutable.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)
    profile: str = Field(min_length=1)
    models_dir: Path
    textures_dir: Path
    output_dir: Path
    expected_pair_count: int = Field(ge=2)
    specimen_ids: tuple[str, ...] = ()
    orientation_reference_id: str = "UF_IZ_438751"
    render_size: int = Field(default=2048, ge=256)
    texture_max_size: int = Field(default=4096, ge=0)
    render_margin: float = Field(default=1.08, gt=0)
    crop_padding_pixels: int = Field(default=12, ge=0)
    crop_alpha_tolerance: int = Field(default=8, ge=0, le=254)
    patch_alpha_tolerance: int = Field(default=10, ge=0, le=254)
    force_square_normalization: bool = True
    patch_size_px: int = Field(default=256, ge=1)
    patches_per_specimen: int = Field(default=64, ge=1)
    max_overlap_iou: float = Field(default=0.6, ge=0, lt=1)
    patch_sample_seed: int = 20240601
    checkpoint: Literal["facebook/ijepa_vith14_1k"] = "facebook/ijepa_vith14_1k"
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    device: Literal["cuda", "mps", "cpu"] = "cuda"
    allow_cpu: bool = False
    batch_size: int = Field(default=8, ge=1)
    standardize_pca_features: bool = True
    umap_neighbors: int = Field(default=10, ge=2)
    umap_min_dist: float = Field(default=0.1, ge=0, le=1)
    random_seed: int = 42
    nearest_neighbors: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def validate_protocol(self):
        """Reject incompatible crop dimensions and ambiguous specimen filters."""
        if self.patch_size_px > self.render_size:
            raise ValueError("patch_size_px exceeds render_size")
        if len(self.specimen_ids) != len(set(self.specimen_ids)):
            raise ValueError("Duplicate specimen IDs")
        if self.specimen_ids and len(self.specimen_ids) != self.expected_pair_count:
            raise ValueError("specimen_ids must match expected_pair_count")
        if self.device == "cpu" and not self.allow_cpu:
            raise ValueError("CPU inference requires allow_cpu=true")
        return self

    @staticmethod
    def load(path: Path):
        """Read JSON and normalize paths before starting any numerical work."""
        path = path.expanduser().resolve()
        values = json.loads(path.read_text())
        for field in ("models_dir", "textures_dir", "output_dir"):
            value = Path(values[field]).expanduser()
            values[field] = (
                (path.parent / value).resolve()
                if not value.is_absolute()
                else value.resolve()
            )
        return Config.model_validate(values)
