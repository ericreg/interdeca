"""Strict, versioned experiment boundaries; paths are relative to their JSON file."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from experiments.fishy.models import Parameters


class Boundary(BaseModel):
    """Reject misspelled settings and nonfinite scientific parameters."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Asset(Boundary):
    """A local file and optional archived SHA256; no network fetching is implicit."""

    path: Path
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class Donor(Boundary):
    """One prepared donor, whose triangulation and original UVs must be retained."""

    donor_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    mesh: Asset
    landmarks: Asset
    mesh_coordinates: Literal["RAS", "LPS"] = "RAS"
    landmark_coordinates: Literal["RAS", "LPS"] | None = None


class Generation(Boundary):
    """Explicit generator controls; checked-in profile is not a verified paper run."""

    samples: int = Field(default=250, ge=2)
    seed: int = Field(default=49, ge=0, le=4294967295)
    expected_landmarks: int = Field(default=47, ge=4)
    tps_lambda: float = Field(default=1e-6, ge=0)
    resolution: int = Field(default=2048, ge=32, le=8192)
    emission_margin: int = Field(default=16, ge=0, le=64)
    noise_amount: float = Field(default=0.10, ge=0, le=1)
    noise_scale: float = Field(default=8.0, gt=0)


class Transfer(Boundary):
    """The paper's stated transfer parameters, in the declared mesh units."""

    merge_distance: float = Field(default=0.0, ge=0)
    angle_degrees: float = Field(default=66.0, gt=0, le=89)
    island_margin: float = Field(default=0.002, ge=0, lt=1)
    resolution: int = Field(default=2048, ge=32, le=8192)
    cage_extrusion: float = Field(default=0.001, ge=0)
    margin: int = Field(default=2, ge=0, le=64)
    timeout: float = Field(default=300, gt=0)


class Config(Boundary):
    """Configuration pins donor order, which determines seeded assignment."""

    schema_version: Literal[1] = 1
    profile: str = "cichlid-rerun-unverified"
    units: str = Field(min_length=1)
    reference_id: str
    donors: list[Donor] = Field(min_length=1)
    generation: Generation = Field(default_factory=Generation)
    transfer: Transfer = Field(default_factory=Transfer)

    @model_validator(mode="after")
    def identities(self) -> Self:
        """Prevent duplicate donors and an ambiguous reference selection."""
        names = [item.donor_id for item in self.donors]
        if len(set(names)) != len(names) or self.reference_id not in names:
            raise ValueError("Donor IDs must be unique and include reference_id.")
        return self


class ParameterData(Boundary):
    """CSV/JSON parameter rows validated before entering numerical code."""

    base_color_hue: float = Field(ge=0, lt=1)
    base_color_sat: float = Field(ge=0, le=1)
    base_color_val: float = Field(ge=0, le=1)
    stripe_count: Literal[4, 5]
    stripe_spacing: float
    stripe_width: float = Field(gt=0)
    stripe_longitudinal_offset: float
    belly_hue: float = Field(ge=0, lt=1)
    belly_strength: float = Field(ge=0, le=1)
    belly_translation: float
    tail_hue: float = Field(ge=0, lt=1)
    tail_strength: float = Field(ge=0, le=1)
    rosy_cheeks_present: Literal[0, 1]
    rosy_cheeks_hue: float = Field(ge=0, lt=1)
    rosy_cheeks_strength: float = Field(ge=0, le=1)
    rosy_cheeks_translation_y: float
    rosy_cheeks_translation_z: float

    def to_internal(self) -> Parameters:
        """Numerical code consumes named tuples rather than serialization models."""
        return Parameters(**self.model_dump())


class ArchiveRow(ParameterData):
    """Original generator CSV; historical paths are metadata, never file inputs."""

    sample_tag: str = Field(pattern=r"^[A-Za-z0-9_-]+_fishy_[0-9]{6}$")
    sample_index: int = Field(ge=0)
    specimen_index: int = Field(ge=0)
    mesh_file: str
    landmark_file: str
    tps_lambda: float = Field(ge=0)
    n_landmarks_used: int = Field(ge=4)
    tps_rmse: float = Field(ge=0)

    @field_validator("stripe_count", "rosy_cheeks_present", mode="before")
    @classmethod
    def csv_integer(cls, value: object) -> object:
        """CSV stores categories as text; retain the parent enum restrictions."""
        return int(value) if isinstance(value, str) else value


class Sample(Boundary):
    """Explicit pairing and immutable hashes replace filename/CSV row assumptions."""

    sample_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    donor_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    mesh: Asset
    landmarks: Asset
    texture: Asset


class Manifest(Boundary):
    """Inputs consumed by transfer; contains no factor labels."""

    schema_version: Literal[1] = 1
    kind: Literal["generated", "atlas"]
    backend: Literal["generator", "interdeca", "legacy-atlas"]
    samples: list[Sample] = Field(min_length=1)
    atlas_mesh: Asset | None = None
    atlas_landmarks: Asset | None = None

    @model_validator(mode="after")
    def identities(self) -> Self:
        """Labels and colors must join one-to-one by stable specimen IDs."""
        if (self.kind == "generated") != (self.backend == "generator"):
            raise ValueError(
                "Generated inputs use generator; atlas outputs use interdeca or legacy-atlas."
            )
        ids = [item.sample_id for item in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample IDs.")
        if self.kind == "atlas" and (
            self.atlas_mesh is None or self.atlas_landmarks is None
        ):
            raise ValueError("Atlas manifests require an atlas mesh and landmarks.")
        return self


class Evaluation(Boundary):
    """Explicit baseline evaluation; these defaults are not archived run metadata."""

    palette_bins: int = Field(default=24, ge=2)
    seed: int = Field(default=0, ge=0, le=4294967295)
    folds: int = Field(default=5, ge=2)
    repeats: int = Field(default=4, ge=1)
    logistic_c: float = Field(default=1.0, gt=0)


class CompositionIdentity(BaseModel):
    """Read only identity from stage metadata, ignoring stage-specific fields."""

    sample_ids: list[str]
    donor_ids: list[str]

    @model_validator(mode="after")
    def identities(self) -> Self:
        if (
            not self.sample_ids
            or len(self.sample_ids) != len(self.donor_ids)
            or len(set(self.sample_ids)) != len(self.sample_ids)
        ):
            raise ValueError("Expected unique samples and one donor per sample.")
        return self


class Labels(Boundary):
    """Extraction metadata; specimen identity travels with every numerical array."""

    sample_ids: list[str]
    donor_ids: list[str]
    topology_key: str
    backend: Literal["interdeca", "legacy-atlas"]
    sampling: str
    coverage: str

    @model_validator(mode="after")
    def identities(self) -> Self:
        if len(self.sample_ids) != len(self.donor_ids) or len(
            set(self.sample_ids)
        ) != len(self.sample_ids):
            raise ValueError(
                "Extraction requires unique samples and one donor per sample."
            )
        return self
