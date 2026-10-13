"""Validated wire contracts for the frozen surface-appearance experiment.

JSON stage lists retain the version-1 wire format, identified by run.json's
pipeline_version. Legacy completions may lack config/analysis hashes; callers
must explicitly decide whether that evidence is sufficient. CSV alone stores
numbers as strings, so its reader uses Pydantic's dedicated string-input mode.
"""

import csv
import json
import math
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

from experiments.learned_color.models import (
    AppearanceOptions,
    Completion,
    Coordinate,
    EmbeddingMetadata,
    InputRecord,
    Neighbor,
    PatchRecord,
    RenderRecord,
    RunRecord,
)

VERSION = "documented-ijepa-reconstruction-1"
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[^/\\]+$")]
Vector3 = tuple[float, float, float]


class BoundaryData(BaseModel):
    """All owned schemas reject mutable, nonfinite or coerced scalar inputs."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
        from_attributes=True,
    )


class Config(BoundaryData):
    """Visible protocol choices; paths resolve relative to this configuration.

    An expected count is required: 47 available mussels must never silently stand
    in for the original 32-specimen experiment. The checkpoint is immutable.
    """

    profile: str = Field(min_length=1)
    models_dir: Path
    textures_dir: Path
    output_dir: Path
    expected_pair_count: int = Field(ge=2)
    specimen_ids: tuple[Identifier, ...] = ()
    orientation_reference_id: Identifier = "UF_IZ_438751"
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
    def validate_protocol(self) -> Self:
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

    def to_internal(self) -> AppearanceOptions:
        """Cross the boundary once; numerical stages receive only this record."""
        return AppearanceOptions(**self.model_dump())

    @classmethod
    def load(cls, path: Path) -> Self:
        """Validate JSON scalars first, then resolve portable paths without coercion."""
        path = path.expanduser().resolve()
        config = cls.model_validate_json(path.read_bytes())
        # JSON mode deliberately permits path strings and tuple arrays while
        # strict numeric/bool fields still reject quoted values.
        paths = {}
        for field in ("models_dir", "textures_dir", "output_dir"):
            value = getattr(config, field).expanduser()
            paths[field] = (
                (path.parent / value).resolve()
                if not value.is_absolute()
                else value.resolve()
            )
        return cls.model_validate(config.model_dump() | paths)


class InputData(BoundaryData):
    """The paths and hashes of one native specimen."""

    specimen_id: Identifier
    mesh: Path
    texture: Path
    mesh_sha256: Digest
    texture_sha256: Digest

    def to_internal(self) -> InputRecord:
        return InputRecord(**self.model_dump())


class RunData(BoundaryData):
    """Versioned identity of a prepared or completed experiment."""

    pipeline_version: Literal["documented-ijepa-reconstruction-1"]
    status: Literal["prepared", "completed"]
    inputs: tuple[InputData, ...] = Field(min_length=2)
    source_sha256: dict[str, Digest] = Field(min_length=1)
    platform: str
    packages: dict[str, str]
    scope: str

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        identifiers = [row.specimen_id for row in self.inputs]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Duplicate run specimen IDs")
        if any(
            Path(name).is_absolute() or ".." in Path(name).parts
            for name in self.source_sha256
        ):
            raise ValueError(
                "Source checksums must name files inside the implementation"
            )
        return self

    def to_internal(self) -> RunRecord:
        return RunRecord(
            self.pipeline_version,
            self.status,
            tuple(row.to_internal() for row in self.inputs),
            self.source_sha256,
            self.platform,
            self.packages,
            self.scope,
        )


class RenderData(BoundaryData):
    """A normalized render and finite camera diagnostics."""

    specimen_id: Identifier
    path: Path
    sha256: Digest
    direction: Vector3
    up: Vector3
    skewness: Vector3
    orientation_score: float
    reference_id: Identifier

    def to_internal(self) -> RenderRecord:
        return RenderRecord(**self.model_dump())


class PatchData(BoundaryData):
    """An ordered source rectangle and immutable saved crop identity."""

    specimen_id: Identifier
    patch_index: int = Field(ge=0)
    path: Path
    sha256: Digest
    source_render: Path
    source_render_sha256: Digest
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    size: int = Field(ge=1)
    background_fraction: float = Field(ge=0, le=1)

    def to_internal(self) -> PatchRecord:
        return PatchRecord(**self.model_dump())


class EmbeddingData(BoundaryData):
    """Pinned inference choices and hashes of both saved feature matrices."""

    checkpoint: Literal["facebook/ijepa_vith14_1k"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    device: Literal["cuda", "mps", "cpu"]
    dtype: Literal["float32"]
    pooling: Literal["all-spatial-token-mean"]
    aggregation: Literal["raw-crop-mean-then-single-L2"]
    processor: dict[str, JsonValue]
    model: dict[str, JsonValue]
    patch_manifest_sha256: Digest
    specimen_embeddings_sha256: Digest
    crop_embeddings_sha256: Digest

    @field_validator("processor", "model")
    @classmethod
    def validate_upstream_numbers(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        # Upstream configs are extensible maps, but their recursively nested
        # JSON numbers must obey the same finite-value contract as owned fields.
        pending = list(value.values())
        while pending:
            item = pending.pop()
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Upstream model/processor numbers must be finite")
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        return value

    def to_internal(self) -> EmbeddingMetadata:
        return EmbeddingMetadata(**self.model_dump())


class CoordinateData(BoundaryData):
    """One identified row in the saved population projection table."""

    specimen_id: Identifier
    PC1: float
    PC2: float
    UMAP1: float
    UMAP2: float

    def to_internal(self) -> Coordinate:
        return Coordinate(**self.model_dump())


class NeighborData(BoundaryData):
    """An identified and finite cosine neighbor row."""

    specimen_id: Identifier
    rank: int = Field(ge=1)
    neighbor_id: Identifier
    cosine_similarity: float = Field(ge=-1.000001, le=1.000001)

    def to_internal(self) -> Neighbor:
        return Neighbor(**self.model_dump())


class CompletionData(BoundaryData):
    """Current completion fields, preserving absent hashes in historical runs."""

    profile: str
    specimens: int = Field(ge=2)
    crops_per_specimen: int = Field(ge=1)
    embedding_features: int = Field(ge=1)
    PCA_explained_variance_ratio: tuple[float, float]
    UMAP_method: str
    scope: str
    supervised_performance_claim: Literal[False]
    checkpoint_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: Digest | None = None
    population_plot_sha256: Digest
    analysis_sha256: dict[str, Digest] | None = None

    @field_validator("supervised_performance_claim", mode="before")
    @classmethod
    def validate_claim(cls, value: object) -> object:
        # Literal[False] also compares equal to integer zero in Python; require
        # an actual JSON boolean so strict scalar validation remains deliberate.
        if type(value) is not bool:
            raise ValueError("supervised_performance_claim must be a boolean")
        return value

    @model_validator(mode="after")
    def validate_analysis_paths(self) -> Self:
        if any(Path(name).name != name for name in self.analysis_sha256 or ()):
            raise ValueError("Analysis checksums must name files inside analysis/")
        return self

    def to_internal(self) -> Completion:
        return Completion(**self.model_dump())


class StageData:
    """Read entire stage documents before work, and write the existing wire format."""

    @staticmethod
    def load_options(path: Path) -> AppearanceOptions:
        return Config.load(path).to_internal()

    @staticmethod
    def read_config_value(path: Path) -> dict[str, object]:
        # Compare the saved path spellings as written, without relocating them.
        # Resolving paths here could silently forgive an edited prepared config.
        content = path.read_bytes()
        Config.model_validate_json(content)
        return json.loads(content)

    @staticmethod
    def config_value(options: AppearanceOptions) -> dict[str, object]:
        return Config.model_validate(options).model_dump(mode="json")

    @staticmethod
    def read_run(path: Path) -> RunRecord:
        return RunData.model_validate_json(path.read_bytes()).to_internal()

    @staticmethod
    def read_renders(path: Path) -> tuple[RenderRecord, ...]:
        rows = TypeAdapter(tuple[RenderData, ...]).validate_json(path.read_bytes())
        return tuple(row.to_internal() for row in rows)

    @staticmethod
    def read_patches(path: Path) -> tuple[PatchRecord, ...]:
        rows = TypeAdapter(tuple[PatchData, ...]).validate_json(path.read_bytes())
        return tuple(row.to_internal() for row in rows)

    @staticmethod
    def read_embedding(path: Path) -> EmbeddingMetadata:
        return EmbeddingData.model_validate_json(path.read_bytes()).to_internal()

    @staticmethod
    def read_completion(path: Path) -> Completion:
        return CompletionData.model_validate_json(path.read_bytes()).to_internal()

    @staticmethod
    def matching_completion(first: Path, second: Path) -> bool:
        # Validate both sides while retaining the original presence/absence of
        # optional legacy keys in the completion-marker identity comparison.
        StageData.read_completion(first)
        StageData.read_completion(second)
        return json.loads(first.read_bytes()) == json.loads(second.read_bytes())

    @staticmethod
    def read_coordinates(path: Path) -> tuple[Coordinate, ...]:
        with path.open(newline="") as stream:
            return tuple(
                CoordinateData.model_validate_strings(row, strict=True).to_internal()
                for row in csv.DictReader(stream)
            )

    @staticmethod
    def read_neighbors(path: Path) -> tuple[Neighbor, ...]:
        with path.open(newline="") as stream:
            return tuple(
                NeighborData.model_validate_strings(row, strict=True).to_internal()
                for row in csv.DictReader(stream)
            )

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        """Write atomically, preserving the exact bytes of unchanged JSON data."""
        encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
        if path.exists() and json.loads(path.read_bytes()) == value:
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(path)

    @staticmethod
    def write_run(path: Path, record: RunRecord) -> None:
        StageData.write_json(
            path, RunData.model_validate(record).model_dump(mode="json")
        )

    @staticmethod
    def write_render(path: Path, record: RenderRecord) -> None:
        StageData.write_json(
            path, RenderData.model_validate(record).model_dump(mode="json")
        )

    @staticmethod
    def write_renders(path: Path, records: tuple[RenderRecord, ...]) -> None:
        StageData.write_json(
            path,
            [RenderData.model_validate(row).model_dump(mode="json") for row in records],
        )

    @staticmethod
    def write_patches(path: Path, records: tuple[PatchRecord, ...]) -> None:
        StageData.write_json(
            path,
            [PatchData.model_validate(row).model_dump(mode="json") for row in records],
        )

    @staticmethod
    def _unique_upstream_mapping(
        pairs: list[tuple[str, JsonValue]],
    ) -> dict[str, JsonValue]:
        """Reject upstream keys that become ambiguous after JSON normalization."""
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Upstream configuration keys collide in JSON: {key}")
            result[key] = value
        return result

    @staticmethod
    def _upstream_json(value: dict[str, object]) -> dict[str, JsonValue]:
        """Apply the original JSON encoder's key semantics to upstream maps only."""
        # Transformers uses integer id2label keys in memory; persisted JSON has
        # string keys. Roundtripping through the original encoder preserves its
        # key spelling, nested lists, insertion order and finite-value checks.
        return json.loads(
            json.dumps(value, allow_nan=False),
            object_pairs_hook=StageData._unique_upstream_mapping,
        )

    @staticmethod
    def write_embedding(path: Path, record: EmbeddingMetadata) -> None:
        # Normalize external processor/model maps before strict owned-schema
        # validation. Config choices and other stage fields retain strict types.
        normalized = record._replace(
            processor=StageData._upstream_json(record.processor),
            model=StageData._upstream_json(record.model),
        )
        StageData.write_json(
            path, EmbeddingData.model_validate(normalized).model_dump(mode="json")
        )

    @staticmethod
    def write_completion(path: Path, record: Completion) -> None:
        StageData.write_json(
            path,
            CompletionData.model_validate(record).model_dump(
                mode="json", exclude_none=True
            ),
        )

    @staticmethod
    def write_table(
        path: Path,
        rows: tuple[PatchRecord, ...] | tuple[Coordinate, ...] | tuple[Neighbor, ...],
    ) -> None:
        """NamedTuple field order retains the historical CSV columns and values."""
        if not rows:
            raise ValueError("Cannot write an empty stage table")
        # Validate all rows before opening the destination. NamedTuple typing
        # alone cannot prevent a direct Python caller from supplying infinity.
        if isinstance(rows[0], PatchRecord):
            for row in rows:
                PatchData.model_validate(row)
        elif isinstance(rows[0], Coordinate):
            for row in rows:
                CoordinateData.model_validate(row)
        elif isinstance(rows[0], Neighbor):
            for row in rows:
                NeighborData.model_validate(row)
        else:
            raise TypeError("Unsupported stage table record")
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(rows[0]._fields)
            writer.writerows(rows)
