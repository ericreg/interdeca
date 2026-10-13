"""Strict, versioned experiment boundaries; paths are relative to their JSON file."""

import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from experiments.fishy.records import (
    BuildIdentity,
    BundleConfig,
    BundleManifest,
    BundleProvenance,
    BundleRights,
    Completion,
    EvaluationOptions,
    ExportManifest,
    InputFile,
    InventoryFile,
    PaperRun,
    Parameters,
    SchemeScore,
    Specimen,
    StageRun,
)


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

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )

    palette_bins: int = Field(default=24, ge=2)
    seed: int = Field(default=0, ge=0, le=4294967295)
    folds: int = Field(default=5, ge=2)
    repeats: int = Field(default=4, ge=1)
    logistic_c: float = Field(default=1.0, gt=0)

    def to_internal(self) -> EvaluationOptions:
        """Freeze validated options before any numerical classifier work."""
        return EvaluationOptions(**self.model_dump())


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


# Runner schemas deliberately import only standard-library records. The separate
# tooling requirements add validation without importing core/scientific packages.
SHA256 = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]


class RecordBoundary(Boundary):
    """Owned metadata is strict and immutable; legacy layouts stay explicit."""

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )


class VersionOne(RecordBoundary):
    """A version is mandatory; omitted/boolean/future versions are unsupported."""

    schema_version: Literal[1]

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1.")
        return value


class InputFileData(RecordBoundary):
    """Version-one input row; early runs omitted modification timestamps."""

    path: str = Field(min_length=1)
    bytes: int = Field(ge=0)
    sha256: SHA256
    mtime_ns: int | None = Field(default=None, ge=0)

    def to_internal(self) -> InputFile:
        return InputFile(
            **self.model_dump(), present_fields=frozenset(self.model_fields_set)
        )


class StageRunData(RecordBoundary):
    """Running evidence and completed evidence retain their historical layouts."""

    exit_code: int | None
    status: Literal["running"] | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0)
    thread_environment: dict[str, str] | None = None

    @model_validator(mode="after")
    def state(self) -> Self:
        # A running record cannot simultaneously assert completion. Historical
        # completed stages carry an exit code and need no status discriminator.
        if self.status == "running" and self.exit_code is not None:
            raise ValueError("Running stages cannot have a completed exit code.")
        if self.exit_code is None and self.status != "running":
            raise ValueError("Incomplete stages must explicitly say running.")
        return self

    def to_internal(self) -> StageRun:
        return StageRun(
            **self.model_dump(), present_fields=frozenset(self.model_fields_set)
        )


class PaperRunData(VersionOne):
    """The persisted version-one paper coordinator contract, including old fields."""

    upstream_revision: Revision
    source: str = Field(min_length=1)
    module_run: str = Field(min_length=1)
    donor_scans: str | None
    python: str
    platform: str
    packages: dict[str, str]
    inputs: tuple[InputFileData, ...]
    stages: dict[str, StageRunData]
    scope: str
    analysis_python: str | None = None
    analysis_packages: dict[str, str] | None = None
    donor_scans_attached_unix_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def identities(self) -> Self:
        paths = [item.path for item in self.inputs]
        if len(paths) != len(set(paths)):
            raise ValueError("Paper inputs must have unique paths.")
        return self

    def to_internal(self) -> PaperRun:
        return PaperRun(
            self.schema_version,
            self.upstream_revision,
            self.source,
            self.module_run,
            self.donor_scans,
            self.python,
            self.platform,
            self.packages,
            tuple(item.to_internal() for item in self.inputs),
            {name: stage.to_internal() for name, stage in self.stages.items()},
            self.scope,
            self.analysis_python,
            self.analysis_packages,
            self.donor_scans_attached_unix_seconds,
            frozenset(self.model_fields_set),
        )


class PortablePaths(RecordBoundary):
    """Inventory/config paths are relative and never escape their bundle root."""

    @staticmethod
    def relative(value: str) -> str:
        # Backslashes and drive prefixes would change meaning across platforms.
        if (
            not value
            or "\\" in value
            or ":" in value
            or Path(value).is_absolute()
            or any(part in (".", "..") for part in value.split("/"))
        ):
            raise ValueError("Expected a portable relative bundle path.")
        return value


class BundleConfigData(VersionOne, PortablePaths):
    """Historical version-one config; numerical requirements remain author-pinned."""

    python: Literal["3.12.8"]
    requirements: str
    source: str
    module_run: str
    donor_scans: str
    author_arrays: str

    @field_validator(
        "requirements", "source", "module_run", "donor_scans", "author_arrays"
    )
    @classmethod
    def paths(cls, value: str) -> str:
        return cls.relative(value)

    def to_internal(self) -> BundleConfig:
        return BundleConfig(**self.model_dump())


class InventoryFileData(PortablePaths):
    """Manifest payload entry, without input-family metadata."""

    path: str
    bytes: int = Field(ge=0)
    sha256: SHA256

    @field_validator("path")
    @classmethod
    def path_relative(cls, value: str) -> str:
        return cls.relative(value)

    def to_internal(self) -> InventoryFile:
        return InventoryFile(**self.model_dump())


class InputInventoryData(InventoryFileData):
    """Explicit unversioned legacy input inventory entry."""

    family: Literal["synthetic", "module", "donor-scans"]


class BundleManifestData(VersionOne):
    """Hash identity is checked separately from the structural contract."""

    source_tree_sha256: SHA256
    files: tuple[InventoryFileData, ...]

    @model_validator(mode="after")
    def unique_paths(self) -> Self:
        names = [item.path for item in self.files]
        if len(names) != len(set(names)):
            raise ValueError("Manifest payload paths must be unique.")
        return self

    def to_internal(self) -> BundleManifest:
        return BundleManifest(
            self.schema_version,
            self.source_tree_sha256,
            tuple(item.to_internal() for item in self.files),
        )


class CompletionData(RecordBoundary):
    """Explicit unversioned legacy completion document; never infer a version."""

    upstream_revision: Revision
    scope: str

    def to_internal(self) -> Completion:
        return Completion(**self.model_dump())


class BuildIdentityData(RecordBoundary):
    """Explicit unversioned legacy interrupted-build identity."""

    identity: SHA256

    def to_internal(self) -> BuildIdentity:
        return BuildIdentity(self.identity)


class BundleRightsData(RecordBoundary):
    dataset_redistribution: str
    code_license: str
    access: str


class BundleProvenanceData(RecordBoundary):
    """Explicit unversioned legacy provenance, preserving every original field."""

    bundle_scope: str
    source_head: Revision
    source_snapshot: str
    author_revision: Revision
    author_repository: str
    input_assets: int = Field(ge=0)
    input_bytes: int = Field(ge=0)
    input_origin: dict[str, str]
    rights: BundleRightsData
    historical_parity: str
    limitations: tuple[str, ...]

    def to_internal(self) -> BundleProvenance:
        values = self.model_dump()
        values["rights"] = BundleRights(**values["rights"])
        return BundleProvenance(**values)


class SpecimenData(RecordBoundary):
    sample_id: str = Field(min_length=1)
    donor_id: int = Field(ge=0)
    donor_name: str = Field(min_length=1)

    def to_internal(self) -> Specimen:
        return Specimen(**self.model_dump())


class ExportManifestData(RecordBoundary):
    """Explicit legacy export inventory (different from version-one bundles)."""

    bundle_manifest_sha256: SHA256
    files: tuple[InventoryFileData, ...]

    def to_internal(self) -> ExportManifest:
        return ExportManifest(
            self.bundle_manifest_sha256,
            tuple(item.to_internal() for item in self.files),
        )


class FoldScoreData(RecordBoundary):
    """Serialize fixed fold records only after numerical fitting finishes."""

    fold: int = Field(ge=0)
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    null_train_indices: tuple[int, ...]
    null_test_indices: tuple[int, ...]
    balanced_accuracy: float | None = Field(ge=0, le=1)
    shuffled_accuracy: float | None = Field(ge=0, le=1)
    null_corrected: float | None = Field(ge=-1, le=1)


class SchemeScoreData(RecordBoundary):
    folds: tuple[FoldScoreData, ...]
    note: str


class Documents:
    """Read owned JSON into named records and serialize at the file boundary."""

    @staticmethod
    def payload(record: object) -> object:
        """Restore historical object/array JSON without leaking NamedTuple arrays.

        Presence sets retain the distinction between missing and explicit null
        optional fields. Only these boundary adapters use object dictionaries.
        """
        if isinstance(record, tuple) and hasattr(record, "_fields"):
            fields = record._asdict()
            present = fields.pop("present_fields", None)
            if present is not None:
                fields = {
                    key: value
                    for key, value in fields.items()
                    if key not in record._field_defaults or key in present
                }
            if isinstance(record, InventoryFile) and record.family is None:
                fields.pop("family")
            return {key: Documents.payload(value) for key, value in fields.items()}
        if isinstance(record, dict):
            return {key: Documents.payload(value) for key, value in record.items()}
        if isinstance(record, (tuple, list)):
            return [Documents.payload(value) for value in record]
        return record

    @staticmethod
    def release_payload(path: Path, value: object) -> object:
        """Validate owned release outputs without changing their historical layout."""
        schemas = {
            "bundle.json": BundleConfigData,
            "provenance.json": BundleProvenanceData,
            ".building.json": BuildIdentityData,
        }
        if path.name == "manifest.json":
            schema = (
                BundleManifestData if "schema_version" in value else ExportManifestData
            )
            return schema.model_validate_json(
                json.dumps(value, allow_nan=False)
            ).model_dump(mode="json", exclude_unset=True)
        if path.name in schemas:
            return (
                schemas[path.name]
                .model_validate_json(json.dumps(value, allow_nan=False))
                .model_dump(mode="json", exclude_unset=True)
            )
        if path.name in ("input-inventory.json", "specimens.json"):
            schema = (
                InputInventoryData
                if path.name == "input-inventory.json"
                else SpecimenData
            )
            rows = TypeAdapter(tuple[schema, ...]).validate_json(
                json.dumps(value, allow_nan=False)
            )
            return [row.model_dump(mode="json", exclude_unset=True) for row in rows]
        return value

    @staticmethod
    def run(path: Path) -> PaperRun:
        return PaperRunData.model_validate_json(path.read_text()).to_internal()

    @staticmethod
    def run_payload(record: PaperRun) -> dict:
        return PaperRunData.model_validate_json(
            json.dumps(Documents.payload(record), allow_nan=False)
        ).model_dump(mode="json", exclude_unset=True)

    @staticmethod
    def bundle(path: Path) -> BundleConfig:
        return BundleConfigData.model_validate_json(path.read_text()).to_internal()

    @staticmethod
    def manifest(path: Path) -> BundleManifest:
        return BundleManifestData.model_validate_json(path.read_text()).to_internal()

    @staticmethod
    def legacy_completion(path: Path) -> Completion:
        return CompletionData.model_validate_json(path.read_text()).to_internal()

    @staticmethod
    def legacy_build_identity(path: Path) -> BuildIdentity:
        return BuildIdentityData.model_validate_json(path.read_text()).to_internal()

    @staticmethod
    def legacy_provenance(path: Path) -> BundleProvenance:
        return BundleProvenanceData.model_validate_json(path.read_text()).to_internal()

    @staticmethod
    def legacy_input_inventory(path: Path) -> tuple[InventoryFile, ...]:
        rows = TypeAdapter(tuple[InputInventoryData, ...]).validate_json(
            path.read_text()
        )
        names = [row.path for row in rows]
        if len(names) != len(set(names)):
            raise ValueError("Input inventory paths must be unique.")
        return tuple(row.to_internal() for row in rows)

    @staticmethod
    def legacy_specimens(path: Path) -> tuple[Specimen, ...]:
        rows = TypeAdapter(tuple[SpecimenData, ...]).validate_json(path.read_text())
        if len({row.sample_id for row in rows}) != len(rows):
            raise ValueError("Specimen sample IDs must be unique.")
        return tuple(row.to_internal() for row in rows)

    @staticmethod
    def legacy_export_manifest(path: Path) -> ExportManifest:
        return ExportManifestData.model_validate_json(path.read_text()).to_internal()

    @staticmethod
    def scores(results: dict[str, SchemeScore]) -> dict:
        # Validation through JSON preserves strict JSON-array tuple semantics.
        return {
            name: SchemeScoreData.model_validate_json(
                json.dumps(Documents.payload(score), allow_nan=False)
            ).model_dump(mode="json")
            for name, score in results.items()
        }
