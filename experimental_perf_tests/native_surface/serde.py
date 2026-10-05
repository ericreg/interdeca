"""Persisted parity measurements; failed gates remain explicit in reports."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Agreement(BaseModel):
    """Unit-range RGB errors on joint coverage, with coverage measured separately."""

    model_config = ConfigDict(allow_inf_nan=False)
    samples: int
    joint_samples: int
    mean_rgb: float | None
    p95_max_channel: float | None
    coverage_loss_pp: float
    coverage_gain_pp: float
    passed: bool


class Measurement(BaseModel):
    specimen: str
    backend: str
    resolution: int
    rust_seconds: float
    blender_seconds: float | None = None
    agreement: Agreement
    analysis_agreement: Agreement | None = None
    input_sha256: str = ""
    reference_sha256: str = ""
    output_sha256: str = ""
    repeat: int = 0


class Report(BaseModel):
    algorithm: str = "blender-5.2.1-source-port"
    measurements: list[Measurement]
    all_passed: bool
    scope: str = "baker_on_reference_uv"
    platform: str = ""
    python: str = ""
    implementation_sha256: str = ""
    extension_sha256: str = ""
    elapsed_seconds: float = 0
    peak_rss_bytes: int = 0


class FixtureResult(BaseModel):
    case: str
    reference_covered: int
    rust_covered: int
    mae: float | None
    maximum: float | None


class CleanupFixtureResult(BaseModel):
    positions: list[float]
    blender_vertices: list[list[float]]
    rust_vertices: list[list[float]]
    blender_faces: list[list[int]]
    rust_faces: list[list[int]]


class MeshReport(BaseModel):
    backend: str
    resolution: int
    vertices: int
    triangles: int
    charts: int
    cleanup_seconds: float
    uv_seconds: float
    occupied_fraction: float
    blender_occupied_fraction: float
    subpixel_triangles: int
    cleanup_same_connectivity: bool | None = None
    cleanup_max_distance: float | None = None
    cleanup_coordinate_tolerance: float | None = None
    cleanup_passed: bool | None = None


class RepeatedResult(BaseModel):
    backend: str
    resolution: int
    repetitions: int
    specimens_per_repetition: int
    primary_passes: list[int]
    analysis_passes: list[int]
    combined_passes: list[int]
    deterministic_pixels: bool
    native_kernel_total_seconds: list[float]
    whole_process_seconds: list[float]
    peak_rss_bytes: list[int]
    maximum_mean_rgb: float
    maximum_p95_channel: float
    maximum_coverage_loss_pp: float
    passed: bool
    implementation_sha256: list[str]
    extension_sha256: list[str]


class QualificationSnapshot(BaseModel):
    status: str = "experimental; production replacement blocked"
    scope: str = "Baker comparisons use preserved Blender UVs; not whole-workflow qualification."
    results: list[RepeatedResult]
    layouts: list[MeshReport]
    custom_layout_pilots: list[Measurement]
    blockers: list[str]
    uv_comparisons: list["UVComparison"] = Field(default_factory=list)
    workflow_comparisons: list["WorkflowComparison"] = Field(default_factory=list)


class BlenderSourceFile(BaseModel):
    path: str
    sha256: str
    license: str
    symbols: list[str]
    status: str
    adaptation: str


class BlenderSourceManifest(BaseModel):
    version: str
    revision: str
    repository: str
    distribution_license: str
    files: list[BlenderSourceFile]


class UVComparison(BaseModel):
    """Direct per-corner comparison with fresh Smart UV Project output."""

    case: str
    faces: int
    charts: int
    margin: float
    native_seconds: float
    blender_seconds: float | None
    maximum_uv_error: float
    exact_corners: int
    total_corners: int
    input_sha256: str
    native_sha256: str
    reference_sha256: str


class ArrayDifference(BaseModel):
    name: str
    same_shape: bool
    exact: bool
    maximum: float | None = None
    mean: float | None = None
    rms: float | None = None


class WorkflowComparison(BaseModel):
    backend: str
    reference: Path
    candidate: Path
    cleanup_bijective: bool
    cleanup_same_connectivity: bool
    cleanup_max_distance: float
    cleanup_coordinate_tolerance: float
    cleanup_passed: bool
    reference_wall_seconds: float
    candidate_wall_seconds: float
    runtime_passed: bool
    reference_peak_rss_bytes: int
    candidate_peak_rss_bytes: int
    memory_passed: bool
    downstream_arrays_exact: bool
    downstream_passed: bool = False
    pca_score_relative_rms: float | None = None
    # Named limits remain in the saved report so qualification is reproducible.
    numerical_limits: dict[str, float] = Field(default_factory=dict)
    arrays: list[ArrayDifference]


class CompletionEvidence(BaseModel):
    """Local integration, quality, workflow and distributable evidence."""

    date: str
    platform: str
    extension_sha256: str
    population_summary: Path
    complete_workflows: list[WorkflowComparison]
    artifact_sha256: dict[str, str]
    packaging_verified: bool
    wheel_smoke_verified: bool
    integration_verified: bool
    automated_tests_run: bool = False
    limitations: list[str]
