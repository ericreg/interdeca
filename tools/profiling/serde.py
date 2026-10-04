"""Validated, versioned files exchanged by the profiling parent and child."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from interdeca.serde import BlenderInstallationData


class Data(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Config(Data):
    schema_version: Literal[1] = 1
    backend: Literal["native", "morphoweave"]
    mode: Literal["baseline", "spans", "cpu", "memory"]
    inputs: Path
    output: Path
    cache_budget_bytes: int | None = Field(default=None, ge=0)


class CacheCapture(Data):
    """Run-local reuse counters, with cleanup and Python overhead kept explicit."""

    budget_bytes: int
    bytes_before_close: int
    hits: int
    misses: int
    bypasses: int
    evictions: int
    invalidations: int
    array_bytes: int
    peak_array_bytes: int
    entries: int
    peak_entries: int
    metadata_bytes: int
    peak_metadata_bytes: int
    write_admissions: int = 0
    write_bypasses: int = 0
    closed_after_task: str | None = None
    closed_during_task: str | None = None


class Span(Data):
    """Nested elapsed intervals; exclusive time subtracts instrumented children."""

    kind: Literal["task", "function", "pipeline", "session"]
    name: str
    pipeline: str = ""
    task: str = ""
    start: float
    end: float
    wall_seconds: float
    process_cpu_seconds: float
    thread_cpu_seconds: float
    children_cpu_seconds: float
    exclusive_wall_seconds: float
    exclusive_thread_cpu_seconds: float
    rss_start: int | None = None
    rss_end: int | None = None
    retained_array_bytes: int | None = None
    array_copy_bytes: int | None = None
    file_bytes: int | None = None
    file_direction: Literal["read", "write"] | None = None
    file_path: str | None = None
    failed: bool = False


class ProcessSample(Data):
    pid: int
    created: float
    parent: bool
    name: str
    rss_bytes: int
    uss_bytes: int | None = None
    user_seconds: float
    system_seconds: float
    threads: int


class Sample(Data):
    time: float
    processes: list[ProcessSample]


class Run(Data):
    schema_version: Literal[1] = 1
    config: Config
    exit_code: int
    wall_seconds: float
    sample_interval_seconds: float
    parent_peak_rss_bytes: int
    tree_peak_rss_bytes: int
    parent_peak_uss_bytes: int | None
    tree_peak_uss_bytes: int | None
    sampled_processes: int
    output_hashes: dict[str, str] = Field(default_factory=dict)


class ResourceUsage(Data):
    """Operating-system process totals saved by the original and current recorders.

    The recorded maxrss value uses native platform units (bytes on macOS).
    Keeping it separate from sampled RSS exposes short peaks between samples.
    """

    user_seconds: float
    system_seconds: float
    maxrss_platform_units: int
    major_faults: int
    minor_faults: int


class FunctionCost(Data):
    calls: int = 0
    inclusive_wall_seconds: float = 0
    exclusive_wall_seconds: float = 0
    exclusive_thread_cpu_seconds: float = 0
    array_copy_bytes: int = 0
    logical_read_bytes: int = 0
    logical_write_bytes: int = 0


class AllocationSite(Data):
    bytes: int
    allocations: int
    python_stack: list[str]


class TaskSummary(Data):
    pipeline: str
    task: str
    wall_seconds: float
    parent_cpu_seconds: float
    children_cpu_seconds: float
    parent_peak_rss_bytes: int
    tree_peak_rss_bytes: int
    retained_array_bytes: int


class Summary(Data):
    schema_version: Literal[1] = 1
    runs: list[Run]
    tasks: dict[str, list[TaskSummary]]
    functions: dict[str, dict[str, FunctionCost]]
    peak_allocations: dict[str, list[AllocationSite]]
    tracked_heap_peaks: dict[str, int]


class MetricChange(Data):
    reference: float
    candidate: float
    reduction: float
    reduction_percent: float | None


class Comparison(Data):
    """Correctness and measurements stay separate from the acceptance decision."""

    schema_version: Literal[1] = 1
    reference: Path
    candidate: Path
    reference_revision: str
    candidate_revision: str
    compatible_environment: bool
    identical_outputs: bool
    issues: list[str]
    investigation_alerts: list[str]
    metrics: dict[str, dict[str, MetricChange]]


class Distribution(Data):
    """Distances in the reported coordinate or color units; never NaN for empty data."""

    count: int
    mean: float | None = None
    rms: float | None = None
    p50: float | None = None
    p95: float | None = None
    maximum: float | None = None


class PopulationDataset(Data):
    source: Path
    specimens: tuple[str, ...]
    source_sha256: dict[str, str]
    removed_zero_area_faces: dict[str, list[int]]


class PopulationEnvironment(Data):
    """Source, input, and runtime identity required to resume a population capture.

    Version and hash dictionaries are lookups by dependency or file name. The
    surrounding fixed fields are validated before a run can reuse the archive.
    """

    platform: str
    python: str
    logical_cpus: int | None
    physical_cpus: int | None
    ram_bytes: int
    blender: BlenderInstallationData
    versions: dict[str, str]
    git_head: str
    lock_sha256: str
    source_sha256: dict[str, str]
    inputs_sha256: dict[str, str]
    thread_environment: dict[str, str | None]


class HostMemorySample(Data):
    """Whole-machine context, not memory attributable to the application alone.

    The swap counter fields preserve psutil's sin/sout values. On macOS psutil
    7.2.2 these are broader page-in/page-out counters, not dedicated swap events.
    """

    time: float
    unix_time: float
    available_bytes: int
    swap_used_bytes: int
    swap_in_bytes: int
    swap_out_bytes: int
    free_disk_bytes: int


class MeshAccuracy(Data):
    vertices: int
    triangles: int
    collapsed_triangles: int
    collapsed_fraction: float
    area: float
    bbox_diagonal: float


class SurfaceDifference(Data):
    first_to_second: Distribution
    second_to_first: Distribution


class ColorDifference(Data):
    samples: int
    first_covered: int
    second_covered: int
    jointly_covered: int
    mean_absolute_channel_0_255: float | None = None
    rms_channel_0_255: float | None = None
    delta_e_76: Distribution


class SpecimenAccuracy(Data):
    aligned: MeshAccuracy
    resampled: MeshAccuracy
    input_centroid_size: float
    aligned_centroid_size: float
    size_ratio: float
    vertex_to_input: Distribution
    interiors_to_input: Distribution
    input_to_resampled: Distribution
    transferred_landmarks_to_input: Distribution
    transferred_landmarks_error_over_centroid_size: Distribution
    baked_to_source_color: ColorDifference


class BackendAccuracy(Data):
    directory: Path
    atlas: MeshAccuracy
    atlas_centroid_size: float
    mean_input_centroid_size: float
    atlas_landmark_to_surface: Distribution
    specimens: dict[str, SpecimenAccuracy] = Field(default_factory=dict)


class CrossBackendSpecimen(Data):
    aligned_landmark_rigid_residual: Distribution
    resampled_surface_after_specimen_rigid_fit: SurfaceDifference
    color_at_common_atlas_locations: ColorDifference


class CrossBackendAccuracy(Data):
    atlas_landmark_rigid_residual: Distribution
    atlas_landmark_similarity_residual: Distribution
    morphoweave_to_native_scale: float
    atlas_surface_rigid: SurfaceDifference
    atlas_surface_similarity: SurfaceDifference
    average_color_at_common_atlas_locations: ColorDifference
    pca_pairwise_distance_correlation: float
    pca_pairwise_distance_relative_rms: float
    specimens: dict[str, CrossBackendSpecimen] = Field(default_factory=dict)


class PopulationAccuracy(Data):
    schema_version: Literal[1] = 1
    inputs: Path
    samples_per_surface: int
    seed: int
    methods: dict[str, str]
    backends: dict[str, BackendAccuracy]
    cross_backend: CrossBackendAccuracy
