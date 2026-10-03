"""Validated, versioned files exchanged by the profiling parent and child."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
