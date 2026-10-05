"""Validated, portable evidence for the isolated spatial-search experiment."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Timing(Record):
    backend: str
    specimen: str
    repeat: int
    method: str
    vertices: int
    triangles: int
    queries: int
    build_seconds: float = Field(ge=0)
    query_seconds: float = Field(ge=0)
    build_cpu_seconds: float = Field(ge=0)
    query_cpu_seconds: float = Field(ge=0)


class Agreement(Record):
    backend: str
    specimen: str
    queries: int
    bbox_diagonal: float
    point_max_error: float
    point_rms_error: float
    distance_max_error: float
    different_triangle_ids: int
    points_outside_tolerance: int
    distances_outside_tolerance: int
    parallel_bitwise_equal: bool
    recovered_native_max_error: float | None = None
    different_float32_output_points: int


class Experiment(Record):
    schema_version: Literal[1] = 1
    revision: str
    platform: str
    python: str
    processor: str
    physical_cpus: int | None
    logical_cpus: int | None
    versions: dict[str, str]
    thread_environment: dict[str, str | None]
    implementation_sha256: dict[str, str]
    source_sha256: dict[str, str] = Field(default_factory=dict)
    command: list[str]
    captures: str
    repeats: int
    parallel_threads: int
    verification_cases: list[str]
    timings: list[Timing] = Field(default_factory=list)
    agreement: list[Agreement] = Field(default_factory=list)
