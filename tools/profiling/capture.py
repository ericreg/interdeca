"""Child-only observation hooks around the unchanged application and upstream code."""

import inspect
import json
import pstats
import resource
import threading
import time
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
from pathlib import Path
from typing import NamedTuple

import memray
import numpy as np
import psutil
import yappi

from interdeca import _native, morphoweave
from interdeca._mesh_cache import MeshCache
from interdeca._morphoweave_host import DataNode, Host
from interdeca.blender import Blender
from interdeca.cli import main
from interdeca.dataset import Dataset
from interdeca.io import LandmarkIO, MeshIO, TextureIO
from interdeca.models import Arrays, Landmarks, Mesh
from interdeca.serial_execution import SerialExecutionManager
from interdeca.spatial import Spatial
from interdeca.surface import MeshProcessing, TextureBaking
from tools.profiling.serde import CacheCapture, Config, Span


class Clock(NamedTuple):
    wall: float
    process: float
    thread: float
    children: float


class Frame(NamedTuple):
    name: str
    start: Clock
    child_times: list[tuple[float, float]]


def clock() -> Clock:
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    return Clock(
        time.perf_counter(),
        time.process_time(),
        time.thread_time(),
        child.ru_utime + child.ru_stime,
    )


def retained_bytes(value) -> int:
    """Count unique NumPy backing buffers, without keeping results alive afterward.

    This excludes Python objects, estimators, VTK allocations, and allocator slack.
    The read-only walk neither copies numerical arrays nor changes their lifetime.
    """
    seen, buffers = set(), {}

    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            while isinstance(item.base, np.ndarray):
                item = item.base
            buffers[id(item)] = item.nbytes
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return sum(buffers.values())


class Recorder:
    """Keep hooks local to this disposable child; production code stays unchanged."""

    def __init__(self, config: Config):
        self.config = config
        self.local = threading.local()
        self.process = psutil.Process()
        self.stream = (config.output / "spans.jsonl").open("x")
        self.lock = threading.Lock()
        self.pipeline_index = 0

    def write(self, span: Span):
        with self.lock:
            self.stream.write(span.model_dump_json() + "\n")
            self.stream.flush()

    @contextmanager
    def span(self, name, kind="function"):
        stack = getattr(self.local, "stack", [])
        self.local.stack = stack
        frame = Frame(name, clock(), [])
        rss = self.process.memory_info().rss if kind != "function" else None
        stack.append(frame)
        values = {}
        failed = False
        try:
            yield values
        except BaseException:
            failed = True
            raise
        finally:
            end = clock()
            stack.pop()
            wall, cpu = end.wall - frame.start.wall, end.thread - frame.start.thread
            if stack:
                stack[-1].child_times.append((wall, cpu))
            self.write(
                Span(
                    kind=kind,
                    name=name,
                    pipeline=getattr(self.local, "pipeline", ""),
                    task=getattr(self.local, "task", ""),
                    start=frame.start.wall,
                    end=end.wall,
                    wall_seconds=wall,
                    process_cpu_seconds=end.process - frame.start.process,
                    thread_cpu_seconds=cpu,
                    children_cpu_seconds=end.children - frame.start.children,
                    exclusive_wall_seconds=wall - sum(t[0] for t in frame.child_times),
                    exclusive_thread_cpu_seconds=cpu
                    - sum(t[1] for t in frame.child_times),
                    rss_start=rss,
                    rss_end=self.process.memory_info().rss if rss is not None else None,
                    failed=failed or values.pop("failed", False),
                    **values,
                )
            )

    def wrap(self, owner, method, *, file_direction=None, path_index=0):
        descriptor = inspect.getattr_static(owner, method)
        function = getattr(owner, method)
        label = f"{owner.__name__}.{method}"

        @wraps(function)
        def observed(*args, **kwargs):
            with self.span(label) as values:
                result = function(*args, **kwargs)
                # Count logical OBJ traffic once at the codec boundary. These bytes
                # are not physical disk traffic, since the OS may serve cached data.
                if file_direction is not None:
                    path = Path(args[path_index])
                    values.update(
                        file_bytes=path.stat().st_size,
                        file_direction=file_direction,
                        file_path=str(path),
                    )
                if owner is Arrays and method == "frozen":
                    values["array_copy_bytes"] = result.nbytes
                return result

        setattr(
            owner,
            method,
            staticmethod(observed)
            if isinstance(descriptor, staticmethod)
            else observed,
        )

    def install(self):
        # Resource settings are experimental controls, not numerical application
        # options. Leave the production default in place for acceptance captures.
        if self.config.cache_budget_bytes is not None:
            morphoweave.MorphoWeave.stage_inputs = staticmethod(
                partial(
                    morphoweave.MorphoWeave.stage_inputs,
                    cache_budget_bytes=self.config.cache_budget_bytes,
                )
            )
        original_close = MeshCache.close

        def close(cache):
            before = cache.stats.array_bytes
            original_close(cache)
            capture = CacheCapture(
                budget_bytes=cache.budget_bytes,
                bytes_before_close=before,
                closed_after_task=(
                    getattr(self.local, "last_completed_task", None)
                    if not getattr(self.local, "task", "") else None
                ),
                closed_during_task=getattr(self.local, "task", "") or None,
                **cache.stats._asdict(),
            )
            with (self.config.output / "mesh-cache.jsonl").open("a") as stream:
                stream.write(capture.model_dump_json() + "\n")

        MeshCache.close = close
        original_emit = SerialExecutionManager._emit
        original_execute = SerialExecutionManager._execute
        recorder = self

        def emit(manager, submission, event):
            if event.kind == "task_started":
                recorder.local.task = event.task_id
                recorder.local.active_task = recorder.span(event.task_id, "task")
                recorder.local.task_values = recorder.local.active_task.__enter__()
            elif event.kind in ("task_completed", "failed"):
                if event.kind == "task_completed":
                    recorder.local.last_completed_task = event.task_id
                active = getattr(recorder.local, "active_task", None)
                if active is not None:
                    if event.kind == "failed":
                        recorder.local.task_values["failed"] = True
                    # The emitting frame is the manager's _execute method.
                    # Its results lookup is inspected only here and never stored.
                    caller = inspect.currentframe().f_back
                    recorder.local.task_values["retained_array_bytes"] = retained_bytes(
                        caller.f_locals.get("results", {})
                    )
                    del caller
                    active.__exit__(None, None, None)
                    recorder.local.active_task = None
                    recorder.local.task = ""
            original_emit(manager, submission, event)

        def execute(manager, submission):
            recorder.pipeline_index += 1
            recorder.local.pipeline = submission.pipeline.label
            recorder.local.last_completed_task = None
            profile = recorder.config.mode == "cpu"
            try:
                with recorder.span(submission.pipeline.label, "pipeline"):
                    if profile:
                        # Thread-aware clocks keep CPU attribution valid when the
                        # application owner and execution thread run concurrently.
                        yappi.clear_stats()
                        yappi.set_clock_type("cpu")
                        yappi.start(builtins=True, profile_threads=True)
                    return original_execute(manager, submission)
            finally:
                if profile:
                    yappi.stop()
                    path = recorder.config.output / f"cpu-{recorder.pipeline_index:02d}"
                    threads = yappi.get_thread_stats()
                    yappi.get_func_stats().save(
                        str(path.with_suffix(".pstats")), type="pstat"
                    )
                    with path.with_suffix(".txt").open("w") as stream:
                        threads.print_all(out=stream)
                        stats = pstats.Stats(
                            str(path.with_suffix(".pstats")), stream=stream
                        )
                        stats.sort_stats("cumulative").print_stats(70)
                        stats.sort_stats("tottime").print_stats(70)
                    yappi.clear_stats()
                recorder.local.pipeline = ""

        SerialExecutionManager._emit = emit
        SerialExecutionManager._execute = execute
        if self.config.mode == "baseline":
            return

        # Nested spans separate formatting, validation, copying, upstream kernels,
        # and subprocess waits. They preserve all arguments and return values.
        for owner, names in (
            (Arrays, ("frozen",)),
            (Mesh, ("validated",)),
            (Landmarks, ("validated",)),
            (Dataset, ("validate_specimens",)),
            (MeshIO, ("read", "to_polydata")),
            (LandmarkIO, ("read", "write_json")),
            (TextureIO, ("read", "validate_file", "write_png")),
            (Host, ("loadModel", "saveNode", "mesh")),
            (DataNode, ("SetAndObservePolyData",)),
            (Blender, ("_run", "_save_mesh")),
            (MeshProcessing, ("clean_atlas", "prepare_atlas_uv")),
            (TextureBaking, ("bake_specimen_texture",)),
            (morphoweave.MorphoWeave, ("_inputs",)),
        ):
            for name in names:
                self.wrap(owner, name)
        # Native entry points replace the removed Python OBJ codec. Keep file
        # traffic at these boundaries; nested wrappers otherwise double-count it.
        self.wrap(_native, "read_obj", file_direction="read")
        self.wrap(_native, "write_obj", file_direction="write")
        self.wrap(MeshIO, "write_obj")
        # These spans include all worker CPU in their process totals; thread CPU
        # and Yappi alone cannot attribute work performed on Rust's native pool.
        self.wrap(Spatial, "index")
        self.wrap(Spatial, "project")
        for name in (
            "texture_from_rgba8", "texture_to_rgba8",
            "texture_to_linear", "texture_from_linear",
        ):
            self.wrap(_native, name)
        original_load = morphoweave.load_atlas_builder

        def load(host):
            module = original_load(host)
            for name in (
                "runAlign",
                "runMean",
                "importPaired",
                "denseCorrespondence",
                "computeAverageModelFromGroup",
                "_corr_pair_kdtree",
                "_warp_polydata_tps",
                "procrustesImposition",
                "cleanPolyData",
                "_coord_check_ratio",
                "_mesh_density_score",
                "sparseCorrespondenceBaseMesh",
            ):
                self.wrap(module.MorphoWeaveAtlasBuilderLogic, name)
            self.wrap(module.MorphoWeaveAtlasBuilderWidget, "_generateAtlas")
            return module

        morphoweave.load_atlas_builder = load


def run_child(config: Config):
    """Run the real Click client and shared core with the e2e numerical settings."""
    recorder = Recorder(config)
    recorder.install()
    tracker = (
        memray.Tracker(
            config.output / "allocations.bin",
            native_traces=True,
            file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS,
        )
        if config.mode == "memory"
        else nullcontext()
    )
    arguments = [
        "--quiet",
        "--output",
        str(config.output / "cli"),
        "atlas",
        "--backend",
        config.backend,
        "--models",
        str(config.inputs / "models"),
        "--landmarks",
        str(config.inputs / "landmarks"),
        "--textures",
        str(config.inputs / "textures"),
        "--output",
        str(config.output / "atlas"),
        "--bake-size",
        "512",
        "colors",
        "--sample-only",
        "--samples",
        "1000",
        "--neighbor-average",
        "fit",
        "--method",
        "pca",
        "--components",
        "2",
        "axes",
        "--x",
        "2",
        "--y",
        "1",
        "reconstruct",
        "--x",
        "0.7",
        "--y",
        "0.3",
        "UF_IZ_507781",
        "show",
        "UF_IZ_507782",
        "select",
        "--curve",
        str(config.inputs / "curve.npy"),
        "--both-sides",
        "export",
        "clear-selection",
    ]
    try:
        with tracker, recorder.span("CLI session", "session"):
            main.main(args=arguments, standalone_mode=False)
    finally:
        recorder.stream.close()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        (config.output / "rusage.json").write_text(
            json.dumps(
                {
                    "user_seconds": usage.ru_utime,
                    "system_seconds": usage.ru_stime,
                    "maxrss_platform_units": usage.ru_maxrss,
                    "major_faults": usage.ru_majflt,
                    "minor_faults": usage.ru_minflt,
                },
                indent=2,
            )
            + "\n"
        )
