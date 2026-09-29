"""Application-facing task contracts; no worker or UI implementation lives here."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

import numpy as np

type RunId = str
type Observer = Callable[["ExecutionEvent"], None]


class ResultRef(NamedTuple):
    """Refer to an earlier task's result, optionally selecting named fields/items."""

    task_id: str
    path: tuple[str | int, ...] = ()


class Task(NamedTuple):
    """One registered operation with a fixed-field input record."""

    task_id: str
    operation: str
    inputs: tuple
    label: str = ""


class Pipeline(NamedTuple):
    """Stages are barriers: tasks within a stage must not depend on each other."""

    stages: list[list[Task]] | tuple[tuple[Task, ...], ...]
    outputs: tuple[ResultRef, ...]
    label: str = "Analysis"


class Failure(NamedTuple):
    """Diagnostics cross the interface without retaining exception tracebacks."""

    operation: str
    message: str
    traceback: str


class ExecutionEvent(NamedTuple):
    """Immutable notification; completed/total count tasks, not elapsed time."""

    run_id: RunId
    kind: Literal[
        "started", "task_started", "task_completed", "completed", "failed", "cancelled"
    ]
    task_id: str = ""
    label: str = ""
    completed: int = 0
    total: int = 0
    stage: int = 0
    result: Any = None
    failure: Failure | None = None
    stages: int = 0


class EventDispatcher(Protocol):
    """The host posts callbacks to the thread that owns application state.

    post must be thread-safe and asynchronous: callbacks cannot execute before
    submit has returned its run identifier. Hosts detach callbacks on teardown.
    """

    def post(self, callback: Callable[[], None]) -> None: ...


class ExecutionManager(Protocol):
    """The complete controller dependency, independent of scheduling mechanics."""

    def submit(self, pipeline: Pipeline, observer: Observer) -> RunId: ...
    def cancel(self, run_id: RunId) -> None: ...
    def close(self) -> None: ...


class TaskValues:
    """Validate/snapshot task inputs and resolve explicit upstream references."""

    @staticmethod
    def map(
        value: Any, reference: Callable[[ResultRef], Any], snapshot: bool = False
    ) -> Any:
        """Traverse records and sequences without interpreting arbitrary objects."""
        if isinstance(value, ResultRef):
            return reference(value)
        if isinstance(value, np.ndarray):
            # Owned immutable buffers prevent scene edits from changing queued work.
            if snapshot:
                value = value.copy()
                value.setflags(write=False)
            return value
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(
                *(TaskValues.map(item, reference, snapshot) for item in value)
            )
        if isinstance(value, (tuple, list)):
            return tuple(TaskValues.map(item, reference, snapshot) for item in value)
        if isinstance(value, (str, int, float, bool, Path, type(None))):
            return value
        # Fitted estimators are read-only domain objects; they are never UI handles.
        if callable(getattr(value, "inverse_transform", None)):
            return value
        raise TypeError(f"Unsupported task input: {type(value).__name__}")

    @staticmethod
    def resolve(reference: ResultRef, results: Mapping[str, Any]) -> Any:
        """Select named record fields or sequence positions from completed results."""
        value = results[reference.task_id]
        for component in reference.path:
            value = (
                getattr(value, component)
                if isinstance(component, str)
                else value[component]
            )
        return value

    @staticmethod
    def snapshot(pipeline: Pipeline, operations: Mapping[str, Callable]) -> Pipeline:
        """Reject ambiguous graphs before any operation can produce side effects."""
        seen: set[str] = set()
        stages = []
        for stage in pipeline.stages:
            current: set[str] = set()
            copied = []

            def check(reference: ResultRef) -> ResultRef:
                if reference.task_id not in seen:
                    raise ValueError(
                        f"Dependency {reference.task_id!r} must belong to an earlier stage."
                    )
                return reference

            for task in stage:
                if not task.task_id or task.task_id in seen | current:
                    raise ValueError(f"Duplicate or empty task ID: {task.task_id!r}")
                if task.operation not in operations:
                    raise ValueError(f"Unknown operation: {task.operation}")
                if not isinstance(task.inputs, tuple) or not hasattr(
                    task.inputs, "_fields"
                ):
                    raise TypeError("Task inputs must be a NamedTuple record.")
                copied.append(
                    task._replace(inputs=TaskValues.map(task.inputs, check, True))
                )
                current.add(task.task_id)
            # Publish the current IDs only after all siblings have been checked.
            stages.append(tuple(copied))
            seen.update(current)
        if not seen:
            raise ValueError("A pipeline needs at least one task.")
        for output in pipeline.outputs:
            if output.task_id not in seen:
                raise ValueError(f"Unknown pipeline output: {output.task_id}")
        return Pipeline(tuple(stages), tuple(pipeline.outputs), pipeline.label)
