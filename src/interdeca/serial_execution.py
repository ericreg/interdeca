"""FIFO implementation of the execution interface with one private worker."""

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from queue import Queue
from threading import Event, Lock, Thread
from traceback import format_exc
from typing import Any, NamedTuple

from interdeca.execution import (
    EventDispatcher,
    ExecutionEvent,
    Failure,
    Observer,
    Pipeline,
    RunId,
    TaskValues,
)


class _Submission(NamedTuple):
    """Private worker state; callers receive only the run identifier."""

    run_id: RunId
    pipeline: Pipeline
    observer: Observer
    cancelled: Event


class SerialExecutionManager:
    """Execute complete pipelines serially; publish events through the host dispatcher.

    Cancellation is observed between operations. close is nonblocking because
    an active native call or Blender subprocess cannot be safely killed as a thread.
    """

    def __init__(self, operations: Mapping[str, Callable], dispatcher: EventDispatcher):
        # Copy the registry so later registration cannot change a queued operation.
        self._operations = dict(operations)
        self._dispatcher = dispatcher
        self._queue: Queue[_Submission | None] = Queue()
        self._pending: dict[RunId, Event] = {}
        self._lock = Lock()
        self._closed = False
        self._worker = Thread(target=self._work, name="interdeca-serial", daemon=True)
        self._worker.start()

    def submit(self, pipeline: Pipeline, observer: Observer, *, run_id: RunId) -> RunId:
        """Submit inputs using the run ID already allocated by the application worker."""
        frozen = TaskValues.snapshot(pipeline, self._operations)
        with self._lock:
            if self._closed:
                raise RuntimeError("Execution manager is closed.")
            if not run_id:
                raise ValueError(
                    "The application worker must allocate a run identifier."
                )
            if run_id in self._pending:
                raise ValueError("A run identifier is already active.")
            cancellation = Event()
            self._pending[run_id] = cancellation
            self._queue.put(_Submission(run_id, frozen, observer, cancellation))
        return run_id

    def cancel(self, run_id: RunId) -> None:
        """Request cancellation; an already completed run is unaffected."""
        with self._lock:
            cancellation = self._pending.get(run_id)
            if cancellation is not None:
                cancellation.set()

    def close(self) -> None:
        """Detach notifications and drain pending runs without blocking the UI."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for cancellation in self._pending.values():
                cancellation.set()
            self._queue.put(None)

    def join(self, timeout: float | None = None) -> bool:
        """Wait for cancelled work to finish after close, outside any UI thread."""
        self._worker.join(timeout)
        return not self._worker.is_alive()

    def _emit(self, submission: _Submission, event: ExecutionEvent) -> None:
        # Recheck at delivery time because the user may close the view after posting.
        def deliver() -> None:
            if not self._closed:
                submission.observer(event)

        self._dispatcher.post(deliver)

    def _work(self) -> None:
        # Taking an entire run preserves FIFO order even when several controllers submit.
        while (submission := self._queue.get()) is not None:
            try:
                self._execute(submission)
            finally:
                with self._lock:
                    self._pending.pop(submission.run_id, None)

    @staticmethod
    def _manage(resources: ExitStack, resource: Any) -> Any:
        """Enter a task resource without allowing cleanup to hide task failures."""
        try:
            enter, leave = type(resource).__enter__, type(resource).__exit__
        except AttributeError:
            raise TypeError("Managed task results must be context managers.") from None
        value = enter(resource)

        # A registered operation's exception remains a failed run even if a
        # general-purpose context manager would normally suppress that exception.
        def release(*error):
            leave(resource, *error)
            return False

        resources.push(release)
        return value

    def _execute(self, submission: _Submission) -> None:
        results = {}
        count = 0
        total = sum(len(stage) for stage in submission.pipeline.stages)
        base = ExecutionEvent(
            submission.run_id,
            "started",
            label=submission.pipeline.label,
            total=total,
            stages=len(submission.pipeline.stages),
        )
        self._emit(submission, base)
        current = base
        operation = "resolve_outputs"
        try:
            # Only executed producers acquire resources. The stack releases them
            # in reverse order on success, task failure, cancellation, or close.
            terminal = None
            with ExitStack() as resources:
                for index, stage in enumerate(submission.pipeline.stages):
                    for task in stage:
                        if submission.cancelled.is_set():
                            terminal = current._replace(
                                kind="cancelled", completed=count
                            )
                            break
                        current = base._replace(
                            kind="task_started",
                            task_id=task.task_id,
                            label=task.label or task.operation,
                            completed=count,
                            stage=index + 1,
                        )
                        self._emit(submission, current)
                        operation = task.operation
                        inputs = TaskValues.map(
                            task.inputs, lambda ref: TaskValues.resolve(ref, results)
                        )
                        result = self._operations[task.operation](inputs)
                        if task.manage_resource:
                            result = self._manage(resources, result)
                        results[task.task_id] = result
                        count += 1
                        self._emit(
                            submission,
                            current._replace(
                                kind="task_completed", completed=count, result=result
                            ),
                        )
                    if terminal is not None:
                        break
                # A cancel during the final operation still cancels the run. Keep
                # the event private until the resource stack has completed cleanup.
                if submission.cancelled.is_set():
                    terminal = current._replace(kind="cancelled", completed=count)
                else:
                    operation = "resolve_outputs"
                    outputs = tuple(
                        TaskValues.resolve(ref, results)
                        for ref in submission.pipeline.outputs
                    )
                    terminal = base._replace(
                        kind="completed",
                        completed=count,
                        stage=len(submission.pipeline.stages),
                        result=outputs,
                    )
                operation = "release_resources"
            # Cleanup can itself take time. A cancellation received while the
            # resource stack was closing still precedes terminal publication.
            if submission.cancelled.is_set():
                terminal = current._replace(kind="cancelled", completed=count)
            self._emit(submission, terminal)
        except Exception as error:  # noqa: BLE001 -- isolate arbitrary registered operation failures
            failure = Failure(operation, str(error), format_exc())
            self._emit(
                submission,
                current._replace(kind="failed", completed=count, failure=failure),
            )
