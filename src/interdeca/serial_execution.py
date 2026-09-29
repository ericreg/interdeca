"""FIFO implementation of the execution interface with one private worker."""

from collections.abc import Callable, Mapping
from queue import Queue
from threading import Event, Lock, Thread
from traceback import format_exc
from typing import NamedTuple
from uuid import uuid4

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

    def submit(self, pipeline: Pipeline, observer: Observer) -> RunId:
        """Snapshot and validate inputs before accepting a run."""
        frozen = TaskValues.snapshot(pipeline, self._operations)
        with self._lock:
            if self._closed:
                raise RuntimeError("Execution manager is closed.")
            run_id = uuid4().hex
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
            for index, stage in enumerate(submission.pipeline.stages):
                for task in stage:
                    if submission.cancelled.is_set():
                        self._emit(
                            submission,
                            current._replace(kind="cancelled", completed=count),
                        )
                        return
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
                    results[task.task_id] = result
                    count += 1
                    self._emit(
                        submission,
                        current._replace(
                            kind="task_completed", completed=count, result=result
                        ),
                    )
            # A cancel arriving during the final operation still remains a cancelled run.
            if submission.cancelled.is_set():
                self._emit(
                    submission, current._replace(kind="cancelled", completed=count)
                )
            else:
                operation = "resolve_outputs"
                outputs = tuple(
                    TaskValues.resolve(ref, results)
                    for ref in submission.pipeline.outputs
                )
                self._emit(
                    submission,
                    base._replace(
                        kind="completed",
                        completed=count,
                        stage=len(submission.pipeline.stages),
                        result=outputs,
                    ),
                )
        except Exception as error:  # noqa: BLE001 -- isolate arbitrary registered operation failures
            failure = Failure(operation, str(error), format_exc())
            self._emit(
                submission,
                current._replace(kind="failed", completed=count, failure=failure),
            )
