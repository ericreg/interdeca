"""Managed task resources must close before terminal events on every exit path."""

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event
from typing import NamedTuple

import pytest

from interdeca.execution import ExecutionEvent, Pipeline, ResultRef, TaskValues
from interdeca.serial_execution import SerialExecutionManager
from interdeca.workflows import Workflows


class Dispatcher(NamedTuple):
    callbacks: Queue[Callable]

    def post(self, callback):
        self.callbacks.put(callback)


class Harness(NamedTuple):
    manager: SerialExecutionManager
    dispatcher: Dispatcher
    events: list[ExecutionEvent]

    def terminal(self, run_id):
        # Deliver callbacks on the test thread, as a UI/event-loop host would.
        while True:
            for event in self.events:
                if event.run_id == run_id and event.kind in (
                    "completed",
                    "failed",
                    "cancelled",
                ):
                    return event
            self.dispatcher.callbacks.get(timeout=10)()


@pytest.fixture
def harness():
    active = []

    def create(operations):
        dispatcher = Dispatcher(Queue())
        runner = Harness(SerialExecutionManager(operations, dispatcher), dispatcher, [])
        active.append(runner)
        return runner

    yield create
    for runner in active:
        runner.manager.close()
        assert runner.manager.join(10)


def resource_pipeline():
    return Pipeline(
        (
            (Workflows.task("resource", "acquire")._replace(manage_resource=True),),
            (Workflows.task("use", "use", ResultRef("resource")),),
        ),
        (ResultRef("use"),),
    )


def test_success_closes_resources_in_reverse_order_before_notification(
    tmp_path, harness
):
    roots, closed = [], []

    @contextmanager
    def acquire(name):
        with TemporaryDirectory(prefix=name, dir=tmp_path) as directory:
            path = Path(directory)
            roots.append(path)
            (path / "data").write_text(name)
            yield path
        closed.append(name)

    def use(inputs):
        return tuple((root / "data").read_text() for root in inputs.arguments)

    runner = harness(
        {"acquire": lambda inputs: acquire(inputs.arguments[0]), "use": use}
    )
    pipeline = Pipeline(
        (
            tuple(
                Workflows.task(name, "acquire", name)._replace(manage_resource=True)
                for name in ("first", "second")
            ),
            (Workflows.task("use", "use", ResultRef("first"), ResultRef("second")),),
        ),
        (ResultRef("use"),),
    )

    def observe(event):
        if event.kind == "completed":
            assert closed == ["second", "first"]
            assert all(not root.exists() for root in roots)
        runner.events.append(event)

    runner.manager.submit(pipeline, observe, run_id="success")
    assert runner.terminal("success").result == (("first", "second"),)


@pytest.mark.parametrize("suppress", [False, True])
def test_failure_releases_resources_and_keeps_original_diagnostic(
    tmp_path, harness, suppress
):
    roots = []

    @contextmanager
    def acquire():
        with TemporaryDirectory(dir=tmp_path) as directory:
            roots.append(Path(directory))
            try:
                yield roots[-1]
            except ValueError:
                if not suppress:
                    raise

    def fail(inputs):
        assert inputs.arguments[0].is_dir()
        raise ValueError("operation failed")

    runner = harness({"acquire": lambda inputs: acquire(), "use": fail})
    runner.manager.submit(resource_pipeline(), runner.events.append, run_id="failure")
    terminal = runner.terminal("failure")
    assert terminal.kind == "failed"
    assert terminal.failure.operation == "use"
    assert terminal.failure.message == "operation failed"
    assert "ValueError: operation failed" in terminal.failure.traceback
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("close", [False, True])
def test_active_cancel_or_close_keeps_resource_until_operation_finishes(
    tmp_path, harness, close
):
    entered, release = Event(), Event()
    roots = []

    @contextmanager
    def acquire():
        with TemporaryDirectory(dir=tmp_path) as directory:
            roots.append(Path(directory))
            yield roots[-1]

    def use(inputs):
        root = inputs.arguments[0]
        entered.set()
        assert release.wait(10)
        assert root.is_dir()
        return "finished operation"

    runner = harness({"acquire": lambda inputs: acquire(), "use": use})
    runner.manager.submit(resource_pipeline(), runner.events.append, run_id="active")
    assert entered.wait(10)
    if close:
        runner.manager.close()
    else:
        runner.manager.cancel("active")
    # Cancellation is cooperative: cleanup cannot remove files under an active
    # numerical operation, but must finish before its eventual terminal event.
    assert roots[0].is_dir()
    release.set()
    if close:
        assert runner.manager.join(10)
        while not runner.dispatcher.callbacks.empty():
            runner.dispatcher.callbacks.get_nowait()()
        assert runner.events == []
    else:
        assert runner.terminal("active").kind == "cancelled"
    assert not roots[0].exists()


def test_queued_cancellation_does_not_acquire_a_resource(harness):
    entered, release = Event(), Event()

    def block(inputs):
        entered.set()
        assert release.wait(10)

    def forbidden(inputs):
        pytest.fail("A cancelled queued run must never acquire its resources")

    runner = harness({"block": block, "acquire": forbidden, "use": forbidden})
    blocker = Pipeline(((Workflows.task("block", "block"),),), (ResultRef("block"),))
    runner.manager.submit(blocker, runner.events.append, run_id="blocker")
    assert entered.wait(10)
    runner.manager.submit(resource_pipeline(), runner.events.append, run_id="queued")
    runner.manager.cancel("queued")
    release.set()
    assert runner.terminal("blocker").kind == "completed"
    assert runner.terminal("queued").kind == "cancelled"


def test_cleanup_failure_is_not_reported_as_completed(tmp_path, harness):
    roots = []

    @contextmanager
    def acquire():
        with TemporaryDirectory(dir=tmp_path) as directory:
            roots.append(Path(directory))
            yield roots[-1]
        raise OSError("cleanup failed")

    runner = harness(
        {"acquire": lambda inputs: acquire(), "use": lambda inputs: "value"}
    )
    runner.manager.submit(resource_pipeline(), runner.events.append, run_id="cleanup")
    terminal = runner.terminal("cleanup")
    assert terminal.kind == "failed"
    assert terminal.failure.operation == "release_resources"
    assert terminal.failure.message == "cleanup failed"
    assert not roots[0].exists()
    assert not any(event.kind == "completed" for event in runner.events)


def test_resource_flag_is_validated_before_submission():
    task = Workflows.task("acquire", "acquire")._replace(manage_resource="yes")
    with pytest.raises(TypeError, match="boolean"):
        TaskValues.snapshot(
            Pipeline(((task,),), (ResultRef("acquire"),)),
            {"acquire": lambda inputs: None},
        )


def test_cancellation_during_cleanup_is_not_reported_as_completed(tmp_path, harness):
    cleaning, release = Event(), Event()
    roots = []

    @contextmanager
    def acquire():
        with TemporaryDirectory(dir=tmp_path) as directory:
            roots.append(Path(directory))
            yield roots[-1]
            cleaning.set()
            assert release.wait(10)

    runner = harness(
        {"acquire": lambda inputs: acquire(), "use": lambda inputs: "value"}
    )
    runner.manager.submit(
        resource_pipeline(), runner.events.append, run_id="cleanup-cancel"
    )
    try:
        assert cleaning.wait(10)
        runner.manager.cancel("cleanup-cancel")
        release.set()
        assert runner.terminal("cleanup-cancel").kind == "cancelled"
        assert not roots[0].exists()
    finally:
        release.set()
