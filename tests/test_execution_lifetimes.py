"""Explicit resource scopes and ordinary array ownership are different contracts."""

import weakref
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any, NamedTuple

import numpy as np
import pytest

from interdeca import application, serial_execution
from interdeca.application import ApplicationSession
from interdeca.execution import ExecutionEvent, Pipeline, ResultRef, TaskValues
from interdeca.serial_execution import SerialExecutionManager
from interdeca.workflows import Workflows


class Dispatcher(NamedTuple):
    callbacks: Queue[Callable]

    def post(self, callback):
        self.callbacks.put(callback)


class Driver(NamedTuple):
    manager: SerialExecutionManager
    callbacks: Queue[Callable]
    events: list[ExecutionEvent]

    def observe(self, event):
        # This observer keeps final results but does not retain task intermediates.
        # Controller diagnostic retention is tested separately.
        self.events.append(
            event._replace(result=None) if event.kind == "task_completed" else event
        )

    def drain(self):
        while not self.callbacks.empty():
            self.callbacks.get_nowait()()

    def terminal(self, run_id):
        while True:
            for event in self.events:
                if event.run_id == run_id and event.kind in (
                    "completed", "failed", "cancelled"
                ):
                    return event
            self.callbacks.get(timeout=30)()


class Nested(NamedTuple):
    first: Any
    rest: tuple


class ObservedQueue(Queue):
    """Signal when the owning thread waits after processing its first item."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.idle = Event()

    def get(self, *args, **kwargs):
        self.calls += 1
        if self.calls > 1:
            self.idle.set()
        return super().get(*args, **kwargs)


@pytest.fixture
def driver_factory():
    drivers = []

    def create(operations):
        callbacks = Queue()
        driver = Driver(
            SerialExecutionManager(operations, Dispatcher(callbacks)), callbacks, []
        )
        drivers.append(driver)
        return driver

    yield create
    for driver in drivers:
        driver.manager.close()
        assert driver.manager.join(10)
        driver.drain()


def test_nested_fanout_releases_last_input_and_keeps_final_outputs(driver_factory):
    references = []
    reached, proceed = Event(), Event()

    def produce(inputs):
        data = np.arange(4096, dtype=float)
        references.append(weakref.ref(data))
        return data

    def right(inputs):
        nested = inputs.arguments[0]
        assert nested.first is nested.rest[0]
        return np.array([nested.first.sum() + nested.rest[0].sum()])

    def check(inputs):
        reached.set()
        assert proceed.wait(10)
        assert references[0]() is None
        return float(inputs.arguments[0][0] + inputs.arguments[1][0])

    driver = driver_factory({
        "produce": produce,
        "left": lambda inputs: np.array([inputs.arguments[0].sum()]),
        "right": right,
        "check": check,
    })
    task, ref = Workflows.task, ResultRef
    pipeline = Pipeline(
        (
            (task("source", "produce"),),
            (
                task("left", "left", ref("source")),
                task("right", "right", Nested(ref("source"), (ref("source"),))),
            ),
            (task("check", "check", ref("left"), ref("right")),),
        ),
        (ref("check"), ref("left")),
    )
    driver.manager.submit(pipeline, driver.observe, run_id="fanout")
    try:
        assert reached.wait(10)
        # Drain queued task events so the assertion tests the manager's ownership,
        # without pretending an undelivered observer event has released its data.
        driver.drain()
    finally:
        proceed.set()
    terminal = driver.terminal("fanout")
    assert terminal.kind == "completed", terminal.failure
    expected = np.arange(4096, dtype=float).sum()
    assert terminal.result[0] == 3 * expected
    np.testing.assert_array_equal(terminal.result[1], [expected])


def test_idle_worker_and_queued_notifications_release_input_snapshot(
    monkeypatch, driver_factory
):
    monkeypatch.setattr(serial_execution, "Queue", ObservedQueue)
    references = []

    def consume(inputs):
        references.append(weakref.ref(inputs.arguments[0]))
        return float(inputs.arguments[0].sum())

    driver = driver_factory({"consume": consume})
    source = np.arange(4096, dtype=float)
    pipeline = Workflows.single("consume", source)
    driver.manager.submit(pipeline, driver.observe, run_id="idle")
    assert driver.manager._queue.idle.wait(10)
    # No callbacks have been delivered yet. Neither the idle worker nor those
    # queued closures should keep the completed pipeline's private input copy.
    assert references[0]() is None
    assert driver.terminal("idle").result == (float(source.sum()),)


def test_idle_application_releases_delivered_callback(monkeypatch, tmp_path):
    monkeypatch.setattr(application, "Queue", ObservedQueue)
    session = ApplicationSession(tmp_path, Queue())

    class Callback:
        def __init__(self):
            self.values = np.zeros(4096)

        def __call__(self):
            assert self.values.shape == (4096,)

    callback = Callback()
    reference = weakref.ref(callback)
    session.post(callback)
    del callback
    try:
        assert session.queue.idle.wait(10)
        assert reference() is None
    finally:
        session.close()
        session.thread.join(10)
        assert not session.thread.is_alive()
        assert session.execution.manager.join(10)


def test_explicit_boundary_closes_once_in_reverse_order_before_later_work(
    tmp_path, driver_factory
):
    roots, closed = [], []

    @contextmanager
    def acquire(name):
        with TemporaryDirectory(dir=tmp_path) as directory:
            root = Path(directory)
            roots.append(root)
            (root / "value").write_text(name)
            yield root
        closed.append(name)

    def consume(inputs):
        return tuple((root / "value").read_text() for root in inputs.arguments)

    def later(inputs):
        assert closed == ["second", "first"]
        assert all(not root.exists() for root in roots)
        return inputs.arguments[0]

    driver = driver_factory({
        "acquire": lambda inputs: acquire(inputs.arguments[0]),
        "consume": consume,
        "later": later,
    })
    task, ref = Workflows.task, ResultRef
    pipeline = Pipeline(
        (
            tuple(task(name, "acquire", name)._replace(
                manage_resource=True, release_after="consume"
            ) for name in ("first", "second")),
            (task("consume", "consume", ref("first"), ref("second")),),
            (task("later", "later", ref("consume")),),
        ),
        (ref("later"),),
    )
    driver.manager.submit(pipeline, driver.observe, run_id="boundary")
    terminal = driver.terminal("boundary")
    assert terminal.kind == "completed", terminal.failure
    assert terminal.result == (("first", "second"),)
    assert closed == ["second", "first"]


def test_default_resource_scope_preserves_forwarded_paths(tmp_path, driver_factory):
    roots = []

    @contextmanager
    def acquire():
        with TemporaryDirectory(dir=tmp_path) as directory:
            root = Path(directory)
            roots.append(root)
            (root / "value").write_text("indirect consumer")
            yield root

    driver = driver_factory({
        "acquire": lambda inputs: acquire(),
        "forward": lambda inputs: inputs.arguments[0] / "value",
        "read": lambda inputs: inputs.arguments[0].read_text(),
    })
    task, ref = Workflows.task, ResultRef
    pipeline = Pipeline(
        (
            (task("resource", "acquire")._replace(manage_resource=True),),
            (task("forward", "forward", ref("resource")),),
            (task("read", "read", ref("forward")),),
        ),
        (ref("read"),),
    )
    driver.manager.submit(pipeline, driver.observe, run_id="forwarded")
    terminal = driver.terminal("forwarded")
    assert terminal.kind == "completed", terminal.failure
    assert terminal.result == ("indirect consumer",)
    assert not roots[0].exists()


@pytest.mark.parametrize("failure", ["operation", "cleanup"])
def test_early_scope_failure_keeps_diagnostics_and_stops_later_work(
    tmp_path, driver_factory, failure
):
    roots, received = [], []

    @contextmanager
    def acquire():
        with TemporaryDirectory(dir=tmp_path) as directory:
            roots.append(Path(directory))
            try:
                yield roots[-1]
            except ValueError as error:
                received.append(str(error))
                # Even a suppressing resource cannot hide operation failure.
        if failure == "cleanup":
            raise OSError("cleanup failed")

    def consume(inputs):
        assert inputs.arguments[0].is_dir()
        if failure == "operation":
            raise ValueError("operation failed")
        return "owned result"

    driver = driver_factory({
        "acquire": lambda inputs: acquire(),
        "consume": consume,
        "later": lambda inputs: pytest.fail("Work continued after failure"),
    })
    task, ref = Workflows.task, ResultRef
    pipeline = Pipeline(
        (
            (task("resource", "acquire")._replace(
                manage_resource=True, release_after="consume"
            ),),
            (task("consume", "consume", ref("resource")),),
            (task("later", "later", ref("consume")),),
        ),
        (ref("later"),),
    )
    driver.manager.submit(pipeline, driver.observe, run_id="failure")
    terminal = driver.terminal("failure")
    assert terminal.kind == "failed"
    assert terminal.failure.message == f"{failure} failed"
    assert terminal.failure.operation == (
        "consume" if failure == "operation" else "release_resources"
    )
    assert received == (["operation failed"] if failure == "operation" else [])
    assert all(not root.exists() for root in roots)


@pytest.mark.parametrize("close", [False, True])
def test_early_scope_cancel_or_close_waits_for_active_consumer(
    tmp_path, driver_factory, close
):
    roots, closed = [], []
    entered, proceed = Event(), Event()

    @contextmanager
    def acquire():
        with TemporaryDirectory(dir=tmp_path) as directory:
            roots.append(Path(directory))
            yield roots[-1]
        closed.append(True)

    def consume(inputs):
        entered.set()
        assert proceed.wait(10)
        assert inputs.arguments[0].is_dir()
        return "owned"

    driver = driver_factory({
        "acquire": lambda inputs: acquire(), "consume": consume,
        "later": lambda inputs: pytest.fail("Cancelled work continued"),
    })
    task, ref = Workflows.task, ResultRef
    pipeline = Pipeline(
        (
            (task("resource", "acquire")._replace(
                manage_resource=True, release_after="consume"
            ),),
            (task("consume", "consume", ref("resource")),),
            (task("later", "later", ref("consume")),),
        ),
        (ref("later"),),
    )
    driver.manager.submit(pipeline, driver.observe, run_id="cancel")
    try:
        assert entered.wait(10)
        if close:
            driver.manager.close()
        else:
            driver.manager.cancel("cancel")
        assert roots[0].is_dir()
    finally:
        proceed.set()
    if close:
        assert driver.manager.join(10)
        driver.drain()
        assert driver.events == []
    else:
        assert driver.terminal("cancel").kind == "cancelled"
    assert closed == [True]
    assert not roots[0].exists()


@pytest.mark.parametrize("case", ["ordinary", "missing", "earlier", "later", "output", "empty", "type"])
def test_invalid_resource_boundaries_fail_before_execution(case):
    task, ref = Workflows.task, ResultRef
    resource = task("resource", "noop")._replace(
        manage_resource=case != "ordinary",
        release_after={"missing": "unknown", "earlier": "before", "empty": "", "type": 3}.get(case, "use"),
    )
    pipeline = Pipeline(
        (
            (task("before", "noop"),),
            (resource,),
            (task("use", "noop", ref("resource")),),
            (task("after", "noop", ref("resource") if case == "later" else ref("use")),),
        ),
        (ref("resource") if case == "output" else ref("after"),),
    )
    with pytest.raises((TypeError, ValueError), match="release|boundary"):
        TaskValues.snapshot(pipeline, {"noop": lambda inputs: None})


def test_nested_final_reference_survives_last_task_consumer(driver_factory):
    driver = driver_factory({
        "produce": lambda inputs: Nested(np.arange(8), (np.arange(4),)),
        "consume": lambda inputs: int(inputs.arguments[0].sum()),
    })
    task, ref = Workflows.task, ResultRef
    pipeline = Pipeline(
        ((task("source", "produce"),),
         (task("consume", "consume", ref("source", ("first",))),)),
        (ref("consume"), ref("source", ("rest", 0))),
    )
    driver.manager.submit(pipeline, driver.observe, run_id="final-path")
    terminal = driver.terminal("final-path")
    assert terminal.kind == "completed", terminal.failure
    assert terminal.result[0] == 28
    np.testing.assert_array_equal(terminal.result[1], np.arange(4))


def test_queued_cancellation_does_not_acquire_early_scope(driver_factory):
    entered, proceed = Event(), Event()

    def block(inputs):
        entered.set()
        assert proceed.wait(10)

    driver = driver_factory({
        "block": block,
        "acquire": lambda inputs: pytest.fail("Cancelled scope was acquired"),
        "use": lambda inputs: None,
    })
    task, ref = Workflows.task, ResultRef
    queued = Pipeline(
        ((task("resource", "acquire")._replace(manage_resource=True, release_after="use"),),
         (task("use", "use", ref("resource")),)),
        (ref("use"),),
    )
    driver.manager.submit(Workflows.single("block"), driver.observe, run_id="block")
    try:
        assert entered.wait(10)
        driver.manager.submit(queued, driver.observe, run_id="queued")
        driver.manager.cancel("queued")
    finally:
        proceed.set()
    assert driver.terminal("block").kind == "completed"
    assert driver.terminal("queued").kind == "cancelled"
