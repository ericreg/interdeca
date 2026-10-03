"""Keep useful session results and failure diagnostics, not successful intermediates."""

import weakref
from queue import Queue
from typing import NamedTuple

import numpy as np
import pytest

from interdeca.application_models import ColorOptions, ColorResult, PopulationResult
from interdeca.controller import ApplicationController
from interdeca.execution import ExecutionEvent, Failure
from interdeca.models import Mesh, ProcessedColors
from interdeca.population import Population
from interdeca.serde import ModelRequest
from interdeca.serial_execution import SerialExecutionManager
from interdeca.workflows import Operations, Workflows


class PassiveManager(NamedTuple):
    calls: list[str]

    def submit(self, pipeline, observer, *, run_id):
        self.calls.append(run_id)
        return run_id

    def cancel(self, run_id):
        self.calls.append("cancel:" + run_id)

    def close(self):
        self.calls.append("close")


class View:
    """Store presentation identity/status without retaining numerical history."""

    def __init__(self):
        self.actions = []
        self.errors = []
        self.fail = None

    def render(self, state):
        if self.fail == "render" and state.status == "completed":
            raise OSError("render failed")

    def present(self, action, result):
        if self.fail in ("present", action):
            raise OSError("presentation failed")
        self.actions.append(action)

    def show_error(self, message, details):
        self.errors.append((message, details))


class Dispatcher(NamedTuple):
    queue: Queue

    def post(self, callback):
        self.queue.put(callback)


def partial_controller():
    controller = ApplicationController(PassiveManager([]), View())
    controller.start("display", Workflows.single("noop"), run_id="run")
    buffer = np.arange(4096, dtype=float)
    reference = weakref.ref(buffer)
    controller._on_event(
        ExecutionEvent("run", "task_completed", task_id="partial", result=buffer), 0
    )
    return controller, reference


def test_successful_publication_releases_partial_buffers():
    controller, reference = partial_controller()
    controller._on_event(ExecutionEvent("run", "completed", result=("display",)), 0)
    assert controller.state.status == "completed"
    assert controller._view.actions == ["display"]
    assert controller.completed_tasks == ()
    assert reference() is None


@pytest.mark.parametrize("kind", ["failed", "cancelled", "stale_revision", "stale_cancel"])
def test_failure_and_cancellation_keep_partial_diagnostics(kind):
    controller, reference = partial_controller()
    if kind == "stale_revision":
        controller.invalidate("all")
    elif kind == "stale_cancel":
        controller.cancel()
    event = ExecutionEvent(
        "run", "completed" if kind.startswith("stale") else kind,
        result=("display",),
        failure=Failure("operation", "failed", "trace") if kind == "failed" else None,
    )
    controller._on_event(event, 0)
    assert reference() is not None
    assert controller.completed_tasks[0].task_id == "partial"
    assert controller._view.actions == []
    assert controller.state.status == ("failed" if kind == "failed" else "cancelled")
    controller.start("display", Workflows.single("noop"), run_id="next")
    assert reference() is None


@pytest.mark.parametrize("failure", ["present", "render"])
def test_publication_failure_does_not_discard_diagnostics(failure):
    controller, reference = partial_controller()
    controller._view.fail = failure
    with pytest.raises(OSError):
        controller._on_event(ExecutionEvent("run", "completed", result=("display",)), 0)
    assert reference() is not None
    assert len(controller.completed_tasks) == 1


@pytest.fixture
def colors(tmp_path):
    mesh = Mesh(
        np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., 3.]]),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
        np.arange(24, dtype=float).reshape(4, 3, 2) / 24,
    ).validated()
    analysis = Operations.prepare_analysis(mesh, None)
    rng = np.random.default_rng(42)
    # Four small synthetic specimens cover UMAP's minimum population size without
    # adding real clams or large images to the acceptance dataset.
    specimens = tuple(
        ProcessedColors(
            f"synthetic-{index}", rng.normal([50, 0, 0], [8, 4, 4], (4, 3)),
            rng.uniform(0.2, 0.8, (4, 3)), np.ones(4, dtype=bool), analysis.sampling_key,
        ).validated()
        for index in range(4)
    )
    return ColorResult(
        analysis, specimens, None, ColorOptions(tmp_path, clustering=False, sample_count=None)
    )


@pytest.mark.parametrize("method", ["pca", "umap"])
def test_repeated_population_and_reconstruction_preserve_state_and_results(colors, method):
    # Direct numerical calls use the unchanged core, independently of task result
    # lifetimes and controller publication. Fixed seeds make both fits comparable.
    features = Population.assemble_features(colors.specimens, colors.analysis, missing="drop")
    expected_model = Population.fit_model(features, method=method, n_components=2)
    expected = PopulationResult(features, expected_model)
    coordinates = expected.model.scores[0].copy()
    reconstructed = Operations.reconstruct(expected, colors, coordinates)
    callbacks = Queue()
    manager = SerialExecutionManager(Operations.registry(), Dispatcher(callbacks))
    view = View()
    controller = ApplicationController(manager, view)
    controller.state = controller.state._replace(colors=colors)

    def finish():
        while controller.state.run_id is not None:
            callbacks.get(timeout=60)()
        assert controller.state.status == "completed", view.errors
        assert controller.completed_tasks == ()

    try:
        controller.fit_population(ModelRequest(method=method, components=2), run_id="first")
        finish()
        first = controller.state.population
        np.testing.assert_array_equal(first.model.scores, expected.model.scores)
        assert first.features.feature_names == expected.features.feature_names
        reference = weakref.ref(first.model.estimator)
        del first
        controller.set_axes(1, 0)
        controller.reconstruct(colors.specimens[0].specimen_id, None, run_id="reconstruct")
        finish()
        actual = Operations.reconstruct(controller.state.population, colors, coordinates)
        for before, after in zip(reconstructed, actual, strict=True):
            np.testing.assert_array_equal(before, after)
        controller.fit_population(ModelRequest(method=method, components=2), run_id="second")
        finish()
        assert reference() is None
        np.testing.assert_array_equal(controller.state.population.model.scores, expected.model.scores)
        assert controller.state.colors is colors
        assert view.actions.count("population") == 2
        assert "morphospace" in view.actions
    finally:
        controller.close()
        assert manager.join(10)
