"""MVC controller shared by Slicer and future standalone presentation hosts."""

from typing import Any, NamedTuple, Protocol

import numpy as np

from interdeca.application_models import (
    ApplicationInputs,
    ApplicationState,
    PlotData,
)
from interdeca.execution import ExecutionEvent, ExecutionManager, Pipeline
from interdeca.models import Arrays, Mesh
from interdeca.serde import AtlasRequest, ColorRequest, ModelRequest
from interdeca.workflows import Workflows


class ApplicationView(Protocol):
    """Passive view contract; concrete hosts own scene and window lifecycles."""

    def render(self, state: ApplicationState) -> None: ...
    def present(self, action: str, result: Any) -> None: ...
    def show_error(self, message: str, details: str = "") -> None: ...


class CompletedTask(NamedTuple):
    """Partial results remain inspectable after a failed or cancelled run."""

    task_id: str
    result: Any


class ApplicationController:
    """Own one session; execution is accessed only through its public protocol."""

    def __init__(self, manager: ExecutionManager, view: ApplicationView):
        self._manager = manager
        self._view = view
        self.state = ApplicationState()
        self.completed_tasks: tuple[CompletedTask, ...] = ()
        self._closed = False
        self._pending_morph: np.ndarray | None = None
        self._morph_generation = 0
        self._axes = (0, 1)

    def _render(self) -> None:
        if not self._closed:
            self._view.render(self.state)

    def start(
        self,
        action: str,
        pipeline: Pipeline,
        inputs: ApplicationInputs | None = None,
    ) -> None:
        """Submit a snapshot and associate every notification with its input revision."""
        if self._closed:
            raise RuntimeError("The application session is closed.")
        if self.state.run_id is not None:
            raise RuntimeError("An operation is already running.")
        revision, morph_generation = self.state.revision, self._morph_generation
        self.completed_tasks = ()
        # The dispatcher contract guarantees notification delivery after submit returns.
        run_id = self._manager.submit(
            pipeline, lambda event: self._on_event(event, revision, morph_generation)
        )
        self.state = self.state._replace(
            run_id=run_id,
            action=action,
            status="queued",
            completed=0,
            total=0,
            stage=0,
            stages=len(pipeline.stages),
            message=pipeline.label,
            inputs=self.state.inputs if inputs is None else inputs,
        )
        self._render()

    def run_atlas(self, request: AtlasRequest) -> None:
        """Validate a request and submit the atlas workflow."""
        options = request.to_internal()
        self.start(
            "atlas",
            Workflows.atlas(options),
            self.state.inputs._replace(atlas=options),
        )

    def process_colors(self, mesh: Mesh, request: ColorRequest) -> None:
        """Process a snapshot of the selected mesh and its texture population."""
        # Store immutable input records alongside results so another view can restore a session.
        mesh, options = mesh.validated(), request.to_internal()
        self.start(
            "colors",
            Workflows.colors(mesh, options),
            self.state.inputs._replace(color_mesh=mesh, colors=options),
        )

    def fit_population(self, request: ModelRequest) -> None:
        """Fit the requested model using completed color results."""
        if self.state.colors is None:
            raise ValueError("Run color processing first.")
        options = request.to_internal()
        self.start(
            "population",
            Workflows.population(self.state.colors, options),
            self.state.inputs._replace(population=options),
        )

    def show_specimen(self, specimen_id: str) -> None:
        """Reuse processed samples to prepare one specimen for display."""
        if self.state.colors is None:
            raise ValueError("Run color processing first.")
        self.start(
            "display", Workflows.single("display", self.state.colors, specimen_id)
        )

    def select_region(
        self,
        mesh: Mesh,
        points: np.ndarray,
        same_side_only: bool,
        vertex_colors: np.ndarray | None = None,
    ) -> None:
        """Select a region from plain curve coordinates and canonical mesh data."""
        mesh, points = mesh.validated(), Arrays.frozen(points)
        self.start(
            "selection",
            Workflows.single(
                "select",
                mesh,
                points,
                same_side_only=same_side_only,
                vertex_colors=vertex_colors,
                previous=self.state.selection,
            ),
            self.state.inputs._replace(
                selection_mesh=mesh,
                curve_points=points,
                same_side_only=same_side_only,
            ),
        )

    def export_region(self, mesh: Mesh) -> None:
        """Extract the accumulated selection without reading scene objects."""
        if self.state.selection is None:
            raise ValueError("Apply a region selection first.")
        self.start("export", Workflows.single("extract", mesh, self.state.selection))

    def clear_selection(self) -> None:
        """Clear numerical selection state and restore the normal mesh display."""
        self.invalidate("selection")
        self._view.present("clear_selection", None)

    def invalidate(self, scope: str = "colors") -> None:
        """Invalidate dependents after data edits, while display changes stay local."""
        self._pending_morph = None
        self._morph_generation += 1
        self.cancel()
        changes = {"revision": self.state.revision + 1}
        if scope in ("colors", "mesh", "all"):
            changes.update(colors=None, population=None)
        if scope in ("population", "all"):
            changes.update(population=None)
        if scope in ("selection", "mesh", "all"):
            changes.update(selection=None)
        if scope == "all":
            changes.update(atlas=None, inputs=ApplicationInputs())
        # _replace is the record-update boundary; no mutable state dictionary is retained.
        self.state = self.state._replace(**changes)
        self._render()

    def cancel(self) -> None:
        """Request cancellation without waiting for the active operation."""
        self._pending_morph = None
        if self.state.run_id is not None:
            self._manager.cancel(self.state.run_id)
            self.state = self.state._replace(
                status="cancelling", message="Cancelling after the active task finishes"
            )
            self._render()

    def set_axes(self, x_axis: int, y_axis: int) -> None:
        """Change presentation axes while reusing the fitted population scores."""
        population = self.state.population
        if population is None:
            return
        count = population.model.scores.shape[1]
        if not 0 <= x_axis < count or not 0 <= y_axis < count:
            raise ValueError("Plot axes must refer to fitted components.")
        self._axes = (x_axis, y_axis)
        self._view.present(
            "plot",
            PlotData(
                population.model.specimen_ids,
                population.model.scores,
                population.model.method,
                x_axis,
                y_axis,
            ),
        )

    def explore(self, coordinates: np.ndarray) -> None:
        """Keep only the newest slider request while a reconstruction is running."""
        if self.state.population is None or self.state.colors is None:
            raise ValueError("Fit a population model first.")
        self._morph_generation += 1
        if self.state.run_id is not None:
            if self.state.action != "morphospace":
                raise RuntimeError("Wait for the current analysis to finish.")
            self._pending_morph = np.asarray(coordinates, dtype=float).copy()
            self._manager.cancel(self.state.run_id)
            return
        self.start(
            "morphospace",
            Workflows.single(
                "reconstruct", self.state.population, self.state.colors, coordinates
            ),
        )

    def _on_event(
        self, event: ExecutionEvent, revision: int, morph_generation: int
    ) -> None:
        if self._closed or event.run_id != self.state.run_id:
            return
        # A completed event may already be queued when Cancel is clicked. The UI's
        # cancellation decision still prevents that delayed result from being applied.
        stale = (
            self.state.status == "cancelling"
            or revision != self.state.revision
            or (
                self.state.action == "morphospace"
                and morph_generation != self._morph_generation
            )
        )
        terminal = event.kind in ("completed", "failed", "cancelled")
        if event.kind == "task_completed":
            self.completed_tasks += (CompletedTask(event.task_id, event.result),)
        if not terminal:
            if not stale:
                status = (
                    self.state.status
                    if self.state.status == "cancelling"
                    else "running"
                )
                self.state = self.state._replace(
                    status=status,
                    completed=event.completed,
                    total=event.total,
                    stage=event.stage,
                    stages=event.stages,
                    message=event.label,
                )
                self._render()
            return
        action = self.state.action
        self.state = self.state._replace(
            run_id=None,
            status="cancelled" if stale else event.kind,
            completed=event.completed,
            total=event.total,
            stage=event.stage,
            stages=event.stages,
            message=event.label,
        )
        if not stale and event.kind == "completed":
            result = event.result[0]
            # Only successful complete workflows replace the last useful session result.
            if action == "atlas":
                self.state = self.state._replace(
                    atlas=result, colors=None, population=None, selection=None
                )
            elif action == "colors":
                self.state = self.state._replace(colors=result, population=None)
            elif action == "population":
                self.state = self.state._replace(population=result)
            elif action == "selection":
                self.state = self.state._replace(selection=result)
            self._view.present(action, result)
            if action == "population":
                self.set_axes(0, min(1, result.model.scores.shape[1] - 1))
        elif not stale and event.failure is not None:
            self._view.show_error(
                f"{event.failure.operation}: {event.failure.message}",
                event.failure.traceback,
            )
        self._render()
        # A newer request replaces the obsolete display result without restarting fitting.
        pending, self._pending_morph = self._pending_morph, None
        if pending is not None and self.state.population is not None:
            self.explore(pending)

    def close(self) -> None:
        """Detach the view before asking the execution implementation to shut down."""
        self._closed = True
        self._pending_morph = None
        self._manager.close()
