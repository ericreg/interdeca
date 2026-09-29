"""Slicer composition root: the only UI module that constructs an executor."""

import logging

import qt
import vtk

import slicer
from interdeca.controller import ApplicationController
from interdeca.serial_execution import SerialExecutionManager
from interdeca.workflows import Operations
from interdeca_slicer.atlas_tab import AtlasTab
from interdeca_slicer.color_tab import ColorTab
from interdeca_slicer.dispatch import SlicerDispatcher
from interdeca_slicer.scene import PopulationPlot, SceneBridge
from interdeca_slicer.selection_tab import SelectionTab

logger = logging.getLogger(__name__)


class SlicerApplication:
    """Bind passive workflow panels to one shared application session."""

    def __init__(self):
        self.widget = qt.QWidget()
        layout = qt.QVBoxLayout(self.widget)
        self.settings = qt.QSettings()
        self.settings.beginGroup("InterDeCANext")
        self.dispatcher = SlicerDispatcher()
        self.controller = ApplicationController(
            SerialExecutionManager(Operations.registry(), self.dispatcher), self
        )
        self.plot = PopulationPlot()
        self._ready = False
        self._closed = False
        self._presenting = False
        self._source = None
        self._observers = []
        self._last_message = ""
        self.tabs = qt.QTabWidget()
        layout.addWidget(self.tabs)
        self.atlas = AtlasTab(self)
        self.selection = SelectionTab(self)
        self.colors = ColorTab(self)
        for panel, label in (
            (self.atlas, "ATLAS"),
            (self.selection, "Mesh Selection"),
            (self.colors, "MultiRecolor"),
        ):
            scroll = qt.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(panel.widget)
            self.tabs.addTab(scroll, label)
        self.progress = qt.QProgressBar()
        self.cancel = qt.QPushButton("Cancel Operation")
        self.cancel.connect("clicked()", self.controller.cancel)
        self.log = qt.QPlainTextEdit()
        self.log.readOnly = True
        self.log.maximumHeight = 140
        layout.addWidget(self.progress)
        layout.addWidget(self.cancel)
        layout.addWidget(self.log)
        self._scene_observer = slicer.mrmlScene.AddObserver(
            slicer.vtkMRMLScene.StartCloseEvent, self.scene_closing
        )
        self._ready = True
        self.render(self.controller.state)
        self.watch_sources()

    def guard(self, function, *args):
        """Show contextual action failures instead of letting them escape Qt callbacks."""
        try:
            return function(*args)
        except Exception as error:
            logger.exception("InterDeCA action failed")
            self.show_error(str(error))

    def render(self, state):
        """Reflect immutable session state in controls, task counts, and logs."""
        if not self._ready or self._closed:
            return
        busy = state.run_id is not None
        self.dispatcher.set_active(busy)
        self.atlas.render(busy)
        self.selection.render(state)
        self.colors.render(state)
        self.progress.visible = busy
        self.cancel.visible = busy
        self.cancel.enabled = state.status != "cancelling"
        self.progress.setRange(0, max(state.total, 1))
        self.progress.setValue(state.completed)
        self.progress.setFormat(
            f"Stage {state.stage}/{state.stages} · "
            f"{state.completed}/{state.total} tasks — {state.status}"
        )
        message = f"{state.status}: {state.message}"
        if message != self._last_message:
            self.log.appendPlainText(message)
            self._last_message = message

    def show_error(self, message, details=""):
        """Show an actionable message while retaining detailed diagnostics in the log."""
        if self._closed:
            return
        self.log.appendPlainText(message)
        if details:
            logger.error(details)
        slicer.util.errorDisplay(message)

    def remember_source(self, snapshot):
        """Associate the next result with the geometry snapshot that produced it."""
        self._source = snapshot

    def source_current(self):
        """Reject results after source geometry changes or its node disappears."""
        if self._source is None:
            return True
        node = slicer.mrmlScene.GetNodeByID(self._source.node_id)
        return (
            node is not None
            and SceneBridge.snapshot(node).signature == self._source.signature
        )

    def present(self, action, result):
        """Apply a completed result to the scene on the application thread."""
        if self._closed:
            return
        # Recheck identity at delivery, even if no scene observer fired during native work.
        if (
            action not in ("atlas", "plot", "clear_selection")
            and not self.source_current()
        ):
            self.controller.invalidate("mesh")
            self.log.appendPlainText(
                "Discarded a result because its source mesh changed."
            )
            return
        self._presenting = True
        try:
            if action == "atlas":
                node = SceneBridge.create_model(result.atlas.mesh, "ATLAS Model")
                self.colors.model.setCurrentNode(node)
                self.selection.model.setCurrentNode(node)
                # An atlas without new bakes must not inherit textures from an older UV map.
                self.colors.directory.setCurrentPath(
                    str(result.directory / "textures") if result.textures else ""
                )
                if result.average is not None:
                    SceneBridge.texture(node, result.average)
                self.tabs.setCurrentIndex(2)
                SceneBridge.focus(node)
                self.log.appendPlainText(f"Saved results: {result.directory}")
            elif action == "colors":
                self.colors.refresh_textures()
            elif action in ("display", "morphospace"):
                SceneBridge.colors(self.colors.model.currentNode(), result)
                if action == "display":
                    SceneBridge.focus(self.colors.model.currentNode())
            elif action == "population":
                self.colors.population_ready(result)
            elif action == "plot":
                self.plot.show(result)
            elif action == "selection":
                if self.selection.appearance is None:
                    self.selection.appearance_node = self.selection.model.currentNode()
                    self.selection.appearance = SceneBridge.capture_display(
                        self.selection.appearance_node
                    )
                SceneBridge.highlight(self.selection.model.currentNode(), result)
                SceneBridge.focus(self.selection.model.currentNode())
            elif action == "clear_selection":
                self.selection.restore_appearance()
            elif action == "export":
                node = SceneBridge.export(
                    self.selection.model.currentNode(),
                    self.selection.export_snapshot,
                    result,
                    self.selection.name.text.strip() or "SelectedRegion",
                    self.selection.appearance,
                )
                SceneBridge.focus(node)
        finally:
            self._presenting = False

    def watch_sources(self):
        """Observe selected geometry while keeping display edits out of invalidation."""
        if not self._ready:
            return
        for obj, tag in self._observers:
            obj.RemoveObserver(tag)
        self._observers = []
        seen = set()
        for node in (
            self.selection.model.currentNode(),
            self.colors.model.currentNode(),
        ):
            if node is None or node.GetID() in seen or node.GetPolyData() is None:
                continue
            seen.add(node.GetID())
            try:
                snapshot = SceneBridge.snapshot(node)
            except ValueError:
                continue
            # Capture immutable geometry identity; scalar/display edits leave it unchanged.
            callback = lambda *_args, source=snapshot: self.geometry_changed(source)
            for obj in (node, node.GetPolyData(), node.GetParentTransformNode()):
                if obj is not None:
                    self._observers.append(
                        (obj, obj.AddObserver(vtk.vtkCommand.ModifiedEvent, callback))
                    )

    def geometry_changed(self, snapshot):
        """Invalidate numerical dependents when a selected surface snapshot changes."""
        if self._closed or self._presenting:
            return
        node = slicer.mrmlScene.GetNodeByID(snapshot.node_id)
        try:
            changed = (
                node is None
                or SceneBridge.snapshot(node).signature != snapshot.signature
            )
        except ValueError:
            changed = True
        if changed:
            self.controller.invalidate("mesh")
            self.watch_sources()

    def scene_closing(self, *args):
        """Cancel work and clear session references before a scene is removed."""
        self.controller.invalidate("all")
        self._source = None
        self.plot.clear()

    def close(self):
        """Release MRML-backed Qt models before Slicer destroys the scene."""
        if self._closed:
            return
        self._closed = True
        self.controller.close()
        self.dispatcher.close()
        self.colors.close()
        self.selection.close()
        for obj, tag in self._observers:
            obj.RemoveObserver(tag)
        slicer.mrmlScene.RemoveObserver(self._scene_observer)
        # Scene-backed combo boxes otherwise retain Qt model/accessibility references
        # into MRML while the scene is being destroyed during application shutdown.
        for selector in (self.colors.model, self.selection.model, self.selection.curve):
            selector.blockSignals(True)
            selector.setMRMLScene(None)
        self.plot.clear()
        self.widget.hide()
        self.widget.deleteLater()
