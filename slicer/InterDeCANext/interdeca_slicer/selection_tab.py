"""Curve interaction is host-specific; selection and extraction are core tasks."""

import qt

import slicer
from interdeca_slicer.scene import SceneBridge
from interdeca_slicer.widgets import Widgets


class SelectionTab:
    """Present the active legacy region-selection workflow."""

    def __init__(self, app):
        self.app = app
        self.widget = qt.QWidget()
        layout = qt.QFormLayout(self.widget)
        section = Widgets.section(layout, "Mesh Region Selection")
        self.model = Widgets.node(section, "Target Mesh:")
        self.curve = Widgets.node(
            section, "Selection Markup:", "vtkMRMLMarkupsClosedCurveNode"
        )
        self.side = Widgets.check(section, "Select only one side:", True)
        self.draw = Widgets.button(section, "Draw Curve", lambda: app.guard(self.place))
        self.clear = Widgets.button(
            section,
            "Clear Selection",
            lambda: app.guard(app.controller.clear_selection),
        )
        self.apply = Widgets.button(
            section, "Apply Selection", lambda: app.guard(self.select)
        )
        self.name = qt.QLineEdit("SelectedRegion")
        section.addRow("Export Name:", self.name)
        self.export = Widgets.button(
            section, "Export Selected Region as Model", lambda: app.guard(self.extract)
        )
        self.info = qt.QLabel("No region selected")
        section.addRow(self.info)
        self.model.connect("currentNodeChanged(vtkMRMLNode*)", self.model_changed)
        self._curve_observers = []
        self.curve.connect(
            "currentNodeChanged(vtkMRMLNode*)",
            self.curve_changed,
        )
        self.side.connect("toggled(bool)", self.curve_edited)
        self.shortcut = qt.QShortcut(qt.QKeySequence("Escape"), self.widget)
        self.shortcut.connect("activated()", self.stop_placing)
        self.export_snapshot = None
        self.appearance = None
        self.appearance_node = None

    def model_changed(self, *args):
        """Release the previous overlay before switching the selection target."""
        self.restore_appearance()
        if not self.app._presenting:
            self.app.controller.invalidate("selection")
        self.app.watch_sources()

    def restore_appearance(self):
        """Selection highlighting must not overwrite a model's previous appearance."""
        if (
            self.appearance_node is not None
            and self.appearance_node.GetScene() is not None
        ):
            SceneBridge.clear_highlight(self.appearance_node)
            SceneBridge.restore_display(self.appearance_node, self.appearance)
        self.appearance = self.appearance_node = None

    def curve_changed(self, *args):
        """Observe only the current curve and release the previous node's callbacks."""
        for node, tag in self._curve_observers:
            node.RemoveObserver(tag)
        self._curve_observers = []
        node = self.curve.currentNode()
        if node is not None:
            for event in (
                slicer.vtkMRMLMarkupsNode.PointModifiedEvent,
                slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
                slicer.vtkMRMLMarkupsNode.PointRemovedEvent,
            ):
                self._curve_observers.append(
                    (node, node.AddObserver(event, self.curve_edited))
                )
        self.curve_edited()

    def curve_edited(self, *args):
        """Discard in-flight selection work while keeping previously applied regions."""
        if (
            self.app.controller.state.action == "selection"
            and self.app.controller.state.run_id
        ):
            self.app.controller.invalidate("curve")
        self.app.render(self.app.controller.state)

    def place(self):
        """Enter Slicer curve-placement mode, creating a curve when necessary."""
        curve = self.curve.currentNode()
        if curve is None:
            curve = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsClosedCurveNode", "SelectionCurve"
            )
            self.curve.setCurrentNode(curve)
        selection = slicer.app.applicationLogic().GetSelectionNode()
        selection.SetActivePlaceNodeID(curve.GetID())
        interaction = slicer.app.applicationLogic().GetInteractionNode()
        interaction.SetPlaceModePersistence(True)
        interaction.SetCurrentInteractionMode(interaction.Place)

    def stop_placing(self):
        """Return to ordinary camera interaction when Escape is pressed."""
        interaction = slicer.app.applicationLogic().GetInteractionNode()
        interaction.SetCurrentInteractionMode(interaction.ViewTransform)

    def select(self):
        """Snapshot the curve and surface before submitting region selection."""
        snapshot = SceneBridge.snapshot(self.model.currentNode())
        self.app.remember_source(snapshot)
        self.app.controller.select_region(
            snapshot.mesh,
            SceneBridge.curve_points(self.curve.currentNode()),
            bool(self.side.checked),
            SceneBridge.vertex_colors(self.model.currentNode(), snapshot),
        )

    def extract(self):
        """Export the applied selection using the current canonical source geometry."""
        self.export_snapshot = SceneBridge.snapshot(self.model.currentNode())
        self.app.remember_source(self.export_snapshot)
        self.app.controller.export_region(self.export_snapshot.mesh)

    def render(self, state):
        """Enable selection actions according to available inputs and execution state."""
        busy = state.run_id is not None
        self.widget.enabled = not busy
        self.apply.enabled = (
            not busy
            and self.model.currentNode() is not None
            and self.curve.currentNode() is not None
            and self.curve.currentNode().GetNumberOfControlPoints() >= 3
        )
        self.clear.enabled = self.export.enabled = (
            not busy and state.selection is not None
        )
        self.info.text = (
            "No region selected"
            if state.selection is None
            else f"Selected: {len(state.selection.vertices)} vertices, {len(state.selection.faces)} faces"
        )

    def close(self):
        """Restore display state and remove markup observers and keyboard shortcuts."""
        self.restore_appearance()
        for node, tag in self._curve_observers:
            node.RemoveObserver(tag)
        self._curve_observers = []
        self.shortcut.disconnect("activated()", self.stop_placing)
        self.shortcut.setParent(None)
