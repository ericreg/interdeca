"""Small widget constructors keep workflow panels readable without hiding behavior."""

import ctk
import qt

import slicer


class Widgets:
    """Presentation-only factories; none invoke application operations."""

    @staticmethod
    def section(layout, title, collapsed=False):
        """Group related controls without coupling them to a workflow controller."""
        group = ctk.ctkCollapsibleButton()
        group.text = title
        group.collapsed = collapsed
        layout.addRow(group)
        return qt.QFormLayout(group)

    @staticmethod
    def path(layout, label, files=False):
        """Create a path selector restricted to files or directories."""
        widget = ctk.ctkPathLineEdit()
        widget.filters = (
            ctk.ctkPathLineEdit.Files if files else ctk.ctkPathLineEdit.Dirs
        )
        layout.addRow(label, widget)
        return widget

    @staticmethod
    def spin(layout, label, minimum, maximum, value, decimals=None):
        """Constrain a numerical setting to the range supported by its operation."""
        widget = qt.QSpinBox() if decimals is None else qt.QDoubleSpinBox()
        if decimals is not None:
            widget.setDecimals(decimals)
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        layout.addRow(label, widget)
        return widget

    @staticmethod
    def check(layout, label, checked=False):
        """Create a labeled Boolean workflow choice."""
        widget = qt.QCheckBox()
        widget.checked = checked
        layout.addRow(label, widget)
        return widget

    @staticmethod
    def node(layout, label, kind="vtkMRMLModelNode"):
        """Select existing scene objects without allowing accidental node creation."""
        widget = slicer.qMRMLNodeComboBox()
        widget.nodeTypes = [kind]
        widget.noneEnabled = True
        widget.addEnabled = False
        widget.removeEnabled = False
        widget.setMRMLScene(slicer.mrmlScene)
        layout.addRow(label, widget)
        return widget

    @staticmethod
    def button(layout, text, callback):
        """Connect a user action to its host callback."""
        widget = qt.QPushButton(text)
        widget.connect("clicked()", callback)
        layout.addRow(widget)
        return widget

    @staticmethod
    def combo(layout, label):
        """Create a labeled selector whose choices are supplied by session results."""
        widget = qt.QComboBox()
        layout.addRow(label, widget)
        return widget
