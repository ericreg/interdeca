"""Slicer client entry point; the external worker owns application logic."""

from interdeca_slicer.application import SlicerApplication
from interdeca_slicer.bootstrap import WorkerPanel
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
)


class InterDeCANext(ScriptedLoadableModule):
    """Register a distinct name so the reference module can remain installed."""

    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "InterDeCA Next"
        parent.categories = ["Shape Analysis"]
        parent.contributors = ["InterDeCA"]
        parent.helpText = "Atlas correspondence, texture transfer, region selection, and population color analysis."
        parent.acknowledgementText = "InterDeCA"


class InterDeCANextWidget(ScriptedLoadableModuleWidget):
    """Own the host lifetime; no numerical work is performed by this entry point."""

    def setup(self):
        """Create passive panels and the independent worker launcher."""
        super().setup()
        self.application = SlicerApplication()
        self.worker = WorkerPanel(self.application)
        self.layout.addWidget(self.worker.widget)
        self.layout.addWidget(self.application.widget)

    def cleanup(self):
        """Detach application callbacks before Slicer destroys the module widgets."""
        # Detach scene observers and queued callbacks before Slicer destroys widgets.
        self.application.close()
