"""Small Slicer entry point; application logic and rendering live in packages."""

from interdeca_slicer.bootstrap import PackagePanel
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
)

# Missing computational dependencies must not hide the installation controls.
try:
    from interdeca_slicer.application import SlicerApplication

    IMPORT_ERROR = ""
except ImportError as error:
    SlicerApplication = None
    IMPORT_ERROR = str(error)


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
        """Create the application when available and always expose dependency management."""
        super().setup()
        self.application = None
        if SlicerApplication is not None:
            self.application = SlicerApplication()
            self.layout.addWidget(self.application.widget)
        self.packages = PackagePanel(IMPORT_ERROR)
        self.layout.addWidget(self.packages.widget)

    def cleanup(self):
        """Detach application callbacks before Slicer destroys the module widgets."""
        # Detach scene observers and queued callbacks before Slicer destroys widgets.
        if self.application is not None:
            self.application.close()
        self.packages.close()
