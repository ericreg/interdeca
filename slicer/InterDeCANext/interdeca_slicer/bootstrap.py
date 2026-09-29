"""Dependency UI that remains usable without importing InterDeCA's core."""

import importlib.metadata
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import ClassVar

import ctk
import qt
from packaging.requirements import Requirement

import slicer


class PackagePanel:
    """Install missing analysis packages using Slicer's Python, preserving host pins."""

    requirements: ClassVar[dict[str, str]] = {
        "pydantic": "pydantic>=2.11,<3",
        "sklearn": "scikit-learn>=1.6,<2",
        "skimage": "scikit-image>=0.25,<1",
        "umap": "umap-learn>=0.5.7,<0.6",
    }

    def __init__(self, import_error=""):
        self.widget = ctk.ctkCollapsibleButton()
        self.widget.text = "Package Management"
        self.widget.collapsed = not bool(import_error)
        layout = qt.QFormLayout(self.widget)
        self.status = qt.QLabel()
        self.status.wordWrap = True
        layout.addRow(self.status)
        self.check = qt.QPushButton("Check & Install Missing Packages")
        self.all = qt.QPushButton("Install All Recommended Packages")
        self.log = qt.QPlainTextEdit()
        self.log.readOnly = True
        self.log.maximumHeight = 120
        layout.addRow(self.check)
        layout.addRow(self.all)
        layout.addRow(self.log)
        self.check.connect("clicked()", lambda: self.install(False))
        self.all.connect("clicked()", lambda: self.install(True))
        self.process = None
        self._constraints = None
        self.refresh()
        if import_error:
            self.log.appendPlainText(import_error)
            self.log.appendPlainText(
                "Install the interdeca package with Slicer's Python, then restart Slicer. See README.md."
            )

    def refresh(self):
        """Check distribution versions without loading compiled numerical modules."""
        missing = []
        details = []
        # An importable but incompatible release also needs installation or an upgrade.
        for name, specification in self.requirements.items():
            requirement = Requirement(specification)
            try:
                version = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                version = None
            if (
                importlib.util.find_spec(name) is None
                or version is None
                or version not in requirement.specifier
            ):
                missing.append(name)
                details.append(f"{requirement.name} ({version or 'missing'})")
        self.status.text = (
            "Missing or incompatible: " + ", ".join(details)
            if missing
            else "All analysis packages are available."
        )
        return missing

    def install(self, all_packages):
        """Install requested dependencies in the host environment with binary versions pinned."""
        if self.process is not None:
            return
        packages = list(self.requirements) if all_packages else self.refresh()
        if not packages:
            return
        # Pin the host's binary stack so pip cannot replace Slicer's VTK or NumPy ABI.
        descriptor, filename = tempfile.mkstemp(
            prefix="interdeca-constraints-", suffix=".txt"
        )
        with os.fdopen(descriptor, "w") as stream:
            for package in ("numpy", "scipy", "vtk", "pillow"):
                try:
                    stream.write(f"{package}=={importlib.metadata.version(package)}\n")
                except importlib.metadata.PackageNotFoundError:
                    pass
        self._constraints = Path(filename)
        executable = (
            Path(slicer.app.slicerHome)
            / "bin"
            / ("PythonSlicer.exe" if os.name == "nt" else "PythonSlicer")
        )
        self.process = qt.QProcess(self.widget)
        self.process.setProcessChannelMode(qt.QProcess.MergedChannels)
        self.process.connect("readyReadStandardOutput()", self._read)
        self.process.connect("finished(int,QProcess::ExitStatus)", self._finished)
        self.process.connect("errorOccurred(QProcess::ProcessError)", self._error)
        self.check.enabled = self.all.enabled = False
        self.log.appendPlainText(
            "Installing analysis dependencies with the existing host libraries pinned…"
        )
        self.process.start(
            str(executable),
            ["-m", "pip", "install", "--constraint", filename]
            + [self.requirements[name] for name in packages],
        )

    def _read(self):
        if self.process is not None:
            self.log.appendPlainText(
                bytes(self.process.readAllStandardOutput()).decode(
                    "utf-8", errors="replace"
                )
            )

    def _finished(self, code, status):
        self._read()
        self.log.appendPlainText(
            "Restart Slicer to load the installed packages."
            if code == 0
            else "Installation failed; see diagnostics above."
        )
        self.process = None
        self.check.enabled = self.all.enabled = True
        if self._constraints is not None:
            self._constraints.unlink(missing_ok=True)
            self._constraints = None

    def _error(self, error):
        if self.process is not None:
            self.log.appendPlainText(self.process.errorString())
            if self.process.state() == qt.QProcess.NotRunning:
                self._finished(1, None)

    def close(self):
        """Release the installation process and its temporary constraint file."""
        # A package installation owns this process; clean it up before its widgets vanish.
        if self.process is not None:
            self.process.kill()
            self.process.waitForFinished(1000)
        if self._constraints is not None:
            self._constraints.unlink(missing_ok=True)
