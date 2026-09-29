"""Dependency UI that remains usable without importing InterDeCA's core."""

import importlib
import importlib.metadata
import importlib.util
import os
import tempfile
from codecs import getincrementaldecoder
from functools import partial
from pathlib import Path
from typing import ClassVar, NamedTuple

import ctk
import qt
from packaging.requirements import Requirement

import slicer


class InstallationStep(NamedTuple):
    """Resolve optional dependencies separately so they cannot block core installation."""

    label: str
    packages: tuple[str, ...]
    optional: bool = False


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
        # Keep the bootstrap form compact when the rest of the module cannot load yet.
        self.widget.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Maximum)
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
        self._pending: tuple[InstallationStep, ...] = ()
        self._step: InstallationStep | None = None
        self._closed = False
        # Process output can split a multibyte character across separate Qt notifications.
        self._decoder = getincrementaldecoder("utf-8")(errors="replace")
        self.refresh()
        if import_error:
            self.log.appendPlainText(import_error)
            self.log.appendPlainText(
                "Install missing required packages below, then restart Slicer. "
                "If interdeca itself is missing, follow the editable-install instructions in docs/slicer.md."
            )

    def refresh(self):
        """Check distribution versions without loading compiled numerical modules."""
        missing = []
        required_details = []
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
                if name != "umap":
                    required_details.append(
                        f"{requirement.name} ({version or 'missing'})"
                    )
        required_status = (
            "Required packages missing or incompatible: " + ", ".join(required_details)
            if required_details
            else "Required analysis packages are available."
        )
        optional_status = (
            "Optional UMAP is unavailable. ATLAS, color processing, PCA, and ICA do not require it."
            if "umap" in missing
            else "Optional UMAP is installed."
        )
        self.status.text = required_status + "\n" + optional_status
        return missing

    def install(self, all_packages):
        """Install core dependencies first, then optionally attempt UMAP with host pins."""
        if self._closed or self.process is not None:
            return
        missing = self.refresh()
        required = tuple(
            name
            for name in self.requirements
            if name != "umap" and (all_packages or name in missing)
        )
        steps = []
        if required:
            steps.append(InstallationStep("Required analysis packages", required))
        if all_packages:
            steps.append(InstallationStep("Optional UMAP", ("umap",), optional=True))
        if not steps:
            self.log.appendPlainText(
                "Required packages are ready. Restart Slicer to load them."
            )
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
        self._pending = tuple(steps)
        self.check.enabled = self.all.enabled = False
        self._start_next()

    def _start_next(self):
        # A separate pip transaction lets a platform-specific UMAP failure leave core usable.
        self._step, self._pending = self._pending[0], self._pending[1:]
        self._decoder.reset()
        executable = (
            Path(slicer.app.slicerHome)
            / "bin"
            / ("PythonSlicer.exe" if os.name == "nt" else "PythonSlicer")
        )
        process = qt.QProcess(self.widget)
        self.process = process
        process.setProcessChannelMode(qt.QProcess.MergedChannels)
        # Bind callbacks to their own process so delayed signals cannot finish a later step.
        process.connect("readyReadStandardOutput()", partial(self._read, process))
        process.connect(
            "finished(int,QProcess::ExitStatus)", partial(self._finished, process)
        )
        process.connect(
            "errorOccurred(QProcess::ProcessError)", partial(self._error, process)
        )
        self.log.appendPlainText(
            f"Installing {self._step.label.lower()} with the existing host libraries pinned…"
        )
        process.start(
            str(executable),
            [
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--constraint",
                str(self._constraints),
            ]
            + [self.requirements[name] for name in self._step.packages],
        )

    def _read(self, process, final=False):
        if self._closed or process is not self.process:
            return
        # PythonQt exposes QByteArray bytes through data(); bytes(QByteArray) is unsupported.
        output = self._decoder.decode(
            process.readAllStandardOutput().data(), final=final
        )
        if output:
            self.log.moveCursor(qt.QTextCursor.End)
            self.log.insertPlainText(output)

    def _finished(self, process, code, status):
        if process is not self.process:
            return
        succeeded = code == 0 and status == qt.QProcess.NormalExit
        try:
            if not self._closed:
                self._read(process, final=True)
                self.log.appendPlainText(
                    f"{self._step.label} installed. Restart Slicer to load them."
                    if succeeded
                    else "Optional UMAP could not be installed for this Slicer environment. "
                    "Required packages remain available; see pip diagnostics above."
                    if self._step.optional
                    else "Required package installation failed; see pip diagnostics above."
                )
        finally:
            # Output handling must never leave the buttons disabled or retain a finished process.
            self.process = None
            process.deleteLater()
            if self._closed or not succeeded:
                self._pending = ()
            if self._pending:
                self._start_next()
            else:
                self._end_installation()

    def _end_installation(self):
        self._step = None
        if self._constraints is not None:
            self._constraints.unlink(missing_ok=True)
            self._constraints = None
        if not self._closed:
            self.check.enabled = self.all.enabled = True
            importlib.invalidate_caches()
            self.refresh()

    def _error(self, process, error):
        if process is self.process and not self._closed:
            self.log.appendPlainText(process.errorString())
            if process.state() == qt.QProcess.NotRunning:
                self._finished(process, 1, None)

    def close(self):
        """Release the installation process and its temporary constraint file."""
        # A package installation owns this process; clean it up before its widgets vanish.
        self._closed = True
        self._pending = ()
        if self.process is not None:
            process = self.process
            process.kill()
            process.waitForFinished(1000)
        self._end_installation()
