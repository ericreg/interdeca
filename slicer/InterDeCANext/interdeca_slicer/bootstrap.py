"""Configure a separate analysis environment without installing core packages in Slicer."""

import os
from pathlib import Path

import ctk
import qt

import slicer


class WorkerPanel:
    """The host starts/stops the worker and exposes its independent Python path."""

    def __init__(self, application):
        self.application = application
        self.widget = ctk.ctkCollapsibleButton()
        self.widget.text = "Application Worker"
        layout = qt.QFormLayout(self.widget)
        self.python = ctk.ctkPathLineEdit()
        self.python.filters = ctk.ctkPathLineEdit.Files
        # A repository checkout has a useful default; installed extensions require a path.
        repository = Path(__file__).resolve().parents[3]
        default = os.environ.get("INTERDECA_WORKER_PYTHON", "")
        candidate = (
            repository
            / ".venv"
            / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        if not default and candidate.is_file():
            default = str(candidate)
        saved = application.settings.value("worker/python", default)
        self.python.setCurrentPath(str(saved))
        layout.addRow("Worker Python:", self.python)
        # Each task may hold a dense specimen and use Rust's internal threads;
        # one is the conservative default, with an explicit CPU-bounded choice.
        available = os.cpu_count() or 1
        self.jobs = qt.QSpinBox()
        self.jobs.setRange(1, available)
        self.jobs.setValue(int(application.settings.value("worker/jobs", 1)))
        self.jobs.toolTip = "1 executes tasks serially; higher counts execute independent tasks in parallel. More workers use more memory. Stop and Start Worker to apply a changed count."
        layout.addRow(f"Task workers ({available} logical CPUs):", self.jobs)
        restart_hint = qt.QLabel("Worker count takes effect on Start Worker; stop the current worker first to change it.")
        restart_hint.wordWrap = True
        layout.addRow(restart_hint)
        help_text = qt.QLabel(
            "Install InterDeCA in a separate Python 3.12+ environment. "
            "Select that environment's Python executable here. "
            "Slicer needs only its bundled NumPy and the websockets transport package."
        )
        help_text.wordWrap = True
        layout.addRow(help_text)
        self.start = qt.QPushButton("Start Worker")
        self.stop = qt.QPushButton("Stop Worker")
        self.install = qt.QPushButton("Install Client Transport")
        layout.addRow(self.start)
        layout.addRow(self.stop)
        layout.addRow(self.install)
        self.start.connect("clicked()", lambda: application.guard(self.launch))
        self.stop.connect("clicked()", self.shutdown)
        self.install.connect(
            "clicked()", lambda: application.guard(self.install_transport)
        )
        if saved:
            # Defer until all module widgets and observers exist.
            qt.QTimer.singleShot(
                0,
                lambda: (
                    application.guard(self.launch) if not application._closed else None
                ),
            )

    def launch(self):
        """Persist the executable and concurrency choices before starting a session."""
        path = self.python.currentPath
        self.application.settings.setValue("worker/python", path)
        jobs = int(self.jobs.value)
        self.application.settings.setValue("worker/jobs", jobs)
        self.application.client.start(path, max_workers=jobs)

    def shutdown(self):
        """Detach old state before stopping a worker that may still be computing."""
        self.application.client.close()
        self.application.plot.clear()
        self.application.selection.restore_appearance()
        self.application.render(self.application.client.state)

    def install_transport(self):
        """Install only the small pure-Python socket dependency into the host."""
        slicer.util.pip_install("websockets>=15,<16")
        slicer.util.infoDisplay(
            "Client transport installed. Restart Slicer to load it."
        )
