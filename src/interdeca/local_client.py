"""In-process application client for command-line hosts and small integration checks.

The same ApplicationSession handles Slicer's socket commands and these direct
calls. No socket or worker subprocess is created. Numerical presentation files
are retained so a CLI run can be inspected after its live estimators are gone.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from typing import NamedTuple
from uuid import uuid4

from interdeca_client.artifacts import ArtifactStore

from interdeca.application import ApplicationSession
from interdeca.models import Mesh
from interdeca.serde import (
    ApplicationCommand,
    BoundaryData,
    CapabilitiesEvent,
    ControlStateData,
    LogEntryData,
    LogEvent,
    ResultEvent,
    StateEvent,
    TerminalEvent,
)


class PendingCommand(NamedTuple):
    """Keep the input references until the application acknowledges their consumption."""

    action: str
    payload: dict


class ApplicationError(RuntimeError):
    """An application failure with its complete diagnostic traceback."""

    def __init__(self, event: TerminalEvent):
        super().__init__(event.message)
        self.details = event.details
        self.request_id = event.request_id


class LocalApplicationClient:
    """Own one application session; poll delivers events on the calling host thread.

    The supplied directory must be new. Outputs are persistent presentation data,
    not serialized estimators or a restorable application session. close cancels
    outstanding work and waits for active serial operations before releasing inputs.
    """

    def __init__(self, root: Path, receive: Callable[[BoundaryData], None]):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.artifacts = ArtifactStore(self.root)
        self.receive = receive
        self.events = Queue()
        self.pending: dict[str, PendingCommand] = {}
        self.results: dict[str, ResultEvent] = {}
        self.failures: list[TerminalEvent] = []
        self.state: ControlStateData | None = None
        self.capabilities: CapabilitiesEvent | None = None
        self.revision = 0
        self.cancelled: set[str] = set()
        self.latest_reconstruction: str | None = None
        self.closed = False
        # File logging remains independent of progress, just as in the socket host.
        self.log_handler = logging.FileHandler(
            self.root / "worker.log", encoding="utf-8"
        )
        self.log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s request=%(request_id)s run=%(run_id)s %(name)s: %(message)s",
                defaults={"request_id": "-", "run_id": "-"},
            )
        )
        logger = logging.getLogger("interdeca")
        self.previous_log_level = logger.level
        logger.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.log_handler)
        self.application = ApplicationSession(self.root, self.events)

    def send(self, action: str, **payload) -> str:
        """Queue one user action directly, preserving the common application contract."""
        if self.closed:
            raise RuntimeError("The application session is closed.")
        # Apply the host's invalidation/cancellation decision before delivering
        # results that may already be queued, matching the Slicer client boundary.
        if action in ("invalidate", "clear_selection"):
            self.revision += 1
        if action == "cancel":
            self.cancelled.update(self.pending)
        self.poll()
        request_id = uuid4().hex
        command = ApplicationCommand(
            protocol_version=1,
            type="command",
            request_id=request_id,
            input_revision=self.revision,
            action=action,
            payload=payload,
        ).model_copy(deep=True)
        # Frozen Pydantic models do not freeze nested dictionaries; take ownership
        # before another thread validates the payload or a caller edits its inputs.
        self.pending[request_id] = PendingCommand(action, command.payload)
        if action == "reconstruct":
            self.latest_reconstruction = request_id
        # Only the application thread touches its controller; no host reaches into state.
        self.application.submit(command)
        return request_id

    def mesh(self, mesh: Mesh) -> dict:
        """Publish a canonical host snapshot using the same numerical boundary as Slicer."""
        return {
            "vertices": self.artifacts.write(mesh.vertices, "inputs"),
            "faces": self.artifacts.write(mesh.faces, "inputs"),
            "uv": None if mesh.uv is None else self.artifacts.write(mesh.uv, "inputs"),
        }

    def poll(self, timeout: float = 0) -> None:
        """Deliver queued events in order; reject obsolete results before updating the view."""
        while True:
            try:
                event = self.events.get(timeout=timeout)
            except Empty:
                return
            timeout = 0
            if isinstance(event, LogEntryData):
                # The socket host batches these records; local output uses the same
                # public log envelope so every JSON line has a message type.
                event = LogEvent(entries=(event,))
            if isinstance(event, CapabilitiesEvent):
                self.capabilities = event
            elif isinstance(event, StateEvent):
                if event.payload.revision >= self.revision:
                    self.state = event.payload
            elif isinstance(event, ResultEvent):
                obsolete = (
                    event.input_revision != self.revision
                    or event.request_id in self.cancelled
                    or (
                        event.action in ("morphospace", "morph_controls")
                        and event.request_id != self.latest_reconstruction
                    )
                )
                if not obsolete:
                    self.results[event.action] = event
            elif isinstance(event, TerminalEvent):
                pending = self.pending.pop(event.request_id, None)
                if pending is not None:
                    self.artifacts.discard(pending.payload, "inputs")
                if event.type == "failed":
                    self.failures.append(event)
            # Retain the complete event stream, including superseded results, for diagnosis.
            self.receive(event)

    def wait_ready(self) -> None:
        """Wait for capability discovery before a host offers estimator choices."""
        while self.capabilities is None:
            self.poll(0.1)

    def wait(self) -> None:
        """Pump notifications until all accepted commands have terminal acknowledgements."""
        while self.pending:
            self.poll(0.1)
        self.poll()
        if self.failures:
            failure = self.failures[0]
            self.failures.clear()
            raise ApplicationError(failure)

    def close(self) -> None:
        """Cancel, detach, and drain operations before deleting their input snapshots."""
        if self.closed:
            return
        self.closed = True
        self.application.close()
        self.application.join()
        for pending in self.pending.values():
            self.artifacts.discard(pending.payload, "inputs")
        self.pending.clear()
        # A host-side loading error can leave a snapshot that was never submitted.
        # This namespace belongs entirely to this new session, not to source files.
        for path in (self.root / "inputs").glob("*.npy"):
            path.unlink(missing_ok=True)
        logger = logging.getLogger("interdeca")
        logging.getLogger().removeHandler(self.log_handler)
        logger.setLevel(self.previous_log_level)
        self.log_handler.close()
