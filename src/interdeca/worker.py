"""Local application server: controllers, state, pipelines, and execution live here.

The WebSocket reader and writer never run numerical operations. A session thread
owns the controller, and the serial executor runs operations on its own thread.
This leaves cancellation and socket heartbeats responsive during native calls.
"""

import argparse
import importlib
import json
import logging
import os
import secrets
from collections import deque
from functools import partial
from http import HTTPStatus
from pathlib import Path
from queue import Full, Queue
from threading import Condition, Event, Thread, current_thread
from traceback import format_exc
from typing import NamedTuple
from uuid import uuid4

from interdeca_client.artifacts import ArtifactStore
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from interdeca.controller import ApplicationController
from interdeca.serde import (
    AcceptedEvent,
    ApplicationCommand,
    ArrayArtifact,
    CapabilitiesEvent,
    ColorsPresentation,
    CommandIdentityData,
    ControlStateData,
    EmptyPayload,
    ExportPresentation,
    FaceColorsPresentation,
    LogEntryData,
    LogEvent,
    MeshArtifacts,
    MeshData,
    MeshPresentation,
    MorphControlsData,
    PlotPresentation,
    PopulationPresentationData,
    ProgressEvent,
    ResultEvent,
    SelectionPresentation,
    StateEvent,
    TerminalEvent,
)
from interdeca.serial_execution import SerialExecutionManager
from interdeca.workflows import Operations

logger = logging.getLogger(__name__)


class RequestContext(NamedTuple):
    """Identity of one user action, retained through asynchronous task completion."""

    request_id: str | None = None
    run_id: str | None = None
    input_revision: int = 0

    def fields(self):
        """A boundary keyword mapping, not mutable application state."""
        return self._asdict()


class PendingRequest(NamedTuple):
    command: ApplicationCommand
    payload: object
    context: RequestContext


class EventBuffer:
    """Coalesce progress and state at 20 Hz; keep results and terminal events ordered."""

    def __init__(self):
        self.condition = Condition()
        self.reliable = deque()
        self.latest = {}
        self.logs = deque(maxlen=200)
        self.dropped = 0
        self.closed = False

    def put(self, event):
        with self.condition:
            if self.closed:
                return
            if isinstance(event, LogEntryData):
                self.dropped += int(len(self.logs) == self.logs.maxlen)
                self.logs.append(event)
            elif isinstance(event, (ProgressEvent, StateEvent)):
                self.latest[(event.type, event.run_id)] = event
            else:
                # Flush snapshots before results/terminal events so final controls cannot lag.
                self.reliable.extend(self.latest.values())
                self.latest.clear()
                self.reliable.append(event)
            self.condition.notify()

    def send(self, socket):
        try:
            while not self.closed:
                with self.condition:
                    self.condition.wait(timeout=0.05)
                    batch = list(self.reliable) + list(self.latest.values())
                    self.reliable.clear()
                    self.latest.clear()
                    if self.logs:
                        batch.append(
                            LogEvent(entries=tuple(self.logs), dropped=self.dropped)
                        )
                        self.logs.clear()
                        self.dropped = 0
                for event in batch:
                    socket.send(event.model_dump_json())
                # Frequent producers cannot turn coalescing into one write per log line.
                Event().wait(0.05)
        except ConnectionClosed:
            pass
        except Exception:
            logger.exception("Application event stream failed")
            socket.close(1011, "Event stream failed")
        finally:
            self.closed = True


class SessionLogHandler(logging.Handler):
    """Save full diagnostics to disk separately; only bounded batches reach the UI."""

    def __init__(self, session):
        super().__init__()
        self.session = session

    def emit(self, record):
        session = self.session
        context = (
            session.active_context
            if current_thread().name == "interdeca-serial"
            else session.context
        )
        # Correlation is also written to the complete on-disk diagnostic log.
        record.request_id = context.request_id or "-"
        record.run_id = context.run_id or "-"
        session.events.put(
            LogEntryData(
                **context.fields(),
                severity=record.levelname,
                message=self.format(record)[:3000],
            )
        )


class SessionExecution:
    """Attach request identity to an otherwise host-independent execution manager."""

    def __init__(self, session):
        self.session = session
        self.manager = SerialExecutionManager(Operations.registry(), session)

    def submit(self, pipeline, observer, *, run_id):
        context = self.session.context
        self.session.active_context = context
        return self.manager.submit(
            pipeline,
            partial(self.deliver, observer, context),
            run_id=run_id,
        )

    def deliver(self, observer, context, event):
        session = self.session
        session.context = context
        if event.run_id != session.controller.state.run_id:
            return
        if event.kind == "failed":
            logger.error(
                "%s: %s\n%s",
                event.failure.operation,
                event.failure.message,
                event.failure.traceback,
            )
        elif event.kind == "task_started":
            logger.info("Starting %s", event.label)
        # Controller callbacks and user actions execute on the same session thread.
        try:
            observer(event)
        except Exception as error:
            # Presentation I/O can fail after computation succeeds. Release busy controls
            # and report that publication failure without stranding the accepted request.
            logger.exception("Could not publish execution result")
            self.manager.cancel(event.run_id)
            session.controller.state = session.controller.state._replace(
                run_id=None,
                status="failed",
                message=str(error),
            )
            session.render(session.controller.state)
            session.events.put(
                TerminalEvent(
                    **context.fields(),
                    type="failed",
                    message=str(error),
                    details=format_exc(),
                )
            )
            session.cancel_pending()
            return
        if event.kind in ("completed", "failed", "cancelled"):
            session.events.put(
                TerminalEvent(
                    **context.fields(),
                    type="cancelled"
                    if session.controller.state.status == "cancelled"
                    else event.kind,
                    message=event.failure.message if event.failure else event.label,
                    details=event.failure.traceback if event.failure else "",
                )
            )
            session.active_context = RequestContext()
            if session.pending is not None:
                pending, session.pending = session.pending, None
                session.execute(pending)
        else:
            session.events.put(
                ProgressEvent(
                    **context.fields(),
                    operation=event.label,
                    completed=event.completed,
                    total=event.total,
                    stage=event.stage,
                    stages=event.stages,
                )
            )

    def cancel(self, run_id):
        self.manager.cancel(run_id)

    def close(self):
        self.manager.close()


class ApplicationSession:
    """One authenticated connection owns one nonpersistent application session."""

    run_actions = frozenset(
        {
            "run_atlas",
            "process_colors",
            "fit_population",
            "show_specimen",
            "select_region",
            "export_region",
            "reconstruct",
        }
    )

    def __init__(self, root, events):
        self.root, self.events = root, events
        self.artifacts = ArtifactStore(root)
        self.queue = Queue(maxsize=256)
        self.context = self.active_context = RequestContext()
        self.pending = None
        self.dataset_id = self.model_id = None
        self.seen = set()
        self.closed = False
        self.execution = SessionExecution(self)
        self.controller = ApplicationController(self.execution, self)
        self.log_handler = SessionLogHandler(self)
        logging.getLogger("interdeca").addHandler(self.log_handler)
        self.thread = Thread(
            target=self.work, name="interdeca-application", daemon=True
        )
        self.thread.start()

    def post(self, callback):
        """Execution notifications enter the state owner's queue asynchronously."""
        if not self.closed:
            self.queue.put(callback)

    def work(self):
        try:
            # Actually importing optional UMAP reports broken binary stacks as unavailable.
            methods, umap_error = ("pca", "ica"), ""
            try:
                importlib.import_module("umap")
                methods += ("umap",)
            except Exception as error:
                umap_error = str(error)
                logger.info("Optional UMAP unavailable: %s", error, exc_info=True)
            self.methods = methods
            self.events.put(
                CapabilitiesEvent(
                    methods=methods,
                    umap_error=umap_error,
                    log_path=str(self.root / "worker.log"),
                )
            )
            self.render(self.controller.state)
            while not self.closed:
                item = self.queue.get()
                if item is None:
                    break
                try:
                    item() if callable(item) else self.receive(item)
                except Exception:
                    logger.exception("Application notification failed")
                    self.events.put(
                        TerminalEvent(
                            **self.context.fields(),
                            type="failed",
                            message="Application notification failed",
                            details=format_exc(),
                        )
                    )
        finally:
            self.controller.close()
            logging.getLogger("interdeca").removeHandler(self.log_handler)

    def receive(self, raw):
        context = RequestContext()
        try:
            identity = CommandIdentityData.model_validate_json(raw)
            context = RequestContext(identity.request_id, None, identity.input_revision)
            self.context = context
            command = ApplicationCommand.model_validate_json(raw)
            if command.request_id in self.seen:
                raise ValueError("Duplicate request_id; commands must not be replayed.")
            self.seen.add(command.request_id)
            if command.input_revision < self.controller.state.revision:
                raise ValueError("The command refers to an obsolete input revision.")
            if (
                command.action not in ("invalidate", "clear_selection")
                and command.input_revision != self.controller.state.revision
            ):
                raise ValueError("Invalidate inputs before using a new revision.")
            payload = command.validated_payload()
            context = context._replace(
                run_id=uuid4().hex if command.action in self.run_actions else None,
            )
            self.context = context
            self.events.put(AcceptedEvent(**context.fields(), action=command.action))
            pending = PendingRequest(command, payload, context)
            if command.action == "reconstruct" and self.controller.state.run_id:
                if self.controller.state.action != "morphospace":
                    raise ValueError("Wait for the current analysis to finish.")
                self.cancel_pending()
                self.pending = pending
                self.controller.cancel()
                return
            if command.action in ("cancel", "invalidate", "clear_selection"):
                self.cancel_pending()
            self.execute(pending)
        except Exception as error:
            logger.exception("Request rejected")
            self.events.put(
                TerminalEvent(
                    **context.fields(),
                    type="failed",
                    message=str(error),
                    details=format_exc(),
                )
            )

    def cancel_pending(self):
        if self.pending is not None:
            self.events.put(
                TerminalEvent(
                    **self.pending.context.fields(),
                    type="cancelled",
                    message="Superseded before reconstruction started.",
                )
            )
            self.pending = None

    def load(self, reference):
        return self.artifacts.read(reference.model_dump(), "inputs")

    def mesh(self, value):
        # Core validation rejects bad geometry after the file metadata has been checked.
        return MeshData(
            vertices=self.load(value.vertices),
            faces=self.load(value.faces),
            uv=None if value.uv is None else self.load(value.uv),
        ).to_internal()

    def execute(self, pending):
        command, payload, self.context = pending
        controller = self.controller
        try:
            action = command.action
            if hasattr(payload, "dataset_id") and payload.dataset_id != self.dataset_id:
                raise ValueError(
                    "Dataset is no longer available; process colors again."
                )
            if hasattr(payload, "model_id") and payload.model_id != self.model_id:
                raise ValueError(
                    "Model is no longer available; fit the population again."
                )
            if action == "run_atlas":
                controller.run_atlas(payload, run_id=self.context.run_id)
            elif action == "process_colors":
                controller.process_colors(
                    self.mesh(payload.mesh), payload.options, run_id=self.context.run_id
                )
            elif action == "fit_population":
                if payload.options.method not in self.methods:
                    raise ValueError(
                        "The requested estimator is unavailable in this worker."
                    )
                controller.fit_population(payload.options, run_id=self.context.run_id)
            elif action == "show_specimen":
                controller.show_specimen(
                    payload.specimen_id, run_id=self.context.run_id
                )
            elif action == "select_region":
                controller.select_region(
                    self.mesh(payload.mesh),
                    self.load(payload.points),
                    payload.same_side_only,
                    None
                    if payload.vertex_colors is None
                    else self.load(payload.vertex_colors),
                    run_id=self.context.run_id,
                )
            elif action == "export_region":
                controller.export_region(
                    self.mesh(payload.mesh), run_id=self.context.run_id
                )
            elif action in ("invalidate", "clear_selection"):
                if command.input_revision <= controller.state.revision:
                    raise ValueError("Input changes require a strictly newer revision.")
                # The host allocates revisions so it can reject stale updates immediately.
                controller.state = controller.state._replace(
                    revision=command.input_revision - 1
                )
                controller.clear_selection() if action == "clear_selection" else controller.invalidate(
                    payload.scope
                )
            elif action == "cancel":
                controller.cancel()
            elif action == "set_axes":
                controller.set_axes(payload.x_axis, payload.y_axis)
            elif action == "reconstruct":
                controller.reconstruct(
                    payload.specimen_id, payload.positions, run_id=self.context.run_id
                )
            if action not in self.run_actions:
                self.events.put(
                    TerminalEvent(**self.context.fields(), type="completed")
                )
        except Exception as error:
            logger.exception("Application action failed")
            self.events.put(
                TerminalEvent(
                    **self.context.fields(),
                    type="failed",
                    message=str(error),
                    details=format_exc(),
                )
            )

    def render(self, state):
        if state.colors is None:
            self.dataset_id = None
        if state.population is None:
            self.model_id = None
        # Control snapshots carry counts and identifiers, never retained numerical state.
        self.events.put(
            StateEvent(
                **self.context.fields(),
                payload=ControlStateData(
                    revision=state.revision,
                    run_id=state.run_id,
                    action=state.action,
                    status=state.status,
                    completed=state.completed,
                    total=state.total,
                    stage=state.stage,
                    stages=state.stages,
                    message=state.message,
                    dataset_id=self.dataset_id,
                    model_id=self.model_id,
                    morph_active=state.morph is not None,
                    selection_vertices=0
                    if state.selection is None
                    else len(state.selection.vertices),
                    selection_faces=0
                    if state.selection is None
                    else len(state.selection.faces),
                ),
            )
        )

    def array(self, value):
        reference = self.artifacts.write(value, "outputs")
        return ArrayArtifact.model_validate_json(json.dumps(reference))

    def mesh_result(self, mesh):
        return MeshArtifacts(
            vertices=self.array(mesh.vertices),
            faces=self.array(mesh.faces),
            uv=None if mesh.uv is None else self.array(mesh.uv),
        )

    def present(self, action, result):
        if action == "atlas":
            payload = MeshPresentation(
                mesh=self.mesh_result(result.atlas.mesh),
                directory=str(result.directory),
                has_textures=bool(result.textures),
                average=None if result.average is None else str(result.average),
            )
        elif action == "colors":
            self.dataset_id = uuid4().hex
            payload = ColorsPresentation(dataset_id=self.dataset_id)
        elif action == "population":
            self.model_id = uuid4().hex
            payload = PopulationPresentationData(
                model_id=self.model_id,
                method=result.model.method,
                specimen_ids=result.model.specimen_ids,
                components=result.model.scores.shape[1],
                representation=result.features.representation,
                can_reconstruct=result.features.representation == "colors"
                and callable(
                    getattr(result.model.estimator, "inverse_transform", None)
                ),
            )
        elif action == "plot":
            payload = PlotPresentation(
                specimen_ids=result.specimen_ids,
                scores=self.array(result.scores),
                method=result.method,
                x_axis=result.x_axis,
                y_axis=result.y_axis,
            )
        elif action == "morph_controls":
            payload = MorphControlsData(
                positions=result.positions, coordinates=result.coordinates
            )
        elif action in ("display", "morphospace"):
            payload = FaceColorsPresentation(
                rgb=self.array(result.rgb), valid=self.array(result.valid)
            )
        elif action == "selection":
            payload = SelectionPresentation(
                vertices=self.array(result.vertices), faces=self.array(result.faces)
            )
        elif action == "export":
            payload = ExportPresentation(
                mesh=self.mesh_result(result.mesh),
                original_vertices=self.array(result.original_vertices),
                original_faces=self.array(result.original_faces),
            )
        else:
            payload = EmptyPayload()
        self.events.put(
            ResultEvent(**self.context.fields(), action=action, payload=payload)
        )

    def show_error(self, message, details=""):
        # The execution adapter emits the correlated failure exactly once.
        logger.error("%s\n%s", message, details)

    def close(self):
        self.closed = True
        self.execution.close()
        try:
            self.queue.put_nowait(None)
        except Full:
            pass


class WorkerServer:
    """Bind only loopback, authenticate a single host, and exit on disconnect."""

    def __init__(self, root, token):
        self.root, self.token = root, token
        self.connected = Event()
        self.stopped = Event()

    def authenticate(self, connection, request):
        authorization = request.headers.get("Authorization", "")
        if request.headers.get("Origin") is not None or not secrets.compare_digest(
            authorization, f"Bearer {self.token}"
        ):
            return connection.respond(HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
        if self.connected.is_set():
            return connection.respond(
                HTTPStatus.CONFLICT, "Session already connected\n"
            )
        self.connected.set()
        return None

    def handle(self, socket):
        events = EventBuffer()
        session = ApplicationSession(self.root, events)
        sender = Thread(
            target=events.send, args=(socket,), name="interdeca-events", daemon=True
        )
        sender.start()
        try:
            for raw in socket:
                if not isinstance(raw, str):
                    socket.close(1003, "JSON text frames required")
                    break
                session.queue.put(raw)
        except ConnectionClosed:
            pass
        finally:
            session.close()
            events.closed = True
            self.stopped.set()

    def run(self):
        with serve(
            self.handle,
            "127.0.0.1",
            0,
            process_request=self.authenticate,
            compression=None,
            max_size=1024 * 1024,
            max_queue=16,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=2,
        ) as server:
            # A descriptor is published only after bind succeeds; stdout remains diagnostic.
            descriptor = self.root / "endpoint.json"
            temporary = descriptor.with_suffix(".partial")
            temporary.write_text(
                json.dumps(
                    {
                        "protocol_version": 1,
                        "port": server.socket.getsockname()[1],
                    }
                )
            )
            os.replace(temporary, descriptor)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            if not self.connected.wait(60):
                raise TimeoutError("No application client connected within 60 seconds.")
            self.stopped.wait()
            server.shutdown()


def main():
    """Start a private worker; the launcher passes its secret through the environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-directory", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.session_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    diagnostic_file = logging.FileHandler(root / "worker.log", encoding="utf-8")
    diagnostic_file.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(threadName)s request=%(request_id)s run=%(run_id)s %(name)s: %(message)s",
            defaults={"request_id": "-", "run_id": "-"},
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[diagnostic_file])
    token = os.environ.pop("INTERDECA_SESSION_TOKEN")
    WorkerServer(root, token).run()


if __name__ == "__main__":
    main()
