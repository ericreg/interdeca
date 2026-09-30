"""Transport and presentation adapter for Slicer or a future standalone Qt host."""

import json
import os
import secrets
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic
from typing import NamedTuple
from uuid import uuid4

from interdeca_client.artifacts import ArtifactStore
from interdeca_client.records import (
    AtlasPresentation,
    ExtractedRegion,
    FaceColors,
    Mesh,
    MorphControls,
    PlotData,
    PopulationPresentation,
    PresentationState,
    Selection,
    SelectionSummary,
)

# Hosts may expose their setup UI before the only additional client dependency exists.
try:
    from websockets.sync.client import connect
except ImportError:
    connect = None


class PendingCommand(NamedTuple):
    """Keep input file ownership until the corresponding terminal acknowledgement."""

    action: str
    payload: dict


class ApplicationClient:
    """Send user actions and dispatch presentation updates on the host's UI thread.

    The dispatcher must offer a thread-safe asynchronous post(callback). The view
    implements render, present, show_error, receive_log, and capabilities_changed.
    No MRML, Qt, controller, task, or fitted model enters this transport package.
    """

    def __init__(self, dispatcher, view):
        self.dispatcher, self.view = dispatcher, view
        self.state = PresentationState()
        self.connected = False
        self.methods = ()
        self.root = None
        self.process = None
        self._closed = Event()
        self._outgoing = Queue()
        self._pending = {}
        self._revision = 0
        self._population = None
        self._cancelled_requests = set()
        self._latest_reconstruction = None
        self._thread = None

    @property
    def busy(self):
        """Transport readiness and outstanding submissions also disable analysis buttons."""
        return (
            not self.connected
            or self.state.run_id is not None
            or any(
                item.action
                in (
                    "run_atlas",
                    "process_colors",
                    "fit_population",
                    "show_specimen",
                    "select_region",
                    "export_region",
                    "reconstruct",
                    "invalidate",
                    "clear_selection",
                )
                for item in self._pending.values()
            )
        )

    def start(self, executable):
        """Launch an independently installed Python environment and capture early failures."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(
                "The worker is running or stopping; wait for it to exit before starting another."
            )
        if connect is None:
            raise RuntimeError(
                "Install websockets>=15,<16 in Slicer's Python, then restart Slicer."
            )
        executable = Path(executable).expanduser()
        if not executable.is_file():
            raise ValueError(
                "Select the Python executable in the InterDeCA worker environment."
            )
        self.root = Path(tempfile.mkdtemp(prefix="interdeca-session-"))
        self.artifacts = ArtifactStore(self.root)
        self._closed.clear()
        self.state = PresentationState(
            status="starting", message="Starting application worker…"
        )
        self._revision = 0
        self._population = None
        self._cancelled_requests = set()
        self._latest_reconstruction = None
        self._pending.clear()
        self._outgoing = Queue()
        token = secrets.token_urlsafe(32)
        environment = os.environ.copy()
        # Slicer injects its runtime paths; the external interpreter must find its own ABI.
        for key in (
            "PYTHONHOME",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
            "DYLD_FRAMEWORK_PATH",
            "DYLD_FALLBACK_LIBRARY_PATH",
            "QT_PLUGIN_PATH",
            "QT_QPA_PLATFORM_PLUGIN_PATH",
        ):
            environment.pop(key, None)
        environment["INTERDECA_SESSION_TOKEN"] = token
        environment["PATH"] = (
            str(executable.parent) + os.pathsep + environment.get("PATH", "")
        )
        try:
            with (
                (self.root / "stdout.log").open("wb") as stdout,
                (self.root / "stderr.log").open("wb") as stderr,
            ):
                self.process = subprocess.Popen(
                    [
                        str(executable),
                        "-I",
                        "-u",
                        "-m",
                        "interdeca.worker",
                        "--session-directory",
                        str(self.root),
                    ],
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            self._thread = Thread(
                target=self._communicate,
                args=(token,),
                name="interdeca-client",
                daemon=True,
            )
            self._thread.start()
            self.view.render(self.state)
        except Exception:
            self._cleanup_artifacts()
            raise

    def _post(self, callback):
        # A stopped worker's callbacks must not run against a newly launched session.
        root = self.root
        self.dispatcher.post(
            lambda: (
                callback() if not self._closed.is_set() and self.root == root else None
            )
        )

    def _communicate(self, token):
        try:
            deadline = monotonic() + 60
            endpoint = self.root / "endpoint.json"
            while not endpoint.exists():
                if self._closed.wait(0.05):
                    return
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Worker exited during startup (code {self.process.returncode})."
                    )
                if monotonic() > deadline:
                    raise TimeoutError("Worker startup timed out after 60 seconds.")
            descriptor = json.loads(endpoint.read_text())
            if descriptor["protocol_version"] != 1:
                raise ValueError("Unsupported worker protocol version.")
            # Local traffic must never be sent to an environment-configured HTTP proxy.
            with connect(
                f"ws://127.0.0.1:{int(descriptor['port'])}",
                proxy=None,
                additional_headers={"Authorization": f"Bearer {token}"},
                compression=None,
                max_size=1024 * 1024,
                max_queue=16,
                open_timeout=5,
                close_timeout=2,
            ) as socket:
                ready = False
                deadline = monotonic() + 60
                while not self._closed.is_set():
                    if not ready and monotonic() > deadline:
                        raise TimeoutError(
                            "Worker capability detection timed out after 60 seconds."
                        )
                    for _ in range(20):
                        try:
                            socket.send(self._outgoing.get_nowait())
                        except Empty:
                            break
                    try:
                        raw = socket.recv(timeout=0.05)
                    except TimeoutError:
                        continue
                    message = json.loads(raw)
                    if message["protocol_version"] != 1:
                        raise ValueError("Unsupported application event version.")
                    ready = ready or message["type"] == "capabilities"
                    self._post(lambda message=message: self._deliver(message))
        except Exception as error:  # noqa: BLE001 -- isolate transport/startup failures from the UI
            if not self._closed.is_set():
                self._post(lambda message=str(error): self._disconnected(message))
        finally:
            # Preserve queued presentation artifacts until the UI consumes earlier callbacks.
            self._stop_process()
            if self._closed.is_set():
                self._cleanup_artifacts()
            else:
                self._post(self._cleanup_artifacts)

    def _disconnected(self, message):
        self.connected = False
        self._pending.clear()
        self._population = None
        self.state = PresentationState(status="disconnected", message=message)
        self.view.render(self.state)
        details = ""
        for filename in ("stdout.log", "stderr.log"):
            path = self.root / filename
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 16000))
                    details += f"\n{filename}:\n" + stream.read().decode(
                        "utf-8", errors="replace"
                    )
        self.view.show_error(f"{message}\nDiagnostics: {self.root}", details)

    def _deliver(self, message):
        kind = message["type"]
        request_id = message["request_id"]
        stale = message["input_revision"] < self._revision
        try:
            if kind == "capabilities":
                self.connected = True
                self.methods = tuple(message["methods"])
                self.view.capabilities_changed(self.methods)
                self.view.receive_log(
                    "INFO", f"Worker diagnostics: {message['log_path']}"
                )
            elif kind == "log":
                for entry in message["entries"]:
                    self.view.receive_log(entry["severity"], entry["message"])
                if message["dropped"]:
                    self.view.receive_log(
                        "INFO",
                        f"{message['dropped']} additional log entries saved to disk.",
                    )
            elif kind == "presentation":
                value = message["payload"]
                # State may describe a new revision even when triggered by an older run ending.
                if value["revision"] < self._revision:
                    return
                population = self._population
                if population is not None and population.model_id != value["model_id"]:
                    population = self._population = None
                self.state = PresentationState(
                    value["revision"],
                    value["run_id"],
                    value["action"],
                    value["status"],
                    value["completed"],
                    value["total"],
                    value["stage"],
                    value["stages"],
                    value["message"],
                    value["dataset_id"],
                    population,
                    SelectionSummary(
                        value["selection_vertices"], value["selection_faces"]
                    )
                    if value["selection_vertices"]
                    else None,
                    value["morph_active"],
                )
            elif kind == "result":
                # File ownership ends even if the user changed inputs before delivery.
                try:
                    if (
                        not stale
                        and request_id not in self._cancelled_requests
                        and not (
                            message["action"] in ("morphospace", "morph_controls")
                            and request_id != self._latest_reconstruction
                        )
                    ):
                        result = self._decode(message["action"], message["payload"])
                        self.view.present(message["action"], result)
                finally:
                    self.artifacts.discard(message["payload"], "outputs")
            elif kind in ("completed", "cancelled", "failed"):
                pending = self._pending.pop(request_id, None)
                if pending is not None:
                    self.artifacts.discard(pending.payload, "inputs")
                if (
                    kind == "failed"
                    and not stale
                    and request_id not in self._cancelled_requests
                ):
                    self.view.show_error(message["message"], message["details"])
                self._cancelled_requests.discard(request_id)
            # Progress remains structured; controls render the worker's accompanying snapshot.
            self.view.render(self.state)
        except Exception as error:  # noqa: BLE001 -- host rendering failures must not stop event delivery
            self.view.show_error(f"Could not apply worker update: {error}")

    def _array(self, value):
        return self.artifacts.read(value, "outputs")

    def _mesh(self, value):
        return Mesh(
            self._array(value["vertices"]),
            self._array(value["faces"]),
            None if value["uv"] is None else self._array(value["uv"]),
        )

    def _decode(self, action, value):
        if action == "atlas":
            return AtlasPresentation(
                self._mesh(value["mesh"]),
                Path(value["directory"]),
                value["has_textures"],
                None if value["average"] is None else Path(value["average"]),
            )
        if action == "population":
            self._population = PopulationPresentation(
                value["model_id"],
                value["method"],
                tuple(value["specimen_ids"]),
                value["components"],
                value["representation"],
                value["can_reconstruct"],
            )
            return self._population
        if action == "plot":
            return PlotData(
                tuple(value["specimen_ids"]),
                self._array(value["scores"]),
                value["method"],
                value["x_axis"],
                value["y_axis"],
            )
        if action == "morph_controls":
            return MorphControls(tuple(value["positions"]), tuple(value["coordinates"]))
        if action in ("display", "morphospace"):
            return FaceColors(self._array(value["rgb"]), self._array(value["valid"]))
        if action == "selection":
            return Selection(
                self._array(value["vertices"]), self._array(value["faces"])
            )
        if action == "export":
            return ExtractedRegion(
                self._mesh(value["mesh"]),
                self._array(value["original_vertices"]),
                self._array(value["original_faces"]),
            )
        if action in ("colors", "clear_selection"):
            return None
        raise ValueError(f"Unsupported result action: {action}")

    def send(self, action, **payload):
        """Submit a JSON operation; the worker validates all application input fields."""
        if not self.connected:
            raise RuntimeError(
                "Start the application worker before running an operation."
            )
        request_id = uuid4().hex
        message = {
            "protocol_version": 1,
            "type": "command",
            "request_id": request_id,
            "input_revision": self._revision,
            "action": action,
            "payload": payload,
        }
        try:
            encoded = json.dumps(message, allow_nan=False)
        except (TypeError, ValueError):
            # A command that never reaches the worker cannot receive a cleanup acknowledgement.
            self.artifacts.discard(payload, "inputs")
            raise
        self._pending[request_id] = PendingCommand(action, payload)
        self._outgoing.put(encoded)
        if action == "reconstruct":
            self._latest_reconstruction = request_id
        self.view.render(self.state)
        return request_id

    def _snapshot(self, mesh):
        if not self.connected:
            raise RuntimeError(
                "Start the application worker before publishing scene snapshots."
            )
        return {
            "vertices": self.artifacts.write(mesh.vertices, "inputs"),
            "faces": self.artifacts.write(mesh.faces, "inputs"),
            "uv": None if mesh.uv is None else self.artifacts.write(mesh.uv, "inputs"),
        }

    def run_atlas(self, **options):
        """Validate atlas options in the worker and start its pipeline."""
        return self.send("run_atlas", **options)

    def process_colors(self, mesh, **options):
        """Publish an ordered mesh snapshot and request a new processed dataset."""
        return self.send("process_colors", mesh=self._snapshot(mesh), options=options)

    def fit_population(self, **options):
        """Fit the current dataset; the returned model stays in the worker."""
        return self.send(
            "fit_population", dataset_id=self.state.colors, options=options
        )

    def show_specimen(self, specimen_id):
        """Request display colors from an already processed specimen."""
        return self.send(
            "show_specimen", dataset_id=self.state.colors, specimen_id=specimen_id
        )

    def select_region(self, mesh, points, same_side_only, vertex_colors=None):
        """Submit canonical mesh indices and world-coordinate curve points."""
        return self.send(
            "select_region",
            mesh=self._snapshot(mesh),
            points=self.artifacts.write(points, "inputs"),
            same_side_only=same_side_only,
            vertex_colors=None
            if vertex_colors is None
            else self.artifacts.write(vertex_colors, "inputs"),
        )

    def export_region(self, mesh):
        """Request the selected submesh and its original index maps."""
        return self.send("export_region", mesh=self._snapshot(mesh))

    def invalidate(self, scope="colors"):
        """Advance the input revision before the worker invalidates dependent state."""
        if self.connected:
            self._revision += 1
            return self.send("invalidate", scope=scope)

    def clear_selection(self):
        """Invalidate pending selection work and restore the scene appearance."""
        if self.connected:
            self._revision += 1
            return self.send("clear_selection")

    def cancel(self):
        """Suppress queued local results and request cooperative worker cancellation."""
        if self.connected:
            # A result may already be queued locally when Cancel is clicked.
            self._cancelled_requests.update(self._pending)
            return self.send("cancel")

    def set_axes(self, x_axis, y_axis):
        """Change zero-based plot columns without refitting the worker model."""
        if self.state.population is not None:
            return self.send(
                "set_axes",
                model_id=self.state.population.model_id,
                x_axis=x_axis,
                y_axis=y_axis,
            )

    def reconstruct(self, specimen_id, positions):
        """Send a starting specimen and slider fractions; model coordinates stay in the worker."""
        if self.state.population is not None:
            return self.send(
                "reconstruct",
                model_id=self.state.population.model_id,
                specimen_id=specimen_id,
                positions=positions,
            )

    def _cleanup_artifacts(self):
        if self.root is not None:
            for owner in ("inputs", "outputs"):
                shutil.rmtree(self.root / owner, ignore_errors=True)

    def _stop_process(self):
        if self.process is None:
            return
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        # POSIX workers own a fresh process group, including any active Blender child.
        # Disconnect ends that session; children must not continue writing orphaned files.
        if os.name == "posix":
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def close(self):
        """Detach updates immediately; process termination and cleanup happen off the UI thread."""
        self._closed.set()
        self.connected = False
        self._pending.clear()
        self._population = None
        self.state = PresentationState(status="disconnected", message="Worker stopped.")
