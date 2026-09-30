"""Authenticated WebSocket host for the shared InterDeCA application session."""

import argparse
import json
import logging
import os
import secrets
from collections import deque
from http import HTTPStatus
from pathlib import Path
from threading import Condition, Event, Thread

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from interdeca.application import ApplicationSession
from interdeca.serde import LogEntryData, LogEvent, ProgressEvent, StateEvent

logger = logging.getLogger(__name__)


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
                session.submit(raw)
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
