"""Thread-safe event delivery implemented entirely within the Slicer host."""

import logging
from queue import Empty, Queue
from time import sleep

import qt

logger = logging.getLogger(__name__)


class SlicerDispatcher:
    """The worker only posts callbacks; a UI-owned timer performs their delivery."""

    def __init__(self):
        self._callbacks = Queue()
        self._closed = False
        self._active = False
        self._timer = qt.QTimer()
        self._timer.setInterval(25)
        self._timer.connect("timeout()", self._deliver)
        self._timer.start()

    def post(self, callback):
        """Accept a worker notification without calling any Qt method from that worker."""
        if not self._closed:
            self._callbacks.put(callback)

    def set_active(self, active):
        """Yield Python execution while a session has work, as Slicer's SimpleFilters does."""
        self._active = active

    def _deliver(self):
        # PythonQt can hold the GIL while waiting in its C++ event loop. A short,
        # bounded yield lets the Python worker reacquire it after native/I/O calls.
        if self._active and not self._closed:
            sleep(0.005)
        # Bound each batch so a long event backlog cannot monopolize the event loop.
        for _ in range(100):
            try:
                callback = self._callbacks.get_nowait()
            except Empty:
                break
            if not self._closed:
                try:
                    callback()
                except Exception:
                    logger.exception("InterDeCA view notification failed")

    def close(self):
        """Stop delivery and release callbacks retaining closed application objects."""
        self._closed = True
        self._timer.stop()
        while not self._callbacks.empty():
            try:
                self._callbacks.get_nowait()
            except Empty:
                break
