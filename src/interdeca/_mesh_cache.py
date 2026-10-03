"""Bounded reuse of parsed files inside one private MorphoWeave workspace."""

from collections import OrderedDict
from pathlib import Path
from sys import getsizeof
from typing import NamedTuple, Self

from interdeca.io import MeshIO
from interdeca.models import CoordinateSystem, Mesh


class FileVersion(NamedTuple):
    """A private file's identity, including replacement and in-place modification."""

    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def read(cls, path: Path) -> Self:
        status = path.stat()
        return cls(
            status.st_dev,
            status.st_ino,
            status.st_size,
            status.st_mtime_ns,
            status.st_ctime_ns,
        )


class ReadKey(NamedTuple):
    path: str
    coordinate_system: CoordinateSystem
    allow_degenerate: bool


class CacheEntry(NamedTuple):
    version: FileVersion
    mesh: Mesh
    array_bytes: int


class CacheStats(NamedTuple):
    """Array storage and Python-object estimates are separate from VTK/RSS."""

    hits: int = 0
    misses: int = 0
    bypasses: int = 0
    evictions: int = 0
    invalidations: int = 0
    array_bytes: int = 0
    peak_array_bytes: int = 0
    entries: int = 0
    peak_entries: int = 0
    metadata_bytes: int = 0
    peak_metadata_bytes: int = 0


class MeshCache:
    """Reuse immutable parsed originals; evict the least recently read entry.

    The byte budget bounds retained NumPy buffers, not transient file parsing or
    the caller's VTK copies. Python metadata is reported separately and bounded
    to at most 256 entries. Only files directly in the private originals folder
    are admitted: intermediate files read once must not evict reusable originals.
    Each workspace owns one instance and closes it before its run ends. This
    object is confined to the serial numerical thread, with no global cache.
    """

    def __init__(self, originals: Path, budget_bytes: int):
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("Mesh cache budget must be a nonnegative integer.")
        self.originals = originals.resolve()
        self.budget_bytes = budget_bytes
        self._entries: OrderedDict[ReadKey, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.closed = False

    def _account(self) -> None:
        # MeshIO returns owned buffers. Count each retained array once and report
        # object/header estimates separately rather than presenting them as RSS.
        arrays = sum(entry.array_bytes for entry in self._entries.values())
        metadata = getsizeof(self._entries)
        for key, entry in self._entries.items():
            metadata += (
                getsizeof(key)
                + getsizeof(key.path)
                + getsizeof(entry)
                + getsizeof(entry.version)
                + getsizeof(entry.mesh)
                + sum(getsizeof(value) for value in entry.version)
                + sum(
                    getsizeof(array) - array.nbytes
                    for array in entry.mesh
                    if array is not None
                )
            )
        self.stats = self.stats._replace(
            array_bytes=arrays,
            peak_array_bytes=max(arrays, self.stats.peak_array_bytes),
            entries=len(self._entries),
            peak_entries=max(len(self._entries), self.stats.peak_entries),
            metadata_bytes=metadata,
            peak_metadata_bytes=max(metadata, self.stats.peak_metadata_bytes),
        )

    def invalidate(self, path: Path | str) -> None:
        """Forget every conversion of a path before a save, including failed saves."""
        source = str(Path(path).expanduser().resolve())
        removed = [key for key in self._entries if key.path == source]
        for key in removed:
            del self._entries[key]
        if removed:
            self.stats = self.stats._replace(
                invalidations=self.stats.invalidations + len(removed)
            )
            self._account()

    def read(
        self,
        path: Path | str,
        *,
        coordinate_system: CoordinateSystem = "RAS",
        allow_degenerate: bool = False,
    ) -> Mesh:
        """Read through the ordinary validator on misses; hits retain its result.

        File versions and conversion settings both participate in identity. These
        are private staged files, not a synchronization mechanism for concurrent
        external writers. A file changed during parsing is never admitted.
        """
        if self.closed:
            raise RuntimeError("Mesh cache has already closed.")
        source = Path(path).expanduser().resolve(strict=True)
        key = ReadKey(str(source), coordinate_system, allow_degenerate)
        version = FileVersion.read(source)
        entry = self._entries.get(key)
        if entry is not None and entry.version == version:
            self._entries.move_to_end(key)
            self.stats = self.stats._replace(hits=self.stats.hits + 1)
            return entry.mesh
        # Discard all settings for a changed file; stale variants must not consume
        # capacity or become hits after a later coordinate-system change.
        if any(
            k.path == key.path and value.version != version
            for k, value in self._entries.items()
        ):
            self.invalidate(source)
        self.stats = self.stats._replace(misses=self.stats.misses + 1)
        mesh = MeshIO.read(
            source,
            coordinate_system=coordinate_system,
            allow_degenerate=allow_degenerate,
        )
        size = sum(array.nbytes for array in mesh if array is not None)
        if (
            self.budget_bytes == 0
            or source.parent != self.originals
            or size > self.budget_bytes
            or FileVersion.read(source) != version
        ):
            self.stats = self.stats._replace(bypasses=self.stats.bypasses + 1)
            return mesh
        # Eviction releases only the cache's references. A caller already using
        # a returned immutable mesh keeps valid buffers through its own reference.
        live = self.stats.array_bytes
        while self._entries and (
            live + size > self.budget_bytes or len(self._entries) >= 256
        ):
            _, old = self._entries.popitem(last=False)
            live -= old.array_bytes
            self.stats = self.stats._replace(evictions=self.stats.evictions + 1)
        self._entries[key] = CacheEntry(version, mesh, size)
        self._account()
        return mesh

    def close(self) -> None:
        """Release buffers while retaining only small diagnostic counters."""
        self._entries.clear()
        self.closed = True
        self._account()
