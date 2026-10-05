"""Lossless, verified mesh archives created only after timed captures finish."""

import gzip
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from interdeca.io import MeshIO


def archive_path(path: Path) -> Path | None:
    """Locate a capture-local compressed counterpart of a logical OBJ path."""
    for parent in path.parents:
        if (parent / "run.json").is_file():
            return parent / "mesh-archive" / (str(path.relative_to(parent)) + ".gz")
    return None


def digest(path: Path) -> str:
    """Hash original bytes, independently of whether their storage is compressed."""
    if path.is_file():
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    compressed = archive_path(path)
    if compressed is None or not compressed.is_file():
        raise FileNotFoundError(path)
    with gzip.open(compressed, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_meshes(directory: Path) -> None:
    """Preserve complete OBJ bytes before removing their uncompressed copies.

    Metadata, textures, presentation arrays, and profiler files remain ordinary
    files. The existing omission manifest retains original hashes for the standard
    fingerprint reader. Unlike omission alone, mesh-archive permits re-analysis.
    """
    manifest = directory / "omitted-artifacts.json"
    omitted = json.loads(manifest.read_text()) if manifest.exists() else {}
    for folder in ("atlas", "cli"):
        for path in sorted((directory / folder).rglob("*.obj")):
            expected = digest(path)
            destination = directory / "mesh-archive" / (str(path.relative_to(directory)) + ".gz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # A partial archive is disposable scratch while the original still
            # exists. Completed evidence is verified and never overwritten.
            if not destination.exists():
                temporary = destination.with_suffix(".partial")
                temporary.unlink(missing_ok=True)
                with path.open("rb") as source, temporary.open("xb") as output:
                    with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=3, mtime=0) as zipped:
                        shutil.copyfileobj(source, zipped, length=1024 * 1024)
                temporary.rename(destination)
            with gzip.open(destination, "rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Compressed mesh verification failed: {path}")
            omitted[str(path.relative_to(directory))] = expected
            # Record the verified identity before deletion so interruption leaves
            # either the original, its lossless archive, or both available.
            temporary = manifest.with_suffix(".partial")
            temporary.write_text(json.dumps(omitted, indent=2) + "\n")
            temporary.replace(manifest)
            path.unlink()


def read_mesh(path: Path, *, allow_degenerate: bool = False):
    """Read a preserved output without restoring an entire multi-GiB capture."""
    if path.is_file():
        return MeshIO.read(path, allow_degenerate=allow_degenerate)
    compressed = archive_path(path)
    if compressed is None or not compressed.is_file():
        raise FileNotFoundError(path)
    # The parser receives one temporary plain OBJ. Verify restored bytes against
    # the capture's original hash before interpreting geometry for accuracy work.
    run = next(parent for parent in path.parents if (parent / "run.json").is_file())
    expected = json.loads((run / "omitted-artifacts.json").read_text())[str(path.relative_to(run))]
    with tempfile.TemporaryDirectory(prefix="interdeca-profile-mesh-") as temporary:
        restored = Path(temporary) / "mesh.obj"
        with gzip.open(compressed, "rb") as source, restored.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if digest(restored) != expected:
            raise ValueError(f"Archived mesh hash differs from the capture: {path}")
        return MeshIO.read(restored, allow_degenerate=allow_degenerate)
