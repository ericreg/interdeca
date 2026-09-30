"""Numeric file exchange usable without Pydantic or the computational package."""

import os
from pathlib import Path
from uuid import uuid4

import numpy as np


class ArtifactStore:
    """Publish complete .npy files atomically inside one private session directory.

    Each reference has exactly one consumer. Inputs live until their command's
    terminal event; outputs live until the client has copied or discarded them.
    Closing the session removes both namespaces. Logs are stored outside them.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        for owner in ("inputs", "outputs"):
            (self.root / owner).mkdir(parents=True, exist_ok=True)

    def path(self, reference, owner):
        """Reject traversal, wrong ownership, and symlinks escaping the session."""
        relative = Path(reference["path"])
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] != owner
            or relative.suffix != ".npy"
        ):
            raise ValueError("Invalid session artifact path.")
        path = (self.root / relative).resolve()
        if path.parent != self.root / owner:
            raise ValueError("Artifact escapes its session directory.")
        return path

    def write(self, array, owner):
        """Keep dtype, shape, and ordering explicit; object arrays are never accepted."""
        array = np.asarray(array)
        if array.dtype.kind not in "biuf" or array.ndim > 4:
            raise ValueError(
                "Artifacts must be numerical arrays of at most four dimensions."
            )
        relative = f"{owner}/{uuid4().hex}.npy"
        destination = self.root / relative
        temporary = destination.with_suffix(".partial")
        # Renaming on the same filesystem prevents the peer from seeing partial data.
        try:
            with temporary.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return {"path": relative, "dtype": array.dtype.str, "shape": list(array.shape)}

    def read(self, reference, owner):
        """Validate metadata before returning an owned, read-only numerical buffer."""
        path = self.path(reference, owner)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            array.dtype.kind not in "biuf"
            or array.dtype.str != reference["dtype"]
            or list(array.shape) != list(reference["shape"])
            or array.ndim > 4
        ):
            raise ValueError("Artifact metadata does not match its numerical file.")
        result = np.array(array)
        result.setflags(write=False)
        return result

    def discard(self, value, owner):
        """Release references in a JSON tree after their single consumer is finished."""
        if isinstance(value, dict):
            if set(value) == {"path", "dtype", "shape"}:
                self.path(value, owner).unlink(missing_ok=True)
            else:
                for item in value.values():
                    self.discard(item, owner)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.discard(item, owner)
