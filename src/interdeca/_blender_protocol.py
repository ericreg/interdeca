"""Small, standard-library-only records shared with Blender's Python.

Keep this file compatible with Python 3.10 (Blender 3.6). The host's Pydantic
models own wire validation and serialization. The private worker consumes only
those validated requests, converting the JSON transport into named records.
"""

import json
from pathlib import Path
from typing import Literal, NamedTuple


class BlenderUVRequest(NamedTuple):
    """Paths and options for one atlas UV operation."""

    # Plain paths let the worker find artifacts without importing host-side mesh types.
    mesh: str
    output: str
    # Carry the layout choices with the request so the worker needs no shared settings.
    angle_limit_degrees: float
    island_margin: float
    # Identify the operation and format explicitly so mismatched workers can reject it.
    operation: Literal["uv"] = "uv"
    schema_version: Literal[1] = 1


class BlenderBakeRequest(NamedTuple):
    """Paths, image dimensions, and ray settings for one texture transfer."""

    # Separate source, target, and output paths prevent confusing image ownership.
    source: str
    target: str
    texture: str
    output: str
    # Make image size and ray choices explicit because they depend on this dataset.
    resolution: tuple[int, int]
    cage_extrusion: float
    max_ray_distance: float
    margin_pixels: int
    # The discriminator distinguishes this payload from a UV request with other fields.
    operation: Literal["bake"] = "bake"
    schema_version: Literal[1] = 1


class BlenderProtocol:
    """Decode the host-validated transport without importing host dependencies."""

    @staticmethod
    def read_request(path: Path) -> BlenderUVRequest | BlenderBakeRequest:
        """Convert a private JSON request immediately into its typed record.

        The JSON object exists only at this transport boundary. No worker
        computation receives a dictionary. This is not a public input API;
        external requests must first pass through the host Pydantic models.
        """
        # Refuse unknown formats before interpreting their fields as a known
        # operation; the host and worker must agree on the same contract.
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported Blender request schema.")
        # Convert immediately to a named record so computation cannot depend on
        # arbitrary dictionary keys or silently ignore unexpected request fields.
        operation = payload.get("operation")
        if operation == "uv":
            return BlenderUVRequest(**payload)
        if operation == "bake":
            request = BlenderBakeRequest(**payload)
            # JSON arrays become lists; restore the record's fixed tuple contract.
            return request._replace(resolution=tuple(request.resolution))
        raise ValueError(f"Unknown Blender operation: {operation!r}")
