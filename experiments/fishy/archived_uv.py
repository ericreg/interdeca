"""Narrow reader for VTK's archived triangle-expanded OBJ texture layout."""

from pathlib import Path

import numpy as np


class ArchivedUV:
    """Verify texture correspondence without declaring collapsed geometry valid."""

    @staticmethod
    def inspect(path: Path, atlas_uv: np.ndarray) -> np.ndarray:
        """Return a degenerate-face mask after exact archived-layout verification.

        Only triangle-expanded positive v/vt indices are accepted. Every ordered
        corner must equal the atlas UV rounded to float32, matching the legacy
        vtkFloatArray export. No arbitrary tolerance, face sorting or repair is
        allowed. This contract supports texture sampling, never surface analysis.
        """
        vertices, texture_vertices = [], []
        face_count = 0
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.partition("#")[0].split()
                if not fields:
                    continue
                try:
                    if fields[0] == "v":
                        if len(fields) != 4:
                            raise ValueError("Expected three vertex coordinates.")
                        vertices.append([float(v) for v in fields[1:]])
                    elif fields[0] == "vt":
                        if len(fields) != 3:
                            raise ValueError("Expected two texture coordinates.")
                        texture_vertices.append([float(v) for v in fields[1:]])
                    elif fields[0] == "f":
                        if len(fields) != 4:
                            raise ValueError("Only triangular faces are supported.")
                        # VTK's expanded export enumerates each face corner once.
                        # A reorder, shared index, or missing UV must fail visibly.
                        for corner, token in enumerate(fields[1:]):
                            indices = token.split("/")
                            expected = 3 * face_count + corner + 1
                            if len(indices) != 2 or any(
                                int(index) != expected for index in indices
                            ):
                                raise ValueError(
                                    "Expected sequential expanded v/vt indices."
                                )
                            if expected > min(len(vertices), len(texture_vertices)):
                                raise ValueError("Face references an undefined corner.")
                        face_count += 1
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
        if (
            face_count != len(atlas_uv)
            or len(vertices) != 3 * face_count
            or len(texture_vertices) != 3 * face_count
        ):
            raise ValueError(f"{path}: archived corner/face counts do not match atlas.")
        points = np.asarray(vertices)
        uv = np.asarray(texture_vertices).reshape(face_count, 3, 2)
        if not np.isfinite(points).all() or not np.isfinite(uv).all():
            raise ValueError(f"{path}: nonfinite archived coordinates.")
        expected_uv = atlas_uv.astype(np.float32).astype(np.float64)
        if not np.array_equal(uv, expected_uv):
            raise ValueError(f"{path}: ordered corners differ from float32 atlas UVs.")
        # Keep the historical defects measurable, but do not use these specimen
        # positions for sampling, areas, centroids, or downstream shape claims.
        triangles = points.reshape(face_count, 3, 3)
        edges = triangles[:, 1:] - triangles[:, :1]
        area_twice = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
        threshold = (
            np.finfo(float).eps
            * 32
            * np.linalg.norm(edges[:, 0], axis=1)
            * np.linalg.norm(edges[:, 1], axis=1)
        )
        return area_twice <= threshold
