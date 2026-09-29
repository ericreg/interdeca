"""Input inspection and population-wide validation before expensive operations."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from interdeca.io import LandmarkIO, MeshIO, TextureIO
from interdeca.models import CoordinateSystem, Specimen


class Dataset:
    """Namespace for loading specimens and checking their shared assumptions."""

    @staticmethod
    def inspect_specimen(
        specimen_id: str,
        mesh_path: Path | str,
        landmarks_path: Path | str,
        texture_path: Path | str | None = None,
        *,
        mesh_coordinate_system: CoordinateSystem = "RAS",
        landmark_coordinate_system: CoordinateSystem | None = None,
    ) -> Specimen:
        """Read and validate one specimen without searching for implicit files.

        Mesh and landmark axes can differ on disk; both are converted to RAS.
        Their length units must already agree. Texture files are decoded once
        here to expose corrupt input before Blender is started.
        """
        # Normalize each file's coordinate convention before any comparison;
        # mesh and landmark files may encode the same specimen using different axes.
        mesh = MeshIO.read(mesh_path, coordinate_system=mesh_coordinate_system)
        landmarks = LandmarkIO.read(
            landmarks_path, coordinate_system=landmark_coordinate_system
        )
        # Discover missing or unreadable textures now, before expensive geometry
        # and Blender work has been scheduled for this specimen.
        texture = (
            None
            if texture_path is None
            else Path(texture_path).expanduser().resolve(strict=True)
        )
        if texture is not None:
            if mesh.uv is None:
                raise ValueError(
                    f"{specimen_id}: a textured source mesh needs UV coordinates."
                )
            TextureIO.read(texture)
        # Apply the same alignment requirements used for a whole population so
        # inspection cannot report a specimen as usable when alignment would fail.
        specimen = Specimen(specimen_id, mesh, landmarks, texture).validated()
        Dataset.validate_specimens([specimen])
        return specimen

    @staticmethod
    def validate_specimens(
        specimens: Sequence[Specimen], *, require_textures: bool = False
    ) -> tuple[Specimen, ...]:
        """Validate identities and geometry, returning homologous landmark order.

        At least three noncollinear landmarks are needed for rigid alignment.
        Nonrigid resampling imposes the stronger requirement of at least four
        noncoplanar landmarks. Label matching is exact and case sensitive.
        """
        # Stable ordering and unique IDs keep population statistics from counting
        # a specimen twice or associating results with the wrong input.
        specimens = tuple(specimens)
        if not specimens:
            raise ValueError("At least one specimen is required.")
        identifiers = [specimen.specimen_id for specimen in specimens]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Specimen IDs must be unique within a population.")
        # Anatomical labels establish correspondence; file row order does not.
        labels = specimens[0].landmarks.labels
        ordered = []
        for specimen in specimens:
            landmarks = specimen.landmarks.reordered(labels)
            # Collinear landmarks leave some rotations undetermined, and repeated
            # positions make the later spline equations singular.
            centered = landmarks.points - landmarks.points.mean(axis=0)
            if len(centered) < 3 or np.linalg.matrix_rank(centered) < 2:
                raise ValueError(
                    f"{specimen.specimen_id}: alignment needs three noncollinear landmarks."
                )
            if len(np.unique(landmarks.points, axis=0)) != len(landmarks.points):
                raise ValueError(
                    f"{specimen.specimen_id}: two landmarks occupy the same position."
                )
            # Texture-free shape work is allowed, but a baking workflow needs both
            # an image and coordinates describing where that image belongs.
            if require_textures and (
                specimen.texture_path is None or specimen.mesh.uv is None
            ):
                raise ValueError(
                    f"{specimen.specimen_id}: baking requires a texture and source UVs."
                )
            if (
                specimen.texture_path is not None
                and not specimen.texture_path.is_file()
            ):
                raise FileNotFoundError(specimen.texture_path)
            # Return reordered copies so other tasks can keep reading the original
            # specimens without seeing their landmark order change.
            ordered.append(
                Specimen(
                    specimen.specimen_id,
                    specimen.mesh,
                    landmarks,
                    specimen.texture_path,
                ).validated()
            )
        return tuple(ordered)
