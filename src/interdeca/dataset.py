"""Input inspection and population-wide validation before expensive operations."""

from collections.abc import Sequence
from pathlib import Path
from shutil import copyfile, rmtree

import numpy as np

from interdeca.io import Files, LandmarkIO, MeshIO, TextureIO
from interdeca.models import CoordinateSystem, Mesh, Specimen
from interdeca.serde import InputPreparationData


class Dataset:
    """Namespace for loading specimens and checking their shared assumptions."""

    @staticmethod
    def reset_workspace(input_dir: Path, protected_paths: tuple[Path, ...]) -> None:
        """Delete generated storage for an explicitly requested overwrite run.

        This operation runs before preparation, within the workflow's execution
        order. Raw inputs and atlas overrides must not point into that storage.
        """
        workspace = input_dir.resolve() / "_interdeca"
        # Recheck links at execution time as well as request validation, before
        # deleting anything. Linked source files must survive the cleanup too.
        if workspace.is_symlink():
            raise ValueError("Cannot overwrite a symlinked _interdeca directory.")
        if any(path.resolve().is_relative_to(workspace) for path in protected_paths):
            raise ValueError("Overwrite inputs and atlas overrides must be outside _interdeca.")
        if workspace.exists():
            rmtree(workspace)

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
        here to expose corrupt input before atlas construction and baking.
        """
        # Normalize each file's coordinate convention before any comparison;
        # mesh and landmark files may encode the same specimen using different axes.
        mesh = MeshIO.read(mesh_path, coordinate_system=mesh_coordinate_system)
        landmarks = LandmarkIO.read(
            landmarks_path, coordinate_system=landmark_coordinate_system
        )
        # Discover missing or unreadable textures now, before expensive geometry
        # and baking work has been scheduled for this specimen.
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
            TextureIO.validate_file(texture)
        # Apply the same alignment requirements used for a whole population so
        # inspection cannot report a specimen as usable when alignment would fail.
        specimen = Specimen(specimen_id, mesh, landmarks, texture).validated()
        Dataset.validate_specimens([specimen])
        return specimen

    @staticmethod
    def prepare_specimen(
        specimen_id: str,
        mesh_path: Path | str,
        landmarks_path: Path | str,
        texture_path: Path | str,
        directory: Path,
        *,
        mesh_coordinate_system: CoordinateSystem = "LPS",
    ) -> Specimen:
        """Normalize one raw specimen and publish its private RAS input copy.

        LPS to RAS reverses X/Y and preserves distances and triangle winding.
        Only exactly zero-area source faces are removed before correspondence;
        original vertex order and surviving face order and UVs remain unchanged.
        Repair indices are recorded in a JSON file. No original file is modified.
        Every workflow supplies a fresh directory under its input's _interdeca.
        """
        # A raw face can collapse to a line or point and cannot support source
        # projection. Thin nonzero faces remain subject to ordinary validation.
        mesh = MeshIO.read(
            mesh_path, coordinate_system=mesh_coordinate_system, allow_degenerate=True
        )
        triangles = mesh.vertices[mesh.faces]
        doubled_area = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        removed = np.flatnonzero(doubled_area == 0)
        if removed.size:
            keep = doubled_area != 0
            mesh = Mesh(
                mesh.vertices,
                mesh.faces[keep],
                None if mesh.uv is None else mesh.uv[keep],
            ).validated()
            del keep
        else:
            mesh = mesh.validated()
        # Large geometry snapshots have no role in image validation or export.
        removed_faces = removed.tolist()
        del triangles, doubled_area, removed

        # Both records enter the same RAS frame; source image bytes and UVs do
        # not change axes. Reject unusable inputs before publishing their copies.
        landmarks = LandmarkIO.read(landmarks_path)
        texture = Path(texture_path).expanduser().resolve(strict=True)
        if mesh.uv is None:
            raise ValueError(f"{specimen_id}: a textured source mesh needs UV coordinates.")
        TextureIO.validate_file(texture)
        specimen = Specimen(specimen_id, mesh, landmarks, texture).validated()
        Dataset.validate_specimens([specimen], require_textures=True)

        # Private per-run names prevent concurrent preparations and failed runs
        # from replacing each other. Images are streamed as unaltered file bytes.
        directory = Path(directory)
        MeshIO.write_obj(mesh, directory / "models" / f"{specimen_id}.obj")
        LandmarkIO.write_json(landmarks, directory / "landmarks" / f"{specimen_id}.mrk.json")
        destination = directory / "textures" / texture.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        copyfile(texture, destination)
        preparation = InputPreparationData(
            specimen_id=specimen_id,
            source_mesh=Path(mesh_path).resolve(),
            source_landmarks=Path(landmarks_path).resolve(),
            source_texture=texture,
            mesh_coordinates=mesh_coordinate_system,
            removed_zero_area_faces=removed_faces,
        )
        Files.write(
            directory / "preparation" / f"{specimen_id}.json",
            (preparation.model_dump_json(indent=2) + "\n").encode(),
        )
        # Downstream algorithms reuse the validated arrays rather than parsing
        # the just-written OBJ; the copied image snapshots this workflow's input.
        return Specimen(specimen_id, mesh, landmarks, destination.resolve())

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
