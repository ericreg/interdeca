"""MorphoWeave atlas backend, using upstream workflows through a private host.

The original InterDeCA Atlas remains a separate comparison backend. This module
does not substitute its algorithms when MorphoWeave fails. TPS, upstream mean
construction and float32 outputs remain upstream properties. The application
restores representative input scale after mean construction, without changing
the upstream calculation or the original comparison backend.
"""

import logging
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet, vtkPolyData

from interdeca._morphoweave_host import Host, load_atlas_builder
from interdeca.dataset import Dataset
from interdeca.io import LandmarkIO, MeshIO
from interdeca.models import (
    AtlasModel,
    AtlasScaleConversion,
    AtlasScaleResult,
    Landmarks,
    Mesh,
    ResampledSpecimen,
    Specimen,
)
from interdeca.serde import CorrespondenceMeshData, ResampledSpecimenData

logger = logging.getLogger(__name__)


class CoordinateCheck(NamedTuple):
    """The upstream workflow reads this control; coordinate validation stays enabled."""

    def isChecked(self) -> bool:
        return False


class BuildContext(NamedTuple):
    """Only the three host inputs read by the upstream atlas-construction workflow."""

    logic: Any
    status: Callable[[str], None]
    overrideCoordCheck: CoordinateCheck = CoordinateCheck()

    def _status(self, message: str) -> None:
        self.status(message)


class MorphoWeave:
    """Numerical entry points shared by CLI and the independent Slicer worker."""

    @staticmethod
    def _validate_landmarks(landmarks: Landmarks) -> None:
        # VTK may otherwise report a singular spline without raising a Python
        # exception. Preserve the core's requirement for a determinate 3-D warp.
        if (
            len(landmarks.points) < 4
            or np.linalg.matrix_rank(landmarks.points - landmarks.points.mean(axis=0))
            < 3
        ):
            raise ValueError(
                "MorphoWeave TPS needs at least four noncoplanar landmarks."
            )
        if len(np.unique(landmarks.points, axis=0)) != len(landmarks.points):
            raise ValueError("MorphoWeave TPS landmarks must have distinct positions.")

    @staticmethod
    def _inputs(specimens: Sequence[Specimen], root: Path) -> dict[str, str]:
        # Use the upstream directory contract in a private workspace. Source UVs
        # survive the RAS OBJ staging; input files and the user's scene are untouched.
        folders = {}
        for name in ("originalModels", "originalLMs", "alignedModels", "alignedLMs"):
            path = root / name
            path.mkdir()
            folders[name] = str(path)
        identifiers = [specimen.specimen_id.lower() for specimen in specimens]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(
                "MorphoWeave pairs filenames case-insensitively; specimen IDs must remain unique after lowercasing."
            )
        for specimen in specimens:
            if Path(specimen.specimen_id).name != specimen.specimen_id:
                raise ValueError("A specimen ID must be a filename stem.")
            MeshIO.write_obj(
                specimen.mesh,
                Path(folders["originalModels"]) / f"{specimen.specimen_id}.obj",
            )
            LandmarkIO.write_json(
                specimen.landmarks,
                Path(folders["originalLMs"]) / f"{specimen.specimen_id}.mrk.json",
            )
        return folders

    @staticmethod
    def build_atlas(
        specimens: Sequence[Specimen], *, normalize_scale: bool = False
    ) -> AtlasModel:
        """Run upstream reference selection, pre-alignment, and mean construction.

        normalize_scale controls upstream runAlign. Upstream reference/mean GPA
        independently uses similarity alignment even when this setting is false.
        Exact duplicate vertices are cleaned before establishing atlas topology;
        the returned atlas still uses upstream scale. The application calls
        restore_input_scale before Blender cleanup and UV generation.
        """
        specimens = Dataset.validate_specimens(specimens)
        for specimen in specimens:
            MorphoWeave._validate_landmarks(specimen.landmarks)
        host = Host(specimens[0].landmarks.labels)
        upstream = load_atlas_builder(host)
        context = BuildContext(upstream.MorphoWeaveAtlasBuilderLogic(), logger.info)
        with tempfile.TemporaryDirectory(
            prefix="interdeca-morphoweave-mean-"
        ) as directory:
            folders = MorphoWeave._inputs(specimens, Path(directory))
            # Call the actual upstream high-level workflow without constructing widgets.
            model, landmarks = upstream.MorphoWeaveAtlasBuilderWidget._generateAtlas(
                context, folders, normalize_scale, "tps", False
            )
            # Nearest-vertex mean construction can create duplicate positions. Cleanup
            # is legal here because no atlas UVs or population index contract exists yet.
            surface = context.logic.cleanPolyData(model.GetPolyData(), normals=False)
            return AtlasModel(host.mesh(surface), landmarks.landmarks)

    @staticmethod
    def restore_input_scale(
        atlas: AtlasModel, specimens: Sequence[Specimen]
    ) -> AtlasScaleResult:
        """Scale an upstream atlas to the arithmetic mean input centroid size.

        Each specimen contributes sqrt(sum(||landmark - centroid||**2)) with
        equal weight. All inputs must use the same length units and landmark
        labels. The mesh and landmarks undergo one uniform transform about the
        atlas landmark centroid, preserving pose, faces, UVs, and mean shape.

        Measure the actual upstream size instead of assuming it is exactly one.
        This changes only the atlas coordinate scale; individual specimen size
        normalization remains a separate alignment choice. Call this before
        distance-based cleanup, and never on a supplied atlas override.
        """
        specimens = Dataset.validate_specimens(specimens)
        landmarks = atlas.landmarks.validated()
        center = landmarks.points.mean(axis=0)
        source_size = float(np.linalg.norm(landmarks.points - center))
        # Average sizes of original configurations, not coordinates or squared
        # sizes: translations and the alignment mode must not change this target.
        sizes = []
        for specimen in specimens:
            points = specimen.landmarks.reordered(landmarks.labels).points
            sizes.append(float(np.linalg.norm(points - points.mean(axis=0))))
        target_size = float(np.mean(sizes))
        if (
            not np.isfinite([source_size, target_size]).all()
            or min(source_size, target_size) <= 0
        ):
            raise ValueError(
                "Atlas scale conversion requires finite positive centroid sizes."
            )
        factor = target_size / source_size
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(
                "Atlas scale conversion requires a finite positive factor."
            )
        # Apply the identical map to the surface and anatomical anchors. New
        # records leave the normalized atlas and the input specimens untouched.
        scaled = AtlasModel(
            Mesh(
                center + factor * (atlas.mesh.vertices - center),
                atlas.mesh.faces,
                atlas.mesh.uv,
            ).validated(),
            Landmarks(
                center + factor * (landmarks.points - center), landmarks.labels
            ).validated(),
        )
        conversion = AtlasScaleConversion(
            source_size, target_size, factor, tuple(float(value) for value in center)
        )
        return AtlasScaleResult(scaled, conversion)

    @staticmethod
    def align_specimens(
        specimens: Sequence[Specimen],
        atlas: AtlasModel,
        *,
        normalize_scale: bool = False,
    ) -> tuple[Specimen, ...]:
        """Use upstream runAlign, preserving native triangle order and corner UVs."""
        specimens = Dataset.validate_specimens(specimens)
        host = Host(atlas.landmarks.labels)
        upstream = load_atlas_builder(host)
        logic = upstream.MorphoWeaveAtlasBuilderLogic()
        base = host.AddNewNodeByClass("vtkMRMLModelNode", "atlas")
        base.SetAndObservePolyData(MeshIO.to_polydata(atlas.mesh))
        landmarks = host.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "atlas")
        landmarks.landmarks = atlas.landmarks
        with tempfile.TemporaryDirectory(
            prefix="interdeca-morphoweave-align-"
        ) as directory:
            folders = MorphoWeave._inputs(specimens, Path(directory))
            logic.runAlign(
                base,
                landmarks,
                folders["originalModels"],
                folders["originalLMs"],
                folders["alignedModels"],
                folders["alignedLMs"],
                normalize_scale,
                progress=lambda done, total, name: logger.info(
                    "Aligned %s (%s/%s)", name, done, total
                ),
            )
            aligned = []
            for specimen in specimens:
                # Upstream lowercases file pairing; restore the original application ID.
                stem = specimen.specimen_id.lower()
                mesh = MeshIO.read(Path(folders["alignedModels"]) / f"{stem}.obj")
                if mesh.topology_key != specimen.mesh.topology_key:
                    raise ValueError(
                        f"MorphoWeave alignment changed source topology or UVs: {specimen.specimen_id}"
                    )
                points = LandmarkIO.read(
                    Path(folders["alignedLMs"]) / f"{stem}.mrk.json"
                )
                aligned.append(
                    Specimen(
                        specimen.specimen_id, mesh, points, specimen.texture_path
                    ).validated()
                )
            return tuple(aligned)

    @staticmethod
    def _landmark_polydata(landmarks: Landmarks) -> vtkPolyData:
        points = vtkPoints()
        points.SetData(numpy_to_vtk(landmarks.points, deep=True))
        data = vtkPolyData()
        data.SetPoints(points)
        return data

    @staticmethod
    def resample_specimen(specimen: Specimen, atlas: AtlasModel) -> ResampledSpecimen:
        """Project every atlas vertex through the upstream TPS correspondence operation.

        The selected index sequence is exactly 0..N-1, not the upstream default
        landmark subset. Faces and UVs are attached unchanged after projection.
        Diagnostics measure query-to-surface distance in the aligned specimen's
        space, unlike the original backend's common warped space.
        """
        host = Host(atlas.landmarks.labels)
        logic = load_atlas_builder(host).MorphoWeaveAtlasBuilderLogic()
        MorphoWeave._validate_landmarks(atlas.landmarks)
        MorphoWeave._validate_landmarks(specimen.landmarks)
        # A single-specimen batch keeps upstream behavior while preserving our task
        # boundaries for progress and cooperative cancellation between specimens.
        surfaces, landmarks = vtkMultiBlockDataSet(), vtkMultiBlockDataSet()
        surfaces.SetBlock(0, MeshIO.to_polydata(specimen.mesh))
        subject = MorphoWeave._landmark_polydata(
            specimen.landmarks.reordered(atlas.landmarks.labels)
        )
        landmarks.SetBlock(0, subject)
        atlas_lm = MorphoWeave._landmark_polydata(atlas.landmarks).GetPoints()
        surface = MeshIO.to_polydata(atlas.mesh)
        indices = np.arange(len(atlas.mesh.vertices), dtype=np.int64)
        result = logic.sparseCorrespondenceBaseMesh(
            landmarks,
            surfaces,
            surface,
            atlas_lm,
            indices,
            warp_method="tps",
            allow_fallback=False,
        )[0]
        # Upstream exports only projected points. Reuse its TPS implementation on
        # the same float32 queries to retain honest projection-distance diagnostics.
        queries = vtkPolyData()
        points = vtkPoints()
        points.SetData(numpy_to_vtk(atlas.mesh.vertices.astype(np.float32), deep=True))
        queries.SetPoints(points)
        warped = logic._warp_polydata_tps(queries, atlas_lm, subject.GetPoints())
        distances = np.linalg.norm(
            result - vtk_to_numpy(warped.GetPoints().GetData()), axis=1
        )
        mesh = CorrespondenceMeshData(
            vertices=result, faces=atlas.mesh.faces, uv=atlas.mesh.uv
        ).to_internal()
        return ResampledSpecimenData.from_internal(
            ResampledSpecimen(specimen.specimen_id, mesh, specimen.landmarks, distances)
        ).to_internal()
