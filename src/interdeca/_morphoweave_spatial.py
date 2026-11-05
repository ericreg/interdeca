"""Local surface-query adapters for the pinned, unmodified MorphoWeave source.

Only coordinate-distance checks and closest-triangle projection are replaced.
Upstream TPS, mean construction, nearest-vertex matching, and float32 rounding
remain unchanged. These adapters are installed on each private imported class.
"""

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData

from interdeca.models import Mesh
from interdeca.spatial import Spatial


class MorphoWeaveSpatial:
    """Surface-query implementations with the upstream method signatures."""

    @staticmethod
    def index(logic, mesh_pd):
        """Snapshot a triangular VTK surface without keeping the VTK object alive."""
        if mesh_pd.GetNumberOfCells() != mesh_pd.GetNumberOfPolys():
            raise ValueError("Surface queries require triangles without line/point cells.")
        offsets = vtk_to_numpy(mesh_pd.GetPolys().GetOffsetsArray())
        if not np.all(np.diff(offsets) == 3):
            raise ValueError("Surface queries require triangular faces.")
        faces = vtk_to_numpy(mesh_pd.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        vertices = vtk_to_numpy(mesh_pd.GetPoints().GetData())
        return Spatial.index(Mesh(vertices, faces))

    @staticmethod
    def coordinate_ratio(logic, pd, pts_np, threshold=None):
        """Preserve upstream's median absolute-distance coordinate convention."""
        diagonal = max(logic._bbox_diag(pd), 1e-9)
        base = logic.cleanPolyData(pd, tri=True, normals=False)
        if base.GetNumberOfPoints() == 0 or pts_np.size == 0:
            return float("inf"), float("inf")
        index = logic._build_surface_locator(base)
        # Only magnitudes were consumed from the old signed-distance API. The
        # unsigned closest-surface distance therefore keeps the same check.
        projected = Spatial.project(index, pts_np)
        ratio = float(np.median(np.sqrt(projected.squared_distances))) / diagonal
        if threshold is not None and ratio <= threshold:
            return ratio, float("inf")
        flipped = pts_np.copy()
        flipped[:, :2] *= -1.0
        projection = Spatial.project(index, flipped)
        other = float(np.median(np.sqrt(projection.squared_distances))) / diagonal
        return ratio, other

    @staticmethod
    def correspondence(logic, LMs, meshes, atlasMesh, atlasLM, keep_idx, *,
                       warp_method="tps", allow_fallback=True, progress=None,
                       mesh_locators=None):
        """Batch the pinned workflow's warped queries through parallel Parry.

        Upstream writes selected atlas points into float32 vtkPoints before TPS
        and rounds projected points to float32 afterward. Both conversions are
        retained, including when the incoming atlas vertices use float64.
        """
        if warp_method not in ("tps", "biharmonic"):
            raise ValueError(f"Unsupported MorphoWeave warp method: {warp_method}")
        n_landmarks = atlasLM.GetNumberOfPoints()
        if progress:
            progress(f"Warping atlas queries via {warp_method.upper()} and projecting with Parry…")
        # Selection preserves caller order and duplicates; no UV or topology
        # changes are permitted while moving these atlas-indexed query points.
        selected = vtk_to_numpy(atlasMesh.GetPoints().GetData())[keep_idx].astype(np.float32)
        points = vtkPoints()
        points.SetData(numpy_to_vtk(selected, deep=True))
        queries = vtkPolyData()
        queries.SetPoints(points)
        output = []
        for i in range(LMs.GetNumberOfBlocks()):
            subject = LMs.GetBlock(i).GetPoints()
            if subject is None or subject.GetNumberOfPoints() != n_landmarks:
                raise ValueError(f"Landmark count mismatch for specimen {i + 1}: expected {n_landmarks}.")
            if warp_method == "biharmonic":
                warped = logic.biharmonicWarpPolyData(
                    atlasMesh, atlasLM, subject, allow_fallback=allow_fallback
                )
                query_points = vtk_to_numpy(warped.GetPoints().GetData())[keep_idx].astype(np.float32)
            else:
                warped = logic._warp_polydata_tps(queries, atlasLM, subject)
                query_points = vtk_to_numpy(warped.GetPoints().GetData())
            index = (mesh_locators[i] if mesh_locators is not None
                     else logic._build_surface_locator(meshes.GetBlock(i)))
            projected = Spatial.project(index, query_points)
            output.append(projected.points.astype(np.float32))
        return output

    @staticmethod
    def install(module) -> None:
        """Replace methods only on this host's private class, never global VTK."""
        logic = module.MorphoWeaveAtlasBuilderLogic
        logic._build_surface_locator = MorphoWeaveSpatial.index
        logic._coord_check_ratio = MorphoWeaveSpatial.coordinate_ratio
        logic.sparseCorrespondenceBaseMesh = MorphoWeaveSpatial.correspondence
