"""Closest points on triangle surfaces through an owned parallel Rust index."""

from typing import NamedTuple

from interdeca import _native
from interdeca.models import FloatArray, IntArray, Mesh
from interdeca.serde import SpatialMeshInput, SpatialQueryInput


class SurfaceProjection(NamedTuple):
    """One surface point, original triangle ID, and squared distance per query.

    Triangle IDs index the input mesh's unchanged faces and per-corner UVs.
    An edge or vertex may belong to several equally close triangles; the selected
    face can differ from VTK even when the projected position is the same.
    """

    points: FloatArray
    faces: IntArray
    squared_distances: FloatArray


class Spatial:
    """Stateless boundary operations; each returned index owns its mesh snapshot."""

    @staticmethod
    def index(mesh: Mesh) -> _native.SurfaceIndex:
        """Build a reusable float64 triangle BVH, preserving every input index.

        The index snapshots vertices and faces before releasing the GIL. It does
        not retain Python arrays, weld vertices, repair topology, or process UVs.
        Meshes must contain finite coordinates and valid triangles. Degenerate
        triangles are retained, as required by correspondence mesh indexing.
        """
        data = SpatialMeshInput(vertices=mesh.vertices, faces=mesh.faces)
        return _native.SurfaceIndex(data.vertices, data.faces)

    @staticmethod
    def project(index: _native.SurfaceIndex, points: FloatArray) -> SurfaceProjection:
        """Project onto the surface, including for points inside a closed mesh.

        Outputs are independent read-only float64/int64 arrays in query order.
        Empty (0, 3) batches return empty outputs. Failed or nonfinite projections
        raise ValueError; there is no nearest-vertex or VTK fallback. Returned
        squared distances are measured in the input coordinate system.
        """
        data = SpatialQueryInput(points=points)
        return SurfaceProjection(*index.project(data.points))

    @staticmethod
    def workers_for(query_count: int) -> int:
        """Return the maximum active query chunks selected by the worker policy.

        Default limit: min(8, available CPUs). INTERDECA_SPATIAL_THREADS overrides
        eight, still capped by available CPUs, and is read once per process.
        Use at least 8,192 queries per worker; batches below 16,384 stay serial.
        Empty batches use zero workers. A shared lazy pool bounds simultaneous
        queries across all indices. This policy does not change BVH construction
        or the thread settings of SciPy, VTK, or Blender.
        """
        if isinstance(query_count, bool) or not isinstance(query_count, int) or query_count < 0:
            raise ValueError("query_count must be a nonnegative integer.")
        return _native.spatial_worker_count(query_count)
