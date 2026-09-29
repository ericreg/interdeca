"""Surface correspondence and shared geometric measurements for color analysis."""

from itertools import combinations

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from vtkmodules.vtkCommonCore import reference
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

from interdeca.io import MeshIO
from interdeca.models import (
    AnalysisGeometry,
    AtlasModel,
    FloatArray,
    Landmarks,
    Mesh,
    ResampledSpecimen,
    SpatialTransform,
    Specimen,
)


class Geometry:
    """Stateless geometry operations on NumPy-backed surfaces and landmarks."""

    @staticmethod
    def prepare_analysis(
        atlas: AtlasModel, *, sample_count: int | None = None
    ) -> AnalysisGeometry:
        """Compute atlas face areas, adjacency, and a common face sampling scheme.

        Sampling uses deterministic farthest-point selection with shortest-path
        distances between adjacent face centroids. Each disconnected component
        receives a seed, so no component can silently disappear. With no sample
        count, every face is used. The shared atlas supplies area weights; these
        are deliberately independent of each specimen's anatomical size.

        Farthest-point selection runs one graph shortest-path search per chosen
        face. It favors clarity and reproducibility over maximum speed.
        """
        # Every specimen will sample this same image tile, so missing or tiled
        # atlas UVs would make later color comparisons ambiguous.
        mesh = atlas.mesh
        if mesh.uv is None or np.any((mesh.uv < 0) | (mesh.uv > 1)):
            raise ValueError(
                "Analysis requires atlas UVs within one [0, 1] texture tile."
            )
        # Surface area provides physical weights, while face centers provide
        # representative locations for measuring separation along the mesh.
        triangles = mesh.vertices[mesh.faces]
        areas = (
            np.linalg.norm(
                np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                ),
                axis=1,
            )
            / 2
        )
        centers = triangles.mean(axis=1)
        count = len(mesh.faces)
        # A sample budget must select real faces; repeated or nonexistent samples
        # would not provide additional surface coverage.
        requested = count if sample_count is None else sample_count
        if not isinstance(requested, int) or not 1 <= requested <= count:
            raise ValueError(f"sample_count must be an integer from 1 to {count}.")

        # Follow shared surface edges so nearby but disconnected pieces do not
        # exchange colors or appear connected during sampling.
        edge_faces: dict[tuple[int, int], list[int]] = {}
        neighbors: list[set[int]] = [set() for _ in range(count)]
        for face_index, face in enumerate(mesh.faces):
            for first, second in ((0, 1), (1, 2), (2, 0)):
                # Both orientations of an edge must identify the same shared boundary.
                edge = tuple(sorted((int(face[first]), int(face[second]))))
                edge_faces.setdefault(edge, []).append(face_index)
        for attached in edge_faces.values():
            # If several faces share an edge, include every connection rather than
            # silently dropping one branch of the surface.
            for first, second in combinations(attached, 2):
                neighbors[first].add(second)
                neighbors[second].add(first)
        # Sorting makes neighbor order reproducible rather than dependent on set order.
        adjacency = tuple(np.asarray(sorted(row), dtype=np.int64) for row in neighbors)
        if requested == count:
            # When every face is measured, no shortest-path searches are necessary.
            samples = np.arange(count, dtype=np.int64)
            nearest = samples.copy()
        else:
            # Weight adjacent-face connections by physical separation so sample
            # spacing reflects geometry rather than only the number of triangles.
            rows = np.repeat(np.arange(count), [len(row) for row in adjacency])
            columns = np.concatenate(adjacency)
            distances = np.linalg.norm(centers[rows] - centers[columns], axis=1)
            # A positive floor keeps coincident centroids connected in the sparse graph.
            length_scale = max(
                float(np.linalg.norm(np.ptp(mesh.vertices, axis=0))),
                np.finfo(float).tiny,
            )
            distances = np.maximum(distances, length_scale * 1e-12)
            graph = csr_matrix((distances, (rows, columns)), shape=(count, count))
            # Each disconnected piece needs its own seed because no path from
            # another piece can reach it.
            component_count, components = connected_components(graph, directed=False)
            if requested < component_count:
                raise ValueError(
                    f"Need at least {component_count} samples to cover every disconnected component."
                )
            seeds = []
            for component in range(component_count):
                members = np.flatnonzero(components == component)
                seeds.append(int(members[np.argmax(areas[members])]))
            # Keep only the best distance and owner for each face, avoiding a
            # full sample-by-face distance matrix.
            chosen: list[int] = []
            minimum = np.full(count, np.inf)
            nearest = np.full(count, -1, dtype=np.int64)
            for sample_index in range(requested):
                # After seeding every piece, add a sample where coverage is weakest.
                face = (
                    seeds[sample_index]
                    if sample_index < len(seeds)
                    else int(np.argmax(minimum))
                )
                chosen.append(face)
                # Update ownership only where the new sample is closer; strict
                # comparison also keeps the earlier sample when distances tie.
                distance = dijkstra(graph, directed=False, indices=face)
                closer = distance < minimum
                minimum[closer] = distance[closer]
                nearest[closer] = sample_index
                minimum[face] = 0
            samples = np.asarray(chosen, dtype=np.int64)
        # Each sampled color stands for its surrounding region, so carry that
        # region's total area into later cluster-area features.
        represented_areas = np.bincount(nearest, weights=areas, minlength=len(samples))
        return AnalysisGeometry(
            mesh, areas, centers, adjacency, samples, nearest, represented_areas
        ).validated()

    @staticmethod
    def resample_specimen(
        specimen: Specimen,
        atlas: AtlasModel,
        *,
        mean_landmarks: Landmarks | None = None,
        max_projection_distance: float | None = None,
    ) -> ResampledSpecimen:
        """Place atlas vertices onto an aligned specimen using landmark-guided warps.

        Both surfaces are warped to a common landmark configuration using a
        three-dimensional polyharmonic spline with radial basis r and an affine
        term. For each warped atlas vertex, the closest point on the warped
        specimen determines a triangle and barycentric weights. Those weights
        are applied to the *original* specimen triangle, giving a point exactly
        on its piecewise-linear surface without an approximate inverse warp.

        The output retains atlas faces and UVs. Landmarks retain the specimen's
        aligned positions. At least four noncoplanar landmarks are required.
        Distances and the optional rejection threshold use the common warped
        coordinate system, in the caller's length units.
        """
        # Warp both surfaces toward the same labeled configuration so proximity
        # reflects corresponding anatomy rather than their original pose or shape.
        labels = atlas.landmarks.labels
        subject_landmarks = specimen.landmarks.reordered(labels)
        common = (
            mean_landmarks.reordered(labels)
            if mean_landmarks is not None
            else Landmarks(
                (subject_landmarks.points + atlas.landmarks.points) / 2, labels
            ).validated()
        )
        # Reject meaningless distance limits before building the search structure.
        if max_projection_distance is not None and (
            not np.isfinite(max_projection_distance) or max_projection_distance < 0
        ):
            raise ValueError("max_projection_distance must be finite and nonnegative.")
        # A shared warped space makes it possible to locate a specimen surface
        # point for each atlas vertex without changing either input mesh.
        warped_subject = Geometry._warp(
            specimen.mesh.vertices, subject_landmarks, common
        )
        warped_atlas = Geometry._warp(atlas.mesh.vertices, atlas.landmarks, common)
        warped_mesh = Mesh(warped_subject, specimen.mesh.faces).validated()
        # Reuse one spatial index for all queries instead of searching every
        # specimen triangle separately for every atlas vertex.
        surface = MeshIO.to_polydata(warped_mesh)
        locator = vtkStaticCellLocator()
        locator.SetDataSet(surface)
        locator.BuildLocator()
        # Retain the closest triangle as well as its point: recovering coordinates
        # on the original specimen will need both.
        closest = np.empty_like(warped_atlas)
        cell_ids = np.empty(len(warped_atlas), dtype=np.int64)
        squared_distances = np.empty(len(warped_atlas))
        for index, point in enumerate(warped_atlas):
            # VTK writes its answers into mutable arguments; each query gets its
            # own containers so previous results cannot be overwritten.
            destination = [0.0, 0.0, 0.0]
            cell_id, sub_id, distance_squared = (
                reference(0),
                reference(0),
                reference(0.0),
            )
            locator.FindClosestPoint(
                point, destination, cell_id, sub_id, distance_squared
            )
            if int(cell_id) < 0:
                raise ValueError(
                    f"{specimen.specimen_id}: surface projection failed at vertex {index}."
                )
            closest[index], cell_ids[index], squared_distances[index] = (
                destination,
                int(cell_id),
                float(distance_squared),
            )
        # Report distances in ordinary length units and enforce the caller's
        # quality limit before accepting these correspondences.
        projection_distances = np.sqrt(np.maximum(squared_distances, 0))
        if max_projection_distance is not None and np.any(
            projection_distances > max_projection_distance
        ):
            raise ValueError(
                f"{specimen.specimen_id}: maximum projection distance {projection_distances.max():.6g} exceeds {max_projection_distance:.6g}."
            )

        # Express each closest point as a mixture of its triangle's three corners;
        # the same mixture identifies a point on the original, unwarped triangle.
        triangles = warped_subject[specimen.mesh.faces[cell_ids]]
        first, second = (
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        relative = closest - triangles[:, 0]
        dot00, dot01, dot11 = (
            np.sum(first * first, axis=1),
            np.sum(first * second, axis=1),
            np.sum(second * second, axis=1),
        )
        dot20, dot21 = (
            np.sum(relative * first, axis=1),
            np.sum(relative * second, axis=1),
        )
        # Very thin triangles make the weight calculation unstable, so reject
        # them before dividing by a value that is effectively zero.
        determinant = dot00 * dot11 - dot01 * dot01
        if np.any(determinant <= np.finfo(float).eps * dot00 * dot11):
            raise ValueError(
                f"{specimen.specimen_id}: warped triangles are too thin for stable barycentric projection."
            )
        weight1 = (dot11 * dot20 - dot01 * dot21) / determinant
        weight2 = (dot00 * dot21 - dot01 * dot20) / determinant
        # Remove small rounding excursions and make the weights sum to one so
        # the recovered point stays inside its original triangle.
        weights = np.clip(
            np.column_stack((1 - weight1 - weight2, weight1, weight2)), 0, 1
        )
        weights /= weights.sum(axis=1, keepdims=True)
        # Recover points directly on the native surface instead of assuming that
        # fitting a second spline would exactly invert the first warp.
        native_triangles = specimen.mesh.vertices[specimen.mesh.faces[cell_ids]]
        vertices = np.einsum("ni,nij->nj", weights, native_triangles)
        # Retain atlas indexing and UVs so all baked textures describe the same
        # face locations, even though specimen vertex positions differ.
        mesh = Mesh(vertices, atlas.mesh.faces, atlas.mesh.uv).validated()
        return ResampledSpecimen(
            specimen.specimen_id, mesh, subject_landmarks, projection_distances
        ).validated()

    @staticmethod
    def _warp(points: FloatArray, source: Landmarks, target: Landmarks) -> FloatArray:
        """Interpolate a 3-D r-basis spline after nondimensionalizing coordinates.

        SciPy names the kernel ``linear`` because it is -r; its default kernel
        ``thin_plate_spline`` is r² log(r), the familiar two-dimensional form.
        The sign difference in r is absorbed by the fitted coefficients when
        smoothing is zero. Degree one reproduces affine maps exactly.
        """
        # Corresponding labels and a normalized coordinate scale make the spline
        # equations describe anatomy without depending on the chosen length unit.
        target = target.reordered(source.labels)
        center = source.points.mean(axis=0)
        scale = float(np.linalg.norm(source.points - center))
        # The affine part of a 3-D spline needs independent directions in space;
        # coplanar or duplicate landmarks cannot determine a unique solution.
        if (
            scale == 0
            or len(source.points) < 4
            or np.linalg.matrix_rank(source.points - center) < 3
        ):
            raise ValueError(
                "3-D spline resampling requires at least four noncoplanar source landmarks."
            )
        if len(np.unique(source.points, axis=0)) != len(source.points):
            raise ValueError("Spline source landmarks must occupy distinct positions.")
        # Include an affine term so translations, rotations, and scaling can be
        # reproduced exactly rather than approximated by the radial terms.
        normalized = (source.points - center) / scale
        interpolator = RBFInterpolator(
            normalized, target.points, kernel="linear", degree=1, smoothing=0
        )
        # Chunk queries to bound the query-by-landmark distance matrix in memory.
        result = np.empty_like(points, dtype=np.float64)
        chunk_size = max(1, 1_000_000 // len(source.points))
        for start in range(0, len(points), chunk_size):
            result[start : start + chunk_size] = interpolator(
                (points[start : start + chunk_size] - center) / scale
            )
        return result

    @staticmethod
    def _fit_transform(
        source: FloatArray, target: FloatArray, *, allow_scaling: bool = False
    ) -> SpatialTransform:
        """Solve proper row-vector Procrustes alignment by singular values.

        For H = XᵀY = UΣVᵀ, the least-squares row-vector rotation is UVᵀ.
        A sign correction on the final singular direction excludes reflections.
        """
        # Each row must identify one corresponding point, and the points must
        # constrain rotation before a least-squares alignment is meaningful.
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("Alignment requires matching (N, 3) landmark matrices.")
        # Removing the centroids separates rotation and scale from translation.
        source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
        centered_source, centered_target = source - source_mean, target - target_mean
        if (
            np.linalg.matrix_rank(centered_source) < 2
            or np.linalg.matrix_rank(centered_target) < 2
        ):
            raise ValueError(
                "Alignment needs at least three noncollinear points in both configurations."
            )
        # Find the closest rotation, correcting a negative determinant so a
        # reflected specimen cannot be accepted as a rotated specimen.
        left, singular_values, right = np.linalg.svd(
            centered_source.T @ centered_target
        )
        correction = np.ones(3)
        correction[-1] = 1 if np.linalg.det(left @ right) >= 0 else -1
        rotation = (left * correction) @ right
        # Preserve anatomical size unless the caller explicitly requests scaling;
        # then restore the translation needed to align the original centroids.
        scale = (
            float(np.sum(singular_values * correction) / np.sum(centered_source**2))
            if allow_scaling
            else 1.0
        )
        translation = target_mean - scale * source_mean @ rotation
        return SpatialTransform(rotation, translation, scale).validated()
