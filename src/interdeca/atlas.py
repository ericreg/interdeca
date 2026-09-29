"""Landmark alignment and mean-atlas construction without application state."""

from collections.abc import Sequence

import numpy as np

from interdeca.dataset import Dataset
from interdeca.geometry import Geometry
from interdeca.models import (
    AlignmentResult,
    AtlasModel,
    AtlasReference,
    Landmarks,
    Mesh,
    ResampledSpecimen,
    Specimen,
)


class Atlas:
    """Namespace for constructing and aligning a common anatomical reference."""

    @staticmethod
    def compute_reference(
        specimens: Sequence[Specimen],
        *,
        allow_scaling: bool = False,
        max_iterations: int = 100,
        tolerance: float = 1e-7,
    ) -> AtlasReference:
        """Choose the specimen nearest a generalized Procrustes landmark mean.

        Every iteration aligns the original landmark configurations to the
        current mean, then updates that mean. Reflections are forbidden. Rigid
        alignment retains size differences by default. With allow_scaling, the
        mean's centroid size is fixed to the population's average centroid size
        so repeated least-squares fits cannot collapse the mean toward zero.

        Ties select the first specimen in input order. The result reports both
        iteration count and convergence; failure to converge is an error rather
        than a silently accepted partial atlas.
        """
        # Matching labels make rows comparable; valid stopping rules keep the
        # iterative alignment from accepting arbitrary or unfinished results.
        specimens = Dataset.validate_specimens(specimens)
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive.")
        # Remove position differences before estimating shape, and choose a fixed
        # population size to prevent repeated scaling from shrinking the mean.
        configurations = np.stack([specimen.landmarks.points for specimen in specimens])
        centered = configurations - configurations.mean(axis=1, keepdims=True)
        target_size = float(np.linalg.norm(centered, axis=(1, 2)).mean())
        mean = centered[0].copy()
        if allow_scaling:
            mean *= target_size / np.linalg.norm(mean)
        # Always align the original coordinates so rounding errors from earlier
        # iterations do not accumulate in the specimen configurations.
        converged = False
        for iteration in range(1, max_iterations + 1):
            transforms = tuple(
                Geometry._fit_transform(points, mean, allow_scaling=allow_scaling)
                for points in configurations
            )
            aligned = np.stack(
                [
                    transform.apply(points)
                    for transform, points in zip(
                        transforms, configurations, strict=True
                    )
                ]
            )
            # Update the reference from all aligned specimens with equal weight;
            # recentering prevents small numerical shifts from moving the origin.
            updated = aligned.mean(axis=0)
            updated -= updated.mean(axis=0)
            # Restore the fixed population size only when size differences are
            # intentionally being removed from the analysis.
            if allow_scaling:
                magnitude = np.linalg.norm(updated)
                if magnitude <= np.finfo(float).eps * target_size:
                    raise ValueError(
                        "The mean landmark configuration collapsed during alignment."
                    )
                updated *= target_size / magnitude
            # Measure change relative to specimen size so the stopping decision
            # does not depend on whether coordinates use millimeters or meters.
            relative_change = np.linalg.norm(updated - mean) / target_size
            mean = updated
            if relative_change <= tolerance:
                converged = True
                break
        # Downstream stages must not mistake an unfinished alignment for a valid atlas.
        if not converged:
            raise RuntimeError(
                f"Generalized Procrustes alignment did not converge within {max_iterations} iterations."
            )
        # Refit once to ensure stored transforms correspond to the returned mean.
        transforms = tuple(
            Geometry._fit_transform(points, mean, allow_scaling=allow_scaling)
            for points in configurations
        )
        aligned = np.stack(
            [
                transform.apply(points)
                for transform, points in zip(transforms, configurations, strict=True)
            ]
        )
        # Use the specimen nearest the population mean to avoid choosing an
        # unusually shaped surface as the source of the atlas connectivity.
        residuals = np.sum((aligned - mean) ** 2, axis=(1, 2))
        reference_index = int(np.argmin(residuals))
        specimen, transform = specimens[reference_index], transforms[reference_index]
        # Native source UVs have no role in choosing a new atlas; Blender makes its UVs later.
        template = AtlasModel(
            Mesh(
                transform.apply(specimen.mesh.vertices), specimen.mesh.faces
            ).validated(),
            Landmarks(aligned[reference_index], specimen.landmarks.labels).validated(),
        )
        # Keep specimen order and its transforms together so later tasks can
        # identify exactly which population produced this reference.
        return AtlasReference(
            specimen.specimen_id,
            template,
            Landmarks(mean, specimen.landmarks.labels).validated(),
            tuple(item.specimen_id for item in specimens),
            transforms,
            allow_scaling,
            iteration,
            converged,
        )

    @staticmethod
    def prepare_contribution(
        specimen: Specimen,
        reference: AtlasReference,
        *,
        max_projection_distance: float | None = None,
    ) -> ResampledSpecimen:
        """Align and resample one specimen for independent mean-atlas accumulation.

        Each contribution receives the same template and common landmark target.
        Contributions can therefore be computed concurrently by a future manager.
        """
        # A contribution must come from the population that defined this reference;
        # otherwise the requested mean would mix two different input populations.
        if specimen.specimen_id not in reference.specimen_ids:
            raise ValueError(
                "An atlas contribution must belong to the reference population."
            )
        # Remove overall pose before looking for local surface correspondences,
        # using the same scaling choice that produced the shared reference.
        aligned = Atlas.align_specimen(
            specimen, reference.mean_landmarks, allow_scaling=reference.allow_scaling
        )
        return Geometry.resample_specimen(
            aligned.specimen,
            reference.template,
            mean_landmarks=reference.mean_landmarks,
            max_projection_distance=max_projection_distance,
        )

    @staticmethod
    def assemble_mean(contributions: Sequence[ResampledSpecimen]) -> AtlasModel:
        """Average corresponding vertices and landmarks with equal specimen weight.

        Different connectivity or UV maps are rejected because coordinate-wise
        averaging is meaningful only for homologous points. Contributions should
        all come from the same AtlasReference; caller-selected subsets are allowed.
        """
        # Count each specimen once so duplicate inputs cannot bias the mean.
        contributions = tuple(contributions)
        if not contributions:
            raise ValueError("At least one atlas contribution is required.")
        if len({item.specimen_id for item in contributions}) != len(contributions):
            raise ValueError("Duplicate specimens would bias the mean atlas.")
        # Accumulate corresponding coordinates without stacking every surface
        # into another large array in memory.
        first = contributions[0]
        vertex_sum = np.zeros_like(first.mesh.vertices)
        landmark_sum = np.zeros_like(first.landmarks.points)
        # Equal array positions must refer to the same anatomical locations;
        # matching shapes alone would not establish that correspondence.
        for contribution in contributions:
            if contribution.mesh.topology_key != first.mesh.topology_key:
                raise ValueError(
                    "Atlas contributions must share connectivity and UV indexing."
                )
            vertex_sum += contribution.mesh.vertices
            landmark_sum += contribution.landmarks.reordered(
                first.landmarks.labels
            ).points
        # Divide both surfaces and landmarks by the same count so the finished
        # atlas gives every contributing specimen equal influence.
        count = len(contributions)
        mesh = Mesh(vertex_sum / count, first.mesh.faces, first.mesh.uv).validated()
        return AtlasModel(
            mesh, Landmarks(landmark_sum / count, first.landmarks.labels).validated()
        )

    @staticmethod
    def align_specimen(
        specimen: Specimen, target: Landmarks, *, allow_scaling: bool = False
    ) -> AlignmentResult:
        """Fit landmarks, then apply the same proper transform to native geometry.

        Texture UVs remain unchanged because rigid/similarity transforms alter
        spatial coordinates without changing how a texture is attached.
        """
        # Fit corresponding landmark pairs rather than relying on input file order.
        landmarks = specimen.landmarks.reordered(target.labels)
        transform = Geometry._fit_transform(
            landmarks.points, target.points, allow_scaling=allow_scaling
        )
        # Move the surface and its landmarks together while leaving texture
        # attachment unchanged; baking needs their spatial relationship preserved.
        aligned = Specimen(
            specimen.specimen_id,
            Mesh(
                transform.apply(specimen.mesh.vertices),
                specimen.mesh.faces,
                specimen.mesh.uv,
            ).validated(),
            Landmarks(transform.apply(landmarks.points), target.labels).validated(),
            specimen.texture_path,
        ).validated()
        return AlignmentResult(aligned, transform)
