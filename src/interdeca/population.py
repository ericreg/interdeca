"""Cross-specimen cluster matching, feature assembly, and statistical models."""

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.color import lab2rgb
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning

from interdeca.color_analysis import ColorAnalysis
from interdeca.models import (
    AnalysisGeometry,
    ClusterAlignment,
    PopulationFeatures,
    PopulationModel,
    ProcessedColors,
)

# Keep PCA and ICA available without installing the optional UMAP dependency.
try:
    from umap import UMAP
except ModuleNotFoundError as error:
    # Only absence of the optional package is optional; broken dependencies must surface.
    if error.name != "umap":
        raise
    UMAP = None


class Population:
    """Namespace for operations that need multiple specimens at once."""

    @staticmethod
    def align_clusters(
        specimens: Sequence[ProcessedColors], *, reference_id: str | None = None
    ) -> ClusterAlignment:
        """Match local Lab clusters to a reference using minimum-cost assignment.

        Cost is Euclidean distance in Lab (Delta E 76). The Hungarian algorithm
        finds a one-to-one assignment rather than independently matching each
        centroid to its nearest neighbor. The input's first specimen is the
        deterministic default reference; choosing a reference is an analytical
        decision and can affect results when colors are ambiguous.

        Matching changes labels and centroid order, not measured sample colors.
        The returned common palette gives each specimen equal weight.
        """
        # Use one explicit reference so a cluster index has a consistent meaning
        # across the population rather than the arbitrary meaning assigned by K-means.
        specimens = ColorAnalysis._validate_collection(specimens)
        reference_id = (
            specimens[0].specimen_id if reference_id is None else reference_id
        )
        reference = next(
            (
                specimen
                for specimen in specimens
                if specimen.specimen_id == reference_id
            ),
            None,
        )
        if reference is None:
            raise ValueError(
                f"Reference specimen {reference_id!r} is not in this population."
            )
        if reference.centroids is None:
            raise ValueError("Cluster alignment requires clustered specimen colors.")
        # One-to-one matching needs the same number of available clusters on both
        # sides; mismatches cannot be repaired by silently dropping a centroid.
        aligned = []
        for specimen in specimens:
            if (
                specimen.centroids is None
                or specimen.labels is None
                or specimen.centroids.shape != reference.centroids.shape
            ):
                raise ValueError(
                    "Every specimen must have the same number of nonempty clusters."
                )
            if specimen.specimen_id == reference_id:
                # Preserve the reference's identity even if two centroids happen to tie.
                rows = columns = np.arange(len(reference.centroids))
            else:
                # Solve the assignments together so two local clusters cannot
                # both claim the same nearest reference cluster.
                rows, columns = linear_sum_assignment(
                    cdist(reference.centroids, specimen.centroids, metric="euclidean")
                )
            # Apply the same relabeling to both samples and centroids so looking
            # up a sample's cluster still returns its associated color center.
            local_to_reference = np.empty(len(columns), dtype=np.int64)
            local_to_reference[columns] = rows
            labels = np.full(len(specimen.labels), -1, dtype=np.int64)
            labels[specimen.valid] = local_to_reference[specimen.labels[specimen.valid]]
            centroids = np.empty_like(specimen.centroids)
            centroids[rows] = specimen.centroids[columns]
            # Retain measured colors; matching establishes names rather than
            # replacing observations with a population-wide display palette.
            aligned.append(
                ProcessedColors(
                    specimen.specimen_id,
                    specimen.lab,
                    specimen.rgb,
                    specimen.valid,
                    specimen.sampling_key,
                    labels,
                    centroids,
                ).validated()
            )
        # Average aligned centers with equal specimen weight so a heavily sampled
        # specimen cannot dominate the shared display colors.
        palette = np.mean([specimen.centroids for specimen in aligned], axis=0)
        return ClusterAlignment(
            tuple(aligned), reference_id, palette, np.clip(lab2rgb(palette), 0, 1)
        ).validated()

    @staticmethod
    def assemble_features(
        specimens: Sequence[ProcessedColors] | ClusterAlignment,
        analysis: AnalysisGeometry,
        *,
        representation: Literal["colors", "cluster_areas"] = "colors",
        missing: Literal["error", "drop"] = "error",
        area_normalization: Literal["fraction", "l2", "none"] = "fraction",
    ) -> PopulationFeatures:
        """Build corresponding Lab features or matched-cluster surface-area features.

        Color rows flatten [L*,a*,b*] in shared sample-face order. Cluster-area
        rows sum the common atlas area represented by each cluster's samples;
        they require an explicit ClusterAlignment to avoid mixing local labels.
        Area fractions sum to one; l2 instead gives a Euclidean unit-length row.

        Missing samples normally raise. With missing='drop', the intersection
        of valid samples is used for every specimen, preserving column meanings.
        For area features this also removes those samples' represented regions
        from the area denominator. No imputation occurs.
        """
        # Require explicit feature and missing-data choices so misspelled settings
        # cannot silently select a different analysis.
        if representation not in ("colors", "cluster_areas") or missing not in (
            "error",
            "drop",
        ):
            raise ValueError("Unknown representation or missing-data policy.")
        if area_normalization not in ("fraction", "l2", "none"):
            raise ValueError("area_normalization must be fraction, l2, or none.")
        # Area in cluster zero is comparable between specimens only after their
        # independently fitted clusters have been matched to a common reference.
        if representation == "cluster_areas" and not isinstance(
            specimens, ClusterAlignment
        ):
            raise ValueError(
                "Cluster-area features require Population.align_clusters output."
            )
        # Verify that every feature row uses the supplied atlas sampling scheme
        # before using its face identities or represented-area weights.
        collection = (
            specimens.specimens
            if isinstance(specimens, ClusterAlignment)
            else specimens
        )
        collection = ColorAnalysis._validate_collection(collection)
        for specimen in collection:
            if specimen.sampling_key != analysis.sampling_key or len(
                specimen.lab
            ) != len(analysis.sample_faces):
                raise ValueError("Color samples do not match the analysis geometry.")
        # Use the same retained positions for every row; removing missing samples
        # independently would give the same column different anatomical meanings.
        common_valid = np.logical_and.reduce(
            [specimen.valid for specimen in collection]
        )
        if missing == "error" and not common_valid.all():
            raise ValueError(
                "Missing color samples exist; correct the textures or explicitly select missing='drop'."
            )
        if not common_valid.any():
            raise ValueError("No samples have valid colors in every specimen.")
        if representation == "colors":
            # Keep channels next to their face and name each column so later model
            # results can be traced back to a particular surface measurement.
            values = np.stack(
                [specimen.lab[common_valid].reshape(-1) for specimen in collection]
            )
            names = tuple(
                f"face_{int(face)}.{channel}"
                for face in analysis.sample_faces[common_valid]
                for channel in ("L", "a", "b")
            )
        else:
            # Count represented atlas area rather than sample count so unequal
            # triangle sizes do not bias estimates of a color region's extent.
            cluster_count = len(specimens.palette_lab)
            values = np.stack(
                [
                    np.bincount(
                        specimen.labels[common_valid],
                        weights=analysis.sample_areas[common_valid],
                        minlength=cluster_count,
                    )
                    for specimen in collection
                ]
            )
            # Fractions describe area shares, while unit-length rows support
            # comparisons based on direction; retain raw areas only when requested.
            if area_normalization == "fraction":
                values /= values.sum(axis=1, keepdims=True)
            elif area_normalization == "l2":
                values /= np.linalg.norm(values, axis=1, keepdims=True)
            names = tuple(
                f"cluster_{index}.area_{area_normalization}"
                for index in range(cluster_count)
            )
        # Carry row and column identities with the matrix so a model's output
        # remains attributable after the numerical arrays leave this operation.
        return PopulationFeatures(
            tuple(specimen.specimen_id for specimen in collection),
            values,
            names,
            representation,
        ).validated()

    @staticmethod
    def fit_model(
        features: PopulationFeatures,
        *,
        method: Literal["pca", "ica", "umap"] = "pca",
        n_components: int = 2,
        standardize: bool = False,
        random_state: int = 0,
        max_iterations: int = 1000,
        tolerance: float = 1e-4,
        n_neighbors: int | None = None,
        min_dist: float = 0.1,
    ) -> PopulationModel:
        """Fit PCA, FastICA, or optional UMAP to a specimen-by-feature matrix.

        Standardization, when requested, is fitted across specimens per feature.
        Its parameters are returned for consistent later transforms. PCA and ICA
        component counts cannot exceed the centered matrix's numerical rank;
        this also prevents undefined whitening and meaningless zero-variance axes.

        UMAP requires the 'umap' package extra and at least four specimens. Its
        random initialization avoids spectral initialization failures in small
        populations. A fixed seed and one UMAP worker favor reproducibility.
        """
        # Reject impossible model requests before fitting, including populations
        # too small to contain differences between specimens.
        if method not in ("pca", "ica", "umap"):
            raise ValueError("method must be pca, ica, or umap.")
        count, width = features.values.shape
        if count < 2:
            raise ValueError("Population modeling requires at least two specimens.")
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if (
            not isinstance(max_iterations, int)
            or max_iterations < 1
            or not np.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise ValueError("max_iterations and tolerance must be positive.")
        # Learn preprocessing from this population and retain its parameters for
        # future observations; constant columns must not cause division by zero.
        center = features.values.mean(axis=0) if standardize else np.zeros(width)
        scale = features.values.std(axis=0) if standardize else np.ones(width)
        scale = np.where(scale > 0, scale, 1)
        values = (features.values - center) / scale
        # Only independent directions of variation can support PCA or ICA axes;
        # requesting more would produce meaningless or unstable components.
        rank = int(np.linalg.matrix_rank(values - values.mean(axis=0)))
        if rank == 0:
            raise ValueError(
                "All specimens have identical features; there is no population variation to model."
            )
        if method in ("pca", "ica") and n_components > rank:
            raise ValueError(
                f"n_components={n_components} exceeds the centered feature rank {rank}."
            )
        if method == "pca":
            # A full decomposition avoids randomized approximation when estimating
            # the population's principal directions of variation.
            estimator = PCA(n_components=n_components, svd_solver="full")
        elif method == "ica":
            # Normalize component variance and fix initialization so ICA's search
            # uses consistent scaling and repeatable starting conditions.
            estimator = FastICA(
                n_components=n_components,
                whiten="unit-variance",
                random_state=random_state,
                max_iter=max_iterations,
                tol=tolerance,
            )
        else:
            # Validate neighborhood choices against the actual population size;
            # UMAP cannot find more distinct neighbors than available specimens.
            if UMAP is None:
                raise ImportError(
                    "UMAP support is optional. Install it with: uv sync --extra umap"
                )
            if count < 4:
                raise ValueError(
                    "UMAP requires at least four specimens in this core API."
                )
            neighbors = min(15, count - 1) if n_neighbors is None else n_neighbors
            if not isinstance(neighbors, int) or not 2 <= neighbors < count:
                raise ValueError(
                    "n_neighbors must be at least 2 and less than the specimen count."
                )
            if not np.isfinite(min_dist) or not 0 <= min_dist <= 1:
                raise ValueError(
                    "min_dist must lie in [0, 1] (UMAP spread is fixed to 1)."
                )
            # Random initialization avoids small-population spectral failures;
            # one worker and a fixed seed make repeated fitting more reproducible.
            estimator = UMAP(
                n_components=n_components,
                n_neighbors=neighbors,
                min_dist=min_dist,
                random_state=random_state,
                n_jobs=1,
                init="random",
            )
        # Scope warning handling to this fit so another caller's warning behavior
        # is unchanged, while still treating failed convergence as a failed result.
        with warnings.catch_warnings():
            # A nonconverged ICA fit is not silently presented as a completed model.
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                scores = estimator.fit_transform(values)
            except ConvergenceWarning as error:
                raise RuntimeError(
                    "Population model did not converge; reconsider its settings or input features."
                ) from error
        # Return the estimator together with its preprocessing and identities so
        # callers can interpret scores and transform later feature rows consistently.
        return PopulationModel(
            method,
            features.specimen_ids,
            features.feature_names,
            scores,
            center,
            scale,
            estimator,
        ).validated()
