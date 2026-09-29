"""Texture sampling, population normalization, and per-specimen color processing."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import AgglomerativeClustering, KMeans

from interdeca.io import TextureIO
from interdeca.models import (
    AnalysisGeometry,
    BakedTexture,
    ColorMoments,
    FloatArray,
    NormalizationStats,
    ProcessedColors,
    SampledColors,
    SpecimenColorMoments,
)


class ColorAnalysis:
    """Stateless operations on shared atlas samples and their texture colors."""

    @staticmethod
    def sample_specimen_colors(
        texture: BakedTexture,
        analysis: AnalysisGeometry,
        *,
        smoothing_steps: int = 0,
        minimum_alpha: float = 0.99,
    ) -> SampledColors:
        """Bilinearly sample each selected face at its UV centroid.

        Face centroids provide a reproducible point approximation, not an exact
        integral of texture color over a triangle. Optional smoothing performs
        area-weighted averaging over each face and its edge neighbors before
        subsampling. Repeated steps diffuse color farther through the mesh.
        Invalid faces neither contribute to smoothing nor become valid through it.

        UV v increases upward, while image row numbers increase downward.
        Alpha marks valid image coverage; RGB=(0,0,0) remains a valid black color.
        """
        # Pixel values are comparable only when the image uses this exact atlas
        # layout; matching image dimensions alone would not establish correspondence.
        if texture.topology_key != analysis.mesh.topology_key:
            raise ValueError(
                "Texture and analysis geometry use different atlas topology or UVs."
            )
        # Reject invalid smoothing and coverage settings before decoding a large image.
        if not isinstance(smoothing_steps, int) or smoothing_steps < 0:
            raise ValueError("smoothing_steps must be a nonnegative integer.")
        if not np.isfinite(minimum_alpha) or not 0 < minimum_alpha <= 1:
            raise ValueError("minimum_alpha must lie in (0, 1].")
        # Check the actual file against its metadata so a replaced or misidentified
        # image cannot be sampled using assumptions from a different artifact.
        rgb, alpha = TextureIO.read(texture.path)
        if (rgb.shape[1], rgb.shape[0]) != texture.resolution:
            raise ValueError(
                "Texture dimensions do not match the BakedTexture metadata."
            )
        uv = analysis.mesh.uv
        if uv is None:
            raise ValueError("Texture sampling requires atlas UVs.")
        # Sample a consistent location on every face and use coverage to mark
        # missing data; testing for black would discard real black surface colors.
        face_rgb, face_alpha = ColorAnalysis._sample_image(rgb, alpha, uv.mean(axis=1))
        valid = face_alpha >= minimum_alpha
        face_rgb[~valid] = 0
        # Read from the previous pass and write to a fresh array so smoothing is
        # independent of face traversal order.
        for _step in range(smoothing_steps):
            smoothed = np.zeros_like(face_rgb)
            for face_index, neighbors in enumerate(analysis.adjacency):
                if not valid[face_index]:
                    continue
                # Keep the face itself in the average and exclude missing neighbors
                # so absent texture coverage cannot pull colors toward black.
                neighborhood = np.concatenate(([face_index], neighbors))
                neighborhood = neighborhood[valid[neighborhood]]
                # Weight by surface area so many tiny triangles do not outweigh
                # a larger neighboring region merely because there are more of them.
                weights = analysis.face_areas[neighborhood]
                smoothed[face_index] = np.average(
                    face_rgb[neighborhood], axis=0, weights=weights
                )
            face_rgb = smoothed
        # Subsample after smoothing so even unselected neighboring faces can
        # contribute to a selected face's local color estimate.
        return SampledColors(
            texture.specimen_id,
            face_rgb[analysis.sample_faces],
            valid[analysis.sample_faces],
            analysis.sampling_key,
        ).validated()

    @staticmethod
    def compute_normalization_stats(
        samples: Sequence[SampledColors],
    ) -> NormalizationStats:
        """Pool L* and chroma moments over valid shared samples across specimens.

        Chroma is C*=sqrt(a*²+b*²). Each valid sampled face receives equal weight;
        this is a distribution of *sampled surface colors*, not all image pixels.
        Population variance includes both within-specimen and between-specimen
        variation. Transparent padding cannot bias the result.
        """
        # Shared sampling and unique IDs keep the pooled statistics comparable
        # and prevent a repeated specimen from receiving extra weight.
        samples = ColorAnalysis._validate_collection(samples)
        records: list[SpecimenColorMoments] = []
        # Accumulate moments rather than concatenate all colors, keeping memory
        # use independent of the number of specimens being combined.
        total = 0
        pooled_mean = np.zeros(2)
        pooled_squared_deviations = np.zeros(2)
        for specimen in samples:
            # Missing samples must not influence either the specimen's own
            # distribution or the population distribution used as its target.
            lab = rgb2lab(specimen.rgb[specimen.valid])
            moments = ColorAnalysis._moments(lab)
            records.append(SpecimenColorMoments(specimen.specimen_id, moments))
            # Merge within-specimen spread and differences between specimen means;
            # this avoids losing precision by subtracting nearly equal large sums.
            combined = total + moments.count
            difference = moments.mean - pooled_mean
            pooled_squared_deviations += (
                moments.std**2 * moments.count
                + difference**2 * total * moments.count / combined
            )
            pooled_mean += difference * moments.count / combined
            total = combined
        # Convert the accumulated squared deviations into a population standard
        # deviation, allowing only tiny negative rounding errors to collapse to zero.
        pooled = ColorMoments(
            total,
            pooled_mean,
            np.sqrt(np.maximum(pooled_squared_deviations / total, 0)),
        ).validated()
        return NormalizationStats(pooled, tuple(records), samples[0].sampling_key)

    @staticmethod
    def process_specimen_colors(
        samples: SampledColors,
        *,
        normalization: NormalizationStats | None = None,
        normalize_luminance: bool = True,
        normalize_chroma: bool = True,
        n_clusters: int | None = None,
        initial_clusters: int | None = None,
        random_state: int = 0,
    ) -> ProcessedColors:
        """Convert samples to Lab, optionally normalize L*/C*, and cluster colors.

        Normalization matches a specimen's mean and standard deviation to the
        pooled moments while retaining hue. A constant channel maps to the target
        mean; it cannot acquire variance that was absent from its input. Nearly
        neutral grays (C* <= 0.01) retain zero chroma because hue is unstable there.

        With n_clusters, deterministic-seeded K-means fits Lab colors. An optional
        larger initial_clusters value is consolidated by average-linkage centroid
        clustering. Final centroids are recomputed from assigned samples, so an
        initial cluster containing many samples receives its proper weight.
        No cluster identity matching occurs here; Population.align_clusters does it.
        """
        # An empty mask has no distribution to normalize or cluster. Lab then
        # gives separate brightness and color components for those operations.
        if not samples.valid.any():
            raise ValueError(f"{samples.specimen_id}: no valid color samples.")
        lab = rgb2lab(samples.rgb)
        # Normalization parameters belong to a particular population and sampling
        # scheme; applying unrelated parameters could silently change the analysis.
        if normalization is not None:
            if normalization.sampling_key != samples.sampling_key:
                raise ValueError(
                    "Normalization statistics use a different sampling scheme."
                )
            moments = next(
                (
                    item.moments
                    for item in normalization.per_specimen
                    if item.specimen_id == samples.specimen_id
                ),
                None,
            )
            if moments is None:
                raise ValueError(
                    "Specimen is absent from the normalization population."
                )
            # Check the summary statistics again so stale parameters are rejected
            # when the specimen's sampled color distribution has changed.
            current = ColorAnalysis._moments(lab[samples.valid])
            if (
                current.count != moments.count
                or not np.allclose(current.mean, moments.mean)
                or not np.allclose(current.std, moments.std)
            ):
                raise ValueError(
                    "Specimen colors changed after normalization statistics were computed."
                )
            lab = ColorAnalysis._normalize(
                lab,
                moments,
                normalization.pooled,
                normalize_luminance,
                normalize_chroma,
            )
        # Keep placeholders finite while preserving the separate validity mask;
        # placeholder values must never participate in clustering.
        lab[~samples.valid] = 0
        labels, centroids = None, None
        # Consolidation needs a meaningful final cluster count, and the requested
        # counts must fit the number of distinct colors actually present.
        if n_clusters is None and initial_clusters is not None:
            raise ValueError("initial_clusters requires a final n_clusters value.")
        if n_clusters is not None:
            initial = n_clusters if initial_clusters is None else initial_clusters
            if (
                not isinstance(n_clusters, int)
                or not isinstance(initial, int)
                or not 1 <= n_clusters <= initial
            ):
                raise ValueError(
                    "Cluster counts must be integers with 1 <= n_clusters <= initial_clusters."
                )
            colors = lab[samples.valid]
            if len(np.unique(colors, axis=0)) < initial:
                raise ValueError(
                    f"{samples.specimen_id}: fewer distinct sampled colors than requested initial clusters."
                )
            # Several initial fits reduce sensitivity to a poor starting layout;
            # a fixed seed makes repeated runs use the same initialization choices.
            fitted = KMeans(
                n_clusters=initial, n_init=10, random_state=random_state
            ).fit(colors)
            local_labels = fitted.labels_
            # A finer initial partition can be merged into the requested number
            # of broader color groups without rerunning K-means on all samples.
            if initial > n_clusters:
                if n_clusters == 1:
                    consolidation = np.zeros(initial, dtype=np.int64)
                else:
                    consolidation = AgglomerativeClustering(
                        n_clusters=n_clusters, linkage="average", metric="euclidean"
                    ).fit_predict(fitted.cluster_centers_)
                local_labels = consolidation[local_labels]
            # Recompute each final center from its samples rather than averaging
            # initial centers, which would overrepresent small initial clusters.
            counts = np.bincount(local_labels, minlength=n_clusters)
            if np.any(counts == 0):
                raise ValueError(
                    "Clustering produced an empty cluster; reduce the requested cluster count."
                )
            centroids = np.column_stack(
                [
                    np.bincount(
                        local_labels, weights=colors[:, channel], minlength=n_clusters
                    )
                    / counts
                    for channel in range(3)
                ]
            )
            # Preserve full sample order and reserve -1 for missing data so later
            # area calculations cannot confuse missing samples with cluster zero.
            labels = np.full(len(lab), -1, dtype=np.int64)
            labels[samples.valid] = local_labels
        # Keep the analytical Lab values intact while producing bounded RGB for
        # display; some normalized colors cannot be represented exactly in sRGB.
        display_rgb = np.clip(lab2rgb(lab), 0, 1)
        display_rgb[~samples.valid] = 0
        return ProcessedColors(
            samples.specimen_id,
            lab,
            display_rgb,
            samples.valid,
            samples.sampling_key,
            labels,
            centroids,
        ).validated()

    @staticmethod
    def average_textures(
        textures: Sequence[BakedTexture],
        output_path: Path | str,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Average matching atlas textures in linear light, ignoring transparent pixels.

        Images must share UV topology and resolution. Input alpha acts as a
        fractional contribution weight; output alpha is the maximum input alpha,
        preserving coverage wherever any specimen supplies valid image data.
        """
        # Each specimen should contribute once, and output must not overwrite
        # an input that another task may still be using.
        textures = tuple(textures)
        if not textures:
            raise ValueError("At least one texture is required.")
        if len({texture.specimen_id for texture in textures}) != len(textures):
            raise ValueError("Duplicate specimens would bias the average texture.")
        first = textures[0]
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".png":
            raise ValueError("Average texture output must use the .png extension.")
        if any(destination == Path(texture.path).resolve() for texture in textures):
            raise ValueError("Average output cannot replace an input texture.")
        # Accumulate one image at a time so the whole population need not be
        # resident in memory while forming the average.
        height, width = first.resolution[1], first.resolution[0]
        color_sum, weight_sum, coverage = (
            np.zeros((height, width, 3)),
            np.zeros((height, width)),
            np.zeros((height, width)),
        )
        # The same pixel must represent the same atlas location in every image;
        # otherwise an image average would blend unrelated parts of the surface.
        for texture in textures:
            if (
                texture.topology_key != first.topology_key
                or texture.resolution != first.resolution
            ):
                raise ValueError("Textures must share atlas UVs and image dimensions.")
            rgb, alpha = TextureIO.read(texture.path)
            if rgb.shape != color_sum.shape:
                raise ValueError("Texture dimensions disagree with metadata.")
            # Average light values rather than encoded sRGB numbers, weighting
            # coverage so transparent padding cannot darken the mean.
            color_sum += TextureIO.to_linear(rgb) * alpha[..., None]
            weight_sum += alpha
            coverage = np.maximum(coverage, alpha)
        # Leave uncovered pixels at zero without dividing by zero; coverage stays
        # separate so these pixels remain distinguishable from measured black.
        mean = np.divide(
            color_sum,
            weight_sum[..., None],
            out=np.zeros_like(color_sum),
            where=weight_sum[..., None] > 0,
        )
        return TextureIO.write_png(
            TextureIO.from_linear(mean), coverage, destination, overwrite=overwrite
        )

    @staticmethod
    def _sample_image(
        rgb: FloatArray, alpha: FloatArray, uv: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Bilinearly interpolate premultiplied color at Blender-style pixel centers."""
        height, width = alpha.shape
        # Align UVs with pixel centers and reverse the vertical direction to match
        # Blender's texture coordinates to the image's top-to-bottom row order.
        x = np.clip(uv[:, 0] * width - 0.5, 0, width - 1)
        y = np.clip((1 - uv[:, 1]) * height - 0.5, 0, height - 1)
        # Find the four surrounding pixels so colors vary smoothly as a sample
        # moves across the image rather than jumping to the nearest pixel.
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        fraction_x, fraction_y = x - x0, y - y0
        # Weight colors by alpha before interpolation to keep arbitrary colors
        # in transparent pixels from bleeding into valid surface samples.
        sampled_rgb, sampled_alpha = np.zeros((len(uv), 3)), np.zeros(len(uv))
        for ix, iy, weight in (
            (x0, y0, (1 - fraction_x) * (1 - fraction_y)),
            (x1, y0, fraction_x * (1 - fraction_y)),
            (x0, y1, (1 - fraction_x) * fraction_y),
            (x1, y1, fraction_x * fraction_y),
        ):
            contribution = weight * alpha[iy, ix]
            sampled_alpha += contribution
            sampled_rgb += rgb[iy, ix] * contribution[:, None]
        # Remove the alpha weighting only where coverage exists; samples with no
        # coverage keep a finite placeholder and a zero alpha value.
        sampled_rgb = np.divide(
            sampled_rgb,
            sampled_alpha[:, None],
            out=np.zeros_like(sampled_rgb),
            where=sampled_alpha[:, None] > 0,
        )
        return np.clip(sampled_rgb, 0, 1), sampled_alpha

    @staticmethod
    def _moments(lab: FloatArray) -> ColorMoments:
        """Measure luminance and chroma without inventing data for an empty image."""
        # An empty image has no meaningful moments, so do not invent fallback
        # values that could be mistaken for measured population statistics.
        if len(lab) == 0:
            raise ValueError(
                "Cannot compute normalization statistics without valid colors."
            )
        # Use color magnitude rather than separate a/b components so normalization
        # can adjust saturation while retaining the original hue direction.
        luminance_chroma = np.column_stack(
            (lab[:, 0], np.linalg.norm(lab[:, 1:], axis=1))
        )
        return ColorMoments(
            len(lab), luminance_chroma.mean(axis=0), luminance_chroma.std(axis=0)
        ).validated()

    @staticmethod
    def _normalize(
        lab: FloatArray,
        source: ColorMoments,
        target: ColorMoments,
        luminance: bool,
        chroma: bool,
    ) -> FloatArray:
        """Apply independent affine L*/C* transforms while preserving hue direction."""
        # Work on a copy so other tasks can continue using the original samples.
        result = lab.copy()
        original_chroma = np.linalg.norm(lab[:, 1:], axis=1)
        channels = np.column_stack((lab[:, 0], original_chroma))
        # A constant channel cannot supply variation; mapping it to the target
        # mean avoids amplifying numerical noise with a near-infinite gain.
        gain = np.divide(
            target.std, source.std, out=np.zeros(2), where=source.std > 1e-12
        )
        transformed = (channels - source.mean) * gain + target.mean
        # Respect independently selected adjustments, keeping brightness in its
        # defined range and chroma nonnegative so hue is not accidentally reversed.
        if luminance:
            result[:, 0] = np.clip(transformed[:, 0], 0, 100)
        if chroma:
            magnitude = np.maximum(transformed[:, 1], 0)
            # Nominal grays can gain tiny chroma through conversion rounding;
            # do not turn that unreliable hue into a strongly colored output.
            ratio = np.divide(
                magnitude,
                original_chroma,
                out=np.zeros_like(magnitude),
                where=original_chroma > 0.01,
            )
            result[:, 1:] *= ratio[:, None]
        return result

    @staticmethod
    def _validate_collection(
        samples: Sequence[SampledColors | ProcessedColors],
    ) -> tuple:
        """Require unique identities, compatible sample ordering, and usable colors."""
        # Stable, unique specimen identities prevent ambiguous lookups and
        # accidental duplicate weighting in downstream population operations.
        samples = tuple(samples)
        if not samples:
            raise ValueError("At least one specimen's colors are required.")
        if len({sample.specimen_id for sample in samples}) != len(samples):
            raise ValueError("Specimen IDs must be unique.")
        # Equal sample counts alone are insufficient: ordering must match too,
        # and each specimen needs actual observations rather than only placeholders.
        for sample in samples:
            if sample.sampling_key != samples[0].sampling_key or len(sample.rgb) != len(
                samples[0].rgb
            ):
                raise ValueError(
                    "All specimens must share an identical sample ordering."
                )
            if not sample.valid.any():
                raise ValueError(f"{sample.specimen_id}: no valid color samples.")
        return samples
