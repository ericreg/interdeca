"""Data exchanged by core operations, independent of Slicer and task schedulers.

Positions use RAS axes (right, anterior, superior) and one consistent length
unit chosen by the caller. RGB values are sRGB numbers in [0, 1]; Lab values
use the CIE D65 white point. Mesh UVs are stored per triangle corner so a seam
can have two texture coordinates without duplicating a geometric vertex.

Internal records use typing.NamedTuple. Construction itself is deliberately
lightweight: call validated() when creating numerical records inside algorithms,
or use the Pydantic data models in serde.py when accepting external data.
Validation returns a new record with owned, read-only numerical buffers.
"""

from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, NamedTuple, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Consistent precision and index types keep numerical and file adapters from
# disagreeing about the representation of the same data.
type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
type CoordinateSystem = Literal["RAS", "LPS"]


class Arrays:
    """Internal helpers for owning, validating, and freezing numerical arrays."""

    @staticmethod
    def frozen(values: ArrayLike, dtype: Any = np.float64) -> NDArray:
        """Copy data into a contiguous read-only array with finite entries.

        Copying prevents a caller's subsequent edits from changing a result
        that another task might already be reading. Read-only flags discourage
        accidental mutation; they are not a security boundary.
        """
        # Check integer inputs before conversion so a fractional index cannot
        # be silently rounded into a reference to the wrong vertex or face.
        original = np.asarray(values)
        if np.issubdtype(np.dtype(dtype), np.integer):
            if not np.issubdtype(original.dtype, np.integer):
                raise ValueError(
                    "Index arrays must contain integers, not rounded floats."
                )
        # Own contiguous storage so external changes cannot alter this record,
        # and reject nonfinite values before they contaminate later calculations.
        array = np.array(values, dtype=dtype, order="C", copy=True)
        if not np.isfinite(array).all():
            raise ValueError("Arrays must not contain NaN or infinity.")
        # Make accidental in-place edits fail instead of changing data shared by tasks.
        array.setflags(write=False)
        return array

    @staticmethod
    def matrix(values: ArrayLike, columns: int, name: str) -> FloatArray:
        """Require a nonempty matrix with the specified number of columns."""
        # An explicit matrix shape prevents NumPy broadcasting from disguising
        # missing rows or the wrong number of coordinate/color channels.
        array = Arrays.frozen(values)
        if array.ndim != 2 or array.shape[1] != columns or len(array) == 0:
            raise ValueError(f"{name} must have shape (N, {columns}), with N > 0.")
        return array


class Mesh(NamedTuple):
    """A triangular surface; ``uv[f, c]`` belongs to corner c of face f.

    Triangle order and vertex order define correspondence across specimens.
    UV values may lie outside [0, 1] on a source mesh (Blender repeats its
    texture); an atlas used for analysis must occupy a single [0, 1] tile.
    """

    # Store shape, connectivity, and corner UVs together so a surface cannot
    # lose the indexing information needed to interpret its coordinates or texture.
    vertices: FloatArray
    faces: IntArray
    uv: FloatArray | None = None

    def validated(self) -> Self:
        """Reject invalid connectivity and geometrically degenerate triangles."""
        # Check connectivity before indexing vertices so bad face references
        # produce a clear validation error rather than an unrelated NumPy failure.
        vertices = Arrays.matrix(self.vertices, 3, "vertices")
        faces = Arrays.frozen(self.faces, np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError("faces must have shape (M, 3), with M > 0.")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("A face references a nonexistent vertex.")
        # Degenerate triangles cannot support stable area or barycentric
        # calculations, even when their vertex indices are otherwise valid.
        triangles = vertices[faces]
        edges = triangles[:, 1:] - triangles[:, :1]
        areas_twice = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
        # A relative threshold behaves consistently under changes of length unit.
        edge_products = np.linalg.norm(edges[:, 0], axis=1) * np.linalg.norm(
            edges[:, 1], axis=1
        )
        if np.any(areas_twice <= np.finfo(float).eps * 32 * edge_products):
            raise ValueError(
                "Mesh contains zero-area or numerically degenerate triangles."
            )
        # A UV value belongs to each face corner, which allows seams without
        # duplicating geometric vertices and changing correspondence.
        uv = None if self.uv is None else Arrays.frozen(self.uv)
        if uv is not None and uv.shape != (len(faces), 3, 2):
            raise ValueError("uv must have shape (number_of_faces, 3, 2).")
        return Mesh(vertices, faces, uv)

    @property
    def topology_key(self) -> str:
        """Fingerprint vertex indexing, face order, and UVs, excluding shape.

        Two deformed copies of the same atlas share this key. Including UVs
        prevents accidentally comparing textures baked using different maps.
        """
        # Hash indexing and texture layout, not positions, so deformed copies
        # remain compatible. Explicit byte order makes the key portable.
        digest = sha256(str(len(self.vertices)).encode("ascii"))
        digest.update(self.faces.astype("<i8", copy=False).tobytes())
        digest.update(
            b"no-uv" if self.uv is None else self.uv.astype("<f8", copy=False).tobytes()
        )
        return digest.hexdigest()


class RegionSelection(NamedTuple):
    """A selection in canonical mesh indexing; independent of display scalars."""

    topology_key: str
    vertices: IntArray
    faces: IntArray

    def validated(self) -> Self:
        """Require distinct nonnegative indices; mesh bounds are checked on application."""
        vertices = Arrays.frozen(self.vertices, np.int64)
        faces = Arrays.frozen(self.faces, np.int64)
        for indices in (vertices, faces):
            if (
                indices.ndim != 1
                or not len(indices)
                or np.any(indices < 0)
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError(
                    "Selection indices must be nonempty, unique, nonnegative vectors."
                )
        return RegionSelection(self.topology_key, vertices, faces)


class ExtractedRegion(NamedTuple):
    """Submesh and source indices needed to copy external scene attributes."""

    mesh: Mesh
    original_vertices: IntArray
    original_faces: IntArray

    def validated(self) -> Self:
        """Every new element must map to one original element."""
        selection = RegionSelection(
            self.mesh.topology_key, self.original_vertices, self.original_faces
        ).validated()
        if len(selection.vertices) != len(self.mesh.vertices) or len(
            selection.faces
        ) != len(self.mesh.faces):
            raise ValueError(
                "Extraction maps must match the new vertex and face counts."
            )
        return ExtractedRegion(self.mesh, selection.vertices, selection.faces)


class FaceColors(NamedTuple):
    """Display RGB with explicit coverage so missing values never masquerade as black."""

    rgb: FloatArray
    valid: BoolArray

    def validated(self) -> Self:
        """Keep bounded display values and their coverage mask in the same face order."""
        rgb = Arrays.matrix(self.rgb, 3, "rgb")
        valid = Arrays.frozen(self.valid, bool)
        if np.any((rgb < 0) | (rgb > 1)) or valid.shape != (len(rgb),):
            raise ValueError(
                "Face colors need bounded RGB and one coverage value per face."
            )
        return FaceColors(rgb, valid)


class Landmarks(NamedTuple):
    """Ordered, uniquely named homologous positions in RAS coordinates."""

    # Labels travel with positions because row order alone cannot identify anatomy.
    points: FloatArray
    labels: tuple[str, ...]

    def validated(self) -> Self:
        """Require one distinct, nonempty name for each finite position."""
        # Require a one-to-one label/point relationship so alignment never has
        # to guess which of several identically named points should be matched.
        points = Arrays.matrix(self.points, 3, "landmarks")
        labels = tuple(self.labels)
        if len(labels) != len(points) or len(set(labels)) != len(labels):
            raise ValueError(
                "Landmark labels must be unique and match the point count."
            )
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("Each landmark needs a nonempty string label.")
        return Landmarks(points, labels)

    def reordered(self, labels: tuple[str, ...]) -> Self:
        """Return the same points in a requested homologous landmark order."""
        # Reordering may change positions in the array but must not add, drop,
        # or substitute an anatomical landmark.
        if len(labels) != len(self.labels) or set(labels) != set(self.labels):
            raise ValueError("Specimens must contain exactly the same landmark labels.")
        # Look up each requested label once rather than repeatedly scanning all
        # labels, then copy coordinates into the requested homologous order.
        indices = {label: index for index, label in enumerate(self.labels)}
        return Landmarks(
            self.points[[indices[label] for label in labels]], labels
        ).validated()


class Specimen(NamedTuple):
    """A native specimen surface, landmarks, and optional single sRGB texture.

    A texture path is an external resource: do not modify that file while any
    core operation is using it. Texture coordinates remain on the native mesh
    during rigid alignment and are used as the source of Blender baking.
    """

    # Keep the specimen's identity attached to its native geometry and image
    # so separately executed stages can associate their results correctly.
    specimen_id: str
    mesh: Mesh
    landmarks: Landmarks
    texture_path: Path | None = None

    def validated(self) -> Self:
        """Normalize identifiers and filesystem paths without opening files."""
        # A usable identifier is required for matching independent stage outputs.
        if not isinstance(self.specimen_id, str) or not self.specimen_id.strip():
            raise ValueError("specimen_id must be a nonempty string.")
        # Resolve paths before a worker changes its working directory; this
        # normalizes the reference without reading or altering the image itself.
        texture_path = (
            None
            if self.texture_path is None
            else Path(self.texture_path).expanduser().resolve()
        )
        return Specimen(self.specimen_id, self.mesh, self.landmarks, texture_path)


class SpatialTransform(NamedTuple):
    """A proper similarity transform using row vectors: ``s * points @ R + t``."""

    # Keep pose and scale as separate terms so size normalization remains an
    # explicit choice rather than an unnoticed effect of the rotation matrix.
    rotation: FloatArray
    translation: FloatArray
    scale: float = 1.0

    def validated(self) -> Self:
        """Keep rotations orthogonal and forbid reflection or negative scale."""
        # Enforce matrix/vector dimensions before testing geometric properties,
        # avoiding unintended broadcasting in the orthogonality check.
        rotation = Arrays.frozen(self.rotation)
        translation = Arrays.frozen(self.translation)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(
                "Transform requires a (3, 3) rotation and (3,) translation."
            )
        # Orthogonality preserves angles and unit lengths; a positive determinant
        # and scale exclude mirror reflections and collapsed coordinate systems.
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            or np.linalg.det(rotation) < 0
        ):
            raise ValueError(
                "rotation must be an orthogonal, orientation-preserving matrix."
            )
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive.")
        return SpatialTransform(rotation, translation, self.scale)

    def apply(self, points: ArrayLike) -> FloatArray:
        """Transform an N-by-3 collection without changing the source array."""
        # Apply the row-vector convention used when fitting the transform, with
        # translation last so it is expressed in the destination coordinate frame.
        return (
            Arrays.matrix(points, 3, "points") @ self.rotation * self.scale
            + self.translation
        )


class AlignmentResult(NamedTuple):
    """An aligned native specimen and the transform from its original frame."""

    # Retaining the transform makes the aligned result traceable to its original frame.
    specimen: Specimen
    transform: SpatialTransform


class AtlasModel(NamedTuple):
    """One common mesh topology and its ordered anatomical landmarks."""

    # The surface and its anatomical anchors must move through the workflow together.
    mesh: Mesh
    landmarks: Landmarks


class AtlasReference(NamedTuple):
    """Generalized Procrustes result used to create independent contributions."""

    # Record which specimen supplies topology and which landmark shape supplies
    # the common target so independent contributions use the same reference.
    reference_id: str
    template: AtlasModel
    mean_landmarks: Landmarks
    # Positional pairing identifies the transform belonging to each specimen.
    specimen_ids: tuple[str, ...]
    transforms: tuple[SpatialTransform, ...]
    # Retain alignment choices and completion status for later interpretation.
    allow_scaling: bool
    iterations: int
    converged: bool


class ResampledSpecimen(NamedTuple):
    """A specimen on atlas connectivity, retaining its anatomical shape.

    Projection distances are measured in the common warped landmark space;
    they diagnose correspondence quality, not the final surface error.
    """

    # Keep correspondence diagnostics with the resampled surface so quality can
    # be assessed before its geometry is used for baking or atlas averaging.
    specimen_id: str
    mesh: Mesh
    landmarks: Landmarks
    projection_distances: FloatArray

    def validated(self) -> Self:
        """Require one finite nonnegative projection distance per atlas vertex."""
        # One distance per vertex makes every reported quality measurement
        # attributable to a specific correspondence on the atlas.
        distances = Arrays.frozen(self.projection_distances)
        if distances.shape != (len(self.mesh.vertices),) or np.any(distances < 0):
            raise ValueError(
                "projection_distances must contain one distance per vertex."
            )
        return ResampledSpecimen(self.specimen_id, self.mesh, self.landmarks, distances)


class AnalysisGeometry(NamedTuple):
    """Shared face measurements, adjacency, and deterministic sample coverage.

    ``nearest_samples[f]`` is an index into ``sample_faces``, not a face ID.
    ``sample_areas[k]`` is the total atlas area represented by sample k.
    """

    # Shared surface measurements keep spatial weighting identical across specimens.
    mesh: Mesh
    face_areas: FloatArray
    face_centers: FloatArray
    adjacency: tuple[IntArray, ...]
    # Store sample identities and their represented regions together so area
    # features can account for faces that were not sampled directly.
    sample_faces: IntArray
    nearest_samples: IntArray
    sample_areas: FloatArray

    def validated(self) -> Self:
        """Own buffers and reject inconsistent dimensions or out-of-range indices."""
        # Own all buffers before returning this shared record, since many
        # specimen tasks may read the same analysis geometry concurrently.
        areas, centers = (
            Arrays.frozen(self.face_areas),
            Arrays.frozen(self.face_centers),
        )
        adjacency = tuple(Arrays.frozen(row, np.int64) for row in self.adjacency)
        samples = Arrays.frozen(self.sample_faces, np.int64)
        nearest = Arrays.frozen(self.nearest_samples, np.int64)
        represented = Arrays.frozen(self.sample_areas)
        # Every face needs a center, a positive area, and a neighbor list for
        # spatial sampling and smoothing to refer to the same surface.
        count = len(self.mesh.faces)
        if (
            areas.shape != (count,)
            or centers.shape != (count, 3)
            or len(adjacency) != count
        ):
            raise ValueError(
                "Analysis measurements and adjacency must match the atlas face count."
            )
        if np.any(areas <= 0):
            raise ValueError("Analysis face areas must be positive.")
        # Reject graph edges pointing outside the atlas before shortest-path or
        # smoothing operations try to use them as array indices.
        for row in adjacency:
            if row.ndim != 1 or np.any((row < 0) | (row >= count)):
                raise ValueError(
                    "Adjacency must contain one-dimensional arrays of valid face indices."
                )
        # Samples must identify distinct faces, and every atlas face must be
        # assigned to one of those samples for its area to be represented.
        if (
            samples.ndim != 1
            or len(samples) == 0
            or len(np.unique(samples)) != len(samples)
            or np.any((samples < 0) | (samples >= count))
        ):
            raise ValueError("Sample faces must be distinct, valid atlas face indices.")
        if nearest.shape != (count,) or np.any(
            (nearest < 0) | (nearest >= len(samples))
        ):
            raise ValueError("Nearest-sample assignments must cover every atlas face.")
        # Recompute represented areas from their assignments to catch stale
        # weights that would otherwise bias population cluster-area features.
        expected = np.bincount(nearest, weights=areas, minlength=len(samples))
        if (
            represented.shape != (len(samples),)
            or np.any(represented <= 0)
            or not np.allclose(represented, expected)
        ):
            raise ValueError(
                "Represented areas must agree with the nearest-sample assignments."
            )
        return AnalysisGeometry(
            self.mesh, areas, centers, adjacency, samples, nearest, represented
        )

    @property
    def sampling_key(self) -> str:
        """Identify the atlas and exact sample ordering used by color features."""
        # Include sample order as well as topology so two equally sized sample
        # arrays cannot be mistaken for matching anatomical measurements.
        digest = sha256(self.mesh.topology_key.encode("ascii"))
        digest.update(self.sample_faces.astype("<i8", copy=False).tobytes())
        return digest.hexdigest()


class BlenderInstallation(NamedTuple):
    """A discovered executable and the version reported by that executable."""

    # Keep the verified version with its executable so later calls use the
    # installation that was actually inspected.
    executable: Path
    version: tuple[int, int, int]
    description: str


class BakedTexture(NamedTuple):
    """An sRGB PNG baked onto one specimen using a particular atlas UV map."""

    # Carry identity and layout metadata with the path so a readable image
    # cannot be mistaken for one baked for a different specimen or atlas.
    specimen_id: str
    path: Path
    topology_key: str
    resolution: tuple[int, int]


class SampledColors(NamedTuple):
    """sRGB samples in shared face order; invalid values are masked explicitly.

    Black is a legitimate color. Missing data is represented by ``valid``,
    never inferred from RGB being zero.
    """

    # A separate validity mask preserves true black colors while recording which
    # sample positions lack image coverage.
    specimen_id: str
    rgb: FloatArray
    valid: BoolArray
    sampling_key: str

    def validated(self) -> Self:
        """Require bounded colors and a matching validity mask."""
        # Keep one coverage flag per bounded color sample so later masking cannot
        # shift sample positions or treat display values as missing-data markers.
        rgb = Arrays.matrix(self.rgb, 3, "rgb")
        valid = Arrays.frozen(self.valid, np.bool_)
        if valid.shape != (len(rgb),) or np.any((rgb < 0) | (rgb > 1)):
            raise ValueError(
                "rgb must lie in [0, 1] and valid must have one entry per row."
            )
        return SampledColors(self.specimen_id, rgb, valid, self.sampling_key)


class ColorMoments(NamedTuple):
    """Population (ddof=0) mean and standard deviation of [L*, chroma]."""

    # Retain sample count so pooled moments weight each specimen by its actual
    # observations rather than treating small and large sample sets as identical.
    count: int
    mean: FloatArray
    std: FloatArray

    def validated(self) -> Self:
        """Validate the two statistics without adding epsilon to true variances."""
        # Keep the two L*/chroma channels explicit; a zero standard deviation is
        # a meaningful constant channel and must remain distinguishable from noise.
        mean, std = Arrays.frozen(self.mean), Arrays.frozen(self.std)
        if self.count < 1 or mean.shape != (2,) or std.shape != (2,) or np.any(std < 0):
            raise ValueError(
                "Color moments need positive count and two nonnegative standard deviations."
            )
        return ColorMoments(self.count, mean, std)


class SpecimenColorMoments(NamedTuple):
    """A specimen identifier and its normalization moments, with named fields."""

    # Named ownership avoids relying on collection order when normalizing one specimen.
    specimen_id: str
    moments: ColorMoments


class NormalizationStats(NamedTuple):
    """Pooled and per-specimen moments, all computed on one sampling scheme."""

    # Store source and target moments together with their sampling identity so
    # normalization can reject statistics from an unrelated set of observations.
    pooled: ColorMoments
    per_specimen: tuple[SpecimenColorMoments, ...]
    sampling_key: str


class ProcessedColors(NamedTuple):
    """Normalized Lab samples and optional local cluster labels/centroids.

    Cluster label -1 marks an invalid sample. Centroids are expressed in Lab;
    ``rgb`` is a clipped display conversion and can lose out-of-gamut detail.
    """

    # Keep analytical Lab colors alongside bounded display RGB so clipping for
    # visualization does not discard information needed for population features.
    specimen_id: str
    lab: FloatArray
    rgb: FloatArray
    valid: BoolArray
    sampling_key: str
    # Clustering is optional, but labels and their centers must be retained together.
    labels: IntArray | None = None
    centroids: FloatArray | None = None

    def validated(self) -> Self:
        """Freeze color buffers and require complete, consistent cluster data."""
        # All color representations must refer to the same ordered samples;
        # mismatched masks or display arrays would otherwise mislabel observations.
        lab = Arrays.matrix(self.lab, 3, "lab")
        rgb = Arrays.matrix(self.rgb, 3, "rgb")
        valid = Arrays.frozen(self.valid, np.bool_)
        if rgb.shape != lab.shape or valid.shape != (len(lab),):
            raise ValueError(
                "Lab, RGB, and validity arrays must have matching lengths."
            )
        # Partial cluster data cannot explain a label's meaning, so accept either
        # a complete clustering result or no clustering result at all.
        labels, centroids = None, None
        if (self.labels is None) != (self.centroids is None):
            raise ValueError("labels and centroids must be supplied together.")
        if self.labels is not None:
            labels = Arrays.frozen(self.labels, np.int64)
            centroids = Arrays.matrix(self.centroids, 3, "centroids")
            # Reserve -1 for missing observations and require valid labels to
            # reference real centers before later palette or area lookups.
            if labels.shape != (len(lab),) or np.any(labels[~valid] != -1):
                raise ValueError(
                    "Cluster labels must match samples; invalid samples use -1."
                )
            if np.any(labels[valid] < 0) or np.any(labels[valid] >= len(centroids)):
                raise ValueError("A cluster label is outside the centroid array.")
        return ProcessedColors(
            self.specimen_id, lab, rgb, valid, self.sampling_key, labels, centroids
        )


class ClusterAlignment(NamedTuple):
    """Clustered results relabeled against one reference, plus a mean palette."""

    # Record the shared reference and palette with the relabeled specimens so
    # callers can distinguish local cluster numbering from population numbering.
    specimens: tuple[ProcessedColors, ...]
    reference_id: str
    palette_lab: FloatArray
    palette_rgb: FloatArray

    def validated(self) -> Self:
        """Keep the common display palette independent of temporary arrays."""
        # Copy palette arrays because they may outlive the temporary arrays
        # used to compute averages in the alignment operation.
        return ClusterAlignment(
            self.specimens,
            self.reference_id,
            Arrays.matrix(self.palette_lab, 3, "palette_lab"),
            Arrays.matrix(self.palette_rgb, 3, "palette_rgb"),
        )


class PopulationFeatures(NamedTuple):
    """A specimen-by-feature matrix with explicit row and column identities."""

    # Names identify both axes, allowing model scores and feature loadings to
    # remain interpretable after the matrix leaves the assembly operation.
    specimen_ids: tuple[str, ...]
    values: FloatArray
    feature_names: tuple[str, ...]
    representation: Literal["colors", "cluster_areas"]

    def validated(self) -> Self:
        """Reject empty, nonfinite, or ambiguously labeled feature matrices."""
        # Reject empty matrices and inconsistent axis labels before fitting can
        # produce results with missing or misleading specimen attribution.
        values = Arrays.frozen(self.values)
        if values.ndim != 2 or 0 in values.shape:
            raise ValueError(
                "Population features must be a nonempty two-dimensional matrix."
            )
        if values.shape != (len(self.specimen_ids), len(self.feature_names)):
            raise ValueError(
                "Feature names and specimen IDs must describe the matrix shape."
            )
        if len(set(self.specimen_ids)) != len(self.specimen_ids):
            raise ValueError("Population specimen IDs must be unique.")
        return PopulationFeatures(
            tuple(self.specimen_ids),
            values,
            tuple(self.feature_names),
            self.representation,
        )


class PopulationModel(NamedTuple):
    """Fitted estimator, embedding, and training preprocessing parameters.

    New feature rows must be standardized with ``(X - center) / scale`` before
    calling estimator.transform. Estimators are mutable third-party objects;
    treat them as read-only after fitting. UMAP does not promise an exact inverse.
    """

    # Retain identities with fitted scores so each output row remains attributable.
    method: Literal["pca", "ica", "umap"]
    specimen_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    scores: FloatArray
    # New observations must use the training preprocessing, not newly estimated
    # parameters, before they are passed to the fitted estimator.
    center: FloatArray
    scale: FloatArray
    estimator: Any

    def validated(self) -> Self:
        """Freeze numerical results while retaining the fitted estimator."""
        # Freeze numerical results while preserving the live estimator object;
        # serializable snapshots are a separate, explicitly limited representation.
        return PopulationModel(
            self.method,
            tuple(self.specimen_ids),
            tuple(self.feature_names),
            Arrays.frozen(self.scores),
            Arrays.frozen(self.center),
            Arrays.frozen(self.scale),
            self.estimator,
        )


class TexturePixels(NamedTuple):
    """Decoded sRGB and alpha arrays; image rows run from top to bottom."""

    # Coverage remains separate so missing image regions do not masquerade as black.
    rgb: FloatArray
    alpha: FloatArray

    def validated(self) -> Self:
        """Require bounded RGB/alpha with matching, nonempty image dimensions."""
        # Require one alpha value for each RGB pixel before image interpolation
        # can safely combine color and coverage weights.
        rgb, alpha = Arrays.frozen(self.rgb), Arrays.frozen(self.alpha)
        if (
            rgb.ndim != 3
            or rgb.shape[-1] != 3
            or 0 in rgb.shape
            or alpha.shape != rgb.shape[:2]
        ):
            raise ValueError("Textures need RGB (height, width, 3) and matching alpha.")
        # Reject invalid ranges rather than silently clipping numerical problems
        # before the data is used for averaging or saved as an image.
        if np.any((rgb < 0) | (rgb > 1)) or np.any((alpha < 0) | (alpha > 1)):
            raise ValueError("Texture RGB and alpha must lie in [0, 1].")
        return TexturePixels(rgb, alpha)


class PopulationModelSnapshot(NamedTuple):
    """Serializable numerical model results, excluding a live fitted estimator.

    A snapshot can be stored or displayed but cannot transform new observations.
    It deliberately does not pretend to reconstruct a fitted third-party model.
    """

    # Store enough numerical context to display and interpret results without
    # claiming to reconstruct a live estimator from JSON data.
    method: Literal["pca", "ica", "umap"]
    specimen_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    scores: FloatArray
    center: FloatArray
    scale: FloatArray

    def validated(self) -> Self:
        """Freeze model outputs and check their specimen and feature dimensions."""
        # Check score ownership and preprocessing dimensions together so a saved
        # snapshot cannot pair an embedding with another feature set's parameters.
        scores = Arrays.frozen(self.scores)
        center, scale = Arrays.frozen(self.center), Arrays.frozen(self.scale)
        if (
            scores.ndim != 2
            or len(scores) != len(self.specimen_ids)
            or 0 in scores.shape
        ):
            raise ValueError("Model scores must have one nonempty row per specimen.")
        if (
            center.shape != (len(self.feature_names),)
            or scale.shape != center.shape
            or np.any(scale <= 0)
        ):
            raise ValueError(
                "Preprocessing needs one center and positive scale per feature."
            )
        return PopulationModelSnapshot(
            self.method, self.specimen_ids, self.feature_names, scores, center, scale
        )
