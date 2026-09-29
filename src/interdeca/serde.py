"""Pydantic validation and serialization at InterDeCA's data boundaries.

Core computations use NamedTuple records. These models describe their wire
representations: JSON objects have named fields, arrays become nested numeric
lists, and paths become strings. Decoding rebuilds validated, read-only NumPy
buffers before a record reaches an algorithm. Pydantic stays out of the
geometry/color computation code and out of Blender's Python installation.
"""

from pathlib import Path
from typing import Annotated, ClassVar, Generic, Literal, Self, TypeVar

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PlainSerializer,
    PlainValidator,
    PrivateAttr,
    StrictInt,
    StringConstraints,
    WithJsonSchema,
    model_validator,
)

from interdeca._blender_protocol import BlenderBakeRequest, BlenderUVRequest
from interdeca.models import (
    AlignmentResult,
    AnalysisGeometry,
    Arrays,
    AtlasModel,
    AtlasReference,
    BakedTexture,
    BlenderInstallation,
    ClusterAlignment,
    ColorMoments,
    Landmarks,
    Mesh,
    NormalizationStats,
    PopulationFeatures,
    PopulationModel,
    PopulationModelSnapshot,
    ProcessedColors,
    ResampledSpecimen,
    SampledColors,
    SpatialTransform,
    Specimen,
    SpecimenColorMoments,
    TexturePixels,
)


class ArraySerde:
    """Narrow array conversions shared by Pydantic field annotations."""

    @staticmethod
    def floating(value: object) -> np.ndarray:
        """Accept actual numbers, rejecting string-to-number coercion and NaN."""
        # Check the original value types before conversion so textual numbers
        # and truth values cannot silently become analytical measurements.
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise ValueError(
                "Expected a numeric array, not strings, booleans, or objects."
            )
        return Arrays.frozen(array)

    @staticmethod
    def integer(value: object) -> np.ndarray:
        """Keep index values integral; an empty JSON array needs an explicit dtype."""
        # Fractional values must not be rounded into different face or vertex
        # indices; empty lists are allowed because adjacency rows may be empty.
        array = np.asarray(value)
        if array.size and array.dtype.kind not in "iu":
            raise ValueError(
                "Index arrays require integers without rounding or coercion."
            )
        return Arrays.frozen(
            array.astype(np.int64) if not array.size else array, np.int64
        )

    @staticmethod
    def boolean(value: object) -> np.ndarray:
        """Accept boolean masks without interpreting arbitrary numbers as truth."""
        # Require actual boolean values so malformed coverage data such as 2 or
        # "false" cannot unexpectedly mark a sample as valid.
        array = np.asarray(value)
        if array.size and array.dtype.kind != "b":
            raise ValueError("Validity masks require JSON booleans or boolean arrays.")
        return Arrays.frozen(array, np.bool_)

    @staticmethod
    def nested_lists(value: np.ndarray) -> list:
        """Expose numeric values to Pydantic's JSON encoder without pickle."""
        # Plain nested numbers keep the wire representation independent of
        # NumPy's binary object layout and the receiver's Python installation.
        return value.tolist()


# Pair validation and JSON conversion in reusable annotations so every boundary
# applies the same numeric rules while computation receives only NumPy buffers.
FloatBuffer = Annotated[
    np.ndarray,
    PlainValidator(ArraySerde.floating),
    PlainSerializer(ArraySerde.nested_lists, return_type=list, when_used="json"),
    WithJsonSchema({"type": "array"}),
]
IntBuffer = Annotated[
    np.ndarray,
    PlainValidator(ArraySerde.integer),
    PlainSerializer(ArraySerde.nested_lists, return_type=list, when_used="json"),
    WithJsonSchema({"type": "array"}),
]
BoolBuffer = Annotated[
    np.ndarray,
    PlainValidator(ArraySerde.boolean),
    PlainSerializer(ArraySerde.nested_lists, return_type=list, when_used="json"),
    WithJsonSchema({"type": "array"}),
]
# Shared scalar constraints prevent identifiers and sizes from having different
# acceptance rules in otherwise related file and subprocess formats.
NonemptyString = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
PositiveInteger = Annotated[StrictInt, Field(gt=0)]
NonnegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
Record = TypeVar("Record", bound=tuple)


class BoundaryData(BaseModel):
    """Strict named-field schema for data owned by InterDeCA."""

    # Reject misspelled fields and implicit scalar coercions instead of silently
    # accepting a different meaning from the one the sender intended.
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
        allow_inf_nan=False,
        strict=True,
    )


class CoreData(BoundaryData, Generic[Record]):
    """Reusable conversion between a boundary model and its internal NamedTuple.

    A validated internal record is cached during model validation. That allows
    to_internal() to return the same checked buffers instead of repeatedly
    copying large meshes. Nested models perform their own checks first.
    """

    # Keep conversion metadata and the checked record outside the wire fields;
    # only the declared data values should appear in serialized output.
    record_type: ClassVar[type[tuple]]
    _internal: Record = PrivateAttr()

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        """Apply numerical invariants after Pydantic validates field types."""
        # Follow the NamedTuple's declared field order rather than JSON key order
        # so positional construction always assigns values to the right fields.
        values = tuple(
            CoreData._internal_value(getattr(self, name))
            for name in self.record_type._fields
        )
        # Type checks alone cannot establish geometric or array invariants;
        # apply the record's numerical checks before caching it for computation.
        record = self.record_type(*values)
        validator = getattr(record, "validated", None)
        self._internal = record if validator is None else validator()
        return self

    @staticmethod
    def _internal_value(value: object) -> object:
        """Convert nested boundary models without constructing record dictionaries."""
        # Unwrap nested models and collections recursively so Pydantic objects
        # do not leak into the geometry and color-analysis implementations.
        if isinstance(value, CoreData):
            return value.to_internal()
        if isinstance(value, tuple):
            return tuple(CoreData._internal_value(item) for item in value)
        return value

    @classmethod
    def from_internal(cls, record: Record) -> Self:
        """Validate named attributes of a core record for transport or storage."""
        # Read named attributes rather than tuple positions so the external model
        # validates field meanings instead of relying on incidental field order.
        return cls.model_validate(record, from_attributes=True)

    def to_internal(self) -> Record:
        """Return the validated NamedTuple; no Pydantic model enters computation."""
        # Reuse the checked buffers instead of copying a large mesh each time a
        # boundary consumer asks for the corresponding internal record.
        return self._internal


class MeshData(CoreData[Mesh]):
    """Serializable triangular mesh including optional per-corner UVs."""

    # Share mesh validation with the core so decoding cannot bypass connectivity
    # checks; optional UVs also allow storing atlases before their unwrap stage.
    record_type = Mesh
    vertices: FloatBuffer
    faces: IntBuffer
    uv: FloatBuffer | None = None


class LandmarksData(CoreData[Landmarks]):
    """Named RAS landmarks; anatomical labels retain exact case and spelling."""

    # Serialize labels alongside coordinates so correspondence survives a round trip.
    record_type = Landmarks
    points: FloatBuffer
    labels: tuple[NonemptyString, ...]


class SpecimenData(CoreData[Specimen]):
    """A complete native specimen record at an API or persistence boundary."""

    # Validate nested geometry and landmarks before handing a complete specimen
    # to the core; image paths stay references rather than embedded image payloads.
    record_type = Specimen
    specimen_id: NonemptyString
    mesh: MeshData
    landmarks: LandmarksData
    texture_path: Path | None = None


class SpatialTransformData(CoreData[SpatialTransform]):
    """Proper row-vector similarity transform."""

    # Preserve separate transform terms so decoding can reject reflection,
    # invalid rotation matrices, or a scale that collapses the specimen.
    record_type = SpatialTransform
    rotation: FloatBuffer
    translation: FloatBuffer
    scale: Annotated[FiniteFloat, Field(gt=0)] = 1.0


class AlignmentResultData(CoreData[AlignmentResult]):
    """Aligned native specimen and the transform applied to it."""

    # Keep the fitted transform with the aligned specimen for coordinate traceability.
    record_type = AlignmentResult
    specimen: SpecimenData
    transform: SpatialTransformData


class AtlasModelData(CoreData[AtlasModel]):
    """Common mesh and anatomical landmarks for an atlas."""

    # Decode surface and landmarks together to retain the atlas's anatomical anchors.
    record_type = AtlasModel
    mesh: MeshData
    landmarks: LandmarksData


class AtlasReferenceData(CoreData[AtlasReference]):
    """Serializable reference selection and population alignment results."""

    # Persist the reference population and alignment choices so later contribution
    # tasks can reproduce the intended frame rather than infer it from a mesh alone.
    record_type = AtlasReference
    reference_id: NonemptyString
    template: AtlasModelData
    mean_landmarks: LandmarksData
    specimen_ids: tuple[NonemptyString, ...]
    transforms: tuple[SpatialTransformData, ...]
    allow_scaling: bool
    iterations: PositiveInteger
    converged: bool

    @model_validator(mode="after")
    def check_population(self) -> Self:
        """Ensure reference identity and transforms describe the same population."""
        # Unique population membership prevents ambiguous ownership of transforms.
        if not self.specimen_ids or len(set(self.specimen_ids)) != len(
            self.specimen_ids
        ):
            raise ValueError("Reference specimen IDs must be nonempty and unique.")
        # Require a real reference member and one transform per member so an
        # incomplete serialized alignment cannot masquerade as a complete result.
        if self.reference_id not in self.specimen_ids or len(self.transforms) != len(
            self.specimen_ids
        ):
            raise ValueError(
                "Reference ID and transforms must match the reference population."
            )
        return self


class ResampledSpecimenData(CoreData[ResampledSpecimen]):
    """Corresponding surface and its projection diagnostics."""

    # Carry quality measurements with the surface so persistence does not strip
    # away the evidence needed to assess correspondence before baking.
    record_type = ResampledSpecimen
    specimen_id: NonemptyString
    mesh: MeshData
    landmarks: LandmarksData
    projection_distances: FloatBuffer


class AnalysisGeometryData(CoreData[AnalysisGeometry]):
    """Shared atlas measurements, graph adjacency, and sample coverage."""

    # Save the sampling assignments and their area weights together; loading one
    # without the other could silently change population feature values.
    record_type = AnalysisGeometry
    mesh: MeshData
    face_areas: FloatBuffer
    face_centers: FloatBuffer
    adjacency: tuple[IntBuffer, ...]
    sample_faces: IntBuffer
    nearest_samples: IntBuffer
    sample_areas: FloatBuffer


class BlenderInstallationData(CoreData[BlenderInstallation]):
    """Blender executable metadata; decoding does not execute or locate Blender."""

    # Treat executable information as data; loading it must not launch a process.
    record_type = BlenderInstallation
    executable: Path
    version: tuple[StrictInt, StrictInt, StrictInt]
    description: str


class BakedTextureData(CoreData[BakedTexture]):
    """Texture artifact metadata without embedding the image file itself."""

    # Persist layout and size with the image reference so consumers can detect
    # an incompatible artifact before interpreting its pixels as atlas samples.
    record_type = BakedTexture
    specimen_id: NonemptyString
    path: Path
    topology_key: NonemptyString
    resolution: tuple[PositiveInteger, PositiveInteger]


class TexturePixelsData(CoreData[TexturePixels]):
    """Decoded image boundary with explicit RGB and alpha channels."""

    # Validate color and coverage as one image so their pixel grids cannot diverge.
    record_type = TexturePixels
    rgb: FloatBuffer
    alpha: FloatBuffer


class SampledColorsData(CoreData[SampledColors]):
    """Surface color samples with a shared ordering and validity mask."""

    # The sampling identity and mask preserve which anatomical observations each
    # color represents, including explicit absence of data at some positions.
    record_type = SampledColors
    specimen_id: NonemptyString
    rgb: FloatBuffer
    valid: BoolBuffer
    sampling_key: NonemptyString


class ColorMomentsData(CoreData[ColorMoments]):
    """Luminance/chroma moments over valid samples."""

    # Sample count is part of the statistical result, not incidental metadata;
    # later pooling needs it to combine unequal sample sets correctly.
    record_type = ColorMoments
    count: PositiveInteger
    mean: FloatBuffer
    std: FloatBuffer


class SpecimenColorMomentsData(CoreData[SpecimenColorMoments]):
    """One named specimen's normalization moments."""

    # Named ownership makes moment lookup independent of serialized collection order.
    record_type = SpecimenColorMoments
    specimen_id: NonemptyString
    moments: ColorMomentsData


class NormalizationStatsData(CoreData[NormalizationStats]):
    """Population and per-specimen moments with explicit record fields."""

    # Keep population targets and specimen sources in one checked boundary object
    # so parameters from unrelated normalization runs are not accidentally mixed.
    record_type = NormalizationStats
    pooled: ColorMomentsData
    per_specimen: tuple[SpecimenColorMomentsData, ...]
    sampling_key: NonemptyString

    @model_validator(mode="after")
    def check_population(self) -> Self:
        """Prevent ambiguous moment lookup and incomplete population totals."""
        # Duplicate owners would make normalization lookup ambiguous, while a
        # mismatched pooled count signals missing or inconsistent specimen records.
        identifiers = tuple(item.specimen_id for item in self.per_specimen)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("Normalization specimen IDs must be nonempty and unique.")
        if sum(item.moments.count for item in self.per_specimen) != self.pooled.count:
            raise ValueError("Pooled count must equal the sum of per-specimen counts.")
        return self


class ProcessedColorsData(CoreData[ProcessedColors]):
    """Normalized colors and optional local clustering."""

    # Preserve analytical values separately from display colors, and let the
    # core enforce that optional cluster labels and centers form a complete pair.
    record_type = ProcessedColors
    specimen_id: NonemptyString
    lab: FloatBuffer
    rgb: FloatBuffer
    valid: BoolBuffer
    sampling_key: NonemptyString
    labels: IntBuffer | None = None
    centroids: FloatBuffer | None = None


class ClusterAlignmentData(CoreData[ClusterAlignment]):
    """Relabeled specimen clusters and a shared display palette."""

    # Store the matching reference and common palette with relabeled results so
    # their cluster numbers retain the same meaning after serialization.
    record_type = ClusterAlignment
    specimens: tuple[ProcessedColorsData, ...]
    reference_id: NonemptyString
    palette_lab: FloatBuffer
    palette_rgb: FloatBuffer


class PopulationFeaturesData(CoreData[PopulationFeatures]):
    """Feature matrix with explicit specimen and feature identities."""

    # Axis names and representation travel with the matrix so another process
    # can interpret both specimen rows and measured feature columns correctly.
    record_type = PopulationFeatures
    specimen_ids: tuple[NonemptyString, ...]
    values: FloatBuffer
    feature_names: tuple[NonemptyString, ...]
    representation: Literal["colors", "cluster_areas"]


class PopulationModelData(CoreData[PopulationModelSnapshot]):
    """Numerical model snapshot; a fitted estimator is not JSON data."""

    # Limit this format to numerical results and preprocessing parameters;
    # a live estimator would require a separate reconstruction contract.
    record_type = PopulationModelSnapshot
    method: Literal["pca", "ica", "umap"]
    specimen_ids: tuple[NonemptyString, ...]
    feature_names: tuple[NonemptyString, ...]
    scores: FloatBuffer
    center: FloatBuffer
    scale: FloatBuffer

    @classmethod
    def from_model(cls, model: PopulationModel) -> Self:
        """Export only named numerical outputs, never pickle executable estimator state."""
        # Read only the declared snapshot fields, leaving estimator state outside
        # the JSON artifact rather than implying it can be restored from these values.
        return cls.model_validate(model, from_attributes=True)


class BlenderUVRequestData(CoreData[BlenderUVRequest]):
    """Validated, versioned JSON request for the UV worker."""

    # Check operator ranges and protocol identity on the host so Blender receives
    # a complete request without needing the host's validation dependencies.
    record_type = BlenderUVRequest
    mesh: NonemptyString
    output: NonemptyString
    angle_limit_degrees: Annotated[FiniteFloat, Field(gt=0, le=89)]
    island_margin: Annotated[FiniteFloat, Field(ge=0, lt=1)]
    operation: Literal["uv"] = "uv"
    schema_version: Literal[1] = 1


class BlenderBakeRequestData(CoreData[BlenderBakeRequest]):
    """Validated, versioned JSON request for the texture worker."""

    # Validate image dimensions and ray settings before starting the expensive
    # subprocess; the worker consumes this same request as a lightweight record.
    record_type = BlenderBakeRequest
    source: NonemptyString
    target: NonemptyString
    texture: NonemptyString
    output: NonemptyString
    resolution: tuple[PositiveInteger, PositiveInteger]
    cage_extrusion: NonnegativeFloat
    max_ray_distance: NonnegativeFloat
    margin_pixels: Annotated[StrictInt, Field(ge=0)]
    operation: Literal["bake"] = "bake"
    schema_version: Literal[1] = 1


class MarkupControlPointData(BoundaryData):
    """The Slicer control-point fields used by InterDeCA.

    Slicer also stores display/orientation metadata. Ignoring those extra fields
    is an explicit interoperability choice restricted to this external format.
    """

    # Accept unrelated Slicer display metadata while preserving its external field
    # names, but require a defined position before a landmark can enter alignment.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    id: str = ""
    label: NonemptyString
    position: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    position_status: Literal["defined"] = Field(
        default="defined", alias="positionStatus"
    )


class MarkupSetData(BoundaryData):
    """A labeled point set and its on-disk coordinate convention."""

    # Retain coordinate metadata until the reader converts positions to RAS;
    # accepting both numeric and named conventions supports existing Slicer files.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    type: str = "Fiducial"
    coordinate_system: Literal["RAS", "LPS", "0", "1", 0, 1] = Field(
        default="RAS", alias="coordinateSystem"
    )
    control_points: tuple[MarkupControlPointData, ...] = Field(
        alias="controlPoints", min_length=1
    )


class MarkupsDocumentData(BoundaryData):
    """Pydantic reader/writer for a single Slicer Markups landmark set."""

    # Require exactly one landmark set so a reader never has to guess which
    # anatomy to use from an otherwise valid multi-set document.
    model_config = ConfigDict(extra="ignore")
    markups: tuple[
        MarkupSetData
    ]  # Exactly one set; ambiguous multi-set files are rejected.


class LandmarkRowData(BoundaryData):
    """Typed numeric/label columns at the CSV/FCSV parsing boundary."""

    # CSV represents every scalar as text, so numeric conversion is appropriate here.
    model_config = ConfigDict(extra="ignore", strict=False)
    # Labels and all three finite coordinates are required even though unrelated
    # CSV columns are ignored, preventing partial landmarks from entering the core.
    label: NonemptyString
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat
