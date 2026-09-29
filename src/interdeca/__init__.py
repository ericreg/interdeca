"""InterDeCA's application-independent core operations and data contracts.

These operations can be called directly. A future execution manager can wrap
them as tasks without requiring any operation to understand Slicer or a UI.
"""

# Expose the public workflow from one import location so callers need not depend
# on the package's internal file layout.
from interdeca.atlas import Atlas
from interdeca.blender import Blender, BlenderError
from interdeca.color_analysis import ColorAnalysis
from interdeca.dataset import Dataset
from interdeca.geometry import Geometry
from interdeca.io import LandmarkIO, MeshIO, TextureIO
from interdeca.models import (
    AlignmentResult,
    AnalysisGeometry,
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
from interdeca.population import Population

# Keep the public surface explicit so internal helpers and serialization details
# do not become accidental API commitments through wildcard imports.
__all__ = [
    "AlignmentResult",
    "AnalysisGeometry",
    "Atlas",
    "AtlasModel",
    "AtlasReference",
    "BakedTexture",
    "Blender",
    "BlenderError",
    "BlenderInstallation",
    "ClusterAlignment",
    "ColorAnalysis",
    "ColorMoments",
    "Dataset",
    "Geometry",
    "LandmarkIO",
    "Landmarks",
    "Mesh",
    "MeshIO",
    "NormalizationStats",
    "Population",
    "PopulationFeatures",
    "PopulationModel",
    "PopulationModelSnapshot",
    "ProcessedColors",
    "ResampledSpecimen",
    "SampledColors",
    "SpatialTransform",
    "Specimen",
    "SpecimenColorMoments",
    "TextureIO",
    "TexturePixels",
]
