"""InterDeCA's application-independent core operations and data contracts.

These operations can be called directly or through staged task pipelines.
Computational operations never depend on Slicer or an execution implementation.
"""

# Expose the public workflow from one import location so callers need not depend
# on the package's internal file layout.
from interdeca.atlas import Atlas
from interdeca.blender import Blender, BlenderError
from interdeca.color_analysis import ColorAnalysis
from interdeca.dataset import Dataset
from interdeca.execution import ExecutionManager, Pipeline, ResultRef, Task
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
    ExtractedRegion,
    FaceColors,
    Landmarks,
    Mesh,
    NormalizationStats,
    PopulationFeatures,
    PopulationModel,
    PopulationModelSnapshot,
    ProcessedColors,
    RegionSelection,
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
    "ExecutionManager",
    "ExtractedRegion",
    "FaceColors",
    "Geometry",
    "LandmarkIO",
    "Landmarks",
    "Mesh",
    "MeshIO",
    "NormalizationStats",
    "Pipeline",
    "Population",
    "PopulationFeatures",
    "PopulationModel",
    "PopulationModelSnapshot",
    "ProcessedColors",
    "RegionSelection",
    "ResampledSpecimen",
    "ResultRef",
    "SampledColors",
    "SpatialTransform",
    "Specimen",
    "SpecimenColorMoments",
    "Task",
    "TextureIO",
    "TexturePixels",
]
