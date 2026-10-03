"""Small file/data host for MorphoWeave's unmodified Atlas Builder.

The supported upstream operations use model/markup handles, file I/O, and VTK.
There is no MRML scene, Qt application, or Slicer runtime here. Each load gets a
private import context; Python's global imports and module registry are untouched.
This is an integration boundary for a reviewed revision, not a Slicer emulator.
"""

import builtins
import importlib.util
import logging
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from types import ModuleType

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkPolyData

from interdeca._mesh_cache import MeshCache
from interdeca.io import LandmarkIO, MeshIO
from interdeca.models import Mesh
from interdeca.serde import LandmarksData, MeshData

logger = logging.getLogger(__name__)
REVISION = "6fd9e448fb3441b71f7fed2bec9434a9b2190d24"
SOURCE_SHA256 = "bed88b0b2f3759161bef3cde54ac2d75063a9897fbc09b47fca9602dea9e3f94"
SOURCE = (
    Path(__file__).parent
    / "_vendor/morphoweave/MorphoWeaveAtlasBuilder/MorphoWeaveAtlasBuilder.py"
)
CORNER_UV = "InterDeCA_CornerUV"


class DataNode:
    """Private mutable handle implementing only Atlas Builder's model/markup API."""

    def __init__(self):
        self.polydata = None
        self.landmarks = None

    def GetPolyData(self):
        return self.polydata

    def SetAndObservePolyData(self, polydata):
        # Upstream reuses VTK filters and scratch handles, so retain an owned copy.
        self.polydata = vtkPolyData()
        self.polydata.DeepCopy(polydata)

    def GetNumberOfControlPoints(self):
        return len(self.landmarks.points)

    def GetParentTransformNode(self):
        # Files and input records already use world RAS coordinates.
        return None

    def GetDisplayNode(self):
        return None

    def CreateDefaultDisplayNodes(self):
        """Display creation has no effect in a numerical host."""

    def SetHideFromEditors(self, hidden):
        """Editor visibility has no effect in a numerical host."""


class Host:
    """Implement the file/node calls made by runAlign, runMean, and atlas generation."""

    def __init__(self, labels: tuple[str, ...], cache: MeshCache | None = None):
        self.labels = labels
        self.cache = cache
        self.nodes: set[DataNode] = set()

    def AddNewNodeByClass(self, kind, name):
        # Unknown node types indicate a new upstream contract, not a silent fallback.
        if kind not in ("vtkMRMLModelNode", "vtkMRMLMarkupsFiducialNode"):
            raise RuntimeError(f"Unsupported MorphoWeave host node: {kind}")
        node = DataNode()
        self.nodes.add(node)
        return node

    def RemoveNode(self, node):
        self.nodes.remove(node)

    def loadModel(self, path):
        mesh = MeshIO.read(path) if self.cache is None else self.cache.read(path)
        polydata = MeshIO.to_polydata(mesh)
        # UVs belong to corners, not vertices. VTK carries this cell array through
        # the upstream alignment transform without merging texture seams.
        if mesh.uv is not None:
            coordinates = numpy_to_vtk(mesh.uv.reshape(-1, 6), deep=True)
            coordinates.SetName(CORNER_UV)
            polydata.GetCellData().AddArray(coordinates)
        node = self.AddNewNodeByClass("vtkMRMLModelNode", str(path))
        node.SetAndObservePolyData(polydata)
        return node

    def loadMarkups(self, path):
        node = self.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", str(path))
        node.landmarks = LandmarkIO.read(path).reordered(self.labels)
        return node

    @staticmethod
    def arrayFromMarkupsControlPoints(node):
        return node.landmarks.points.copy()

    def updateMarkupsControlPointsFromArray(self, node, points):
        # Shared input labels restore anatomical identity lost by upstream scratch nodes.
        node.landmarks = LandmarksData(points=points, labels=self.labels).to_internal()

    @staticmethod
    def mesh(polydata) -> Mesh:
        """Read a triangular VTK boundary without reordering geometry or UV corners."""
        if polydata.GetNumberOfPolys() != polydata.GetNumberOfCells():
            raise ValueError("MorphoWeave output contains non-surface cells.")
        offsets = vtk_to_numpy(polydata.GetPolys().GetOffsetsArray())
        if not np.all(np.diff(offsets) == 3):
            raise ValueError("MorphoWeave output contains non-triangular faces.")
        faces = vtk_to_numpy(polydata.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        uv = polydata.GetCellData().GetArray(CORNER_UV)
        return MeshData(
            vertices=vtk_to_numpy(polydata.GetPoints().GetData()),
            faces=faces,
            uv=None if uv is None else vtk_to_numpy(uv).reshape(-1, 3, 2),
        ).to_internal()

    def saveNode(self, node, path):
        # All staging files are private RAS OBJ/Markups files, never user scene nodes.
        if self.cache is not None:
            self.cache.invalidate(path)
        if node.polydata is not None:
            if Path(path).suffix != ".obj":
                raise ValueError("MorphoWeave host model outputs must be OBJ files.")
            MeshIO.write_obj(Host.mesh(node.polydata), path, overwrite=True)
        else:
            LandmarkIO.write_json(node.landmarks, path, overwrite=True)
        return True

    @staticmethod
    def NodeModify(node):
        # There are no scene observers or render notifications to batch.
        return nullcontext(node)

    @staticmethod
    def setRenderPaused(paused):
        """Rendering has no effect in a numerical host."""

    @staticmethod
    def processEvents():
        """Networking runs on its own thread; this host owns no GUI event loop."""

    @staticmethod
    def warningDisplay(message, **options):
        logger.warning("%s", message)


def unavailable(name):
    """Reject unsupported host APIs instead of pretending they succeeded."""
    raise RuntimeError(f"Unsupported Slicer/Qt API in MorphoWeave host: {name}")


def load_atlas_builder(host: Host) -> ModuleType:
    """Load the exact reviewed upstream file with isolated host-only imports."""
    # A missing submodule or changed upstream file must fail before numerical work.
    if not SOURCE.is_file():
        raise RuntimeError(
            "MorphoWeave sources are missing. Run git submodule update --init --recursive."
        )
    content = SOURCE.read_bytes()
    if sha256(content).hexdigest() != SOURCE_SHA256:
        raise RuntimeError(
            "MorphoWeave Atlas Builder differs from the reviewed revision; update and validate the adapter before changing its pin."
        )
    # Only the imported module sees these names. In particular, a real Slicer
    # process can import InterDeCA without its own slicer/qt modules being replaced.
    replacements = {
        name: ModuleType(name)
        for name in ("slicer", "qt", "ctk", "slicer.ScriptedLoadableModule")
    }
    for module in replacements.values():
        module.__getattr__ = unavailable
    replacements["slicer"].util = host
    replacements["slicer"].mrmlScene = host
    replacements["slicer"].app = host
    replacements["qt"].QApplication = host
    bases = replacements["slicer.ScriptedLoadableModule"]
    bases.__all__ = (
        "ScriptedLoadableModule",
        "ScriptedLoadableModuleWidget",
        "ScriptedLoadableModuleLogic",
        "ScriptedLoadableModuleTest",
    )
    for name in bases.__all__:
        setattr(bases, name, object)

    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in replacements:
            return replacements[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    # Compile the same bytes that were checked, retaining upstream filenames in tracebacks.
    spec = importlib.util.spec_from_file_location(
        "_interdeca_morphoweave_atlas", SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    module.__dict__["__builtins__"] = {**vars(builtins), "__import__": scoped_import}
    exec(compile(content, str(SOURCE), "exec"), module.__dict__)  # noqa: S102 -- pinned, hash-verified upstream source
    return module
