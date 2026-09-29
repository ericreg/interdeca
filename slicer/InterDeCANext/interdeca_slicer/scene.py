"""Convert between immutable core data and Slicer-owned scene objects."""

from hashlib import sha256
from typing import NamedTuple

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

import slicer
from interdeca.models import FaceColors, Mesh
from interdeca.serde import MeshData


class SceneMesh(NamedTuple):
    """Index maps connect a canonical snapshot to the source display attributes."""

    mesh: Mesh
    node_id: str
    point_ids: np.ndarray
    signature: str


class DisplayState(NamedTuple):
    """Host-only display state restored after a temporary selection overlay."""

    texture: object
    scalar: str | None
    location: int
    visible: bool
    color_node: str | None
    scalar_range: tuple[float, float]


class SceneBridge:
    """All methods execute on Slicer's UI thread; workers never receive this object."""

    index_name = "InterDeCA_CanonicalVertex"

    @staticmethod
    def capture_display(node):
        """Keep display choices separate from the mesh's numerical state."""
        display = node.GetDisplayNode()
        return DisplayState(
            display.GetTextureImageDataConnection(),
            display.GetActiveScalarName(),
            display.GetActiveAttributeLocation(),
            bool(display.GetScalarVisibility()),
            display.GetColorNodeID(),
            tuple(display.GetScalarRange()),
        )

    @staticmethod
    def restore_display(node, state):
        """Remove an overlay without losing the user's raw texture or processed colors."""
        display = node.GetDisplayNode()
        display.SetTextureImageDataConnection(state.texture)
        if state.scalar:
            display.SetActiveScalar(state.scalar, state.location)
        # PythonQt expects a string even when the original model had no color map.
        display.SetAndObserveColorNodeID(state.color_node or "")
        display.SetScalarRange(*state.scalar_range)
        display.SetScalarVisibility(state.visible)

    @staticmethod
    def snapshot(node) -> SceneMesh:
        """Copy canonical geometry and world coordinates out of a live scene model."""
        if node is None or node.GetPolyData() is None:
            raise ValueError("Select a surface model.")
        poly = node.GetPolyData()
        # Require triangles rather than silently changing cell identities during extraction.
        cells = vtk_to_numpy(poly.GetPolys().GetData())
        if (
            poly.GetNumberOfVerts()
            or poly.GetNumberOfLines()
            or poly.GetNumberOfStrips()
        ):
            raise ValueError(
                "Analysis requires a triangular surface without lines or strips."
            )
        if len(cells) % 4 or not np.all(cells.reshape(-1, 4)[:, 0] == 3):
            raise ValueError("Triangulate the selected model before analysis.")
        faces = cells.reshape(-1, 4)[:, 1:].copy()
        vertices = vtk_to_numpy(poly.GetPoints().GetData()).copy()
        tcoords = poly.GetPointData().GetTCoords()
        uv = None if tcoords is None else vtk_to_numpy(tcoords)[faces, :2].copy()
        indices = poly.GetPointData().GetArray(SceneBridge.index_name)
        point_ids = np.arange(len(vertices))
        if indices is not None:
            mapping = vtk_to_numpy(indices).astype(np.int64)
            unique, point_ids = np.unique(mapping, return_index=True)
            if not np.array_equal(unique, np.arange(len(unique))):
                raise ValueError("Canonical vertex identities were modified.")
            canonical = vertices[point_ids]
            if not np.allclose(vertices, canonical[mapping], rtol=0, atol=1e-7):
                raise ValueError(
                    "Texture-seam copies have diverged; restore consistent geometry before analysis."
                )
            vertices, faces = canonical, mapping[faces]
        if node.GetParentTransformNode() is not None:
            # Bake scene transforms into the snapshot so curves and surfaces share world RAS.
            transform = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
                node.GetParentTransformNode(), None, transform
            )
            vertices = np.asarray(
                [transform.TransformPoint(point) for point in vertices]
            )
        mesh = MeshData(vertices=vertices, faces=faces, uv=uv).to_internal()
        signature = sha256(
            mesh.vertices.tobytes() + mesh.topology_key.encode()
        ).hexdigest()
        return SceneMesh(mesh, node.GetID(), point_ids, signature)

    @staticmethod
    def polydata(mesh: Mesh):
        """Build display geometry while recording canonical identities across UV seams."""
        # A display vertex has one UV; duplicate corners only in this presentation copy.
        indices = (
            mesh.faces.ravel() if mesh.uv is not None else np.arange(len(mesh.vertices))
        )
        vertices = mesh.vertices[indices]
        faces = (
            np.arange(len(indices)).reshape(-1, 3)
            if mesh.uv is not None
            else mesh.faces
        )
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices), deep=True))
        poly.SetPoints(points)
        cells = vtk.vtkCellArray()
        cells.SetCells(
            len(faces),
            numpy_to_vtkIdTypeArray(
                np.column_stack((np.full(len(faces), 3), faces))
                .astype(np.int64)
                .ravel(),
                deep=True,
            ),
        )
        poly.SetPolys(cells)
        ids = numpy_to_vtk(indices.astype(np.int64), deep=True)
        ids.SetName(SceneBridge.index_name)
        poly.GetPointData().AddArray(ids)
        if mesh.uv is not None:
            coordinates = numpy_to_vtk(
                np.ascontiguousarray(mesh.uv.reshape(-1, 2)), deep=True
            )
            coordinates.SetName("TextureCoordinates")
            poly.GetPointData().SetTCoords(coordinates)
        return poly

    @staticmethod
    def create_model(mesh: Mesh, name: str):
        """Create a scene-owned surface with ordinary neutral display properties."""
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        node.SetAndObservePolyData(SceneBridge.polydata(mesh))
        node.CreateDefaultDisplayNodes()
        node.GetDisplayNode().SetColor(0.8, 0.8, 0.8)
        return node

    @staticmethod
    def curve_points(curve):
        """Copy markup positions into the same world RAS frame as mesh snapshots."""
        if curve is None:
            raise ValueError("Draw or select a closed curve.")
        points = []
        for index in range(curve.GetNumberOfControlPoints()):
            position = [0.0, 0.0, 0.0]
            curve.GetNthControlPointPositionWorld(index, position)
            points.append(position)
        return np.asarray(points, dtype=float).reshape(-1, 3)

    @staticmethod
    def vertex_colors(node, snapshot):
        """Use explicit source RGB arrays, excluding analytical and selection scalars."""
        data = node.GetPolyData().GetPointData()
        for name in ("RGB", "Colors", "RGBColors"):
            array = data.GetArray(name)
            if array is not None and array.GetNumberOfComponents() in (3, 4):
                values = vtk_to_numpy(array)[snapshot.point_ids, :3].astype(float)
                return values / 255 if values.max() > 1 else values
        return None

    @staticmethod
    def texture(node, path):
        """Display an existing image through the selected model UVs."""
        factory = vtk.vtkImageReader2Factory()
        reader = factory.CreateImageReader2(str(path))
        if reader is None:
            raise ValueError(f"Cannot display texture: {path}")
        reader.SetFileName(str(path))
        reader.Update()
        display = node.GetDisplayNode()
        display.SetScalarVisibility(False)
        display.SetTextureImageDataConnection(reader.GetOutputPort())
        display.SetVisibility(True)

    @staticmethod
    def colors(node, colors: FaceColors):
        """Apply face colors with a neutral shade for missing measurements."""
        poly = node.GetPolyData()
        if poly.GetNumberOfCells() != len(colors.rgb):
            raise ValueError("Face-color result no longer matches the displayed mesh.")
        rgb = colors.rgb.copy()
        rgb[~colors.valid] = (0.45, 0.45, 0.45)
        values = numpy_to_vtk(np.rint(rgb * 255).astype(np.uint8), deep=True)
        values.SetName("InterDeCA_Colors")
        poly.GetCellData().SetScalars(values)
        display = node.GetDisplayNode()
        display.SetTextureImageDataConnection(None)
        display.SetActiveScalar("InterDeCA_Colors", vtk.vtkAssignAttribute.CELL_DATA)
        display.SetScalarVisibility(True)
        display.SetVisibility(True)
        poly.Modified()

    @staticmethod
    def highlight(node, selection):
        """Overlay a canonical selection on its corresponding display vertices."""
        poly = node.GetPolyData()
        identities = poly.GetPointData().GetArray(SceneBridge.index_name)
        mapping = (
            np.arange(poly.GetNumberOfPoints())
            if identities is None
            else vtk_to_numpy(identities)
        )
        values = np.isin(mapping, selection.vertices).astype(np.uint8)
        array = numpy_to_vtk(values, deep=True)
        array.SetName("InterDeCA_Selection")
        poly.GetPointData().AddArray(array)
        display = node.GetDisplayNode()
        display.SetTextureImageDataConnection(None)
        display.SetActiveScalar(
            "InterDeCA_Selection", vtk.vtkAssignAttribute.POINT_DATA
        )
        display.SetScalarRange(0, 1)
        display.SetAndObserveColorNodeID(
            "vtkMRMLColorTableNodeFileColdToHotRainbow.txt"
        )
        display.SetScalarVisibility(True)

    @staticmethod
    def clear_highlight(node):
        """Remove the temporary selection array from a scene model."""
        if node is not None and node.GetPolyData() is not None:
            node.GetPolyData().GetPointData().RemoveArray("InterDeCA_Selection")
            node.GetDisplayNode().SetScalarVisibility(False)

    @staticmethod
    def export(source, snapshot: SceneMesh, extracted, name, appearance=None):
        """Create a region model with mapped source attributes and preserved UVs."""
        node = SceneBridge.create_model(extracted.mesh, name)
        original, target = source.GetPolyData(), node.GetPolyData()
        if source.GetParentTransformNode() is not None:
            # Exported positions are in world RAS, so normals and vectors must use it too.
            transform = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
                source.GetParentTransformNode(), None, transform
            )
            transformed = vtk.vtkTransformPolyDataFilter()
            transformed.SetInputData(original)
            transformed.SetTransform(transform)
            transformed.Update()
            original = transformed.GetOutput()
        display_ids = target.GetPointData().GetArray(SceneBridge.index_name)
        mapping = vtk_to_numpy(display_ids)
        # UV meshes duplicate corners for display. Copy each original corner's attributes,
        # since seam normals or colors can differ even when canonical positions agree.
        if extracted.mesh.uv is not None:
            source_corners = vtk_to_numpy(original.GetPolys().GetData()).reshape(-1, 4)[
                :, 1:
            ]
            point_indices = source_corners[extracted.original_faces].ravel()
        else:
            point_indices = snapshot.point_ids[extracted.original_vertices[mapping]]
        for source_data, destination, indices in (
            (original.GetPointData(), target.GetPointData(), point_indices),
            (original.GetCellData(), target.GetCellData(), extracted.original_faces),
        ):
            for index in range(source_data.GetNumberOfArrays()):
                array = source_data.GetArray(index)
                if (
                    array is None
                    or array.GetName()
                    in (SceneBridge.index_name, "InterDeCA_Selection")
                    or array is source_data.GetTCoords()
                ):
                    continue
                copy = numpy_to_vtk(
                    np.ascontiguousarray(vtk_to_numpy(array)[indices]), deep=True
                )
                copy.SetName(array.GetName())
                # Preserve geometric attribute roles, not just their array names.
                if array is source_data.GetNormals():
                    destination.SetNormals(copy)
                elif array is source_data.GetVectors():
                    destination.SetVectors(copy)
                else:
                    destination.AddArray(copy)
        node.GetDisplayNode().SetColor(source.GetDisplayNode().GetColor())
        node.GetDisplayNode().SetOpacity(source.GetDisplayNode().GetOpacity())
        connection = source.GetDisplayNode().GetTextureImageDataConnection()
        if connection is not None:
            node.GetDisplayNode().SetTextureImageDataConnection(connection)
        if appearance is not None:
            SceneBridge.restore_display(node, appearance)
        return node

    @staticmethod
    def focus(node=None):
        """Center the selected surface in a dedicated three-dimensional view."""
        if node is not None:
            node.SetDisplayVisibility(True)
        manager = slicer.app.layoutManager()
        if manager is not None:
            manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
            slicer.util.resetThreeDViews()


class PopulationPlot:
    """Keep one chart and moving point per session rather than accumulating nodes."""

    def __init__(self):
        self.nodes = []
        self.table = None
        self.chart = None

    def show(self, data):
        """Display stored population scores and prepare a separate moving-point series."""
        self.clear()
        self.table = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLTableNode", "InterDeCA population"
        )
        self.nodes.append(self.table)
        for name, values in (
            ("X", data.scores[:, data.x_axis]),
            ("Y", data.scores[:, data.y_axis]),
        ):
            array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
            array.SetName(name)
            self.table.AddColumn(array)
        labels = vtk.vtkStringArray()
        labels.SetName("Specimen")
        for name in data.specimen_ids:
            labels.InsertNextValue(name)
        self.table.AddColumn(labels)
        series = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLPlotSeriesNode", "Specimens"
        )
        series.SetAndObserveTableNodeID(self.table.GetID())
        series.SetXColumnName("X")
        series.SetYColumnName("Y")
        series.SetLabelColumnName("Specimen")
        series.SetPlotType(series.PlotTypeScatter)
        series.SetLineStyle(series.LineStyleNone)
        series.SetMarkerStyle(series.MarkerStyleCircle)
        self.chart = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLPlotChartNode", "InterDeCA population"
        )
        self.chart.AddAndObservePlotSeriesNodeID(series.GetID())
        self.chart.SetTitle(data.method.upper())
        self.chart.SetXAxisTitle(f"Component {data.x_axis + 1}")
        self.chart.SetYAxisTitle(f"Component {data.y_axis + 1}")
        self.nodes.extend((series, self.chart))
        # A second series represents the current morphospace coordinate.
        self.point_table = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLTableNode", "Morphospace position"
        )
        for name in ("X", "Y"):
            array = vtk.vtkDoubleArray()
            array.SetName(name)
            self.point_table.AddColumn(array)
        marker = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLPlotSeriesNode", "Current position"
        )
        marker.SetAndObserveTableNodeID(self.point_table.GetID())
        marker.SetXColumnName("X")
        marker.SetYColumnName("Y")
        marker.SetPlotType(marker.PlotTypeScatter)
        marker.SetLineStyle(marker.LineStyleNone)
        marker.SetMarkerSize(12)
        marker.SetColor(1, 0, 0)
        self.chart.AddAndObservePlotSeriesNodeID(marker.GetID())
        self.nodes.extend((self.point_table, marker))
        manager = slicer.app.layoutManager()
        if manager is not None:
            # A private layout keeps surface changes visible beside their score plot.
            identifier = 731
            xml = '<layout type="horizontal"><item><view class="vtkMRMLViewNode" singletontag="1"><property name="viewlabel" action="default">1</property></view></item><item><view class="vtkMRMLPlotViewNode" singletontag="InterDeCA"><property name="viewlabel" action="default">P</property></view></item></layout>'
            manager.layoutLogic().GetLayoutNode().AddLayoutDescription(identifier, xml)
            manager.setLayout(identifier)
            manager.plotWidget(0).mrmlPlotViewNode().SetPlotChartNodeID(
                self.chart.GetID()
            )

    def point(self, x, y):
        """Update the morphospace marker without rebuilding the population plot."""
        if self.table is not None:
            self.point_table.GetTable().SetNumberOfRows(1)
            self.point_table.SetCellText(0, 0, str(x))
            self.point_table.SetCellText(0, 1, str(y))
            self.point_table.Modified()

    def clear(self):
        """Remove only the chart nodes owned by this application session."""
        for node in self.nodes:
            if node.GetScene() is not None:
                slicer.mrmlScene.RemoveNode(node)
        self.nodes = []
        self.table = None
