"""File adapters for core data; no application scene or UI objects are involved."""

import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image
from vtkmodules.util.numpy_support import (
    numpy_to_vtk,
    numpy_to_vtkIdTypeArray,
    vtk_to_numpy,
)
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkTriangleFilter
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkIOPLY import vtkPLYReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

from interdeca.models import (
    CoordinateSystem,
    FloatArray,
    Landmarks,
    Mesh,
    TexturePixels,
)
from interdeca.serde import (
    LandmarkRowData,
    LandmarksData,
    MarkupControlPointData,
    MarkupsDocumentData,
    MarkupSetData,
    MeshData,
    TexturePixelsData,
)


class Files:
    """Small filesystem boundary shared by explicit output-producing tasks."""

    @staticmethod
    def write(path: Path | str, content: bytes, *, overwrite: bool = False) -> Path:
        """Write a caller-selected artifact, refusing replacement by default.

        Exclusive creation also prevents two tasks from silently choosing the
        same output path. Managers should assign a unique path to each task.
        """
        # Resolve the artifact location now so later working-directory changes
        # cannot redirect where this task writes its result.
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents two tasks from both passing an earlier
        # existence check and then overwriting the same output.
        with destination.open("wb" if overwrite else "xb") as stream:
            stream.write(content)
        return destination


class MeshIO:
    """Read triangular surfaces and preserve separate geometry/UV indexing."""

    @staticmethod
    def read(path: Path | str, *, coordinate_system: CoordinateSystem = "RAS") -> Mesh:
        """Load triangular OBJ, or triangulate PLY/STL/VTP using VTK.

        OBJ must already be triangulated; silently fan-triangulating a concave
        polygon would change its surface. Materials are not imported: supply a
        single texture explicitly to Dataset.inspect_specimen. UV seams in OBJ
        remain per-corner values instead of becoming duplicate mesh vertices.
        """
        # Establish a real input path and an explicit axis convention before
        # interpreting coordinates that later alignment will treat as comparable.
        source = Path(path).expanduser().resolve(strict=True)
        if coordinate_system not in ("RAS", "LPS"):
            raise ValueError("coordinate_system must be RAS or LPS.")
        # OBJ needs separate vertex and UV indexing to preserve seams; use the
        # format-specific VTK readers for the other supported surface formats.
        if source.suffix.lower() == ".obj":
            mesh = MeshIO._read_obj(source)
        else:
            readers = {
                ".ply": vtkPLYReader,
                ".stl": vtkSTLReader,
                ".vtp": vtkXMLPolyDataReader,
            }
            reader_type = readers.get(source.suffix.lower())
            if reader_type is None:
                raise ValueError(f"Unsupported mesh extension: {source.suffix}")
            reader = reader_type()
            reader.SetFileName(str(source))
            reader.Update()
            # The core's area and projection calculations require triangles;
            # lines and isolated point cells are not part of that surface contract.
            triangulator = vtkTriangleFilter()
            triangulator.SetInputData(reader.GetOutput())
            triangulator.PassLinesOff()
            triangulator.PassVertsOff()
            triangulator.Update()
            # A reader can report no surface without raising a Python exception;
            # reject that result before passing empty arrays into the algorithms.
            polydata = triangulator.GetOutput()
            if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfPolys() == 0:
                raise ValueError(f"No readable surface triangles in {source}")
            vertices = vtk_to_numpy(polydata.GetPoints().GetData())
            # TriangleFilter guarantees a legacy cell record is [3, i, j, k].
            faces = vtk_to_numpy(polydata.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
            # Expand any vertex UVs into corner values so every input format uses
            # the same texture representation once it enters the core.
            tcoords = polydata.GetPointData().GetTCoords()
            uv = None if tcoords is None else vtk_to_numpy(tcoords)[faces, :2]
            mesh = MeshData(vertices=vertices, faces=faces, uv=uv).to_internal()
        if coordinate_system == "LPS":
            # Flipping both x and y has positive determinant and preserves winding.
            mesh = MeshData(
                vertices=mesh.vertices * [-1, -1, 1], faces=mesh.faces, uv=mesh.uv
            ).to_internal()
        return mesh

    @staticmethod
    def _read_obj(path: Path) -> Mesh:
        """Parse vertex and UV indices independently, including negative indices."""
        # Keep geometry and texture indices separate because one geometric
        # vertex may have a different texture coordinate on each side of a seam.
        vertices, texture_vertices, faces, face_uv = [], [], [], []
        uv_presence: set[bool] = set()
        with path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                # Strip file comments before parsing so annotations cannot be
                # mistaken for extra coordinate values or face corners.
                fields = line.partition("#")[0].split()
                if not fields:
                    continue
                try:
                    if fields[0] == "v":
                        if len(fields) < 4:
                            raise ValueError("Incomplete vertex.")
                        vertices.append([float(value) for value in fields[1:4]])
                    elif fields[0] == "vt":
                        if len(fields) < 3:
                            raise ValueError("Incomplete texture coordinate.")
                        texture_vertices.append([float(value) for value in fields[1:3]])
                    elif fields[0] == "f":
                        # Reject polygons rather than inventing a triangulation
                        # that might change the surface of a concave face.
                        if len(fields) != 4:
                            raise ValueError(
                                "OBJ faces must be triangles; triangulate the source first."
                            )
                        # Resolve every corner against its own vertex and texture
                        # lists instead of assuming the two index sequences agree.
                        triangle, triangle_uv = [], []
                        for corner in fields[1:]:
                            indices = corner.split("/")
                            triangle.append(
                                MeshIO._obj_index(indices[0], len(vertices))
                            )
                            has_uv = len(indices) > 1 and bool(indices[1])
                            uv_presence.add(has_uv)
                            if has_uv:
                                triangle_uv.append(
                                    texture_vertices[
                                        MeshIO._obj_index(
                                            indices[1], len(texture_vertices)
                                        )
                                    ]
                                )
                        faces.append(triangle)
                        if triangle_uv:
                            face_uv.append(triangle_uv)
                except (ValueError, IndexError) as error:
                    # Include the source line so malformed data can be corrected
                    # without debugging the entire downstream mesh pipeline.
                    raise ValueError(f"{path}:{line_number}: {error}") from error
        # Partial UV coverage would leave some faces with undefined colors, so
        # require either complete corner coordinates or an entirely untextured mesh.
        if len(uv_presence) > 1:
            raise ValueError(
                f"{path}: texture coordinates are missing from some face corners."
            )
        # Structural parsing alone cannot detect invalid geometry; apply the
        # boundary model's numerical checks before exposing the mesh to callers.
        return MeshData(
            vertices=np.asarray(vertices),
            faces=np.asarray(faces, dtype=np.int64),
            uv=np.asarray(face_uv) if face_uv else None,
        ).to_internal()

    @staticmethod
    def _obj_index(value: str, count: int) -> int:
        """Translate OBJ's one-based or negative-relative index into zero-based."""
        # OBJ counts positive indices from one and negative indices backward
        # from the current list end; zero has no defined meaning in that format.
        index = int(value)
        resolved = index - 1 if index > 0 else count + index
        if index == 0 or not 0 <= resolved < count:
            raise ValueError(f"Invalid OBJ index {index} for {count} entries.")
        return resolved

    @staticmethod
    def write_obj(mesh: Mesh, path: Path | str, *, overwrite: bool = False) -> Path:
        """Save RAS geometry and corner UVs without changing vertex/face order.

        One vt record per face corner is intentional: a geometric vertex can
        belong to several UV islands. This file contains no material library.
        """
        # Check the complete record before encoding so an invalid internal mesh
        # cannot become an apparently valid artifact for another process.
        mesh = MeshData.from_internal(mesh).to_internal()
        stream = io.StringIO()
        stream.write(
            "# InterDeCA; RAS coordinates; triangle and vertex order preserved\n"
        )
        # Preserve enough digits to recover double-precision coordinates when
        # the OBJ is loaded again, without changing its vertex order.
        for vertex in mesh.vertices:
            stream.write(
                "v " + " ".join(format(value, ".17g") for value in vertex) + "\n"
            )
        # One texture coordinate per corner preserves seams without forcing
        # duplicate geometric vertices into the exported surface.
        if mesh.uv is not None:
            for uv in mesh.uv.reshape(-1, 2):
                stream.write(
                    "vt " + " ".join(format(value, ".17g") for value in uv) + "\n"
                )
        # Pair each geometric index with its exported corner UV index while
        # converting back to OBJ's one-based numbering.
        for face_index, face in enumerate(mesh.faces):
            corners = [str(int(index) + 1) for index in face]
            if mesh.uv is not None:
                corners = [
                    f"{vertex}/{face_index * 3 + corner + 1}"
                    for corner, vertex in enumerate(corners)
                ]
            stream.write("f " + " ".join(corners) + "\n")
        return Files.write(path, stream.getvalue().encode("utf-8"), overwrite=overwrite)

    @staticmethod
    def to_polydata(mesh: Mesh) -> vtkPolyData:
        """Create a private VTK surface for geometric queries (UVs are not needed)."""
        # Deep copies prevent VTK from mutating or outliving borrowed NumPy
        # storage that another task may still be reading.
        points = vtkPoints()
        points.SetData(numpy_to_vtk(mesh.vertices, deep=True))
        cells = vtkCellArray()
        # VTK's legacy cell buffer prefixes every triangle with its vertex count.
        records = np.column_stack(
            (np.full(len(mesh.faces), 3, dtype=np.int64), mesh.faces)
        ).ravel()
        cells.SetCells(len(mesh.faces), numpy_to_vtkIdTypeArray(records, deep=True))
        # Supply only the surface geometry needed by spatial queries; the
        # separate core mesh remains responsible for texture correspondence.
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)
        return polydata


class LandmarkIO:
    """Load labeled landmarks, honoring Slicer coordinate-system metadata."""

    @staticmethod
    def read(
        path: Path | str, *, coordinate_system: CoordinateSystem | None = None
    ) -> Landmarks:
        """Read Markups JSON, FCSV, or CSV with label,x,y,z columns.

        An explicit coordinate_system overrides file metadata. Without either,
        plain CSV uses RAS. FCSV metadata encodes RAS as 0 and LPS as 1.
        Only defined control points are accepted; missing landmarks are errors.
        """
        source = Path(path).expanduser().resolve(strict=True)
        # Retain the file's coordinate convention until all rows are parsed;
        # conversion to the shared RAS frame happens once at the end.
        system: str | int = "RAS"
        points, labels = [], []
        if source.suffix.lower() == ".json":
            # Validate the external schema first so missing or undefined control
            # points cannot silently shorten the landmark set.
            document = MarkupsDocumentData.model_validate_json(
                source.read_text(encoding="utf-8-sig")
            )
            markup = document.markups[0]
            system = markup.coordinate_system
            for point in markup.control_points:
                points.append(point.position)
                labels.append(point.label)
        elif source.suffix.lower() == ".fcsv":
            # FCSV puts column names and coordinate metadata in comment headers;
            # read those before assigning meanings to the data columns.
            lines = source.read_text(encoding="utf-8-sig").splitlines()
            columns = (
                "id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID".split(
                    ","
                )
            )
            for line in lines:
                if line.startswith("# CoordinateSystem"):
                    system = line.split("=", 1)[1].strip()
                elif line.startswith("# columns"):
                    columns = [
                        column.strip() for column in line.split("=", 1)[1].split(",")
                    ]
            # Exclude metadata lines and validate each row at the file boundary
            # before its text coordinates enter numerical calculations.
            rows = csv.DictReader(
                (line for line in lines if line.strip() and not line.startswith("#")),
                fieldnames=columns,
            )
            for row in rows:
                landmark = LandmarkRowData.model_validate(row)
                points.append((landmark.x, landmark.y, landmark.z))
                labels.append(landmark.label)
        elif source.suffix.lower() == ".csv":
            # Named columns make plain CSV independent of column ordering while
            # still requiring the labels needed for anatomical correspondence.
            with source.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    landmark = LandmarkRowData.model_validate(row)
                    points.append((landmark.x, landmark.y, landmark.z))
                    labels.append(landmark.label)
        else:
            raise ValueError("Landmarks must be Markups JSON, FCSV, or labeled CSV.")
        # Let an explicit caller convention override metadata, then normalize all
        # accepted file encodings before deciding whether axes need conversion.
        system = str(
            coordinate_system if coordinate_system is not None else system
        ).upper()
        if system not in ("RAS", "LPS", "0", "1"):
            raise ValueError(f"Unknown landmark coordinate system: {system}")
        # Convert coordinates before constructing the internal record so every
        # downstream operation can assume the same RAS convention.
        positions = np.asarray(points, dtype=float).reshape(-1, 3)
        if system in ("LPS", "1"):
            positions *= [-1, -1, 1]
        return LandmarksData(points=positions, labels=tuple(labels)).to_internal()

    @staticmethod
    def write_json(
        landmarks: Landmarks, path: Path | str, *, overwrite: bool = False
    ) -> Path:
        """Save a minimal Slicer Markups-compatible landmark set with RAS metadata."""
        # Validate label/point correspondence before encoding control points;
        # pairing mismatched sequences would misidentify anatomical locations.
        landmarks = LandmarksData.from_internal(landmarks).to_internal()
        control_points = tuple(
            MarkupControlPointData(id=str(index), label=label, position=tuple(point))
            for index, (label, point) in enumerate(
                zip(landmarks.labels, landmarks.points, strict=True)
            )
        )
        # Use the schema's external aliases so Slicer receives its expected field
        # names even though Python code accesses those fields in snake_case.
        data = MarkupsDocumentData(
            markups=(MarkupSetData(control_points=control_points),)
        )
        return Files.write(
            path,
            (data.model_dump_json(by_alias=True, indent=2) + "\n").encode("utf-8"),
            overwrite=overwrite,
        )


class TextureIO:
    """Read/write 8-bit sRGB images with alpha as an explicit validity signal."""

    @staticmethod
    def read(path: Path | str) -> TexturePixels:
        """Return RGB in [0, 1] and alpha in [0, 1], without treating black as missing.

        Images must already be encoded as sRGB; ICC conversion, HDR, tiled
        textures, and material graphs are outside this first core API.
        """
        # Restrict conversion to supported 8-bit inputs so higher-precision or
        # differently encoded images are not silently quantized or misinterpreted.
        with Image.open(Path(path).expanduser()) as image:
            if image.mode not in ("RGB", "RGBA", "P", "L", "LA"):
                raise ValueError(
                    f"Unsupported texture mode {image.mode}; supply an 8-bit sRGB image."
                )
            # A uniform RGBA representation gives opaque images full coverage
            # while retaining explicit transparency where the source provides it.
            pixels = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
        return TexturePixelsData(
            rgb=pixels[..., :3], alpha=pixels[..., 3]
        ).to_internal()

    @staticmethod
    def write_png(
        rgb: FloatArray, alpha: FloatArray, path: Path | str, *, overwrite: bool = False
    ) -> Path:
        """Quantize unit-range color and alpha to an explicitly encoded PNG file."""
        # Validate bounds before quantization so invalid numerical results are
        # rejected instead of hidden by clipping or integer overflow.
        image = TexturePixelsData(rgb=rgb, alpha=alpha).to_internal()
        pixels = np.rint(np.dstack((image.rgb, image.alpha)) * 255).astype(np.uint8)
        # Finish image encoding in memory before creating the destination file,
        # keeping encoding failures from leaving a partial PNG artifact.
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        return Files.write(path, buffer.getvalue(), overwrite=overwrite)

    @staticmethod
    def to_linear(rgb: FloatArray) -> FloatArray:
        """Decode sRGB's transfer curve before physically meaningful averaging."""
        # Undo sRGB's nonlinear encoding so arithmetic averages represent average
        # light levels instead of averages distorted by display encoding.
        return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    @staticmethod
    def from_linear(rgb: FloatArray) -> FloatArray:
        """Encode nonnegative linear-light RGB back into bounded sRGB."""
        # Limit output to representable display values before applying the sRGB
        # curve, which is undefined for negative inputs in its power-law branch.
        bounded = np.clip(rgb, 0, 1)
        return np.where(
            bounded <= 0.0031308, 12.92 * bounded, 1.055 * bounded ** (1 / 2.4) - 0.055
        )
