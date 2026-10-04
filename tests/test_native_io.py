"""Public mesh and texture contracts at the new native I/O boundary."""

import numpy as np
import pytest
from PIL import Image

from interdeca import _native
from interdeca.io import MeshIO, TextureIO
from interdeca.models import Mesh


def test_obj_axes_seams_and_exact_roundtrip(tmp_path):
    source = tmp_path / "input.obj"
    source.write_text(
        "\ufeffv -0.0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
        "vt 0.12345678901234567 0.8765432109876543\nvt 1 0\nvt 0 1\n"
        "f 1/1 2/2 3/3\ng other\nf -4/2 -2/1 -1/3\n"
    )
    ras = MeshIO.read(source)
    lps = MeshIO.read(source, coordinate_system="LPS")
    assert lps.vertices.tobytes() == (ras.vertices * [-1, -1, 1]).tobytes()
    assert lps.faces.tobytes() == ras.faces.tobytes()
    assert lps.uv.tobytes() == ras.uv.tobytes()
    # Unused vertices exercise finite double extremes without changing the
    # geometry of the two valid triangles or the area validation threshold.
    extremes = np.array([[np.finfo(float).max, np.nextafter(0.0, 1.0), -0.0]])
    mesh = ras._replace(vertices=np.vstack((ras.vertices, extremes)))
    output = MeshIO.write_obj(mesh, tmp_path / "nested/output.obj")
    actual = MeshIO.read(output)
    for expected, value in zip(mesh, actual, strict=True):
        assert expected.tobytes() == value.tobytes()
        assert value.flags.c_contiguous and not value.flags.writeable
    with pytest.raises(FileExistsError):
        MeshIO.write_obj(mesh, output)
    MeshIO.write_obj(mesh, output, overwrite=True)


@pytest.mark.parametrize("field,values", [
    ("vertices", [[np.nan, 0, 0], [1, 0, 0], [0, 1, 0]]),
    ("vertices", [[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
    ("faces", [[0, 0, 2]]),
    ("faces", [[0, 1, 3]]),
    ("faces", [[0.0, 1.0, 2.0]]),
    ("uv", [[[0, 0], [1, 0]]]),
])
def test_writer_validates_before_overwrite(tmp_path, field, values):
    mesh = Mesh(np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]]))
    output = tmp_path / "preserved.obj"
    output.write_bytes(b"existing bytes")
    with pytest.raises(ValueError):
        MeshIO.write_obj(mesh._replace(**{field: np.array(values)}), output, overwrite=True)
    assert output.read_bytes() == b"existing bytes"


def test_png_rounding_ties_and_strided_channels(tmp_path):
    half = (np.arange(255) + 0.5) / 255.0
    values = np.concatenate((half, np.nextafter(half, 0), np.nextafter(half, 1)))
    rgb = np.broadcast_to(values[:, None, None], (len(values), 2, 3))[::-1]
    alpha = np.broadcast_to(values[:, None], (len(values), 2))[::-1]
    path = TextureIO.write_png(rgb, alpha, tmp_path / "pixels.png")
    with Image.open(path) as image:
        actual = np.asarray(image)
    expected = np.rint(np.dstack((rgb, alpha)) * 255).astype(np.uint8)
    np.testing.assert_array_equal(actual, expected)


def test_transfer_curves_keep_thresholds_and_clipping():
    values = np.array([-np.inf, -0.0, 0.0, 0.0031308, 0.04045, 0.5, 1.0, np.inf, np.nan])
    with np.errstate(invalid="ignore"):
        decoded = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
        bounded = np.clip(values, 0, 1)
        encoded = np.where(bounded <= 0.0031308, 12.92 * bounded,
                           1.055 * bounded ** (1 / 2.4) - 0.055)
    for actual, expected in ((TextureIO.to_linear(values), decoded),
                             (TextureIO.from_linear(values), encoded)):
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0, equal_nan=True)
        assert np.array_equal(np.signbit(actual[expected == 0]), np.signbit(expected[expected == 0]))


def unaligned(values):
    """Build an ordinary NumPy buffer view that cannot back Rust references."""
    source = np.asarray(values)
    result = np.ndarray(source.shape, dtype=source.dtype,
                        buffer=bytearray(source.nbytes + 1), offset=1)
    result[...] = source
    assert not result.flags.aligned
    return result


def test_unaligned_mesh_and_texture_inputs(tmp_path):
    # Odd row strides also occur in packed records, independently of the first
    # element's alignment. Public schemas must preserve logical element values.
    vertices = np.ndarray((3, 3), dtype=np.float64, buffer=bytearray(80),
                          offset=1, strides=(25, 8))
    vertices[...] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    faces = unaligned(np.array([[0, 1, 2]], dtype=np.int64))
    uv = unaligned(np.array([[[0., 0.], [1., 0.], [0., 1.]]]))
    mesh = Mesh(vertices, faces, uv)
    actual = MeshIO.read(MeshIO.write_obj(mesh, tmp_path / "unaligned.obj"))
    for expected, value in zip(mesh, actual, strict=True):
        np.testing.assert_array_equal(value, expected)
    rgb = unaligned(np.array([[[0., .5, 1.], [.25, .75, .125]]]))
    alpha = unaligned(np.array([[1., .5]]))
    for operation in (TextureIO.to_linear, TextureIO.from_linear):
        np.testing.assert_array_equal(operation(rgb), operation(rgb.copy()))
    path = TextureIO.write_png(rgb, alpha, tmp_path / "unaligned.png")
    with Image.open(path) as image:
        np.testing.assert_array_equal(np.asarray(image),
                                      np.rint(np.dstack((rgb, alpha)) * 255).astype(np.uint8))
    # The private extension also defends itself when called without schemas.
    with pytest.raises(ValueError, match="aligned"):
        _native.write_obj(tmp_path / "invalid.obj", vertices, faces, uv)
    with pytest.raises(ValueError, match="aligned"):
        _native.texture_to_rgba8(rgb, alpha)
    with pytest.raises(ValueError, match="aligned"):
        _native.texture_to_linear(rgb)


def test_empty_and_excessive_rank_native_inputs(tmp_path):
    empty = np.ndarray((0, 3), dtype=np.float64, buffer=bytearray(25),
                       offset=1, strides=(-24, 8))
    for operation in (TextureIO.to_linear, TextureIO.from_linear):
        result = operation(empty)
        assert result.shape == empty.shape and not result.flags.writeable
    with pytest.raises(ValueError, match="vertices must have shape"):
        _native.write_obj(tmp_path / "empty.obj", empty, np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="nonempty"):
        _native.texture_from_rgba8(np.empty((0, 1, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="nonempty"):
        _native.texture_to_rgba8(np.empty((0, 1, 3)), np.empty((0, 1)))
    with pytest.raises(ValueError, match="at most 32"):
        _native.texture_from_linear(np.ones((1,) * 33))
