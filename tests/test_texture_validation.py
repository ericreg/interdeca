"""Source validation decodes files without creating discarded numerical colors."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from interdeca.blender import Blender
from interdeca.cli import main
from interdeca.dataset import Dataset
from interdeca.io import LandmarkIO, MeshIO, TextureIO
from interdeca.models import BlenderInstallation, Landmarks, Mesh, ResampledSpecimen
from interdeca.serde import RawTexturePresentation, TexturePixelsData


@pytest.fixture
def pixels():
    # Incompressible data leaves a substantial pixel payload after each header,
    # so truncation below exercises decoding rather than only format discovery.
    return np.random.default_rng(42).integers(0, 256, (128, 128, 4), dtype=np.uint8)


@pytest.fixture
def opened_images(monkeypatch):
    original = Image.open
    streams = []

    def track(*args, **kwargs):
        image = original(*args, **kwargs)
        streams.append(image.fp)
        return image

    monkeypatch.setattr(Image, "open", track)
    return streams


@pytest.fixture
def forbid_float_pixels(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("File validation must not materialize float64 texture pixels")

    monkeypatch.setattr(TexturePixelsData, "from_rgba8", unexpected)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P", "L", "LA"])
@pytest.mark.parametrize("suffix", [".png", ".tiff"])
def test_valid_file_decodes_and_closes_without_float_colors(
    tmp_path, pixels, opened_images, forbid_float_pixels, mode, suffix
):
    path = tmp_path / (mode + suffix)
    Image.fromarray(pixels).convert(mode).save(path)
    before = path.read_bytes()
    assert TextureIO.validate_file(path) is None
    assert path.read_bytes() == before
    assert opened_images and all(stream.closed for stream in opened_images)


@pytest.mark.parametrize("reader", [TextureIO.read, TextureIO.validate_file])
@pytest.mark.parametrize("suffix", [".png", ".tiff"])
def test_valid_header_with_truncated_pixels_fails_and_closes(
    tmp_path, pixels, opened_images, reader, suffix
):
    path = tmp_path / ("truncated" + suffix)
    Image.fromarray(pixels).save(path, compress_level=0)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    # Header-only inspection still succeeds; accepting this file would let a
    # corrupt source get as far as Blender and obscure the actual input error.
    with Image.open(path) as image:
        assert image.size == (128, 128)
    # The raw TIFF decoder reports a short mapped buffer as ValueError; the PNG
    # decoder reports truncated compressed data as OSError. Preserve both paths.
    with pytest.raises(OSError if suffix == ".png" else ValueError):
        reader(path)
    assert all(stream.closed for stream in opened_images)


@pytest.mark.parametrize("reader", [TextureIO.read, TextureIO.validate_file])
@pytest.mark.parametrize("suffix", [".png", ".tiff"])
def test_corrupt_header_and_missing_file_fail(tmp_path, reader, suffix):
    path = tmp_path / ("broken" + suffix)
    with pytest.raises(FileNotFoundError):
        reader(path)
    path.write_bytes(b"not a readable image")
    with pytest.raises(OSError):
        reader(path)


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.int32])
def test_unsupported_encoding_fails_and_closes(tmp_path, opened_images, dtype):
    path = tmp_path / "unsupported.tiff"
    Image.fromarray(np.ones((4, 4), dtype=dtype)).save(path)
    with pytest.raises(ValueError, match="Unsupported texture mode"):
        TextureIO.validate_file(path)
    assert opened_images and all(stream.closed for stream in opened_images)


@pytest.fixture
def specimen_files(tmp_path, pixels):
    landmarks = Landmarks(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ("origin", "x", "y", "z"),
    ).validated()
    mesh = Mesh(
        landmarks.points[:3],
        np.array([[0, 1, 2]]),
        np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
    ).validated()
    mesh_path = MeshIO.write_obj(mesh, tmp_path / "source.obj")
    landmark_path = LandmarkIO.write_json(landmarks, tmp_path / "source.mrk.json")
    texture_path = tmp_path / "source.png"
    Image.fromarray(pixels).save(texture_path, compress_level=0)
    return mesh_path, landmark_path, texture_path


@pytest.mark.parametrize("change", ["truncated", "unsupported", "missing"])
def test_baking_rechecks_source_modified_after_inspection(
    tmp_path, specimen_files, forbid_float_pixels, monkeypatch, change
):
    mesh_path, landmark_path, texture_path = specimen_files
    specimen = Dataset.inspect_specimen("one", mesh_path, landmark_path, texture_path)
    target = ResampledSpecimen(
        "one", specimen.mesh, specimen.landmarks, np.zeros(len(specimen.mesh.vertices))
    ).validated()
    if change == "truncated":
        data = texture_path.read_bytes()
        texture_path.write_bytes(data[: len(data) // 2])
        error = OSError
    elif change == "unsupported":
        Image.fromarray(np.ones((4, 4), dtype=np.float32)).save(
            texture_path, format="TIFF"
        )
        error = ValueError
    else:
        texture_path.unlink()
        error = FileNotFoundError

    # Invalid input must fail before a subprocess or published artifact exists.
    def unexpected_run(*args, **kwargs):
        pytest.fail("Blender must not start with an unusable source texture")

    monkeypatch.setattr(Blender, "_run", unexpected_run)
    destination = tmp_path / "baked.png"
    installation = BlenderInstallation(Path("unused-blender"), (5, 2, 1), "unused")
    with pytest.raises(error):
        Blender.bake_specimen_texture(specimen, target, installation, destination)
    with pytest.raises(error):
        Dataset.inspect_specimen("one", mesh_path, landmark_path, texture_path)
    assert not destination.exists()


def test_cli_raw_texture_publishes_same_file_without_loading_float_colors(
    tmp_path, specimen_files, forbid_float_pixels
):
    mesh_path, _landmark_path, texture_path = specimen_files
    output = tmp_path / "session"
    result = CliRunner().invoke(
        main,
        [
            "--output",
            str(output),
            "raw-texture",
            "--mesh",
            str(mesh_path),
            str(texture_path),
        ],
    )
    assert result.exit_code == 0, result.output
    presentations = list((output / "results").glob("*_raw_texture.json"))
    assert len(presentations) == 1
    presentation = RawTexturePresentation.model_validate_json(
        presentations[0].read_text()
    )
    assert presentation.texture == texture_path


def test_cli_raw_texture_refuses_corrupt_source(tmp_path, specimen_files):
    mesh_path, _landmark_path, texture_path = specimen_files
    texture_path.write_bytes(b"corrupted after selection")
    output = tmp_path / "session"
    result = CliRunner().invoke(
        main,
        [
            "--output",
            str(output),
            "raw-texture",
            "--mesh",
            str(mesh_path),
            str(texture_path),
        ],
    )
    assert result.exit_code != 0
    assert not list(output.glob("results/*_raw_texture.json"))
