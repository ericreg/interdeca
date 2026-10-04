"""Check benchmark measurements against analytic geometry and known image samples."""

import numpy as np
import pytest
from PIL import Image

from interdeca.color_analysis import ColorAnalysis
from interdeca.io import LandmarkIO, MeshIO, TextureIO
from interdeca.models import Landmarks, Mesh
from tools.profiling.accuracy import (
    Colors,
    Projection,
    SurfaceIndex,
    area_samples,
    barycentric,
    color_difference,
    fit,
    mesh_metrics,
    sample_image,
    surface_difference,
    transformed,
)
from tools.profiling.population import prepare


def triangle():
    return Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0, 1, 2]]),
        np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
    ).validated()


def test_pose_and_scale_are_separate_and_reflections_are_rejected():
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = 2 * source @ rotation + [4.0, 5.0, 6.0]
    similarity = fit(source, target, scaling=True)
    assert similarity.scale == pytest.approx(2)
    np.testing.assert_allclose(transformed(source, similarity), target, atol=1e-14)
    rigid = fit(source, target)
    assert rigid.scale == 1
    assert np.linalg.norm(transformed(source, rigid) - target) > 1
    reflected = source * [-1, 1, 1]
    proper = fit(source, reflected)
    assert np.linalg.det(proper.rotation) == pytest.approx(1)
    assert np.linalg.norm(transformed(source, proper) - reflected) > 0.1


def test_parallel_surfaces_have_known_distance_and_barycentric_projection():
    mesh = triangle()
    shifted = Mesh(mesh.vertices + [0.0, 0.0, 3.0], mesh.faces, mesh.uv)
    difference = surface_difference(mesh, shifted, 100, 413)
    assert difference.first_to_second.rms == pytest.approx(3)
    assert difference.second_to_first.maximum == pytest.approx(3)
    projection = SurfaceIndex(mesh).project(np.array([[0.2, 0.3, 2.0]]))
    np.testing.assert_allclose(barycentric(mesh, projection), [[0.5, 0.2, 0.3]])
    samples = area_samples(mesh, 10000, 413)
    np.testing.assert_allclose(
        samples.points.mean(axis=0), [1 / 3, 1 / 3, 0], atol=0.01
    )


def test_texture_diagnostics_match_application_sampling_and_report_no_coverage(
    tmp_path,
):
    # Distinct colors/alphas expose vertical flips, pixel-center shifts, and
    # interpolation of transparent RGB as if it were visible color.
    pixels = np.array(
        [[[255, 0, 0, 255], [0, 255, 0, 128]], [[0, 0, 255, 0], [150, 100, 50, 255]]],
        dtype=np.uint8,
    )
    path = tmp_path / "image.png"
    Image.fromarray(pixels).save(path)
    uv = np.random.default_rng(413).random((100, 2))
    expected = ColorAnalysis._sample_image(*TextureIO.read(path), uv)
    actual = sample_image(path, uv)
    np.testing.assert_allclose(actual.rgb, expected[0], atol=1e-15)
    np.testing.assert_allclose(actual.alpha, expected[1], atol=1e-15)
    np.testing.assert_allclose(
        sample_image(path, np.array([[0.25, 0.75]])).rgb, [[1, 0, 0]]
    )
    missing = Colors(np.zeros((3, 3)), np.zeros(3))
    comparison = color_difference(missing, missing)
    assert comparison.jointly_covered == 0
    assert comparison.mean_absolute_channel_0_255 is None
    assert comparison.delta_e_76.count == 0


def test_barycentric_coordinates_of_a_valid_very_thin_triangle():
    mesh = Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-10, 0.0]]),
        np.array([[0, 1, 2]]),
    ).validated()
    projection = Projection(
        np.zeros(1), np.array([0]), np.array([[0.75, 2.5e-11, 0.0]])
    )
    np.testing.assert_allclose(
        barycentric(mesh, projection), [[0.25, 0.5, 0.25]], atol=1e-15
    )


def test_input_repair_is_explicit_recorded_and_keeps_remaining_corner_order(tmp_path):
    source = tmp_path / "source"
    for name in ("models", "textures", "landmarks"):
        (source / name).mkdir(parents=True)
    mesh = triangle()
    broken = Mesh(
        np.vstack((mesh.vertices, mesh.vertices[1])),
        np.array([[0, 1, 3], [0, 1, 2]]),
        np.repeat(mesh.uv, 2, axis=0),
    )
    model = source / "models/example.obj"
    MeshIO.write_obj(broken, model, allow_degenerate=True)
    original = model.read_bytes()
    LandmarkIO.write_json(
        Landmarks(mesh.vertices, ("a", "b", "c")), source / "landmarks/example.mrk.json"
    )
    Image.new("RGBA", (2, 2), "red").save(source / "textures/example.png")
    with pytest.raises(ValueError, match="degenerate"):
        prepare(source, tmp_path / "rejected", 1)
    manifest = prepare(source, tmp_path / "repaired", 1, repair_zero_area=True)
    assert manifest.removed_zero_area_faces == {"example": [0]}
    observed = MeshIO.read(tmp_path / "repaired/models/example.obj")
    np.testing.assert_array_equal(observed.vertices, broken.vertices * [-1, -1, 1])
    np.testing.assert_array_equal(observed.faces, mesh.faces)
    np.testing.assert_array_equal(observed.uv, mesh.uv)
    assert model.read_bytes() == original
    assert mesh_metrics(broken).collapsed_triangles == 1
