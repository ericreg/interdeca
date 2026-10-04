"""Native diffusion preserves weighted neighborhoods, coverage and sample order."""

import numpy as np
import pytest

from interdeca import _native
from interdeca.geometry import Geometry
from interdeca.models import Mesh
from interdeca.serde import AnalysisGeometryData, FaceAdjacencyData


@pytest.mark.parametrize("steps", [0, 1, 2, 3, 5])
def test_diffusion_matches_weighted_transition_matrix(steps):
    # Face 2 is uncovered, face 3 is valid black, and face 4 is disconnected.
    # A matrix power independently specifies simultaneous diffusion at each step.
    rgb = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [0, 0, 0], [.25, .5, .75]])
    valid = np.array([True, True, False, True, True])
    areas = np.array([1., 2, 3, 4, 5])
    offsets = np.array([0, 2, 5, 7, 8, 8], dtype=np.int64)
    neighbors = np.array([1, 2, 0, 2, 3, 0, 1, 1], dtype=np.int64)
    samples = np.array([3, 0, 4, 2], dtype=np.int64)
    transition = np.array([
        [1/3, 2/3, 0, 0, 0], [1/7, 2/7, 0, 4/7, 0], [0, 0, 0, 0, 0],
        [0, 1/3, 0, 2/3, 0], [0, 0, 0, 0, 1],
    ])
    covered = rgb.copy()
    covered[~valid] = 0
    expected = (np.linalg.matrix_power(transition, steps) @ covered)[samples]
    before = rgb.tobytes()
    result = _native.smooth_face_colors(rgb, valid, areas, offsets, neighbors, samples, steps)
    np.testing.assert_allclose(result, expected, rtol=2e-15, atol=2e-15)
    assert rgb.tobytes() == before
    assert result.flags.c_contiguous and not result.flags.writeable


def test_high_degree_nonmanifold_neighborhood():
    rng = np.random.default_rng(23)
    rgb = rng.random((20, 3))
    areas = rng.random(20) + .1
    # Every other face touches face zero; do not assume three neighbors maximum.
    offsets = np.array([0] + [19] * 20, dtype=np.int64)
    indices = np.arange(1, 20, dtype=np.int64)
    result = _native.smooth_face_colors(
        rgb, np.ones(20, dtype=bool), areas, offsets, indices, np.array([0]), 1,
    )
    np.testing.assert_allclose(result[0], np.average(rgb, axis=0, weights=areas),
                               rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("offsets,indices", [
    ([0, 1], [0]), ([1, 1, 1], [0]), ([0, 2, 1], [0]),
    ([0, 0, 1], [2]), ([0, 0, 1], [-1]),
])
def test_native_rejects_malformed_graph(offsets, indices):
    with pytest.raises(ValueError, match="adjacency"):
        _native.smooth_face_colors(
            np.ones((2, 3)), np.ones(2, dtype=bool), np.ones(2),
            np.array(offsets, dtype=np.int64), np.array(indices, dtype=np.int64),
            np.array([0]), 1,
        )


def test_packed_adjacency_schema_and_nonmanifold_preparation():
    mesh = Mesh(
        np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0], [0, -1., 0], [0, 0, 1.]]),
        np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]), np.zeros((3, 3, 2)),
    ).validated()
    analysis = Geometry.prepare_analysis(mesh)
    np.testing.assert_array_equal(analysis.adjacency.offsets, [0, 2, 4, 6])
    np.testing.assert_array_equal(analysis.adjacency.indices, [1, 2, 0, 2, 0, 1])
    restored = AnalysisGeometryData.model_validate_json(
        AnalysisGeometryData.from_internal(analysis).model_dump_json()
    ).to_internal()
    for before, after in zip(analysis.adjacency, restored.adjacency, strict=True):
        np.testing.assert_array_equal(after, before)
        assert not after.flags.writeable
    with pytest.raises(ValueError):
        FaceAdjacencyData(offsets=np.array([0, 1]), indices=np.array([1]))


def test_native_rejects_column_major_colors():
    # A flat Rust slice would otherwise interpret a Fortran array in the wrong
    # face/channel order even though its memory is contiguous.
    with pytest.raises(ValueError, match="C-contiguous"):
        _native.smooth_face_colors(
            np.asfortranarray(np.arange(6.).reshape(2, 3)),
            np.ones(2, dtype=bool), np.ones(2), np.array([0, 0, 0]),
            np.array([], dtype=np.int64), np.array([0, 1]), 1,
        )
