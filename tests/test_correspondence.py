"""Regression checks for many-to-one projection without changing atlas indexing."""

import numpy as np
import pytest

from interdeca.blender import Blender
from interdeca.geometry import Geometry
from interdeca.io import MeshIO
from interdeca.models import AtlasModel, Landmarks, Mesh, Specimen
from interdeca.serde import CorrespondenceMeshData, MeshData, ResampledSpecimenData


def test_projected_faces_survive_serialization_and_blender_handoff(tmp_path):
    """Two atlas vertices hit the same source corner; no face or UV may disappear."""
    landmarks = Landmarks(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ("origin", "x", "y", "z"),
    ).validated()
    source = Specimen(
        "triangle",
        Mesh(landmarks.points[:3], np.array([[0, 1, 2]])).validated(),
        landmarks,
    ).validated()
    uv = np.array(
        [[[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]], [[0.5, 0.0], [1.0, 0.0], [0.0, 1.0]]]
    )
    atlas = AtlasModel(
        Mesh(
            np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            np.array([[0, 1, 3], [1, 2, 3]]),
            uv,
        ).validated(),
        landmarks,
    )
    projected = Geometry.resample_specimen(source, atlas)
    np.testing.assert_allclose(
        projected.mesh.vertices[1], projected.mesh.vertices[2], atol=1e-14
    )
    np.testing.assert_array_equal(projected.mesh.faces, atlas.mesh.faces)
    np.testing.assert_array_equal(projected.mesh.uv, uv)
    np.testing.assert_allclose(projected.projection_distances, [0, 0, 1, 0], atol=1e-14)

    # The explicit output schema accepts this geometry; ordinary source and analysis
    # paths must still refuse it rather than use unstable barycentric or area values.
    decoded = ResampledSpecimenData.model_validate_json(
        ResampledSpecimenData.from_internal(projected).model_dump_json()
    ).to_internal()
    np.testing.assert_array_equal(decoded.mesh.vertices, projected.mesh.vertices)
    assert not decoded.mesh.vertices.flags.writeable
    with pytest.raises(ValueError, match="degenerate"):
        MeshData.from_internal(projected.mesh)
    with pytest.raises(ValueError, match="degenerate"):
        Geometry.prepare_analysis(projected.mesh)
    with pytest.raises(ValueError, match="degenerate"):
        MeshIO.write_obj(projected.mesh, tmp_path / "strict.obj")
    path = MeshIO.write_obj(
        projected.mesh, tmp_path / "projected.obj", allow_degenerate=True
    )
    with pytest.raises(ValueError, match="degenerate"):
        MeshIO.read(path)
    restored = MeshIO.read(path, allow_degenerate=True)
    for before, after in zip(projected.mesh, restored, strict=True):
        np.testing.assert_array_equal(before, after)
    Blender._save_mesh(projected.mesh, tmp_path / "target.npz", allow_degenerate=True)
    with np.load(tmp_path / "target.npz", allow_pickle=False) as target:
        for field in ("vertices", "faces", "uv"):
            np.testing.assert_array_equal(target[field], getattr(projected.mesh, field))


@pytest.mark.parametrize("faces", [np.array([[0, 1, 3]]), np.array([[0, 0, 1]])])
def test_correspondence_schema_still_rejects_invalid_indexing(faces):
    """A collapse of positions never permits out-of-range or repeated index identities."""
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        CorrespondenceMeshData(vertices=vertices, faces=faces)


def test_correspondence_policy_is_not_an_application_wire_option():
    """A client cannot weaken strict source-mesh validation with an extra JSON field."""
    with pytest.raises(ValueError):
        MeshData.model_validate_json(
            '{"vertices":[[0,0,0],[1,0,0],[0,1,0]],"faces":[[0,1,2]],"allow_degenerate":true}'
        )
