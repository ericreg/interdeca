"""Exercise the unmodified upstream workflows and the independent host contract."""

import sys
from pathlib import Path

import numpy as np
import pytest

from interdeca import _morphoweave_host, morphoweave
from interdeca._morphoweave_host import Host, load_atlas_builder
from interdeca.atlas import Atlas
from interdeca.models import AtlasModel, Landmarks, Mesh, Specimen
from interdeca.morphoweave import MorphoWeave
from interdeca.serde import AtlasRequest


@pytest.fixture
def specimens():
    """Asymmetric tetrahedra include a UV seam and labeled noncoplanar landmarks."""
    points = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    uv = np.arange(24, dtype=float).reshape(4, 3, 2) / 24
    labels = ("origin", "x", "y", "z")
    return tuple(
        Specimen(
            name,
            Mesh(points + offset, faces, uv).validated(),
            Landmarks(points + offset, labels).validated(),
        ).validated()
        for name, offset in (
            ("First", np.zeros(3)),
            ("Second", np.array([3.0, -1.0, 2.0])),
        )
    )


def test_high_level_workflow_runs_without_slicer_and_preserves_source_uvs(
    specimens, monkeypatch
):
    """The adapter must call upstream orchestration, never silently use native Atlas."""
    calls = []
    original_loader = morphoweave.load_atlas_builder

    def load(host):
        module = original_loader(host)
        for cls, name in (
            (module.MorphoWeaveAtlasBuilderWidget, "_generateAtlas"),
            (module.MorphoWeaveAtlasBuilderLogic, "runMean"),
            (module.MorphoWeaveAtlasBuilderLogic, "runAlign"),
        ):
            original = getattr(cls, name)

            def record(*args, _method=original, _name=name, **kwargs):
                calls.append(_name)
                return _method(*args, **kwargs)

            monkeypatch.setattr(cls, name, record)
        return module

    def forbidden(*args, **kwargs):
        raise AssertionError("MorphoWeave must not fall back to native Atlas")

    monkeypatch.setattr(morphoweave, "load_atlas_builder", load)
    monkeypatch.setattr(Atlas, "compute_reference", forbidden)
    monkeypatch.setattr(Atlas, "align_specimen", forbidden)
    originals = tuple(item.mesh.vertices.copy() for item in specimens)
    atlas = MorphoWeave.build_atlas(specimens)
    aligned = MorphoWeave.align_specimens(specimens, atlas)
    assert calls == ["_generateAtlas", "runAlign", "runMean", "runAlign"]
    assert atlas.landmarks.labels == specimens[0].landmarks.labels
    assert len(atlas.mesh.faces) == 4
    for before, source, result in zip(originals, specimens, aligned, strict=True):
        assert result.specimen_id == source.specimen_id
        assert result.landmarks.labels == source.landmarks.labels
        np.testing.assert_array_equal(before, source.mesh.vertices)
        np.testing.assert_array_equal(source.mesh.faces, result.mesh.faces)
        np.testing.assert_array_equal(source.mesh.uv, result.mesh.uv)
    # The translated inputs have the same shape and should align to the same surface.
    np.testing.assert_allclose(
        aligned[0].mesh.vertices, aligned[1].mesh.vertices, atol=1e-12
    )


def test_alignment_scale_choice_is_obeyed(specimens):
    """Upstream runAlign's setting must control size independently of the mean GPA."""
    source = specimens[0]
    target = AtlasModel(
        source.mesh._replace(vertices=source.mesh.vertices * 2),
        source.landmarks._replace(points=source.landmarks.points * 2),
    )
    rigid = MorphoWeave.align_specimens((source,), target)[0]
    similarity = MorphoWeave.align_specimens((source,), target, normalize_scale=True)[0]
    original_size = np.linalg.norm(source.mesh.vertices[1] - source.mesh.vertices[0])
    assert np.linalg.norm(
        rigid.mesh.vertices[1] - rigid.mesh.vertices[0]
    ) == pytest.approx(original_size)
    assert np.linalg.norm(
        similarity.mesh.vertices[1] - similarity.mesh.vertices[0]
    ) == pytest.approx(2 * original_size)


def test_full_projection_preserves_collapsed_faces_and_uvs(specimens):
    source = specimens[0]
    surface = source._replace(
        mesh=Mesh(source.mesh.vertices[:3], np.array([[0, 1, 2]])).validated()
    )
    atlas = AtlasModel(
        Mesh(
            np.array(
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            np.array([[0, 1, 3], [1, 2, 3]]),
            source.mesh.uv[:2],
        ).validated(),
        source.landmarks,
    )
    result = MorphoWeave.resample_specimen(surface, atlas)
    assert len(result.mesh.vertices) == len(atlas.mesh.vertices)
    np.testing.assert_array_equal(result.mesh.vertices[1], result.mesh.vertices[2])
    np.testing.assert_array_equal(result.mesh.faces, atlas.mesh.faces)
    np.testing.assert_array_equal(result.mesh.uv, atlas.mesh.uv)
    np.testing.assert_allclose(result.projection_distances, [0, 0, 2, 0], atol=1e-6)


def test_host_imports_are_isolated(specimens):
    names = ("qt", "ctk", "slicer", "slicer.ScriptedLoadableModule")
    before = {name: sys.modules.get(name) for name in names}
    first, second = (Host(specimens[0].landmarks.labels) for _ in range(2))
    first_module, second_module = load_atlas_builder(first), load_atlas_builder(second)
    node = first_module.MorphoWeaveAtlasBuilderLogic().numpyToFiducialNode(
        specimens[0].landmarks.points, "test"
    )
    assert node in first.nodes and node not in second.nodes
    assert second_module.slicer.util is second
    with pytest.raises(RuntimeError, match="Unsupported"):
        _ = first_module.qt.QWidget
    assert before == {name: sys.modules.get(name) for name in names}


def test_changed_or_missing_upstream_source_fails(tmp_path, monkeypatch):
    source = tmp_path / "upstream.py"
    monkeypatch.setattr(_morphoweave_host, "SOURCE", source)
    with pytest.raises(RuntimeError, match="submodule"):
        load_atlas_builder(Host(()))
    source.write_text("raise AssertionError('must never execute')\n")
    with pytest.raises(RuntimeError, match="reviewed revision"):
        load_atlas_builder(Host(()))


def test_ambiguous_specimen_ids_are_rejected(specimens):
    first = specimens[0]
    with pytest.raises(ValueError, match="case-insensitively"):
        MorphoWeave.build_atlas(
            (first, first._replace(specimen_id=first.specimen_id.lower()))
        )


def test_coordinate_mismatch_is_not_silently_overridden(specimens):
    first = specimens[0]
    wrong = first._replace(mesh=first.mesh._replace(vertices=first.mesh.vertices + 100))
    with pytest.raises(ValueError, match="coordinate mismatch"):
        MorphoWeave.align_specimens((wrong,), AtlasModel(first.mesh, first.landmarks))


def test_backend_request_defaults_keep_original_atlas():
    request = AtlasRequest(models=Path("."), landmarks=Path("."), output=Path("output"))
    assert request.to_internal().backend == "native"
    assert not request.normalize_scale
    with pytest.raises(ValueError):
        AtlasRequest(
            models=Path("."),
            landmarks=Path("."),
            output=Path("output"),
            backend="unknown",
        )


def test_singular_landmark_warp_is_rejected_before_vtk(specimens):
    first = specimens[0]
    points = first.landmarks.points.copy()
    points[3] = [1, 1, 0]
    planar = first._replace(landmarks=Landmarks(points, first.landmarks.labels))
    with pytest.raises(ValueError, match="noncoplanar"):
        MorphoWeave.build_atlas((planar,))


def test_duplicate_warp_positions_are_rejected_before_vtk(specimens):
    first = specimens[0]
    # A fifth landmark at the first position retains rank three but makes TPS singular.
    landmarks = Landmarks(
        np.vstack((first.landmarks.points, first.landmarks.points[0])),
        (*first.landmarks.labels, "duplicate"),
    ).validated()
    source = first._replace(landmarks=landmarks)
    with pytest.raises(ValueError, match="distinct positions"):
        MorphoWeave.resample_specimen(source, AtlasModel(first.mesh, landmarks))
