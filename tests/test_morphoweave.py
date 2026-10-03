"""Exercise the unmodified upstream workflows and the independent host contract."""

import sys
from collections.abc import Callable
from pathlib import Path
from queue import Queue
from threading import Event
from typing import NamedTuple

import numpy as np
import pytest

from interdeca import _morphoweave_host, morphoweave
from interdeca._morphoweave_host import Host, load_atlas_builder
from interdeca.application_models import AtlasOptions
from interdeca.atlas import Atlas
from interdeca.io import LandmarkIO, MeshIO
from interdeca.models import AtlasModel, AtlasScaleConversion, Landmarks, Mesh, Specimen
from interdeca.morphoweave import MorphoWeave
from interdeca.serde import AtlasRequest, AtlasRunData, AtlasScaleConversionData
from interdeca.serial_execution import SerialExecutionManager
from interdeca.workflows import Operations, Workflows


class Callbacks(NamedTuple):
    """Post thread-safe callbacks for delivery by the test's owning thread."""

    post: Callable


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
    normalized = MorphoWeave.build_atlas(specimens)
    scaled = MorphoWeave.restore_input_scale(normalized, specimens)
    atlas = scaled.atlas
    assert scaled.conversion.source_centroid_size == pytest.approx(1)
    assert scaled.conversion.target_centroid_size == pytest.approx(np.sqrt(10.5))
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


def test_atlas_scale_restoration_preserves_pose_shape_and_correspondence(specimens):
    """Unequal, translated specimens distinguish arithmetic mean size from RMS size."""
    first = specimens[0]
    offset = np.array([12.0, -3.0, 8.0])
    large = Specimen(
        "Large",
        first.mesh._replace(vertices=3 * first.mesh.vertices + offset),
        first.landmarks._replace(points=3 * first.landmarks.points + offset),
    ).validated()
    atlas = AtlasModel(
        first.mesh._replace(vertices=0.5 * first.mesh.vertices + offset).validated(),
        first.landmarks._replace(
            points=0.5 * first.landmarks.points + offset
        ).validated(),
    )
    original_vertices = atlas.mesh.vertices.copy()
    original_landmarks = atlas.landmarks.points.copy()
    result = MorphoWeave.restore_input_scale(atlas, (first, large))
    center = original_landmarks.mean(axis=0)
    assert result.conversion.source_centroid_size == pytest.approx(0.5 * np.sqrt(10.5))
    assert result.conversion.target_centroid_size == pytest.approx(2 * np.sqrt(10.5))
    assert result.conversion.factor == pytest.approx(4)
    np.testing.assert_allclose(result.conversion.center, center)
    np.testing.assert_allclose(result.atlas.landmarks.points.mean(axis=0), center)
    np.testing.assert_allclose(
        result.atlas.mesh.vertices, center + 4 * (original_vertices - center)
    )
    np.testing.assert_allclose(
        result.atlas.landmarks.points, center + 4 * (original_landmarks - center)
    )
    np.testing.assert_array_equal(result.atlas.mesh.faces, atlas.mesh.faces)
    np.testing.assert_array_equal(result.atlas.mesh.uv, atlas.mesh.uv)
    assert result.atlas.landmarks.labels == atlas.landmarks.labels
    np.testing.assert_array_equal(atlas.mesh.vertices, original_vertices)
    np.testing.assert_array_equal(atlas.landmarks.points, original_landmarks)
    # The target is based on the input population, so applying the conversion a
    # second time has unit factor rather than scaling the atlas a second time.
    repeated = MorphoWeave.restore_input_scale(result.atlas, (first, large))
    assert repeated.conversion.factor == pytest.approx(1)
    np.testing.assert_allclose(repeated.atlas.mesh.vertices, result.atlas.mesh.vertices)


@pytest.mark.parametrize("normalize_scale", [False, True])
def test_restored_atlas_size_is_independent_of_specimen_alignment_mode(
    specimens, normalize_scale
):
    first = specimens[0]
    large = Specimen(
        "Large",
        first.mesh._replace(vertices=3 * first.mesh.vertices),
        first.landmarks._replace(points=3 * first.landmarks.points),
    ).validated()
    population = (first, large)
    raw = MorphoWeave.build_atlas(population, normalize_scale=normalize_scale)
    restored = MorphoWeave.restore_input_scale(raw, population)
    points = restored.atlas.landmarks.points
    assert np.linalg.norm(points - points.mean(axis=0)) == pytest.approx(
        2 * np.sqrt(10.5)
    )
    aligned = MorphoWeave.align_specimens(
        population, restored.atlas, normalize_scale=normalize_scale
    )
    # Identical tetrahedron shapes have exact expected sizes: a rigid fit keeps
    # the 1:3 ratio; a similarity fit takes both to the representative 2x size.
    expected = (2, 2) if normalize_scale else (1, 3)
    for result, multiplier in zip(aligned, expected, strict=True):
        vertices = result.mesh.vertices
        size = np.linalg.norm(vertices - vertices.mean(axis=0))
        assert size == pytest.approx(multiplier * np.sqrt(10.5), rel=1e-6)


def test_zero_size_upstream_atlas_is_rejected(specimens):
    first = specimens[0]
    atlas = AtlasModel(first.mesh, first.landmarks._replace(points=np.zeros((4, 3))))
    with pytest.raises(ValueError, match="positive centroid sizes"):
        MorphoWeave.restore_input_scale(atlas, specimens)


@pytest.mark.parametrize(
    "backend,override",
    [("native", False), ("morphoweave", True), ("morphoweave", False)],
)
def test_run_metadata_records_conversion_only_when_applied(
    tmp_path, specimens, backend, override
):
    atlas = AtlasModel(specimens[0].mesh, specimens[0].landmarks)
    options = AtlasOptions(
        tmp_path,
        tmp_path,
        tmp_path,
        backend=backend,
        atlas_model=tmp_path / "provided.obj" if override else None,
        atlas_landmarks=tmp_path / "provided.mrk.json" if override else None,
    )
    conversion = (
        AtlasScaleConversion(1.0, 2.0, 2.0, (1.0, 2.0, 3.0))
        if backend == "morphoweave" and not override
        else None
    )
    Operations.atlas_result(atlas, tmp_path, (), None, options, conversion)
    manifest = AtlasRunData.model_validate_json(
        (tmp_path / "atlas_run.json").read_text()
    )
    assert manifest.schema_version == 2
    if conversion is None:
        assert manifest.atlas_scale_conversion is None
    else:
        assert manifest.atlas_scale_conversion.to_internal() == conversion
        assert (
            manifest.atlas_scale_conversion.method
            == "mean_input_landmark_centroid_size"
        )


@pytest.mark.parametrize("factor", [0.0, -1.0, float("inf"), float("nan"), 3.0])
def test_scale_metadata_rejects_invalid_or_inconsistent_factor(factor):
    with pytest.raises(ValueError):
        AtlasScaleConversionData(
            source_centroid_size=1.0,
            target_centroid_size=2.0,
            factor=factor,
            center=(0.0, 0.0, 0.0),
        )


@pytest.mark.parametrize("normalize_scale", [False, True])
def test_shared_staging_matches_standalone_results_and_writes_originals_once(
    specimens, monkeypatch, normalize_scale
):
    # Independently owned calls provide the previous staging behavior as a
    # numerical control; shared staging must change only repeated file handling.
    reference_mean = MorphoWeave.build_atlas(specimens, normalize_scale=normalize_scale)
    reference_atlas = MorphoWeave.restore_input_scale(reference_mean, specimens).atlas
    reference_aligned = MorphoWeave.align_specimens(
        specimens, reference_atlas, normalize_scale=normalize_scale
    )
    writes, align_directories = [], []
    original_write, original_loader = MeshIO.write_obj, morphoweave.load_atlas_builder

    def write(mesh, path, **kwargs):
        if Path(path).parent.name == "originalModels":
            writes.append(Path(path))
        return original_write(mesh, path, **kwargs)

    def load(host):
        module = original_loader(host)
        original_align = module.MorphoWeaveAtlasBuilderLogic.runAlign

        def align(*args, **kwargs):
            align_directories.append((Path(args[3]), Path(args[5])))
            return original_align(*args, **kwargs)

        monkeypatch.setattr(module.MorphoWeaveAtlasBuilderLogic, "runAlign", align)
        return module

    monkeypatch.setattr(MeshIO, "write_obj", write)
    monkeypatch.setattr(morphoweave, "load_atlas_builder", load)
    with MorphoWeave.stage_inputs(specimens) as workspace:
        root = workspace.root
        before = {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
        mean = MorphoWeave.build_atlas(
            specimens, normalize_scale=normalize_scale, workspace=workspace
        )
        atlas = MorphoWeave.restore_input_scale(mean, specimens).atlas
        aligned = MorphoWeave.align_specimens(
            specimens, atlas, normalize_scale=normalize_scale, workspace=workspace
        )
        assert {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        } == before
        assert len(writes) == len(specimens)
        assert [pair[0] for pair in align_directories] == [root / "originalModels"] * 2
        assert align_directories[0][1] != align_directories[1][1]
        assert all(not pair[1].exists() for pair in align_directories)
    assert not root.exists()
    for expected, actual in zip(reference_mean.mesh, mean.mesh, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        mean.landmarks.points, reference_mean.landmarks.points
    )
    for expected, actual in zip(reference_aligned, aligned, strict=True):
        assert actual.specimen_id == expected.specimen_id
        assert actual.landmarks.labels == expected.landmarks.labels
        np.testing.assert_array_equal(
            actual.landmarks.points, expected.landmarks.points
        )
        for before, after in zip(expected.mesh, actual.mesh, strict=True):
            np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize(
    "change", ["vertices", "faces", "uv", "landmarks", "labels", "id", "order", "count"]
)
def test_staging_rejects_different_input_revision(specimens, change):
    with MorphoWeave.stage_inputs(specimens) as workspace:
        first = specimens[0]
        if change in ("vertices", "faces", "uv"):
            values = getattr(first.mesh, change).copy()
            if change == "faces":
                values = values[::-1]
            else:
                values.flat[0] += 0.01
            changed = (
                first._replace(mesh=first.mesh._replace(**{change: values})),
                *specimens[1:],
            )
        elif change == "landmarks":
            points = first.landmarks.points.copy()
            points[0, 0] += 0.01
            changed = (
                first._replace(landmarks=first.landmarks._replace(points=points)),
                *specimens[1:],
            )
        elif change == "labels":
            changed = tuple(
                item._replace(
                    landmarks=item.landmarks._replace(
                        labels=("changed", *item.landmarks.labels[1:])
                    )
                )
                for item in specimens
            )
        elif change == "id":
            changed = (first._replace(specimen_id="changed"), *specimens[1:])
        elif change == "order":
            changed = specimens[::-1]
        else:
            changed = specimens[:1]
        # Both entry points must reject a workspace from another input revision,
        # including geometrically equivalent changes to index/label ordering.
        with pytest.raises(ValueError, match="workspace does not match"):
            MorphoWeave.build_atlas(changed, workspace=workspace)
        with pytest.raises(ValueError, match="workspace does not match"):
            MorphoWeave.align_specimens(
                changed, AtlasModel(first.mesh, first.landmarks), workspace=workspace
            )


def test_workspaces_are_isolated_and_closed_workspaces_are_rejected(specimens):
    with MorphoWeave.stage_inputs(specimens) as first:
        with MorphoWeave.stage_inputs(specimens) as second:
            assert first.root != second.root
        assert not second.root.exists()
        assert first.root.is_dir()
        MorphoWeave.build_atlas(specimens, workspace=first)
    with pytest.raises(RuntimeError, match="already closed"):
        MorphoWeave.build_atlas(specimens, workspace=first)
    with pytest.raises(RuntimeError, match="already closed"):
        MorphoWeave.align_specimens(
            specimens,
            AtlasModel(specimens[0].mesh, specimens[0].landmarks),
            workspace=first,
        )


@pytest.mark.parametrize("invalid_id", ["first", "nested/name"])
def test_staging_preserves_filename_rules_and_cleans_partial_files(
    specimens, tmp_path, monkeypatch, invalid_id
):
    monkeypatch.setattr(morphoweave.tempfile, "tempdir", str(tmp_path))
    invalid = (specimens[0], specimens[1]._replace(specimen_id=invalid_id))
    with pytest.raises(ValueError, match="lowercasing|filename stem"):
        MorphoWeave.stage_inputs(invalid)
    assert list(tmp_path.iterdir()) == []


def test_staging_cleans_up_if_writing_an_input_fails(specimens, tmp_path, monkeypatch):
    monkeypatch.setattr(morphoweave.tempfile, "tempdir", str(tmp_path))
    original = MeshIO.write_obj
    calls = []

    def write(mesh, path, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("staging write failed")
        return original(mesh, path, **kwargs)

    monkeypatch.setattr(MeshIO, "write_obj", write)
    with pytest.raises(OSError, match="staging write failed"):
        MorphoWeave.stage_inputs(specimens)
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_atlas_workflow_releases_shared_inputs_if_cleanup_fails_before_alignment(
    specimens, tmp_path
):
    models, landmarks = tmp_path / "models", tmp_path / "landmarks"
    for item in specimens:
        MeshIO.write_obj(item.mesh, models / f"{item.specimen_id}.obj")
        LandmarkIO.write_json(
            item.landmarks, landmarks / f"{item.specimen_id}.mrk.json"
        )
    options = AtlasOptions(
        models, landmarks, tmp_path / "output", backend="morphoweave"
    )
    operations = Operations.registry()
    operations["blender"] = lambda inputs: None

    def fail_cleanup(inputs):
        raise ValueError("cleanup deliberately failed")

    operations["clean"] = fail_cleanup
    callbacks, events = Queue(), []
    manager = SerialExecutionManager(operations, Callbacks(callbacks.put))
    try:
        manager.submit(Workflows.atlas(options), events.append, run_id="atlas-failure")
        while not events or events[-1].kind not in ("completed", "failed", "cancelled"):
            callbacks.get(timeout=10)()
        terminal = events[-1]
        assert terminal.kind == "failed"
        assert terminal.failure.operation == "clean"
        assert terminal.failure.message == "cleanup deliberately failed"
        resources = [
            event.result
            for event in events
            if event.kind == "task_completed" and event.task_id == "morphoweave_inputs"
        ]
        assert len(resources) == 1
        assert not resources[0].root.exists()
        assert resources[0].cache.closed
        assert resources[0].cache.stats.array_bytes == 0
        assert resources[0].cache.stats.peak_array_bytes > 0
        assert all(event.task_id != "morphoweave_aligned" for event in events)
        assert len(list(models.glob("*.obj"))) == len(specimens)
    finally:
        manager.close()
        assert manager.join(10)


@pytest.mark.parametrize("close", [False, True])
def test_atlas_workflow_releases_staging_on_cancel_or_close(specimens, tmp_path, close):
    models, landmarks = tmp_path / "models", tmp_path / "landmarks"
    for item in specimens:
        MeshIO.write_obj(item.mesh, models / f"{item.specimen_id}.obj")
        LandmarkIO.write_json(
            item.landmarks, landmarks / f"{item.specimen_id}.mrk.json"
        )
    options = AtlasOptions(
        models, landmarks, tmp_path / "output", backend="morphoweave"
    )
    entered, release = Event(), Event()
    operations = Operations.registry()
    operations["blender"] = lambda inputs: None

    def wait_at_cleanup(inputs):
        entered.set()
        assert release.wait(10)
        return inputs.arguments[0]

    # Stop between real upstream mean construction and final alignment. This
    # exercises the application lifetime, not just a standalone with-statement.
    operations["clean"] = wait_at_cleanup
    callbacks, events = Queue(), []
    manager = SerialExecutionManager(operations, Callbacks(callbacks.put))
    try:
        manager.submit(Workflows.atlas(options), events.append, run_id="atlas-cancel")
        assert entered.wait(10)
        while not events or events[-1].task_id != "clean":
            callbacks.get(timeout=10)()
        workspace = next(
            event.result
            for event in events
            if event.kind == "task_completed" and event.task_id == "morphoweave_inputs"
        )
        if close:
            manager.close()
        else:
            manager.cancel("atlas-cancel")
        assert workspace.root.is_dir()
        release.set()
        if close:
            assert manager.join(10)
        else:
            while events[-1].kind not in ("completed", "failed", "cancelled"):
                callbacks.get(timeout=10)()
            assert events[-1].kind == "cancelled"
        assert not workspace.root.exists()
        assert workspace.cache.closed
        assert workspace.cache.stats.array_bytes == 0
        assert workspace.cache.stats.peak_array_bytes > 0
        assert all(event.task_id != "morphoweave_aligned" for event in events)
    finally:
        release.set()
        manager.close()
        assert manager.join(10)


@pytest.mark.parametrize("budget", [0, 1, 384, 8192])
@pytest.mark.parametrize("normalize_scale", [False, True])
def test_bounded_cache_matches_disabled_upstream_workflow(
    specimens, budget, normalize_scale
):
    def run(limit):
        with MorphoWeave.stage_inputs(specimens, cache_budget_bytes=limit) as workspace:
            mean = MorphoWeave.build_atlas(
                specimens, workspace=workspace, normalize_scale=normalize_scale
            )
            atlas = MorphoWeave.restore_input_scale(mean, specimens).atlas
            aligned = MorphoWeave.align_specimens(
                specimens, atlas, workspace=workspace, normalize_scale=normalize_scale
            )
            stats = workspace.cache.stats
            assert stats.array_bytes <= limit
        assert workspace.cache.closed
        assert workspace.cache.stats.array_bytes == 0
        return mean, aligned, stats

    baseline, expected, _ = run(0)
    mean, aligned, stats = run(budget)
    for left, right in zip(baseline.mesh, mean.mesh, strict=True):
        if left is None or right is None:
            assert left is right
        else:
            assert left.dtype == right.dtype
            assert left.tobytes() == right.tobytes()
    np.testing.assert_array_equal(baseline.landmarks.points, mean.landmarks.points)
    assert baseline.landmarks.labels == mean.landmarks.labels
    for before, after in zip(expected, aligned, strict=True):
        for left, right in zip(before.mesh, after.mesh, strict=True):
            assert left.dtype == right.dtype
            assert left.tobytes() == right.tobytes()
        assert before.landmarks.labels == after.landmarks.labels
        assert before.landmarks.points.tobytes() == after.landmarks.points.tobytes()
    if budget == 8192:
        assert stats.hits == 3
        assert stats.entries == len(specimens)
        assert stats.bypasses == len(specimens)
    elif budget < 384:
        assert stats.hits == stats.peak_array_bytes == 0


def test_invalid_cache_budget_cleans_workspace(specimens, tmp_path, monkeypatch):
    monkeypatch.setattr(morphoweave.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(ValueError, match="budget"):
        MorphoWeave.stage_inputs(specimens, cache_budget_bytes=-1)
    assert list(tmp_path.iterdir()) == []
