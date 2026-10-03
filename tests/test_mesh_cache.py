"""Cache reuse must preserve file validation, precision, and upstream isolation."""

import os
import weakref
from pathlib import Path

import numpy as np
import pytest
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from interdeca._mesh_cache import MeshCache
from interdeca._morphoweave_host import CORNER_UV, Host, load_atlas_builder
from interdeca.io import MeshIO
from interdeca.models import Mesh


@pytest.fixture
def files(tmp_path):
    mesh = Mesh(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]]),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
        np.arange(24, dtype=float).reshape(4, 3, 2) / 24,
    ).validated()
    originals = tmp_path / "originals"
    paths = tuple(originals / f"{name}.obj" for name in ("a", "b", "c"))
    for index, path in enumerate(paths):
        MeshIO.write_obj(mesh._replace(vertices=mesh.vertices + index), path)
    return mesh, paths


def array_bytes(mesh):
    return sum(array.nbytes for array in mesh if array is not None)


def equal_mesh(actual, expected):
    for left, right in zip(actual, expected, strict=True):
        if left is None or right is None:
            assert left is right
        else:
            assert left.dtype == right.dtype
            assert left.tobytes() == right.tobytes()


def test_repeated_loads_use_owned_parsed_arrays_and_mutable_nodes_stay_private(files):
    mesh, paths = files
    cache = MeshCache(paths[0].parent, array_bytes(mesh))
    host = Host((), cache)
    first = host.loadModel(paths[0])
    points = vtk_to_numpy(first.GetPolyData().GetPoints().GetData())
    points[:] = 42
    uv = vtk_to_numpy(first.GetPolyData().GetCellData().GetArray(CORNER_UV))
    uv[:] = -2
    second = host.loadModel(paths[0])
    equal_mesh(Host.mesh(second.GetPolyData()), MeshIO.read(paths[0]))
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.array_bytes == array_bytes(mesh)
    assert cache.stats.metadata_bytes > 0
    assert first.GetPolyData() is not second.GetPolyData()
    cache.close()
    # Closing the run releases the cache, while already returned VTK handles own
    # their buffers and remain valid for the caller that retained them.
    equal_mesh(Host.mesh(second.GetPolyData()), MeshIO.read(paths[0]))
    assert cache.stats.array_bytes == cache.stats.entries == 0
    assert cache.stats.peak_array_bytes == array_bytes(mesh)
    with pytest.raises(RuntimeError, match="closed"):
        cache.read(paths[0])


@pytest.mark.parametrize("budget", [0, 1])
def test_disabled_and_oversized_entries_use_disk_and_do_not_retain_buffers(
    files, budget
):
    _, paths = files
    cache = MeshCache(paths[0].parent, budget)
    first, second = cache.read(paths[0]), cache.read(paths[0])
    equal_mesh(first, second)
    assert first is not second
    assert cache.stats.hits == cache.stats.entries == cache.stats.peak_array_bytes == 0
    assert cache.stats.bypasses == cache.stats.misses == 2


def test_eviction_respects_recency_and_retains_live_callers(files):
    mesh, paths = files
    size = array_bytes(mesh)
    cache = MeshCache(paths[0].parent, 2 * size)
    a, b = cache.read(paths[0]), cache.read(paths[1])
    assert cache.read(paths[0]) is a
    cache.read(paths[2])
    assert cache.stats.evictions == 1
    assert cache.stats.peak_array_bytes == 2 * size
    assert cache.read(paths[0]) is a
    assert cache.read(paths[1]) is not b
    equal_mesh(b, MeshIO.read(paths[1]))
    assert cache.stats.array_bytes <= cache.budget_bytes


def test_intermediates_do_not_evict_reusable_originals(files, tmp_path):
    mesh, paths = files
    other = tmp_path / "aligned" / "a.obj"
    MeshIO.write_obj(mesh, other)
    cache = MeshCache(paths[0].parent, array_bytes(mesh))
    first = cache.read(paths[0])
    equal_mesh(cache.read(other), mesh)
    assert cache.read(paths[0]) is first
    assert cache.stats.bypasses == 1
    assert cache.stats.evictions == 0


@pytest.mark.parametrize("replacement", [False, True])
def test_file_version_detects_overwrite_with_preserved_mtime(files, replacement):
    mesh, paths = files
    path = paths[0]
    cache = MeshCache(path.parent, 4 * array_bytes(mesh))
    first = cache.read(path)
    old = path.stat()
    content = path.read_text().replace("v 0 0 0\n", "v 1 0 0\n")
    assert len(content.encode()) == old.st_size
    if replacement:
        alternate = path.with_suffix(".replacement")
        alternate.write_text(content)
        alternate.replace(path)
    else:
        path.write_text(content)
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    second = cache.read(path)
    assert second is not first
    assert second.vertices[0, 0] == 1
    assert first.vertices[0, 0] == 0
    assert cache.stats.invalidations == 1


def test_conversion_settings_and_strict_geometry_validation_are_part_of_identity(files):
    mesh, paths = files
    cache = MeshCache(paths[0].parent, 4 * array_bytes(mesh))
    ras, lps = cache.read(paths[0]), cache.read(paths[0], coordinate_system="LPS")
    np.testing.assert_array_equal(lps.vertices[:, :2], -ras.vertices[:, :2])
    assert cache.read(paths[0]) is ras
    assert cache.read(paths[0], coordinate_system="LPS") is lps
    collapsed = paths[0].parent / "collapsed.obj"
    MeshIO.write_obj(
        mesh._replace(vertices=np.zeros_like(mesh.vertices)),
        collapsed,
        allow_degenerate=True,
    )
    cache.read(collapsed, allow_degenerate=True)
    with pytest.raises(ValueError):
        cache.read(collapsed)
    with pytest.raises(ValueError, match="coordinate_system"):
        cache.read(paths[0], coordinate_system="wrong")


def test_host_saves_invalidate_before_success_or_failure(files, monkeypatch):
    mesh, paths = files
    cache = MeshCache(paths[0].parent, 4 * array_bytes(mesh))
    host = Host((), cache)
    node = host.loadModel(paths[0])
    vtk_to_numpy(node.GetPolyData().GetPoints().GetData())[:] += 0.25
    host.saveNode(node, paths[0])
    assert cache.stats.invalidations == 1
    assert cache.stats.write_admissions == 1
    equal_mesh(cache.read(paths[0]), Host.mesh(node.GetPolyData()))

    def fail(*args, **kwargs):
        raise OSError("save failed")

    monkeypatch.setattr(MeshIO, "write_obj", fail)
    with pytest.raises(OSError, match="save failed"):
        host.saveNode(node, paths[0])
    assert cache.stats.entries == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("textured", [False, True])
def test_written_mesh_matches_real_obj_roundtrip_and_owns_input(
    files, dtype, textured, monkeypatch
):
    mesh, paths = files
    vertices = mesh.vertices.astype(dtype)
    vertices[0] = [-0.0, np.nextafter(dtype(0), dtype(1)), -0.0]
    vertices[1, 0] = np.nextafter(dtype(2), dtype(3))
    # Extreme finite UVs exercise decimal precision without introducing huge or
    # collapsed geometric triangles; repeated vertices keep distinct UV corners.
    uv = np.arange(48, dtype=dtype).reshape(4, 3, 4)[..., ::2]
    uv[0, 0] = [-0.0, np.nextafter(dtype(0), dtype(1))]
    uv[0, 1] = [np.finfo(dtype).max, np.finfo(dtype).tiny]
    mesh = Mesh(vertices, mesh.faces[::-1], uv if textured else None)
    path = paths[0].parent.parent / "aligned" / "saved.obj"
    cache = MeshCache(paths[0].parent, 8192)
    cache.write_obj(mesh, path)
    expected = MeshIO.read(path)
    assert path in tuple(path.parent.glob("*.obj"))
    assert cache.stats.write_admissions == 1
    # A cache result is owned even when the writer received writable arrays and
    # strided views. The fully serialized file remains readable independently.
    vertices[:] = 17
    uv[:] = 19

    def unexpected_parse(*args, **kwargs):
        raise AssertionError("A saved mesh should cross this boundary in memory")

    monkeypatch.setattr(MeshIO, "read", unexpected_parse)
    equal_mesh(cache.read(path), expected)
    for array in cache.read(path):
        if array is not None:
            assert not array.flags.writeable
    first = Host((), cache).loadModel(path)
    vtk_to_numpy(first.GetPolyData().GetPoints().GetData())[:] = 23
    second = Host((), cache).loadModel(path)
    equal_mesh(Host.mesh(second.GetPolyData()), expected)


def test_saved_mesh_settings_versions_and_failed_overwrites(files, monkeypatch):
    mesh, paths = files
    cache = MeshCache(paths[0].parent, 8192)
    path = paths[0].parent.parent / "aligned" / "saved.obj"
    cache.write_obj(mesh, path)
    equal_mesh(cache.read(path, coordinate_system="LPS"), MeshIO.read(path, coordinate_system="LPS"))
    assert cache.stats.hits == 0
    assert cache.stats.misses == 1
    with pytest.raises(FileExistsError):
        cache.write_obj(mesh, path)
    assert cache.stats.entries == 0
    cache.write_obj(mesh, path, overwrite=True)
    path.write_text("v 0 0 0\nf 1 1 1\n")
    with pytest.raises(ValueError):
        cache.read(path)
    assert cache.stats.entries == 0
    cache.write_obj(mesh, path, overwrite=True)

    def fail(*args, **kwargs):
        raise OSError("partial write")

    monkeypatch.setattr(MeshIO, "write_obj", fail)
    with pytest.raises(OSError, match="partial write"):
        cache.write_obj(mesh, path, overwrite=True)
    assert cache.stats.entries == 0


@pytest.mark.parametrize("budget", [0, 1, 384, 8192])
def test_written_entries_share_budget_and_fall_back_to_disk(files, budget):
    mesh, paths = files
    cache = MeshCache(paths[0].parent, budget)
    saved = tuple(paths[0].parent.parent / "aligned" / path.name for path in paths)
    for path in (*paths, *saved):
        cache.write_obj(mesh, path, overwrite=True)
        assert cache.stats.array_bytes <= budget
    for path in (*paths, *saved):
        equal_mesh(cache.read(path), MeshIO.read(path))
    if budget < array_bytes(mesh):
        assert cache.stats.write_admissions == 0
        assert cache.stats.write_bypasses == 6
    elif budget == array_bytes(mesh):
        assert cache.stats.evictions > 0
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        cache.write_obj(mesh, saved[0], overwrite=True)


@pytest.mark.parametrize("fail", [False, True])
def test_operation_entries_release_before_directory_cleanup(files, fail):
    mesh, paths = files
    cache = MeshCache(paths[0].parent, 8192)
    cache.read(paths[0])
    directory = paths[0].parent.parent / "aligned"
    try:
        with cache.forget_after(directory):
            cache.write_obj(mesh, directory / "saved.obj")
            parsed = cache.read(directory / "saved.obj")
            references = [weakref.ref(array) for array in parsed]
            del parsed
            if fail:
                raise ValueError("operation failed")
    except ValueError:
        assert fail
    assert all(reference() is None for reference in references)
    assert cache.stats.entries == 1
    assert cache.stats.array_bytes == array_bytes(mesh)
    assert (directory / "saved.obj").is_file()


def test_written_readonly_alias_isolated_and_external_paths_not_retained(files, tmp_path):
    mesh, paths = files
    base = mesh.vertices.copy()
    view = base.view()
    view.setflags(write=False)
    cache = MeshCache(paths[0].parent, 8192)
    saved = paths[0].parent / "saved.obj"
    cache.write_obj(mesh._replace(vertices=view), saved)
    base[:] = -7
    equal_mesh(cache.read(saved), mesh)
    external = tmp_path.parent / (tmp_path.name + "-outside.obj")
    try:
        cache.write_obj(mesh, external)
        equal_mesh(cache.read(external), mesh)
        assert cache.stats.write_bypasses == 1
        assert cache.stats.entries == 1
    finally:
        external.unlink(missing_ok=True)


def test_saved_float32_vtk_scratch_reloads_as_exact_owned_float64(files):
    mesh, paths = files
    cache = MeshCache(paths[0].parent, 8192)
    host = Host((), cache)
    scratch = host.loadModel(paths[0])
    points = mesh.vertices.astype(np.float32)
    points[0, 0] = -0.0
    points[1, 0] = np.nextafter(np.float32(2), np.float32(3))
    scratch.GetPolyData().GetPoints().SetData(numpy_to_vtk(points, deep=True))
    path = paths[0].parent.parent / "aligned" / "scratch.obj"
    host.saveNode(scratch, path)
    expected = MeshIO.read(path)
    vtk_to_numpy(scratch.GetPolyData().GetPoints().GetData())[:] = 43
    loaded = host.loadModel(path)
    actual = Host.mesh(loaded.GetPolyData())
    assert actual.vertices.dtype == np.float64
    assert np.signbit(actual.vertices[0, 0])
    equal_mesh(actual, expected)


@pytest.mark.parametrize("field", ["vertices", "faces", "uv"])
def test_invalid_saved_mesh_cannot_create_a_file_or_entry(files, field):
    mesh, paths = files
    values = getattr(mesh, field).copy()
    values.flat[0] = len(mesh.vertices) if field == "faces" else np.nan
    cache = MeshCache(paths[0].parent, 8192)
    path = paths[0].parent / "invalid.obj"
    with pytest.raises(ValueError):
        cache.write_obj(mesh._replace(**{field: values}), path)
    assert not path.exists()
    assert cache.stats.entries == 0


@pytest.mark.parametrize("fail", [False, True])
def test_host_scope_releases_nodes_without_closing_borrowed_cache(files, fail):
    mesh, paths = files
    cache = MeshCache(paths[0].parent, 8192)
    try:
        with Host((), cache) as host:
            upstream = load_atlas_builder(host)
            node = host.loadModel(paths[0])
            node_reference = weakref.ref(node)
            surface_reference = weakref.ref(node.GetPolyData())
            result = Host.mesh(node.GetPolyData())
            del node
            if fail:
                raise ValueError("host operation failed")
    except ValueError as error:
        assert fail and str(error) == "host operation failed"
    # Keep the upstream module and its host import context alive deliberately.
    # Explicit node cleanup must work without waiting for cyclic GC.
    assert upstream.MorphoWeaveAtlasBuilderLogic is not None
    assert not host.nodes and host.cache is None
    assert node_reference() is None
    assert surface_reference() is None
    assert not cache.closed
    equal_mesh(cache.read(paths[0]), mesh)
    cache.close()
    equal_mesh(result, mesh)


def test_changed_during_parse_is_not_cached_and_missing_files_are_not_hits(
    files, monkeypatch
):
    mesh, paths = files
    cache = MeshCache(paths[0].parent, array_bytes(mesh))
    original = MeshIO.read

    def change(path, **kwargs):
        result = original(path, **kwargs)
        Path(path).write_text(Path(path).read_text() + "# changed\n")
        return result

    monkeypatch.setattr(MeshIO, "read", change)
    cache.read(paths[0])
    assert cache.stats.entries == 0
    monkeypatch.setattr(MeshIO, "read", original)
    cache.read(paths[0])
    paths[0].unlink()
    with pytest.raises(FileNotFoundError):
        cache.read(paths[0])


def test_close_releases_all_cached_arrays(files):
    mesh, paths = files
    cache = MeshCache(paths[0].parent, 3 * array_bytes(mesh))
    references = []
    for path in paths:
        parsed = cache.read(path)
        references.extend(weakref.ref(array) for array in parsed if array is not None)
    del parsed
    assert all(reference() is not None for reference in references)
    cache.close()
    assert all(reference() is None for reference in references)
    cache.close()


def test_tiny_meshes_cannot_accumulate_unbounded_metadata(files):
    mesh, paths = files
    cache = MeshCache(paths[0].parent, 1024 * array_bytes(mesh))
    # A generous byte budget still has an entry ceiling, so Python/path metadata
    # stays bounded when a population contains many very small meshes.
    for index in range(257):
        path = paths[0].parent / f"tiny-{index}.obj"
        MeshIO.write_obj(mesh, path)
        cache.read(path)
    assert cache.stats.entries == cache.stats.peak_entries == 256
    assert cache.stats.evictions == 1
    assert cache.stats.peak_array_bytes <= cache.budget_bytes
    assert cache.stats.peak_metadata_bytes > 0
    cache.close()
    assert cache.stats.entries == cache.stats.array_bytes == 0


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_budgets_are_rejected(files, budget):
    _, paths = files
    with pytest.raises(ValueError, match="budget"):
        MeshCache(paths[0].parent, budget)
