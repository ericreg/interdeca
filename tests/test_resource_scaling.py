"""Bounded accounting experiments, not timings or large-population benchmarks."""

import hashlib
import weakref
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from PIL import Image

from interdeca._mesh_cache import MeshCache
from interdeca.io import MeshIO, TextureIO
from interdeca.models import Mesh
from interdeca.serde import TexturePixelsData


class MeshFiles(NamedTuple):
    meshes: tuple[Mesh, ...]
    paths: tuple[Path, ...]
    bytes_per_mesh: int
    hashes: tuple[str, ...]


class ImageBuffers(NamedTuple):
    images: int
    payload_bytes: int


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_bytes(mesh):
    return sum(array.nbytes for array in mesh if array is not None)


def equal_mesh(actual, expected):
    # Byte/order equality includes the UV assigned to each triangle corner.
    for left, right in zip(actual, expected, strict=True):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes() == right.tobytes()


def mesh_files(root, count):
    # A fixed 17 by 17 grid gives 289 vertices and 512 nondegenerate triangles.
    # Translations distinguish specimens without changing their storage sizes.
    x, y = np.meshgrid(np.arange(17, dtype=float), np.arange(17, dtype=float))
    vertices = np.column_stack((x.ravel(), y.ravel(), 0.1 * np.sin(x.ravel())))
    faces = np.array([
        triangle
        for row in range(16)
        for column in range(16)
        for base in (row * 17 + column,)
        for triangle in ((base, base + 1, base + 17),
                         (base + 1, base + 18, base + 17))
    ])
    uv = vertices[faces, :2] / 16
    meshes = tuple(
        Mesh(vertices + [index, 0, 0], faces, uv).validated()
        for index in range(count)
    )
    paths = tuple(root / f"synthetic-{index}.obj" for index in range(count))
    for mesh, path in zip(meshes, paths, strict=True):
        MeshIO.write_obj(mesh, path)
    return MeshFiles(meshes, paths, array_bytes(meshes[0]), tuple(map(checksum, paths)))


@pytest.mark.parametrize("count", [1, 4, 16])
def test_population_arrays_grow_with_count_when_every_mesh_fits(tmp_path, count):
    data = mesh_files(tmp_path / "originals", count)
    # Hold the budget and mesh density fixed while changing specimen count.
    # Caller-owned inputs and separately parsed cache buffers are distinct costs.
    cache = MeshCache(data.paths[0].parent, 16 * data.bytes_per_mesh)
    try:
        for path, mesh in zip(data.paths, data.meshes, strict=True):
            parsed = cache.read(path)
            equal_mesh(parsed, mesh)
            assert not np.shares_memory(parsed.vertices, mesh.vertices)
        expected = count * data.bytes_per_mesh
        assert sum(map(array_bytes, data.meshes)) == expected
        assert cache.stats.array_bytes == cache.stats.peak_array_bytes == expected
        assert cache.stats.entries == count
        assert cache.stats.evictions == 0
        assert tuple(map(checksum, data.paths)) == data.hashes
        print(f"SCALING population specimens={count} input_bytes={expected} "
              f"cache_peak_bytes={cache.stats.peak_array_bytes}")
    finally:
        cache.close()
    assert cache.stats.array_bytes == cache.stats.entries == 0


@pytest.mark.parametrize("capacity", [0, 1, 2, 4, 8])
def test_cache_pressure_changes_reuse_not_geometry(tmp_path, capacity):
    data = mesh_files(tmp_path / "originals", 8)
    cache = MeshCache(data.paths[0].parent, capacity * data.bytes_per_mesh)
    try:
        # A reverse second sweep visits the retained suffix first. It gives an
        # exact hit/eviction expectation independent of filesystem-cache speed.
        for order in (range(8), reversed(range(8))):
            for index in order:
                equal_mesh(cache.read(data.paths[index]), data.meshes[index])
                assert cache.stats.array_bytes <= cache.budget_bytes
        assert cache.stats.hits == capacity
        assert cache.stats.misses == 16 - capacity
        assert cache.stats.evictions == (2 * (8 - capacity) if capacity else 0)
        assert cache.stats.peak_array_bytes == capacity * data.bytes_per_mesh
        assert tuple(map(checksum, data.paths)) == data.hashes
        print(f"SCALING pressure specimens=8 capacity={capacity} "
              f"cache_peak_bytes={cache.stats.peak_array_bytes} "
              f"hits={cache.stats.hits} misses={cache.stats.misses} "
              f"evictions={cache.stats.evictions}")
    finally:
        cache.close()
    assert cache.stats.array_bytes == cache.stats.entries == 0


def test_cache_budget_does_not_bound_caller_owned_population(tmp_path):
    data = mesh_files(tmp_path / "originals", 8)
    cache = MeshCache(data.paths[0].parent, 2 * data.bytes_per_mesh)
    # A population algorithm can keep all returned meshes even when the adapter
    # evicts most entries. A cache budget therefore is not a process-memory cap.
    try:
        retained = tuple(cache.read(path) for path in data.paths)
        assert cache.stats.array_bytes == 2 * data.bytes_per_mesh
        assert sum(map(array_bytes, retained)) == 8 * data.bytes_per_mesh
    finally:
        cache.close()
    assert cache.stats.array_bytes == cache.stats.entries == 0
    for actual, expected in zip(retained, data.meshes, strict=True):
        equal_mesh(actual, expected)
    print(f"SCALING caller retained_bytes={sum(map(array_bytes, retained))} "
          f"cache_budget_bytes={cache.budget_bytes} cache_after_close=0")


@pytest.mark.parametrize("count,edge", [
    (1, 256), (4, 256), (16, 256), (4, 128), (4, 512), (4, 1024),
])
def test_source_image_buffers_scale_with_area_not_sequential_file_count(
    tmp_path, monkeypatch, count, edge
):
    paths = tuple(tmp_path / f"synthetic-{index}.tiff" for index in range(count))
    for index, path in enumerate(paths):
        with Image.new("RGBA", (edge, edge), (index, 32, 64, 255)) as image:
            image.save(path, compression="tiff_lzw")
    hashes = tuple(map(checksum, paths))
    references, observations = [], []
    convert = Image.Image.convert

    def image_buffers():
        # Count logical decoded byte-image payloads, not Python metadata, decoder
        # scratch storage, allocator retention, or total RSS. Closed wrappers may
        # remain alive, but they must no longer expose a native pixel buffer.
        sizes = []
        for reference in references:
            image = reference()
            if image is not None:
                try:
                    image.getpixel((0, 0))
                except ValueError:
                    continue
                sizes.append(image.width * image.height * len(image.getbands()))
        return ImageBuffers(len(sizes), sum(sizes))

    def observed(image, *args, **kwargs):
        result = convert(image, *args, **kwargs)
        references.extend((weakref.ref(image), weakref.ref(result)))
        observations.append(image_buffers())
        return result

    def reject_float_colors(*args, **kwargs):
        pytest.fail("Source validation allocated numerical texture colors")

    monkeypatch.setattr(Image.Image, "convert", observed)
    monkeypatch.setattr(TexturePixelsData, "from_rgba8", reject_float_colors)
    for path in paths:
        assert TextureIO.validate_file(path) is None
        assert image_buffers() == ImageBuffers(0, 0)
    assert len(observations) == count
    assert all(value == ImageBuffers(2, 2 * edge * edge * 4) for value in observations)
    assert tuple(map(checksum, paths)) == hashes
    print(f"SCALING images files={count} edge={edge} "
          f"peak_decoded_payload_bytes={max(value.payload_bytes for value in observations)} "
          "payload_after_each_file=0")
