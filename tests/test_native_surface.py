"""Analytical contracts and saved Blender oracles for the native surface kernels."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from interdeca.blender import Blender
from interdeca.models import AtlasModel, Landmarks, Mesh, ResampledSpecimen, Specimen
from interdeca.surface import MeshProcessing, TextureBaking


@pytest.fixture
def square():
    # Two triangles share the diagonal, testing consistent texel ownership.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return Mesh(vertices, faces, vertices[faces, :2]).validated()


@pytest.fixture
def image():
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[:, :, :3] = [31, 127, 211]
    pixels[:, :, 3] = 255
    return pixels


def test_cleanup_zero_preserves_record(square):
    result = MeshProcessing.clean(square, merge_distance=0)
    assert result.mesh is square
    np.testing.assert_array_equal(result.vertex_map, np.arange(4))
    assert not result.vertex_map.flags.writeable


def test_cleanup_maps_match_blender_cluster_survivor(square):
    # Captured Blender 5.2.1 keeps vertex 4 here, then reuses its existing
    # triangle. Input order alone does not determine which vertex survives.
    mesh = Mesh(np.vstack([square.vertices, [0.00001, 0, 0], [9, 9, 9]]),
                np.vstack([square.faces, [4, 1, 2]]))
    result = MeshProcessing.clean(mesh)
    expected_vertices = mesh.vertices[[1, 2, 3, 4]].astype(np.float32).astype(np.float64)
    np.testing.assert_array_equal(result.mesh.vertices, expected_vertices)
    np.testing.assert_array_equal(result.mesh.faces, [[3, 1, 2], [3, 0, 1]])
    np.testing.assert_array_equal(result.vertex_map, [3, 0, 1, 2, 3, -1])
    np.testing.assert_array_equal(result.face_map, [1, 0, 1])
    assert result.mesh.uv is None


def test_cleanup_is_not_transitive():
    vertices = np.array([[0, 0, 0], [.75, 0, 0], [1.5, 0, 0], [0, 5, 0], [0, 0, 5]], float)
    mesh = Mesh(vertices, np.array([[0, 3, 4], [1, 3, 4], [2, 3, 4]], np.int64))
    result = MeshProcessing.clean(mesh, merge_distance=1)
    assert result.vertex_map[0] == result.vertex_map[1]
    assert result.vertex_map[0] != result.vertex_map[2]


def test_cleanup_blender_weighted_centroid_fixture():
    # This seven-point chain was the counterexample to independent greedy
    # welding. Blender's tree order and centroid weighting retain its midpoint.
    v = np.array([[x, 0, 0] for x in (0, .2, .4, .6, .8, 1, 1.2)] + [[0, 5, 0], [0, 0, 5]])
    f = np.array([[i, 7, 8] for i in range(7)], np.int64)
    result = MeshProcessing.clean(Mesh(v, f), merge_distance=1)
    np.testing.assert_array_equal(result.mesh.vertices, v[[3, 7, 8]].astype(np.float32).astype(np.float64))
    np.testing.assert_array_equal(result.mesh.faces, [[0, 1, 2]])
    np.testing.assert_array_equal(result.vertex_map, [0, 0, 0, 0, 0, 0, 0, 1, 2])


@pytest.mark.parametrize("distance", [-1, np.nan, np.inf])
def test_cleanup_bad_distance(square, distance):
    with pytest.raises(ValueError):
        MeshProcessing.clean(square, merge_distance=distance)


def test_cleanup_empty_surface(square):
    with pytest.raises(ValueError, match="no valid surface"):
        MeshProcessing.clean(square, merge_distance=2)


def test_uv_plane_and_resolution_independence(square):
    layout = MeshProcessing.uv_layout(square, resolution=(64, 64), margin_pixels=2)
    assert len(set(layout.chart_ids)) == 1
    assert layout.uv.min() == 0
    assert layout.uv.max() <= 1
    np.testing.assert_array_equal(layout.uv[0, 0], layout.uv[1, 0])
    np.testing.assert_array_equal(layout.uv[0, 2], layout.uv[1, 1])
    assert not layout.uv.flags.writeable
    np.testing.assert_array_equal(layout.uv, MeshProcessing.uv_layout(square, resolution=(64, 64), margin_pixels=2).uv)
    np.testing.assert_array_equal(layout.uv, MeshProcessing.uv_layout(square, resolution=(512, 512), margin_pixels=0).uv)


def test_uv_overlap_splits_charts(square):
    # Coincident disconnected surfaces must occupy distinct parts of the tile.
    mesh = Mesh(np.vstack([square.vertices, square.vertices]), np.vstack([square.faces, square.faces + 4]))
    layout = MeshProcessing.uv_layout(mesh, resolution=(64, 64))
    assert len(set(layout.chart_ids)) == 2
    boxes = [(layout.uv[:2].min(axis=(0, 1)), layout.uv[:2].max(axis=(0, 1))),
             (layout.uv[2:].min(axis=(0, 1)), layout.uv[2:].max(axis=(0, 1)))]
    assert np.any(boxes[0][1] <= boxes[1][0]) or np.any(boxes[1][1] <= boxes[0][0])


@pytest.mark.parametrize("case", ["square", "triangle", "disconnected", "tetrahedron", "strip", "translated_strip", "nonmanifold", "degenerate"])
def test_smart_uv_matches_saved_blender_corners(case):
    # These independent references were exported by the pinned Blender binary,
    # including its area sorting, island traversal, packing and bounds scaling.
    path = Path(__file__).parent / "fixtures/blender_5_2_1/uv" / f"{case}.npz"
    with np.load(path, allow_pickle=False) as archive:
        mesh = Mesh(archive["vertices"], archive["faces"])
        expected = archive["uv"]
    layout = MeshProcessing.uv_layout(mesh)
    np.testing.assert_array_equal(layout.uv, expected)
    if case == "strip":
        assert len(set(layout.chart_ids)) == 3
    assert layout.uv.dtype == np.float64
    assert not layout.uv.flags.writeable


def test_bake_ray_misses_are_removed_before_padding(square):
    # Half the source is missing. Keeping the full target's coverage mask would
    # block Blender's preliminary gap filling and produce a different image.
    source = Mesh(square.vertices, square.faces[:1], square.uv[:1])
    pixels = np.full((16,16,4), 255, np.uint8)
    pixels[:,:,0] = np.arange(16)[None,:] * 16
    pixels[:,:,1] = np.arange(16)[:,None] * 16
    pixels[:,:,2] = 37
    observed = TextureBaking.bake_pixels(source, square, pixels, resolution=(16,16), cage_extrusion=.01)
    with Image.open(Path(__file__).parent / "fixtures/blender_5_2_1/partial_source.png") as image:
        expected = np.asarray(image.convert("RGBA"))
    np.testing.assert_array_equal(observed, expected)


def test_bake_subpixel_triangle_does_not_invent_coverage(square, image):
    target = square._replace(uv=square.uv * .02 + .49)
    result = TextureBaking.bake_pixels(square, target, image, resolution=(16, 16), margin_pixels=0)
    assert not result.any()


def test_bake_constant_and_black(square, image):
    result = TextureBaking.bake_pixels(square, square, image, resolution=(16, 16), margin_pixels=0)
    np.testing.assert_array_equal(result, image)
    assert not result.flags.writeable
    image[:, :, :3] = 0
    result = TextureBaking.bake_pixels(square, square, image, resolution=(16, 16), margin_pixels=0)
    np.testing.assert_array_equal(result, image)


def test_bake_source_alpha_is_not_shader(square, image):
    image[:, :, 3] = 0
    result = TextureBaking.bake_pixels(square, square, image, resolution=(16, 16), margin_pixels=0)
    assert np.all(result[:, :, 3] == 255)
    # The source loader associates RGB with alpha, but the diffuse material has
    # no transparency shader: successful projections still have opaque coverage.
    assert not result[:, :, :3].any()


def test_bake_source_alpha_associated_in_byte_storage(square, image):
    image[:, :, 3] = 128
    result = TextureBaking.bake_pixels(square, square, image, resolution=(16,16), margin_pixels=0)
    expected = (image[:, :, :3].astype(np.uint32) * 128 // 255).astype(np.uint8)
    np.testing.assert_array_equal(result[:, :, :3], expected)
    assert np.all(result[:, :, 3] == 255)


def test_bake_shared_vertex_uses_blender_bvh_tie_order(square):
    # Both source faces intersect the collapsed target at vertex zero. Blender
    # visits face 1 first; its vertex nudge selects a visibly different pixel.
    pixels = np.full((16,1024,4),255,np.uint8)
    pixels[:,:,:3] = np.linspace(0,255,1024).astype(np.uint8)[None,:,None]
    target = square._replace(vertices=np.zeros_like(square.vertices))
    observed = TextureBaking.bake_pixels(square,target,pixels,resolution=(16,16),cage_extrusion=.01,margin_pixels=0)
    with Image.open(Path(__file__).parent / "fixtures/blender_5_2_1/vertex_sample.png") as image:
        expected = np.asarray(image.convert("RGBA"))
    np.testing.assert_array_equal(observed,expected)


def test_bake_exact_half_byte_rounds_up(square):
    # Constant source UVs isolate the exact 0.5-byte sample from raster offsets.
    # The Blender 5.2.1 reference produces byte one, not ties-to-even byte zero.
    source=square._replace(uv=np.full_like(square.uv,.5))
    pixels=np.array([[[0,0,0,255],[1,1,1,255]]],np.uint8)
    observed=TextureBaking.bake_pixels(source,square,pixels,resolution=(1,1),cage_extrusion=.01,margin_pixels=0)
    np.testing.assert_array_equal(observed,[[[1,1,1,255]]])


def test_bake_maximum_ray_distance_is_strict(square, image):
    # A power-of-two cutoff represents the same distance exactly in float32.
    missed = TextureBaking.bake_pixels(square, square, image, resolution=(8,8), cage_extrusion=.125, max_ray_distance=.125, margin_pixels=0)
    assert not missed.any()
    reached = TextureBaking.bake_pixels(square, square, image, resolution=(8,8), cage_extrusion=.125,
        max_ray_distance=float(np.nextafter(np.float32(.125), np.float32(1))), margin_pixels=0)
    assert np.all(reached[:, :, 3] == 255)


def test_bake_gradient_orientation(square):
    image = np.full((8, 8, 4), 255, dtype=np.uint8)
    image[:, :, 0] = np.arange(8, dtype=np.uint8)[None, :] * 30
    image[:, :, 1] = np.arange(8, dtype=np.uint8)[:, None] * 30
    result = TextureBaking.bake_pixels(square, square, image, resolution=(8, 8), margin_pixels=0)
    np.testing.assert_array_equal(result, image)


def test_bake_blender_byte_image_filtering(square):
    image = np.array([[[0, 0, 0, 255], [255, 255, 255, 255]]], np.uint8)
    result = TextureBaking.bake_pixels(square, square, image, resolution=(1, 1), margin_pixels=0)
    np.testing.assert_array_equal(result[0, 0], [128, 128, 128, 255])


def test_bake_repeat(square, image):
    source = square._replace(uv=square.uv + [2, -3])
    result = TextureBaking.bake_pixels(source, square, image, resolution=(16, 16), margin_pixels=0)
    np.testing.assert_array_equal(result, image)


def test_bake_ray_miss_and_distance(square, image):
    above = square._replace(vertices=square.vertices + [0, 0, 1])
    missed = TextureBaking.bake_pixels(square, above, image, resolution=(8, 8), max_ray_distance=.5, margin_pixels=0)
    assert not missed.any()
    hit = TextureBaking.bake_pixels(square, above, image, resolution=(8, 8), max_ray_distance=0, margin_pixels=0)
    assert np.all(hit[:, :, 3] == 255)


def test_bake_collapsed_target_keeps_indices(square, image):
    target = square._replace(vertices=np.zeros_like(square.vertices))
    output = TextureBaking.bake_pixels(square, target, image, resolution=(8, 8), margin_pixels=0)
    assert np.all(output[:, :, 3] == 255)
    np.testing.assert_array_equal(target.faces, square.faces)


def test_bake_padding_is_bounded(square, image):
    target = square._replace(uv=square.uv * .5 + .25)
    plain = TextureBaking.bake_pixels(square, target, image, resolution=(16, 16), margin_pixels=0)
    padded = TextureBaking.bake_pixels(square, target, image, resolution=(16, 16), margin_pixels=2)
    assert np.count_nonzero(padded[:, :, 3]) > np.count_nonzero(plain[:, :, 3])
    assert not padded[0].any()
    np.testing.assert_array_equal(padded[plain[:, :, 3] > 0], plain[plain[:, :, 3] > 0])


@pytest.mark.parametrize("setting", [{"resolution": (True, 8)}, {"resolution": (0, 8)}, {"margin_pixels": -1}, {"cage_extrusion": np.nan}, {"max_ray_distance": -1}])
def test_bake_bad_settings(square, image, setting):
    with pytest.raises(ValueError):
        TextureBaking.bake_pixels(square, square, image, **setting)


def test_native_surface_thread_determinism():
    # Separate processes initialize the process-wide pool with different limits.
    script = '''import hashlib
import numpy as np
from interdeca.models import Mesh
from interdeca.surface import TextureBaking
v=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float)
f=np.array([[0,1,2],[0,2,3]],np.int64)
m=Mesh(v,f,v[f,:2])
p=np.random.default_rng(42).integers(0,256,(33,33,4),dtype=np.uint8)
a=TextureBaking.bake_pixels(m,m,p,resolution=(128,128),margin_pixels=2)
print(hashlib.sha256(a.tobytes()).hexdigest())
'''
    results = [subprocess.check_output([sys.executable, "-c", script], env={**os.environ, "INTERDECA_SPATIAL_THREADS": str(n)}, text=True) for n in (1, 8)]
    assert results[0] == results[1]


def test_native_atlas_and_png_need_no_blender(square, image, tmp_path, monkeypatch):
    def unexpected_blender(*args, **kwargs):
        pytest.fail("Native surface operations invoked Blender.")

    monkeypatch.setattr(Blender, "_run", unexpected_blender)
    landmarks = Landmarks(np.eye(3), ("a", "b", "c"))
    atlas = MeshProcessing.prepare_atlas_uv(AtlasModel(square, landmarks), resolution=(32, 32))
    assert atlas.mesh.vertices is square.vertices
    assert atlas.mesh.faces is square.faces
    assert atlas.landmarks is landmarks
    source_path = tmp_path / "source.png"
    Image.fromarray(image).save(source_path)
    specimen = Specimen("sample", square, landmarks, source_path)
    target = ResampledSpecimen("sample", atlas.mesh, landmarks, np.zeros(4))
    output = tmp_path / "baked.png"
    baked = TextureBaking.bake_specimen_texture(specimen, target, output, resolution=(32, 32))
    assert baked.path == output
    assert baked.topology_key == atlas.mesh.topology_key
    with Image.open(output) as encoded:
        assert encoded.mode == "RGBA"
        assert encoded.size == (32, 32)
        pixels = np.asarray(encoded)
        np.testing.assert_array_equal(pixels[16, 16], image[0, 0])
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        TextureBaking.bake_specimen_texture(specimen, target, output, resolution=(32, 32))
    assert output.read_bytes() == original


def test_native_bake_rejects_different_aligned_frame(square, image, tmp_path):
    landmarks = Landmarks(np.eye(3), ("a", "b", "c"))
    source_path = tmp_path / "source.png"
    Image.fromarray(image).save(source_path)
    specimen = Specimen("sample", square, landmarks, source_path)
    shifted = Landmarks(landmarks.points + 1, landmarks.labels)
    target = ResampledSpecimen("sample", square, shifted, np.zeros(4))
    output = tmp_path / "baked.png"
    with pytest.raises(ValueError, match="aligned frame"):
        TextureBaking.bake_specimen_texture(specimen, target, output)
    assert not output.exists()
