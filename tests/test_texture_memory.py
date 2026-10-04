"""Preserve image arithmetic, validation, and ownership while removing copies."""

import numpy as np
import pytest
from PIL import Image

from interdeca.io import TextureIO
from interdeca.models import TexturePixels
from interdeca.serde import TexturePixelsData


@pytest.fixture
def rgba():
    # Exercise every possible byte, including transparent and fully opaque alpha.
    pixels = np.arange(256, dtype=np.uint8).reshape(4, 64)
    return np.stack([pixels, np.roll(pixels, 1, axis=1), 255 - pixels, pixels], axis=-1)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P", "L", "LA"])
@pytest.mark.parametrize("suffix", [".png", ".tiff"])
def test_decoded_pixels_match_previous_float64_path(tmp_path, rgba, mode, suffix):
    path = tmp_path / (mode + suffix)
    Image.fromarray(rgba).convert(mode).save(path)
    with Image.open(path) as image:
        expected = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
    actual = TextureIO.read(path)
    np.testing.assert_array_equal(actual.rgb, expected[..., :3])
    np.testing.assert_array_equal(actual.alpha, expected[..., 3])
    for array in actual:
        assert array.dtype == np.float64
        # Rust transfers ownership through a NumPy base capsule. The contract
        # requires independent storage, not NumPy's particular OWNDATA flag.
        assert array.flags.c_contiguous
        assert not np.shares_memory(array, expected)
        assert not array.flags.writeable


def test_byte_conversion_owns_input_and_reuses_validated_fields(rgba):
    expected = rgba.astype(np.float64) / 255.0
    data = TexturePixelsData.from_rgba8(rgba)
    record = data.to_internal()
    rgba[:] = 0
    np.testing.assert_array_equal(record.rgb, expected[..., :3])
    np.testing.assert_array_equal(record.alpha, expected[..., 3])
    # The boundary and internal record share the same owned allocations, which
    # prevents the old second image-sized validation copy from returning.
    assert data.rgb is record.rgb
    assert data.alpha is record.alpha
    assert data.to_internal() is record
    decoded = TexturePixelsData.model_validate_json(
        data.model_dump_json()
    ).to_internal()
    for left, right in zip(record, decoded, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("view", ["writable", "noncontiguous", "readonly_alias"])
@pytest.mark.parametrize("boundary", [True, False])
def test_external_float_alias_cannot_mutate_owned_record(view, boundary):
    original = np.full((4, 4, 4), 0.5)
    pixels = original[::2] if view == "noncontiguous" else original.view()
    if view == "readonly_alias":
        pixels.setflags(write=False)
    rgb, alpha = pixels[..., :3], pixels[..., 3]
    record = (
        TexturePixelsData(rgb=rgb, alpha=alpha).to_internal()
        if boundary
        else TexturePixels(rgb, alpha).validated()
    )
    expected = tuple(array.copy() for array in record)
    original[:] = 0.75
    for actual, unchanged in zip(record, expected, strict=True):
        np.testing.assert_array_equal(actual, unchanged)
        assert not actual.flags.writeable
        assert actual.flags.c_contiguous
        assert not np.shares_memory(actual, original)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.01, 1.01])
@pytest.mark.parametrize("channel", ["rgb", "alpha"])
def test_unit_channels_reject_invalid_values(bad, channel):
    rgb, alpha = np.ones((2, 2, 3)), np.ones((2, 2))
    (rgb if channel == "rgb" else alpha).flat[0] = bad
    with pytest.raises(ValueError):
        TexturePixelsData(rgb=rgb, alpha=alpha)
    with pytest.raises(ValueError):
        TexturePixels(rgb, alpha).validated()


@pytest.mark.parametrize(
    "shape,alpha",
    [((0, 2, 3), (0, 2)), ((2, 2, 4), (2, 2)), ((2, 3), (2,)), ((2, 2, 3), (2, 3))],
)
def test_invalid_dimensions_rejected(shape, alpha):
    with pytest.raises(ValueError):
        TexturePixelsData(rgb=np.zeros(shape), alpha=np.zeros(alpha))


@pytest.mark.parametrize("dtype", [np.bool_, object, str])
def test_unit_channels_require_actual_numbers(dtype):
    with pytest.raises(ValueError):
        TexturePixelsData(rgb=np.ones((2, 2, 3), dtype=dtype), alpha=np.ones((2, 2)))


def test_regular_uint8_fields_remain_unit_values():
    # Only the explicit image factory interprets bytes as 0..255 encoded colors.
    record = TexturePixelsData(
        rgb=np.ones((2, 2, 3), dtype=np.uint8), alpha=np.ones((2, 2), dtype=np.uint8)
    ).to_internal()
    assert np.all(record.rgb == 1) and np.all(record.alpha == 1)
    with pytest.raises(ValueError):
        TexturePixelsData(
            rgb=np.full((2, 2, 3), 255, dtype=np.uint8), alpha=np.ones((2, 2))
        )


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.int32])
def test_image_factory_rejects_other_encodings(dtype):
    with pytest.raises(ValueError):
        TexturePixelsData.from_rgba8(np.ones((2, 2, 4), dtype=dtype))


@pytest.mark.parametrize("shape", [(2, 2, 3), (2, 4), (0, 2, 4)])
def test_image_factory_rejects_bad_shapes(shape):
    with pytest.raises(ValueError):
        TexturePixelsData.from_rgba8(np.ones(shape, dtype=np.uint8))


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.int32])
def test_file_reader_rejects_high_precision_images(tmp_path, dtype):
    path = tmp_path / "unsupported.tiff"
    Image.fromarray(np.ones((2, 2), dtype=dtype)).save(path)
    with pytest.raises(ValueError, match="Unsupported texture mode"):
        TextureIO.read(path)
