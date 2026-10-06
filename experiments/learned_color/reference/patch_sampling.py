"""Content-mask-aware texture patch sampling shared by the texture-embedding experiments.

A "surface patch" is a square crop of a rendered shell view that lies entirely on the
shell surface (no background / white pixels). Patches may overlap but must not be
duplicates, and a deterministic, seeded jitter is applied to each sampled patch.

The same sampler is used by both ``dinov2_texture_embedding`` and
``ijepa_texture_embeddings`` so the two experiments draw patches identically.
"""


import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .util import white_background_content_mask


@dataclass(frozen=True)
class PatchSpec:
    patch_size: int
    target_count: int
    min_center_distance: float
    max_overlap_iou: float
    require_full_content: bool
    max_background_fraction: float
    background_tolerance: int


@dataclass(frozen=True)
class JitterSpec:
    enabled: bool
    brightness: float
    contrast: float
    saturation: float
    hue: float
    horizontal_flip: bool
    vertical_flip: bool

    @property
    def active(self) -> bool:
        return self.enabled and (
            self.brightness > 0
            or self.contrast > 0
            or self.saturation > 0
            or self.hue > 0
            or self.horizontal_flip
            or self.vertical_flip
        )


@dataclass(frozen=True)
class PatchSample:
    index: int
    left: int
    top: int
    size: int
    background_fraction: float
    flipped_horizontal: bool
    flipped_vertical: bool
    brightness_factor: float
    contrast_factor: float
    saturation_factor: float
    hue_shift: float
    image: Image.Image

    def manifest_row(self) -> dict[str, object]:
        row = asdict(self)
        row.pop("image")
        return row


def specimen_seed(global_seed: int, specimen_id: str) -> int:
    """Derive a stable per-specimen seed so each render samples reproducibly."""
    digest = hashlib.blake2b(f"{global_seed}:{specimen_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def content_mask(image: Image.Image, background_tolerance: int) -> np.ndarray:
    """Boolean mask where True marks shell-surface (content) pixels."""
    return white_background_content_mask(image, background_tolerance) > 0


def _window_background_counts(background: np.ndarray, size: int) -> np.ndarray:
    """Background-pixel count for every ``size`` x ``size`` window via an integral image."""
    integral = np.zeros((background.shape[0] + 1, background.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(background.astype(np.int64), axis=0), axis=1)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def candidate_positions(mask: np.ndarray, spec: PatchSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(positions, background_fractions)`` for every valid top-left corner.

    ``positions`` is an ``(n, 2)`` array of ``(top, left)`` pixel coordinates.
    """
    size = spec.patch_size
    if size > mask.shape[0] or size > mask.shape[1]:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)

    background = ~mask
    counts = _window_background_counts(background, size)
    area = size * size
    threshold = 0 if spec.require_full_content else int(spec.max_background_fraction * area)
    valid = counts <= threshold
    positions = np.argwhere(valid)
    fractions = counts[valid] / area
    return positions, fractions


def _iou_equal_squares(dx: float, dy: float, size: int) -> float:
    overlap_w = max(0.0, size - abs(dx))
    overlap_h = max(0.0, size - abs(dy))
    intersection = overlap_w * overlap_h
    if intersection <= 0:
        return 0.0
    union = 2 * size * size - intersection
    return intersection / union


def select_positions(
    positions: np.ndarray,
    fractions: np.ndarray,
    spec: PatchSpec,
    rng: np.random.Generator,
) -> list[tuple[int, int, float]]:
    """Greedily pick non-duplicate patches honouring min distance / max overlap."""
    if positions.shape[0] == 0:
        return []

    order = rng.permutation(positions.shape[0])
    accepted: list[tuple[int, int, float]] = []
    accepted_centers: list[tuple[float, float]] = []
    half = spec.patch_size / 2.0

    for index in order:
        if len(accepted) >= spec.target_count:
            break
        top, left = int(positions[index, 0]), int(positions[index, 1])
        center = (left + half, top + half)
        if _conflicts(center, accepted_centers, spec):
            continue
        accepted.append((left, top, float(fractions[index])))
        accepted_centers.append(center)
    return accepted


def _conflicts(
    center: tuple[float, float],
    accepted_centers: list[tuple[float, float]],
    spec: PatchSpec,
) -> bool:
    for other in accepted_centers:
        dx = center[0] - other[0]
        dy = center[1] - other[1]
        if spec.min_center_distance > 0:
            if (dx * dx + dy * dy) < spec.min_center_distance * spec.min_center_distance:
                return True
        if spec.max_overlap_iou < 1.0:
            if _iou_equal_squares(dx, dy, spec.patch_size) > spec.max_overlap_iou:
                return True
    return False


def _apply_jitter(
    patch: Image.Image,
    jitter: JitterSpec,
    rng: np.random.Generator,
) -> tuple[Image.Image, dict[str, object]]:
    params = {
        "flipped_horizontal": False,
        "flipped_vertical": False,
        "brightness_factor": 1.0,
        "contrast_factor": 1.0,
        "saturation_factor": 1.0,
        "hue_shift": 0.0,
    }
    if not jitter.active:
        return patch, params

    if jitter.horizontal_flip and rng.random() < 0.5:
        patch = ImageOps.mirror(patch)
        params["flipped_horizontal"] = True
    if jitter.vertical_flip and rng.random() < 0.5:
        patch = ImageOps.flip(patch)
        params["flipped_vertical"] = True

    if jitter.brightness > 0:
        factor = float(rng.uniform(1 - jitter.brightness, 1 + jitter.brightness))
        patch = ImageEnhance.Brightness(patch).enhance(factor)
        params["brightness_factor"] = factor
    if jitter.contrast > 0:
        factor = float(rng.uniform(1 - jitter.contrast, 1 + jitter.contrast))
        patch = ImageEnhance.Contrast(patch).enhance(factor)
        params["contrast_factor"] = factor
    if jitter.saturation > 0:
        factor = float(rng.uniform(1 - jitter.saturation, 1 + jitter.saturation))
        patch = ImageEnhance.Color(patch).enhance(factor)
        params["saturation_factor"] = factor
    if jitter.hue > 0:
        shift = int(round(float(rng.uniform(-jitter.hue, jitter.hue)) * 255))
        if shift != 0:
            hsv = np.array(patch.convert("HSV"), dtype=np.uint8)
            hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + shift) % 256).astype(np.uint8)
            patch = Image.fromarray(hsv, mode="HSV").convert("RGB")
        params["hue_shift"] = shift / 255.0

    return patch, params


def sample_patches(
    image: Image.Image,
    spec: PatchSpec,
    jitter: JitterSpec,
    seed: int,
) -> list[PatchSample]:
    """Sample non-white, non-duplicate, jittered patches from a rendered view."""
    rgb_image = image.convert("RGB")
    mask = content_mask(image, spec.background_tolerance)
    positions, fractions = candidate_positions(mask, spec)
    rng = np.random.default_rng(seed)
    selected = select_positions(positions, fractions, spec, rng)

    samples: list[PatchSample] = []
    seen_hashes: set[str] = set()
    for left, top, background_fraction in selected:
        crop = rgb_image.crop((left, top, left + spec.patch_size, top + spec.patch_size))
        jittered, params = _apply_jitter(crop, jitter, rng)
        digest = hashlib.blake2b(jittered.tobytes(), digest_size=16).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        samples.append(
            PatchSample(
                index=len(samples),
                left=left,
                top=top,
                size=spec.patch_size,
                background_fraction=background_fraction,
                image=jittered,
                **params,
            )
        )
    return samples


def patches_are_full_content(image: Image.Image, background_tolerance: int) -> bool:
    """True when an extracted patch contains no background pixels."""
    return bool(content_mask(image, background_tolerance).all())


def save_patches(samples: list[PatchSample], specimen_dir: Path, specimen_id: str) -> list[Path]:
    specimen_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sample in samples:
        path = specimen_dir / f"{specimen_id}_p{sample.index:04d}.png"
        sample.image.save(path)
        paths.append(path)
    return paths
