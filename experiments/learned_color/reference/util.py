from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps
from pydantic import BaseModel


class IndentedSafeDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


def load_yaml_mapping(path: Path) -> dict[str, object]:
    loaded_data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded_data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return loaded_data


def file_index(directory: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    suffix_set = {suffix.lower() for suffix in suffixes}
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in suffix_set:
            files.setdefault(path.stem, path)
    return files


def config_path(path: object, base_dir: Path) -> Path:
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path.resolve()
    return (base_dir / resolved_path).resolve()


def resolve_config_paths(
    config_data: dict[str, object],
    base_dir: Path,
    field_names: tuple[str, ...],
) -> dict[str, object]:
    resolved_data = dict(config_data)
    for field_name in field_names:
        if field_name in resolved_data and resolved_data[field_name] is not None:
            resolved_data[field_name] = config_path(resolved_data[field_name], base_dir)
    return resolved_data


def documented_yaml(config: BaseModel) -> str:
    config_data = config.model_dump(mode="json")
    lines = []
    for field_name, field_info in config.__class__.model_fields.items():
        if field_info.description:
            lines.extend(f"# {line}" for line in field_info.description.splitlines())
        lines.extend(documented_field_yaml(field_name, config_data[field_name], field_info.json_schema_extra))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def documented_field_yaml(
    field_name: str,
    value: object,
    schema_extra: dict[str, object] | None,
) -> list[str]:
    commented_example = None if schema_extra is None else schema_extra.get("commented_example")
    if not value and commented_example is not None:
        return [
            f"{field_name}:",
            f"  # - {commented_example}",
        ]

    field_yaml = yaml.dump(
        {field_name: value},
        Dumper=IndentedSafeDumper,
        default_flow_style=False,
        sort_keys=False,
    )
    return field_yaml.rstrip().splitlines()


def write_documented_config(config: BaseModel, path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(documented_yaml(config))


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def pixel_positions(
    coordinates: np.ndarray,
    width: int,
    height: int,
    padding: int,
) -> np.ndarray:
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    unit = (coordinates - minimum) / span
    x_positions = padding + unit[:, 0] * max(width - 2 * padding, 1)
    y_positions = height - padding - unit[:, 1] * max(height - 2 * padding, 1)
    return np.column_stack([x_positions, y_positions])


def non_overlapping_positions(
    anchor_positions: np.ndarray,
    item_sizes: np.ndarray,
    bounds: tuple[float, float, float, float],
    gap: float = 0.0,
    search_step: float = 6.0,
) -> np.ndarray:
    """Place rectangles near anchors without overlap, preserving crowded anchors first."""
    anchors = np.asarray(anchor_positions, dtype=float)
    sizes = np.asarray(item_sizes, dtype=float)
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError(f"Expected anchor positions shaped [items, 2], got {anchors.shape}.")
    if sizes.shape != anchors.shape:
        raise ValueError(f"Expected one width and height per anchor, got {sizes.shape}.")
    if np.any(~np.isfinite(anchors)) or np.any(~np.isfinite(sizes)):
        raise ValueError("Layout anchors and item sizes must be finite.")
    if np.any(sizes <= 0.0):
        raise ValueError("Layout item sizes must be positive.")
    if gap < 0.0 or search_step <= 0.0:
        raise ValueError("Layout gap must be nonnegative and search step must be positive.")
    if len(anchors) == 0:
        return anchors.copy()

    left, top, right, bottom = (float(value) for value in bounds)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid layout bounds: {bounds}.")

    half_sizes = sizes / 2.0
    minimum_centers = np.column_stack(
        [
            np.full(len(sizes), left),
            np.full(len(sizes), top),
        ]
    ) + half_sizes
    maximum_centers = np.column_stack(
        [
            np.full(len(sizes), right),
            np.full(len(sizes), bottom),
        ]
    ) - half_sizes
    if np.any(minimum_centers > maximum_centers):
        raise ValueError("At least one layout item is larger than the available bounds.")

    clamped_anchors = np.minimum(np.maximum(anchors, minimum_centers), maximum_centers)
    coordinate_scale = np.maximum(np.mean(sizes, axis=0), 1.0)
    pairwise_delta = (clamped_anchors[:, None, :] - clamped_anchors[None, :, :]) / coordinate_scale
    pairwise_distance = np.linalg.norm(pairwise_delta, axis=2)
    density = np.sum(np.exp(-pairwise_distance), axis=1) - 1.0
    areas = np.prod(sizes, axis=1)
    placement_order = sorted(
        range(len(anchors)),
        key=lambda index: (-density[index], -areas[index], index),
    )

    positions = np.empty_like(clamped_anchors)
    placed_indices: list[int] = []
    for index in placement_order:
        anchor = clamped_anchors[index]
        maximum_offset = float(
            np.max(
                [
                    anchor - minimum_centers[index],
                    maximum_centers[index] - anchor,
                ]
            )
        )
        maximum_ring = int(np.ceil(maximum_offset / search_step)) + 1
        selected = None
        visited: set[tuple[float, float]] = set()

        for ring in range(maximum_ring + 1):
            if ring == 0:
                offsets = [(0, 0)]
            else:
                offsets = []
                for horizontal in range(-ring, ring + 1):
                    offsets.append((horizontal, -ring))
                    offsets.append((horizontal, ring))
                for vertical in range(-ring + 1, ring):
                    offsets.append((-ring, vertical))
                    offsets.append((ring, vertical))

            candidates = []
            for horizontal, vertical in offsets:
                candidate = anchor + search_step * np.asarray([horizontal, vertical], dtype=float)
                candidate = np.minimum(
                    np.maximum(candidate, minimum_centers[index]),
                    maximum_centers[index],
                )
                key = (float(candidate[0]), float(candidate[1]))
                if key in visited:
                    continue
                visited.add(key)
                candidates.append(candidate)

            candidates.sort(key=lambda candidate: float(np.sum((candidate - anchor) ** 2)))
            for candidate in candidates:
                collision = False
                for placed_index in placed_indices:
                    separation = np.abs(candidate - positions[placed_index])
                    required = half_sizes[index] + half_sizes[placed_index] + gap
                    if bool(np.all(separation < required)):
                        collision = True
                        break
                if not collision:
                    selected = candidate
                    break
            if selected is not None:
                break

        if selected is None:
            raise ValueError(
                "Could not place every plot item without overlap; enlarge the canvas or reduce "
                "thumbnail size."
            )
        positions[index] = selected
        placed_indices.append(index)

    return positions


def white_background_content_mask(image: Image.Image, background_tolerance: int) -> np.ndarray:
    pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    alpha = pixels[:, :, 3]
    if np.any(alpha < 255):
        return np.where(alpha > background_tolerance, 255, 0).astype(np.uint8)

    rgb_pixels = pixels[:, :, :3].astype(np.int16)
    difference = np.max(np.abs(rgb_pixels - 255), axis=2)
    return np.where(difference > background_tolerance, 255, 0).astype(np.uint8)


def mean_content_color(image: Image.Image, background_tolerance: int) -> tuple[int, int, int]:
    """Return the alpha-weighted mean RGB color of non-background image content."""
    pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb_pixels = pixels[:, :, :3].astype(np.float64)
    alpha = pixels[:, :, 3].astype(np.float64)
    if np.any(alpha < 255.0):
        weights = np.where(alpha > background_tolerance, alpha / 255.0, 0.0)
    else:
        weights = white_background_content_mask(image, background_tolerance) / 255.0

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("Cannot calculate content color from an empty image.")

    mean_rgb = np.sum(rgb_pixels * weights[:, :, None], axis=(0, 1)) / total_weight
    return tuple(int(round(value)) for value in mean_rgb)


def remove_white_background(image: Image.Image, background_tolerance: int) -> Image.Image:
    pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    if np.any(pixels[:, :, 3] < 255):
        return Image.fromarray(pixels)

    rgb_pixels = pixels[:, :, :3].astype(np.int16)
    difference = np.max(np.abs(rgb_pixels - 255), axis=2)
    pixels[:, :, 3] = np.where(difference <= background_tolerance, 0, pixels[:, :, 3])
    return Image.fromarray(pixels)


def thumbnail_from_path(path: Path, thumbnail_size: int, background_tolerance: int) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        image = remove_white_background(image, background_tolerance)
        image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
        return image.copy()
