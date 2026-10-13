from pathlib import Path
from typing import Literal

import click
import numpy as np
import pyvista as pv
import yaml
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)
from .util import (
    config_path,
    file_index,
    load_yaml_mapping,
    resolve_config_paths,
    write_documented_config,
)
from vtkmodules.vtkCommonCore import vtkObject


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "All_Clams"
TEXTURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
CONFIG_PATH_FIELDS = ("models_dir", "textures_dir", "output_dir")
DEPRECATED_CONFIG_FIELDS = ("crop_to_content", "render_all")
WORLD_UP = np.array((0.0, 0.0, 1.0))


class RenderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    models_dir: Path = Field(
        default=DEFAULT_DATASET_DIR / "models",
        description="Directory containing OBJ mesh files.",
    )
    textures_dir: Path = Field(
        default=DEFAULT_DATASET_DIR / "textures",
        description="Directory containing texture images matched to meshes by specimen id.",
    )
    specimen_ids: tuple[str, ...] = Field(
        default=(),
        description="Specimen ids to render. Leave empty to render every OBJ that has a matching texture.",
        json_schema_extra={"commented_example": "UF_IZ_438751"},
    )
    view: Literal["above", "beneath", "both"] = Field(
        default="both",
        description="Standard camera view to render: exterior above, interior beneath, or both.",
    )
    output_dir: Path = Field(
        default=Path("renders"),
        description="Directory where rendered PNG files are written.",
    )
    width: int = Field(
        default=1600,
        gt=0,
        description="Final per-view image width in pixels after cropping and resizing.",
    )
    height: int = Field(
        default=1000,
        gt=0,
        description="Final per-view image height in pixels after cropping and resizing.",
    )
    layout: Literal["horizontal", "vertical"] = Field(
        default="horizontal",
        description="Combined image layout when view is both.",
    )
    background: str = Field(
        default="white",
        description="Background color passed to PyVista.",
    )
    transparent: bool = Field(
        default=False,
        description="Write PNGs with a transparent background.",
    )
    margin: float = Field(
        default=1.08,
        gt=0,
        description="Orthographic framing margin around the mesh.",
    )
    texture_max_size: int = Field(
        default=4096,
        ge=0,
        description="Downsample texture long edge to this size before rendering. Use 0 for original size.",
    )
    show_edges: bool = Field(
        default=False,
        description="Render polygon edges on top of the textured mesh.",
    )
    flat_shading: bool = Field(
        default=False,
        description="Disable smooth mesh shading.",
    )
    show_vtk_warnings: bool = Field(
        default=False,
        description="Show VTK warnings such as missing DISPLAY messages.",
    )
    crop_padding_pixels: int = Field(
        default=12,
        ge=0,
        description="Padding in pixels to keep around content before resizing to the final image dimensions.",
    )
    crop_background_tolerance: int = Field(
        default=8,
        ge=0,
        le=255,
        description="Per-channel tolerance for treating pixels as background during cropping.",
    )

    @field_validator("models_dir", "textures_dir", "output_dir", mode="before")
    @classmethod
    def expand_path(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value

    @field_validator("specimen_ids", mode="before")
    @classmethod
    def normalize_specimen_ids(cls, value: object) -> object:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return value


def load_config(path: Path) -> RenderConfig:
    path = path.expanduser().resolve()
    loaded_data = load_config_data(path)

    resolved_data = resolve_config_paths(loaded_data, path.parent, CONFIG_PATH_FIELDS)
    config = RenderConfig.model_validate(resolved_data)
    if "output_dir" not in loaded_data:
        config = config.model_copy(update={"output_dir": config_path(config.output_dir, path.parent)})
    return config


def load_config_data(path: Path) -> dict[str, object]:
    return load_yaml_mapping(path)


def load_documented_config_source(path: Path) -> RenderConfig:
    loaded_data = load_config_data(path.expanduser())
    documented_data = {
        field_name: value
        for field_name, value in loaded_data.items()
        if field_name in RenderConfig.model_fields
    }
    return RenderConfig.model_validate(documented_data)


def default_documented_config() -> RenderConfig:
    return RenderConfig(
        models_dir=Path("../data/All_Clams/models"),
        textures_dir=Path("../data/All_Clams/textures"),
    )


def texture_index(textures_dir: Path) -> dict[str, Path]:
    return file_index(textures_dir, TEXTURE_SUFFIXES)


def specimen_paths(config: RenderConfig) -> list[tuple[str, Path, Path]]:
    models = {path.stem: path for path in sorted(config.models_dir.glob("*.obj"))}
    textures = texture_index(config.textures_dir)

    if config.specimen_ids:
        specimen_ids = config.specimen_ids
    else:
        specimen_ids = sorted(set(models) & set(textures))

    pairs = []
    for specimen_id in specimen_ids:
        model_path = models.get(specimen_id)
        texture_path = textures.get(specimen_id)
        if model_path is None:
            raise FileNotFoundError(f"No OBJ found for specimen {specimen_id!r}")
        if texture_path is None:
            raise FileNotFoundError(f"No texture found for specimen {specimen_id!r}")
        pairs.append((specimen_id, model_path, texture_path))

    if not pairs:
        raise FileNotFoundError(
            f"No OBJ/texture pairs found in {config.models_dir} and {config.textures_dir}"
        )
    return pairs


def read_texture(path: Path, max_size: int) -> pv.Texture:
    with Image.open(path) as image:
        image = image.convert("RGBA")
        if max_size > 0:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        texture_array = np.asarray(image)
    return pv.numpy_to_texture(texture_array)


def content_bounds(image: Image.Image, background_tolerance: int) -> tuple[int, int, int, int] | None:
    if "A" in image.getbands():
        alpha = np.asarray(image.getchannel("A"))
        if np.any(alpha < 255):
            mask = alpha > background_tolerance
        else:
            mask = opaque_content_mask(image, background_tolerance)
    else:
        mask = opaque_content_mask(image, background_tolerance)

    rows, columns = np.where(mask)
    if rows.size == 0 or columns.size == 0:
        return None
    return columns.min(), rows.min(), columns.max() + 1, rows.max() + 1


def opaque_content_mask(image: Image.Image, background_tolerance: int) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    background = pixels[0, 0]
    difference = np.max(np.abs(pixels - background), axis=2)
    return difference > background_tolerance


def padded_bounds(
    bounds: tuple[int, int, int, int],
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    return (
        max(left - padding, 0),
        max(top - padding, 0),
        min(right + padding, width),
        min(bottom + padding, height),
    )


def crop_and_resize_image(image: Image.Image, target_size: tuple[int, int], config: RenderConfig) -> Image.Image:
    bounds = content_bounds(image, config.crop_background_tolerance)
    if bounds is None:
        cropped_image = image.copy()
    else:
        crop_box = padded_bounds(bounds, image.width, image.height, config.crop_padding_pixels)
        cropped_image = image.crop(crop_box)

    if cropped_image.size != target_size:
        cropped_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)
    return cropped_image


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm


def mesh_points(mesh: pv.PolyData, max_points: int = 250_000) -> np.ndarray:
    points = np.asarray(mesh.points, dtype=float)
    stride = max(1, len(points) // max_points)
    return points[::stride]


def shell_orientation(mesh: pv.PolyData) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    points = mesh_points(mesh)
    center = points.mean(axis=0)
    centered_points = points - center
    covariance = centered_points.T @ centered_points / len(centered_points)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    major_axis = normalized(eigenvectors[:, 0])
    minor_axis = normalized(eigenvectors[:, 1])
    plane_normal = normalized(eigenvectors[:, 2])
    exterior_direction = exterior_normal(centered_points, major_axis, minor_axis, plane_normal)
    view_up = view_up_axis(minor_axis, exterior_direction)
    return tuple(exterior_direction), tuple(view_up)


def exterior_normal(
    centered_points: np.ndarray,
    major_axis: np.ndarray,
    minor_axis: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    major_projection = centered_points @ major_axis
    minor_projection = centered_points @ minor_axis
    radial_distance = np.sqrt(major_projection * major_projection + minor_projection * minor_projection)
    projected_depth = centered_points @ plane_normal
    center_depth = np.mean(projected_depth[radial_distance <= np.quantile(radial_distance, 0.25)])
    rim_depth = np.mean(projected_depth[radial_distance >= np.quantile(radial_distance, 0.75)])
    if center_depth < rim_depth:
        return -plane_normal
    return plane_normal


def view_up_axis(in_plane_axis: np.ndarray, direction: np.ndarray) -> np.ndarray:
    view_up = in_plane_axis - direction * np.dot(in_plane_axis, direction)
    view_up = normalized(view_up)
    if np.dot(view_up, WORLD_UP) < 0:
        view_up = -view_up
    return view_up


def camera_position(
    mesh: pv.PolyData,
    direction: tuple[float, float, float],
    view_up: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    center = np.asarray(mesh.center, dtype=float)
    direction = normalized(np.asarray(direction, dtype=float))
    distance = max(float(mesh.length) * 2.5, 1.0)
    position = center + direction * distance
    return tuple(position), tuple(center), tuple(view_up)


def view_basis(
    direction: tuple[float, float, float],
    view_up: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction_vector = normalized(np.asarray(direction, dtype=float))
    up_vector = np.asarray(view_up, dtype=float)
    up_vector = normalized(up_vector - direction_vector * np.dot(up_vector, direction_vector))
    right_vector = normalized(np.cross(direction_vector, up_vector))
    return right_vector, up_vector, direction_vector


def parallel_scale(
    mesh: pv.PolyData,
    direction: tuple[float, float, float],
    view_up: tuple[float, float, float],
    aspect: float,
    margin: float,
) -> float:
    points = mesh_points(mesh)
    center = points.mean(axis=0)
    centered_points = points - center
    right_vector, up_vector, _ = view_basis(direction, view_up)
    horizontal_projection = centered_points @ right_vector
    vertical_projection = centered_points @ up_vector
    horizontal_length = np.ptp(horizontal_projection)
    vertical_length = np.ptp(vertical_projection)
    visible_height = max(vertical_length, horizontal_length / aspect)
    return visible_height * margin / 2.0


def add_clam_scene(
    plotter: pv.Plotter,
    mesh: pv.PolyData,
    texture: pv.Texture,
    direction: tuple[float, float, float],
    view_up: tuple[float, float, float],
    width: int,
    height: int,
    config: RenderConfig,
) -> None:
    plotter.set_background(config.background)
    plotter.add_mesh(
        mesh,
        texture=texture,
        show_edges=config.show_edges,
        smooth_shading=not config.flat_shading,
        ambient=0.38,
        diffuse=0.75,
        specular=0.08,
    )

    plotter.camera_position = camera_position(mesh, direction, view_up)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = parallel_scale(mesh, direction, view_up, width / height, config.margin)


def enable_antialiasing(plotter: pv.Plotter) -> None:
    plotter.enable_anti_aliasing("ssaa")


def render_clam_image(
    mesh: pv.PolyData,
    texture: pv.Texture,
    direction: tuple[float, float, float],
    view_up: tuple[float, float, float],
    image_size: tuple[int, int],
    config: RenderConfig,
) -> Image.Image:
    plotter = pv.Plotter(off_screen=True, window_size=image_size)
    enable_antialiasing(plotter)
    add_clam_scene(plotter, mesh, texture, direction, view_up, image_size[0], image_size[1], config)
    image_array = plotter.screenshot(transparent_background=config.transparent, return_img=True)
    plotter.close()
    image = Image.fromarray(image_array)
    return crop_and_resize_image(image, image_size, config)


def render_view(
    mesh: pv.PolyData,
    texture: pv.Texture,
    direction: tuple[float, float, float],
    view_up: tuple[float, float, float],
    output_path: Path,
    config: RenderConfig,
) -> None:
    image = render_clam_image(mesh, texture, direction, view_up, (config.width, config.height), config)
    image.save(output_path)


def render_combined(above_path: Path, beneath_path: Path, output_path: Path, config: RenderConfig) -> None:
    with Image.open(above_path) as above_image:
        above = above_image.copy()
    with Image.open(beneath_path) as beneath_image:
        beneath = beneath_image.copy()

    if config.layout == "horizontal":
        canvas_size = (above.width + beneath.width, max(above.height, beneath.height))
        above_position = (0, (canvas_size[1] - above.height) // 2)
        beneath_position = (above.width, (canvas_size[1] - beneath.height) // 2)
    else:
        canvas_size = (max(above.width, beneath.width), above.height + beneath.height)
        above_position = ((canvas_size[0] - above.width) // 2, 0)
        beneath_position = ((canvas_size[0] - beneath.width) // 2, above.height)

    canvas = Image.new(combined_image_mode(above, beneath), canvas_size, combined_background(above, config))
    paste_combined_image(canvas, above, above_position)
    paste_combined_image(canvas, beneath, beneath_position)
    canvas.save(output_path)


def combined_image_mode(above: Image.Image, beneath: Image.Image) -> str:
    if "A" in above.getbands() or "A" in beneath.getbands():
        return "RGBA"
    return "RGB"


def combined_background(image: Image.Image, config: RenderConfig) -> tuple[int, ...]:
    if config.transparent:
        return (0, 0, 0, 0)
    return image.convert("RGB").getpixel((0, 0))


def paste_combined_image(canvas: Image.Image, image: Image.Image, position: tuple[int, int]) -> None:
    if canvas.mode == "RGBA":
        canvas.paste(image.convert("RGBA"), position, image.convert("RGBA"))
    else:
        canvas.paste(image.convert("RGB"), position)


def opposite_direction(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(-component for component in direction)


def view_direction(view: str, exterior_direction: tuple[float, float, float]) -> tuple[float, float, float]:
    if view == "above":
        return exterior_direction
    return opposite_direction(exterior_direction)


def render_specimen(specimen_id: str, model_path: Path, texture_path: Path, config: RenderConfig) -> None:
    mesh = pv.read(model_path)
    texture = read_texture(texture_path, config.texture_max_size)
    exterior_direction, view_up = shell_orientation(mesh)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    views = ("above", "beneath") if config.view == "both" else (config.view,)
    rendered_paths = {}
    for view in views:
        output_path = config.output_dir / f"{specimen_id}_{view}.png"
        render_view(mesh, texture, view_direction(view, exterior_direction), view_up, output_path, config)
        rendered_paths[view] = output_path
        print(f"Rendered {output_path}")

    if config.view == "both":
        output_path = config.output_dir / f"{specimen_id}_above_beneath.png"
        render_combined(rendered_paths["above"], rendered_paths["beneath"], output_path, config)
        print(f"Rendered {output_path}")


@click.command()
@click.option(
    "--config",
    "config_file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("render_config.yaml"),
    show_default=True,
    help="YAML render configuration file.",
)
@click.option(
    "--write-config",
    "write_config_file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write a documented YAML config and exit.",
)
def main(config_file: Path, write_config_file: Path | None) -> None:
    try:
        if write_config_file is not None:
            if config_file.expanduser().exists():
                config = load_documented_config_source(config_file)
            else:
                config = default_documented_config()
            write_documented_config(config, write_config_file)
            click.echo(f"Wrote {write_config_file}")
            return

        config = load_config(config_file)
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as error:
        raise click.ClickException(str(error)) from error

    if not config.show_vtk_warnings:
        vtkObject.GlobalWarningDisplayOff()
    pv.OFF_SCREEN = True

    for specimen_id, model_path, texture_path in specimen_paths(config):
        render_specimen(specimen_id, model_path, texture_path, config)


if __name__ == "__main__":
    main()
