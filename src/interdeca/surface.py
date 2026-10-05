"""Native atlas preparation and color transfer derived from Blender source.

Cleanup may change topology before correspondence. UV generation changes only
per-corner coordinates. Texture baking never changes either mesh's indexing.
These APIs do not select or invoke Blender and remain independent of scheduling.
"""

import io
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from interdeca import _native
from interdeca.io import Files, TextureIO
from interdeca.models import AtlasModel, BakedTexture, Mesh, ResampledSpecimen, Specimen
from interdeca.serde import NativeBakeOptions, NativeCleanupOptions, NativeUVOptions


class CleanupResult(NamedTuple):
    """Clean mesh and original-to-output maps; removed elements have index -1."""

    mesh: Mesh
    vertex_map: np.ndarray
    face_map: np.ndarray


class UVLayout(NamedTuple):
    """Per-corner UVs and one deterministic chart identifier per original face."""

    uv: np.ndarray
    chart_ids: np.ndarray


class MeshProcessing:
    """Rust atlas operations; source-derived paths are tracked in BLENDER_SOURCE.json."""

    @staticmethod
    def clean(mesh: Mesh, *, merge_distance: float = 0.0001) -> CleanupResult:
        """Weld using Blender 5.2.1's float32 cluster and representative selection.

        Clusters form in balanced KD-tree order. The existing vertex nearest a
        weighted centroid survives; lower indices settle exact distance ties.
        Unchanged faces retain their ordering. Changed faces reuse an existing
        matching triangle when present. Output positions pass through float32.
        This operation is only valid before correspondence has been established.
        """
        # Validate distance at the public boundary before native float32 clustering.
        settings = NativeCleanupOptions(merge_distance=merge_distance)
        vertices, faces, vertex_map, face_map = _native.surface_clean(
            mesh.vertices, mesh.faces, settings.merge_distance
        )
        # Old UV corners cease to address the mesh after topology changes. The
        # zero-radius bypass, however, must preserve the entire original record.
        cleaned = mesh if settings.merge_distance == 0 else Mesh(vertices, faces)
        return CleanupResult(cleaned, vertex_map, face_map)

    @staticmethod
    def clean_atlas(atlas: AtlasModel, *, merge_distance: float = 0.0001) -> AtlasModel:
        """Clean a new atlas without moving its anatomical landmarks."""
        result = MeshProcessing.clean(atlas.mesh, merge_distance=merge_distance)
        return atlas if result.mesh is atlas.mesh else AtlasModel(result.mesh, atlas.landmarks)

    @staticmethod
    def uv_layout(
        mesh: Mesh, *, angle_limit_degrees: float = 66,
        island_margin: float = 0.002, resolution: tuple[int, int] = (2048, 2048),
        margin_pixels: int = 2,
    ) -> UVLayout:
        """Run Blender 5.2.1's Smart UV Project defaults on a fresh UV map.

        Face normals choose planar projection groups. Shared UV edges form
        islands; their minimum-area rectangles are rotated upright, packed with
        Blender's scaled island margin, and stretched to the unit square.
        Coordinates pass through float32, while geometry arrays and their order
        are retained. Collapsed faces follow Blender's zero-area UV behavior.
        Resolution and margin_pixels are retained for prototype API compatibility;
        Smart UV Project's layout depends only on angle and island_margin.
        """
        # Settings cross the boundary once; geometry and corner indexing stay intact.
        settings = NativeUVOptions(
            angle_limit_degrees=angle_limit_degrees, island_margin=island_margin,
            resolution=resolution, margin_pixels=margin_pixels,
        )
        return UVLayout(*_native.surface_uv(
            mesh.vertices, mesh.faces, settings.angle_limit_degrees,
            settings.island_margin, *settings.resolution, settings.margin_pixels,
        ))

    @staticmethod
    def prepare_atlas_uv(
        atlas: AtlasModel, *, angle_limit_degrees: float = 66,
        island_margin: float = 0.002, resolution: tuple[int, int] = (2048, 2048),
        margin_pixels: int = 2,
    ) -> AtlasModel:
        """Attach a common UV layout while retaining exact geometry arrays."""
        layout = MeshProcessing.uv_layout(
            atlas.mesh, angle_limit_degrees=angle_limit_degrees,
            island_margin=island_margin, resolution=resolution, margin_pixels=margin_pixels,
        )
        return AtlasModel(Mesh(atlas.mesh.vertices, atlas.mesh.faces, layout.uv), atlas.landmarks)


class TextureBaking:
    """Color-only ray transfer; source image alpha is not a transparency shader."""

    @staticmethod
    def bake_pixels(
        source: Mesh, target: Mesh, pixels: np.ndarray, *,
        resolution: tuple[int, int] = (2048, 2048), cage_extrusion: float = 0.001,
        max_ray_distance: float = 0, margin_pixels: int = 2,
    ) -> np.ndarray:
        """Return owned read-only RGBA8 pixels with coverage in the alpha channel.

        Pixels must be an (H,W,4) uint8 array; resolution is (width,height).
        Inputs share a coordinate frame. Source UVs repeat; target UVs occupy one
        tile. Source RGB is associated with alpha in byte storage before filtering,
        as in Blender; baked alpha records coverage rather than source opacity.
        Without padding, missing ray hits remain transparent; padding can
        fill small gaps. Zero maximum distance disables the distance cutoff.
        Invalid targets are never repaired or reindexed.
        """
        # Bound allocations and physical ray distances before copying mesh inputs.
        settings = NativeBakeOptions(
            resolution=resolution, cage_extrusion=cage_extrusion,
            max_ray_distance=max_ray_distance, margin_pixels=margin_pixels,
        )
        if source.uv is None or target.uv is None:
            raise ValueError("Baking requires both source and target UVs.")
        return _native.surface_bake(
            source.vertices, source.faces, source.uv,
            target.vertices, target.faces, target.uv, pixels,
            *settings.resolution, settings.cage_extrusion,
            settings.max_ray_distance, settings.margin_pixels,
        )

    @staticmethod
    def bake_specimen_texture(
        specimen: Specimen, target: ResampledSpecimen, output_path: Path | str, *,
        resolution: tuple[int, int] = (2048, 2048), cage_extrusion: float = 0.001,
        max_ray_distance: float = 0, margin_pixels: int = 2, overwrite: bool = False,
    ) -> BakedTexture:
        """Decode once, bake in memory, and publish a checked PNG after success."""
        # Identity and landmark agreement prevent transfer between incompatible frames.
        if specimen.specimen_id != target.specimen_id:
            raise ValueError("Source specimen and resampled target IDs must agree.")
        landmarks = specimen.landmarks.reordered(target.landmarks.labels)
        if not np.allclose(landmarks.points, target.landmarks.points, rtol=1e-8, atol=1e-10):
            raise ValueError("Source and target landmarks must share the aligned frame.")
        if specimen.texture_path is None:
            raise ValueError("Baking requires a source texture.")
        # Protect the source and existing artifacts before allocating bake images.
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".png":
            raise ValueError("Baked texture output must use the .png extension.")
        if destination == specimen.texture_path.expanduser().resolve():
            raise ValueError("A bake output cannot replace its source texture.")
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        # Byte decoding avoids allocating a full float64 copy of the source.
        # The kernel snapshots these bytes before releasing Python's GIL.
        with Image.open(specimen.texture_path) as image:
            TextureIO._check_encoding(image)
            with image.convert("RGBA") as converted:
                pixels = np.asarray(converted)
                baked = TextureBaking.bake_pixels(
                    specimen.mesh, target.mesh, pixels, resolution=resolution,
                    cage_extrusion=cage_extrusion, max_ray_distance=max_ray_distance,
                    margin_pixels=margin_pixels,
                )
        # Encoding completes in memory before an atomic write exposes the new artifact.
        buffer = io.BytesIO()
        Image.fromarray(baked).save(buffer, format="PNG")
        Files.write(destination, buffer.getvalue(), overwrite=overwrite)
        return BakedTexture(specimen.specimen_id, destination, target.mesh.topology_key, resolution)
