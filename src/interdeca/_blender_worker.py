"""Private script executed by Blender's Python, never imported by InterDeCA.

This file intentionally depends only on Blender's own bpy and NumPy plus the
standard library. JSON and NPZ inputs isolate it from the host Python version.
"""

import math
import sys
from pathlib import Path

import bpy
import numpy as np
from _blender_protocol import BlenderBakeRequest, BlenderProtocol, BlenderUVRequest


class BlenderWorker:
    """Operations that must run inside Blender's own process and scene."""

    @staticmethod
    def load_mesh(path: str, name: str) -> bpy.types.Object:
        """Build mesh data directly, preserving vertex order and UV corner order."""
        # Numeric archives avoid coupling Blender to the host's Python objects,
        # and direct mesh construction preserves the correspondence indices.
        with np.load(path, allow_pickle=False) as archive:
            vertices, faces = archive["vertices"], archive["faces"]
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata(vertices.tolist(), [], faces.tolist())
            mesh.update()
            # Attach UVs to face corners so a shared vertex can cross a texture seam.
            if "uv" in archive:
                layer = mesh.uv_layers.new(name="InterDeCA_UV")
                layer.data.foreach_set("uv", archive["uv"].astype(np.float32).ravel())
        # Mesh data must belong to a scene object before Blender operators can use it.
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        return obj

    @staticmethod
    def prepare_uv(config: BlenderUVRequest) -> None:
        """Unwrap triangles and export UVs, asserting topology is unchanged."""
        # Blender's operators act on the active selection, so explicitly select
        # the atlas rather than relying on scene defaults.
        obj = BlenderWorker.load_mesh(config.mesh, "atlas")
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        # Start with one fresh map so every specimen will bake to the same atlas
        # layout instead of inheriting an arbitrary source specimen's texture map.
        while obj.data.uv_layers:
            obj.data.uv_layers.remove(obj.data.uv_layers[0])
        obj.data.uv_layers.new(name="InterDeCA_UV")
        # Remember corner indices so any unexpected topology change is detected
        # before the new UVs can be paired with the original atlas.
        before = np.asarray([loop.vertex_index for loop in obj.data.loops])
        # Smart UV projection operates on selected faces in edit mode; include
        # every face so no part of the atlas keeps an uninitialized UV mapping.
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(config.angle_limit_degrees),
            island_margin=config.island_margin,
            correct_aspect=True,
            scale_to_bounds=True,
        )
        bpy.ops.object.mode_set(mode="OBJECT")
        after = np.asarray([loop.vertex_index for loop in obj.data.loops])
        if not np.array_equal(before, after):
            raise RuntimeError("UV generation unexpectedly changed mesh connectivity.")
        # Return only the new UV coordinates because the host already owns the
        # unchanged geometry and must retain its original vertex precision.
        uv = np.empty(len(obj.data.loops) * 2, dtype=np.float32)
        obj.data.uv_layers.active.data.foreach_get("uv", uv)
        np.save(config.output, uv.reshape(-1, 3, 2), allow_pickle=False)

    @staticmethod
    def bake(config: BlenderBakeRequest) -> None:
        """Transfer only diffuse albedo; source shading and illumination are excluded."""
        # Keep the textured source and corresponding target as separate objects:
        # colors come from the source, while the atlas UV layout belongs to the target.
        source = BlenderWorker.load_mesh(config.source, "source")
        target = BlenderWorker.load_mesh(config.target, "target")
        texture = bpy.data.images.load(config.texture, check_existing=False)
        texture.colorspace_settings.name = "sRGB"
        # Build a simple diffuse material so baked values represent the image's
        # surface color without inheriting unrelated source material settings.
        source_material = bpy.data.materials.new("source_color")
        source_material.use_nodes = True
        nodes = source_material.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        diffuse = nodes.new("ShaderNodeBsdfDiffuse")
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.image = texture
        image_node.interpolation = "Linear"
        image_node.extension = "REPEAT"
        source_material.node_tree.links.new(
            image_node.outputs["Color"], diffuse.inputs["Color"]
        )
        source_material.node_tree.links.new(
            diffuse.outputs["BSDF"], output.inputs["Surface"]
        )
        source.data.materials.append(source_material)

        # Transparent initial pixels distinguish untouched image regions from
        # genuine black surface colors when the host reads the result.
        width, height = config.resolution
        baked = bpy.data.images.new(
            "baked_color", width=width, height=height, alpha=True
        )
        baked.generated_color = (0, 0, 0, 0)
        baked.colorspace_settings.name = "sRGB"
        # Give the target a dedicated image destination; otherwise Blender may
        # write into an unrelated material image or have nowhere to bake.
        target_material = bpy.data.materials.new("target_color")
        target_material.use_nodes = True
        target_node = target_material.node_tree.nodes.new("ShaderNodeTexImage")
        target_node.image = baked
        # Blender writes to the active image node of the active target material.
        target_material.node_tree.nodes.active = target_node
        target.data.materials.append(target_material)
        # CPU Cycles keeps this operation usable without a configured GPU;
        # disabling lighting passes leaves only diffuse color in the result.
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1
        bake = scene.render.bake
        bake.use_selected_to_active = True
        bake.use_clear = True
        bake.use_pass_direct = False
        bake.use_pass_indirect = False
        bake.use_pass_color = True
        # Ray distances control which source surface is reached, while the image
        # margin supplies padding that reduces filtering artifacts at UV seams.
        bake.cage_extrusion = config.cage_extrusion
        bake.max_ray_distance = config.max_ray_distance
        bake.margin = config.margin_pixels
        # In selected-to-active baking, the active object is the destination;
        # selecting these objects explicitly prevents reversing the transfer.
        bpy.ops.object.select_all(action="DESELECT")
        source.select_set(True)
        target.select_set(True)
        bpy.context.view_layer.objects.active = target
        bpy.ops.object.bake(type="DIFFUSE")
        # Save to the private output path only after baking finishes, allowing the
        # host to validate the result before publishing the user's final artifact.
        baked.filepath_raw = config.output
        baked.file_format = "PNG"
        baked.save()

    @staticmethod
    def main() -> None:
        """Dispatch one requested operation in a clean, private Blender scene."""
        # Blender consumes arguments before '--'; only the remaining path belongs
        # to our worker, preventing Blender options from being read as input data.
        arguments = sys.argv[sys.argv.index("--") + 1 :]
        if len(arguments) != 1:
            raise ValueError("Expected exactly one JSON configuration path.")
        config = BlenderProtocol.read_request(Path(arguments[0]))
        # Startup objects must not participate in ray projection or become the
        # accidental active target of a later operator.
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        # Dispatch on the typed record so each operation receives only its own options.
        if isinstance(config, BlenderUVRequest):
            BlenderWorker.prepare_uv(config)
        elif isinstance(config, BlenderBakeRequest):
            BlenderWorker.bake(config)
        else:
            raise ValueError(f"Unknown operation: {config.operation}")


if __name__ == "__main__":
    # Execute scene operations only when Blender launches this file as a worker.
    BlenderWorker.main()
