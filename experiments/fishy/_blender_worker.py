"""Emission bake worker; executed only by Blender's Python, never by the host."""

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


class EmissionBake:
    """Write the same BYTE_COLOR / linear assignment used by Alek's generator."""

    @staticmethod
    def run(source: Path, output: Path, resolution: int, margin: int) -> None:
        """Build mesh directly so OBJ import cannot rotate or reorder its vertices."""
        if output.exists():
            raise FileExistsError(output)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        with np.load(source, allow_pickle=False) as data:
            vertices, faces, uv, rgb = (
                data[key] for key in ("vertices", "faces", "uv", "rgb")
            )
            mesh = bpy.data.meshes.new("fishy")
            mesh.from_pydata(vertices.tolist(), [], faces.tolist())
            mesh.update()
            layer = mesh.uv_layers.new(name="NativeUV")
            layer.data.foreach_set("uv", uv.astype(np.float32).ravel())
            # Color is attached per corner; UV seams retain their native layout.
            colors = mesh.color_attributes.new(
                name="SynColor", type="BYTE_COLOR", domain="CORNER"
            )
            rgba = np.column_stack([rgb[faces.ravel()], np.ones(faces.size)])
            colors.data.foreach_set("color", rgba.astype(np.float32).ravel())
        obj = bpy.data.objects.new("fishy", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        material = bpy.data.materials.new("SyntheticEmission")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        output_node = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        attribute = nodes.new("ShaderNodeAttribute")
        attribute.attribute_name = "SynColor"
        material.node_tree.links.new(
            attribute.outputs["Color"], emission.inputs["Color"]
        )
        material.node_tree.links.new(
            emission.outputs["Emission"], output_node.inputs["Surface"]
        )
        image = bpy.data.images.new(
            "SyntheticTexture",
            width=resolution,
            height=resolution,
            alpha=False,
            float_buffer=False,
        )
        target = nodes.new("ShaderNodeTexImage")
        target.image = image
        nodes.active = target
        obj.data.materials.append(material)
        # CPU is available on supported installations; record this deliberate
        # departure from the old GPU setting in host-side provenance.
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 1
        scene.render.bake.margin = margin
        scene.render.bake.target = "IMAGE_TEXTURES"
        bpy.ops.object.bake(type="EMIT")
        image.filepath_raw = str(output)
        image.file_format = "PNG"
        image.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("resolution", type=int)
    parser.add_argument("margin", type=int)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    EmissionBake.run(args.source, args.output, args.resolution, args.margin)
