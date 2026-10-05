"""Compare analytical inputs against the pinned Blender reference executable."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from interdeca._blender_protocol import BlenderBakeRequest
from interdeca.blender import Blender
from interdeca.models import Mesh
from interdeca.surface import TextureBaking
from serde import FixtureResult


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path, default=Path("build/native-surface/fixtures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    installation = Blender.check_available()
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    source = Mesh(v, f, v[f, :2])
    image = np.full((16, 16, 4), 255, np.uint8)
    image[:, :, 0] = np.arange(16)[None, :] * 16
    image[:, :, 1] = np.arange(16)[:, None] * 16
    image[:, :, 2] = 37
    results = []
    # Offset tests distinguish forward, backward, extrusion, and maximum-distance
    # behavior. Inset UVs isolate padding from the actual surface pixels.
    for name, z, extrusion, distance, inset, margin in [
        ("gradient", 0, .01, 0, 0, 0),
        ("linear_filter", 0, .01, 0, 0, 0),
        ("half_byte", 0, .01, 0, 0, 0),
        ("folded", 0, .01, 0, 0, 0),
        ("collapsed", 0, .01, 0, 0, 0),
        ("below", -.1, .01, 0, 0, 0),
        ("above", .1, .01, 0, 0, 0),
        ("within_extrusion", -.005, .01, 0, 0, 0),
        ("distance_miss", .1, .01, .05, 0, 0),
        ("padding", 0, .01, 0, .25, 2),
        ("subpixel", 0, .01, 0, .49, 0),
        ("partial_source", 0, .01, 0, 0, 2),
        ("alpha_zero", 0, .01, 0, 0, 0),
        ("alpha_half", 0, .01, 0, 0, 0),
        ("repeat_negative", 0, .01, 0, 0, 0),
        ("distance_boundary", 0, .125, .125, 0, 0),
        ("vertex_sample", 0, .01, 0, 0, 0),
    ]:
        if args.case and args.case != name:
            continue
        resolution = (1, 1) if name in ("linear_filter","half_byte") else (16, 16)
        case_image = np.array([[[0,0,0,255],[255,255,255,255]]], np.uint8) if name in ("linear_filter","half_byte") else image
        if name=="half_byte":
            case_image[0,1,:3]=1
        if name in ("alpha_zero", "alpha_half"):
            case_image = image.copy()
            case_image[:, :, 3] = 0 if name == "alpha_zero" else 128
        if name == "vertex_sample":
            case_image = np.full((16,1024,4),255,np.uint8)
            case_image[:,:,:3] = np.linspace(0,255,1024).astype(np.uint8)[None,:,None]
        root = args.output / name
        root.mkdir(exist_ok=True)
        target = Mesh(v + [0, 0, z], f, source.uv * (1 - 2 * inset) + inset)
        # A missing half of the source makes target ray misses observable. The
        # bake caller must remove them from its mask before margin generation.
        case_source = source if name != "partial_source" else Mesh(v, f[:1], source.uv[:1])
        if name == "repeat_negative":
            case_source = source._replace(uv=source.uv * 3 - 1)
        if name == "half_byte":
            # Constant source UV removes the raster's deliberate subpixel offset
            # so this case samples the exact midpoint of byte values zero/one.
            case_source = source._replace(uv=np.full_like(source.uv,.5))
        if name == "folded":
            vertices = v.copy()
            vertices[:, 2] = [.1, .1, .3, .1]
            target = target._replace(vertices=vertices)
        if name == "collapsed":
            vertices = v.copy()
            vertices[:, 1] = .5
            vertices[:, 2] = .1
            target = target._replace(vertices=vertices)
        if name == "vertex_sample":
            target = target._replace(vertices=np.zeros_like(v))
        Blender._save_mesh(case_source, root / "source.npz")
        Blender._save_mesh(target, root / "target.npz", allow_degenerate=True)
        Image.fromarray(case_image).save(root / "source.png")
        config = BlenderBakeRequest(
            str((root / "source.npz").resolve()), str((root / "target.npz").resolve()),
            str((root / "source.png").resolve()), str((root / "blender.png").resolve()),
            resolution, extrusion, distance, margin,
        )
        Blender._run(installation, root.resolve(), config, timeout=120, threads=1)
        native = TextureBaking.bake_pixels(case_source, target, case_image, resolution=resolution,
            cage_extrusion=extrusion, max_ray_distance=distance, margin_pixels=margin)
        Image.fromarray(native).save(root / "rust.png")
        with Image.open(root / "blender.png") as reference:
            expected = np.asarray(reference.convert("RGBA"))
        valid = (native[:, :, 3] > 0) & (expected[:, :, 3] > 0)
        error = np.abs(native[:, :, :3].astype(float) - expected[:, :, :3])
        row = FixtureResult(case=name, reference_covered=int(np.count_nonzero(expected[:, :, 3])),
            rust_covered=int(np.count_nonzero(native[:, :, 3])),
            mae=float(error[valid].mean()) if valid.any() else None,
            maximum=float(error[valid].max()) if valid.any() else None)
        results.append(row)
        print(row.model_dump_json(), flush=True)
        # Individual records survive later runs of a different fixture.
        (root / "report.json").write_text(row.model_dump_json(indent=2) + "\n")



if __name__ == "__main__":
    main()
