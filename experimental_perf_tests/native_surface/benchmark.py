"""Compare the native baker on preserved population geometry and Blender UVs.

Input archives are immutable. Timings separate loading from native computation;
reference refreshes include Blender transport, and are labeled accordingly.
"""

import argparse
import hashlib
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from interdeca import _native
from interdeca._blender_protocol import BlenderBakeRequest
from interdeca.blender import Blender
from interdeca.surface import TextureBaking
from tools.profiling.storage import read_mesh
from serde import Agreement, Measurement, Report


def sample_image(image: np.ndarray, uv: np.ndarray, *, production: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Sample the same surface positions in both images, including UV borders."""
    h, w = image.shape[:2]
    xy = uv * [w, -h] + [-.5, h - .5]
    ij = np.floor(xy).astype(np.int64)
    fraction = xy - ij
    rgb = np.zeros((len(uv), 3))
    alpha = np.zeros(len(uv))
    for dx in (0, 1):
        for dy in (0, 1):
            x = np.clip(ij[:, 0] + dx, 0, w - 1)
            y = np.clip(ij[:, 1] + dy, 0, h - 1)
            weight = (fraction[:, 0] if dx else 1 - fraction[:, 0]) * (fraction[:, 1] if dy else 1 - fraction[:, 1])
            if production:
                weight = weight * image[y, x, 3] / 255
                alpha += weight
            rgb += image[y, x, :3] * weight[:, None] / 255
    nearest = np.floor(xy + .5).astype(np.int64)
    covered = image[np.clip(nearest[:, 1], 0, h - 1), np.clip(nearest[:, 0], 0, w - 1), 3] > 0
    if production:
        rgb = np.divide(rgb, alpha[:, None], out=np.zeros_like(rgb), where=alpha[:, None] > 0)
        covered = alpha >= .99
    return rgb, covered


def agreement(reference: np.ndarray, native: np.ndarray, uv: np.ndarray, *, production: bool = False, native_uv: np.ndarray | None = None) -> Agreement:
    a, covered_a = sample_image(reference, uv, production=production)
    b, covered_b = sample_image(native, uv if native_uv is None else native_uv, production=production)
    joint = covered_a & covered_b
    error = np.abs(a[joint] - b[joint])
    mean = float(error.mean()) if joint.any() else None
    p95 = float(np.quantile(error.max(axis=1), .95)) if joint.any() else None
    loss = float(np.mean(covered_a & ~covered_b) * 100)
    gain = float(np.mean(~covered_a & covered_b) * 100)
    return Agreement(samples=len(uv), joint_samples=int(joint.sum()), mean_rgb=mean,
        p95_max_channel=p95, coverage_loss_pp=loss, coverage_gain_pp=gain,
        passed=mean is not None and mean <= 1/255 and p95 <= 5/255 and loss <= .1)


def samples(mesh, count: int) -> np.ndarray:
    triangles = mesh.vertices[mesh.faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    rng = np.random.default_rng(413)
    faces = rng.choice(len(area), count, p=area / area.sum())
    random = rng.random((count, 2))
    root = np.sqrt(random[:, 0])
    weights = np.column_stack((1 - root, root * (1 - random[:, 1]), root * random[:, 1]))
    return np.einsum("nc,ncd->nd", weights, mesh.uv[faces])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=Path("build/profiling/parry-47-2026-10-04/current"))
    parser.add_argument("--inputs", type=Path, default=Path.home() / "Library/Application Support/InterDeCA/full-population-2026-10-03/inputs")
    parser.add_argument("--output", type=Path, default=Path("build/native-surface/comparison"))
    parser.add_argument("--backend", choices=("native", "morphoweave"), default="native")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--float32", action="store_true")
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--limit", type=int, default=47)
    parser.add_argument("--specimen", help="Select one specimen for a targeted diagnostic.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--refresh-reference", action="store_true")
    parser.add_argument("--reference-dir", type=Path, help="Reuse a previous refresh's specimen/blender.png files.")
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 47 or args.repeat < 1:
        parser.error("Use 1–47 specimens and at least one repetition.")
    if args.refresh_reference and args.reference_dir:
        parser.error("Choose either refreshing or reusing references.")
    if args.margin != 2 and not (args.refresh_reference or args.reference_dir):
        parser.error("Captured references use margin 2; refresh other settings.")
    started = time.perf_counter()
    implementation = hashlib.sha256()
    project = Path(__file__).resolve().parents[2]
    for code in sorted((project / "rust/src").rglob("*.rs")) + [project / "src/interdeca/surface.py", Path(__file__)]:
        implementation.update(str(code.relative_to(project)).encode() + b"\0")
        implementation.update(code.read_bytes())
    extension_sha256 = hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    root = next((args.captures / f"{args.backend}-baseline-1" / "atlas").iterdir())
    installation = Blender.check_available() if args.refresh_reference else None
    if args.resolution != 512 and not (args.refresh_reference or args.reference_dir):
        parser.error("Non-512 references need --refresh-reference or --reference-dir.")
    records = []
    textures = sorted((args.inputs / "textures").iterdir())
    if args.specimen:
        textures = [p for p in textures if p.stem == args.specimen]
    textures = textures[:args.limit]
    if len(textures) != (1 if args.specimen else args.limit):
        parser.error("The requested population is incomplete.")
    for path in textures:
        specimen = path.stem
        source = read_mesh(root / "aligned" / f"{specimen}.obj")
        target = read_mesh(root / "resampled" / f"{specimen}.obj", allow_degenerate=True)
        if args.float32:
            source = source._replace(vertices=source.vertices.astype(np.float32).astype(np.float64))
            target = target._replace(vertices=target.vertices.astype(np.float32).astype(np.float64))
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGBA"))
        output = args.output / specimen
        output.mkdir(exist_ok=True)
        reference_path = root / "textures" / f"{specimen}.png"
        if args.reference_dir:
            reference_path = args.reference_dir / specimen / "blender.png"
        blender_seconds = None
        if installation is not None:
            # Time identical Blender geometry/settings including its required
            # transport. Source decode for native is reported outside its kernel.
            start = time.perf_counter()
            Blender._save_mesh(source, output / "source.npz")
            Blender._save_mesh(target, output / "target.npz", allow_degenerate=True)
            reference_path = output / "blender.png"
            config = BlenderBakeRequest(str((output / "source.npz").resolve()),
                str((output / "target.npz").resolve()), str(path.resolve()),
                str(reference_path.resolve()), (args.resolution, args.resolution), .001, 0, args.margin)
            Blender._run(installation, output.resolve(), config, timeout=600, threads=1)
            blender_seconds = time.perf_counter() - start
            (output / "source.npz").unlink()
            (output / "target.npz").unlink()
        with Image.open(reference_path) as image:
            reference = np.asarray(image.convert("RGBA"))
        if reference.shape[:2] != (args.resolution, args.resolution):
            raise ValueError("Reference dimensions differ from the requested bake.")
        uv = samples(target, 100_000)
        if args.layout:
            with np.load(args.layout, allow_pickle=False) as layout:
                if not np.array_equal(layout["faces"], target.faces):
                    raise ValueError("Layout face indexing differs from the captured target.")
                target = target._replace(uv=layout["uv"])
        native_uv = samples(target, 100_000)
        digest = hashlib.sha256(path.read_bytes())
        for mesh in (source, target):
            for array in (mesh.vertices, mesh.faces, mesh.uv):
                digest.update(array.tobytes())
        for repeat in range(args.repeat):
            start = time.perf_counter()
            native = TextureBaking.bake_pixels(source, target, pixels,
                resolution=(args.resolution, args.resolution), cage_extrusion=.001, margin_pixels=args.margin)
            elapsed = time.perf_counter() - start
            result = agreement(reference, native, uv, native_uv=native_uv)
            row = Measurement(specimen=specimen, backend=args.backend, resolution=args.resolution,
                rust_seconds=elapsed, blender_seconds=blender_seconds, agreement=result,
                analysis_agreement=agreement(reference, native, uv, production=True, native_uv=native_uv),
                input_sha256=digest.hexdigest(), reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                output_sha256=hashlib.sha256(native.tobytes()).hexdigest(), repeat=repeat)
            records.append(row)
            print(row.model_dump_json(), flush=True)
            if repeat == 0:
                Image.fromarray(native).save(output / "rust.png")
            # Persist after each specimen so interruption never erases evidence.
            report = Report(measurements=records,
                all_passed=all(r.agreement.passed and r.analysis_agreement.passed for r in records),
                scope="custom_uv_and_baker" if args.layout else "baker_on_reference_uv",
                platform=platform.platform(), python=sys.version,
                implementation_sha256=implementation.hexdigest(), extension_sha256=extension_sha256,
                elapsed_seconds=time.perf_counter()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))
            (args.output / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    return 0 if report.all_passed else 2


if __name__ == "__main__":
    sys.exit(main())
