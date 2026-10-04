"""Compare Rust smoothing with a Git revision on three already-baked clam textures."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
import types
from pathlib import Path
from typing import NamedTuple

import numpy as np

from interdeca import _native
from interdeca.color_analysis import ColorAnalysis
from interdeca.geometry import Geometry
from interdeca.io import MeshIO
from interdeca.models import BakedTexture


class TimedSample(NamedTuple):
    wall_seconds: float
    cpu_seconds: float


def measure(operation, texture, analysis, steps):
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    result = operation(texture, analysis, smoothing_steps=steps)
    return result, TimedSample(time.perf_counter() - start_wall, time.process_time() - start_cpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--reference-revision", default="584fa4f")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    # Read the actual predecessor from Git instead of keeping a second production
    # implementation. Its adjacency rows are views of the same packed graph.
    source = subprocess.check_output(
        ["git", "show", f"{args.reference_revision}:src/interdeca/color_analysis.py"],
        cwd=repository, text=True,
    )
    previous = types.ModuleType("reference_color_analysis")
    exec(compile(source, f"{args.reference_revision}:color_analysis.py", "exec"), previous.__dict__)
    operations = {"python": previous.ColorAnalysis.sample_specimen_colors,
                  "rust": ColorAnalysis.sample_specimen_colors}
    evidence = {"reference_revision": args.reference_revision,
                "reference_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "platform": platform.platform(), "numpy": np.__version__,
                "native_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
                "backends": {}}
    for backend in ("native", "morphoweave"):
        run = args.reference_run / f"{backend}-baseline-1"
        atlas_path, = (run / "atlas").glob("*/atlas.obj")
        analysis = Geometry.prepare_analysis(MeshIO.read(atlas_path), sample_count=1000)
        graph = analysis.adjacency
        reference = analysis._replace(adjacency=tuple(
            graph.indices[start:end] for start, end in zip(graph.offsets[:-1], graph.offsets[1:], strict=True)
        ))
        textures = [BakedTexture(path.stem, path, analysis.mesh.topology_key, (512, 512))
                    for path in sorted((atlas_path.parent / "textures").glob("UF_IZ_*.png"))]
        assert len(textures) == 3, "Use the three-clam fixture, not a different population."
        checks = []
        # Zero, one and repeated passes exercise the public operation, including
        # image decoding, UV sampling, missing coverage and output validation.
        for steps in (0, 1, 3):
            for texture in textures:
                old, _ = measure(operations["python"], texture, reference, steps)
                new, _ = measure(operations["rust"], texture, analysis, steps)
                np.testing.assert_array_equal(new.valid, old.valid)
                np.testing.assert_allclose(new.rgb, old.rgb, rtol=2e-14, atol=2e-15)
                assert new.specimen_id == old.specimen_id and new.sampling_key == old.sampling_key
                checks.append({"specimen": texture.specimen_id, "steps": steps,
                               "max_absolute_error": float(np.max(np.abs(new.rgb - old.rgb))),
                               "byte_identical_rgb": new.rgb.tobytes() == old.rgb.tobytes()})
        times = {"python": [], "rust": []}
        # Alternate execution order over three warm-file repeats. This measures
        # the full sampling operation, rather than excluding Rust's call overhead.
        for repeat in range(3):
            for name in (("python", "rust") if repeat % 2 == 0 else ("rust", "python")):
                totals = [measure(operations[name], texture,
                                  reference if name == "python" else analysis, 1)[1]
                          for texture in textures]
                times[name].append({"wall_seconds": sum(t.wall_seconds for t in totals),
                                    "cpu_seconds": sum(t.cpu_seconds for t in totals)})
        medians = {name: statistics.median(t["wall_seconds"] for t in rows)
                   for name, rows in times.items()}
        assert medians["rust"] < medians["python"], (backend, medians)
        evidence["backends"][backend] = {
            "faces": len(analysis.mesh.faces), "samples": len(analysis.sample_faces),
            "equivalence": checks, "three_specimen_timings": times,
            "median_wall_seconds": medians, "speedup": medians["python"] / medians["rust"],
        }
        print(backend, medians, "speedup", medians["python"] / medians["rust"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
