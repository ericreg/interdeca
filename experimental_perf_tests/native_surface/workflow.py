"""Measure complete CLI workflows in fresh processes before switching production.

Only the disposable reference child invokes Blender. Both
conditions retain pre-cleanup geometry and downstream arrays for parity checks.
The parent samples simultaneous process-tree RSS, including Blender children.
"""

import argparse
import hashlib
import os
import signal
import shutil
import subprocess
import sys
import time
from functools import wraps
from pathlib import Path

import numpy as np
import psutil

from interdeca import _native
from interdeca.blender import Blender
from interdeca.color_analysis import ColorAnalysis
from interdeca.population import Population
from interdeca.surface import MeshProcessing, TextureBaking
from tools.profiling.__main__ import fingerprint, snapshot
from tools.profiling.capture import run_child
from tools.profiling.serde import Config, Run
from compact import compact


def install_capture(config: Config, native: bool) -> None:
    """Save the same boundaries for both conditions, without changing results."""
    evidence = config.output / "numerical"
    evidence.mkdir()
    original_clean = Blender.clean_atlas
    original_uv = Blender.prepare_atlas_uv
    original_bake = Blender.bake_specimen_texture
    native_clean = MeshProcessing.clean_atlas
    native_uv = MeshProcessing.prepare_atlas_uv
    native_bake = TextureBaking.bake_specimen_texture
    # Reference-only discovery keeps the production condition free of Blender.
    installation = None if native else Blender.check_available()

    def clean(atlas, *, merge_distance=.0001, timeout=600, threads=1):
        Blender._save_mesh(atlas.mesh, evidence / "pre-cleanup.npz")
        result = (native_clean(atlas, merge_distance=merge_distance) if native
                  else original_clean(atlas, installation, merge_distance=merge_distance, timeout=timeout, threads=threads))
        Blender._save_mesh(result.mesh, evidence / "post-cleanup.npz")
        return result

    def uv(atlas, *, angle_limit_degrees=66, island_margin=.02, timeout=600, threads=1):
        result = (native_uv(atlas, angle_limit_degrees=angle_limit_degrees, island_margin=island_margin) if native
                  else original_uv(atlas, installation, angle_limit_degrees=angle_limit_degrees, island_margin=island_margin, timeout=timeout, threads=threads))
        Blender._save_mesh(result.mesh, evidence / "post-uv.npz")
        return result

    def bake(specimen, target, output_path, *, resolution=(2048,2048), cage_extrusion=.01,
             max_ray_distance=0, margin_pixels=8, timeout=1800, threads=1, overwrite=False):
        if native:
            return native_bake(specimen, target, output_path, resolution=resolution,
                cage_extrusion=cage_extrusion, max_ray_distance=max_ray_distance, margin_pixels=margin_pixels, overwrite=overwrite)
        return original_bake(specimen, target, installation, output_path, resolution=resolution,
            cage_extrusion=cage_extrusion, max_ray_distance=max_ray_distance, margin_pixels=margin_pixels,
            timeout=timeout, threads=threads, overwrite=overwrite)

    # Capture the production entry points; only reference wrappers launch Blender.
    MeshProcessing.clean_atlas = staticmethod(clean)
    MeshProcessing.prepare_atlas_uv = staticmethod(uv)
    TextureBaking.bake_specimen_texture = staticmethod(bake)

    original_sample = ColorAnalysis.sample_specimen_colors
    original_process = ColorAnalysis.process_specimen_colors
    original_features = Population.assemble_features
    original_fit = Population.fit_model

    @wraps(original_sample)
    def sample(*args, **kwargs):
        result = original_sample(*args, **kwargs)
        np.savez(evidence / f"sample-{result.specimen_id}.npz", rgb=result.rgb, valid=result.valid)
        return result

    @wraps(original_process)
    def process(*args, **kwargs):
        result = original_process(*args, **kwargs)
        np.savez(evidence / f"colors-{result.specimen_id}.npz", rgb=result.rgb, lab=result.lab, valid=result.valid)
        return result

    @wraps(original_features)
    def features(*args, **kwargs):
        result = original_features(*args, **kwargs)
        np.savez(evidence / "features.npz", values=result.values, specimen_ids=result.specimen_ids, feature_names=result.feature_names)
        return result

    @wraps(original_fit)
    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        np.savez(evidence / "pca.npz", scores=result.scores, center=result.center, scale=result.scale,
                 components=result.estimator.components_, variance=result.estimator.explained_variance_ratio_)
        return result

    ColorAnalysis.sample_specimen_colors = staticmethod(sample)
    ColorAnalysis.process_specimen_colors = staticmethod(process)
    Population.assemble_features = staticmethod(features)
    Population.fit_model = staticmethod(fit)


def measure(config: Config, native: bool) -> Run:
    """Run one condition with identical instrumentation and a bounded timeout."""
    # The 47-specimen workflow writes about six GiB before it can compact the
    # results. Refuse a known insufficient disk instead of failing mid-export.
    project=Path(__file__).resolve().parents[2]
    if shutil.disk_usage(project).free < 7 * 2**30:
        raise OSError("Full-population captures require at least 7 GiB free before running.")
    config.output.mkdir(parents=True)
    path = config.output / "config.json"
    path.write_text(config.model_dump_json(indent=2) + "\n")
    command = [sys.executable, str(Path(__file__).resolve()), "--child", str(path)]
    if native:
        command.append("--native")
    samples = []
    started = time.perf_counter()
    with (config.output / "console.log").open("w") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        monitored = psutil.Process(child.pid)
        try:
            with (config.output / "resources.jsonl").open("w") as stream:
                while child.poll() is None:
                    if time.perf_counter() - started > 2400:
                        raise TimeoutError("Full workflow exceeded 40 minutes.")
                    before = time.perf_counter()
                    row = snapshot(monitored, full=len(samples) % 5 == 0)
                    samples.append(row)
                    stream.write(row.model_dump_json() + "\n")
                    stream.flush()
                    time.sleep(max(.001, .1 - (time.perf_counter()-before)))
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    elapsed = time.perf_counter() - started
    result = Run(config=config, exit_code=child.returncode, wall_seconds=elapsed,
        sample_interval_seconds=.1,
        parent_peak_rss_bytes=max((p.rss_bytes for s in samples for p in s.processes if p.parent),default=0),
        tree_peak_rss_bytes=max((sum(p.rss_bytes for p in s.processes) for s in samples),default=0),
        parent_peak_uss_bytes=max((p.uss_bytes for s in samples for p in s.processes if p.parent and p.uss_bytes is not None),default=None),
        tree_peak_uss_bytes=max((sum(p.uss_bytes for p in s.processes) for s in samples if s.processes and all(p.uss_bytes is not None for p in s.processes)),default=None),
        sampled_processes=len({(p.pid,p.created) for s in samples for p in s.processes}),
        output_hashes=fingerprint(config.output) if child.returncode == 0 else {})
    (config.output / "run.json").write_text(result.model_dump_json(indent=2)+"\n")
    with Path(_native.__file__).open("rb") as stream:
        extension_hash = hashlib.file_digest(stream,"sha256").hexdigest()
    (config.output / "extension.sha256").write_text(extension_hash+"\n")
    print(f"{config.output}: exit={result.exit_code}, wall={elapsed:.2f}s, tree RSS={result.tree_peak_rss_bytes/2**20:.1f} MiB",flush=True)
    # Compaction follows the measurement and hashing. Repeated captures can
    # share verified immutable archives instead of accumulating ASCII copies.
    if result.exit_code == 0:
        compact(config.output,(project/'build/profiling',project/'build/native-surface'))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child",type=Path)
    parser.add_argument("--native",action="store_true")
    parser.add_argument("--backend",choices=("native","morphoweave"),default="native")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--inputs",type=Path,default=Path.home()/"Library/Application Support/InterDeCA/full-population-2026-10-03/inputs")
    args=parser.parse_args()
    if args.child:
        config=Config.model_validate_json(args.child.read_text())
        install_capture(config,args.native)
        run_child(config)
        return 0
    if args.output is None:
        parser.error("--output is required.")
    config=Config(backend=args.backend,mode="baseline",inputs=args.inputs.resolve(),output=args.output.resolve())
    return measure(config,args.native).exit_code


if __name__ == "__main__":
    sys.exit(main())
