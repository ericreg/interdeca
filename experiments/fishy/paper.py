"""Run the paper's original analyses in an isolated, inspectable directory.

The archived atlas is an experimental input. This runner does not substitute
the current geometry backend or a newly generated dataset for that input.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REFERENCE = Path(__file__).with_name("reference")
ANALYSES = (
    "build_cache",
    "e1_fidelity",
    "e1b_correspondence",
    "e2_preservation",
    "e2b_donorblocked",
    "e3_module_eda",
    "e4_byoa_descriptors",
    "e5b_expert_repr",
    "e6b_limits",
    "e2c_landmark_baseline",
    "e6_negative_results",
)
FIGURES = (
    "fig1_dataset",
    "fig2_fidelity",
    "fig2b_correspondence",
    "fig3_preservation",
    "fig4_module_eda",
    "fig5_byoa",
    "fig6_expert",
    "fig7_limits",
)
VERIFICATIONS = ("verify_e2_preservation", "verify_e2_followup")
STAGES = ANALYSES + FIGURES + VERIFICATIONS
DEPENDENCIES = {
    "e2b_donorblocked": ("e2_preservation",),
    "e2c_landmark_baseline": ("e1b_correspondence", "e2_preservation"),
    "e6_negative_results": ("e1_fidelity",),
    "fig2_fidelity": ("e1_fidelity",),
    "fig2b_correspondence": ("e1b_correspondence",),
    "fig3_preservation": (
        "e2_preservation",
        "e2b_donorblocked",
        "e2c_landmark_baseline",
    ),
    "fig4_module_eda": ("e3_module_eda",),
    "fig5_byoa": ("e4_byoa_descriptors",),
    "fig6_expert": ("e5b_expert_repr",),
    "fig7_limits": ("e6_negative_results", "e6b_limits"),
    "verify_e2_preservation": ("e2_preservation",),
    "verify_e2_followup": ("e2_preservation",),
}


class Paper:
    """Static operations for the source-pinned synthetic-paper reproduction."""

    @staticmethod
    def digest(path: Path) -> str:
        """Hash a file without loading multi-gigabyte inputs into memory."""
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
        return result.hexdigest()

    @staticmethod
    def write(path: Path, value: object) -> None:
        """Replace a metadata file atomically after serializing strict JSON."""
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    @staticmethod
    def donor_inputs(source: Path, donor_scans: Path) -> list[Path]:
        """Require the exact seven scan/mesh pairs named by the archived CSV."""
        with (source / "samples.csv").open(newline="") as stream:
            stems = sorted(
                {
                    row["sample_tag"].rsplit("_fishy", 1)[0]
                    for row in csv.DictReader(stream)
                }
            )
        if len(stems) != 7:
            raise ValueError("The paper requires seven distinct donor scans.")
        assets = [
            path
            for stem in stems
            for path in (
                donor_scans / "Texture" / f"{stem}.png",
                donor_scans / "Downsampled_decimate/Models" / f"{stem}.obj",
            )
        ]
        for path in assets:
            if not path.is_file():
                raise FileNotFoundError(path)
        return assets

    @staticmethod
    def attach_donor_scans(output: Path, donor_scans: Path) -> None:
        """Hash newly recovered scan inputs without replacing earlier analyses.

        Run this between stages. An existing scan attachment is refused; the
        record keeps when these additional figure inputs became available.
        """
        output, donor_scans = (
            path.expanduser().resolve() for path in (output, donor_scans)
        )
        if (
            output == donor_scans
            or output.is_relative_to(donor_scans)
            or donor_scans.is_relative_to(output)
        ):
            raise ValueError("Input and output directories must be disjoint.")
        record = json.loads((output / "run.json").read_text())
        if record["donor_scans"]:
            raise ValueError("Donor scans are already attached to this run.")
        assets = Paper.donor_inputs(Path(record["source"]), donor_scans)
        record["inputs"].extend(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": Paper.digest(path),
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in assets
        )
        record["donor_scans"] = str(donor_scans)
        record["donor_scans_attached_unix_seconds"] = time.time()
        Paper.write(output / "run.json", record)

    @staticmethod
    def prepare(
        source: Path, module_run: Path, output: Path, donor_scans: Path | None = None
    ) -> None:
        """Validate exact inputs, hash their identity, and stage the pinned scripts.

        Source and output must be disjoint. Existing outputs are refused so that
        a rerun cannot silently replace either the reference or an earlier run.
        """
        source, module_run, output = (
            path.expanduser().resolve() for path in (source, module_run, output)
        )
        donor_scans = donor_scans.expanduser().resolve() if donor_scans else None
        roots = [source, module_run] + ([donor_scans] if donor_scans else [])
        for root in roots:
            if (
                output == root
                or output.is_relative_to(root)
                or root.is_relative_to(output)
            ):
                raise ValueError("Input and output directories must be disjoint.")
        if output.exists():
            raise FileExistsError(output)
        required = [
            source / "samples.csv",
            module_run / "colorAnalysis/atlasModelUV.obj",
            module_run / "colorAnalysis/atlasLM.mrk.json",
        ]
        groups = (
            (source / "images", "*.png"),
            (source / "meshes", "*.obj"),
            (source / "landmarks", "*.mrk.json"),
            (module_run / "colorAnalysis/atlasTextures", "*.png"),
            (module_run / "colorAnalysis/resampledOBJ_withUV", "*.obj"),
            (module_run / "ATLAS/alignedModels", "*.obj"),
        )
        # All six asset groups are required by the paper, including aligned source
        # surfaces for fidelity. Missing groups must fail before creating a run.
        for directory, pattern in groups:
            assets = sorted(directory.glob(pattern))
            if directory.name == "atlasTextures":
                assets = [p for p in assets if "average" not in p.stem]
            if len(assets) != 250:
                raise ValueError(
                    f"Expected 250 assets in {directory}, found {len(assets)}."
                )
            required.extend(assets)
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        with (source / "samples.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        identifiers = [row["sample_tag"] for row in rows]
        if len(identifiers) != 250 or len(set(identifiers)) != 250:
            raise ValueError("The paper requires 250 unique ground-truth sample IDs.")
        for identifier in identifiers:
            # Ground truth, native assets and atlas assets must describe the same
            # specimens. Count agreement alone does not establish correspondence.
            paired = (
                source / "images" / f"{identifier}.png",
                source / "meshes" / f"{identifier}.obj",
                source / "landmarks" / f"{identifier}.mrk.json",
                module_run / "colorAnalysis/atlasTextures" / f"{identifier}.png",
                module_run
                / "colorAnalysis/resampledOBJ_withUV"
                / f"{identifier}_resampled.obj",
                module_run / "ATLAS/alignedModels" / f"{identifier}_align.obj",
            )
            for path in paired:
                if not path.is_file():
                    raise FileNotFoundError(path)
        if donor_scans:
            required.extend(Paper.donor_inputs(source, donor_scans))
        provenance = json.loads((REFERENCE / "upstream.json").read_text())
        for entry in provenance["files"]:
            path = REFERENCE / entry["path"]
            if Paper.digest(path) != entry["sha256"]:
                raise ValueError(f"Pinned reference changed: {entry['path']}")
        # Hash every consumed asset before execution. These inputs stay in place;
        # the run contains metadata and generated outputs, not another data copy.
        inputs = [
            {
                "path": str(p),
                "bytes": p.stat().st_size,
                "sha256": Paper.digest(p),
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in required
        ]
        output.mkdir(parents=True)
        shutil.copytree(REFERENCE, output / "source")
        work = output / "source/paper_synth"
        # Results committed upstream are a reference, never a warm-start result.
        (work / "results").rename(work / "results_ref")
        (work / "results").mkdir()
        (work / "figures").mkdir()
        (output / "logs").mkdir()
        packages = {
            p.metadata["Name"]: p.version for p in importlib.metadata.distributions()
        }
        Paper.write(
            output / "run.json",
            {
                "schema_version": 1,
                "upstream_revision": provenance["revision"],
                "source": str(source),
                "module_run": str(module_run),
                "donor_scans": str(donor_scans) if donor_scans else None,
                "python": sys.version,
                "platform": platform.platform(),
                "packages": packages,
                "inputs": inputs,
                "stages": {},
                "scope": "Archived 250-specimen analysis; does not rerun atlas construction.",
            },
        )

    @staticmethod
    def run(output: Path, stage: str) -> None:
        """Execute a named upstream stage and retain its log and exit status."""
        output = output.expanduser().resolve()
        record = json.loads((output / "run.json").read_text())
        reference = json.loads((output / "source/upstream.json").read_text())
        for item in reference["files"]:
            # The staged original results were deliberately moved to results_ref.
            relative = item["path"].replace(
                "paper_synth/results/", "paper_synth/results_ref/"
            )
            if Paper.digest(output / "source" / relative) != item["sha256"]:
                raise ValueError(f"Staged reference changed: {relative}")
        expected = {}
        for line in (
            (output / "source/paper_synth/requirements-paper.txt")
            .read_text()
            .splitlines()
        ):
            if "==" in line and not line.startswith("#"):
                package, version = line.split("==")
                expected[package] = version
        for package, version in expected.items():
            if importlib.metadata.version(package) != version:
                raise ValueError(
                    f"Install requirements-paper.txt: {package} must equal {version}."
                )
        if sys.version_info[:3] != (3, 12, 8):
            raise ValueError("Use the paper's Python 3.12.8 environment.")
        record["analysis_python"] = sys.version
        record["analysis_packages"] = expected
        stages = (
            ANALYSES
            if stage == "analyses"
            else FIGURES
            if stage == "figures"
            else (stage,)
        )
        if stage == "all":
            stages = ANALYSES + FIGURES
        elif stage == "figures-7-13":
            stages = FIGURES[1:]
        work = output / "source/paper_synth"
        for name in stages:
            if name == "fig1_dataset" and not record["donor_scans"]:
                raise ValueError(
                    "Figure 6 requires original donor scan textures; supply --donor-scans."
                )
            if name not in STAGES:
                raise ValueError(f"Unknown stage {name}")
            required_stages = DEPENDENCIES.get(name, ())
            if name != "build_cache":
                required_stages += ("build_cache",)
            for dependency in required_stages:
                if record["stages"].get(dependency, {}).get("exit_code") != 0:
                    raise ValueError(f"Run {dependency} before {name}.")
            # Modification time catches same-size edits. Hash changed inputs and
            # refuse altered bytes rather than silently accepting a stale cache.
            for item in record["inputs"]:
                path = Path(item["path"])
                if not path.is_file() or path.stat().st_size != item["bytes"]:
                    raise ValueError(f"Input missing or changed: {path}")
                if path.stat().st_mtime_ns != item.get("mtime_ns"):
                    if Paper.digest(path) != item["sha256"]:
                        raise ValueError(f"Input checksum changed: {path}")
                    item["mtime_ns"] = path.stat().st_mtime_ns
            # Starting a rerun replaces the evidence for this stage. Old success
            # records for its consumers cannot certify the newly computed data,
            # and an interrupted or failed rerun must leave no completion marker.
            affected = {name}
            while True:
                additions = {
                    candidate
                    for candidate in STAGES
                    if candidate not in affected
                    and (
                        "build_cache" in affected
                        or any(
                            dependency in affected
                            for dependency in DEPENDENCIES.get(candidate, ())
                        )
                    )
                }
                if not additions:
                    break
                affected.update(additions)
            for stale in affected:
                record["stages"].pop(stale, None)
            record["stages"][name] = {"exit_code": None, "status": "running"}
            (output / "completed.json").unlink(missing_ok=True)
            Paper.write(output / "run.json", record)
            # E6 resumes completed sections from its own checkpoint, including
            # the hole-exclusion analysis derived from E1. A new upstream run
            # must discard that checkpoint before it can certify fresh results.
            # Rerunning E6 itself retains same-input interrupted-run recovery.
            if name in ("build_cache", "e1_fidelity"):
                for filename in (
                    "e6_negative_results_partial.json",
                    "e6_negative_results_partial_arrays.npz",
                ):
                    (work / "cache" / filename).unlink(missing_ok=True)
            environment = os.environ.copy()
            # Full paper execution must not inherit a smoke-test loop count.
            environment.pop("E6_QUICK", None)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment.pop(key, None)
            environment.update(
                SYNTH_SRC=record["source"],
                SYNTH_MODULE_RUN=record["module_run"],
                MPLBACKEND="Agg",
                PYTHONUNBUFFERED="1",
            )
            if record["donor_scans"]:
                environment["SYNTH_DONOR_SCANS"] = record["donor_scans"]
            if name in ("e4_byoa_descriptors", "e5b_expert_repr"):
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    environment[key] = "4"
            start = time.time()
            with (output / "logs" / f"{name}.log").open("w") as log:
                process = subprocess.run(
                    [sys.executable, f"{name}.py"],
                    cwd=work,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record["stages"][name] = {
                "exit_code": process.returncode,
                "elapsed_seconds": time.time() - start,
                "thread_environment": {
                    k: v
                    for k, v in environment.items()
                    if k
                    in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "BLIS_NUM_THREADS",
                    )
                },
            }
            Paper.write(output / "run.json", record)
            print(
                f"{name}: exit {process.returncode}; log {output / 'logs' / (name + '.log')}",
                flush=True,
            )
            if process.returncode:
                raise RuntimeError(f"Stage {name} failed; inspect its log.")
        # A full completion marker is reserved for a fully successful figure run.
        if all(
            record["stages"].get(s, {}).get("exit_code") == 0
            for s in ANALYSES + FIGURES
        ):
            Paper.write(
                output / "completed.json",
                {
                    "upstream_revision": record["upstream_revision"],
                    "scope": record["scope"],
                },
            )


def main() -> None:
    """Provide source staging and explicit paper stages without loading the core."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--module-run", type=Path, required=True)
    prepare.add_argument("--donor-scans", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--stage",
        choices=("all", "analyses", "figures", "figures-7-13") + STAGES,
        required=True,
    )
    attach = commands.add_parser("attach-donor-scans")
    attach.add_argument("--output", type=Path, required=True)
    attach.add_argument("--donor-scans", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        Paper.prepare(args.source, args.module_run, args.output, args.donor_scans)
    elif args.command == "attach-donor-scans":
        Paper.attach_donor_scans(args.output, args.donor_scans)
    else:
        Paper.run(args.output, args.stage)


if __name__ == "__main__":
    main()
