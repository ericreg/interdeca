"""Build and operate relocatable, hash-inventoried fish research bundles."""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiments.fishy.paper import ANALYSES, FIGURES, Paper


class Release:
    """Keep bundle inputs immutable and generated runs in explicit directories."""

    @staticmethod
    def write(path: Path, value: object) -> None:
        """Write readable strict JSON using the runner's atomic replacement."""
        Paper.write(path, value)

    @staticmethod
    def inside(root: Path, relative: str) -> Path:
        """Reject absolute paths, escapes, and symlink aliases in inventories."""
        path = root / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe bundle path: {relative}")
        if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
            raise ValueError(f"Symlink in bundle path: {relative}")
        return path

    @staticmethod
    def build(run: Path, upstream: Path, output: Path, data_only: bool) -> None:
        """Copy the actual archived inputs and current reviewed source snapshot.

        Interrupted construction can resume only for the identical input record.
        Finished bundles are never overwritten. No environments or caches copy.
        """
        run, upstream, output = (p.expanduser().resolve() for p in (run, upstream, output))
        root = Path(__file__).resolve().parents[2]
        record = json.loads((run / "run.json").read_text())
        families = {
            "synthetic": Path(record["source"]),
            "module": Path(record["module_run"]),
            "donor-scans": Path(record["donor_scans"]),
        }
        if (output / "manifest.json").exists():
            raise FileExistsError("Finished bundle already exists; use a new output.")
        for source in families.values():
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError("Bundle and source input roots must be disjoint.")
        identity = hashlib.sha256(
            json.dumps(record["inputs"], sort_keys=True).encode()
        ).hexdigest()
        marker = output / ".building.json"
        if output.exists():
            if not marker.is_file() or json.loads(marker.read_text())["identity"] != identity:
                raise ValueError("Existing directory is not this interrupted build.")
        else:
            output.mkdir(parents=True)
            Release.write(marker, {"identity": identity})
        inventory = []
        # Every consumed asset is copied independently; hard links would let a
        # later edit change both the historical source and the delivered bundle.
        for i, item in enumerate(record["inputs"]):
            source = Path(item["path"])
            family = next(k for k, base in families.items() if source.is_relative_to(base))
            relative = Path("data") / family / source.relative_to(families[family])
            target = output / relative
            if Paper.digest(source) != item["sha256"]:
                raise ValueError(f"Historical input changed: {source.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if Paper.digest(target) != item["sha256"]:
                    raise ValueError(f"Interrupted bundle asset changed: {relative}")
            else:
                temporary = target.with_suffix(target.suffix + ".copying")
                shutil.copyfile(source, temporary)
                if Paper.digest(temporary) != item["sha256"]:
                    raise ValueError(f"Copy verification failed: {relative}")
                temporary.replace(target)
            inventory.append({"path": relative.as_posix(), "bytes": item["bytes"],
                              "sha256": item["sha256"], "family": family})
            if i % 100 == 0:
                print(f"inputs: {i + 1}/{len(record['inputs'])}", flush=True)
        Release.write(output / "input-inventory.json", inventory)
        if data_only:
            print(f"Immutable data ready: {output / 'data'}", flush=True)
            return
        # Git lists tracked files, but bytes come from the working checkout so
        # the release includes reviewed fixes that have not been committed.
        paths = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=root
        ).decode().split("\0")
        paths += ["experiments/fishy/release.py", "experiments/fishy/smoke.py",
                  "experiments/fishy/RELEASE.md"]
        for relative in sorted(set(paths)):
            if not relative or relative.startswith("docs/paper/"):
                continue
            source = root / relative
            if source.is_symlink():
                raise ValueError(f"Source snapshot cannot contain symlinks: {relative}")
            if source.is_file():
                target = output / "source" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        (output / "source-changes.patch").write_bytes(subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root))
        reference = json.loads((root / "experiments/fishy/reference/upstream.json").read_text())
        author = output / "reference/author"
        local = output / "reference/local-2026-10-06"
        author.mkdir(parents=True, exist_ok=True)
        local.mkdir(parents=True, exist_ok=True)
        # Author arrays come from the pinned Git object, never the fresh local
        # results. Keeping both sets makes execution and historical parity distinct.
        prefix = "fishy_research/paper_synth/results/"
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", reference["revision"], prefix],
            cwd=upstream,
        ).decode().splitlines()
        for name in names:
            if Path(name).suffix not in (".json", ".npz"):
                continue
            with (author / Path(name).name).open("wb") as stream:
                subprocess.run(["git", "show", f"{reference['revision']}:{name}"],
                               cwd=upstream, stdout=stream, check=True)
        work = run / "source/paper_synth"
        for directory in ("results", "figures"):
            for source in sorted((work / directory).iterdir()):
                if source.is_file() and source.suffix in (".json", ".npz", ".png", ".pdf"):
                    target = local / directory / source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        config = {"schema_version": 1, "python": "3.12.8",
                  "requirements": "source/experiments/fishy/reference/paper_synth/requirements-paper.txt",
                  "source": "data/synthetic", "module_run": "data/module",
                  "donor_scans": "data/donor-scans", "author_arrays": "reference/author"}
        Release.write(output / "bundle.json", config)
        with (output / "data/synthetic/samples.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        Release.write(output / "specimens.json", [{"sample_id": row["sample_tag"],
            "donor_id": int(row["specimen_index"]),
            "donor_name": row["sample_tag"].rsplit("_fishy", 1)[0]} for row in rows])
        provenance = {
            "bundle_scope": "Full archived 250-specimen downstream paper reproduction; atlas and synthetic textures are fixed inputs.",
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
            "source_snapshot": "Current working files including dirty fixes; source-changes.patch is relative to HEAD; new release tools are included directly.",
            "author_revision": reference["revision"], "author_repository": reference["repository"],
            "input_assets": len(inventory), "input_bytes": sum(v["bytes"] for v in inventory),
            "input_origin": {"synthetic": "User-supplied cichlid-synth-decimated archive (July 16 synthetic cohort)",
                "module": "User-supplied paired July 16 ColorAtlas run, including aligned meshes and atlas landmarks",
                "donor-scans": "Seven original photogrammetric PNGs recovered from the user's data drive, paired with unchanged decimated donor mesh copies"},
            "rights": {"dataset_redistribution": "Unknown/unverified; this local bundle does not grant public redistribution rights.",
                "code_license": "No repository-wide LICENSE was present at this branch snapshot. Preserve upstream attribution; redistribution permission is not established here.",
                "access": "Actual inputs are included locally. No cloud credentials or external data paths are required."},
            "historical_parity": "DIFFERS_FROM_PINNED_REFERENCE; full strict comparison is expected to exit 1. Execution completion is a separate verdict.",
            "limitations": ["No regeneration of original synthetic textures or atlas construction.",
                "Archived landmark inconsistencies and degenerate triangles are preserved.",
                "Original platform, input hashes, and warm-cache history are unavailable.",
                "Hue-group labels follow author's KMeans inference; original mixture labels were not archived.",
                "Repeated-CV inferential intervals retain the author's dependence limitation."],
        }
        Release.write(output / "provenance.json", provenance)
        shutil.copyfile(root / "experiments/fishy/RELEASE.md", output / "README.md")
        (output / "reproduce.py").write_text(
            '"""Run the research bundle from any working directory."""\n'
            'import sys\nfrom pathlib import Path\n'
            'root = Path(__file__).resolve().parent\n'
            'sys.path.insert(0, str(root / "source"))\n'
            'from experiments.fishy.release import main\n'
            'main(bundle=root)\n'
        )
        files = []
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != marker:
                files.append({"path": path.relative_to(output).as_posix(),
                              "bytes": path.stat().st_size, "sha256": Paper.digest(path)})
        source_files = [f for f in files if f["path"].startswith("source/")]
        source_digest = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
        Release.write(output / "manifest.json", {"schema_version": 1,
            "source_tree_sha256": source_digest, "files": files})
        (output / "manifest.sha256").write_text(Paper.digest(output / "manifest.json") + "  manifest.json\n")
        marker.unlink()
        print(f"Bundle ready: {output}; {len(files)} files; source SHA256 {source_digest}")

    @staticmethod
    def verify(bundle: Path) -> dict:
        """Hash every inventoried payload; generated environments/runs are separate."""
        started = time.monotonic()
        manifest = json.loads((bundle / "manifest.json").read_text())
        expected = (bundle / "manifest.sha256").read_text().split()[0]
        if Paper.digest(bundle / "manifest.json") != expected:
            raise ValueError("Manifest checksum differs from manifest.sha256.")
        source_files = [r for r in manifest["files"] if r["path"].startswith("source/")]
        source_digest = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
        if source_digest != manifest["source_tree_sha256"]:
            raise ValueError("Source-tree identity differs from the manifest inventory.")
        known = {row["path"] for row in manifest["files"]}
        # Python bytecode is an interpreter by-product, but other unlisted
        # source files could introduce undeclared runtime dependencies.
        for path in (bundle / "source").rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                if path.relative_to(bundle).as_posix() not in known:
                    raise ValueError(f"Unlisted source file: {path.relative_to(bundle)}")
        for row in manifest["files"]:
            path = Release.inside(bundle, row["path"])
            if not path.is_file() or path.stat().st_size != row["bytes"] or Paper.digest(path) != row["sha256"]:
                raise ValueError(f"Bundle asset missing or changed: {row['path']}")
        result = {"status": "PASS", "files": len(manifest["files"]),
            "bytes": sum(v["bytes"] for v in manifest["files"]),
            "source_tree_sha256": manifest["source_tree_sha256"],
            "manifest_sha256": expected, "seconds": round(time.monotonic()-started, 3)}
        print(json.dumps(result, indent=2), flush=True)
        return result

    @staticmethod
    def environment(bundle: Path) -> None:
        """Fail before numerical work when the pinned interpreter/packages differ."""
        if sys.version_info[:3] != (3, 12, 8):
            raise ValueError("Use Python 3.12.8 and the pinned paper requirements.")
        requirements = bundle / json.loads((bundle / "bundle.json").read_text())["requirements"]
        for line in requirements.read_text().splitlines():
            if line and not line.startswith("#") and "==" in line:
                name, version = line.split("==")
                if importlib.metadata.version(name) != version:
                    raise ValueError(f"Install pinned requirement {name}=={version}.")


def main(bundle: Path | None = None) -> None:
    """Expose build, integrity, smoke, full-run, comparison and export commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=bundle)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--run", type=Path, required=True)
    build.add_argument("--upstream-checkout", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--data-only", action="store_true")
    commands.add_parser("verify")
    for name in ("smoke", "prepare", "run", "compare", "export"):
        child = commands.add_parser(name)
        child.add_argument("--output", type=Path, required=True)
        if name == "run":
            child.add_argument("--stage", default="all")
        if name == "compare":
            child.add_argument("--reference", choices=("local", "author"), default="author")
        if name == "export":
            child.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        Release.build(args.run, args.upstream_checkout, args.output, args.data_only)
        return
    if args.bundle is None:
        parser.error("Supply --bundle or use the bundle's reproduce.py entrypoint.")
    root = args.bundle.expanduser().resolve()
    Release.verify(root)
    if args.command == "verify":
        return
    config = json.loads((root / "bundle.json").read_text())
    output = args.output.expanduser().resolve()
    if args.command in ("smoke", "prepare", "run", "compare"):
        Release.environment(root)
    if args.command == "prepare":
        Paper.prepare(root / config["source"], root / config["module_run"], output,
                      root / config["donor_scans"])
    elif args.command == "run":
        # Child scientific processes use only their pinned environment and
        # writable run-local caches, independently of a user's home directory.
        os.environ.update(PYTHONNOUSERSITE="1", MPLCONFIGDIR=str(output/"runtime-cache/matplotlib"),
                          NUMBA_CACHE_DIR=str(output/"runtime-cache/numba"),
                          XDG_CACHE_HOME=str(output/"runtime-cache/xdg"))
        Paper.run(output, args.stage)
    elif args.command == "compare":
        # Keep independently reconstructed local results separate from the
        # author reference; never overwrite either to make a comparator pass.
        comparison = importlib.import_module("experiments.fishy.paper_compare")

        reference = root / (config["author_arrays"] if args.reference == "author"
            else "reference/local-2026-10-06/results")
        current = output / "source/paper_synth/results"
        report = {"reference": args.reference, "relative_tolerance": 1e-6,
                  "json": {}, "arrays": {},
                  "array_scope": "Numeric arrays compared with pickle disabled; object arrays unverified."}
        for name in comparison.REQUIRED_JSON:
            report["json"][name] = comparison.Compare.json(reference/name, current/name, 1e-6) if (
                (reference/name).is_file() and (current/name).is_file()) else {"status":"MISSING"}
        for name in comparison.REQUIRED_ARRAYS:
            report["arrays"][name] = comparison.Compare.arrays(reference/name, current/name, 1e-6) if (
                (reference/name).is_file() and (current/name).is_file()) else {"__file__":{"status":"MISSING"}}
        failed = any(r["status"] not in ("IDENTICAL", "CLOSE") for r in report["json"].values()) or any(
            r["status"] not in ("IDENTICAL", "CLOSE", "UNVERIFIED_OBJECT_ARRAY")
            for fields in report["arrays"].values() for r in fields.values())
        report["numerical_parity"] = "DIFFERS" if failed else "PASS"
        Release.write(output/f"comparison-{args.reference}.json", report)
        print(f"{args.reference} numerical comparison: {report['numerical_parity']}")
        raise SystemExit(1 if failed else 0)
    elif args.command == "smoke":
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        shutil.copytree(root/"source/experiments/fishy/reference", output/"reference",
                        ignore=shutil.ignore_patterns("results", "__pycache__"))
        environment = os.environ.copy()
        environment.update(SYNTH_SRC=str(root/config["source"]),
            SYNTH_MODULE_RUN=str(root/config["module_run"]), MPLBACKEND="Agg",
            PYTHONNOUSERSITE="1", MPLCONFIGDIR=str(output/"runtime-cache/matplotlib"),
            NUMBA_CACHE_DIR=str(output/"runtime-cache/numba"), XDG_CACHE_HOME=str(output/"runtime-cache/xdg"),
            PYTHONPATH=os.pathsep.join([str(root/"source"), str(output/"reference/paper_synth"),
                                      str(output/"reference/src")]),
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
            VECLIB_MAXIMUM_THREADS="1", NUMBA_NUM_THREADS="1")
        subprocess.run([sys.executable,"-m","experiments.fishy.smoke","--bundle",str(root),
                        "--output",str(output)],env=environment,cwd=output,check=True)
    elif args.command == "export":
        destination = args.destination.expanduser().resolve()
        if destination.exists():
            raise FileExistsError(destination)
        record = json.loads((output/"run.json").read_text())
        if not (output/"completed.json").is_file() or not all(
            record["stages"].get(s, {}).get("exit_code") == 0 for s in ANALYSES + FIGURES):
            raise ValueError("Export requires a completed analysis/figure run.")
        comparison = importlib.import_module("experiments.fishy.paper_compare")
        for name in comparison.REQUIRED_JSON + comparison.REQUIRED_ARRAYS:
            if not (output/"source/paper_synth/results"/name).is_file():
                raise ValueError(f"Completed run is missing required numerical evidence: {name}")
        # Completion records alone cannot associate a foreign run with this
        # bundle. Check the pinned author revision and every consumed input.
        provenance = json.loads((root/"provenance.json").read_text())
        if record["upstream_revision"] != provenance["author_revision"]:
            raise ValueError("Run author revision does not match this bundle.")
        inventory = json.loads((root/"input-inventory.json").read_text())
        expected_inputs = sorted((row["bytes"], row["sha256"]) for row in inventory)
        actual_inputs = sorted((row["bytes"], row["sha256"]) for row in record["inputs"])
        if actual_inputs != expected_inputs:
            raise ValueError("Run input content inventory does not match this bundle.")
        for stage in FIGURES:
            if not (output/"source/paper_synth/figures"/f"{stage}.png").is_file():
                raise ValueError(f"Completed run is missing figure: {stage}")
        destination.mkdir(parents=True)
        for relative in ("run.json", "completed.json", "comparison.json", "comparison-local.json", "comparison-author.json",
                         "source/paper_synth/results", "source/paper_synth/figures", "logs"):
            source = output / relative
            if source.is_file():
                shutil.copyfile(source, destination/source.name)
            elif source.is_dir():
                shutil.copytree(source, destination/source.name)
        inventory = [{"path":p.relative_to(destination).as_posix(), "bytes":p.stat().st_size,
                      "sha256":Paper.digest(p)} for p in sorted(destination.rglob("*")) if p.is_file()]
        Release.write(destination/"manifest.json", {"bundle_manifest_sha256":Paper.digest(root/"manifest.json"),
                                                  "files":inventory})


if __name__ == "__main__":
    main()
