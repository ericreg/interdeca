"""Build and run a relocatable, checksum-verified research bundle.

This bootstrap module uses only the standard library so a recipient can verify
the source, native data, and checkpoint before installing numerical packages.
Scientific configuration validation remains in serde.Config.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


class Reproduction:
    """Explicit packaging and execution operations; never reuse a prior run."""

    interpreter: Path | None = None

    @staticmethod
    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    @staticmethod
    def write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    @staticmethod
    def build(output: Path, inputs: Path, reference: Path, model: Path) -> None:
        """Copy actual inputs and current working source into a new directory."""
        repository = Path(__file__).resolve().parents[2]
        output = output.resolve()
        if output.exists():
            raise FileExistsError(output)
        config = json.loads((reference / "config.json").read_text())
        revision = config["model_revision"]
        provenance = repository / "data/learned-color/slack-32-clams-provenance.json"
        inventory = json.loads(provenance.read_text())["specimens"]
        if (
            sorted(row["specimen_id"] for row in inventory)
            != sorted(config["specimen_ids"])
            or len(inventory) != 32
        ):
            raise ValueError(
                "Reference does not identify the recovered 32-specimen cohort"
            )
        for row in inventory:
            for kind, directory, suffix in (
                ("mesh", "models", ".obj"),
                ("texture", "textures", None),
            ):
                matches = list(
                    (inputs / directory).glob(row["specimen_id"] + (suffix or ".*"))
                )
                if (
                    len(matches) != 1
                    or Reproduction.digest(matches[0]) != row[f"{kind}_sha256"]
                ):
                    raise ValueError(
                        f"Native input differs from recovered handoff: {row['specimen_id']} {kind}"
                    )
        required_model = (
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
        )
        for filename in required_model:
            if not (model / filename).is_file():
                raise FileNotFoundError(model / filename)
        # Overlay tracked dirty source and only this task's named new files.
        # Unrelated untracked notes must never silently enter a research bundle.
        names = (
            subprocess.check_output(["git", "ls-files", "-z"], cwd=repository)
            .decode()
            .split("\0")
        )
        names = sorted({name for name in names if name})
        names = sorted(
            set(names)
            | {
                "experiments/learned_color/reproduce.py",
                "experiments/learned_color/reproduce_check.py",
                "experiments/learned_color/BUNDLE_README.md",
                "experiments/learned_color/environment/requirements-linux-cuda.txt",
                "experiments/learned_color/reference/model-card.md",
            }
        )
        output.mkdir(parents=True)
        for name in names:
            original = repository / name
            if not original.is_file():
                continue
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        patch = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=repository
        )
        (output / "working-tree.patch").write_bytes(patch)
        Reproduction.write(
            output / "source-identity.json",
            {
                "head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repository
                )
                .decode()
                .strip(),
                "branch": subprocess.check_output(
                    ["git", "branch", "--show-current"], cwd=repository
                )
                .decode()
                .strip(),
                "working_tree_patch_sha256": hashlib.sha256(patch).hexdigest(),
                "source_scope": "Current tracked files with dirty changes plus explicitly named reproduction tooling/docs/environment/model-card",
                "python": "3.12.8",
            },
        )
        shutil.copytree(inputs, output / "inputs")
        # Historical metadata remains unchanged, even where it records old paths.
        # Only current configs below are executable after relocation.
        history = output / "reference" / "historical-2026-10-06-32"
        history.mkdir(parents=True)
        for filename in (
            "config.json",
            "run.json",
            "completed.json",
            "embedding_metadata.json",
            "crop_embeddings.npz",
            "specimen_embeddings.npz",
        ):
            shutil.copy2(reference / filename, history / filename)
        shutil.copytree(reference / "analysis", history / "analysis")
        evidence = repository / "docs/learned-color/evidence/2026-10-06-slack-32"
        shutil.copytree(evidence, output / "reference" / "historical-evidence")
        shutil.copy2(provenance, output / "input-provenance.json")
        Reproduction.write(
            output / "ACCESS.json",
            {
                "native_dataset": "32_clams.zip, Charlie Clark, 2026-05-26 research-channel handoff",
                "source_message": json.loads(provenance.read_text())["source_message"],
                "local_use": "User-authorized private research reproduction copy",
                "public_redistribution": "Not established by the available handoff; this bundle is not publicly published",
                "model": "facebook/ijepa_vith14_1k pinned model card declares cc-by-nc-4.0; see model/model-card.md; native data and source have separate terms",
                "model_card_url": f"https://huggingface.co/facebook/ijepa_vith14_1k/raw/{revision}/README.md",
                "source": "Attribution retained in source/experiments/learned_color/reference/upstream.json; no new license grant inferred",
            },
        )
        checkpoint = (
            output / "model/hub/models--facebook--ijepa_vith14_1k/snapshots" / revision
        )
        checkpoint.mkdir(parents=True)
        # Dereference the original Hub symlinks: no external blob or cache path is needed.
        for filename in required_model:
            shutil.copy2(model / filename, checkpoint / filename)
        Reproduction.write(
            output / "model-contract.json",
            {
                "repo_id": config["checkpoint"],
                "revision": revision,
                "files": {
                    name: Reproduction.digest(checkpoint / name)
                    for name in required_model
                },
                "fetch": "python reproduce.py fetch-model --cache .runtime/fresh-model/hub",
            },
        )
        for profile, ids in (
            ("full", config["specimen_ids"]),
            ("smoke", config["specimen_ids"][:4]),
        ):
            portable = dict(config)
            portable.update(
                profile=f"packaged-{profile}-documented-method-rerun",
                models_dir="../inputs/models",
                textures_dir="../inputs/textures",
                output_dir=f"../runs/{profile}",
                specimen_ids=ids,
                expected_pair_count=len(ids),
                device="cuda",
                allow_cpu=False,
            )
            Reproduction.write(output / "configs" / f"{profile}.json", portable)
        shutil.copy2(
            repository / "experiments/learned_color/requirements.txt",
            output / "requirements.txt",
        )
        shutil.copytree(
            repository / "experiments/learned_color/environment", output / "environment"
        )
        shutil.copy2(
            repository / "experiments/learned_color/reference/model-card.md",
            output / "model/model-card.md",
        )
        shutil.copy2(
            repository / "experiments/learned_color/BUNDLE_README.md",
            output / "README.md",
        )
        (output / "reproduce.py").write_text(
            '"""Portable entrypoint; run from any current directory."""\n'
            "import runpy\nimport sys\nfrom pathlib import Path\n"
            "root = Path(__file__).resolve().parent\n"
            'sys.argv[1:1] = ["--bundle", str(root)]\n'
            'runpy.run_path(str(root / "source/experiments/learned_color/reproduce.py"), run_name="__main__")\n'
        )
        Reproduction.write(
            output / "figure-map.json",
            {
                "current_population": {
                    "command": "python reproduce.py figures --run runs/full --output figures/full",
                    "source": "runs/full/analysis/coordinates.csv",
                    "meaning": "New 32-mussel documented-method PCA/UMAP; not certification of the draft's six-fish image",
                },
                "historical_population": {
                    "path": "reference/historical-evidence/population.png",
                    "meaning": "Preserved October 6 reconstruction; no retrospective completion/config authentication",
                },
            },
        )
        records = {}
        for path in sorted(output.rglob("*")):
            if path.is_file():
                records[path.relative_to(output).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": Reproduction.digest(path),
                }
        Reproduction.write(output / "manifest.json", {"version": 1, "files": records})
        (output / "manifest.sha256").write_text(
            Reproduction.digest(output / "manifest.json") + "\n"
        )
        print(
            json.dumps(
                {
                    "bundle": str(output),
                    "files": len(records),
                    "bytes": sum(r["bytes"] for r in records.values()),
                    "manifest_sha256": Reproduction.digest(output / "manifest.json"),
                },
                indent=2,
            )
        )

    @staticmethod
    def verify(bundle: Path) -> None:
        """Check every shipped byte and disallow unexpected executable source."""
        expected = (bundle / "manifest.sha256").read_text().strip()
        if Reproduction.digest(bundle / "manifest.json") != expected:
            raise ValueError("Bundle manifest checksum changed")
        records = json.loads((bundle / "manifest.json").read_text())["files"]
        for name, record in records.items():
            path = bundle / name
            if path.is_symlink() or any(
                parent.is_symlink()
                for parent in path.parents
                if parent.is_relative_to(bundle)
            ):
                raise ValueError(f"Symlink payload is not permitted: {name}")
            if not path.resolve().is_relative_to(bundle.resolve()):
                raise ValueError(f"Path escapes bundle: {name}")
            if (
                not path.is_file()
                or path.stat().st_size != record["bytes"]
                or Reproduction.digest(path) != record["sha256"]
            ):
                raise ValueError(f"Bundle checksum changed or file missing: {name}")
        extras = [
            p
            for p in (bundle / "source").rglob("*.py")
            if p.relative_to(bundle).as_posix() not in records
        ]
        if extras:
            raise ValueError(f"Unexpected source files: {extras}")
        print(f"Verified {len(records)} shipped files; manifest {expected}", flush=True)

    @staticmethod
    def environment(bundle: Path, cache: Path | None = None) -> dict[str, str]:
        """Use only declared model/cache paths, with offline inference by default."""
        env = os.environ.copy()
        # Older Transformers cache variables otherwise override HF_HUB_CACHE.
        # Remove inherited aliases so the selected declared cache is decisive.
        for name in (
            "TRANSFORMERS_CACHE",
            "PYTORCH_TRANSFORMERS_CACHE",
            "PYTORCH_PRETRAINED_BERT_CACHE",
        ):
            env.pop(name, None)
        runtime = bundle / ".runtime"
        runtime.mkdir(exist_ok=True)
        hub = cache or bundle / "model/hub"
        env.update(
            HF_HOME=str(runtime / "hf"),
            HF_HUB_CACHE=str(hub),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(runtime / "matplotlib"),
            NUMBA_CACHE_DIR=str(runtime / "numba"),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONNOUSERSITE="1",
            PYTHONUNBUFFERED="1",
            PYTHONPATH=str(bundle / "source"),
        )
        return env

    @staticmethod
    def python(bundle: Path) -> Path:
        if Reproduction.interpreter is not None:
            return Reproduction.interpreter
        return (
            bundle
            / ".venv"
            / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )

    @staticmethod
    def invoke(bundle: Path, arguments: list[str], cache: Path | None = None) -> None:
        subprocess.run(
            [str(Reproduction.python(bundle)), *arguments],
            cwd=bundle / "source",
            env=Reproduction.environment(bundle, cache),
            check=True,
        )

    @staticmethod
    def model_verify(bundle: Path, cache: Path) -> None:
        contract = json.loads((bundle / "model-contract.json").read_text())
        snapshot = (
            cache
            / ("models--" + contract["repo_id"].replace("/", "--"))
            / "snapshots"
            / contract["revision"]
        )
        for name, expected in contract["files"].items():
            if Reproduction.digest(snapshot / name) != expected:
                raise ValueError(f"Fetched checkpoint mismatch: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path.cwd())
    parser.add_argument(
        "--python",
        type=Path,
        help="Use an explicitly installed clean interpreter instead of bundle/.venv",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    for name in ("output", "inputs", "reference", "model-snapshot"):
        build.add_argument(f"--{name}", type=Path, required=True)
    sub.add_parser("verify")
    sub.add_parser("install")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--device", choices=("mps", "cuda", "cpu"), required=True)
    fetch = sub.add_parser("fetch-model")
    fetch.add_argument("--cache", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--profile", choices=("smoke", "full"), required=True)
    run.add_argument("--device", choices=("mps", "cuda", "cpu"), required=True)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--model-cache", type=Path)
    validate = sub.add_parser("validate")
    validate.add_argument("--run", type=Path, required=True)
    figures = sub.add_parser("figures")
    figures.add_argument("--run", type=Path, required=True)
    figures.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Virtualenv interpreters are often symlinks to the base Python executable.
    # Preserve the selected path: dereferencing it would discard the environment.
    Reproduction.interpreter = (
        args.python.expanduser().absolute() if args.python else None
    )
    bundle = args.bundle.expanduser().resolve()
    if args.command == "build":
        Reproduction.build(
            args.output, args.inputs, args.reference, args.model_snapshot
        )
        return
    Reproduction.verify(bundle)
    if args.command == "verify":
        return
    if args.command == "install":
        # Python and requirements are pinned. Platform-specific CUDA dependencies
        # are resolved by the matching Torch wheel and captured in the receipt.
        env = os.environ.copy()
        env["UV_CACHE_DIR"] = str(bundle / ".runtime/uv")
        env["UV_PYTHON_INSTALL_DIR"] = str(bundle / ".runtime/python")
        env["PYTHONNOUSERSITE"] = "1"
        if Reproduction.interpreter is None:
            subprocess.run(
                [
                    "uv",
                    "--no-config",
                    "venv",
                    "--python",
                    "3.12.8",
                    str(bundle / ".venv"),
                ],
                env=env,
                check=True,
            )
        requirements = bundle / "requirements.txt"
        if platform.system() == "Linux":
            requirements = bundle / "environment/requirements-linux-cuda.txt"
        subprocess.run(
            [
                "uv",
                "--no-config",
                "pip",
                "install",
                "--python",
                str(Reproduction.python(bundle)),
                "-r",
                str(requirements),
            ],
            env=env,
            check=True,
        )
        Reproduction.invoke(
            bundle,
            [
                "-m",
                "experiments.learned_color.reproduce_check",
                "environment",
                "--bundle",
                str(bundle),
            ],
        )
    elif args.command == "fetch-model":
        cache = (bundle / args.cache).resolve()
        contract = json.loads((bundle / "model-contract.json").read_text())
        env = Reproduction.environment(bundle, cache)
        env.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
        code = "from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1],revision=sys.argv[2],cache_dir=sys.argv[3],allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])"
        subprocess.run(
            [
                str(Reproduction.python(bundle)),
                "-c",
                code,
                contract["repo_id"],
                contract["revision"],
                str(cache),
            ],
            env=env,
            check=True,
        )
        Reproduction.model_verify(bundle, cache)
        print("Fetched checkpoint hashes match bundled pinned contract")
    elif args.command == "preflight":
        Reproduction.invoke(
            bundle,
            [
                "-m",
                "experiments.learned_color.reproduce_check",
                "preflight",
                "--bundle",
                str(bundle),
                "--device",
                args.device,
            ],
        )
    elif args.command == "run":
        cache = (
            (bundle / args.model_cache).resolve()
            if args.model_cache
            else bundle / "model/hub"
        )
        Reproduction.model_verify(bundle, cache)
        config = json.loads((bundle / "configs" / f"{args.profile}.json").read_text())
        config.update(device=args.device, allow_cpu=args.device == "cpu")
        # Runtime configs are separate from the immutable shipped profile and
        # expose any batch override in the new run's own recorded configuration.
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        for name in ("models_dir", "textures_dir", "output_dir"):
            config[name] = str((bundle / "configs" / config[name]).resolve())
        path = bundle / ".runtime" / f"{args.profile}.json"
        Reproduction.write(path, config)
        Reproduction.invoke(
            bundle,
            [
                "-m",
                "experiments.learned_color",
                "--config",
                str(path),
                "--stage",
                "all",
            ],
            cache,
        )
    elif args.command == "validate":
        Reproduction.invoke(
            bundle,
            [
                "-m",
                "experiments.learned_color.reproduce_check",
                "validate",
                "--bundle",
                str(bundle),
                "--run",
                str((bundle / args.run).resolve()),
            ],
        )
    elif args.command == "figures":
        Reproduction.invoke(
            bundle,
            [
                "-m",
                "experiments.learned_color.plot",
                "--run",
                str((bundle / args.run).resolve()),
                "--output",
                str((bundle / args.output).resolve()),
            ],
        )


if __name__ == "__main__":
    main()
