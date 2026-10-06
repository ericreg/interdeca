"""Artifact I/O, strict identity validation and reproduction provenance."""

import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from experiments.fishy.serde import Asset, Config, Manifest


class Artifacts:
    """File boundaries use JSON-compatible dictionaries only for serialization."""

    @staticmethod
    def digest(path: Path) -> str:
        """Hash large meshes without holding their full bytes in memory."""
        with path.open("rb") as stream:
            digest = sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def resolve(asset: Asset, base: Path) -> Path:
        """Resolve relative to the declaring file, and verify a supplied checksum."""
        path = (base / asset.path.expanduser()).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Required asset not found: {path}")
        if asset.sha256 is not None and Artifacts.digest(path) != asset.sha256:
            raise ValueError(f"SHA256 mismatch: {path}")
        return path

    @staticmethod
    def asset(path: Path, base: Path) -> Asset:
        """Published manifests use movable relative paths within their run."""
        return Asset(path=path.relative_to(base), sha256=Artifacts.digest(path))

    @staticmethod
    def config(path: Path) -> Config:
        """Resolve and pin every donor asset before any expensive operation."""
        config = Config.model_validate_json(path.read_text())
        for donor in config.donors:
            for asset in (donor.mesh, donor.landmarks):
                asset.path = Artifacts.resolve(asset, path.resolve().parent)
                asset.sha256 = Artifacts.digest(asset.path)
        return config

    @staticmethod
    def manifest(path: Path) -> Manifest:
        """Validate all asset hashes and identities before processing a population."""
        manifest = Manifest.model_validate_json(path.read_text())
        assets = [
            item
            for sample in manifest.samples
            for item in (sample.mesh, sample.landmarks, sample.texture)
        ]
        assets += [
            item
            for item in (manifest.atlas_mesh, manifest.atlas_landmarks)
            if item is not None
        ]
        for asset in assets:
            asset.path = Artifacts.resolve(asset, path.resolve().parent)
            asset.sha256 = Artifacts.digest(asset.path)
        return manifest

    @staticmethod
    def write(path: Path, value: dict) -> None:
        """Refuse accidental overwrite and reject NaN in machine-readable output."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")

    @staticmethod
    def start(output: Path, stage: str, inputs: list[Path]) -> None:
        """Reserve a new run directory and record exactly which code was executed."""
        output.mkdir(parents=True, exist_ok=False)
        root = Path(__file__).resolve().parents[2]
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            if shutil.which("git")
            else None
        )
        # A source hash identifies unpublished edits as well as tagged releases.
        sources = sorted((root / "src/interdeca").glob("*.py"))
        sources += sorted(Path(__file__).parent.glob("*.py"))
        sources += [
            path
            for path in (root / "pyproject.toml", root / "uv.lock")
            if path.is_file()
        ]
        Artifacts.write(
            output / "provenance.json",
            {
                "stage": stage,
                "started_utc": datetime.now(UTC).isoformat(),
                "commit": commit.stdout.strip()
                if commit is not None and commit.returncode == 0
                else None,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "versions": {
                    name: version(name)
                    for name in (
                        "numpy",
                        "scipy",
                        "scikit-learn",
                        "scikit-image",
                        "pillow",
                        "vtk",
                        "pydantic",
                    )
                },
                "inputs": {
                    str(path.resolve()): Artifacts.digest(path) for path in inputs
                },
                "source_sha256": {
                    str(path.relative_to(root)): Artifacts.digest(path)
                    for path in sources
                },
                "status_note": "A complete.json file is written only when this stage succeeds.",
            },
        )

    @staticmethod
    def complete(output: Path) -> None:
        """Keep a partial run inspectable without letting it look successful."""
        Artifacts.write(
            output / "complete.json", {"completed_utc": datetime.now(UTC).isoformat()}
        )
