"""Artifact I/O, strict identity validation and reproduction provenance."""

import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from experiments.fishy.serde import Asset, Config, Donor, Manifest, Sample


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
        # Frozen boundaries remain immutable. Pin validated replacements rather
        # than mutating paths/hashes after their original validation.
        donors = tuple(
            Donor.model_validate(
                donor.model_dump()
                | {
                    "mesh": Artifacts.pinned(donor.mesh, path.resolve().parent),
                    "landmarks": Artifacts.pinned(
                        donor.landmarks, path.resolve().parent
                    ),
                }
            )
            for donor in config.donors
        )
        return Config.model_validate(config.model_dump() | {"donors": donors})

    @staticmethod
    def pinned(asset: Asset, base: Path) -> Asset:
        """Return a newly validated absolute identity without modifying the input."""
        resolved = Artifacts.resolve(asset, base)
        return Asset(path=resolved, sha256=Artifacts.digest(resolved))

    @staticmethod
    def manifest(path: Path) -> Manifest:
        """Validate all asset hashes and identities before processing a population."""
        manifest = Manifest.model_validate_json(path.read_text())
        base = path.resolve().parent
        samples = tuple(
            Sample.model_validate(
                sample.model_dump()
                | {
                    "mesh": Artifacts.pinned(sample.mesh, base),
                    "landmarks": Artifacts.pinned(sample.landmarks, base),
                    "texture": Artifacts.pinned(sample.texture, base),
                }
            )
            for sample in manifest.samples
        )
        values = manifest.model_dump() | {"samples": samples}
        for name in ("atlas_mesh", "atlas_landmarks"):
            asset = getattr(manifest, name)
            if asset is not None:
                values[name] = Artifacts.pinned(asset, base)
        return Manifest.model_validate(values)

    @staticmethod
    def write(path: Path, value: dict) -> None:
        """Refuse accidental overwrite and reject NaN in machine-readable output."""
        # Encode before reserving files/directories: serialization failure must
        # not leave a new partial stage artifact that resembles valid evidence.
        encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(encoded)

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
