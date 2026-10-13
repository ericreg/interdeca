"""Bounded numerical and environment checks for a packaged reconstruction."""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from PIL import Image

from experiments.learned_color.models import AppearanceOptions, RunRecord
from experiments.learned_color.reproduce import Reproduction
from experiments.learned_color.serde import StageData


class Checks:
    """Check actual inputs and arrays, keeping historical comparisons explicit."""

    @staticmethod
    def environment(bundle: Path) -> dict:
        if sys.version_info[:3] != (3, 12, 8):
            raise ValueError("Use the pinned Python 3.12.8 interpreter")
        expected = dict(
            line.split("==", 1)
            for line in (bundle / "requirements.txt").read_text().splitlines()
            if line and not line.startswith("#")
        )
        for name, version in expected.items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Package version differs: {name} must be {version}")
        installed = {
            p.metadata["Name"]: p.version for p in importlib.metadata.distributions()
        }
        report = {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "required_versions_match": True,
            "installed": installed,
        }
        Reproduction.write(bundle / ".runtime/environment.json", report)
        (bundle / ".runtime/resolved-requirements.txt").write_text(
            "".join(
                f"{name}=={version}\n"
                for name, version in sorted(
                    installed.items(), key=lambda item: item[0].lower()
                )
            )
        )
        return report

    @staticmethod
    def preflight(bundle: Path, device: str) -> dict:
        environment = Checks.environment(bundle)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable")
        # A tiny real render exposes a missing OpenGL/EGL backend before inference.
        plotter = pv.Plotter(off_screen=True, window_size=(64, 64))
        try:
            plotter.add_mesh(pv.Sphere())
            pixels = plotter.screenshot(return_img=True)
            renderer = plotter.ren_win.GetClassName()
        finally:
            plotter.close()
        if pixels.shape[:2] != (64, 64) or np.std(pixels) == 0:
            raise ValueError("Headless render preflight produced an invalid image")
        probe = torch.ones(1, device=device)
        if probe.cpu().item() != 1:
            raise ValueError("Requested compute device failed")
        report = {
            "environment": environment,
            "device": device,
            "render_window": renderer,
            "headless_render_pass": True,
            "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        }
        Reproduction.write(bundle / ".runtime" / f"preflight-{device}.json", report)
        return report

    @staticmethod
    def identity(bundle: Path, config: AppearanceOptions, record: RunRecord) -> str:
        """Bind a run to a shipped protocol, complete source, and native cohort."""
        # Only declared execution/location choices may differ after relocation.
        overrides = {
            "models_dir",
            "textures_dir",
            "output_dir",
            "device",
            "allow_cpu",
            "batch_size",
        }
        protocol = {
            name: value
            for name, value in config._asdict().items()
            if name not in overrides
        }
        profiles = [
            name
            for name in ("smoke", "full")
            if {
                field: value
                for field, value in StageData.load_options(
                    bundle / "configs" / f"{name}.json"
                )
                ._asdict()
                .items()
                if field not in overrides
            }
            == protocol
        ]
        if len(profiles) != 1 or config.allow_cpu != (config.device == "cpu"):
            raise ValueError("Run scientific protocol differs from packaged profiles")
        contract = json.loads((bundle / "model-contract.json").read_text())
        if (config.checkpoint, config.model_revision) != (
            contract["repo_id"],
            contract["revision"],
        ):
            raise ValueError("Run checkpoint differs from packaged model contract")
        prefix = "source/experiments/learned_color/"
        files = json.loads((bundle / "manifest.json").read_text())["files"]
        sources = {
            name.removeprefix(prefix): row["sha256"]
            for name, row in files.items()
            if name.startswith(prefix) and name.endswith(".py")
        }
        if not sources or record.source_sha256 != sources:
            raise ValueError("Run source inventory differs from packaged source")
        for name, value in sources.items():
            if Reproduction.digest(bundle / prefix / name) != value:
                raise ValueError(f"Packaged source checksum changed: {name}")
        ids = [row.specimen_id for row in record.inputs]
        if len(ids) != config.expected_pair_count or ids != sorted(config.specimen_ids):
            raise ValueError("Run cohort differs from packaged profile")
        provenance = {
            row["specimen_id"]: row
            for row in json.loads((bundle / "input-provenance.json").read_text())[
                "specimens"
            ]
        }
        for row in record.inputs:
            for kind in ("mesh", "texture"):
                expected = provenance[row.specimen_id][f"{kind}_sha256"]
                if (
                    getattr(row, f"{kind}_sha256") != expected
                    or Reproduction.digest(getattr(row, kind)) != expected
                ):
                    raise ValueError(
                        f"Input differs from bundled cohort: {row.specimen_id} {kind}"
                    )
        return profiles[0]

    @staticmethod
    def validate(bundle: Path, run: Path) -> dict:
        config = StageData.load_options(run / "config.json")
        summary = StageData.read_completion(run / "completed.json")
        record = StageData.read_run(run / "run.json")
        if record.status != "completed" or summary.config_sha256 != Reproduction.digest(
            run / "config.json"
        ):
            raise ValueError("Run lacks matching current completion/config evidence")
        if not StageData.matching_completion(
            run / "analysis/summary.json", run / "completed.json"
        ):
            raise ValueError("Completion marker and analysis summary differ")
        required_analysis = {
            "coordinates.csv",
            "neighbors.csv",
            "pca_transform.npz",
            "population.png",
            "population.pdf",
        }
        if (
            summary.analysis_sha256 is None
            or set(summary.analysis_sha256) != required_analysis
        ):
            raise ValueError("Run lacks the complete analysis checksum inventory")
        for name, value in summary.analysis_sha256.items():
            if Reproduction.digest(run / "analysis" / name) != value:
                raise ValueError(f"Analysis checksum changed: {name}")
        profile = Checks.identity(bundle, config, record)
        ids = [row.specimen_id for row in record.inputs]
        metadata = StageData.read_embedding(run / "embedding_metadata.json")
        for name in ("crop_embeddings", "specimen_embeddings"):
            if Reproduction.digest(run / f"{name}.npz") != getattr(
                metadata, f"{name}_sha256"
            ):
                raise ValueError(f"Embedding checksum changed: {name}")
        with np.load(run / "crop_embeddings.npz", allow_pickle=False) as data:
            crops = data["vectors"].astype(np.float64)
            crop_ids = data["specimen_ids"].tolist()
        with np.load(run / "specimen_embeddings.npz", allow_pickle=False) as data:
            vectors = data["vectors"].astype(np.float64)
            vector_ids = data["specimen_ids"].tolist()
        if vector_ids != ids or crop_ids != list(
            np.repeat(ids, config.patches_per_specimen)
        ):
            raise ValueError("Embedding order differs from cohort")
        if (
            crops.shape != (len(ids) * config.patches_per_specimen, 1280)
            or vectors.shape != (len(ids), 1280)
            or not np.isfinite(crops).all()
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Invalid embedding dimensions or nonfinite values")
        means = crops.reshape(len(ids), config.patches_per_specimen, 1280).mean(axis=1)
        mean_norms = np.linalg.norm(means, axis=1)
        if not np.isfinite(means).all() or np.any(mean_norms <= 1e-12):
            raise ValueError("Degenerate raw mean vectors")
        means /= mean_norms[:, None]
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6, rtol=0):
            raise ValueError("Specimen vectors are not unit normalized")
        aggregation_error = float(np.max(np.abs(means - vectors)))
        if aggregation_error > 2e-6:
            raise ValueError("Raw mean then single normalization contract failed")
        coordinates = StageData.read_coordinates(run / "analysis/coordinates.csv")
        if [r.specimen_id for r in coordinates] != ids:
            raise ValueError("Coordinate cohort mismatch")
        saved_pca = np.array([[r.PC1, r.PC2] for r in coordinates])
        saved_umap = np.array([[r.UMAP1, r.UMAP2] for r in coordinates])
        if not np.isfinite(saved_pca).all() or not np.isfinite(saved_umap).all():
            raise ValueError("Nonfinite population coordinates")
        if len(ids) < 4 or summary.UMAP_method != "UMAP cosine":
            raise ValueError("Packaged smoke/full runs require a genuine UMAP fit")
        with np.load(
            run / "analysis/pca_transform.npz", allow_pickle=False
        ) as transform:
            calculated = (
                (vectors - transform["scaler_mean"]) / transform["scaler_scale"]
                - transform["mean"]
            ) @ transform["components"].T
        if not np.isfinite(calculated).all():
            raise ValueError("Nonfinite saved PCA transform")
        pca_error = float(np.max(np.abs(calculated - saved_pca)))
        if pca_error > 0.001:
            raise ValueError("Saved PCA transform does not reproduce coordinates")
        unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        similarity = unit @ unit.T
        np.fill_diagonal(similarity, -np.inf)
        expected = [
            (name, rank + 1, ids[j])
            for i, name in enumerate(ids)
            for rank, j in enumerate(
                np.argsort(-similarity[i], kind="stable")[
                    : min(config.nearest_neighbors, len(ids) - 1)
                ]
            )
        ]
        actual = [
            (row.specimen_id, row.rank, row.neighbor_id)
            for row in StageData.read_neighbors(run / "analysis/neighbors.csv")
        ]
        if expected != actual:
            raise ValueError("Independent cosine ranking differs")
        manifest = StageData.read_patches(run / "patch_manifest.json")
        if (
            Reproduction.digest(run / "patch_manifest.json")
            != metadata.patch_manifest_sha256
        ):
            raise ValueError("Patch manifest checksum changed")
        maximum_iou = 0.0
        for name in ids:
            rows = [row for row in manifest if row.specimen_id == name]
            if len(rows) != config.patches_per_specimen:
                raise ValueError("Patch count mismatch")
            with Image.open(rows[0].source_render) as image:
                pixels = np.asarray(image.convert("RGBA"))
            distinct = set()
            for row in rows:
                left, top, size = row.left, row.top, row.size
                if not (
                    pixels[top : top + size, left : left + size, 3]
                    > config.patch_alpha_tolerance
                ).all():
                    raise ValueError("Patch includes background pixels")
                if Reproduction.digest(Path(row.path)) != row.sha256:
                    raise ValueError("Saved crop checksum changed")
                with Image.open(row.path) as image:
                    crop = np.asarray(image)
                if not np.array_equal(
                    crop, pixels[top : top + size, left : left + size, :3]
                ):
                    raise ValueError("Crop pixels differ from original render")
                distinct.add(hashlib.sha256(crop.tobytes()).hexdigest())
            if len(distinct) != config.patches_per_specimen:
                raise ValueError("Duplicate pixel crops")
            for first, second in itertools.combinations(rows, 2):
                size = first.size
                overlap = max(0, size - abs(first.left - second.left)) * max(
                    0, size - abs(first.top - second.top)
                )
                maximum_iou = max(maximum_iou, overlap / (2 * size * size - overlap))
        if maximum_iou > config.max_overlap_iou:
            raise ValueError("Crop overlap exceeds configured limit")
        historical = (
            bundle / "reference/historical-2026-10-06-32/specimen_embeddings.npz"
        )
        with np.load(historical, allow_pickle=False) as data:
            ref_ids = data["specimen_ids"].tolist()
            reference = data["vectors"][[ref_ids.index(name) for name in ids]].astype(
                np.float64
            )
        delta = float(np.max(np.abs(vectors - reference)))
        cosine = np.sum(
            unit * reference / np.linalg.norm(reference, axis=1)[:, None], axis=1
        )
        report = {
            "run": str(run),
            "specimens": len(ids),
            "crops": len(crops),
            "current_run_internal_checks_pass": True,
            "packaged_profile": profile,
            "packaged_protocol_source_and_inputs_match": True,
            "source_manifest_sha256": Reproduction.digest(bundle / "manifest.json"),
            "config_sha256": summary.config_sha256,
            "aggregation_max_error": aggregation_error,
            "pca_transform_max_error": pca_error,
            "cosine_neighbor_rows_verified": len(actual),
            "maximum_crop_iou": maximum_iou,
            "historical_comparison": {
                "reference": "October 6 MPS method reconstruction, not original author arrays",
                "max_absolute_embedding_difference": delta,
                "within_1e_minus_5": delta <= 1e-5,
                "minimum_same_specimen_cosine": float(np.min(cosine)),
                "interpretation": "Device/render backend differences are measured separately from internal correctness; smoke subset PCA/UMAP are not full-cohort coordinates",
            },
        }
        Reproduction.write(run / "validation.json", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("environment", "preflight", "validate"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    if args.command == "environment":
        result = Checks.environment(args.bundle)
    elif args.command == "preflight":
        result = Checks.preflight(args.bundle, args.device)
    else:
        result = Checks.validate(args.bundle, args.run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
