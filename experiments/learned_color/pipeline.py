"""Deterministic rendering, frozen inference, and descriptive population fits.

This reconstructs Eric Regina's documented v2.2 protocol. Its source is not the
missing upstream pipeline, and a new population run is not evidence of equality
to the original 32-mussel experiment.
"""

import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel
from umap import UMAP

from experiments.learned_color.models import Frame, Specimen
from experiments.learned_color.reference import patch_sampling as S
from experiments.learned_color.reference import render_clam as R
from experiments.learned_color.serde import Config

VERSION = "documented-ijepa-reconstruction-1"
matplotlib.use("Agg")


class Appearance:
    """Static operations; every stage consumes an explicit recorded cohort."""

    @staticmethod
    def digest(path: Path) -> str:
        """Stream SHA256 for meshes, textures, model configuration and outputs."""
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    @staticmethod
    def write(path: Path, value: object) -> None:
        """Atomically write strict JSON so interrupted stages cannot look complete."""
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    @staticmethod
    def pairs(config: Config) -> tuple[Specimen, ...]:
        """Pair exact stems and enforce the selected cohort before any writes."""
        models = {p.stem: p for p in sorted(config.models_dir.glob("*.obj"))}
        textures = {}
        for path in sorted(config.textures_dir.iterdir()):
            if path.suffix.lower() in R.TEXTURE_SUFFIXES:
                if path.stem in textures:
                    raise ValueError(f"Ambiguous texture stem {path.stem}")
                textures[path.stem] = path
        identifiers = sorted(config.specimen_ids or (models.keys() & textures.keys()))
        if len(identifiers) != config.expected_pair_count:
            raise ValueError(
                f"Expected {config.expected_pair_count} paired specimens, found {len(identifiers)}."
            )
        if config.orientation_reference_id not in identifiers:
            raise ValueError(
                "orientation_reference_id must belong to the selected cohort"
            )
        if any(i not in models or i not in textures for i in identifiers):
            raise ValueError("Every selected specimen needs both a mesh and a texture")
        return tuple(Specimen(i, models[i], textures[i]) for i in identifiers)

    @staticmethod
    def prepare(config: Config) -> None:
        """Freeze configuration and input hashes without copying large assets."""
        specimens = Appearance.pairs(config)
        output = config.output_dir
        for source in (config.models_dir, config.textures_dir):
            if (
                output == source
                or output.is_relative_to(source)
                or source.is_relative_to(output)
            ):
                raise ValueError("Input and output directories must be disjoint")
        if output.exists():
            raise FileExistsError(output)
        inputs = [
            {
                "specimen_id": s.identifier,
                "mesh": str(s.mesh),
                "texture": str(s.texture),
                "mesh_sha256": Appearance.digest(s.mesh),
                "texture_sha256": Appearance.digest(s.texture),
            }
            for s in specimens
        ]
        code = Path(__file__).parent
        sources = {
            str(p.relative_to(code)): Appearance.digest(p)
            for p in sorted(code.rglob("*.py"))
        }
        output.mkdir(parents=True)
        Appearance.write(output / "config.json", config.model_dump(mode="json"))
        Appearance.write(
            output / "run.json",
            {
                "pipeline_version": VERSION,
                "status": "prepared",
                "inputs": inputs,
                "source_sha256": sources,
                "platform": platform.platform(),
                "packages": {
                    p.metadata["Name"]: p.version
                    for p in importlib.metadata.distributions()
                },
                "scope": "Documented method reconstruction; original 32-specimen run not certified.",
            },
        )

    @staticmethod
    def verify(config: Config) -> dict:
        """Reject edited inputs, configuration or code before reusing stage output."""
        output = config.output_dir
        if json.loads((output / "config.json").read_text()) != config.model_dump(
            mode="json"
        ):
            raise ValueError("Configuration differs from the prepared run")
        record = json.loads((output / "run.json").read_text())
        if record["pipeline_version"] != VERSION:
            raise ValueError("Pipeline version mismatch")
        identifiers = [s.identifier for s in Appearance.pairs(config)]
        if identifiers != [s["specimen_id"] for s in record["inputs"]]:
            raise ValueError("Prepared cohort changed")
        for item in record["inputs"]:
            for kind in ("mesh", "texture"):
                if Appearance.digest(Path(item[kind])) != item[f"{kind}_sha256"]:
                    raise ValueError(f"Changed {kind}: {item['specimen_id']}")
        for path, digest in record["source_sha256"].items():
            if Appearance.digest(Path(__file__).parent / path) != digest:
                raise ValueError(f"Changed implementation: {path}")
        return record

    @staticmethod
    def frame(direction: np.ndarray, up: np.ndarray, points: np.ndarray) -> Frame:
        """Build a right-handed camera frame and standardized skewness signature."""
        direction = R.normalized(direction)
        up = R.normalized(up - direction * np.dot(up, direction))
        axes = np.column_stack((R.normalized(np.cross(direction, up)), up, direction))
        projected = (points - points.mean(axis=0)) @ axes
        rms = np.sqrt(np.mean(projected**2, axis=0))
        if np.any(rms <= 1e-12):
            raise ValueError("Degenerate mesh orientation")
        return Frame(direction, up, np.mean((projected / rms) ** 3, axis=0), 0.0)

    @staticmethod
    def group_frame(mesh: pv.PolyData, reference: Frame) -> Frame:
        """Resolve target PCA signs using the reference's third-moment signature.

        This resolves sign ambiguity, not anatomical axis permutations. The
        trace term is the documented weak preference for world-frame agreement.
        """
        points = R.mesh_points(mesh)
        centered = points - points.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(points))
        axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        ref_axes = np.column_stack(
            (
                np.cross(reference.direction, reference.up),
                reference.up,
                reference.direction,
            )
        )
        candidates = []
        for up_sign in (-1, 1):
            for direction_sign in (-1, 1):
                candidate = Appearance.frame(
                    direction_sign * axes[:, 2], up_sign * axes[:, 1], points
                )
                target_axes = np.column_stack(
                    (
                        np.cross(candidate.direction, candidate.up),
                        candidate.up,
                        candidate.direction,
                    )
                )
                score = -np.sum(
                    (candidate.skewness - reference.skewness) ** 2
                ) + 0.01 * np.trace(target_axes.T @ ref_axes)
                candidates.append(candidate._replace(score=float(score)))
        return max(candidates, key=lambda c: c.score)

    @staticmethod
    def normalize_canvas(image: Image.Image, config: Config) -> Image.Image:
        """Crop alpha content, pad, and apply the explicit square-normalization rule."""
        mask = np.asarray(image.convert("RGBA"))[:, :, 3] > config.crop_alpha_tolerance
        yy, xx = np.nonzero(mask)
        if not len(xx):
            raise ValueError("Render contains no surface pixels")
        padding = config.crop_padding_pixels
        crop = image.crop(
            (
                max(0, int(xx.min()) - padding),
                max(0, int(yy.min()) - padding),
                min(image.width, int(xx.max()) + 1 + padding),
                min(image.height, int(yy.max()) + 1 + padding),
            )
        )
        if not config.force_square_normalization:
            edge = max(crop.size)
            square = Image.new("RGBA", (edge, edge), (0, 0, 0, 0))
            square.paste(crop, ((edge - crop.width) // 2, (edge - crop.height) // 2))
            crop = square
        return crop.resize(
            (config.render_size, config.render_size), Image.Resampling.LANCZOS
        )

    @staticmethod
    def render(config: Config) -> None:
        """Render native textured surfaces in a shared group-PCA camera orientation."""
        Appearance.verify(config)
        specimens = Appearance.pairs(config)
        ref_specimen = next(
            s for s in specimens if s.identifier == config.orientation_reference_id
        )
        ref_mesh = pv.read(ref_specimen.mesh)
        direction, up = R.shell_orientation(ref_mesh)
        reference = Appearance.frame(
            np.asarray(direction), np.asarray(up), R.mesh_points(ref_mesh)
        )
        output = config.output_dir / "renders"
        output.mkdir()
        render_config = R.RenderConfig(
            transparent=True,
            width=config.render_size,
            height=config.render_size,
            margin=config.render_margin,
            texture_max_size=config.texture_max_size,
        )
        manifest = []
        for index, specimen in enumerate(specimens):
            mesh = pv.read(specimen.mesh)
            frame = (
                reference
                if specimen.identifier == ref_specimen.identifier
                else Appearance.group_frame(mesh, reference)
            )
            texture = R.read_texture(specimen.texture, config.texture_max_size)
            # Use the upstream scene and material coefficients, then apply the
            # v2.2 alpha-content canvas normalization before surface sampling.
            plotter = pv.Plotter(
                off_screen=True, window_size=(config.render_size, config.render_size)
            )
            try:
                R.enable_antialiasing(plotter)
                R.add_clam_scene(
                    plotter,
                    mesh,
                    texture,
                    tuple(frame.direction),
                    tuple(frame.up),
                    config.render_size,
                    config.render_size,
                    render_config,
                )
                pixels = plotter.screenshot(
                    transparent_background=True, return_img=True
                )
            finally:
                plotter.close()
            image = Appearance.normalize_canvas(
                Image.fromarray(pixels).convert("RGBA"), config
            )
            path = output / f"{specimen.identifier}_exterior.png"
            image.save(path)
            item = {
                "specimen_id": specimen.identifier,
                "path": str(path),
                "sha256": Appearance.digest(path),
                "direction": frame.direction.tolist(),
                "up": frame.up.tolist(),
                "skewness": frame.skewness.tolist(),
                "orientation_score": frame.score,
                "reference_id": config.orientation_reference_id,
            }
            Appearance.write(path.with_suffix(".json"), item)
            manifest.append(item)
            print(
                f"Rendered {index + 1}/{len(specimens)}: {specimen.identifier}",
                flush=True,
            )
        Appearance.write(config.output_dir / "renders.json", manifest)

    @staticmethod
    def sample(config: Config) -> None:
        """Save exactly the required unaugmented, distinct, all-surface crops."""
        Appearance.verify(config)
        renders = json.loads((config.output_dir / "renders.json").read_text())
        output = config.output_dir / "patches"
        output.mkdir()
        manifest = []
        for item in renders:
            path = Path(item["path"])
            if Appearance.digest(path) != item["sha256"]:
                raise ValueError(f"Render checksum changed: {path}")
            with Image.open(path) as image:
                mask = (
                    np.asarray(image.convert("RGBA"))[:, :, 3]
                    > config.patch_alpha_tolerance
                )
                spec = S.PatchSpec(
                    config.patch_size_px,
                    config.patches_per_specimen,
                    0.0,
                    config.max_overlap_iou,
                    True,
                    0.0,
                    config.patch_alpha_tolerance,
                )
                positions, fractions = S.candidate_positions(mask, spec)
                selected = S.select_positions(
                    positions,
                    fractions,
                    spec,
                    np.random.default_rng(
                        S.specimen_seed(config.patch_sample_seed, item["specimen_id"])
                    ),
                )
                seen = set()
                rows = []
                directory = output / item["specimen_id"]
                directory.mkdir()
                for left, top, fraction in selected:
                    crop = image.crop(
                        (left, top, left + spec.patch_size, top + spec.patch_size)
                    ).convert("RGB")
                    pixel_hash = hashlib.blake2b(
                        crop.tobytes(), digest_size=16
                    ).hexdigest()
                    if pixel_hash in seen:
                        continue
                    seen.add(pixel_hash)
                    destination = (
                        directory / f"{item['specimen_id']}_p{len(rows):04d}.png"
                    )
                    crop.save(destination)
                    rows.append(
                        {
                            "specimen_id": item["specimen_id"],
                            "patch_index": len(rows),
                            "path": str(destination),
                            "sha256": Appearance.digest(destination),
                            "source_render": str(path),
                            "source_render_sha256": item["sha256"],
                            "left": left,
                            "top": top,
                            "size": spec.patch_size,
                            "background_fraction": fraction,
                        }
                    )
            if len(rows) != config.patches_per_specimen:
                raise ValueError(
                    f"{item['specimen_id']}: only {len(rows)} distinct surface patches"
                )
            manifest.extend(rows)
            print(
                f"Sampled {item['specimen_id']}: {len(rows)} full-surface patches",
                flush=True,
            )
        Appearance.write(config.output_dir / "patch_manifest.json", manifest)
        Appearance.table(config.output_dir / "patch_manifest.csv", manifest)

    @staticmethod
    def table(path: Path, rows: list[dict]) -> None:
        """Write an ordered, named-column CSV without implicit row indices."""
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def aggregate(crops: np.ndarray, count: int) -> np.ndarray:
        """Average raw crop vectors, then normalize each specimen exactly once."""
        if (
            crops.ndim != 2
            or not len(crops)
            or len(crops) % count
            or not np.isfinite(crops).all()
        ):
            raise ValueError("Invalid crop embedding matrix")
        means = crops.reshape(-1, count, crops.shape[1]).mean(axis=1)
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("Zero specimen embedding")
        result = means / norms
        if len(result) > 1 and np.max(np.std(result, axis=0)) <= 1e-8:
            raise ValueError("Effectively constant specimen embeddings")
        return result.astype(np.float32)

    @staticmethod
    def embed(config: Config) -> None:
        """Load immutable pretrained weights and mean-pool all spatial tokens."""
        Appearance.verify(config)
        output = config.output_dir
        if (output / "specimen_embeddings.npz").exists():
            raise FileExistsError(output / "specimen_embeddings.npz")
        rows = json.loads((output / "patch_manifest.json").read_text())
        specimens = Appearance.pairs(config)
        expected = [
            (s.identifier, j)
            for s in specimens
            for j in range(config.patches_per_specimen)
        ]
        if [(r["specimen_id"], r["patch_index"]) for r in rows] != expected:
            raise ValueError("Patch manifest has an incomplete or reordered cohort")
        for row in rows:
            if Appearance.digest(Path(row["path"])) != row["sha256"]:
                raise ValueError(f"Patch checksum changed: {row['path']}")
        available = {
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
            "cpu": config.allow_cpu,
        }
        if not available[config.device]:
            raise ValueError(
                f"Requested {config.device} is unavailable; no implicit CPU fallback"
            )
        processor = AutoImageProcessor.from_pretrained(
            config.checkpoint, revision=config.model_revision, use_fast=False
        )
        model = AutoModel.from_pretrained(
            config.checkpoint,
            revision=config.model_revision,
            use_safetensors=True,
            attn_implementation="eager",
        )
        model = model.to(config.device).eval()
        model.requires_grad_(False)
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(rows), config.batch_size):
                images = []
                for row in rows[start : start + config.batch_size]:
                    with Image.open(row["path"]) as image:
                        images.append(image.convert("RGB"))
                inputs = processor(images=images, return_tensors="pt").to(config.device)
                hidden = model(**inputs).last_hidden_state
                if hidden.ndim != 3 or hidden.shape[1:] != (256, 1280):
                    raise ValueError(
                        f"Unexpected I-JEPA token shape: {tuple(hidden.shape)}"
                    )
                vectors.append(hidden.mean(dim=1).cpu().numpy())
                print(
                    f"Encoded {min(start + config.batch_size, len(rows))}/{len(rows)}",
                    flush=True,
                )
        crops = np.concatenate(vectors).astype(np.float32)
        embeddings = Appearance.aggregate(crops, config.patches_per_specimen)
        metadata = {
            "checkpoint": config.checkpoint,
            "revision": config.model_revision,
            "device": config.device,
            "dtype": "float32",
            "pooling": "all-spatial-token-mean",
            "aggregation": "raw-crop-mean-then-single-L2",
            "processor": processor.to_dict(),
            "model": model.config.to_dict(),
            "patch_manifest_sha256": Appearance.digest(output / "patch_manifest.json"),
        }
        np.savez_compressed(
            output / "crop_embeddings.npz",
            vectors=crops,
            specimen_ids=np.array([r["specimen_id"] for r in rows]),
        )
        np.savez_compressed(
            output / "specimen_embeddings.npz",
            vectors=embeddings,
            specimen_ids=np.array([s.identifier for s in specimens]),
        )
        metadata["specimen_embeddings_sha256"] = Appearance.digest(
            output / "specimen_embeddings.npz"
        )
        metadata["crop_embeddings_sha256"] = Appearance.digest(
            output / "crop_embeddings.npz"
        )
        Appearance.write(output / "embedding_metadata.json", metadata)

    @staticmethod
    def analyze(config: Config) -> None:
        """Fit descriptive PCA/UMAP and cosine neighbors to the recorded population."""
        record = Appearance.verify(config)
        output = config.output_dir
        metadata = json.loads((output / "embedding_metadata.json").read_text())
        for name in ("specimen_embeddings", "crop_embeddings"):
            if Appearance.digest(output / f"{name}.npz") != metadata[f"{name}_sha256"]:
                raise ValueError(f"Embedding checksum changed: {name}")
        if (
            Appearance.digest(output / "patch_manifest.json")
            != metadata["patch_manifest_sha256"]
        ):
            raise ValueError("Patch manifest checksum changed")
        with np.load(output / "specimen_embeddings.npz", allow_pickle=False) as data:
            vectors, identifiers = data["vectors"], data["specimen_ids"]
        if list(identifiers) != [s.identifier for s in Appearance.pairs(config)]:
            raise ValueError("Embedding specimen order differs from cohort")
        if not np.isfinite(vectors).all() or not np.allclose(
            np.linalg.norm(vectors, axis=1), 1, atol=1e-6
        ):
            raise ValueError("Specimen vectors must be finite unit vectors")
        directory = output / "analysis"
        directory.mkdir()
        # Standardization affects PCA only. UMAP and neighbor ranking use the
        # unit-normalized original specimen vectors with cosine geometry.
        scaler = (
            StandardScaler().fit(vectors) if config.standardize_pca_features else None
        )
        features = scaler.transform(vectors) if scaler else vectors
        pca = PCA(n_components=2, svd_solver="full").fit(features)
        coordinates = pca.transform(features)
        if len(vectors) <= 3:
            umap_coordinates = coordinates.copy()
            umap_method = "PCA fallback for n<=3; not a UMAP fit"
        else:
            umap_coordinates = UMAP(
                n_neighbors=min(config.umap_neighbors, len(vectors) - 1),
                min_dist=config.umap_min_dist,
                metric="cosine",
                n_jobs=1,
                random_state=config.random_seed,
            ).fit_transform(vectors)
            umap_method = "UMAP cosine"
        if (
            not np.isfinite(coordinates).all()
            or not np.isfinite(umap_coordinates).all()
        ):
            raise ValueError("Nonfinite population coordinates")
        rows = [
            {
                "specimen_id": str(i),
                "PC1": float(p[0]),
                "PC2": float(p[1]),
                "UMAP1": float(u[0]),
                "UMAP2": float(u[1]),
            }
            for i, p, u in zip(identifiers, coordinates, umap_coordinates, strict=True)
        ]
        Appearance.table(directory / "coordinates.csv", rows)
        similarity = vectors @ vectors.T
        np.fill_diagonal(similarity, -np.inf)
        neighbors = []
        for index, identifier in enumerate(identifiers):
            selected = np.argsort(-similarity[index], kind="stable")[
                : min(config.nearest_neighbors, len(vectors) - 1)
            ]
            neighbors.extend(
                {
                    "specimen_id": str(identifier),
                    "rank": rank + 1,
                    "neighbor_id": str(identifiers[j]),
                    "cosine_similarity": float(similarity[index, j]),
                }
                for rank, j in enumerate(selected)
            )
        Appearance.table(directory / "neighbors.csv", neighbors)
        np.savez_compressed(
            directory / "pca_transform.npz",
            components=pca.components_,
            mean=pca.mean_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            scaler_mean=scaler.mean_ if scaler else np.zeros(vectors.shape[1]),
            scaler_scale=scaler.scale_ if scaler else np.ones(vectors.shape[1]),
        )
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for axis, points, title in zip(
            axes,
            (coordinates, umap_coordinates),
            ("PCA of frozen I-JEPA features", umap_method),
            strict=True,
        ):
            axis.scatter(points[:, 0], points[:, 1], s=35, color="#087f8c")
            for identifier, point in zip(identifiers, points, strict=True):
                axis.annotate(
                    str(identifier).replace("UF_IZ_", ""),
                    point,
                    fontsize=5,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
            axis.set_title(title)
            axis.set_xlabel("PC1" if axis is axes[0] else "Coordinate 1")
            axis.set_ylabel("PC2" if axis is axes[0] else "Coordinate 2")
        figure.suptitle(
            f"{config.profile}: {len(vectors)} specimens · descriptive full-cohort fits"
        )
        figure.savefig(directory / "population.png", dpi=180)
        figure.savefig(directory / "population.pdf")
        plt.close(figure)
        summary = {
            "profile": config.profile,
            "specimens": len(vectors),
            "crops_per_specimen": config.patches_per_specimen,
            "embedding_features": vectors.shape[1],
            "PCA_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "UMAP_method": umap_method,
            "scope": record["scope"],
            "supervised_performance_claim": False,
            "checkpoint_revision": config.model_revision,
            # The configuration controls figure labels and the camera reference,
            # so bind its exact saved bytes to the completed numerical results.
            "config_sha256": Appearance.digest(output / "config.json"),
            "population_plot_sha256": Appearance.digest(directory / "population.png"),
            # A later export must verify the saved numerical outputs, not merely
            # trust that another coordinate table has the same number of rows.
            "analysis_sha256": {
                name: Appearance.digest(directory / name)
                for name in (
                    "coordinates.csv",
                    "neighbors.csv",
                    "pca_transform.npz",
                    "population.png",
                    "population.pdf",
                )
            },
        }
        Appearance.write(directory / "summary.json", summary)
        record["status"] = "completed"
        Appearance.write(output / "run.json", record)
        Appearance.write(output / "completed.json", summary)
