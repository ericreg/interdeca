"""Fit the shared comparison palette to archived pre-transfer colors."""

from pathlib import Path

import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

from experiments.fishy.artifacts import Artifacts
from experiments.fishy.serde import Evaluation
from interdeca import MeshIO, TextureIO


class NativePalette:
    """Build a label-free native palette and composition vectors in manifest order."""

    @staticmethod
    def fit(manifest_path: Path, config_path: Path, output: Path) -> None:
        """Use the same pooled 200k-face palette convention as the atlas baseline.

        The paper specifies a palette fitted to native colors for its matched
        comparison. This implements that policy; exact figure settings are still
        required to claim equality to the manuscript's reported scores.
        """
        manifest = Artifacts.manifest(manifest_path)
        boundary_options = Evaluation.model_validate_json(config_path.read_text())
        options = boundary_options.to_internal()
        if manifest.kind != "generated":
            raise ValueError("Native palette requires a native-space input manifest.")
        if output.exists():
            raise FileExistsError(output)
        meshes, colors, areas = {}, [], []
        for index, sample in enumerate(manifest.samples):
            # Hash identity permits reusing repeated donor geometry while all
            # specimen textures retain their own place in the pooled population.
            key = sample.mesh.sha256
            if key not in meshes:
                meshes[key] = MeshIO.read(sample.mesh.path)
            mesh = meshes[key]
            if mesh.uv is None or np.any((mesh.uv < 0) | (mesh.uv > 1)):
                raise ValueError("Native palette requires one-tile UVs.")
            pixels = TextureIO.read(sample.texture.path)
            h, w = pixels.rgb.shape[:2]
            x = (mesh.uv[..., 0] * (w - 1)).astype(int)
            y = ((1 - mesh.uv[..., 1]) * (h - 1)).astype(int)
            if not (pixels.alpha[y, x] >= 0.99).all():
                raise ValueError(
                    "Missing native alpha coverage; no imputation allowed."
                )
            rgb = np.rint(pixels.rgb * 255).astype(np.uint8)
            colors.append(rgb[y, x].astype(float).mean(axis=1).astype(np.uint8))
            t = mesh.vertices[mesh.faces]
            areas.append(
                np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
                / 2
            )
            if (index + 1) % 25 == 0:
                print(
                    f"Sampled {index + 1}/{len(manifest.samples)} native textures.",
                    flush=True,
                )
        # Fit once without labels. The same centers can then encode native and
        # transferred textures; this is a transductive descriptor definition.
        pooled = np.concatenate(colors)
        selected = np.random.default_rng(options.seed).choice(
            len(pooled), min(200_000, len(pooled)), replace=False
        )
        training = rgb2lab(pooled[selected].astype(float) / 255)
        if len(np.unique(training, axis=0)) < options.palette_bins:
            raise ValueError("Insufficient distinct native palette colors.")
        model = KMeans(
            n_clusters=options.palette_bins, n_init=5, random_state=options.seed
        ).fit(training)
        vectors = np.stack(
            [
                np.bincount(
                    model.predict(rgb2lab(c.astype(float) / 255)),
                    weights=a,
                    minlength=options.palette_bins,
                )
                for c, a in zip(colors, areas, strict=True)
            ]
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        Artifacts.start(output, "fit-native-palette", [manifest_path, config_path])
        np.save(output / "palette_lab.npy", model.cluster_centers_)
        np.save(output / "composition.npy", vectors)
        Artifacts.write(
            output / "samples.json",
            {
                "sample_ids": [s.sample_id for s in manifest.samples],
                "donor_ids": [s.donor_id for s in manifest.samples],
                "palette_fitting": "native colors; transductive pooled face sample; no labels",
                "sampled_faces": len(selected),
                "scope": "paper-described comparison policy; exact figure implementation unverified",
            },
        )
        Artifacts.write(
            output / "input-manifest.json", manifest.model_dump(mode="json")
        )
        Artifacts.write(
            output / "evaluation-config.json", boundary_options.model_dump(mode="json")
        )
        Artifacts.complete(output)
