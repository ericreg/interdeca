"""Face-color extraction and the documented baseline factor-recovery protocol.

This module writes inspectable numerical outputs, not fabricated paper figures.
The seven-donor archive is available; exact figure scripts are still required to
certify the manuscript's numerical results (see docs/fishy/provenance.md).
"""

import csv
from pathlib import Path

import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, pairwise_distances_argmin
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.fishy.archived_uv import ArchivedUV
from experiments.fishy.artifacts import Artifacts
from experiments.fishy.serde import Evaluation, Labels
from interdeca import MeshIO, TextureIO


class Analysis:
    """Label-free extraction and label-dependent evaluation are separate stages."""

    @staticmethod
    def extract(
        manifest_path: Path, output: Path, *, archived_uv: bool = False
    ) -> None:
        """Sample triangle-corner colors with the legacy fishpipe pixel convention.

        UV coordinates are clamped, v is flipped, pixel coordinates are truncated,
        and the three RGB corners are averaged then truncated to uint8. Native
        black colors remain valid. Alpha coverage is checked separately.
        """
        manifest = Artifacts.manifest(manifest_path)
        if manifest.kind != "atlas":
            raise ValueError(
                "Extraction needs an atlas manifest with shared topology and UVs."
            )
        if archived_uv and manifest.backend != "legacy-atlas":
            raise ValueError(
                "Archived UV mode is restricted to legacy-atlas manifests."
            )
        mesh = MeshIO.read(manifest.atlas_mesh.path)
        if mesh.uv is None or np.any((mesh.uv < 0) | (mesh.uv > 1)):
            raise ValueError("Expected a shared atlas UV layout within [0,1].")
        Artifacts.start(output, "extract-corner-colors", [manifest_path])
        Artifacts.write(
            output / "input-manifest.json", manifest.model_dump(mode="json")
        )
        shape = (len(manifest.samples), len(mesh.faces), 3)
        colors = np.lib.format.open_memmap(
            output / "colors.npy", mode="w+", dtype=np.uint8, shape=shape
        )
        valid = np.lib.format.open_memmap(
            output / "valid.npy", mode="w+", dtype=bool, shape=shape[:2]
        )
        geometry_mask = None
        compatibility = []
        if archived_uv:
            geometry_mask = np.lib.format.open_memmap(
                output / "archived_degenerate_faces.npy",
                mode="w+",
                dtype=bool,
                shape=shape[:2],
            )
        for index, sample in enumerate(manifest.samples):
            if archived_uv:
                geometry_mask[index] = ArchivedUV.inspect(sample.mesh.path, mesh.uv)
            else:
                specimen_mesh = MeshIO.read(sample.mesh.path)
                if specimen_mesh.topology_key != mesh.topology_key:
                    raise ValueError(
                        f"{sample.sample_id}: topology/UV mismatch; cannot compare face indices."
                    )
            pixels = TextureIO.read(sample.texture.path)
            height, width = pixels.rgb.shape[:2]
            x = (mesh.uv[..., 0] * (width - 1)).astype(np.int64)
            y = ((1 - mesh.uv[..., 1]) * (height - 1)).astype(np.int64)
            if archived_uv:
                # Sampling remains anchored to the original shared atlas. Count
                # pixel changes caused by UV rounding instead of hiding them.
                rounded = mesh.uv.astype(np.float32).astype(np.float64)
                rounded_x = (rounded[..., 0] * (width - 1)).astype(np.int64)
                rounded_y = ((1 - rounded[..., 1]) * (height - 1)).astype(np.int64)
                compatibility.append(
                    {
                        "sample_id": sample.sample_id,
                        "degenerate_faces": int(geometry_mask[index].sum()),
                        "rounded_uv_different_pixel_corners": int(
                            ((x != rounded_x) | (y != rounded_y)).sum()
                        ),
                    }
                )
            # Round decoded sRGB to the original 8-bit bytes before the legacy
            # corner average. Background is identified only by explicit alpha.
            rgb = np.rint(pixels.rgb * 255).astype(np.uint8)
            colors[index] = rgb[y, x].astype(float).mean(axis=1).astype(np.uint8)
            valid[index] = (pixels.alpha[y, x] >= 0.99).all(axis=1)
            if (index + 1) % 25 == 0:
                print(
                    f"Extracted {index + 1}/{len(manifest.samples)} textures.",
                    flush=True,
                )
        if archived_uv:
            geometry_mask.flush()
            Artifacts.write(
                output / "archive-compatibility.json",
                {
                    "contract": "sequential expanded v/vt indices; exact ordered float32 atlas UVs",
                    "sampling_uv": "original shared atlas, not rounded specimen UVs",
                    "geometry_policy": "specimen geometry audited only; atlas supplies areas/centroids",
                    "limitation": "texture extraction does not validate historical transfer fidelity",
                    "samples": compatibility,
                },
            )
        colors.flush()
        valid.flush()
        triangles = mesh.vertices[mesh.faces]
        areas = (
            np.linalg.norm(
                np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                ),
                axis=1,
            )
            / 2
        )
        np.save(output / "areas.npy", areas)
        np.save(output / "centroids.npy", triangles.mean(axis=1))
        Artifacts.write(
            output / "samples.json",
            {
                "sample_ids": [s.sample_id for s in manifest.samples],
                "donor_ids": [s.donor_id for s in manifest.samples],
                "topology_key": mesh.topology_key,
                "backend": manifest.backend,
                "sampling": "legacy nearest-pixel triangle corners averaged to uint8",
                "coverage": "alpha only; opaque black is not classified as bake failure",
            },
        )
        # Copy the atlas through the core writer so extraction can be moved and
        # analyzed without the source checkout or experiment's original paths.
        MeshIO.write_obj(mesh, output / "atlas.obj")
        Artifacts.complete(output)

    @staticmethod
    def composition(
        colors: np.ndarray,
        areas: np.ndarray,
        bins: int,
        seed: int,
        palette: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Port fishpipe.features.area_hist: pooled Lab KMeans then L2 area vectors.

        The palette is fitted on the whole dataset without labels. This is a
        transductive descriptor and must not be described as a held-out palette.
        """
        lab = rgb2lab(colors.astype(float) / 255)
        if palette is None:
            pooled = lab.reshape(-1, 3)
            selected = np.random.default_rng(seed).choice(
                len(pooled), min(200_000, len(pooled)), replace=False
            )
            if len(np.unique(pooled[selected], axis=0)) < bins:
                raise ValueError("Fewer distinct colors than requested palette size.")
            quantizer = KMeans(n_clusters=bins, n_init=5, random_state=seed).fit(
                pooled[selected]
            )
            palette = quantizer.cluster_centers_
        elif palette.shape != (bins, 3) or not np.isfinite(palette).all():
            raise ValueError("Palette must contain bins finite Lab triples.")
        vectors = np.stack(
            [
                np.bincount(
                    pairwise_distances_argmin(row, palette),
                    weights=areas,
                    minlength=bins,
                )
                for row in lab
            ]
        )
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors, palette

    @staticmethod
    def score(
        features: np.ndarray,
        labels: np.ndarray,
        donors: np.ndarray,
        options: Evaluation,
    ) -> dict:
        """Balanced logistic recovery and shuffled-label null with saved fold IDs."""
        _, counts = np.unique(labels, return_counts=True)
        if len(counts) != 2 or counts.min() < options.folds:
            raise ValueError(
                "Each binary factor needs at least folds specimens in both classes."
            )
        random = RepeatedStratifiedKFold(
            n_splits=options.folds, n_repeats=options.repeats, random_state=options.seed
        )
        schemes = {"repeated_stratified": list(random.split(features, labels))}
        if len(np.unique(donors)) >= 2:
            schemes["leave_one_donor_out"] = list(
                LeaveOneGroupOut().split(features, labels, donors)
            )
        result = {}
        shuffled = np.random.default_rng(options.seed).permutation(labels)
        for name, folds in schemes.items():
            rows = []
            # Re-stratify shuffled labels for the random null; otherwise original
            # class stratification can inadvertently constrain the null labels.
            null_folds = (
                list(random.split(features, shuffled))
                if name == "repeated_stratified"
                else folds
            )
            for fold_index, ((train, test), (null_train, null_test)) in enumerate(
                zip(folds, null_folds, strict=True)
            ):
                scores = []
                for response, fitting, held in (
                    (labels, train, test),
                    (shuffled, null_train, null_test),
                ):
                    if (
                        len(np.unique(response[fitting])) != 2
                        or len(np.unique(response[held])) != 2
                    ):
                        scores.append(None)
                        continue
                    # All supervised scaling is fitted within the training fold.
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=options.logistic_c,
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=options.seed,
                        ),
                    )
                    model.fit(features[fitting], response[fitting])
                    scores.append(
                        float(
                            balanced_accuracy_score(
                                response[held], model.predict(features[held])
                            )
                        )
                    )
                rows.append(
                    {
                        "fold": fold_index,
                        "train_indices": train.tolist(),
                        "test_indices": test.tolist(),
                        "null_train_indices": null_train.tolist(),
                        "null_test_indices": null_test.tolist(),
                        "balanced_accuracy": scores[0],
                        "shuffled_accuracy": scores[1],
                        "null_corrected": None
                        if None in scores
                        else scores[0] - scores[1],
                    }
                )
            result[name] = {
                "folds": rows,
                "note": "Nulls and scores are omitted for single-class train/test folds.",
            }
        return result

    @staticmethod
    def evaluate(
        extraction: Path,
        labels_path: Path,
        config_path: Path,
        output: Path,
        palette_path: Path | None = None,
    ) -> None:
        """Compute the baseline using supplied labels and their declared protocol."""
        import_json = (extraction / "samples.json").read_text()
        samples = Labels.model_validate_json(import_json)
        options = Evaluation.model_validate_json(config_path.read_text())
        if not (extraction / "complete.json").exists():
            raise ValueError("Extraction has no completion marker.")
        colors = np.load(extraction / "colors.npy", mmap_mode="r", allow_pickle=False)
        valid = np.load(extraction / "valid.npy", allow_pickle=False)
        areas = np.load(extraction / "areas.npy", allow_pickle=False)
        if (
            colors.shape != (len(samples.sample_ids), len(areas), 3)
            or valid.shape != colors.shape[:2]
        ):
            raise ValueError("Extraction arrays and specimen metadata disagree.")
        if not valid.all():
            raise ValueError(
                "Missing coverage: resolve it explicitly; this reproduction never imputes black pixels."
            )
        with labels_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        by_id = {row["sample_id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != set(samples.sample_ids):
            raise ValueError(
                "Label CSV must have exactly one row per extracted sample ID."
            )
        responses = {}
        for factor in (
            "belly_group",
            "tail_group",
            "stripe_count",
            "rosy_cheeks_present",
        ):
            values = np.array(
                [int(by_id[sample][factor]) for sample in samples.sample_ids]
            )
            allowed = {4, 5} if factor == "stripe_count" else {0, 1}
            if set(values) != allowed:
                raise ValueError(
                    f"{factor} must contain both values {sorted(allowed)}."
                )
            responses[factor] = (values == max(allowed)).astype(int)
        for sample_id, donor_id in zip(
            samples.sample_ids, samples.donor_ids, strict=True
        ):
            if by_id[sample_id]["donor_id"] != donor_id:
                raise ValueError(f"Donor mismatch for {sample_id}")
        Artifacts.start(
            output,
            "evaluate-composition",
            [
                labels_path,
                config_path,
                extraction / "samples.json",
                extraction / "colors.npy",
                extraction / "valid.npy",
                extraction / "areas.npy",
            ]
            + ([] if palette_path is None else [palette_path]),
        )
        vectors, palette = Analysis.composition(
            colors,
            areas,
            options.palette_bins,
            options.seed,
            None if palette_path is None else np.load(palette_path, allow_pickle=False),
        )
        np.save(output / "composition.npy", vectors)
        np.save(output / "palette_lab.npy", palette)
        # Save a label-free exploratory view; the classifier uses the full vector.
        model = PCA(n_components=2, svd_solver="full").fit(vectors)
        np.save(output / "pca_scores.npy", model.transform(vectors))
        Artifacts.write(
            output / "evaluation-config.json", options.model_dump(mode="json")
        )
        report = {
            "backend": samples.backend,
            "descriptor": "pooled Lab palette, L2-normalized surface-area vector",
            "descriptor_fitting": "transductive; all specimens, no factor labels",
            "factor_label_source": "explicit input CSV; see label preparation provenance for inferred vs generating groups",
            "palette_source": "atlas pooled"
            if palette_path is None
            else str(palette_path.resolve()),
            "sample_ids": samples.sample_ids,
            "donor_ids": samples.donor_ids,
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "factors": {
                name: Analysis.score(vectors, y, np.array(samples.donor_ids), options)
                for name, y in responses.items()
            },
            "scope": "Baseline reproduction building block, not certification of Figures 7–13.",
        }
        Artifacts.write(output / "recovery.json", report)
        Artifacts.complete(output)
