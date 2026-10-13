"""Matched native/atlas composition recovery with a shared native palette."""

import csv
from pathlib import Path

import numpy as np

from experiments.fishy.analysis import Analysis
from experiments.fishy.artifacts import Artifacts
from experiments.fishy.serde import CompositionIdentity, Documents, Evaluation


class Comparison:
    """Compare completed stages without changing labels, palette or folds."""

    @staticmethod
    def run(native: Path, atlas: Path, labels: Path, output: Path) -> None:
        """Require matching identities, configurations and exact palette centers.

        Native vectors use native face areas; atlas vectors use shared atlas
        areas. Differences combine transfer, sampling and geometric weighting,
        so they are not a pure pixel-level transfer-error measurement.
        """
        metadata_paths = [native / "samples.json", atlas / "recovery.json"]
        identities = [
            CompositionIdentity.model_validate_json(p.read_text())
            for p in metadata_paths
        ]
        if identities[0] != identities[1]:
            raise ValueError("Native and atlas sample/donor order must match.")
        boundary_options = [
            Evaluation.model_validate_json((p / "evaluation-config.json").read_text())
            for p in (native, atlas)
        ]
        options = [item.to_internal() for item in boundary_options]
        if options[0] != options[1]:
            raise ValueError("Native and atlas evaluation configurations differ.")
        arrays, palettes = [], []
        inputs = [labels, *metadata_paths]
        for stage in (native, atlas):
            if not (stage / "complete.json").is_file():
                raise ValueError(f"Stage is incomplete: {stage}")
            vector_path, palette_path = (
                stage / "composition.npy",
                stage / "palette_lab.npy",
            )
            vectors = np.load(vector_path, allow_pickle=False)
            palette = np.load(palette_path, allow_pickle=False)
            if vectors.shape != (
                len(identities[0].sample_ids),
                options[0].palette_bins,
            ):
                raise ValueError("Composition shape disagrees with metadata.")
            if not np.isfinite(vectors).all() or not np.allclose(
                np.linalg.norm(vectors, axis=1), 1
            ):
                raise ValueError("Expected finite L2-normalized composition vectors.")
            if (
                palette.shape != (options[0].palette_bins, 3)
                or not np.isfinite(palette).all()
            ):
                raise ValueError("Invalid Lab palette.")
            arrays.append(vectors)
            palettes.append(palette)
            inputs.extend([vector_path, palette_path, stage / "evaluation-config.json"])
        if not np.array_equal(palettes[0], palettes[1]):
            raise ValueError("Matched comparison requires exactly the same palette.")
        with labels.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_id = {r["sample_id"]: r for r in rows}
        identity = identities[0]
        if len(by_id) != len(rows) or set(by_id) != set(identity.sample_ids):
            raise ValueError("Labels must match sample IDs one-to-one.")
        if any(
            by_id[s]["donor_id"] != d
            for s, d in zip(identity.sample_ids, identity.donor_ids, strict=True)
        ):
            raise ValueError("Label donor IDs disagree with stage metadata.")
        responses = {}
        for factor in (
            "belly_group",
            "tail_group",
            "stripe_count",
            "rosy_cheeks_present",
        ):
            values = np.array([int(by_id[s][factor]) for s in identity.sample_ids])
            allowed = {4, 5} if factor == "stripe_count" else {0, 1}
            if set(values) != allowed:
                raise ValueError(f"Invalid binary labels for {factor}.")
            responses[factor] = (values == max(allowed)).astype(int)
        Artifacts.start(output, "compare-native-atlas-composition", inputs)
        # Reuse the deterministic scoring protocol for both spaces. Saved fold
        # indices permit verifying that every comparison holds out the same fish.
        reports = {
            space: {
                factor: Analysis.score(
                    vectors, response, np.array(identity.donor_ids), options[0]
                )
                for factor, response in responses.items()
            }
            for space, vectors in zip(("native", "atlas"), arrays, strict=True)
        }
        summaries = {}
        for space, factors in reports.items():
            summaries[space] = {}
            for factor, schemes in factors.items():
                summaries[space][factor] = {}
                for scheme, details in schemes.items():
                    summaries[space][factor][scheme] = {
                        metric: float(np.mean(values)) if values else None
                        for metric in (
                            "balanced_accuracy",
                            "shuffled_accuracy",
                            "null_corrected",
                        )
                        for values in [
                            [
                                getattr(r, metric)
                                for r in details.folds
                                if getattr(r, metric) is not None
                            ]
                        ]
                    }
        Artifacts.write(
            output / "comparison.json",
            {
                **identity.model_dump(mode="json"),
                "configuration": boundary_options[0].model_dump(mode="json"),
                "summary": summaries,
                "folds": {
                    space: {
                        factor: Documents.scores(schemes)
                        for factor, schemes in factors.items()
                    }
                    for space, factors in reports.items()
                },
                "scope": "Shared native palette; transductive unsupervised fit. Native and atlas face areas differ. Not a transfer-fidelity or full paper reproduction claim.",
            },
        )
        Artifacts.complete(output)
