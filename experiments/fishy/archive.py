"""Register an existing native-space dataset without regenerating its evidence."""

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.fishy.artifacts import Artifacts
from experiments.fishy.generation import Generation
from experiments.fishy.serde import ArchiveRow, Asset, Manifest, ParameterData, Sample
from interdeca import Dataset


class Archive:
    """Import paired files while keeping scientific readiness separate from I/O."""

    @staticmethod
    def import_inputs(source: Path, output: Path, seed: int) -> None:
        """Hash all inputs and audit their contracts; never repair source evidence.

        A completed import means the inventory was saved, not that transfer or
        the paper results passed validation. The audit records those distinctions.
        """
        source = source.expanduser().resolve(strict=True)
        output = output.expanduser().resolve()
        if output.is_relative_to(source):
            raise ValueError("Import output must be outside the source dataset.")
        if not 0 <= seed <= 4294967295:
            raise ValueError("Comparison seed must be a uint32.")
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        csv_path = source / "samples.csv"
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(
                reader.fieldnames
            ):
                raise ValueError("Archive CSV needs unique column names.")
            rows = [ArchiveRow.model_validate(row) for row in reader]
        if not rows:
            raise ValueError("Archive CSV is empty.")
        rows.sort(key=lambda row: row.sample_index)
        tags = [row.sample_tag for row in rows]
        if len(set(tags)) != len(tags) or [r.sample_index for r in rows] != list(
            range(len(rows))
        ):
            raise ValueError(
                "Archive requires unique tags and contiguous sample indices."
            )
        # A strict one-to-one inventory prevents a partial download from looking
        # like a complete population. Source path columns are historical metadata.
        for folder, suffix in (
            ("meshes", ".obj"),
            ("landmarks", ".mrk.json"),
            ("images", ".png"),
        ):
            found = {
                p.name[: -len(suffix)] for p in (source / folder).glob("*" + suffix)
            }
            if found != set(tags):
                raise ValueError(f"{folder}: missing or extra sample files.")
        samples = []
        donor_counts = Counter()
        donor_indices = {}
        donor_hashes = defaultdict(set)
        donor_examples = {}
        dimensions = Counter()
        for row in rows:
            donor, index = row.sample_tag.rsplit("_fishy_", 1)
            if index != f"{row.sample_index:06d}":
                raise ValueError(f"Sample tag/index mismatch: {row.sample_tag}")
            if donor_indices.setdefault(row.specimen_index, donor) != donor:
                raise ValueError("A specimen index maps to multiple donors.")
            if (
                donor in donor_indices.values()
                and sum(value == donor for value in donor_indices.values()) != 1
            ):
                raise ValueError("A donor maps to multiple specimen indices.")
            assets = [
                Asset(path=path, sha256=Artifacts.digest(path))
                for path in (
                    source / "meshes" / (row.sample_tag + ".obj"),
                    source / "landmarks" / (row.sample_tag + ".mrk.json"),
                    source / "images" / (row.sample_tag + ".png"),
                )
            ]
            # Decode every image, rather than trusting a PNG header alone.
            with Image.open(assets[2].path) as image:
                image.load()
                dimensions[f"{image.width}x{image.height}:{image.mode}"] += 1
            sample = Sample(
                sample_id=row.sample_tag,
                donor_id=donor,
                mesh=assets[0],
                landmarks=assets[1],
                texture=assets[2],
            )
            samples.append(sample)
            donor_counts[donor] += 1
            donor_hashes[donor].add((assets[0].sha256, assets[1].sha256))
            donor_examples.setdefault(donor, sample)
        if sorted(donor_indices) != list(range(len(donor_indices))):
            raise ValueError("Specimen indices must be contiguous from zero.")
        # Repeated donors may share geometry, but never collapse population rows.
        if any(len(hashes) != 1 for hashes in donor_hashes.values()):
            raise ValueError("Mesh/landmark content changes within an archived donor.")
        landmark_audit = {}
        specimens = []
        for donor, sample in donor_examples.items():
            data = json.loads(sample.landmarks.path.read_text())
            markup = next(m for m in data["markups"] if m["type"] == "Fiducial")
            labels = [p["label"] for p in markup["controlPoints"]]
            counts = Counter(labels)
            entry = {
                "point_count": len(labels),
                "duplicate_labels": {k: v for k, v in counts.items() if v > 1},
                "whitespace_labels": [
                    label for label in labels if label != label.strip()
                ],
                "coordinate_system": markup.get("coordinateSystem"),
                "recorded_landmarks_used": sorted(
                    {
                        r.n_landmarks_used
                        for r in rows
                        if r.sample_tag.startswith(donor + "_fishy_")
                    }
                ),
            }
            # Use the actual core reader to expose incompatibility. Record errors
            # as evidence; do not relax the core's anatomical matching contract.
            try:
                specimen = Dataset.inspect_specimen(
                    donor, sample.mesh.path, sample.landmarks.path, sample.texture.path
                )
                specimens.append(specimen)
                entry["core_error"] = None
            except ValueError as error:
                entry["core_error"] = str(error)
            landmark_audit[donor] = entry
        transfer_errors = [
            f"{donor}: {entry['core_error']}"
            for donor, entry in landmark_audit.items()
            if entry["core_error"]
        ]
        try:
            Dataset.validate_specimens(specimens, require_textures=True)
        except ValueError as error:
            transfer_errors.append(str(error))
        # Same-seed regeneration is only a comparison. Never substitute generated
        # mixture labels when the archived vectors disagree with that sequence.
        generated = Generation.sample_labeled(len(rows), seed)
        differences = {}
        for name in ParameterData.model_fields:
            observed = np.array([getattr(row, name) for row in rows])
            expected = np.array([getattr(s.parameters, name) for s in generated])
            differences[name] = {
                "max_absolute_difference": float(np.max(np.abs(observed - expected))),
                "mismatched_rows_at_1e_minus_12": int(
                    np.sum(~np.isclose(observed, expected, rtol=0, atol=1e-12))
                ),
            }
        assigned = np.random.RandomState(seed).randint(0, len(donor_counts), len(rows))
        report = {
            "source": str(source),
            "sample_count": len(rows),
            "donor_counts": dict(sorted(donor_counts.items())),
            "image_formats": dict(dimensions),
            "cheek_positive": sum(r.rosy_cheeks_present for r in rows),
            "stripe_counts": dict(Counter(r.stripe_count for r in rows)),
            "landmarks": landmark_audit,
            "transfer_ready": not transfer_errors,
            "transfer_errors": transfer_errors,
            "sampler_comparison_seed": seed,
            "parameter_differences": differences,
            "donor_assignment_matches": bool(
                np.array_equal(assigned, [r.specimen_index for r in rows])
            ),
            "labels_note": "Original CSV preserved; no belly/tail mixture labels inferred.",
            "scope": "Input inventory only; no atlas, bake, or paper result verified.",
        }
        manifest = Manifest(kind="generated", backend="generator", samples=samples)
        Artifacts.start(
            output, "import-archived-inputs", [csv_path, source / "README.md"]
        )
        Artifacts.write(output / "manifest.json", manifest.model_dump(mode="json"))
        Artifacts.write(output / "audit.json", report)
        shutil.copyfile(csv_path, output / "samples.original.csv")
        shutil.copyfile(source / "README.md", output / "README.original.md")
        Artifacts.complete(output)
        print(f"Imported {len(rows)} samples from {len(donor_counts)} donors.")
        print(
            f"Current-core transfer ready: {report['transfer_ready']}. See {output / 'audit.json'}"
        )
