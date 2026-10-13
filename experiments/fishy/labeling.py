"""Explicit source-backed labels for archives missing generator mixture IDs."""

import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from experiments.fishy.artifacts import Artifacts
from experiments.fishy.serde import ArchiveRow


class Labeling:
    """Keep legacy hue clustering distinct from actual generating assignments."""

    @staticmethod
    def legacy_kmeans(csv_path: Path, output: Path) -> None:
        """Port fishpipe.data._kmeans_1d_labels at source revision 3db23a1.

        This protocol clusters archived scalar hues, then orders cluster IDs by
        increasing center. It does not recover the original make_blobs IDs.
        """
        with csv_path.open(newline="") as stream:
            rows = [ArchiveRow.from_csv(row) for row in csv.DictReader(stream)]
        if len(rows) < 2 or len({r.sample_tag for r in rows}) != len(rows):
            raise ValueError("Label preparation requires at least two unique samples.")
        # Canonical index order makes the seeded fit independent of CSV row order.
        rows.sort(key=lambda r: r.sample_index)
        if [r.sample_index for r in rows] != list(range(len(rows))):
            raise ValueError("Sample indices must be unique and contiguous from zero.")
        labels, centers = {}, {}
        for name in ("belly", "tail"):
            values = np.array([getattr(r, name + "_hue") for r in rows])
            if len(np.unique(values)) < 2:
                raise ValueError(f"{name}: cannot form two hue clusters.")
            fitted = KMeans(n_clusters=2, n_init=10, random_state=0).fit(
                values[:, None]
            )
            order = np.argsort(fitted.cluster_centers_.ravel())
            remap = np.empty(2, dtype=int)
            remap[order] = np.arange(2)
            labels[name] = remap[fitted.labels_]
            centers[name] = fitted.cluster_centers_.ravel()[order].tolist()
        Artifacts.start(output, "labels-legacy-kmeans", [csv_path])
        with (output / "labels.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "sample_id",
                    "donor_id",
                    "belly_group",
                    "tail_group",
                    "stripe_count",
                    "rosy_cheeks_present",
                ],
            )
            writer.writeheader()
            for index, row in enumerate(rows):
                donor, suffix = row.sample_tag.rsplit("_fishy_", 1)
                if suffix != f"{row.sample_index:06d}":
                    raise ValueError("Sample tag disagrees with sample index.")
                writer.writerow(
                    {
                        "sample_id": row.sample_tag,
                        "donor_id": donor,
                        "belly_group": int(labels["belly"][index]),
                        "tail_group": int(labels["tail"][index]),
                        "stripe_count": row.stripe_count,
                        "rosy_cheeks_present": row.rosy_cheeks_present,
                    }
                )
        Artifacts.write(
            output / "label-protocol.json",
            {
                "protocol": "legacy-kmeans-hue",
                "source_revision": "3db23a1de18eb2a24d073e9a571edda290e5937f",
                "source_path": "fishy_research/src/fishpipe/data.py",
                "settings": {"n_clusters": 2, "n_init": 10, "random_state": 0},
                "ordering": "sample_index for fit; increasing hue center for group IDs",
                "centers": centers,
                "counts": {
                    k: np.bincount(v, minlength=2).tolist() for k, v in labels.items()
                },
                "limitation": "Source-backed inferred hue groups, not generating mixture IDs; figure-script provenance remains unverified.",
            },
        )
        Artifacts.complete(output)
