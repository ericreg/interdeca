"""Export a readable population figure from a completed, recorded analysis.

Dense specimen labels remain in the coordinate table; the plot highlights only
the camera-reference specimen so adjacent points do not obscure one another.
"""

import csv
import hashlib
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np


@click.command()
@click.option(
    "--run", "run_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def main(run_path: Path, output: Path) -> None:
    """Export PNG and PDF to a new directory, retaining source-table provenance."""
    run_path = run_path.expanduser().resolve()
    config = json.loads((run_path / "config.json").read_text())
    summary = json.loads((run_path / "completed.json").read_text())
    table = run_path / "analysis/coordinates.csv"
    with table.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != summary["specimens"] or len(
        {r["specimen_id"] for r in rows}
    ) != len(rows):
        raise ValueError("Coordinate table does not match the completed cohort")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    reference = next(
        i
        for i, r in enumerate(rows)
        if r["specimen_id"] == config["orientation_reference_id"]
    )
    variance = summary["PCA_explained_variance_ratio"]
    for axis, fields, title in zip(
        axes,
        (("PC1", "PC2"), ("UMAP1", "UMAP2")),
        ("PCA · standardized features", summary["UMAP_method"]),
        strict=True,
    ):
        points = np.array([[float(r[field]) for field in fields] for r in rows])
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite coordinates")
        axis.scatter(points[:, 0], points[:, 1], s=28, color="#087f8c", alpha=0.85)
        point = points[reference]
        axis.scatter(
            *point, s=90, facecolors="none", edgecolors="#23313b", linewidths=1.3
        )
        axis.annotate(
            config["orientation_reference_id"],
            point,
            xytext=(10, 13),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#23313b", "lw": 0.7},
        )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel(
            f"PC1 ({variance[0]:.1%} variance)"
            if fields[0] == "PC1"
            else "UMAP coordinate 1"
        )
        axis.set_ylabel(
            f"PC2 ({variance[1]:.1%} variance)"
            if fields[1] == "PC2"
            else "UMAP coordinate 2"
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(0.12)
    figure.suptitle(
        f"Frozen I-JEPA · {len(rows)} mussels · {summary['crops_per_specimen']} surface crops per specimen\n"
        "Descriptive full-cohort fits; outlined point is the camera reference",
        fontsize=12,
    )
    figure.savefig(output / "population.png", dpi=220)
    figure.savefig(output / "population.pdf")
    plt.close(figure)
    provenance = {
        "source_run": str(run_path),
        "source_coordinate_table_sha256": hashlib.sha256(
            table.read_bytes()
        ).hexdigest(),
        "source_summary_sha256": hashlib.sha256(
            (run_path / "completed.json").read_bytes()
        ).hexdigest(),
        "scope": summary["scope"],
        "specimens": len(rows),
        "render_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
