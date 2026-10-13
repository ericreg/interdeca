"""Export a readable population figure from a completed, recorded analysis.

Dense specimen labels remain in the coordinate table; the plot highlights only
the camera-reference specimen so adjacent points do not obscure one another.
"""

import hashlib
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from experiments.learned_color.serde import StageData


@click.command()
@click.option(
    "--run", "run_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option(
    "--allow-legacy-unverified",
    is_flag=True,
    help="Export an old run missing analysis or config hashes, recording the limits.",
)
def main(run_path: Path, output: Path, allow_legacy_unverified: bool) -> None:
    """Export PNG and PDF to a new directory, retaining source-table provenance."""
    run_path = run_path.expanduser().resolve()
    config = StageData.load_options(run_path / "config.json")
    summary = StageData.read_completion(run_path / "completed.json")
    # Coordinates alone cannot certify the labels: the saved configuration also
    # determines PCA standardization and which specimen is the camera reference.
    config_digest = hashlib.sha256((run_path / "config.json").read_bytes()).hexdigest()
    if summary.config_sha256 and summary.config_sha256 != config_digest:
        raise ValueError("Configuration checksum changed")
    record = StageData.read_run(run_path / "run.json")
    if record.status != "completed":
        raise ValueError("Run is not completed")
    table = run_path / "analysis/coordinates.csv"
    rows = StageData.read_coordinates(table)
    identifiers = [row.specimen_id for row in rows]
    recorded_ids = [specimen.specimen_id for specimen in record.inputs]
    # Count alone cannot distinguish 32 mussels from another 32-specimen table.
    # Keep the exact order that joins coordinates to the frozen input manifest.
    if (
        len(rows) != summary.specimens
        or len(rows) != config.expected_pair_count
        or len(set(identifiers)) != len(rows)
        or identifiers != recorded_ids
        or (config.specimen_ids and identifiers != sorted(config.specimen_ids))
    ):
        raise ValueError("Coordinate table does not match the completed cohort")
    if (
        summary.profile != config.profile
        or summary.crops_per_specimen != config.patches_per_specimen
        or summary.checkpoint_revision != config.model_revision
        or config.orientation_reference_id not in identifiers
    ):
        raise ValueError("Figure summary does not match the recorded protocol")
    if not StageData.matching_completion(
        run_path / "analysis/summary.json", run_path / "completed.json"
    ):
        raise ValueError("Completion marker differs from the analysis summary")
    # Historical runs predate these hashes. Explicit opt-in preserves access to
    # them without claiming that the exporter certified their numerical bytes.
    missing_integrity_checks = []
    if not summary.config_sha256:
        missing_integrity_checks.append("config")
    if not summary.analysis_sha256:
        missing_integrity_checks.append("analysis")
    else:
        if "coordinates.csv" not in summary.analysis_sha256:
            raise ValueError("Completion marker has no coordinate checksum")
        for name, expected in summary.analysis_sha256.items():
            if Path(name).name != name:
                raise ValueError("Analysis checksum must name a file inside analysis/")
            if (
                hashlib.sha256((run_path / "analysis" / name).read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError(f"Analysis checksum changed: {name}")
    if missing_integrity_checks and not allow_legacy_unverified:
        raise ValueError(
            "Legacy run lacks completion hashes for "
            f"{', '.join(missing_integrity_checks)}; use --allow-legacy-unverified"
        )
    integrity = (
        "legacy_without_" + "_and_".join(missing_integrity_checks) + "_hashes"
        if missing_integrity_checks
        else "completion_hashes_verified"
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    reference = next(
        i
        for i, r in enumerate(rows)
        if r.specimen_id == config.orientation_reference_id
    )
    variance = summary.PCA_explained_variance_ratio
    for axis, fields, title in zip(
        axes,
        (("PC1", "PC2"), ("UMAP1", "UMAP2")),
        (
            "PCA · standardized features"
            if config.standardize_pca_features
            else "PCA · unstandardized features",
            summary.UMAP_method,
        ),
        strict=True,
    ):
        points = np.array([[getattr(r, field) for field in fields] for r in rows])
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite coordinates")
        axis.scatter(points[:, 0], points[:, 1], s=28, color="#087f8c", alpha=0.85)
        point = points[reference]
        axis.scatter(
            *point, s=90, facecolors="none", edgecolors="#23313b", linewidths=1.3
        )
        axis.annotate(
            config.orientation_reference_id,
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
            if summary.UMAP_method == "UMAP cosine"
            else "Fallback PC1"
        )
        axis.set_ylabel(
            f"PC2 ({variance[1]:.1%} variance)"
            if fields[1] == "PC2"
            else "UMAP coordinate 2"
            if summary.UMAP_method == "UMAP cosine"
            else "Fallback PC2"
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(0.12)
    figure.suptitle(
        f"Frozen I-JEPA · {len(rows)} specimens · {summary.crops_per_specimen} surface crops per specimen\n"
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
        "source_config_sha256": config_digest,
        "scope": summary.scope,
        "profile": summary.profile,
        "specimen_ids": identifiers,
        "analysis_integrity": integrity,
        "missing_integrity_checks": missing_integrity_checks,
        "specimens": len(rows),
        "render_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
