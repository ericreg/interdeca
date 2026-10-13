"""Exercise real fish inputs through the pinned fidelity kernels without full ML."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import common as C
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from e1_fidelity import _closest_point_uv, _sample_tex, deltaE2000
from fishpipe.features import rgb_to_lab
from fishpipe.mesh import load_obj, sample_face_colors
from scipy.spatial import cKDTree
from skimage.color import deltaE_ciede2000

from experiments.fishy.serde import Documents


def main() -> None:
    """Process one real specimen per donor and save fresh arrays plus a plot.

    The parent command supplies a fresh copy of the reference source on
    PYTHONPATH. Its caches/results therefore live only under the new smoke run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    root, output = args.bundle.resolve(), args.output.resolve()
    inventory = Documents.legacy_specimens(root / "specimens.json")
    selected = {}
    for row in sorted(inventory, key=lambda r: r.sample_id):
        selected.setdefault(row.donor_id, row.sample_id)
    if len(selected) != 7:
        raise ValueError("The representative smoke requires all seven donors.")
    atlas = load_obj(C.ATLAS_MESH)
    if atlas.n_faces != 49997 or len(atlas.vertices) != 24999:
        raise ValueError("The archived atlas dimensions differ from the paper.")
    faces = np.linspace(0, atlas.n_faces - 1, 1024, dtype=int)
    reference = root / "reference/local-2026-10-06/results"
    with np.load(
        reference / "e2_preservation_arrays.npz", allow_pickle=False
    ) as arrays:
        names = list(arrays["names"].astype(str))
    with np.load(reference / "e1_fidelity_arrays.npz", allow_pickle=False) as arrays:
        expected = arrays["dE"]
    deltas, records = [], []
    # Each specimen passes through raw OBJ/PNG loading, texture sampling,
    # source-surface projection, Lab conversion, and a fresh CIEDE2000 result.
    for donor, name in sorted(selected.items()):
        native = imageio.imread(C.SRC_IMAGES / f"{name}.png")
        baked = imageio.imread(C.ATLAS_TEX / f"{name}.png")
        aligned = load_obj(C.MOD_RUN / "ATLAS/alignedModels" / f"{name}_align.obj")
        resampled = load_obj(
            C.ATLAS_DIR / "resampledOBJ_withUV" / f"{name}_resampled.obj"
        )
        if resampled.n_faces != atlas.n_faces:
            raise ValueError(f"Correspondence face count changed for {name}.")
        atlas_rgb = sample_face_colors(atlas, baked)[faces]
        triangles = aligned.vertices[aligned.face_v]
        uvs = aligned.uvs[aligned.face_vt]
        corners = triangles.reshape(-1, 3)
        tree = cKDTree(corners)
        query = resampled.vertices[resampled.face_v][faces].mean(axis=1)
        uv, distance = _closest_point_uv(
            query, triangles, uvs, tree, np.repeat(np.arange(len(triangles)), 3)
        )
        reference_lab = rgb_to_lab(_sample_tex(native, uv))
        atlas_lab = rgb_to_lab(atlas_rgb)
        full_precision = deltaE2000(reference_lab, atlas_lab)
        independent = deltaE_ciede2000(reference_lab, atlas_lab)
        color_error = float(np.max(np.abs(full_precision - independent)))
        if not np.isfinite(full_precision).all() or color_error > 1e-10:
            raise ValueError(f"Independent color calculation failed for {name}.")
        delta = full_precision.astype(np.float32)
        retained = expected[names.index(name), faces]
        max_error = float(np.max(np.abs(delta - retained)))
        relative = max_error / max(float(np.max(np.abs(retained))), 1e-12)
        records.append(
            {
                "sample_id": name,
                "donor_id": donor,
                "faces": len(faces),
                "median_dE": float(np.median(delta)),
                "p95_dE": float(np.percentile(delta, 95)),
                "projection_p95": float(np.percentile(distance, 95)),
                "independent_color_max_abs_error": color_error,
                "local_reference_max_abs_error": max_error,
                "local_reference_relative_max_error": relative,
                "local_reference_within_1e_6": relative <= 1e-6,
            }
        )
        deltas.append(delta)
        print(
            f"smoke donor {donor}: {name}; median dE {np.median(delta):.6f}", flush=True
        )
    values = np.stack(deltas)
    np.savez_compressed(
        output / "smoke-arrays.npz",
        dE=values,
        face_idx=faces,
        sample_ids=np.array([r["sample_id"] for r in records]),
    )
    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for row, values_row in zip(records, values, strict=True):
        axis.plot(
            np.sort(values_row),
            np.arange(1, len(values_row) + 1) / len(values_row),
            label=f"Donor {row['donor_id']}",
        )
    axis.set(
        xlabel="CIEDE2000 color difference",
        ylabel="Cumulative fraction",
        title="Representative smoke: 1,024 faces from each of seven donors",
        xlim=(0, 15),
    )
    axis.legend(ncol=2)
    figure.savefig(output / "smoke-fidelity.png", dpi=160)
    plt.close(figure)
    manifest = Documents.manifest(root / "manifest.json")
    receipt = {
        "execution": "PASS",
        "scope": "Seven real specimens, one per donor; 1,024 faces each. Not full paper reconstruction.",
        "local_numerical_parity": "PASS"
        if all(r["local_reference_within_1e_6"] for r in records)
        else "DIFFERS",
        "relative_tolerance": 1e-6,
        "source_tree_sha256": manifest.source_tree_sha256,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            p.metadata["Name"]: p.version for p in importlib.metadata.distributions()
        },
        "seconds": round(time.monotonic() - started, 3),
        "specimens": records,
        "output_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("smoke-arrays.npz", "smoke-fidelity.png")
        },
    }
    (output / "smoke.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in ("packages", "specimens")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
