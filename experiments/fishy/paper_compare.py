"""Compare regenerated paper results to their pinned upstream reference.

Strict numerical agreement is reported separately from cross-platform changes;
this command does not relax tolerance to turn a differing run into a match.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.fishy.paper import Paper

IGNORE_FIELDS = {"seconds", "elapsed", "time", "runtime_s", "wall_s"}
REQUIRED_JSON = (
    "cache_meta.json",
    "e1_fidelity.json",
    "e1b_correspondence.json",
    "e2_preservation.json",
    "e2b_donorblocked.json",
    "e2c_landmark_baseline.json",
    "e3_module_eda.json",
    "e4_byoa_descriptors.json",
    "e5b_expert_repr.json",
    "e6_negative_results.json",
    "e6b_limits.json",
)
REQUIRED_ARRAYS = (
    "e1_fidelity_arrays.npz",
    "e1b_arrays.npz",
    "e2_preservation_arrays.npz",
    "e2c_landmark_baseline_arrays.npz",
    "e3_module_eda_arrays.npz",
    "e4_byoa_descriptors_arrays.npz",
    "e5b_arrays.npz",
    "e6_negative_results_arrays.npz",
)


class Compare:
    """Operations that compare observable outputs without changing the reference."""

    @staticmethod
    def leaves(value, prefix: str = "") -> dict:
        """Flatten JSON while preserving semantic paths and list ordering."""
        output = {}
        if isinstance(value, dict):
            for key, item in value.items():
                if key not in IGNORE_FIELDS:
                    output.update(Compare.leaves(item, f"{prefix}.{key}"))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                output.update(Compare.leaves(item, f"{prefix}[{index}]"))
        else:
            output[prefix] = value
        return output

    @staticmethod
    def json(reference: Path, current: Path, tolerance: float) -> dict:
        """Report all changed values, their absolute error, and relative error."""
        left = Compare.leaves(json.loads(reference.read_text()))
        right = Compare.leaves(json.loads(current.read_text()))
        differences = []
        outside = 0
        for key in sorted(left.keys() | right.keys()):
            a, b = left.get(key), right.get(key)
            if key in left and key in right and a == b:
                continue
            numeric = all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b)
            )
            # Upstream JSON uses NaN for undefined exploratory scores. Preserve
            # undefined status, and distinguish it from finite replacements.
            if numeric and math.isnan(a) and math.isnan(b):
                continue
            absolute = abs(a - b) if numeric else None
            relative = absolute / max(abs(a), abs(b), 1e-12) if numeric else None
            close = numeric and math.isfinite(relative) and relative <= tolerance
            outside += not close
            differences.append(
                {
                    "path": key,
                    "reference": Compare.clean(a),
                    "current": Compare.clean(b),
                    "absolute_difference": Compare.clean(absolute),
                    "relative_difference": Compare.clean(relative),
                    "within_tolerance": bool(close),
                    "reference_present": key in left,
                    "current_present": key in right,
                }
            )
        return {
            "status": "IDENTICAL"
            if not differences
            else "CLOSE"
            if not outside
            else "DIFFERS",
            "leaves": len(left.keys() | right.keys()),
            "outside_tolerance": outside,
            "differences": differences,
        }

    @staticmethod
    def clean(value):
        """Encode nonfinite reference observations explicitly in strict output JSON."""
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value

    @staticmethod
    def arrays(reference: Path, current: Path, tolerance: float) -> dict:
        """Compare shapes, missing fields and every numerical array element."""
        rows = {}
        with (
            np.load(reference, allow_pickle=False) as left,
            np.load(current, allow_pickle=False) as right,
        ):
            for key in sorted(set(left.files) | set(right.files)):
                if key not in left.files or key not in right.files:
                    rows[key] = {"status": "MISSING"}
                    continue
                # Object arrays in the historical bundle are specimen strings,
                # but opening pickled arrays is unnecessary for numeric parity.
                try:
                    a, b = left[key], right[key]
                except ValueError as error:
                    if "Object arrays cannot be loaded" not in str(error):
                        raise
                    rows[key] = {"status": "UNVERIFIED_OBJECT_ARRAY"}
                    continue
                if a.shape != b.shape:
                    rows[key] = {
                        "status": "SHAPE_DIFFERS",
                        "reference_shape": list(a.shape),
                        "current_shape": list(b.shape),
                    }
                elif a.dtype.kind in "fciub" and b.dtype.kind in "fciub":
                    same_nonfinite = (
                        np.array_equal(np.isnan(a), np.isnan(b))
                        and np.array_equal(np.isposinf(a), np.isposinf(b))
                        and np.array_equal(np.isneginf(a), np.isneginf(b))
                    )
                    finite = np.isfinite(a) & np.isfinite(b)
                    absolute = (
                        float(
                            np.max(
                                np.abs(
                                    a[finite].astype(float) - b[finite].astype(float)
                                )
                            )
                        )
                        if finite.any()
                        else 0.0
                    )
                    scale = max(
                        float(np.max(np.abs(a[finite]))) if finite.any() else 0.0, 1e-12
                    )
                    rows[key] = {
                        "status": "CLOSE"
                        if same_nonfinite and absolute / scale <= tolerance
                        else "DIFFERS",
                        "shape": list(a.shape),
                        "max_absolute_difference": absolute,
                        "relative_to_reference_magnitude": absolute / scale,
                        "same_nonfinite_pattern": same_nonfinite,
                    }
                else:
                    rows[key] = {
                        "status": "IDENTICAL" if np.array_equal(a, b) else "DIFFERS",
                        "shape": list(a.shape),
                    }
        return rows


def main() -> None:
    """Write the detailed comparison without overwriting scientific results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-arrays", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()
    if not math.isfinite(args.rtol) or args.rtol < 0:
        parser.error("--rtol must be finite and nonnegative")
    work = args.output.resolve() / "source/paper_synth"
    report = {
        "relative_tolerance": args.rtol,
        "json": {},
        "arrays": {},
        "array_scope": "Required analysis array files; object arrays remain unverified"
        if args.reference_arrays
        else "No reference arrays supplied",
    }
    for name in REQUIRED_JSON:
        current = work / "results" / name
        report["json"][name] = (
            Compare.json(work / "results_ref" / name, current, args.rtol)
            if current.exists()
            else {"status": "MISSING"}
        )
    if args.reference_arrays:
        report["excluded_supporting_reference_arrays"] = sorted(
            path.name
            for path in args.reference_arrays.glob("*.npz")
            if path.name not in REQUIRED_ARRAYS
        )
        for name in REQUIRED_ARRAYS:
            path = args.reference_arrays / name
            current = work / "results" / name
            if not path.exists() or not current.exists():
                report["arrays"][name] = {
                    "__file__": {
                        "status": "MISSING_REFERENCE"
                        if not path.exists()
                        else "MISSING_CURRENT"
                    }
                }
            else:
                report["arrays"][name] = Compare.arrays(path, current, args.rtol)
    Paper.write(args.output.resolve() / "comparison.json", report)
    for name, row in report["json"].items():
        print(f"{name}: {row['status']}")
    raise SystemExit(
        1
        if any(
            r["status"] not in ("IDENTICAL", "CLOSE") for r in report["json"].values()
        )
        or any(
            field["status"] not in ("IDENTICAL", "CLOSE", "UNVERIFIED_OBJECT_ARRAY")
            for fields in report["arrays"].values()
            for field in fields.values()
        )
        else 0
    )


if __name__ == "__main__":
    main()
