"""Compare regenerated paper results to their pinned upstream reference.

Strict numerical agreement is reported separately from cross-platform changes;
this command does not relax tolerance to turn a differing run into a match.
"""

import argparse
import json
import math
from pathlib import Path
from typing import NamedTuple

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


DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-8
POLICY = "symmetric-elementwise-v2"
CRITERION = (
    "abs(a-b) <= atol + rtol * max(abs(a), abs(b)); discrete values and structure exact"
)


class JSONNode(NamedTuple):
    """A JSON node keeps container/type identity even when it has no children."""

    kind: str
    value: object


class FloatingErrors(NamedTuple):
    """Elementwise errors; undefined entries never certify numerical parity."""

    finite: np.ndarray
    undefined: np.ndarray
    outside: np.ndarray
    same_nonfinite: np.ndarray
    absolute: np.ndarray
    relative: np.ndarray
    allowed: np.ndarray
    ratio: np.ndarray


class ArrayFailure(NamedTuple):
    """One bounded failure example, ordered by error relative to its allowance."""

    index: int
    reference: object
    current: object
    absolute: float | None
    allowed: float | None
    ratio: float | None
    reason: str
    priority: float


class ArraySummary(NamedTuple):
    """Scalar reductions and at most ten examples, independent of array length."""

    finite_pairs: int
    changed: int
    outside: int
    undefined: int
    pattern_mismatches: int
    absolute: float
    relative: float
    ratio: float
    reference_scale: float
    worst: int | None
    examples: tuple[ArrayFailure, ...]


class Compare:
    """Compare observable outputs without fitting transforms or changing inputs."""

    @staticmethod
    def tolerances(tolerance: float, atol: float) -> None:
        """Tolerances are finite nonnegative scalars in the stored value's units."""
        for name, value in (("rtol", tolerance), ("atol", atol)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite nonnegative number.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number.")

    @staticmethod
    def nodes(value: object, prefix: str = "") -> dict[str, JSONNode]:
        """Keep every container and leaf; escape keys to prevent path collisions."""
        if isinstance(value, dict):
            output = {
                prefix: JSONNode(
                    "object", sum(key not in IGNORE_FIELDS for key in value)
                )
            }
            for key, item in value.items():
                if key not in IGNORE_FIELDS:
                    suffix = f".{key}" if key.isidentifier() else f"[{json.dumps(key)}]"
                    output.update(Compare.nodes(item, prefix + suffix))
            return output
        if isinstance(value, list):
            output = {prefix: JSONNode("array", len(value))}
            for index, item in enumerate(value):
                output.update(Compare.nodes(item, f"{prefix}[{index}]"))
            return output
        # Python equality conflates True and 1. Keeping kinds makes identifiers,
        # counts, booleans and nulls exact, independently of floating tolerance.
        return {prefix: JSONNode(type(value).__name__, value)}

    @staticmethod
    def leaves(value: object, prefix: str = "") -> dict:
        """Return leaves and empty containers without losing their identities."""
        nodes = Compare.nodes(value, prefix)
        return {
            path: node.value
            if node.kind not in ("array", "object")
            else []
            if node.kind == "array"
            else {}
            for path, node in nodes.items()
            if node.kind not in ("array", "object") or node.value == 0
        }

    @staticmethod
    def clean(value: object) -> object:
        """Encode undefined/overflow diagnostics as strings in strict JSON."""
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (complex, np.complexfloating)):
            return {
                "real": Compare.clean(float(value.real)),
                "imag": Compare.clean(float(value.imag)),
            }
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            return str(value)
        return value

    @staticmethod
    def floating(
        a: np.ndarray, b: np.ndarray, tolerance: float, atol: float
    ) -> FloatingErrors:
        """Apply the symmetric bound independently to corresponding elements.

        Normalize by the largest component before subtracting. This prevents
        overflow for finite opposite-sign values near the floating-point limit.
        Complex finite entries use complex magnitude, not a discarded real part.
        """
        dtype = (
            np.clongdouble
            if np.iscomplexobj(a) or np.iscomplexobj(b)
            else np.longdouble
        )
        a, b = a.astype(dtype), b.astype(dtype)
        finite = np.isfinite(a) & np.isfinite(b)
        same = np.ones(a.shape, dtype=bool)
        components = (
            ((a.real, b.real), (a.imag, b.imag)) if np.iscomplexobj(a) else ((a, b),)
        )
        for left, right in components:
            same &= (
                (np.isnan(left) == np.isnan(right))
                & (np.isposinf(left) == np.isposinf(right))
                & (np.isneginf(left) == np.isneginf(right))
            )
        # Only finite entries enter subtraction. Matching undefined patterns stay
        # visible, and finite/undefined or opposite infinities are discrepancies.
        left, right = np.where(finite, a, 0), np.where(finite, b, 0)
        scale = np.maximum(np.abs(left.real), np.abs(right.real))
        if np.iscomplexobj(left):
            scale = np.maximum(scale, np.maximum(np.abs(left.imag), np.abs(right.imag)))
        divisor = np.where(scale == 0, 1, scale)
        with np.errstate(
            over="ignore", invalid="ignore", divide="ignore", under="ignore"
        ):
            normalized = np.abs(left / divisor - right / divisor)
            magnitude = np.maximum(np.abs(left / divisor), np.abs(right / divisor))
            bound = atol / divisor + tolerance * magnitude
            absolute = normalized * divisor
            allowed = atol + (tolerance * magnitude) * divisor
            relative = np.divide(
                normalized,
                magnitude,
                out=np.zeros_like(normalized),
                where=magnitude != 0,
            )
            ratio = np.divide(
                normalized,
                bound,
                out=np.full_like(normalized, np.inf),
                where=bound != 0,
            )
            ratio[(bound == 0) & (normalized == 0)] = 0
        outside = (~finite & ~same) | (finite & (normalized > bound))
        # A complex entry can have an undefined real part and a changed finite
        # imaginary part. Do not let the undefined component hide that change.
        if np.iscomplexobj(a):
            for left_part, right_part in components:
                both_finite = np.isfinite(left_part) & np.isfinite(right_part)
                component = Compare.floating(left_part, right_part, tolerance, atol)
                outside |= ~finite & both_finite & component.outside
        return FloatingErrors(
            finite, ~finite, outside, same, absolute, relative, allowed, ratio
        )

    @staticmethod
    def json(
        reference: Path,
        current: Path,
        tolerance: float = DEFAULT_RTOL,
        *,
        atol: float = DEFAULT_ATOL,
    ) -> dict:
        """Compare structure/discrete values exactly and finite floats symmetrically."""
        Compare.tolerances(tolerance, atol)
        left = Compare.nodes(json.loads(reference.read_text()))
        right = Compare.nodes(json.loads(current.read_text()))
        differences, undefined = [], []
        finite_pairs = 0
        for key in sorted(left.keys() | right.keys()):
            a, b = left.get(key), right.get(key)
            same_kind = a is not None and b is not None and a.kind == b.kind
            if same_kind and a.kind in ("object", "array"):
                continue
            absolute = relative = allowed = ratio = None
            if same_kind and a.kind == "float":
                errors = Compare.floating(
                    np.array(a.value), np.array(b.value), tolerance, atol
                )
                if bool(errors.undefined):
                    undefined.append(
                        {
                            "path": key,
                            "reference": Compare.clean(a.value),
                            "current": Compare.clean(b.value),
                            "same_nonfinite_pattern": bool(errors.same_nonfinite),
                        }
                    )
                    if not bool(errors.outside):
                        continue
                else:
                    finite_pairs += 1
                    if a.value == b.value:
                        continue
                    absolute, relative, allowed, ratio = [
                        float(value)
                        for value in (
                            errors.absolute,
                            errors.relative,
                            errors.allowed,
                            errors.ratio,
                        )
                    ]
                close = not bool(errors.outside)
            else:
                if same_kind and a.value == b.value:
                    continue
                close = False
            differences.append(
                {
                    "path": key,
                    "reference": Compare.clean(a.value) if a else None,
                    "current": Compare.clean(b.value) if b else None,
                    "reference_type": a.kind if a else None,
                    "current_type": b.kind if b else None,
                    "absolute_difference": Compare.clean(absolute),
                    "relative_difference": Compare.clean(relative),
                    "allowed_absolute_difference": Compare.clean(allowed),
                    "error_to_tolerance_ratio": Compare.clean(ratio),
                    "within_tolerance": close,
                    "reference_present": a is not None,
                    "current_present": b is not None,
                }
            )
        outside = sum(not row["within_tolerance"] for row in differences)
        finite_status = (
            "DIFFERS" if outside else "CLOSE" if differences else "IDENTICAL"
        )
        return {
            "status": "DIFFERS"
            if outside
            else "UNDEFINED"
            if undefined
            else finite_status,
            "finite_status": finite_status,
            "comparison_policy": POLICY,
            "relative_tolerance": tolerance,
            "absolute_tolerance": atol,
            "criterion": CRITERION,
            "leaves": len(
                set(Compare.leaves(json.loads(reference.read_text())))
                | set(Compare.leaves(json.loads(current.read_text())))
            ),
            "nodes": len(left.keys() | right.keys()),
            "finite_float_pairs": finite_pairs,
            "outside_tolerance": outside,
            "undefined_values": len(undefined),
            "undefined": undefined,
            "differences": differences,
        }

    @staticmethod
    def maximum(values: np.ndarray, finite: np.ndarray) -> float:
        """Reduce finite-pair diagnostics; empty arrays have no observed error."""
        return float(np.max(values[finite])) if finite.any() else 0.0

    @staticmethod
    def array_summary(
        a: np.ndarray,
        b: np.ndarray,
        tolerance: float,
        atol: float,
        *,
        chunk_size: int = 250_000,
    ) -> ArraySummary:
        """Accumulate elementwise diagnostics with bounded numerical scratch space.

        NPZ decompression still materializes each input field. Flat-iterator
        slices then copy at most chunk_size entries, including Fortran arrays;
        no full-size longdouble arrays or Python lists of failures are created.
        """
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        finite_pairs = changed = outside = undefined = pattern_mismatches = 0
        absolute = relative = ratio = reference_scale = 0.0
        worst = None
        examples = []
        for start in range(0, a.size, chunk_size):
            stop = min(start + chunk_size, a.size)
            left, right = a.flat[start:stop], b.flat[start:stop]
            errors = Compare.floating(left, right, tolerance, atol)
            finite_pairs += int(errors.finite.sum())
            changed += int((errors.finite & (left != right)).sum())
            outside += int(errors.outside.sum())
            undefined += int(errors.undefined.sum())
            pattern_mismatches += int((~errors.same_nonfinite).sum())
            absolute = max(absolute, Compare.maximum(errors.absolute, errors.finite))
            relative = max(relative, Compare.maximum(errors.relative, errors.finite))
            if errors.finite.any():
                reference_scale = max(
                    reference_scale, float(np.max(np.abs(left[errors.finite])))
                )
                local_worst = int(np.argmax(np.where(errors.finite, errors.ratio, -1)))
                local_ratio = float(errors.ratio[local_worst])
                if worst is None or local_ratio > ratio:
                    ratio, worst = local_ratio, start + local_worst
            bad = np.flatnonzero(errors.outside)
            if len(bad) > 10:
                priorities = np.where(errors.finite[bad], errors.ratio[bad], np.inf)
                bad = bad[np.argpartition(priorities, -10)[-10:]]
            for index in bad:
                finite = bool(errors.finite[index])
                examples.append(
                    ArrayFailure(
                        start + int(index),
                        left[index],
                        right[index],
                        float(errors.absolute[index]) if finite else None,
                        float(errors.allowed[index]) if finite else None,
                        float(errors.ratio[index]) if finite else None,
                        "FINITE_TOLERANCE"
                        if finite
                        else "NONFINITE_PATTERN_OR_COMPONENT_MISMATCH",
                        float(errors.ratio[index]) if finite else math.inf,
                    )
                )
            # Each chunk contributes at most ten candidates. Keep only the ten
            # greatest violations, rather than sorting every failed array index.
            examples = sorted(
                examples, key=lambda item: (item.priority, -item.index), reverse=True
            )[:10]
        return ArraySummary(
            finite_pairs,
            changed,
            outside,
            undefined,
            pattern_mismatches,
            absolute,
            relative,
            ratio,
            reference_scale,
            worst,
            tuple(examples),
        )

    @staticmethod
    def arrays(
        reference: Path,
        current: Path,
        tolerance: float = DEFAULT_RTOL,
        *,
        atol: float = DEFAULT_ATOL,
    ) -> dict:
        """Compare every NPZ field with exact shape/dtype/discrete contracts.

        Object payloads are never unpickled. Their explicit unverified status is
        incomplete evidence, never a pass for strict numerical parity.
        """
        Compare.tolerances(tolerance, atol)
        rows = {}
        with (
            np.load(reference, allow_pickle=False) as left,
            np.load(current, allow_pickle=False) as right,
        ):
            for key in sorted(set(left.files) | set(right.files)):
                if key not in left.files or key not in right.files:
                    rows[key] = {
                        "status": "MISSING",
                        "reference_present": key in left.files,
                        "current_present": key in right.files,
                    }
                    continue
                try:
                    a, b = left[key], right[key]
                except ValueError as error:
                    if "Object arrays cannot be loaded" not in str(error):
                        raise
                    rows[key] = {
                        "status": "UNVERIFIED_OBJECT_ARRAY",
                        "reason": "Pickle loading is disabled; no parity claim.",
                    }
                    continue
                row = {
                    "shape": list(a.shape),
                    "reference_dtype": str(a.dtype),
                    "current_dtype": str(b.dtype),
                    "dtype_matches": a.dtype == b.dtype,
                }
                if a.shape != b.shape:
                    rows[key] = {
                        **row,
                        "status": "SHAPE_DIFFERS",
                        "reference_shape": list(a.shape),
                        "current_shape": list(b.shape),
                    }
                    continue
                if a.dtype.kind in "fc" and b.dtype.kind in "fc":
                    summary = Compare.array_summary(a, b, tolerance, atol)
                    finite_status = (
                        "DIFFERS"
                        if summary.outside or not row["dtype_matches"]
                        else "CLOSE"
                        if summary.changed
                        else "IDENTICAL"
                    )
                    row.update(
                        {
                            "status": "DIFFERS"
                            if finite_status == "DIFFERS"
                            else "UNDEFINED"
                            if summary.undefined
                            else finite_status,
                            "finite_status": finite_status,
                            "comparison_policy": POLICY,
                            "relative_tolerance": tolerance,
                            "absolute_tolerance": atol,
                            "criterion": CRITERION,
                            "elements": int(a.size),
                            "finite_pairs": summary.finite_pairs,
                            "changed_finite_elements": summary.changed,
                            "outside_tolerance": summary.outside,
                            "undefined_elements": summary.undefined,
                            "nonfinite_pattern_mismatches": summary.pattern_mismatches,
                            "same_nonfinite_pattern": summary.pattern_mismatches == 0,
                            "max_absolute_difference": Compare.clean(summary.absolute),
                            "max_relative_difference": Compare.clean(summary.relative),
                            "max_error_to_tolerance_ratio": Compare.clean(
                                summary.ratio
                            ),
                            "relative_to_reference_magnitude": Compare.clean(
                                summary.absolute / summary.reference_scale
                                if summary.reference_scale
                                else 0.0
                                if summary.absolute == 0
                                else math.inf
                            ),
                            "worst_tolerance_index": [
                                int(i) for i in np.unravel_index(summary.worst, a.shape)
                            ]
                            if summary.worst is not None
                            else None,
                            "outside_examples": [
                                {
                                    "index": [
                                        int(i)
                                        for i in np.unravel_index(item.index, a.shape)
                                    ],
                                    "reference": Compare.clean(item.reference),
                                    "current": Compare.clean(item.current),
                                    "reason": item.reason,
                                    "absolute_difference": Compare.clean(item.absolute),
                                    "allowed_absolute_difference": Compare.clean(
                                        item.allowed
                                    ),
                                    "error_to_tolerance_ratio": Compare.clean(
                                        item.ratio
                                    ),
                                }
                                for item in summary.examples
                            ],
                            "comparison_chunk_elements": 250_000,
                        }
                    )
                else:
                    # Comparing signed/unsigned integers through floating-point
                    # promotion loses identity above 2**53. Convert only the
                    # unsigned side of a guarded nonnegative integer pair.
                    if a.dtype.kind == "u" and b.dtype.kind == "i":
                        equal = (b >= 0) & (a == b.astype(np.uint64))
                    elif a.dtype.kind == "i" and b.dtype.kind == "u":
                        equal = (a >= 0) & (a.astype(np.uint64) == b)
                    elif a.dtype.kind == b.dtype.kind and (
                        a.dtype.kind != "V" or a.dtype == b.dtype
                    ):
                        equal = a == b
                    else:
                        equal = np.zeros(a.shape, dtype=bool)
                    count = int(np.count_nonzero(~equal))
                    row.update(
                        {
                            "status": "IDENTICAL"
                            if not count and row["dtype_matches"]
                            else "DIFFERS",
                            "comparison": "exact discrete values and dtype",
                            "elements": int(a.size),
                            "outside_tolerance": count,
                        }
                    )
                rows[key] = row
        return rows

    @staticmethod
    def results(
        reference: Path,
        current: Path,
        tolerance: float = DEFAULT_RTOL,
        *,
        atol: float = DEFAULT_ATOL,
        reference_arrays: Path | None = None,
    ) -> dict:
        """Read existing result directories; return a report without writing inputs."""
        Compare.tolerances(tolerance, atol)
        report = {
            "comparison_policy": POLICY,
            "criterion": CRITERION,
            "relative_tolerance": tolerance,
            "absolute_tolerance": atol,
            "reference_results": str(reference.resolve()),
            "current_results": str(current.resolve()),
            "parity_scope": "Required JSON and required arrays"
            if reference_arrays
            else "Required JSON only; arrays not assessed",
            "json": {},
            "arrays": {},
            "array_scope": "Required arrays; unsafe objects remain unverified"
            if reference_arrays
            else "No reference arrays supplied",
        }
        for name in REQUIRED_JSON:
            a, b = reference / name, current / name
            report["json"][name] = (
                Compare.json(a, b, tolerance, atol=atol)
                if a.is_file() and b.is_file()
                else {
                    "status": "MISSING",
                    "reference_present": a.is_file(),
                    "current_present": b.is_file(),
                }
            )
        if reference_arrays:
            report["excluded_supporting_reference_arrays"] = sorted(
                path.name
                for path in reference_arrays.glob("*.npz")
                if path.name not in REQUIRED_ARRAYS
            )
            for name in REQUIRED_ARRAYS:
                a, b = reference_arrays / name, current / name
                report["arrays"][name] = (
                    Compare.arrays(a, b, tolerance, atol=atol)
                    if a.is_file() and b.is_file()
                    else {
                        "__file__": {
                            "status": "MISSING_REFERENCE"
                            if not a.is_file()
                            else "MISSING_CURRENT"
                        }
                    }
                )
        statuses = [row["status"] for row in report["json"].values()]
        statuses += [
            row["status"]
            for fields in report["arrays"].values()
            for row in fields.values()
        ]
        incomplete = any(
            status in ("UNDEFINED", "UNVERIFIED_OBJECT_ARRAY") for status in statuses
        )
        failed = any(
            status not in ("IDENTICAL", "CLOSE", "UNDEFINED", "UNVERIFIED_OBJECT_ARRAY")
            for status in statuses
        )
        report["numerical_parity"] = (
            "DIFFERS" if failed else "INCOMPLETE" if incomplete else "PASS"
        )
        return report


def main() -> None:
    """Compare a run or explicit existing directories, writing only a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Existing paper run directory (historical interface).",
    )
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--current-results", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="Report destination; required with explicit result directories.",
    )
    parser.add_argument("--reference-arrays", type=Path)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    args = parser.parse_args()
    try:
        Compare.tolerances(args.rtol, args.atol)
    except ValueError as error:
        parser.error(str(error))
    if args.reference_results or args.current_results:
        if not (args.reference_results and args.current_results and args.report):
            parser.error(
                "Explicit directories require --reference-results, --current-results and --report."
            )
        reference, current = (
            args.reference_results.resolve(),
            args.current_results.resolve(),
        )
        destination = args.report.resolve()
    else:
        if args.output is None:
            parser.error("Supply --output or explicit result directories and --report.")
        work = args.output.resolve() / "source/paper_synth"
        reference, current = work / "results_ref", work / "results"
        destination = (
            args.report.resolve()
            if args.report
            else args.output.resolve() / "comparison.json"
        )
    # A report must never replace a scientific input, including an NPZ reference.
    roots = (reference, current) + (
        (args.reference_arrays.resolve(),) if args.reference_arrays else ()
    )
    if any(destination.is_relative_to(root) for root in roots):
        parser.error(
            "Report destination must be outside scientific result/reference directories."
        )
    report = Compare.results(
        reference,
        current,
        args.rtol,
        atol=args.atol,
        reference_arrays=args.reference_arrays,
    )
    Paper.write(destination, report)
    for name, row in report["json"].items():
        print(f"{name}: {row['status']}")
    print(f"Numerical parity: {report['numerical_parity']}")
    raise SystemExit(0 if report["numerical_parity"] == "PASS" else 1)


if __name__ == "__main__":
    main()
