"""Behavioral regressions for strict scientific result comparison."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.fishy.paper_compare import (
    REQUIRED_ARRAYS,
    REQUIRED_JSON,
    Compare,
    main,
)


def record_unpickle(path: str) -> None:
    """This controlled sentinel must never execute when NPZ data is inspected."""
    Path(path).write_text("unsafe pickle executed")


class UnpickleSentinel:
    def __init__(self, path: Path):
        self.path = path

    def __reduce__(self):
        return record_unpickle, (str(self.path),)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def json_pair(self, left, right, **kwargs):
        a, b = self.root / "reference.json", self.root / "current.json"
        a.write_text(json.dumps(left))
        b.write_text(json.dumps(right))
        return Compare.json(a, b, kwargs.pop("rtol", 1e-6), **kwargs)

    def array_pair(self, left, right, **kwargs):
        a, b = self.root / "reference.npz", self.root / "current.npz"
        np.savez(a, **left)
        np.savez(b, **right)
        return Compare.arrays(a, b, kwargs.pop("rtol", 1e-6), **kwargs)

    def test_near_zero_json_uses_absolute_guard(self):
        row = self.json_pair({"score": 0.0}, {"score": 1e-10})
        self.assertEqual(row["status"], "CLOSE")
        self.assertEqual(row["outside_tolerance"], 0)

    def test_zero_absolute_tolerance_retains_stricter_near_zero_rule(self):
        self.assertEqual(self.json_pair(0.0, 1e-10, atol=0)["status"], "DIFFERS")

    def test_large_array_element_cannot_hide_small_element_error(self):
        row = self.array_pair(
            {"values": np.array([1e9, 0.0])}, {"values": np.array([1e9, 1.0])}
        )["values"]
        self.assertEqual(row["status"], "DIFFERS")
        self.assertEqual(row["outside_tolerance"], 1)
        self.assertEqual(row["worst_tolerance_index"], [1])
        self.assertGreater(row["max_error_to_tolerance_ratio"], 1)

    def test_array_tolerance_is_symmetric(self):
        for left, right in ((1.0, 2.0), (2.0, 1.0)):
            with self.subTest(left=left):
                row = self.array_pair(
                    {"x": np.array([left])}, {"x": np.array([right])}, rtol=0.6, atol=0
                )
                self.assertEqual(row["x"]["status"], "CLOSE")

    def test_json_tolerance_is_symmetric(self):
        for left, right in ((1.0, 2.0), (2.0, 1.0)):
            self.assertEqual(self.json_pair(left, right, rtol=0.6)["status"], "CLOSE")

    def test_empty_containers_and_missing_nodes_are_exact(self):
        for left, right in (
            ([], {}),
            ({"x": []}, {"x": {}}),
            ({"x": []}, {}),
            ([], [[]]),
        ):
            with self.subTest(left=left, right=right):
                self.assertEqual(self.json_pair(left, right)["status"], "DIFFERS")

    def test_json_discrete_types_and_values_are_exact(self):
        for left, right in ((True, 1), (1, 1.0), (1000000, 1000001), (None, "None")):
            with self.subTest(left=left, right=right):
                self.assertEqual(self.json_pair(left, right)["status"], "DIFFERS")

    def test_integer_arrays_are_exact(self):
        for left, right in (([10**9, 2], [10**9, 3]), ([2**63 - 2], [2**63 - 1])):
            self.assertEqual(
                self.array_pair(
                    {"ids": np.array(left, dtype=np.int64)},
                    {"ids": np.array(right, dtype=np.int64)},
                )["ids"]["status"],
                "DIFFERS",
            )

    def test_unsigned_and_signed_values_cannot_alias(self):
        row = self.array_pair(
            {"ids": np.array([2**64 - 1], dtype=np.uint64)},
            {"ids": np.array([-1], dtype=np.int64)},
        )["ids"]
        self.assertEqual(row["status"], "DIFFERS")
        self.assertEqual(row["outside_tolerance"], 1)

    def test_boolean_arrays_cannot_equal_integer_arrays(self):
        row = self.array_pair({"x": np.array([True])}, {"x": np.array([1])})["x"]
        self.assertEqual(row["status"], "DIFFERS")

    def test_array_dtype_is_part_of_strict_structure(self):
        row = self.array_pair(
            {"x": np.array([1.0], dtype=np.float32)},
            {"x": np.array([1.0], dtype=np.float64)},
        )["x"]
        self.assertFalse(row["dtype_matches"])
        self.assertEqual(row["status"], "DIFFERS")

    def test_matching_nonfinite_json_is_explicitly_undefined(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            row = self.json_pair(value, value)
            self.assertEqual(row["status"], "UNDEFINED")
            self.assertEqual(row["undefined_values"], 1)
            json.dumps(row, allow_nan=False)

    def test_nonfinite_json_replacements_differ(self):
        for left, right in ((float("nan"), 0.0), (float("inf"), -float("inf"))):
            self.assertEqual(self.json_pair(left, right)["status"], "DIFFERS")

    def test_matching_nonfinite_arrays_are_not_complete_parity(self):
        row = self.array_pair(
            {"x": np.array([np.nan, np.inf, -np.inf, 1.0])},
            {"x": np.array([np.nan, np.inf, -np.inf, 1.0])},
        )["x"]
        self.assertEqual(row["status"], "UNDEFINED")
        self.assertEqual(row["undefined_elements"], 3)
        self.assertEqual(row["finite_pairs"], 1)
        json.dumps(row, allow_nan=False)

    def test_nonfinite_array_patterns_must_match(self):
        row = self.array_pair(
            {"x": np.array([np.nan, np.inf])}, {"x": np.array([0.0, -np.inf])}
        )["x"]
        self.assertEqual(row["status"], "DIFFERS")
        self.assertEqual(row["nonfinite_pattern_mismatches"], 2)

    def test_complex_values_compare_imaginary_components(self):
        row = self.array_pair({"x": np.array([1 + 2j])}, {"x": np.array([1 + 3j])})["x"]
        self.assertEqual(row["status"], "DIFFERS")
        self.assertAlmostEqual(row["max_absolute_difference"], 1)
        json.dumps(row, allow_nan=False)

    def test_undefined_complex_component_does_not_hide_finite_change(self):
        row = self.array_pair(
            {"x": np.array([complex(np.nan, 1)])}, {"x": np.array([complex(np.nan, 2)])}
        )["x"]
        self.assertEqual(row["status"], "DIFFERS")
        self.assertEqual(row["outside_tolerance"], 1)

    def test_extreme_finite_values_do_not_overflow_into_false_match(self):
        row = self.array_pair({"x": np.array([1e308])}, {"x": np.array([-1e308])})["x"]
        self.assertEqual(row["status"], "DIFFERS")
        self.assertAlmostEqual(row["max_relative_difference"], 2)
        json.dumps(row, allow_nan=False)

    def test_empty_numeric_arrays_are_identical(self):
        self.assertEqual(
            self.array_pair({"x": np.array([])}, {"x": np.array([])})["x"]["status"],
            "IDENTICAL",
        )

    def test_missing_arrays_and_shapes_are_explicit(self):
        row = self.array_pair(
            {"x": np.zeros((1, 2)), "left": np.array([1])},
            {"x": np.zeros((2, 1)), "right": np.array([1])},
        )
        self.assertEqual(row["x"]["status"], "SHAPE_DIFFERS")
        self.assertEqual(row["left"]["status"], "MISSING")
        self.assertEqual(row["right"]["status"], "MISSING")

    def test_string_arrays_preserve_order(self):
        self.assertEqual(
            self.array_pair(
                {"ids": np.array(["a", "b"])}, {"ids": np.array(["b", "a"])}
            )["ids"]["status"],
            "DIFFERS",
        )

    def test_object_arrays_are_never_unpickled(self):
        marker = self.root / "executed"
        row = self.array_pair(
            {"x": np.array([UnpickleSentinel(marker)], dtype=object)},
            {"x": np.array([UnpickleSentinel(marker)], dtype=object)},
        )["x"]
        self.assertEqual(row["status"], "UNVERIFIED_OBJECT_ARRAY")
        self.assertFalse(marker.exists())

    def test_keys_with_punctuation_cannot_collide_with_nested_paths(self):
        row = self.json_pair({"a.b": 1, "a": {"b": 2}}, {"a.b": 2, "a": {"b": 2}})
        self.assertEqual(row["status"], "DIFFERS")
        self.assertEqual(row["outside_tolerance"], 1)

    def test_only_documented_elapsed_fields_are_ignored(self):
        self.assertEqual(
            self.json_pair({"seconds": 1.0, "x": []}, {"seconds": 2.0, "x": []})[
                "status"
            ],
            "IDENTICAL",
        )
        self.assertEqual(
            self.json_pair({"score": 0.0}, {"score": 1.0})["status"], "DIFFERS"
        )

    def test_invalid_tolerances_fail_before_file_reads(self):
        for invalid in (-1, float("nan"), float("inf"), True, "1e-6"):
            for function in (Compare.json, Compare.arrays):
                with self.subTest(invalid=invalid, function=function):
                    with self.assertRaises(ValueError):
                        function(self.root / "missing", self.root / "missing", invalid)
                    with self.assertRaises(ValueError):
                        function(
                            self.root / "missing",
                            self.root / "missing",
                            1e-6,
                            atol=invalid,
                        )

    def test_forced_chunks_match_direct_complex_metrics(self):
        left = np.asfortranarray(np.arange(24).reshape(4, 6).astype(complex))
        right = left.copy(order="F")
        right[0, 1] += 1j
        right[2, 3] += 2
        left[1, 2] = complex(np.nan, 1)
        right[1, 2] = complex(np.nan, 2)
        left[3, 4] = right[3, 4] = complex(np.inf, 0)
        direct = Compare.floating(left, right, 1e-6, 1e-8)
        expected_worst = int(np.argmax(np.where(direct.finite, direct.ratio, -1)))
        for size in (1, 3, 7, 100):
            with self.subTest(chunk_size=size):
                summary = Compare.array_summary(
                    left, right, 1e-6, 1e-8, chunk_size=size
                )
                self.assertEqual(summary.outside, int(direct.outside.sum()))
                self.assertEqual(summary.finite_pairs, int(direct.finite.sum()))
                self.assertEqual(summary.undefined, int(direct.undefined.sum()))
                self.assertEqual(
                    summary.pattern_mismatches, int((~direct.same_nonfinite).sum())
                )
                self.assertEqual(summary.worst, expected_worst)
                self.assertEqual(
                    summary.absolute, Compare.maximum(direct.absolute, direct.finite)
                )
                self.assertEqual(
                    summary.relative, Compare.maximum(direct.relative, direct.finite)
                )
                self.assertEqual(
                    summary.ratio, Compare.maximum(direct.ratio, direct.finite)
                )
                self.assertEqual(
                    {item.index for item in summary.examples},
                    set(np.flatnonzero(direct.outside)),
                )

    def test_large_fixture_bounds_floating_workspace_and_examples(self):
        size = 500_003
        original = Compare.floating
        with patch.object(Compare, "floating", wraps=original) as floating:
            row = self.array_pair({"x": np.ones(size)}, {"x": np.full(size, 2.0)})["x"]
        self.assertEqual(row["status"], "DIFFERS")
        self.assertEqual(row["outside_tolerance"], size)
        self.assertEqual(row["worst_tolerance_index"], [0])
        self.assertEqual(len(row["outside_examples"]), 10)
        self.assertEqual(floating.call_count, 3)
        self.assertLessEqual(
            max(call.args[0].size for call in floating.call_args_list), 250_000
        )

    def test_undefined_and_unsafe_arrays_prevent_complete_parity(self):
        reference, current = self.root / "left", self.root / "right"
        reference.mkdir()
        current.mkdir()
        for name in REQUIRED_JSON:
            (reference / name).write_text('{"score": 1.0}')
            (current / name).write_text('{"score": 1.0}')
        for name in REQUIRED_ARRAYS:
            np.savez(reference / name, ids=np.array(["a"], dtype=object))
            np.savez(current / name, ids=np.array(["a"], dtype=object))
        report = Compare.results(reference, current, reference_arrays=reference)
        self.assertEqual(report["numerical_parity"], "INCOMPLETE")
        (reference / REQUIRED_JSON[0]).write_text('{"score": NaN}')
        (current / REQUIRED_JSON[0]).write_text('{"score": NaN}')
        self.assertEqual(
            Compare.results(reference, current)["numerical_parity"], "INCOMPLETE"
        )

    def test_explicit_directory_cli_only_writes_external_report(self):
        reference, current = self.root / "left", self.root / "right"
        reference.mkdir()
        current.mkdir()
        for name in REQUIRED_JSON:
            (reference / name).write_text('{"score": 0.0}')
            (current / name).write_text('{"score": 1e-10}')
        before = {
            str(path): path.read_bytes()
            for root in (reference, current)
            for path in root.iterdir()
        }
        destination = self.root / "report.json"
        arguments = [
            "paper_compare",
            "--reference-results",
            str(reference),
            "--current-results",
            str(current),
            "--report",
            str(destination),
            "--rtol",
            "1e-6",
            "--atol",
            "1e-8",
        ]
        with (
            patch("sys.argv", arguments),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit) as exit_value,
        ):
            main()
        self.assertEqual(exit_value.exception.code, 0)
        self.assertEqual(
            json.loads(destination.read_text())["comparison_policy"],
            "symmetric-elementwise-v2",
        )
        self.assertEqual(
            before,
            {
                str(path): path.read_bytes()
                for root in (reference, current)
                for path in root.iterdir()
            },
        )
        arguments[arguments.index(str(destination))] = str(current / "report.json")
        with (
            patch("sys.argv", arguments),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as exit_value,
        ):
            main()
        self.assertEqual(exit_value.exception.code, 2)
        self.assertFalse((current / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
