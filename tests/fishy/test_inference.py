"""Behavioral checks for the authored donor-held-out E2 supplement."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.fishy.inference import Contrast, Inference

PAIR = (Contrast("post_axial", "pre_spatial_canon", "stripe"),)


def fixture() -> dict[str, np.ndarray]:
    """Three balanced donors with one signal and one specimen index column."""
    labels = np.tile([0, 1, 0, 1, 0, 1], 3)
    specimen = np.arange(len(labels), dtype=float)
    return {
        "names": np.array([f"fish-{i}" for i in range(len(labels))]),
        "donor": np.repeat([0, 1, 2], 6),
        "label_stripe": labels,
        "pre_spatial_canon": np.column_stack((np.zeros(len(labels)), specimen)),
        "post_axial": np.column_stack((2.0 * labels - 1.0, specimen)),
    }


class InferenceTests(unittest.TestCase):
    def test_matched_all_donors_and_macro_interpretation(self) -> None:
        result = Inference.analyze(fixture(), PAIR)
        contrast = result.contrasts[0]
        self.assertEqual(result.donor_ids, ("0", "1", "2"))
        self.assertEqual(contrast.n_scored_donors, 3)
        self.assertEqual(
            [row.donor_id for row in contrast.donor_scores], ["0", "1", "2"]
        )
        self.assertTrue(all(row.status == "scored" for row in contrast.donor_scores))
        self.assertTrue(
            all(row.n_test == 6 and row.n_train == 12 for row in contrast.donor_scores)
        )
        self.assertTrue(
            all(
                row.train_label_counts == {"0": 6, "1": 6}
                for row in contrast.donor_scores
            )
        )
        self.assertTrue(
            all(
                row.test_label_counts == {"0": 3, "1": 3}
                for row in contrast.donor_scores
            )
        )
        self.assertAlmostEqual(
            contrast.macro_post_minus_pre,
            sum(row.post_minus_pre for row in contrast.donor_scores) / 3,
        )
        self.assertNotIn("p_value", result._fields)
        self.assertNotIn("confidence_interval", result._fields)

    def test_scaler_sees_training_rows_only_for_both_paired_models(self) -> None:
        seen = []

        class SpyScaler(StandardScaler):
            def fit(self, X, y=None, sample_weight=None):
                seen.append(tuple(int(value) for value in X[:, 1]))
                return super().fit(X, y=y, sample_weight=sample_weight)

        with patch("experiments.fishy.inference.StandardScaler", SpyScaler):
            result = Inference.analyze(fixture(), PAIR)
        self.assertEqual(len(seen), 6)
        for fold, row in enumerate(result.contrasts[0].donor_scores):
            expected = tuple(i for i in range(18) if i // 6 != int(row.donor_id))
            self.assertEqual(seen[2 * fold], expected)
            self.assertEqual(seen[2 * fold + 1], expected)
            self.assertNotIn(row.donor_id, row.train_donors)

    def test_undefined_class_support_is_explicit(self) -> None:
        arrays = fixture()
        arrays["label_stripe"] = np.array([0] * 6 + [0] * 6 + [0, 1, 0, 1, 0, 1])
        result = Inference.analyze(arrays, PAIR).contrasts[0]
        self.assertEqual(result.n_scored_donors, 0)
        self.assertEqual(
            [row.status for row in result.donor_scores],
            ["undefined_test_class", "undefined_test_class", "undefined_train_class"],
        )
        self.assertIsNone(result.macro_post_minus_pre)
        self.assertTrue(all(row.post_minus_pre is None for row in result.donor_scores))

    def test_small_group_and_nonfinite_input_rejected(self) -> None:
        arrays = fixture()
        arrays["donor"] = np.zeros(18, dtype=int)
        with self.assertRaisesRegex(ValueError, "donor groups"):
            Inference.analyze(arrays, PAIR)
        arrays = fixture()
        arrays["post_axial"][17, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            Inference.analyze(arrays, PAIR)

    def test_empty_text_identifiers_rejected_and_bytes_decoded(self) -> None:
        arrays = fixture()
        arrays["donor"] = np.repeat(["", "a", "b"], 6)
        with self.assertRaisesRegex(ValueError, "identified donor"):
            Inference.analyze(arrays, PAIR)
        arrays["donor"] = np.repeat([" ", "a", "b"], 6)
        with self.assertRaisesRegex(ValueError, "identified donor"):
            Inference.analyze(arrays, PAIR)
        arrays = fixture()
        arrays["names"] = np.array(
            [b"" if i == 0 else f"fish-{i}".encode() for i in range(18)]
        )
        with self.assertRaisesRegex(ValueError, "nonempty specimen"):
            Inference.analyze(arrays, PAIR)
        arrays["names"] = np.array([f"fish-{i}".encode() for i in range(18)])
        arrays["donor"] = np.repeat([b"a", b"b", b"c"], 6)
        self.assertEqual(Inference.analyze(arrays, PAIR).donor_ids, ("a", "b", "c"))

    def test_bad_saved_input_creates_no_output(self) -> None:
        arrays = fixture()
        # A file-backed run uses all predeclared pairs; supply every required
        # descriptor/factor with the same small fixture, then corrupt one row.
        for name in ("pre_hist", "post_hist_matched", "post_spectral"):
            arrays[name] = arrays["pre_spatial_canon"].copy()
        for factor in ("belly", "tail", "cheeks"):
            arrays[f"label_{factor}"] = arrays["label_stripe"].copy()
        arrays["post_spectral"][0, 0] = float("inf")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.npz"
            output = root / "new" / "result.json"
            np.savez_compressed(source, **arrays)
            with self.assertRaisesRegex(ValueError, "finite numeric"):
                Inference.run(source, output)
            self.assertFalse(output.parent.exists())

    def test_same_saved_input_produces_identical_json(self) -> None:
        arrays = fixture()
        for name in ("pre_hist", "post_hist_matched", "post_spectral"):
            arrays[name] = arrays["pre_spatial_canon"].copy()
        for factor in ("belly", "tail", "cheeks"):
            arrays[f"label_{factor}"] = arrays["label_stripe"].copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.npz"
            first, second = root / "first.json", root / "second.json"
            np.savez_compressed(source, **arrays)
            Inference.run(source, first)
            Inference.run(source, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
