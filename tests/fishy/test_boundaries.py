"""CPU-only behavioral contracts with disposable assets, never research data."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from experiments.fishy.artifacts import Artifacts
from experiments.fishy.generation import Generation
from experiments.fishy.models import Brush, CanonicalFrame
from experiments.fishy.paper import ANALYSES, FIGURES, Paper
from experiments.fishy.paper_compare import REQUIRED_ARRAYS, REQUIRED_JSON
from experiments.fishy.release import Release
from experiments.fishy.release import main as release_main
from experiments.fishy.serde import (
    ArchiveRow,
    BundleConfigData,
    CompositionIdentity,
    Config,
    Documents,
    Evaluation,
    ParameterData,
)
from experiments.fishy.serde import (
    Generation as GenerationSettings,
)

PARAMETERS = {
    "base_color_hue": 0.15,
    "base_color_sat": 0.6,
    "base_color_val": 0.8,
    "stripe_count": 4,
    "stripe_spacing": -0.08,
    "stripe_width": 0.025,
    "stripe_longitudinal_offset": 0.0,
    "belly_hue": 0.6,
    "belly_strength": 0.95,
    "belly_translation": 0.0,
    "tail_hue": 0.3,
    "tail_strength": 0.95,
    "rosy_cheeks_present": 1,
    "rosy_cheeks_hue": 0.01,
    "rosy_cheeks_strength": 0.95,
    "rosy_cheeks_translation_y": 0.0,
    "rosy_cheeks_translation_z": 0.0,
}
POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def config_payload():
    return {
        "schema_version": 1,
        "units": "mm",
        "reference_id": "donor",
        "donors": [
            {
                "donor_id": "donor",
                "mesh": {"path": "mesh.obj"},
                "landmarks": {"path": "landmarks.json"},
            }
        ],
    }


def stage_fixture(directory):
    """Represent the documented legacy wire layout with tiny synthetic assets."""
    output = directory / "run"
    work = output / "source/paper_synth"
    (work / "cache").mkdir(parents=True)
    (output / "logs").mkdir()
    asset = directory / "asset"
    asset.write_bytes(b"original bytes")
    source = work / "metadata.txt"
    source.write_bytes(b"pinned fixture source")
    (output / "source/upstream.json").write_text(
        json.dumps(
            {
                "files": [
                    {"path": "paper_synth/metadata.txt", "sha256": Paper.digest(source)}
                ]
            }
        )
    )
    (work / "requirements-paper.txt").write_text("")
    record = {
        "schema_version": 1,
        "upstream_revision": "a" * 40,
        "source": str(directory / "native"),
        "module_run": str(directory / "module"),
        "donor_scans": None,
        "python": sys.version,
        "platform": "test-fixture",
        "packages": {},
        "inputs": [
            {
                "path": str(asset),
                "bytes": asset.stat().st_size,
                "sha256": Paper.digest(asset),
                "mtime_ns": asset.stat().st_mtime_ns,
            }
        ],
        "stages": {name: {"exit_code": 0} for name in ANALYSES + FIGURES},
        "scope": "Synthetic coordinator fixture; no scientific result claimed.",
    }
    (output / "run.json").write_text(json.dumps(record))
    (output / "completed.json").write_text(
        json.dumps({"upstream_revision": "a" * 40, "scope": record["scope"]})
    )
    for name in (
        "e6_negative_results_partial.json",
        "e6_negative_results_partial_arrays.npz",
    ):
        (work / "cache" / name).write_bytes(b"same-input resumable checkpoint")
    return output, work, asset, record


class BoundaryTests(unittest.TestCase):
    def test_strict_frozen_finite_generation_settings(self):
        for field, value in (
            ("samples", "4"),
            ("seed", True),
            ("noise_scale", "8.0"),
            ("noise_scale", float("inf")),
            ("tps_lambda", float("nan")),
        ):
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(ValidationError),
            ):
                GenerationSettings.model_validate_json(json.dumps({field: value}))
        with self.assertRaises(ValidationError):
            GenerationSettings().samples = 0
        with self.assertRaises(ValidationError):
            Evaluation().folds = 1

    def test_json_paths_and_arrays_remain_supported_and_immutable(self):
        config = Config.model_validate_json(json.dumps(config_payload()))
        self.assertEqual(config.donors[0].mesh.path, Path("mesh.obj"))
        self.assertIsInstance(config.donors, tuple)
        with self.assertRaises(ValidationError):
            config.donors[0].mesh.path = Path("changed.obj")
        for version in (True, "1", 1.0, 2):
            with self.subTest(version=version), self.assertRaises(ValidationError):
                Config.model_validate_json(
                    json.dumps(config_payload() | {"schema_version": version})
                )

    def test_frozen_asset_resolution_pins_new_records_without_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("mesh.obj", "landmarks.json"):
                (root / name).write_bytes(name.encode())
            path = root / "config.json"
            path.write_text(json.dumps(config_payload()))
            resolved = Artifacts.config(path)
            self.assertEqual(
                resolved.donors[0].mesh.path, (root / "mesh.obj").resolve()
            )
            self.assertEqual(
                resolved.donors[0].mesh.sha256, Artifacts.digest(root / "mesh.obj")
            )
            self.assertEqual(json.loads(path.read_text()), config_payload())

    def test_csv_legacy_adapter_retains_numeric_values_and_rejects_bad_categories(self):
        values = PARAMETERS | {
            "sample_tag": "donor_fishy_000000",
            "sample_index": 0,
            "specimen_index": 0,
            "mesh_file": "historical/mesh.obj",
            "landmark_file": "historical/lm.json",
            "tps_lambda": 1e-6,
            "n_landmarks_used": 47,
            "tps_rmse": 0.0,
        }
        strings = {name: str(value) for name, value in values.items()}
        row = ArchiveRow.from_csv(strings)
        for field in ("stripe_count", "rosy_cheeks_present"):
            with (
                self.subTest(direct_json_field=field),
                self.assertRaises(ValidationError),
            ):
                ArchiveRow.model_validate_json(
                    json.dumps(values | {field: str(values[field])})
                )
            with (
                self.subTest(direct_python_field=field),
                self.assertRaises(ValidationError),
            ):
                ArchiveRow.model_validate(values | {field: str(values[field])})
        self.assertEqual(row.model_dump(), values)
        for field, value in (
            ("stripe_count", "4.0"),
            ("stripe_count", "6"),
            ("stripe_width", "nan"),
            ("tps_rmse", "inf"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                ArchiveRow.from_csv(strings | {field: value})
        with self.assertRaises(ValidationError):
            ParameterData.model_validate(PARAMETERS | {"base_color_hue": "0.15"})
        with self.assertRaises(ValidationError):
            ParameterData.model_validate(PARAMETERS | {"rosy_cheeks_present": True})

    def test_partial_composition_projection_validates_identities(self):
        richer = {
            "sample_ids": ["fish_0", "fish_1"],
            "donor_ids": ["d0", "d0"],
            "backend": "interdeca",
            "factors": {"documented": "ignored projection"},
        }
        identity = CompositionIdentity.model_validate_json(json.dumps(richer))
        self.assertEqual(set(identity.model_dump()), {"sample_ids", "donor_ids"})
        with self.assertRaises(ValidationError):
            identity.sample_ids = ("replacement",)
        for values in (
            {"sample_ids": [""], "donor_ids": ["d"]},
            {"sample_ids": ["fish"], "donor_ids": [0]},
            {"sample_ids": ["fish", "fish"], "donor_ids": ["d", "d"]},
            {"sample_ids": ["fish"], "donor_ids": []},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                CompositionIdentity.model_validate_json(json.dumps(values))

    def test_malformed_manifest_rejected_before_asset_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": True,
                        "kind": "generated",
                        "backend": "generator",
                        "samples": [
                            {
                                "sample_id": "fish",
                                "donor_id": "d",
                                "mesh": {"path": "missing"},
                                "landmarks": {"path": "missing"},
                                "texture": {"path": "missing"},
                            }
                        ],
                    }
                )
            )
            with (
                patch.object(Artifacts, "resolve") as resolve,
                self.assertRaises(ValidationError),
            ):
                Artifacts.manifest(path)
            resolve.assert_not_called()

    def test_invalid_json_write_allocates_neither_parent_nor_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unallocated/invalid.json"
            with self.assertRaises(ValueError):
                Artifacts.write(path, {"nonfinite": float("nan")})
            self.assertFalse(path.parent.exists())
            self.assertFalse(path.exists())

    def test_legacy_optional_presence_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, _, _, original = stage_fixture(Path(temporary))
            self.assertEqual(
                Documents.run_payload(Documents.run(output / "run.json")), original
            )
            for field in (
                "analysis_python",
                "analysis_packages",
                "donor_scans_attached_unix_seconds",
            ):
                for explicit in (False, True):
                    values = dict(original)
                    if explicit:
                        values[field] = None
                    (output / "run.json").write_text(json.dumps(values))
                    self.assertEqual(
                        Documents.run_payload(Documents.run(output / "run.json")),
                        values,
                    )
            completion = Documents.legacy_completion(output / "completed.json")
            self.assertEqual(
                Documents.payload(completion),
                json.loads((output / "completed.json").read_text()),
            )

    def test_bundle_versions_are_explicit(self):
        values = {
            "schema_version": 1,
            "python": "3.12.8",
            "requirements": "requirements.txt",
            "source": "source",
            "module_run": "module",
            "donor_scans": "donors",
            "author_arrays": "author",
        }
        self.assertEqual(
            BundleConfigData.model_validate_json(json.dumps(values)).schema_version, 1
        )
        for version in (True, "1", 1.0, 2, None):
            invalid = dict(values)
            if version is None:
                invalid.pop("schema_version")
            else:
                invalid["schema_version"] = version
            with self.subTest(version=version), self.assertRaises(ValidationError):
                BundleConfigData.model_validate_json(json.dumps(invalid))

    def test_manifest_contract_is_chosen_explicitly_never_sniffed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            legacy = {"bundle_manifest_sha256": "0" * 64, "files": []}
            with self.assertRaises(ValueError):
                Release.write(path, legacy)
            self.assertFalse(path.exists())
            with self.assertRaises(ValidationError):
                Release.write(path, legacy, manifest_kind="bundle")
            self.assertFalse(path.exists())
            Release.write(path, legacy, manifest_kind="legacy-export")
            self.assertEqual(
                Documents.payload(Documents.legacy_export_manifest(path)), legacy
            )
            for version in (None, True, "1", 2):
                payload = {"source_tree_sha256": "0" * 64, "files": []}
                if version is not None:
                    payload["schema_version"] = version
                with self.subTest(version=version), self.assertRaises(ValidationError):
                    Documents.release_payload(path, payload, manifest_kind="bundle")

    def test_bootstrap_imports_do_not_pull_core_or_scientific_packages(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import experiments.fishy.paper, experiments.fishy.release; "
                    "assert not any(name in sys.modules for name in ('numpy','interdeca','sklearn','skimage','vtk'))"
                ),
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )

    def test_release_comparison_marks_unsafe_arrays_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "author"
            current = root / "run/source/paper_synth/results"
            reference.mkdir()
            current.mkdir(parents=True)
            config = {
                "schema_version": 1,
                "python": "3.12.8",
                "requirements": "requirements.txt",
                "source": "source",
                "module_run": "module",
                "donor_scans": "donors",
                "author_arrays": "author",
            }
            (root / "bundle.json").write_text(json.dumps(config))
            for directory in (reference, current):
                for filename in REQUIRED_JSON:
                    (directory / filename).write_text('{"finite_result": 0.5}')
                for index, filename in enumerate(REQUIRED_ARRAYS):
                    np.savez(
                        directory / filename,
                        values=np.array([{"unsafe": True}], dtype=object)
                        if index == 0
                        else np.array([1.0, 2.0]),
                    )
            arguments = [
                "release",
                "--bundle",
                str(root),
                "compare",
                "--output",
                str(root / "run"),
            ]
            with (
                patch.object(Release, "verify"),
                patch.object(Release, "environment"),
                patch.object(sys, "argv", arguments),
                self.assertRaises(SystemExit) as exit_status,
            ):
                release_main()
            self.assertEqual(exit_status.exception.code, 1)
            report = json.loads((root / "run/comparison-author.json").read_text())
            self.assertEqual(report["numerical_parity"], "INCOMPLETE")
            self.assertEqual(report["comparison_policy"], "symmetric-elementwise-v2")


class GenerationTests(unittest.TestCase):
    def test_spline_rejects_nonfinite_and_negative_controls(self):
        for regularization in (float("nan"), float("inf"), -1.0, "0.1", True):
            with self.subTest(value=regularization), self.assertRaises(ValueError):
                Generation.fit(POINTS, POINTS, regularization)
        for target in (POINTS * float("nan"), np.ones((4, 2))):
            with self.assertRaises(ValueError):
                Generation.fit(POINTS, target, 1e-6)

    def test_valid_affine_spline_and_world_roundtrip(self):
        target = POINTS * 2 + np.array([0.1, 0.2, 0.3])
        spline = Generation.fit(POINTS, target, 1e-6)
        np.testing.assert_allclose(
            Generation.evaluate(POINTS, spline), target, atol=1e-12
        )
        frame = CanonicalFrame(
            np.eye(3), np.array([1.0, 2.0, 3.0]), 2.0, np.array([0.1, 0.2, 0.3])
        )
        canonical = (POINTS - frame.pivot) * frame.scale + frame.translation
        np.testing.assert_allclose(
            Generation.world(canonical, frame), POINTS, atol=1e-12
        )
        with self.assertRaises(ValueError):
            Generation.world(POINTS, frame._replace(scale=float("inf")))

    def test_parameters_and_brushes_reject_nonfinite_before_paint(self):
        parameters = ParameterData.model_validate(PARAMETERS).to_internal()
        with self.assertRaises(ValueError):
            Generation.brushes(parameters._replace(belly_translation=float("nan")))
        brush = Brush(np.zeros(3), np.ones(3), 0.1, 0.03, (0.0, 0.0, 0.0), 1.0)
        for invalid in (
            brush._replace(sigma=float("nan")),
            brush._replace(sigma=-1e-12),
            brush._replace(start=np.array([0.0, float("inf"), 0.0])),
        ):
            with self.subTest(brush=invalid), self.assertRaises(ValueError):
                Generation.paint(POINTS, (invalid,), 0.1, 8.0)
        with self.assertRaises(ValueError):
            Generation.paint(POINTS, (brush,), float("nan"), 8.0)

    def test_negative_width_draw_flows_through_brush_warp_and_paint(self):
        with self.assertWarnsRegex(RuntimeWarning, "non-positive"):
            row = Generation.sample_labeled(250, 11725)[156]
        width = row.parameters.stripe_width
        self.assertEqual(width, -0.0008502049605016274)
        with self.assertWarnsRegex(RuntimeWarning, "non-positive"):
            parameters = ParameterData.model_validate(
                row.parameters._asdict()
            ).to_internal()
        with self.assertWarnsRegex(RuntimeWarning, "non-positive"):
            brushes = Generation.brushes(parameters)
        stripes = [brush for brush in brushes if brush.sigma == width]
        self.assertEqual(len(stripes), 2 * parameters.stripe_count)
        spline = Generation.fit(POINTS, POINTS, 1e-6)
        frame = CanonicalFrame(np.eye(3), np.zeros(3), 1.0, np.zeros(3))
        warped = Generation.warp((stripes[0],), spline, frame)[0]
        np.testing.assert_allclose(warped.sigma, width, rtol=1e-5)
        dot = Brush(np.zeros(3), np.zeros(3), 0.0, width, (1.0, 1.0, 1.0), 1.0)
        painted = Generation.paint(np.array([[width, 0.0, 0.0]]), (dot,), 0.0, 8.0)
        expected = np.exp(-0.5 * (abs(width) / (width + 1e-12)) ** 2)
        np.testing.assert_allclose(painted[0], expected, atol=1e-9)

    def test_sampling_keeps_global_rng_isolated_and_python_categories(self):
        np.random.seed(817)
        expected = np.random.random(5)
        np.random.seed(817)
        with self.assertWarnsRegex(RuntimeWarning, "non-positive"):
            width = Generation.sample(250, 11725)[156].stripe_width
        self.assertEqual(width, -0.0008502049605016274)
        first = Generation.sample(250, 49)
        np.testing.assert_array_equal(np.random.random(5), expected)
        self.assertEqual(first, Generation.sample(250, 49))
        self.assertTrue(
            all(
                type(row.stripe_count) is int and type(row.rosy_cheeks_present) is int
                for row in first
            )
        )
        for row in first:
            parameters = ParameterData.model_validate(row._asdict()).to_internal()
            brushes = Generation.brushes(parameters)
            self.assertEqual(
                [brush.sigma for brush in brushes if brush.sigma == row.stripe_width],
                [row.stripe_width] * (2 * row.stripe_count),
            )


class StageResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output, self.work, self.asset, self.original = stage_fixture(
            Path(self.temporary.name)
        )
        self.python = patch("experiments.fishy.paper.sys.version_info", (3, 12, 8))
        self.python.start()
        self.addCleanup(self.python.stop)

    def test_upstream_rerun_invalidates_consumers_completion_and_partial_cache(self):
        with patch(
            "experiments.fishy.paper.subprocess.run",
            return_value=SimpleNamespace(returncode=0),
        ) as child:
            Paper.run(self.output, "e1_fidelity")
        child.assert_called_once()
        record = Documents.run(self.output / "run.json")
        self.assertEqual(record.stages["e1_fidelity"].exit_code, 0)
        self.assertNotIn("e6_negative_results", record.stages)
        self.assertNotIn("fig7_limits", record.stages)
        self.assertFalse((self.output / "completed.json").exists())
        self.assertFalse(
            (self.work / "cache/e6_negative_results_partial.json").exists()
        )
        self.assertFalse(
            (self.work / "cache/e6_negative_results_partial_arrays.npz").exists()
        )

    def test_failed_child_persists_failure_without_stale_success(self):
        with (
            patch(
                "experiments.fishy.paper.subprocess.run",
                return_value=SimpleNamespace(returncode=7),
            ),
            self.assertRaises(RuntimeError),
        ):
            Paper.run(self.output, "e1_fidelity")
        record = Documents.run(self.output / "run.json")
        self.assertEqual(record.stages["e1_fidelity"].exit_code, 7)
        self.assertNotIn("e6_negative_results", record.stages)
        self.assertFalse((self.output / "completed.json").exists())

    def test_interrupted_e6_preserves_same_input_resume_checkpoint(self):
        with (
            patch(
                "experiments.fishy.paper.subprocess.run", side_effect=KeyboardInterrupt
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            Paper.run(self.output, "e6_negative_results")
        record = Documents.run(self.output / "run.json")
        self.assertEqual(record.stages["e6_negative_results"].status, "running")
        self.assertIsNone(record.stages["e6_negative_results"].exit_code)
        self.assertFalse((self.output / "completed.json").exists())
        self.assertTrue((self.work / "cache/e6_negative_results_partial.json").exists())

    def test_same_bytes_timestamp_refresh_allows_rerun(self):
        timestamp = self.asset.stat().st_mtime_ns + 1_000_000_000
        os.utime(self.asset, ns=(self.asset.stat().st_atime_ns, timestamp))
        with patch(
            "experiments.fishy.paper.subprocess.run",
            return_value=SimpleNamespace(returncode=0),
        ):
            Paper.run(self.output, "e1_fidelity")
        self.assertEqual(
            Documents.run(self.output / "run.json").inputs[0].mtime_ns, timestamp
        )

    def test_missing_prerequisite_rejects_before_state_change(self):
        record = dict(self.original)
        record["stages"] = {}
        (self.output / "run.json").write_text(json.dumps(record))
        before = (self.output / "run.json").read_bytes()
        with (
            patch("experiments.fishy.paper.subprocess.run") as child,
            self.assertRaises(ValueError),
        ):
            Paper.run(self.output, "e1_fidelity")
        child.assert_not_called()
        self.assertEqual((self.output / "run.json").read_bytes(), before)

    def test_metadata_or_input_tamper_stops_before_child_and_state_changes(self):
        for kind in ("schema", "input", "source"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                output, work, asset, original = stage_fixture(Path(temporary))
                if kind == "schema":
                    (output / "run.json").write_text(
                        json.dumps(original | {"schema_version": 2})
                    )
                elif kind == "input":
                    asset.write_bytes(b"changed! bytes")
                    os.utime(
                        asset,
                        ns=(
                            asset.stat().st_atime_ns,
                            original["inputs"][0]["mtime_ns"] + 1_000_000_000,
                        ),
                    )
                else:
                    (work / "metadata.txt").write_bytes(b"changed source")
                before = (output / "run.json").read_bytes()
                with (
                    patch("experiments.fishy.paper.subprocess.run") as child,
                    self.assertRaises(ValueError),
                ):
                    Paper.run(output, "e1_fidelity")
                child.assert_not_called()
                self.assertEqual((output / "run.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
