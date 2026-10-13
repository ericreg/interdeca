"""Exercise integrity and count reconciliation through the public verifier."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.verify_research_evidence import Evidence


class EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report = {
            "json": {
                "result.json": {
                    "leaves": 3,
                    "outside_tolerance": 1,
                    "differences": [{"path": ".x", "within_tolerance": False}],
                }
            },
            "arrays": {
                "result.npz": {"x": {"status": "DIFFERS"}, "y": {"status": "IDENTICAL"}}
            },
        }
        self.file = self.root / "report.json"
        self.file.write_text(json.dumps(self.report))
        self.manifest = {
            "schema_version": 1,
            "files": [
                {
                    "path": "report.json",
                    "bytes": self.file.stat().st_size,
                    "sha256": Evidence.digest(self.file),
                }
            ],
            "comparison_checks": [
                {
                    "path": "report.json",
                    "expected": Evidence.comparison_counts(self.report),
                }
            ],
        }
        self.write_manifest()

    def write_manifest(self) -> None:
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_complete_evidence_and_counts_verify(self) -> None:
        result = Evidence.verify(self.root)
        self.assertEqual(result["files_verified"], 1)
        self.assertEqual(result["comparison_summaries_verified"], 1)
        self.assertEqual(
            Evidence.comparison_counts(self.report)["json_leaves_outside_tolerance"], 1
        )

    def test_same_size_change_is_rejected(self) -> None:
        self.file.write_text(self.file.read_text().replace("DIFFERS", "CHANGED"))
        with self.assertRaisesRegex(ValueError, "Changed evidence"):
            Evidence.verify(self.root)

    def test_missing_and_unlisted_files_are_rejected(self) -> None:
        (self.root / "extra.txt").write_text("unlisted")
        with self.assertRaisesRegex(ValueError, "Unlisted"):
            Evidence.verify(self.root)
        (self.root / "extra.txt").unlink()
        self.file.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            Evidence.verify(self.root)

    def test_wrong_summary_cannot_pass_with_valid_file_hash(self) -> None:
        self.manifest["comparison_checks"][0]["expected"][
            "json_leaves_outside_tolerance"
        ] = 0
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Inconsistent comparison summary"):
            Evidence.verify(self.root)

    def test_report_summary_cannot_contradict_recorded_differences(self) -> None:
        self.report["json"]["result.json"]["outside_tolerance"] = 0
        self.file.write_text(json.dumps(self.report))
        self.manifest["files"][0].update(
            bytes=self.file.stat().st_size, sha256=Evidence.digest(self.file)
        )
        self.manifest["comparison_checks"][0]["expected"][
            "json_leaves_outside_tolerance"
        ] = 0
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "contradicts"):
            Evidence.verify(self.root)

    def test_paths_cannot_escape_or_repeat(self) -> None:
        self.manifest["files"][0]["path"] = "../report.json"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            Evidence.verify(self.root)
        self.manifest["files"][0]["path"] = "report.json"
        self.manifest["files"].append(self.manifest["files"][0].copy())
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            Evidence.verify(self.root)


if __name__ == "__main__":
    unittest.main()
