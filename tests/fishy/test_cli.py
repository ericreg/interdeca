"""Expected invalid inputs should have actionable CLI errors without tracebacks."""

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from experiments.fishy.cli import CLI
from experiments.fishy.serde import Evaluation


class CLITests(unittest.TestCase):
    def test_boundary_error_is_concise_and_keeps_library_exception_contract(
        self,
    ) -> None:
        @CLI.command
        def invalid() -> None:
            Evaluation.model_validate_json('{"folds": "5"}')

        output = StringIO()
        with redirect_stderr(output), self.assertRaises(SystemExit) as failure:
            invalid()
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("folds", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())
        self.assertNotIn("input_value", output.getvalue())

    def test_paper_entrypoint_rejects_bad_run_before_stage_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            (output / "run.json").write_text(json.dumps({"schema_version": 99}))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.fishy.paper",
                    "run",
                    "--output",
                    str(output),
                    "--stage",
                    "build_cache",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("Invalid input", result.stderr)
            self.assertIn("schema_version", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((output / "source").exists())

    def test_programming_failures_are_not_disguised_as_input_errors(self) -> None:
        @CLI.command
        def broken() -> None:
            raise RuntimeError("unexpected internal failure")

        with self.assertRaisesRegex(RuntimeError, "unexpected internal failure"):
            broken()


if __name__ == "__main__":
    unittest.main()
