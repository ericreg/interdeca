"""Check the integrity of a bounded, published research-evidence directory.

This verifies recorded bytes and reconciles comparison counts. It does not
establish that a run occurred or substitute for rerunning the scientific code.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


class Evidence:
    """Operations on the public evidence manifest and its recorded comparisons."""

    @staticmethod
    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    @staticmethod
    def verify(directory: Path) -> dict:
        """Require every listed artifact and reject edits or unsafe paths."""
        directory = directory.resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
        ):
            raise ValueError("Unsupported evidence manifest schema")
        seen = set()
        for record in manifest["files"]:
            relative = record["path"]
            path = directory / relative
            # The index names only regular files inside this evidence directory.
            if (
                not isinstance(relative, str)
                or not relative
                or "\\" in relative
                or ":" in relative
                or Path(relative).is_absolute()
                or any(part in (".", "..", "") for part in relative.split("/"))
                or relative in seen
                or path.is_symlink()
                or not path.resolve().is_relative_to(directory)
            ):
                raise ValueError("Unsafe or duplicate evidence path")
            seen.add(relative)
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError(f"Missing or resized evidence: {relative}")
            if Evidence.digest(path) != record["sha256"]:
                raise ValueError(f"Changed evidence: {relative}")
        actual = {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and path != directory / "manifest.json"
        }
        if actual != seen:
            raise ValueError("Unlisted or missing evidence files")
        for check in manifest.get("comparison_checks", []):
            if check["path"] not in seen:
                raise ValueError("Comparison check refers to unlisted evidence")
            report = json.loads((directory / check["path"]).read_text())
            if Evidence.comparison_counts(report) != check["expected"]:
                raise ValueError(f"Inconsistent comparison summary: {check['path']}")
        return {
            "integrity": "PASS",
            "files_verified": len(seen),
            "comparison_summaries_verified": len(manifest.get("comparison_checks", [])),
        }

    @staticmethod
    def comparison_counts(report: dict) -> dict:
        """Recompute summary counts from the complete persisted differences."""
        outside = 0
        for row in report["json"].values():
            # The detail is the evidence for this count; a self-consistent but
            # incorrect summary in two files must not certify itself.
            differences = row.get("differences", [])
            if any(
                type(item.get("within_tolerance")) is not bool for item in differences
            ):
                raise ValueError("Comparison detail lacks boolean tolerance decisions")
            count = sum(not item["within_tolerance"] for item in differences)
            if count != row.get("outside_tolerance", 0):
                raise ValueError(
                    "Comparison count contradicts its persisted differences"
                )
            outside += count
        return {
            "json_leaves": sum(row.get("leaves", 0) for row in report["json"].values()),
            "json_leaves_changed": sum(
                len(row.get("differences", [])) for row in report["json"].values()
            ),
            "json_leaves_outside_tolerance": outside,
            "array_fields": sum(len(rows) for rows in report["arrays"].values()),
            "array_statuses": dict(
                Counter(
                    row["status"]
                    for rows in report["arrays"].values()
                    for row in rows.values()
                )
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = Evidence.verify(args.directory)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Evidence verification failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
