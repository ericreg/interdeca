"""Summarize complete fresh-process reports without hiding individual failures."""

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from serde import MeshReport, QualificationSnapshot, RepeatedResult, Report, UVComparison, WorkflowComparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("build/native-surface/repeated"))
    parser.add_argument("--layouts", type=Path, default=Path("build/native-surface/uv-conformal"))
    parser.add_argument("--pilots", type=Path, default=Path("build/native-surface"))
    parser.add_argument("--output", type=Path, default=Path("docs/native-surface/results.json"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--source-port", action="store_true", help="Summarize the source-derived implementation; preserve historical prototype results.")
    parser.add_argument("--uv-root", type=Path, default=Path("build/native-surface"))
    parser.add_argument("--workflow-comparisons", type=Path, nargs="+", default=[],
                        help="Retained complete-workflow comparisons for both atlas backends.")
    args = parser.parse_args()
    results = []
    for backend in ("native", "morphoweave"):
        for resolution in (512, 2048):
            reports = [Report.model_validate_json((args.runs / f"{backend}-{resolution}-{i}" / "report.json").read_text())
                       for i in range(1, args.repetitions+1)]
            for report in reports:
                if len(report.measurements) != 47 or len({m.specimen for m in report.measurements}) != 47:
                    raise ValueError("Cannot qualify an incomplete population run.")
                if report.scope != "baker_on_reference_uv":
                    raise ValueError("Do not combine distinct qualification scopes.")
            fingerprints = [{m.specimen: (m.input_sha256, m.reference_sha256, m.output_sha256)
                             for m in r.measurements} for r in reports]
            # Both samplers are release gates, so report worst cases over both.
            primary = [a for r in reports for m in r.measurements for a in (m.agreement,m.analysis_agreement) if a is not None]
            deterministic = all(f == fingerprints[0] for f in fingerprints)
            row = RepeatedResult(backend=backend, resolution=resolution, repetitions=len(reports),
                specimens_per_repetition=47,
                primary_passes=[sum(m.agreement.passed for m in r.measurements) for r in reports],
                analysis_passes=[sum(m.analysis_agreement.passed for m in r.measurements) for r in reports],
                combined_passes=[sum(m.agreement.passed and m.analysis_agreement.passed for m in r.measurements) for r in reports],
                deterministic_pixels=deterministic,
                native_kernel_total_seconds=[sum(m.rust_seconds for m in r.measurements) for r in reports],
                whole_process_seconds=[r.elapsed_seconds for r in reports],
                peak_rss_bytes=[r.peak_rss_bytes for r in reports],
                maximum_mean_rgb=max(a.mean_rgb for a in primary if a.mean_rgb is not None),
                maximum_p95_channel=max(a.p95_max_channel for a in primary if a.p95_max_channel is not None),
                maximum_coverage_loss_pp=max(a.coverage_loss_pp for a in primary),
                passed=all(r.all_passed for r in reports) and deterministic,
                implementation_sha256=sorted({r.implementation_sha256 for r in reports}),
                extension_sha256=sorted({r.extension_sha256 for r in reports}))
            results.append(row)
    layouts = [MeshReport.model_validate_json(p.read_text()) for p in sorted(args.layouts.glob("*-512.json"))]
    pilots = []
    for backend in (() if args.source_port else ("native", "morphoweave")):
        report = Report.model_validate_json((args.pilots / f"final-custom-{backend}" / "report.json").read_text())
        if report.scope != "custom_uv_and_baker":
            raise ValueError("Custom-layout pilots must use a custom layout.")
        pilots.extend(report.measurements)
    blockers = [
        "512 baker color/coverage gates do not pass for every specimen.",
        "Custom UV layouts fail surface-color agreement; baker-only results reuse Blender UVs.",
        "Cleanup representative selection differs from Blender; pre-cleanup workflows remain unqualified.",
        "Downstream features/PCA, complete workflow runtime, and relative peak-memory gates remain unqualified.",
    ]
    uv_comparisons = []
    if args.source_port:
        # UV equality ties the baker's archived reference layout to the port's
        # output. This does not substitute for pre-cleanup workflow validation.
        for backend in ("native","morphoweave"):
            path=args.uv_root/f"source-workflow-uv-{backend}"/"report.json"
            uv_comparisons.extend(TypeAdapter(list[UVComparison]).validate_json(path.read_text()))
        blockers = []
        if not all(r.passed for r in results):
            blockers.append("Population color/coverage or determinism gate failed.")
        if any(r.maximum_uv_error != 0 for r in uv_comparisons):
            blockers.append("Source UV output differs from the reference layout.")
        if not args.workflow_comparisons:
            blockers.append("Full workflow numerical/runtime/memory qualification not supplied.")
    workflows = [WorkflowComparison.model_validate_json(path.read_text())
                 for path in args.workflow_comparisons]
    if workflows:
        # Only paired successful comparisons qualify a complete workflow. Raw
        # byte differences remain in their reports even when tolerances pass.
        if {row.backend for row in workflows} != {"native", "morphoweave"}:
            blockers.append("Both atlas backends need full workflow comparisons.")
        if not all(row.cleanup_passed and row.downstream_passed and row.runtime_passed
                   and row.memory_passed for row in workflows):
            blockers.append("A complete workflow comparison failed.")
    snapshot = QualificationSnapshot(results=results, layouts=layouts, custom_layout_pilots=pilots,
        blockers=blockers, uv_comparisons=uv_comparisons, workflow_comparisons=workflows,
        status=("native production operations; local qualification passed" if workflows and not blockers
                else "experimental; qualification incomplete"),
        scope=("Population baking and complete workflows, with recorded numerical tolerances." if workflows
               else "Baker comparisons use preserved Blender UVs; not whole-workflow qualification."))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(snapshot.model_dump_json(indent=2) + "\n")
    print(snapshot.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
