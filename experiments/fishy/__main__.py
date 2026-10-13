"""Explicit command-line stages for the fishy reproduction bundle."""

import argparse
from pathlib import Path

from experiments.fishy.analysis import Analysis
from experiments.fishy.archive import Archive
from experiments.fishy.artifacts import Artifacts
from experiments.fishy.cli import CLI
from experiments.fishy.comparison import Comparison
from experiments.fishy.labeling import Labeling
from experiments.fishy.native_palette import NativePalette
from experiments.fishy.pipeline import Pipeline


@CLI.command
def main() -> None:
    """Run one stage at a time so archived atlas outputs can bypass recomputation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    archived = commands.add_parser(
        "import-inputs", help="Hash and audit an existing synthetic input directory"
    )
    archived.add_argument("--source", type=Path, required=True)
    archived.add_argument("--output", type=Path, required=True)
    archived.add_argument("--seed", type=int, default=49)
    label = commands.add_parser(
        "label-archive", help="Prepare explicitly inferred legacy hue labels"
    )
    label.add_argument("--csv", type=Path, required=True)
    label.add_argument("--output", type=Path, required=True)
    label.add_argument("--protocol", choices=["legacy-kmeans"], required=True)
    validate = commands.add_parser(
        "validate", help="Check donor assets and landmark/UV contracts without baking"
    )
    validate.add_argument("--config", type=Path, required=True)
    generate = commands.add_parser(
        "generate", help="Generate native-space synthetic cichlid textures"
    )
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--blender")
    transfer = commands.add_parser(
        "transfer",
        help="Rerun atlas construction and baking through the current Interdeca core",
    )
    transfer.add_argument("--config", type=Path, required=True)
    transfer.add_argument("--manifest", type=Path, required=True)
    transfer.add_argument("--output", type=Path, required=True)
    transfer.add_argument("--blender")
    extract = commands.add_parser(
        "extract", help="Extract legacy corner colors from a shared atlas manifest"
    )
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument(
        "--archived-uv",
        action="store_true",
        help="Verify legacy VTK expanded float32 UVs; audit specimen geometry separately",
    )
    native = commands.add_parser(
        "native-palette", help="Fit shared palette on native colors"
    )
    native.add_argument("--manifest", type=Path, required=True)
    native.add_argument("--config", type=Path, required=True)
    native.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate the composition baseline using explicit ground truth"
    )
    evaluate.add_argument("--extraction", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--palette", type=Path)
    compare = commands.add_parser(
        "compare-composition",
        help="Compare native and atlas vectors using the same palette and folds",
    )
    compare.add_argument("--native", type=Path, required=True)
    compare.add_argument("--atlas", type=Path, required=True)
    compare.add_argument("--labels", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Resolve once before Blender or a caller changes the working directory.
    if hasattr(args, "output"):
        args.output = args.output.expanduser().resolve()
    if args.command == "import-inputs":
        Archive.import_inputs(args.source, args.output, args.seed)
    elif args.command == "label-archive":
        Labeling.legacy_kmeans(args.csv, args.output)
    elif args.command == "validate":
        config = Artifacts.config(args.config)
        specimens = Pipeline.donors(config)
        print(f"Validated {len(specimens)} donors; profile {config.profile!r}.")
    elif args.command == "generate":
        Pipeline.generate(args.config, args.output, args.blender)
    elif args.command == "transfer":
        Pipeline.transfer(args.config, args.manifest, args.output, args.blender)
    elif args.command == "extract":
        Analysis.extract(args.manifest, args.output, archived_uv=args.archived_uv)
    elif args.command == "native-palette":
        NativePalette.fit(args.manifest, args.config, args.output)
    elif args.command == "compare-composition":
        Comparison.run(args.native, args.atlas, args.labels, args.output)
    else:
        Analysis.evaluate(
            args.extraction, args.labels, args.config, args.output, args.palette
        )


if __name__ == "__main__":
    main()
