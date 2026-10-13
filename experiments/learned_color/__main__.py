"""Explicit stages for the source-attributed frozen surface-appearance pipeline."""

from pathlib import Path

import click

from experiments.learned_color.pipeline import Appearance
from experiments.learned_color.serde import Config


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
@click.option(
    "--stage",
    type=click.Choice(("prepare", "render", "sample", "embed", "analyze", "all")),
    required=True,
)
def main(config_path: Path, stage: str) -> None:
    """Run one stage or a complete new run with an explicitly selected cohort."""
    config = Config.load(config_path).to_internal()
    selected = (
        ("prepare", "render", "sample", "embed", "analyze")
        if stage == "all"
        else (stage,)
    )
    for name in selected:
        getattr(Appearance, name)(config)
        click.echo(f"{name}: completed")


if __name__ == "__main__":
    main()
