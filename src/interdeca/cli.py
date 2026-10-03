"""Click presentation host for the same application session used by Slicer.

Commands can be chained in one invocation or entered in a persistent shell.
Numerical results are saved as JSON and .npy files; fitted estimators stay live
only for the duration of that invocation. No socket or worker process is started.
"""

import json
import shlex
import tempfile
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

import click
import numpy as np

from interdeca.io import Files, LandmarkIO, MeshIO, TextureIO
from interdeca.local_client import ApplicationError, LocalApplicationClient
from interdeca.models import Mesh
from interdeca.serde import (
    AtlasRequest,
    ColorRequest,
    LogEvent,
    MeshArtifacts,
    ProgressEvent,
    RawTexturePresentation,
    ResultEvent,
    StateEvent,
    TerminalEvent,
)


class HostInputs(NamedTuple):
    """File-backed equivalents of Slicer's selected nodes and texture directory."""

    color_mesh: Mesh | None = None
    selection_mesh: Mesh | None = None
    textures: Path | None = None


class CliApplication:
    """Present application events as durable numerical artifacts and terminal output."""

    def __init__(self, directory: Path, quiet: bool, json_events: bool):
        self.directory = directory
        self.quiet, self.json_events = quiet, json_events
        self.inputs = HostInputs()
        self.exports: dict[str, Path] = {}
        self.last_progress = None
        self._client = None
        self.events_file = None

    @property
    def client(self) -> LocalApplicationClient:
        """Delay session startup so subcommand help has no filesystem or thread effects."""
        if self._client is None:
            self._client = LocalApplicationClient(self.directory, self.receive)
            self.events_file = (self.directory / "events.jsonl").open(
                "x", encoding="utf-8"
            )
            (self.directory / "results").mkdir()
            click.echo(f"Session output: {self._client.root}", err=True)
            self._client.wait_ready()
        return self._client

    def dataset_id(self) -> str:
        self.client.poll()
        if self.client.state is None or self.client.state.dataset_id is None:
            raise click.UsageError(
                "Run colors first in this session, then fit or show a specimen."
            )
        return self.client.state.dataset_id

    def model_id(self) -> str:
        self.client.poll()
        if self.client.state is None or self.client.state.model_id is None:
            raise click.UsageError("Run colors and fit first in this session.")
        return self.client.state.model_id

    def receive(self, event):
        # Archive all validated events, including diagnostics and cancelled requests.
        line = event.model_dump_json()
        self.events_file.write(line + "\n")
        self.events_file.flush()
        if self.json_events:
            click.echo(line)
        if isinstance(event, ResultEvent):
            destination = (
                self.client.root / "results" / f"{event.request_id}_{event.action}.json"
            )
            Files.write(destination, (line + "\n").encode())
            # Historical results stay inspectable but must not replace current inputs.
            if self.client.results.get(event.action) is not event:
                return
            if event.action == "atlas":
                mesh = self.read_mesh(event.payload.mesh)
                self.inputs = HostInputs(
                    mesh,
                    mesh,
                    Path(event.payload.directory) / "textures"
                    if event.payload.has_textures
                    else None,
                )
            elif event.action == "export" and event.request_id in self.exports:
                MeshIO.write_obj(
                    self.read_mesh(event.payload.mesh),
                    self.exports.pop(event.request_id),
                )
            if not self.json_events:
                click.echo(line)
        elif isinstance(event, ProgressEvent) and not self.quiet:
            progress = (event.run_id, event.completed, event.stage, event.operation)
            if progress != self.last_progress:
                click.echo(
                    f"Stage {event.stage}/{event.stages} · {event.completed}/{event.total} tasks · {event.operation}",
                    err=True,
                )
                self.last_progress = progress
        elif isinstance(event, LogEvent):
            for entry in event.entries:
                if entry.severity in ("WARNING", "ERROR"):
                    click.echo(
                        f"{entry.severity}: {entry.message.splitlines()[0]}", err=True
                    )
        elif isinstance(event, TerminalEvent) and not self.quiet:
            click.echo(f"{event.type}: {event.message}", err=True)

    def read_mesh(self, references: MeshArtifacts) -> Mesh:
        store = self.client.artifacts
        return Mesh(
            store.read(references.vertices.model_dump(mode="json"), "outputs"),
            store.read(references.faces.model_dump(mode="json"), "outputs"),
            None
            if references.uv is None
            else store.read(references.uv.model_dump(mode="json"), "outputs"),
        )

    def choose_mesh(self, path: Path | None, coordinates: str, selection=False) -> Mesh:
        previous = self.inputs.selection_mesh if selection else self.inputs.color_mesh
        mesh = MeshIO.read(path, coordinate_system=coordinates) if path else previous
        if mesh is None:
            mesh = self.inputs.color_mesh if selection else self.inputs.selection_mesh
        if mesh is None:
            raise click.UsageError("Supply --mesh or run atlas first in this session.")
        if previous is not None and any(
            not np.array_equal(old, new)
            for old, new in zip(previous, mesh, strict=True)
        ):
            # A changed source invalidates its numerical dependents, like Slicer's observers.
            self.perform("invalidate", scope="selection" if selection else "colors")
        self.inputs = self.inputs._replace(
            **{"selection_mesh" if selection else "color_mesh": mesh}
        )
        return mesh

    def perform(
        self, action: str, *, background=False, export: Path | None = None, **payload
    ):
        request_id = self.client.send(action, **payload)
        if export is not None:
            self.exports[request_id] = export
        if not background:
            self.wait()
        return request_id

    def wait(self):
        try:
            self.client.wait()
        except KeyboardInterrupt:
            # Cancellation remains cooperative: keep pumping until the active task ends.
            click.echo(
                "Cancelling; waiting for the active operation to finish…", err=True
            )
            self.client.send("cancel")
            self.client.wait()
            raise click.exceptions.Exit(130) from None

    def close(self):
        if self._client is not None:
            self._client.close()
        if self.events_file is not None:
            self.events_file.close()


class CliGroup(click.Group):
    """Keep contextual errors concise while complete tracebacks remain in worker.log."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except (ApplicationError, ValueError, OSError) as error:
            raise click.ClickException(str(error)) from error


# All commands use one context-owned host, so a fitted estimator need not be pickled.
pass_application = click.make_pass_decorator(CliApplication)
input_file = click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True)
input_directory = click.Path(
    exists=True, file_okay=False, path_type=Path, resolve_path=True
)
output_path = click.Path(path_type=Path, resolve_path=True)


def background_option(function):
    return click.option(
        "--background",
        is_flag=True,
        help="Return to the shell or next command while this action runs.",
    )(function)


def mesh_options(function):
    function = click.option(
        "--coordinates",
        type=click.Choice(["RAS", "LPS"]),
        default="RAS",
        show_default=True,
        help="Coordinate convention of an explicitly supplied mesh.",
    )(function)
    return click.option(
        "--mesh",
        type=input_file,
        help="Mesh file; defaults to the current mesh or the last atlas.",
    )(function)


@click.group(
    cls=CliGroup, chain=True, context_settings={"help_option_names": ["-h", "--help"]}
)
@click.option(
    "--output",
    type=output_path,
    help="New directory for results and diagnostics; defaults to a temporary directory.",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress progress; retain errors and all diagnostic files.",
)
@click.option(
    "--json-events",
    is_flag=True,
    help="Write every application event as one JSON line to stdout.",
)
@click.pass_context
def main(ctx, output, quiet, json_events):
    """Run InterDeCA in process. Chain commands to share one live application session.

    Example: interdeca colors --mesh atlas.obj --textures textures fit --method pca

    Within each command, place options before positional arguments.
    """
    ctx.meta["owns_application"] = ctx.obj is None
    if ctx.obj is None:
        # A unique directory refuses accidental replacement of earlier diagnostic results.
        directory = (
            output or Path(tempfile.gettempdir()) / f"interdeca-cli-{uuid4().hex}"
        )
        ctx.obj = CliApplication(directory, quiet, json_events)
        ctx.call_on_close(ctx.obj.close)
    elif output is not None or quiet or json_events:
        raise click.UsageError(
            "Set session options when starting the CLI, before entering a shell or script."
        )


@main.result_callback()
@click.pass_context
def finish_commands(ctx, results, **options):
    # Nested shell/script invocations preserve background work; top-level commands drain it.
    if ctx.meta["owns_application"]:
        ctx.obj.wait()


@main.command()
@pass_application
def capabilities(app):
    """Report the installed population estimators and diagnostic location."""
    click.echo(app.client.capabilities.model_dump_json())


@main.command()
@click.option("--models", type=input_directory, required=True)
@click.option("--landmarks", type=input_directory, required=True)
@click.option("--textures", type=input_directory)
@click.option(
    "--output",
    "atlas_output",
    type=output_path,
    required=True,
    help="Parent directory for a new atlas run.",
)
@click.option("--atlas-model", type=input_file)
@click.option("--atlas-landmarks", type=input_file)
@click.option(
    "--backend",
    type=click.Choice(["native", "morphoweave"]),
    default="native",
    show_default=True,
    help="Atlas implementation; keep native for the original comparison baseline.",
)
@click.option(
    "--normalize-scale/--preserve-scale",
    default=False,
    show_default=True,
    help="Allow scale changes during specimen alignment. Generated MorphoWeave atlases are restored to mean input size in either mode.",
)
@click.option(
    "--blender", type=str, help="Blender executable; otherwise use normal discovery."
)
@click.option(
    "--merge-distance", type=click.FloatRange(min=0), default=0.0001, show_default=True
)
@click.option(
    "--uv-angle",
    type=click.FloatRange(min=0, max=89, min_open=True),
    default=66.0,
    show_default=True,
)
@click.option(
    "--island-margin",
    type=click.FloatRange(min=0, max=1, max_open=True),
    default=0.002,
    show_default=True,
)
@click.option(
    "--bake-size", type=click.IntRange(128, 8192), default=2048, show_default=True
)
@click.option(
    "--extrusion", type=click.FloatRange(min=0), default=0.001, show_default=True
)
@click.option("--bake-margin", type=click.IntRange(0, 64), default=2, show_default=True)
@background_option
@pass_application
def atlas(app, atlas_output, background, **options):
    """Build or override an atlas, align/resample specimens, and optionally bake textures.

    Input mesh directories must contain RAS coordinates; landmark file metadata is honored.
    """
    request = AtlasRequest(output=atlas_output, **options)
    app.perform("run_atlas", background=background, **request.model_dump(mode="json"))


@main.command()
@mesh_options
@click.option(
    "--textures",
    type=input_directory,
    help="Specimen textures; defaults to the last selected directory or atlas bakes.",
)
@click.option("--cluster/--sample-only", default=True, show_default=True)
@click.option("--normalize/--no-normalize", default=False, show_default=True)
@click.option(
    "--initial-clusters", type=click.IntRange(2, 64), default=24, show_default=True
)
@click.option("--clusters", type=click.IntRange(2, 64), default=8, show_default=True)
@click.option("--samples", type=click.IntRange(min=1), default=10000, show_default=True)
@click.option(
    "--all-faces", is_flag=True, help="Disable subsampling; overrides --samples."
)
@click.option(
    "--neighbor-average/--no-neighbor-average", default=False, show_default=True
)
@background_option
@pass_application
def colors(
    app, mesh, coordinates, textures, cluster, samples, all_faces, background, **options
):
    """Sample or cluster the texture population on the current mesh."""
    numerical = app.choose_mesh(mesh, coordinates)
    directory = textures or app.inputs.textures
    if directory is None:
        raise click.UsageError("Supply --textures or run an atlas with textures first.")
    request = ColorRequest(
        directory=directory,
        clustering=cluster,
        sample_count=None if all_faces else samples,
        **options,
    )
    app.inputs = app.inputs._replace(textures=directory)
    app.perform(
        "process_colors",
        background=background,
        mesh=app.client.mesh(numerical),
        options=request.model_dump(mode="json"),
    )


@main.command()
@click.option(
    "--method",
    type=click.Choice(["pca", "ica", "umap"]),
    default="pca",
    show_default=True,
)
@click.option("--components", type=click.IntRange(1, 20), default=2, show_default=True)
@background_option
@pass_application
def fit(app, method, components, background):
    """Fit the current processed population; retain the model for later commands."""
    app.client.poll()
    app.perform(
        "fit_population",
        background=background,
        dataset_id=app.dataset_id(),
        options={"method": method, "components": components},
    )


@main.command()
@click.argument("specimen")
@background_option
@pass_application
def show(app, specimen, background):
    """Save an individual specimen's processed face RGB values and coverage mask."""
    app.client.poll()
    app.perform(
        "show_specimen",
        background=background,
        dataset_id=app.dataset_id(),
        specimen_id=specimen,
    )


@main.command("raw-texture")
@click.argument("texture", type=input_file)
@mesh_options
@pass_application
def raw_texture(app, texture, mesh, coordinates):
    """Save a mesh/image presentation for a raw specimen or an average texture.

    This is a host presentation action, like applying an image directly in Slicer.
    """
    numerical = app.choose_mesh(mesh, coordinates)
    if numerical.uv is None:
        raise click.UsageError("A raw texture requires mesh UV coordinates.")
    TextureIO.validate_file(texture)
    store = app.client.artifacts
    references = {
        "vertices": store.write(numerical.vertices, "outputs"),
        "faces": store.write(numerical.faces, "outputs"),
        "uv": None if numerical.uv is None else store.write(numerical.uv, "outputs"),
    }
    event = RawTexturePresentation(
        mesh=MeshArtifacts.model_validate_json(json.dumps(references)), texture=texture
    )
    Files.write(
        app.client.root / "results" / f"{uuid4().hex}_raw_texture.json",
        (event.model_dump_json() + "\n").encode(),
    )
    app.receive(event)
    if not app.json_events:
        click.echo(event.model_dump_json())


@main.command()
@click.option(
    "--x",
    type=click.IntRange(min=1),
    required=True,
    help="Horizontal component, numbered from 1.",
)
@click.option(
    "--y",
    type=click.IntRange(min=1),
    required=True,
    help="Vertical component, numbered from 1.",
)
@pass_application
def axes(app, x, y):
    """Change plot axes without refitting the population."""
    app.client.poll()
    app.perform(
        "set_axes",
        model_id=app.model_id(),
        x_axis=x - 1,
        y_axis=y - 1,
    )


@main.command()
@click.argument("specimen")
@click.option(
    "--x",
    type=click.FloatRange(0, 1),
    help="Horizontal slider fraction; supply both --x and --y.",
)
@click.option(
    "--y",
    type=click.FloatRange(0, 1),
    help="Vertical slider fraction; omit both to start at the specimen.",
)
@background_option
@pass_application
def reconstruct(app, specimen, x, y, background):
    """Reconstruct spatial face colors using the current axes and fitted model."""
    if (x is None) != (y is None):
        raise click.UsageError("Supply both --x and --y, or neither.")
    app.client.poll()
    app.perform(
        "reconstruct",
        background=background,
        model_id=app.model_id(),
        specimen_id=specimen,
        positions=None if x is None else [x, y],
    )


@main.command()
@mesh_options
@click.option(
    "--curve",
    type=input_file,
    required=True,
    help="RAS .npy points, or landmarks in .mrk.json/.fcsv/.csv with file coordinates.",
)
@click.option("--same-side/--both-sides", default=True, show_default=True)
@click.option(
    "--vertex-colors", type=input_file, help="Optional canonical N-by-3 RGB .npy array."
)
@background_option
@pass_application
def select(app, mesh, coordinates, curve, same_side, vertex_colors, background):
    """Accumulate a curve-defined region using the same selection algorithm as Slicer."""
    numerical = app.choose_mesh(mesh, coordinates, selection=True)
    points = (
        np.load(curve, allow_pickle=False)
        if curve.suffix.lower() == ".npy"
        else LandmarkIO.read(curve).points
    )
    store = app.client.artifacts
    app.perform(
        "select_region",
        background=background,
        mesh=app.client.mesh(numerical),
        points=store.write(points, "inputs"),
        same_side_only=same_side,
        vertex_colors=None
        if vertex_colors is None
        else store.write(np.load(vertex_colors, allow_pickle=False), "inputs"),
    )


@main.command()
@mesh_options
@click.option(
    "--output",
    "destination",
    type=output_path,
    help="Optional new OBJ path; arrays and source-index mappings are always retained.",
)
@background_option
@pass_application
def export(app, mesh, coordinates, destination, background):
    """Extract the accumulated region, retaining face/vertex source mappings and corner UVs."""
    if destination is not None and (
        destination.suffix.lower() != ".obj" or destination.exists()
    ):
        raise click.UsageError("--output must name a new .obj file.")
    numerical = app.choose_mesh(mesh, coordinates, selection=True)
    app.perform(
        "export_region",
        background=background,
        export=destination,
        mesh=app.client.mesh(numerical),
    )


@main.command("clear-selection")
@pass_application
def clear_selection(app):
    """Clear the accumulated region while retaining exported results."""
    app.perform("clear_selection")


@main.command()
@click.argument(
    "scope",
    type=click.Choice(["colors", "population", "selection", "mesh", "curve", "all"]),
    default="colors",
)
@pass_application
def invalidate(app, scope):
    """Invalidate state after changing source data or settings."""
    app.perform("invalidate", scope=scope)
    if scope == "all":
        app.inputs = HostInputs()


@main.command()
@pass_application
def cancel(app):
    """Cancel active and pending work; wait for any current serial operation to finish."""
    app.perform("cancel")


@main.command()
@pass_application
def wait(app):
    """Wait for background work and report any application failure."""
    app.wait()


@main.command()
@pass_application
def status(app):
    """Print the current application controls, IDs, progress, and selection counts."""
    app.client.poll()
    if app.client.state is not None:
        click.echo(StateEvent(payload=app.client.state).model_dump_json())


def invoke_line(app, line):
    """Use Click's own parser for both scripts and the interactive shell."""
    arguments = shlex.split(line, comments=True)
    if arguments:
        main.main(args=arguments, obj=app, standalone_mode=False)


@main.command("script")
@click.argument("path", type=input_file)
@pass_application
def script(app, path):
    """Execute one command chain per line, preserving state and stopping on errors.

    Paths are relative to the invocation's working directory. Blank lines and # comments
    are allowed; shell substitutions and arbitrary Python are not evaluated.
    """
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            invoke_line(app, line)
        except click.ClickException as error:
            raise click.ClickException(
                f"{path}:{line_number}: {error.format_message()}"
            ) from error


@main.command()
@pass_application
def shell(app):
    """Keep one live application open. Type --help, exit, or quit.

    Add --background to an analysis command to return to the prompt. Use status,
    wait, and cancel to inspect or control it. Ctrl-C cancels a foreground action.
    """
    while True:
        try:
            app.client.poll()
            line = click.prompt(
                "interdeca", prompt_suffix="> ", default="", show_default=False
            )
            if line.strip() in ("exit", "quit"):
                return
            invoke_line(app, line)
        except click.ClickException as error:
            error.show()
        except click.exceptions.Exit as error:
            if error.exit_code != 130:
                raise
        except (click.Abort, EOFError):
            return
        except ValueError as error:
            click.echo(f"Error: {error}", err=True)


if __name__ == "__main__":
    main()
