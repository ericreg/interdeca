# InterDeCA command-line client

Run `uv sync`, then `uv run interdeca --help`. UMAP is installed by default;
no optional dependency flag is needed. This Click CLI hosts the application
inside the uv Python process. It uses no WebSocket and starts no application worker
subprocess. Atlas operations still invoke Blender when needed.

Both the CLI and Slicer's WebSocket worker use `application.ApplicationSession`:
the same command validation, controller, application state, pipeline definitions,
serial execution manager, capability discovery, cancellation, and result production.
There is no separate CLI implementation of the numerical workflows.

## One session for a sequence of actions

A fitted model is live application state. Chain commands in one invocation to
reuse it; separate invocations start separate sessions. For example:

```sh
uv run interdeca --output /tmp/interdeca-color-run \
  colors --mesh atlas.obj --textures textures --sample-only --samples 1000 \
  fit --method pca --components 2 \
  axes --x 2 --y 1 \
  reconstruct --x 0.75 --y 0.5 specimen-01 \
  show specimen-02 \
  status
```

Replace the file paths and specimen IDs with your own. Options go **before
positional arguments within each command**, as required by Click command chaining.
Axes are numbered from 1; reconstruction positions are slider fractions in [0, 1].
Omit both reconstruction positions to start at the chosen specimen's fitted scores.

The global `--output` must name a **new directory**. If omitted, the CLI creates a
unique temporary directory and prints its location on stderr. Output is retained
after exit for inspection. Existing directories and exported files are not overwritten.

For interactive work:

```sh
uv run interdeca --output /tmp/interdeca-interactive shell
```

Enter the same commands at `interdeca>`, without `uv run interdeca`. `--help` shows
the command list; `colors --help`, for example, shows that command's options. Type
`exit` or `quit` to leave. Session options such as `--output` are chosen at startup.

A command file uses the same parser and live session:

```text
# commands.txt; paths are relative to the invocation's working directory
colors --mesh atlas.obj --textures textures --sample-only --samples 1000
fit --method pca --components 2
axes --x 1 --y 2
reconstruct --x 0.6 --y 0.4 specimen-01
show specimen-02
```

```sh
uv run interdeca --output /tmp/interdeca-script script commands.txt
```

Each line is a command or command chain. Blank lines and `#` comments are allowed.
Shell substitutions and arbitrary Python are not evaluated. A script stops on its
first error and reports the source line; an interactive shell lets you correct it.

## Slicer functionality mapping

| Slicer action | CLI equivalent |
| --- | --- |
| Start/stop the application worker | Start/exit the CLI session; no worker process is needed |
| Worker estimator capabilities | `capabilities` |
| ATLAS construction and texture transfer | `atlas` |
| Atlas mesh/landmark override | `atlas --atlas-model ... --atlas-landmarks ...` |
| Cleanup, UV, and bake controls | `--merge-distance`, `--uv-angle`, `--island-margin`, `--bake-size`, `--extrusion`, `--bake-margin`, `--blender` |
| Sampling-only processing | `colors --sample-only` |
| Clustering, consolidation, normalization | `colors --cluster --initial-clusters 24 --clusters 8 --normalize` |
| Sampling and neighbor averaging | `--samples N`, `--all-faces`, `--neighbor-average` |
| Apply processed specimen colors | `show SPECIMEN` |
| Apply raw specimen or average texture | `raw-texture --mesh atlas.obj image.png` |
| PCA, ICA, UMAP | `fit --method pca/ica/umap --components N` |
| Change plot axes | `axes --x 1 --y 2` |
| Choose starting specimen / move morphospace sliders | `reconstruct SPECIMEN` / `reconstruct --x X --y Y SPECIMEN` |
| Draw/apply a selection curve | Supply curve coordinates to `select --curve ...` |
| Select one side / refine with source colors | `--same-side` / `--both-sides`, optional `--vertex-colors colors.npy` |
| Accumulate applied regions | Repeat `select` on the same mesh |
| Export a selected region | `export --output region.obj` |
| Clear selection | `clear-selection` |
| Source/settings invalidation | `invalidate colors/population/selection/mesh/curve/all` |
| Progress and cancellation | Progress on stderr, `status`, `cancel`, Ctrl-C |

Widgets, curve drawing, plot windows, and MRML rendering are host presentation
responsibilities. The CLI supplies files instead of scene nodes and saves the same
numerical presentation results instead of displaying them. Raw texture application
is a presentation action in both hosts; it does not refit or process the population.
Export retains source-index maps for transferring additional source attributes;
the optional OBJ contains geometry and per-corner UVs, not arbitrary MRML arrays.

`colors`, `select`, `raw-texture`, and `export` can use `--mesh` to read a file.
Otherwise they reuse the relevant mesh chosen in this session. A successful atlas
sets both current meshes and its baked-texture directory automatically. A changed
mesh invalidates its dependent state. `invalidate all` also clears CLI input choices.

## Atlas and selection examples

```sh
uv run interdeca --output /tmp/interdeca-atlas-session \
  atlas --models inputs/models --landmarks inputs/landmarks \
        --textures inputs/textures --output results --bake-size 512 \
  colors --sample-only --samples 1000 \
  fit --method pca --components 2
```

`atlas --output` is the parent directory for an isolated atlas run and may already
exist. It is separate from the global session output. Omit `--textures` for a
geometry-only atlas. An override requires both mesh and landmarks. Set
`--merge-distance 0` to disable merging. All options use the Slicer/core defaults.

Atlas input mesh directories must contain **RAS** coordinates. For individual
`--mesh` inputs, `--coordinates LPS` explicitly converts the mesh to RAS.
Landmark formats honor their own coordinate metadata. Plain `.npy` curve points
must already be world RAS coordinates with shape `(N, 3)`. Vertex-color inputs are
numerical `(vertex_count, 3)` RGB arrays in canonical vertex order. Object arrays
are rejected; NumPy files are loaded with `allow_pickle=False`.

```sh
uv run interdeca --output /tmp/interdeca-region-session \
  select --mesh atlas.obj --curve patch-1.mrk.json --same-side \
  select --curve patch-2.npy --both-sides \
  export --output selected-region.obj \
  clear-selection
```

Like Slicer, selection uses the core's projected convex region and side heuristic;
it is not a concave/geodesic cut. Export preserves source face order and corner UVs.
Changing mesh geometry before export invalidates the old selection.

## Background work, cancellation, and errors

Analysis commands accept `--background`. In a shell, this returns to the prompt
so you can issue `status`, `cancel`, or `wait`. During input, events queue until the
next command; `wait` displays ongoing progress. In a command chain or script, use
`wait` before an action that depends on background results:

```text
colors --mesh atlas.obj --textures textures --sample-only --background
status
wait
fit --method pca --components 2
```

Top-level invocation completion and normal shell exit wait for submitted work.
Use `cancel` before exiting to stop further tasks. Cancellation is cooperative:
an active native call or Blender operation finishes before cancellation completes.
Ctrl-C during a foreground wait requests cancellation, drains the current operation,
and exits with status 130; the interactive shell instead returns to its prompt.
Partial atlas files remain available. Application failures produce a nonzero exit
status; complete diagnostics are retained in `worker.log` and `events.jsonl`.

## Outputs and testing

```text
session-directory/
  events.jsonl                # typed application events, including failures and logs
  worker.log                  # complete diagnostics
  inputs/                     # temporary snapshots; cleared on acknowledgement/exit
  outputs/*.npy               # retained numerical presentation arrays
  results/*_<action>.json     # metadata referencing those arrays
```

Array references are relative to the session root and record dtype and shape.
Plot results contain scores and specimen IDs; display/reconstruction results
contain face RGB arrays and coverage masks. Selection results contain canonical
indices. Export results contain mesh arrays and source vertex/face mappings.
Raw-texture results contain the mesh and the original image path. No fitted
estimator is pickled, and these artifacts do not restore a live model in another
invocation. Replay a command script to reproduce a session.

Default stdout contains JSON result records; progress goes to stderr. Use
`--json-events` for all application events as JSON lines, or `--quiet` to suppress
progress. Every event is archived regardless of terminal verbosity. Historical
results, including superseded requests, remain in the archive; use request IDs,
input revisions, and terminal events when inspecting asynchronous runs.

`LocalApplicationClient` in `local_client.py` also provides direct Python access:
`send`, `poll`, `wait`, and `close` use the same application service as this CLI.
Commands pass validated envelopes through an in-process queue. Numerical files
retain the existing application boundary checks and make test outputs inspectable;
there is no socket or background application process.

Focused smoke checks covered a three-specimen synthetic atlas and Blender baking,
atlas override/no textures/zero merge, sampled and clustered colors, normalization,
neighbor averaging, PCA, reconstruction, selection accumulation/export, cancellation,
scripts, and the shell. Separate CLI and WebSocket runs were compared numerically
for scores, RGB, masks, selection, and exported mesh correspondence. Tests also
blocked socket creation and worker launching during the CLI color workflow. The
large clams dataset was not run. With UMAP installed by default, an eight-specimen
synthetic CLI check also verified finite two-dimensional scores and reconstruction;
the WebSocket worker advertised UMAP while its client remained independent of core
imports. These checks do not replace a comprehensive numerical suite.
