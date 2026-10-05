# InterDeCA Next in Slicer

The new module has its own name and can coexist with the original InterDeCA.
All implementation is in this repository; the original repository is reference
material and is not imported or modified.

The [Click CLI](cli.md) exposes the same application workflows directly from the
uv environment for testing. It shares the application session, controller, and
pipelines used by the Slicer worker.

For a dataset walkthrough, follow the [small clams manual-testing tutorial](tutorials/small_clams.md).

## Development setup

Create the computational environment outside Slicer:

```sh
cd ~/code/interdeca
uv sync
```

Install only the independent client package into Slicer. On macOS:

```sh
/Applications/Slicer.app/Contents/bin/PythonSlicer -m pip install -e ~/code/interdeca/client
```

The `interdeca-client` distribution depends only on NumPy and `websockets`.
It neither installs nor imports `interdeca`, Pydantic, SciPy, scikit-learn, or UMAP. It uses
Slicer's existing NumPy/VTK for scene snapshots and rendering. An installed CMake
extension includes only the client and Slicer packages, so its users need only the
transport installation in Slicer and an independently installed worker environment.

Add `~/code/interdeca/slicer/InterDeCANext` under **Settings → Modules → Additional
module paths**, then restart. Select **Shape Analysis → InterDeCA Next**. The
**Application Worker** panel selects the external Python executable, normally
`~/code/interdeca/.venv/bin/python` (Windows: `.venv/Scripts/python.exe`). A source
checkout supplies this default; `INTERDECA_WORKER_PYTHON` or the saved setting can
override it. The panel starts a configured worker automatically and also provides
Start/Stop controls. **Install Client Transport** installs only `websockets` and
requires a Slicer restart.

The worker reports PCA, ICA, and UMAP as its estimator capabilities. UMAP is
installed by default in the worker environment; its dependency constraints do not
involve Slicer's binary stack. Startup errors show the retained diagnostic directory,
including subprocess stdout/stderr even when the WebSocket never connects.
Atlas cleanup, UV projection, and texture transfer run in the worker's Rust
extension. No Blender installation or executable setting is required.

Do not add the worker environment's `site-packages` to Slicer's Python path.
Communication uses one local WebSocket, versioned JSON, and managed numerical
files. See the [application protocol](application-protocol.md) for schemas,
ownership, revisions, and cancellation behavior.

The root CMake project supports extension packaging with a Slicer development
build. It packages ordinary Python client/UI files, without computational code or
its dependencies. A CMake extension build still requires a Slicer SDK.

## Workflow parity

| Legacy active workflow | New implementation |
| --- | --- |
| ATLAS inputs and optional atlas override | Matching controls; exact specimen-ID pairing, explicit validation, separate output directory per run |
| Merge by distance, Smart UV, and bake options | Legacy defaults; cleanup runs before final UVs and correspondence; zero disables merging |
| Atlas and texture transfer | Inspection, reference/contributions/mean or override, cleanup, UVs, alignment, resampling, baking, averaging, result display |
| Mesh Selection | Closed curve placement, Escape, one-side filtering, accumulation, clear, selection counts, and export to a new model node |
| MultiRecolor Step 1 | Clustering or sampling-only processing, normalization, cluster consolidation, all-face/subsampling and neighbor averaging |
| MultiRecolor Step 2 | Raw texture, sampled colors, or shared cluster palette; existing analysis is reused |
| MultiRecolor Step 3 | PCA/ICA/UMAP, component count, specimen plot, and axis changes without refitting |
| MultiRecolor Step 4 | Starting specimen, coordinate sliders, moving plot point, and reconstructed surface colors |
| Worker configuration, paths, progress and errors | Independent Python executable, startup diagnostics, worker capabilities, shared structured task progress/log panel, cancellation for all workflows |

Disabled Visualize Results features and disconnected DeCAL/subsetting/painting
handlers are not implemented. Styling is compact rather than pixel-identical.
The shared progress panel replaces repeated per-step progress widgets.

## Data and behavior details

- Coordinates cross the scene boundary in world RAS. Canonical vertex IDs are
  retained when display geometry duplicates vertices at texture seams. Geometry
  and UV edits invalidate dependent results; color/display changes do not.
- Selected scene meshes must contain triangular polygon surfaces. Region export
  creates a new scene node from the accumulated applied selection and copies UVs
  and mapped source attributes. Slicer's normal Save action can write that node.
- Selection follows the active legacy projected convex-envelope algorithm with
  a 5% buffer and its shortest-bounding-axis side heuristic. This is not an exact
  concave or geodesic curve cut. Invalid/degenerate curves produce errors.
- Cleanup may change the atlas connectivity before the final correspondence is
  created. Baking never merges independently resampled targets. An existing
  atlas selected for MultiRecolor is not cleaned or unwrapped.
- Atlas output directories contain the atlas OBJ/landmarks, aligned and resampled
  OBJ/landmarks, specimen textures, and an average texture when textures were
  supplied. Output layout is new and does not promise legacy file hashes.
- The UV angle control respects the supported Blender range (up to 89 degrees).
  A sample budget larger than the mesh uses all available faces.
- Average textures are available for display but excluded from population inputs.
  Missing color coverage remains an explicit mask, not an assumption that black
  pixels are missing. Population features use the common valid sample intersection.
- Clustering with subsampling disabled uses cluster-area features; sampling-only
  processing uses spatial colors at every chosen face. Cluster-area features
  cannot reconstruct spatial colors and disable morphospace.
- Slicer sends the starting specimen and normalized slider positions. The worker
  owns fitted coordinate ranges, interpolation, and pending reconstruction requests.
  Morphospace inverts the fitted transform and reverses feature standardization.
  PCA/ICA use their fitted inverse; UMAP's inverse is approximate and may fail.
  Unavailable inverses are disabled. Only display colors are clipped to RGB gamut.
- Sliders are debounced for 120 ms; only the newest pending reconstruction runs.
  The worker receives cancellation during computation and applies it between
  operations. Stop Worker ends its session and discards its fitted estimators.
  Cancelled or failed runs may leave completed persistent output artifacts.
- Mesh order and per-corner UVs cross the process boundary in `.npy` files.
  Presentation results include request/run identities and input revisions;
  obsolete results are discarded before updating the scene.

## Verification and deferred work

The [small-clams pytest](../tests/end_to_end/README.md) launches this module in a
separate Slicer process and compares its full workflow and MRML outputs against
the CLI. Run it with `uv run pytest tests/end_to_end/test_small_clams.py -v -s`;
it is not registered in CMake. For the worker boundary, earlier focused temporary
smoke runs exercised the actual client and separately launched worker through
color sampling, PCA, axis changes, specimen display, coalesced reconstruction,
selection/export, invalidation, file cleanup, and shutdown. Export preserved face
order and per-corner UVs. Client imports did not load the computational package.
Malformed command correlation, stale dataset/model IDs, restart, and captured
startup failures were also checked. A controlled active operation confirmed that
cancellation was acknowledged before the operation finished, followed by a
correlated cancelled event without a result. Unauthenticated connections were
rejected. Syntax/import checks and independent client/worker wheel builds passed.
The end-to-end test now provides a repeatable check of the Slicer UI and full
atlas/texture workflow; consult its README for prerequisites and current limitations.

Exhaustive numerical parity, large datasets, cross-platform Slicer/Blender support,
extension-SDK builds, persistent project restoration, and a standalone Qt view
remain separate validation or implementation work. The existing numerical
algorithm choices in the main README remain authoritative.
