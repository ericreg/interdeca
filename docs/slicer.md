# InterDeCA Next in Slicer

The new module has its own name and can coexist with the original InterDeCA.
All implementation is in this repository; the original repository is reference
material and is not imported or modified.

## Development setup

Use Slicer's Python to install the editable core without replacing its bundled
VTK/NumPy stack. On macOS:

```sh
/Applications/Slicer.app/Contents/bin/PythonSlicer -m pip install --no-deps -e ~/code/interdeca
```

In Slicer, add `~/code/interdeca/slicer/InterDeCANext` under **Settings → Modules →
Additional module paths**, then restart. Select **Shape Analysis → InterDeCA Next**.
If analysis dependencies are missing, the module still displays Package Management.
Its status checks distinguish required packages from optional UMAP support.
**Check & Install Missing Packages** installs the required analysis packages.
**Install All Recommended Packages** installs required packages first, then tries
UMAP separately; an optional dependency failure cannot prevent core installation.
Install actions use Slicer's Python, pin installed NumPy, SciPy, VTK, and Pillow,
and request a restart. The installer uses binary wheels and displays pip's output;
it does not attempt local compiler/LLVM builds or replace pinned host libraries.

On the tested Intel Python runtime bundled with this macOS Slicer installation,
NumPy is 2.4.6, while available Intel macOS Numba wheels require older NumPy.
Consequently optional UMAP cannot currently be installed through this panel.
ATLAS, selection, color processing, PCA, and ICA remain usable without UMAP.
Blender is discovered through the configured path, `BLENDER_EXECUTABLE`, PATH,
or conventional application locations. Install Blender separately if absent.

Do not add the uv virtual environment's `site-packages` to Slicer's Python path:
its Python version, architecture, and compiled libraries can differ. The uv
environment remains useful for independent core development.

The root CMake project supports extension packaging with a Slicer development
build. It installs the small module entry point and both supporting packages as
ordinary Python files. It does not bundle dependencies or concatenate sources.

## Workflow parity

| Legacy active workflow | New implementation |
| --- | --- |
| ATLAS inputs and optional atlas override | Matching controls; exact specimen-ID pairing, explicit validation, separate output directory per run |
| Merge by distance, Smart UV, and bake options | Legacy defaults; cleanup runs before final UVs and correspondence; zero disables merging |
| Atlas and texture transfer | Inspection, reference/contributions/mean or override, cleanup, UVs, alignment, resampling, baking, averaging, result display |
| Mesh Selection | Closed curve placement, Escape, one-side filtering, accumulation, clear, selection counts, and export to a new model node |
| MultiRecolor Step 1 | Clustering or sampling-only processing, normalization, cluster consolidation, all-face/subsampling and neighbor averaging |
| MultiRecolor Step 2 | Raw texture, sampled colors, or shared cluster palette; existing analysis is reused |
| MultiRecolor Step 3 | PCA/ICA/optional UMAP, component count, specimen plot, and axis changes without refitting |
| MultiRecolor Step 4 | Starting specimen, coordinate sliders, moving plot point, and reconstructed surface colors |
| Package management, paths, progress and errors | Separate settings namespace, dependency bootstrap, shared task progress/log panel, cancellation for all workflows |

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
- Morphospace inverts the fitted transform and reverses feature standardization.
  PCA/ICA use their fitted inverse; UMAP's inverse is approximate and may fail.
  Unavailable inverses are disabled. Only display colors are clipped to RGB gamut.
- Sliders are debounced for 120 ms; only the newest pending reconstruction runs.
  The Slicer dispatcher briefly yields Python's GIL during active work, following
  Slicer's bundled SimpleFilters approach; this prevents PythonQt's event loop
  from starving a Python worker between native calls.
  Cancellation waits for an active native/Blender operation to finish. A cancelled
  or failed run may leave completed output artifacts, which are not deleted.

## Verification and deferred work

The implementation phase uses focused smoke checks rather than a new automated
test suite. The review checklist is: Python 3.12 imports; serial ordering and
barriers; cancellation and failure isolation; Blender cleanup/UV/bake; scene and
UV round trips; selection/export; both processing modes; population plots;
morphospace; source invalidation; and teardown with queued callbacks.

Focused checks completed on macOS with Slicer 5.12.4 (Python 3.12.10):

- Built the uv wheel and source distribution; parsed every module with Python
  3.12 grammar and checked imports and critical lint rules.
- Ran synthetic four-specimen pipelines through selection/extraction, both color
  processing modes, PCA, display expansion, inverse reconstruction, and the full
  atlas pipeline with real Blender cleanup, UV generation, four bakes, and averaging.
- Confirmed actual vertex merging preserves landmark coordinates, and region/worker
  boundary objects round-trip through Pydantic serialization.
- Exercised FIFO order, stage-reference validation, failure isolation, and
  cancellation between tasks with a controlled active operation. Delayed completion
  events were rejected after cancellation, input invalidation, and view teardown.
- Verified ICA inverse reconstruction reverses fitted standardization, and
  cluster-area features reject spatial reconstruction.
- Loaded the distinct Slicer module alongside the legacy module; exercised atlas
  generation and override/no-texture paths, selector handoff, scene/UV round trips,
  population plotting, axis changes, raw/processed display, morphospace, selection,
  export, clearing, and clean teardown while the UI event timer continued running.
- Verified export retains per-corner seam attributes and UVs, including transformed
  normals and geometry when the source has a parent scene transform.
- Loaded the dependency-management UI in the unchanged host with the core's
  computational dependencies absent.
- Exercised the actual installer in Slicer: required dependencies installed,
  output streamed through PythonQt, completion released controls and temporary
  files, and an optional UMAP dependency conflict left core packages usable.
  NumPy, SciPy, VTK, and Pillow versions remained unchanged.

These are small-fixture smoke checks, not exhaustive numerical comparisons.
The optional UMAP fit/inverse remains unverified. Initial numerical smoke checks
used temporary dependency installations; the installer check subsequently added
the required packages to Slicer's own environment without replacing its bundled
libraries. The CMake extension build still requires a Slicer SDK, which is not
available in this installation.

Exhaustive numerical parity, large datasets, cross-platform Slicer/Blender support,
extension-SDK builds, persistent project restoration, and a standalone Qt view
remain separate validation or implementation work. The existing numerical
algorithm choices in the main README remain authoritative.
