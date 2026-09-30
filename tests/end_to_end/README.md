# Small clams: Slicer versus CLI

Run from the repository root:

```sh
uv sync
uv run pytest tests/end_to_end/test_small_clams.py -v -s
```

Pytest is a development dependency installed by the default `uv sync`. This is
an ordinary pytest integration test: it is not registered in Slicer or CMake.
It launches the real Click entry point and a separate Slicer application, whose
worker uses the same uv Python interpreter as pytest. Slicer runs with temporary
settings, without loading the user's startup script or showing its main window.
The test creates fresh outputs; it does not read an old atlas as its baseline.

## Requirements

- The raw three-specimen `small_clams` dataset. By default, the test reads
  `../3d_color_package/tests/data/small_clams`. Set `INTERDECA_SMALL_CLAMS` to
  another copy. Only `UF_IZ_507781`, `UF_IZ_507782`, and `UF_IZ_507879` are used,
  even if other files exist there. The reference package and input files are read-only.
- Slicer on `PATH` or at `/Applications/Slicer.app/Contents/MacOS/Slicer`.
  Override with `SLICER_EXECUTABLE` (the application launcher, not PythonSlicer).
- Blender, found by the core's normal discovery. Override with
  `BLENDER_EXECUTABLE`.
- The independent client installed in Slicer's Python. On macOS:

  ```sh
  /Applications/Slicer.app/Contents/bin/PythonSlicer -m pip install -e client
  ```

The test does not install packages or change Slicer settings. It passes only
the client and UI source directories to Slicer, never the uv site-packages.
Missing applications, data, transport dependencies, crashes, timeouts, and missing
outputs fail the test rather than produce a skip or a false success.
This harness uses POSIX process groups for timeout cleanup; macOS is the initial
validation platform. Linux needs a working graphical display, such as Xvfb.
The full raw-data comparison passed with Slicer 5.12.4 and Blender 5.2.1 on macOS
in approximately 2 minutes 38 seconds. These external applications are not pinned
by `uv.lock`; each run compares hosts using the same Blender installation.

## Workflow and assertions

Both hosts independently build an atlas from all three original dense meshes,
align and resample each specimen, and bake 512 × 512 textures. The only fixture
preparation is conversion of the source LPS meshes and landmark metadata to RAS
in pytest's scratch directory. Meshes are not decimated. Both hosts then perform
sampling-only color analysis with 1,000 faces and neighbor averaging, fit a
two-component PCA model, swap axes, reconstruct at slider positions `(0.7, 0.3)`,
display another specimen, select part of the mesh, export it, and clear selection.
UMAP cannot fit this three-specimen fixture because the core requires at least four.

The Slicer driver loads `InterDeCANext`, edits its actual controls, invokes button
signals, pumps Qt, and lets the normal application client and renderer operate.
It replaces only modal error reporting with a fatal recorded error and observes
completed presentations. The comparison checks:

- The complete atlas output file set; aligned and resampled vertices and
  landmarks; exact face order and per-corner UVs; decoded texture pixels.
- Specimen identities, PCA scores, swapped axes, reconstructed and displayed RGB,
  validity masks, morphospace positions, selection indices, and export source maps.
- Actual MRML model snapshots, plot-table values, rendered cell RGB bytes,
  selection overlays, and the exported scene model against the CLI outputs.
- Nonempty results, a proper partial selection, successful worker shutdown,
  input cleanup, and the absence of the computational package and UMAP in Slicer.

Geometry and landmarks use `rtol=1e-12, atol=1e-9`; PCA scores, coordinates, and
RGB use `rtol=1e-10, atol=1e-10`. Indices, masks, UVs, rendered colors, and texture
pixels must match exactly. No sorting, sign alignment, or image-error allowance
hides correspondence or host differences. Session IDs and artifact filenames
are deliberately excluded because each host owns an independent session.

Nearest-surface projection can collapse triangles in individual correspondence
meshes. The regression checks retain and compare those faces with the dedicated
output schema; native meshes and the shared analysis atlas remain strict.
`uv run pytest tests/test_correspondence.py` separately checks the collapse case,
serialization and Blender handoff, and rejection of invalid source geometry.

## Diagnostics

`-s` prints the pytest scratch directory. Each run retains prepared inputs,
`cli.log`, `slicer.log`, both complete atlas outputs, CLI events and numerical
artifacts, and Slicer presentations/MRML snapshots. Slicer also copies its worker's
`worker.log`, `stdout.log`, and `stderr.log` there. On failure, pytest shows the
external process's log tail. Use a new `--basetemp` directory to choose where
pytest keeps artifacts; pytest may delete an existing directory at that path.

Allow several minutes and a few GB of scratch space. Each external host has a
40-minute limit; each Slicer action has a 20-minute limit. This test never runs the
full clams or mussels datasets. Comparing the two current hosts establishes parity,
not independent scientific validation of their shared numerical algorithms.
