# Full-population comparison

Run from the uv environment. This benchmark compares the pre-optimization B0
application with the current application, independently for Native Atlas and
MorphoWeave. It uses archived Git sources in private directories, preserving
the user's checkout and the original reference package. No Slicer/CMake test
registration or production numerical changes are involved.

The local `../3d_color_package/data/mussels_52` dataset has **47**, not 52,
complete specimens. Its 141 input hashes match its reviewed reference manifest.
The raw `UF_IZ_63733.obj` has one exactly zero-area face, zero-based index 177014.
Strict source validation rejects it. The explicit repair option removes that
face only from the common private input copy and records its index. It preserves
all vertex coordinates and the order and corner UVs of all remaining faces.
Nonzero thin triangles are not silently removed. Original input files stay
unchanged. Coordinate conversion from LPS to RAS is recorded by the prepared
files; textures keep their original dimensions and bytes.

```sh
uv run --locked --group profiling python -m tools.profiling.population \
  --dataset ../3d_color_package/data/mussels_52 --expected-specimens 47 \
  --repair-zero-area-inputs --output /path/to/new-capture-directory
uv run --locked --group profiling python -m tools.profiling.accuracy \
  /path/to/new-capture-directory
uv run --locked --group profiling python -m tools.profiling.population_report \
  /path/to/new-capture-directory --output docs/profiling/full-population
uv run --locked --group profiling pytest tests/test_population_accuracy.py
```

The default comparison has three baseline repeats, a nested-timer pass, a CPU
pass, and an allocation pass for each version/backend: 24 sequential captures.
Only baseline captures supply normal runtime and sampled RSS comparisons.
The application chain matches the small-clams profile: dense atlas construction,
alignment, correspondence, 512² texture baking, 1,000-face neighbor-averaged color
sampling, PCA, axes, reconstruction, display, selection, export, and clear.
This measures the shared application core through the CLI, excluding Slicer UI
startup and rendering. UMAP is not part of this fixed comparison workload.
Output fingerprinting, storage compaction, accuracy diagnostics, and report
generation run outside the measured child. Memray tracks native allocations
during the CLI session, with native stacks and aggregated peak records. It does
not enable individual Python-allocator tracing, so Python arenas can include
free slots; its allocation counts are not counts of live Python objects. Import
allocations before tracking begins and executed Blender processes are excluded.

The reference defaults to `31fb5d9`, which adds the original recorder while
retaining the `129c2e2` numerical code. The current snapshot defaults to
`3aa93df`; use `--candidate` explicitly for a different reviewed revision. Both
use the calling uv interpreter and locked environment, with source/dependency
hashes, prepared input hashes, and Blender identity recorded separately.

`--resume` verifies source/runtime/input identity and completed output hashes.
It never combines changed inputs or silently replaces a failed capture. Move a
failed run aside explicitly before restarting. Runs have a configurable four-hour
limit. The harness checks for 8 GiB of free storage before starting each capture.
On APFS, completed byte-identical large files are replaced with independently
writable filesystem clones **after** measurement; no hard links or measurements
of compressed inputs are used. Results remain complete and independently readable.

Accuracy diagnostics use 10,000 samples per surface and seed 413. They report:

- Exact output/metadata preservation within each backend across versions and
  repeated/instrumented captures.
- Atlas pose-only and pose-plus-scale differences, using matching labeled
  landmarks and proper rotations that exclude reflections.
- Bidirectional, area-weighted surface distances, collapsed-face counts, and
  each specimen's vertices and triangle interiors relative to its original
  aligned surface. Sampled maxima are not exact Hausdorff bounds.
- Atlas-landmark barycentric transfer residuals against specimen landmarks,
  both in coordinate units and relative to centroid size. These landmarks also
  guide fitting, so this is an in-sample diagnostic, not independent validation.
- Baked colors versus original source colors at nearest surface locations,
  with alpha coverage, encoded RGB error, and CIELAB Delta E 76. The diagnostic
  decodes byte images and converts only sampled pixels to float64.
- Cross-backend colors at approximately matched atlas surface locations and
  PCA pairwise specimen-distance differences. UV pixels are not compared by
  image index when their surface parameterizations differ.

Neither backend is assumed to be scientific ground truth. Geometric proximity
and source-color fidelity do not establish anatomical correspondence quality.
The JSON report retains every specimen's measurements and the sampling method.

For landmark positions `p_i`, centroid size is
`sqrt(sum_i ||p_i - mean(p)||²)`. It is a measure of the spread of the whole
landmark configuration, not the shell's length. A distance reported as a
percentage of centroid size is `100 × distance / centroid_size`. Surface RMS
is `sqrt(mean(distance²))` over the sampled nearest-surface distances; each
direction weights samples by the area of its own source surface. Rigid and similarity
fits minimize matching landmark residuals, not nearest-surface error, so fitting
scale need not improve the surface-distance measurement.

For PCA pairwise distance vectors `a` (Native) and `b` (MorphoWeave), the relative
RMS difference is `sqrt(mean((a-b)²) / mean(a²))`. Pairwise distances avoid PCA
axis-sign ambiguity, but the backends still use different color feature spaces.
