# Parry closest-surface prototype

This standalone PyO3 extension compares `parry3d-f64` 0.31.1 against the
`vtkStaticCellLocator` search formerly used by InterDeCA. The saved measurements
precede production integration (revision `c272472`). This standalone experiment
does not install into the main extension; the production implementation now lives
in `rust/src/spatial.rs` and has its own documented worker policy.

The prototype accepts an already prepared triangle mesh and projects a batch of
points onto its surface. It returns closest positions, **original input triangle
IDs**, and squared Euclidean distances. Positions remain float64 and input indices
remain int64 at the Python boundary. Parry uses checked u32 triangle indices internally.
There is no vertex welding, triangle removal, reordering of input geometry, or UV work.

## Build and run

From the repository root, with Rust and the development/profiling environment
installed (`uv sync --locked --inexact --group profiling`):

```sh
uv run --no-sync maturin develop --uv --release --locked \
  --manifest-path experimental_perf_tests/parry_spatial/Cargo.toml

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python experimental_perf_tests/parry_spatial/benchmark.py \
  --captures '/path/to/preserved/full-population/current' \
  --repeats 3 --threads 8 --output build/parry-spatial-results
```

The local default `--captures` is the preserved full-population archive under
`~/Library/Application Support/InterDeCA/full-population-2026-10-03/current`.
Each backend must have exactly one `*-baseline-1/atlas/interdeca_*` directory
containing `atlas.obj`, `atlas.mrk.json`, and 47 aligned OBJ/landmark pairs.
Inputs are read in place and never changed. The output directory must not exist.
The archive is not redistributed in Git; saved results retain its input hashes.

For a quick smoke run, add `--limit 1 --repeats 1`. To run only the small analytic
and invalid-input checks, use `--verify-only`. `--backend native` or
`--backend morphoweave` selects just one workload. The full run measures all 47
specimens **three times for each of four implementations, for both workloads**.

```python
from interdeca_parry_spatial import ParryIndex

index = ParryIndex(vertices_float64, triangles_int64, threads=1)
closest_points, triangle_ids, squared_distances = index.project(queries_float64)
```

Arrays must be aligned, C-contiguous, N by 3, and of the stated dtype. Meshes must
be nonempty, coordinates finite, and indices in range. Full production mesh
validation remains the caller's responsibility. This experimental interface
rejects strided, Fortran-only, and unaligned buffers rather than misreading them.
It snapshots NumPy inputs before releasing the GIL, including query points, so
mutable aliases do not race the Rust computation. Returned arrays own their data.
`threads=1` uses a serial iterator; larger values use a dedicated Rayon pool.

## What is measured

The Native workload reproduces the common-space landmark warp used by the saved
CLI pipeline. The MorphoWeave workload reproduces the float32-query VTK TPS warp
onto each aligned specimen. Both engines receive identical float64 arrays for a
given workload. The OBJ inputs come from the preserved earlier population run;
loading and warping are outside all reported times.

Four methods are compared:

- `vtk_existing`: the current Native Python loop, including per-query scratch objects.
- `vtk_reuse`: the same VTK search with those scratch containers reused.
- `parry_serial`: one batched Rust call, one worker.
- `parry_parallel`: the same batch with the requested worker count.

VTK is configured with its Sequential SMP backend. Parry's BVH construction is
serial in both modes. Build time includes input conversion/copies and construction;
parallel Parry also includes creating the thread pool. Query time includes the
NumPy snapshot, validation of query coordinates, search, and returning three NumPy
arrays. Every measurement starts with a new index. Index destruction and garbage
collection are outside the timing intervals for both engines.

Methods run sequentially, with their order rotated by specimen and repeat. The
reported medians are medians of **population totals**, not sums of per-file medians.
These are warm-process spatial-query measurements. The MorphoWeave workload does
not time upstream's `vtkPoints.GetPoint/SetPoint` loop or its float32 output;
all methods intentionally return identical output types for comparison.

## Correctness and interpretation

Every repeated measurement compares the VTK scratch-reuse control exactly with
the original VTK loop and serial Parry exactly with parallel Parry. VTK-versus-Parry
diagnostics record point errors, distance errors, and differing triangle IDs.
The first repeat's diagnostics are saved per specimen. Native's existing
barycentric reconstruction is applied outside timing to check whether choosing
a different triangle changes recovered original-space positions. MorphoWeave
workloads also report whether projected positions differ after float32 rounding.

At an edge or vertex, several triangles can be equally close. Engines can return
different triangle IDs for essentially identical points. An actual replacement
must consider these ties and per-corner texture seams; this prototype does not
claim bitwise equivalence or a complete production compatibility audit.

Small analytic checks cover triangle interiors, edges, vertices, squared distance,
closed-mesh interior projection, preservation of triangle IDs, empty query batches,
parallel equality, and invalid input rejection. No application tests are added.

See [results/RESULTS.md](results/RESULTS.md) for the measured comparison and
[results/results.json](results/results.json) for versions, hashes, per-specimen
wall/CPU times, thread budgets, and agreement evidence. End-to-end correspondence
also includes warping, validation, and reconstruction, so its speedup will be less
than the isolated search speedup. Texture baking and ray casting are not measured.
