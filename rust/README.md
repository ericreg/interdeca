# Native mesh, texture, and spatial operations

This crate builds the private `interdeca._native` module inside the main Python
package. `MeshIO`, `TextureIO`, and `Spatial` are the public interfaces. There is no Python
OBJ parser or formatter fallback, and production OBJ loading always validates.

The native reader preserves vertex order, triangle order, corner UVs, negative
indices, and float64 values. LPS input is converted directly into the final RAS
vertex buffer. The writer snapshots caller-owned arrays, validates them before
opening the destination, and streams through a 256 KiB buffer. Rust's round-trip
decimal formatting preserves float64 values, including signed zero; serialized
text can differ from the previous Python `.17g` formatting.

Texture kernels normalize byte images, validate and quantize unit channels, and
evaluate sRGB transfer curves. Byte normalization and ties-to-even quantization
preserve pixel values exactly. Scalar power implementations can differ from
NumPy by a few rounding bits across platforms; curve checks use `rtol=2e-15`.
Face-color smoothing uses packed adjacency rows shared across specimens. It
accumulates each face and its valid neighbors with atlas-area weights, keeping
invalid faces invalid and reading a separate previous-pass buffer. The final
pass calculates only requested samples; intermediate passes cover the whole
graph. Zero passes select raw covered colors. Neighbor and sample order remain
unchanged; high-degree reductions may differ from NumPy by float64 rounding.
The private smoothing entry point requires contiguous arrays and C-order RGB.
Analysis adjacency is now stored/serialized as `offsets` and `indices`, rather
than one array per face; previously serialized analysis records must be rebuilt.

Pillow continues to handle image codecs and VTK continues to handle other mesh
formats and polydata. Landmark schemas and small file operations remain Python.

Returned arrays are C-contiguous and read-only. Rust allocations are owned by
NumPy base capsules; independence from caller storage does not require NumPy's
OWNDATA flag. The file reader and writer release the GIL over Rust-owned data.
Texture operations hold it while borrowing Python arrays; callers must not
mutate those inputs concurrently. No fast-math or parallel numerical reductions
are enabled.

Public adapters copy unaligned NumPy inputs before borrowing them; ordinary
aligned strided views need no normalization copy. Private native entry points
reject unaligned nonempty arrays. Transfer curves preserve empty shapes and
accept up to 32 dimensions, matching rust-numpy's view limit.

From the repository root, with Rust, a platform linker, `uv`, and `just` installed:

```sh
just rust-setup  # Sync dependencies and build the production extension
just rust-build  # Explicit release rebuild after Rust edits
just rust-check  # Release rebuild, formatting, and non-Slicer tests
just rust-dev    # Optional debug build; unsuitable for performance measurements
just rust-wheel  # Build the combined Python/Rust wheel
```

The recipes operate on the application's `.venv`. Python edits are editable;
Rust edits require rebuilding and restarting processes that imported the old
extension. `uv sync` builds through Maturin, and Cargo reuses unchanged artifacts.
The repository's lockfiles pin Python and Rust dependencies. Release builds keep
line tables for native allocation attribution while retaining optimization.

The parser handles ASCII numeric OBJ syntax and UTF-8 text/comments. It does not
emulate Python's acceptance of Unicode decimal numerals. Polygon triangulation,
materials, normal processing, and a general OBJ scene representation are outside
this deliberately narrow mesh contract.

## Surface projection

`Spatial.index(mesh)` snapshots geometry into an owned `parry3d-f64` 0.31.1
triangle BVH. `Spatial.project(index, points)` returns read-only closest points,
original triangle IDs, and squared distances in the same order as the queries.
Inputs and output positions use float64; faces use int64 at the Python boundary
and checked u32 internally. No triangles or vertices are reordered, merged, or
removed. Degenerate correspondence triangles remain present. Interior points
project onto the surface rather than returning the input as a solid-volume hit.
Nonfinite inputs, invalid indices, and failed/overflowing projections raise errors.

Native correspondence uses this index in common warped space and retains its
existing barycentric reconstruction. The MorphoWeave private host substitutes
batched surface queries and unsigned coordinate-distance checks while retaining
upstream TPS and float32 input/output rounding. Its separate nearest-vertex mean
algorithm stays upstream. The vendored source remains unmodified and hash-checked;
these local adapters would need upstream adoption to benefit standalone MorphoWeave.
There is no production VTK closest-surface fallback. The offline accuracy tool
deliberately retains VTK as an independent comparison implementation.

At shared edges and vertices, different incident triangles can have the same
closest point. Parry may choose a different triangle from VTK. The prototype's
47-specimen checks found only rounding-level changes in recovered geometry and
identical float32 projections, but do not promise identical face IDs at UV seams.
Independent point queries preserve order and serial/parallel numerical results;
there is no parallel floating-point reduction or fast-math change.

### Worker selection

- Default process-wide limit: `min(8, std::thread::available_parallelism())`.
- For N queries, maximum active chunks: `min(limit, max(1, N // 8192))`;
  zero queries use zero chunks. Thus fewer than 16,384 queries run serially.
- A single lazy Rayon pool is shared across indices. The index build is serial.
  Repeated specimens reuse the pool, and concurrent callers share its fixed limit.
- `INTERDECA_SPATIAL_THREADS` overrides eight with a positive integer, capped by
  available CPUs. It is read once per process; restart after changing it. Invalid
  settings raise an error rather than silently selecting an arbitrary budget.
- `Spatial.workers_for(N)` exposes the selection. This is a concurrency bound,
  not a promise that the OS runs that many workers simultaneously.

The 8-worker prototype beat serial search on the 47-clam workloads. Eight is a
conservative default, not an optimum demonstrated for every CPU or mesh. The
8,192-query grain avoids scheduling overhead on small landmark batches; it is a
policy threshold, not an empirically established universal crossover. More than
eight workers may help larger batches on machines with more available CPUs;
benchmark before raising the cap. Lower the limit when several application
processes share the machine. The setting does not control Blender, SciPy's
nearest-vertex mean search, BLAS, or unrelated VTK operations.

Geometry and queries are copied before releasing the GIL, protecting the Rust
work from writable Python aliases. The pool and immutable index can serve
concurrent calls; returned NumPy arrays own their Rust allocations.
