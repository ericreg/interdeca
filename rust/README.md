# Native mesh and texture operations

This crate builds the private `interdeca._native` module inside the main Python
package. `MeshIO` and `TextureIO` are the public interfaces. There is no Python
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
