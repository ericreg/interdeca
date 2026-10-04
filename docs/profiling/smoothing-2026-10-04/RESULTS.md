# Rust face-color smoothing: three-clam check

The production Python smoothing loop has been replaced by
`interdeca._native.smooth_face_colors` in `rust/src/color.rs`. The three-clam
comparison passed and showed a substantial speedup, so there is no Python
smoothing fallback.

## Sampling and smoothing

Median elapsed time to process all three baked textures, over three alternating
Python/Rust repeats with one smoothing pass and 1,000 selected faces:

| Atlas backend | Faces | Previous Python | Rust | Speedup |
|---|---:|---:|---:|---:|
| Native Atlas | 130,350 | 3.033 s | 0.0797 s | **38.0×** |
| MorphoWeave | 127,021 | 2.969 s | 0.0756 s | **39.3×** |

These timings include image decoding, UV sampling, smoothing, sample selection,
and result validation. They exclude atlas construction and analysis preparation.
Both implementations read the same three 512×512 baked textures and use the same
areas, adjacency order and sample indices. Files were warm; no cold-cache claim
is made. Individual repeats and binary identity are in `sampling-comparison.json`.

The gain combines compiled accumulation with avoiding unused final-pass results.
Adjacency is packed once into two arrays. For one pass, Rust computes only the
selected faces while including every valid neighbor's raw color. For multiple
passes, full-face intermediate buffers are reused before computing the selected
final outputs. Missing faces stay missing; valid black faces remain contributors.
Self is accumulated first, then neighbors in their existing order. No fast-math,
parallel reduction, mesh reordering or upstream MorphoWeave edit is involved.

## Equivalence and complete pipeline

For each backend and each of the three specimens, zero, one and three smoothing
passes produced **byte-identical sampled RGB arrays**, identical validity masks,
and matching specimen/sampling identities: 18 comparisons, maximum absolute
color error **0**. High-degree nonmanifold reductions can differ from NumPy's
reduction grouping by float64 rounding; a separate test checks that case with
`rtol=2e-15, atol=2e-15` rather than promising bitwise equivalence on every graph.

One fresh-process whole-pipeline check per implementation and backend also passed:

| Backend | Python session | Rust session | Identical output hashes |
|---|---:|---:|---:|
| Native Atlas | 35.51 s | 32.35 s | **33 / 33** |
| MorphoWeave | 35.21 s | 31.15 s | **33 / 33** |

These runs cover atlas generation, alignment, correspondence, Blender baking,
color sampling, PCA, reconstruction, display data, selection and export. They
exclude Slicer startup/rendering. The single-run session times are a smoke-check
measurement, not a repeated whole-pipeline performance estimate. The sampling
comparison above supplies the repeated performance gate. Analysis preparation
still takes about 12–13 seconds in this small dataset.

All **256 non-Slicer tests passed** (406 existing VTK deprecation warnings).
New coverage checks weighted diffusion over repeated passes, missing coverage,
valid black colors, disconnected faces, sample order, nonmanifold neighborhoods,
malformed graph rejection, array layout and adjacency serialization. Cargo
formatting and Git whitespace checks passed.

## Representation and reproduction

`AnalysisGeometry.adjacency` now contains a `FaceAdjacency(offsets, indices)`
record, with matching Pydantic schemas. This replaces one NumPy object per face.
Previously serialized analysis geometry must be rebuilt; mesh, texture and
presentation formats are unchanged. The old smoothing implementation is retrieved
from Git revision `584fa4f` only by the comparison tool, never by production code.

The three specimens are `UF_IZ_507781`, `UF_IZ_507782`, and `UF_IZ_507879`.
`verification.json` records input/source hashes and dependency versions.
Raw captures remain in:

- `build/profiling/smoothing-python-3-2026-10-04/`
- `build/profiling/smoothing-rust-3-2026-10-04/`

Rebuild the extension and repeat the focused comparison against the saved baked
reference textures with:

```sh
just rust-build
.venv/bin/python -m tools.profiling.smoothing \
  --reference-run build/profiling/smoothing-python-3-2026-10-04 \
  --reference-revision 584fa4f \
  --output build/profiling/smoothing-repeat.json
```

The harness uses a fresh Git-loaded reference, checks results before timing, and
requires the Rust median to be faster for both backends. It changes no production
functions and runs the comparisons sequentially to avoid timing contention.
