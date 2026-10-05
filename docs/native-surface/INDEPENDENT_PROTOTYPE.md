# Independent Rust surface processing — qualification in progress

**This is an experimental implementation, not a completed Blender replacement.**
Production workflows still use Blender. The release gates below are unchanged,
including the 512×512 requirement. No xatlas dependency or Blender source code is
used by these kernels. Blender is used only as a black-box comparison oracle.

## Implemented operations

`interdeca.surface.MeshProcessing` provides:

- `clean`: radius-based welding, duplicate/collapsed triangle removal, removal of
  unused vertices, and original-to-output vertex/face maps. Removed entries use
  `-1`; duplicate faces map to the retained face. The first surviving input vertex
  supplies a cluster's coordinates. Proximity is deliberately nontransitive.
- `clean_atlas`: cleanup before establishing correspondence; landmarks are retained.
- `uv_layout`: intrinsic triangle unfolding, guarded least-squares conformal
  relaxation, minimum-area chart orientation, and deterministic rectangle packing.
- `prepare_atlas_uv`: attaches per-corner UVs while preserving the exact vertex and
  face arrays and their order.

`interdeca.surface.TextureBaking` provides:

- `bake_pixels`: parallel CPU selected-to-active color transfer through a Parry BVH.
- `bake_specimen_texture`: validates aligned specimen identity/landmarks, decodes
  the source image once, bakes, encodes a PNG, and publishes it atomically.

These APIs require no Blender process. They are independent of Slicer, Qt, and
execution managers. The existing Maturin extension packages the new Rust modules;
there is no second native library or extra runtime dependency. Existing wheel CI
and `just rust-build` build these modules with the rest of the extension. Source
installs still require the normal Rust toolchain; a prebuilt wheel does not.

## Algorithms and contracts

Cleanup searches neighboring spatial-grid cells within the merge radius. Only a
surviving vertex can represent later vertices. This avoids the long-range merging
that a transitive connected-component interpretation would allow. **This policy
is not yet equivalent to Blender's representative selection.** Do not replace
cleanup in an established workflow on the strength of the Native-only comparison.
Cleanup discards old UVs when the radius is nonzero; radius zero returns the
original mesh, including its UVs.

UV chart growth unfolds each new triangle around a shared edge using its physical
edge lengths. Closing a loop may introduce strain; positive orientation,
bounded relative edge length, and triangle-interior non-overlap are required.
A conformal relaxation then minimizes the two Cauchy–Riemann residuals per
triangle using matrix-free preconditioned conjugate gradients. Two pinned vertices
remove the trivial zero solution. A relaxed chart is accepted only if it still
passes the orientation, stretch, and overlap checks; otherwise its original
unfolding is retained. Each chart is rotated to its minimum-area bounding
rectangle before deterministic MaxRects packing. Spacing reserves the bake margin
plus one filtering pixel on each side. Insufficient padding space is an error.

The baker assigns UV triangles to 32×32 tiles. Pixel centers determine triangle
coverage; wholly subpixel triangles do not acquire invented center samples. Each
tile has one writer. Rays begin at the target point plus the configured extrusion
along its face normal and travel in the opposite direction. Zero maximum ray
distance means no distance cutoff. The source triangle's barycentric coordinates
interpolate per-corner UVs; repeated, bilinear sampling reads the source image.
RGBA alpha records bake coverage, rather than source-material transparency.
Output arrays are owned and read-only. Inputs are copied before releasing the GIL.

Analytical Blender fixtures established two details that differ from earlier
assumptions: byte images filter encoded RGB (the black/white midpoint is 128),
and collapsed target faces use a +Z fallback normal. Padding is still an
approximation: it propagates covered samples and can follow adjacent triangles,
but does not yet reproduce Blender's fractional boundary alpha and filtering.

## Qualification gates

Both Native and MorphoWeave must pass on all 47 specimens at **512 and 2048**:

| Quantity | Required limit |
|---|---:|
| Mean RGB error on joint coverage | ≤ 1/255 |
| 95th percentile of sample maximum-channel error | ≤ 5/255 |
| Coverage lost relative to reference | ≤ 0.1 percentage points |
| Cleanup connectivity | Identical modulo a bijective vertex-index mapping |
| Cleanup coordinate error | ≤ 1e-6 × input bounding-box diagonal |

UV comparisons sample the **same physical surface points**, even when UV layouts
differ. The harness reports both raw RGB with nearest-alpha coverage and the
production color sampler's alpha-weighted interpolation with coverage ≥0.99.
Geometry/UV order, non-overlap, determinism, and analytical boundary cases have
focused tests. Three fresh-process repetitions record per-specimen kernel time,
input/reference/output hashes, extension and implementation hashes, whole-process
elapsed time, and peak RSS. Timings from reused reference images do **not** measure
Blender performance and do not establish the whole-workflow speed or memory gate.

Passing the isolated baker on Blender UVs is insufficient. Custom UV quality,
cleanup, downstream color features/PCA, complete workflow runtime, and peak memory
must all be qualified before production integration. The agreed runtime criterion
is no regression; peak memory may grow by at most 10% against an equivalent
Blender workflow. Those whole-workflow gates have not been demonstrated.

## Reproducing the work

From the repository root, with the development environment installed:

```sh
just rust-build
.venv/bin/python -m pytest tests/test_native_surface.py -q
PYTHONPATH=. .venv/bin/python experimental_perf_tests/native_surface/fixtures.py
PYTHONPATH=. .venv/bin/python experimental_perf_tests/native_surface/cleanup_fixtures.py
PYTHONPATH=. .venv/bin/python experimental_perf_tests/native_surface/meshcheck.py --reference
PYTHONPATH=. .venv/bin/python experimental_perf_tests/native_surface/benchmark.py --backend native
```

The fixture/reference commands launch the installed Blender. In this macOS
sandbox it must run outside the sandbox because Metal initialization crashes
inside it. The Rust operations themselves do not need that exception.

The population harness defaults to the preserved 2026-10-04 profiling captures
and the 47 local specimen inputs. Override `--captures`, `--inputs`, and `--output`
for another checkout. Archives are read with the existing hash-checking profiling
reader; original files are not modified. Before running the repetition matrix,
create the two 2048 reference sets:

```sh
PYTHONPATH=. .venv/bin/python experimental_perf_tests/native_surface/benchmark.py --backend native --resolution 2048 --refresh-reference --output build/native-surface/native-2048
PYTHONPATH=. .venv/bin/python experimental_perf_tests/native_surface/benchmark.py --backend morphoweave --resolution 2048 --refresh-reference --output build/native-surface/morphoweave-2048
PYTHONPATH=. .venv/bin/python experimental_perf_tests/native_surface/repeat.py
```

The repeat driver runs sequential, fresh interpreters for all 47 specimens,
three times for each backend/resolution combination. It preserves reports when
quality gates fail and returns exit status **2** for a quality failure. Other
nonzero exits are execution failures. Peak RSS collection currently targets
macOS/Linux. Generated images and large reports remain under ignored `build/`.

For a custom-layout comparison, first use `meshcheck.py` at the desired resolution
and pass its `.npz` file as `benchmark.py --layout ...`. The layout must retain
identical target face indexing. A passing baker report without `--layout` cannot
qualify the custom UV generator.

## Remaining work before removing Blender

1. Improve custom chart generation/packing enough to meet the 512 surface-color
   gates. Conformal relaxation is implemented, but the captured atlas still has
   substantial fragmentation and wasted area.
2. Reproduce cleanup representative selection on both backends and fixtures,
   then qualify actual pre-cleanup workflow atlases. Experiments with balanced
   spatial ordering matched the captured Native atlas exactly but still differed
   on MorphoWeave and chain fixtures; those unqualified variants were removed
   from the production kernel.
3. Resolve 512 baker boundary/filtering differences. Keep the 2048 checks as
   regression gates while changing rasterization or padding.
4. Run complete Native/MorphoWeave workflow and downstream color/PCA comparisons,
   plus equivalent fresh-process runtime/memory measurements.
5. Only after all gates pass, switch the workflow registry, remove Blender from
   normal CLI/UI setup, retain deprecated Blender configuration for the agreed
   transition release, and verify platform wheel installation without Blender.

See `results.json` for the recorded qualification snapshot when present. Failures
are retained as evidence rather than weakened thresholds.

## Recorded validation

Three fresh-process runs produced identical pixel hashes for every specimen.
For the **isolated baker using captured Blender UVs**, primary gate counts were:

| Backend | 512×512, each repetition | 2048×2048, each repetition |
|---|---:|---:|
| Native | 34/47 | 47/47 |
| MorphoWeave | 0/47 | 47/47 |

The production sampler calculation passed 36/47 Native and 44/47 MorphoWeave
specimens at 512, and all specimens at 2048. Both calculations are retained;
neither establishes a passing 512 release gate.

The final custom layouts occupy 11.39% of the Native texture and 20.17% of the
MorphoWeave texture at 512, compared with 29.22% and 39.90% for the captured
Blender layouts. Two-specimen custom-layout pilots fail on both backends, so
those pilots were not expanded into a claim of population qualification.
Cleanup also fails the saved-atlas comparison on both backends.

The final non-Slicer suite passes **285 tests**, including 29 focused native
surface tests. A CPython 3.13 macOS arm64 wheel was built, unpacked into an
isolated import location, and exercised for cleanup, UV generation, and baking
without invoking Blender. Other platforms have not been smoke-tested here.
`cargo fmt --check` passes; Clippy reports only the existing argument-count
warning in `color.rs`. Final lint-only Rust edits were checked against the
measured build's bake hashes on two specimens, with identical output.
