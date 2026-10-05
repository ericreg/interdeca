# Native surface processing

Atlas cleanup, Smart UV Project, and selected-to-active color baking now run in
InterDeCA's Rust extension. CLI and Slicer workflows require no Blender executable.
`MeshProcessing` and `TextureBaking` are the public Python namespaces; their core
operations are independent of Slicer, Qt, and execution managers.

The implementation derives from Blender 5.2.1 LTS, commit
[`9e2066aef7ef7e20c142ad7bd3303138a4304c93`](https://github.com/blender/blender/tree/9e2066aef7ef7e20c142ad7bd3303138a4304c93),
and its pinned OpenColorIO output curve. `rust/BLENDER_SOURCE.json` and
`rust/OCIO_SOURCE.json` record source paths, checksums, translated symbols, and
specializations. The package and native extension are GPL-3.0-or-later; original
source notices and compatible licenses are retained in `THIRD_PARTY_NOTICES.md`,
`LICENSE`, and `LICENSES/`.

## Public contracts

- `MeshProcessing.clean/clean_atlas` performs float32 KD-tree clustering,
  representative selection, triangle welding, duplicate/collapsed-face handling,
  and loose-vertex removal. Cleanup precedes correspondence. Nonzero cleanup
  discards old UVs; zero-distance cleanup preserves the original mesh record.
  Original-to-output maps use `-1` for removed elements.
- `MeshProcessing.uv_layout/prepare_atlas_uv` projects normal groups, traverses UV
  islands, rotates minimum-area rectangles, and packs them into one tile. These
  operations preserve original vertices, triangles, their ordering, and per-corner
  associations. UV arithmetic follows Blender's float32 storage.
- `TextureBaking.bake_pixels/bake_specimen_texture` uses translated scan conversion,
  flat normals, ray setup, BVH traversal, watertight intersections, repeat bilinear
  image sampling, byte alpha association, sRGB conversion, and adjacent-face
  texture margins. Baking preserves geometry and indexing. Alpha records coverage;
  zero maximum ray distance disables the cutoff. File baking decodes once and
  publishes a PNG atomically after successful processing.

The scope is triangular meshes, identity object transforms, a fresh single-tile
Smart UV map, and RGB/RGBA8 sRGB color-only transfer. This is the functionality
InterDeCA uses, rather than a general Blender scene renderer. Geometry
correspondence continues to use Parry; the baker uses the translated Blender BVH
because equal-distance ray hits depend on its triangle traversal order.

The explicit `Blender` Python API remains available as a reference oracle. Legacy
`--blender` and serialized `blender` settings are accepted and ignored by normal
workflows for this transition release. Slicer exposes the numerical controls under
**Surface processing (cleanup, UV, bake)**.

## Qualification

Numerical equivalence is the acceptance criterion; downstream byte equality is
recorded as a diagnostic. Cleanup connectivity and winding must agree modulo a
bijective vertex map, with coordinate error at most 1e-6 of the input bounding-box
diagonal. UVs on the captured atlases match every corner exactly.

Population comparisons use all 47 specimens on both atlas backends, at 512 and
2048 pixels, in three fresh processes per condition. Both raw and production
samplers require mean RGB error at most 1/255, p95 maximum-channel error at most
5/255, and coverage loss at most 0.1 percentage points. Repeated image hashes must
be deterministic. Reused images provide quality references, not Blender runtime
measurements.

Full workflows retain pre/post-cleanup geometry, UVs, sampled RGB, processed Lab,
feature matrices, and PCA output. Identity arrays and validity masks remain exact.
The comparison accepts RGB mean/max error of 1/255 and 5/255, Lab/features RMS/max
error of 0.05 and 1, sign-aligned PCA score relative RMS error of 1e-4, component
maximum error of 1e-3, and explained-variance maximum error of 1e-4. PCA score RMS
is divided by the reference score RMS, so that criterion is independent of score
units. Saved reports include the limits and every measured difference.

Complete-workflow comparisons separately require no runtime regression and peak
process-tree RSS growth of at most 10%. Local measurements are host-specific;
Linux/Windows and other architectures require the wheel CI matrix. Slicer form
changes have source-level verification on this host, without an interactive Slicer
run. Automated tests remain deferred at the user's request.

Final measurements and provenance are retained in `completion-results.json` and
`completion-population.json`. Earlier `source-results.json`, `results.json`, and
[prototype notes](INDEPENDENT_PROTOTYPE.md) are historical checkpoints.

## Final local results (2026-10-05)

All 564 specimen bakes passed both samplers using the final extension. Pixels
were deterministic across repetitions. Worst mean RGB error was 1.474e-7;
worst p95 channel error and coverage loss were both zero.

| Complete workflow | Blender seconds | Native seconds | Peak RSS reduction |
|---|---:|---:|---:|
| Native atlas backend | 213.19 | 136.76 | 14.85% |
| MorphoWeave atlas backend | 281.34 | 214.86 | 10.61% |

Both final workflow runs match cleanup vertices, triangles, and every UV corner
exactly. Validity masks are unchanged. Maximum sampled/processed RGB differences
are 0.001138 (Native) and 0.001335 (MorphoWeave), below one byte step. Sign-aligned
PCA score relative RMS errors are 2.271e-6 and 1.894e-6. Every numerical, runtime,
and memory gate passes.

These full-workflow timings pair one fresh final native run per backend with its
saved Blender reference; they are local measurements, not repeated statistical
performance estimates. The baker matrix has three fresh repetitions per condition.
Both final native workflows ran with `BLENDER_EXECUTABLE=/nonexistent-blender`.
Local macOS ARM64 Python 3.13/3.14 wheels and a corresponding source archive build;
a Python 3.13 extracted-wheel smoke check exercises cleanup, UV mapping and baking.

## Reproduction

Build with `just rust-build`. For the offline local development environment:

```sh
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop --skip-install --release --locked --offline --manifest-path rust/Cargo.toml
PYTHONPATH=. .venv/bin/python experimental_perf_tests/native_surface/repeat.py --repetitions 3 --output build/native-surface/completion-repeated
PYTHONPATH=. .venv/bin/python experimental_perf_tests/native_surface/workflow.py --backend native --native --output build/native-surface/completion-workflow-native
PYTHONPATH=. .venv/bin/python experimental_perf_tests/native_surface/workflow.py --backend native --output build/native-surface/completion-reference-native
PYTHONPATH=. .venv/bin/python experimental_perf_tests/native_surface/workflow_compare.py --reference build/native-surface/completion-reference-native --candidate build/native-surface/completion-workflow-native --output build/native-surface/completion-comparison-native.json
```

Repeat the workflow commands with `--backend morphoweave` and distinct outputs.
Only reference conditions launch Blender. Run measured conditions sequentially.
Build wheels with `just rust-wheel` and corresponding source with `just rust-sdist`.
Local wheel/source inspection verifies the extension, Python APIs, translated Rust
sources, Cargo/Python manifests and locks, client source, provenance, and licenses.
Do not modify the upstream reference checkout or `../3d_color_package/`.
