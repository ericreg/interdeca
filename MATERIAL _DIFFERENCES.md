# Material differences in small-clams output

Comparison date: 2026-09-30. Current implementation examined:
`5929c77a686a233a189c0436d55db682ec65fdf9`
(`fix: preserve projected faces and verify Slicer CLI parity`).

The current CLI and Slicer outputs agree with each other under the end-to-end
test's assertions, but they do not reproduce the saved `3d_color_package`
baseline. Aligned input geometry is effectively unchanged after removing pose.
Atlas topology, resampled geometry, UVs, texture resolution and channels, and
texture averaging differ materially.

This document records the comparison before further implementation changes. It
does not accept the current output as a replacement scientific baseline or
establish that every difference is an improvement.

## Inputs, outputs, and scope

Only these three specimens from
`../3d_color_package/tests/data/small_clams` were compared:

- `UF_IZ_507781`
- `UF_IZ_507782`
- `UF_IZ_507879`

The reference was the reviewed baseline at
`../3d_color_package/tests/output/slicer_small_clams/slicer-5.12.4_blender-5.2.1`.
Its README records generation on 2026-09-24 and acceptance on 2026-09-25. Both
the reference and current run used Slicer 5.12.4 and Blender 5.2.1 LTS on macOS.
Matching these versions does not establish that all parameters or Python
dependencies matched.

Current artifacts came from the successful
`tests/end_to_end/test_small_clams.py::test_small_clams_slicer_cli_parity` run:

| Artifact | Location relative to the pytest scratch directory |
| --- | --- |
| CLI atlas | `cli-atlas/interdeca_20260930_193244_dd761e66` |
| Slicer atlas | `slicer/atlas/interdeca_20260930_193347_0cb5e567` |
| CLI actions and results | `cli/events.jsonl` and referenced numerical artifacts |
| Slicer results | `slicer/presentations.json` and scene snapshots |

The scratch directory was
`/private/var/folders/8l/9twf6s3s12sdpcn57c52rpcm0000gn/T/pytest-of-eric/pytest-18/test_small_clams_slicer_cli_pa0`.
It is temporary and may be removed by pytest or operating-system cleanup.
The measurements below preserve the findings independently of that directory.
The CLI atlas represents the current implementation in the legacy comparison,
because the e2e test had already checked agreement between the two current hosts.

This investigation read saved artifacts using the uv environment. It did not
rerun either atlas pipeline, modify the reference package, or run the full
clams or mussels datasets.

## Comparison method

Landmark readers honored RAS/LPS metadata before comparison. Legacy atlas labels
are generic, so its 48 landmarks were matched to current landmarks by stored row
order. The aligned specimen landmark labels already agree between packages.

A least-squares rigid fit removed rotation and translation, with reflections
forbidden and no change of scale. Aligned input meshes used each specimen's
landmark fit. Atlas and resampled surface comparisons used one shared fit of
the 48 atlas landmarks. All distances below are in native mesh coordinate
units; no physical-unit conversion to millimeters was assumed.

Different representations require different comparisons:

- Aligned meshes were compared in triangle-corner order, using
  `vertices[faces]`. This avoids confusing duplicated legacy vertices with a
  change of surface geometry. Faces and corners were not sorted.
- Atlas and resampled surfaces have different topology. For each comparison,
  5,000 points were drawn uniformly by surface area in each direction with
  NumPy random seed 413. VTK's `vtkStaticCellLocator` measured the closest point
  on the other surface's triangles. These are sampled distances, not exhaustive
  Hausdorff distances or proof of anatomical correspondence.
- Projection fidelity used 10,000 evenly indexed points from each resampled
  mesh's lexicographically sorted unique vertices, measuring distance to that
  package's own aligned native surface.
- Surface colors were compared at 5,000 area-weighted locations on the current
  atlas. The nearest legacy triangle after rigid alignment supplied its UV
  coordinates through barycentric interpolation. Both images were sampled
  bilinearly with alpha weighting; only locations with alpha at least 0.99 in
  both images were retained. There were 4,963 valid locations for each image pair.

RMS means the square root of the average squared distance. P95 is the distance
below which 95% of sampled values fall. Reported maxima are sample maxima.

## Aligned input meshes and landmarks

Aligned meshes have the same triangle counts, triangle-corner ordering, and
exactly matching per-corner UVs. The legacy OBJ representation has one vertex
per triangle corner; the current representation shares vertices. Consequently,
vertex counts and face index arrays differ even where the geometry agrees.

| Specimen | Legacy vertices | Current vertices | Triangles in each | Corner-position RMS after rigid fit |
| --- | ---: | ---: | ---: | ---: |
| `UF_IZ_507781` | 1,633,320 | 272,222 | 544,440 | 3.64e-10 |
| `UF_IZ_507782` | 1,005,642 | 167,607 | 335,214 | 9.62e-11 |
| `UF_IZ_507879` | 795,366 | 132,563 | 265,122 | 3.63e-10 |

These differences are consistent with equivalent aligned surfaces at the
precision of the saved files, rather than material deformation. None of these
aligned meshes contains degenerate triangles under the criterion below.

Each specimen has 48 landmarks. After its rigid fit, landmark RMS differences
are below 9e-18. The atlas landmarks have a residual RMS of 4.14e-7 and a maximum
of 6.96e-7. An additional similarity fit gives an atlas scale factor of
1.000000312, so a material overall scale change does not explain the outputs.

The legacy atlas uses labels such as `Atlas Landmarks-1`; the current atlas
retains anatomical labels such as `beak` and `hinge_anterior`. This label change
applies to the atlas, not the aligned specimen landmarks, whose labels match.

## Atlas geometry and topology

| Measurement | Legacy atlas | Current atlas |
| --- | ---: | ---: |
| Vertices in the UV OBJ | 59,200 | 64,490 |
| Triangles | 119,254 | 130,350 |
| Surface area, native units squared | 0.00119990006 | 0.00121604517 |
| Degenerate triangles | 0 | 0 |

The current atlas has 8.94% more vertices, 9.30% more triangles, and approximately
1.35% more surface area. Its connectivity and per-corner UV arrays differ from
the legacy atlas, so same-index vertex subtraction is not meaningful.

| Sampled distance direction | RMS | P95 | Maximum |
| --- | ---: | ---: | ---: |
| Legacy atlas to current atlas | 4.08e-5 | 6.83e-5 | 5.65e-4 |
| Current atlas to legacy atlas | 3.82e-5 | 7.62e-5 | 3.11e-4 |

The legacy atlas's saved bounding-box diagonal is 0.0388997. Relative to that
size, the sampled RMS difference is approximately 0.1%, with larger localized
differences. A small surface distance does not establish that vertices identify
the same anatomical locations.

Template selection is a possible contributor: the legacy pre-UV atlas has
335,214 triangles, matching the input topology size of `UF_IZ_507782`, whereas
the current reference selection identifies `UF_IZ_507781` (544,440 input
triangles). The legacy template identity is inferred from its topology size;
the saved baseline lacks the full run manifest needed to confirm the original
selection and parameters. This observation does not isolate the cause of the
final atlas difference.

## Resampled geometry and collapsed triangles

Every legacy resampled OBJ contains 357,762 vertices and 119,254 triangles.
Every current resampled OBJ contains 64,490 vertices and 130,350 triangles.
The legacy vertex duplication and changed atlas topology account for much of
the count difference, but the surfaces themselves also differ.

| Specimen | Surface RMS, legacy to current | Surface RMS, current to legacy | Legacy vertex-to-native RMS | Current vertex-to-native RMS |
| --- | ---: | ---: | ---: | ---: |
| `UF_IZ_507781` | 5.91e-5 | 5.88e-5 | 2.85e-5 | 4.41e-19 |
| `UF_IZ_507782` | 3.96e-5 | 4.52e-5 | 2.37e-5 | 4.35e-19 |
| `UF_IZ_507879` | 2.71e-5 | 3.06e-5 | 1.76e-5 | 3.93e-19 |

The available legacy implementation projects onto a warped surface and then
applies a separately fitted reverse thin-plate spline. Such a reverse fit need
not be the exact inverse of the forward warp. The current
`Geometry.resample_specimen` instead finds the projected triangle and its
barycentric weights, then applies those weights to the original triangle.
This places the output vertices on the original piecewise-linear surface.
The measured vertex-to-native distances are consistent with that distinction.
Source inspection explains a plausible mechanism; it does not prove the
available legacy source is exactly the version that generated the baseline.

Both outputs have collapsed triangles in individual correspondence meshes:

| Specimen | Legacy count | Legacy fraction | Current count | Current fraction |
| --- | ---: | ---: | ---: | ---: |
| `UF_IZ_507781` | 196 | 0.164% | 326 | 0.250% |
| `UF_IZ_507782` | 34 | 0.0285% | 45 | 0.0345% |
| `UF_IZ_507879` | 1,311 | 1.099% | 2,357 | 1.808% |

A triangle was counted as degenerate when, for edges `e1` and `e2` from one
corner, `norm(cross(e1, e2)) <= 32 * eps * norm(e1) * norm(e2)`, using
float64 machine epsilon. This includes zero-length edges and numerically
collinear corners.

The current correspondence contract retains collapsed faces so atlas triangle
indices and UVs remain paired across specimens. Native input meshes and the
shared analysis atlas remain strict. The higher collapsed-face fractions are
a measured difference, not evidence that correspondence quality improved.
Vertices lying on the native surface also do not imply that every resampled
triangle interior lies on it or that anatomical correspondence is correct.

## Texture resolution, UVs, and colors

All three specimen textures and the average differ in dimensions and channels:

| Property | Legacy | Current e2e run |
| --- | --- | --- |
| Resolution | 2048 x 2048 | 512 x 512 |
| Channels | RGB | RGBA |
| Atlas UV minimum | (0.000982, 0.000982) | (0, 0) |
| Atlas UV maximum | (0.999018, 0.871683) | approximately (1, 1) |

The current pytest deliberately passes a bake size of 512. This is a test
configuration difference, not a maximum supported resolution. Alpha explicitly
separates unpainted pixels from measured black in current textures. Legacy RGB
files contain no equivalent alpha signal.

Different UV layouts mean that the same pixel position can represent different
surface locations. Whole-image averages, nonblack pixel fractions, and direct
pixel differences therefore cannot establish a surface-color change or better
coverage. The surface-based diagnostic instead produced:

| Texture | Mean absolute RGB channel difference, 0-255 | RMS RGB channel difference, 0-255 |
| --- | ---: | ---: |
| `UF_IZ_507781` | 5.56 | 13.27 |
| `UF_IZ_507782` | 7.60 | 15.85 |
| `UF_IZ_507879` | 7.58 | 21.69 |
| `average_texture` | 10.68 | 15.22 |

These differences are over the retained surface samples and all three encoded
RGB channels. They combine changes in atlas geometry, approximate spatial
correspondence, UV packing, resolution, and texture processing. They are not
perceptual color-distance scores or a controlled measurement of baking error.

## Average texture mathematics

This is a confirmed algorithmic difference, independently verified against
each package's own three baked images:

- Legacy: compute the arithmetic mean of encoded RGB bytes, then cast to
  `uint8`, truncating fractional values. Black background participates in the
  arithmetic like any other RGB value.
- Current: decode sRGB to linear light, weight each specimen by alpha, divide
  by the sum of alpha weights, and encode back to sRGB. Uncovered pixels remain
  zero. Output alpha is the maximum input alpha, and bytes are rounded.

Recomputing these formulas reproduced each saved average image exactly, with
zero differing channels. Linear-light averaging can produce a brighter result
than averaging encoded RGB. That explains a systematic difference in the mean
image, but does not explain every difference in the individual baked textures.

## Output layout and validation coverage

The legacy baseline contains 28 non-metadata artifacts, including parallel
OBJ/PLY representations, MTL files, a pre-UV atlas, and a VTP scene export. The
current atlas directory has 19 files. A smaller file count is an output-contract
change, not by itself missing scientific data.

| Logical output | Legacy relative path | Current atlas relative path |
| --- | --- | --- |
| Atlas mesh with UVs | `colorAnalysis/atlasModelUV.obj` | `atlas.obj` |
| Atlas landmarks | `colorAnalysis/atlasLM.mrk.json` | `atlas.mrk.json` |
| Aligned mesh | `ATLAS/alignedModels/<id>_align.obj` | `aligned/<id>.obj` |
| Aligned landmarks | `ATLAS/alignedLMs/<id>_align.mrk.json` | `aligned/<id>.mrk.json` |
| Resampled mesh with UVs | `colorAnalysis/resampledOBJ_withUV/<id>_resampled.obj` | `resampled/<id>.obj` |
| Resampled landmarks | No separate corresponding file | `resampled/<id>.mrk.json` |
| Specimen and average textures | `colorAnalysis/atlasTextures/` | `textures/` |

The current e2e test builds both atlases from raw inputs and then compares color
analysis, PCA scores, axis changes, reconstruction, specimen display, selection,
export, and actual Slicer MRML presentation. It proves agreement between the
two current hosts under its assertions. It does not compare either host with
the legacy baseline and cannot detect a numerical change shared by both hosts.

The saved legacy baseline has no comparable PCA scores, reconstructions,
selection results, or exported selections. Legacy parity for those outputs
remains unmeasured. Its complete parameter manifest is also absent, preventing
a claim that this was an otherwise identical-configuration comparison.

Before accepting numerical parity with the legacy workflow, a separate
comparison would need matched settings, explicit treatment of the changed
topology and correspondence algorithm, and reviewed expectations for texture
averaging. Simply replacing the expected files would conceal these choices.
No such baseline replacement or implementation change was made for this report.

## Source references

- [Current e2e test](tests/end_to_end/test_small_clams.py) and
  [test contract](tests/end_to_end/README.md).
- [Atlas reference selection and construction](src/interdeca/atlas.py).
- [Surface resampling](src/interdeca/geometry.py),
  `Geometry.resample_specimen`.
- [UV generation and baking](src/interdeca/_blender_worker.py).
- [Texture averaging](src/interdeca/color_analysis.py),
  `ColorAnalysis.average_textures`, and
  [sRGB conversion and image serialization](src/interdeca/io.py), `TextureIO`.
- Reference checkout:
  `../3d_color_package/tests/end_to_end/README.md` and
  `../3d_color_package/color_deca/deca3/InterDeCA.py`, especially
  `denseSurfaceCorrespondencePair` and `_calculate_average_texture`.

The read-only investigation also produced a temporary detailed audit at
`/tmp/interdeca-small-clams-comparison.json`. That file and the comparison
scripts in `/tmp` are local diagnostics, not versioned fixtures or automated
legacy regression tests. The quantitative results and interpretation needed
for this record are included above.
