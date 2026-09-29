# InterDeCA core

An independent Python package for landmark-guided surface correspondence,
Blender texture transfer, color analysis, and population models.

This is the first implementation of the new core. It does not import the old
`3d_color_package` repository, Slicer, Qt, or any application scene. Execution
managers, task scheduling, Slicer adapters, and tests are intentionally deferred.
The implementation has not yet undergone numerical parity or integration testing.

## Installation

The existing uv scaffold uses Python 3.13 or newer.

```sh
cd ~/code/interdeca
uv sync
# Optional, only when fitting UMAP models:
uv sync --extra umap
```

Blender is an external application, not a Python dependency. The worker targets
Blender 3.6 and newer; supported-version integration testing remains to be done.
Pass its executable to `Blender.check_available()`, set `BLENDER_EXECUTABLE`, or
let discovery search PATH and conventional installation locations. There is no
CLI entry point or automatic dependency installation inside core operations.

## Operations

Every operation below is a static method. Calling an operation does its work
immediately; it does not create a scheduled task.

| Module | Namespace | Operations |
| --- | --- | --- |
| `dataset.py` | `Dataset` | `inspect_specimen`, `validate_specimens` |
| `blender.py` | `Blender` | `check_available`, `prepare_atlas_uv`, `bake_specimen_texture` |
| `atlas.py` | `Atlas` | `compute_reference`, `prepare_contribution`, `assemble_mean`, `align_specimen` |
| `geometry.py` | `Geometry` | `prepare_analysis`, `resample_specimen` |
| `color_analysis.py` | `ColorAnalysis` | `sample_specimen_colors`, `compute_normalization_stats`, `process_specimen_colors`, `average_textures` |
| `population.py` | `Population` | `align_clusters`, `assemble_features`, `fit_model` |

`models.py` defines internal `typing.NamedTuple` inputs/results. `serde.py`
defines Pydantic boundary models. `io.py` provides `MeshIO`, `LandmarkIO`, and
`TextureIO`. `_blender_worker.py` is an installed package file that runs only
inside Blender, with Blender's own Python and NumPy. The host uses Pydantic to
validate and serialize JSON requests, and validates numeric mesh payloads before
writing NPZ files. The worker decodes host-validated requests into shared
`NamedTuple` records from `_blender_protocol.py`; it needs no Pydantic installation
in Blender. Results are validated again on the host when read back.

Core operations raise contextual exceptions instead of silently substituting
empty outputs. They return arrays, immutable data records, explicit artifact
paths, or fitted estimators. No operation knows about a UI or task graph.

## Data contracts

- **Geometry:** `Mesh.vertices` is `(N, 3)`, `Mesh.faces` is `(M, 3)`, and optional
  `Mesh.uv` is `(M, 3, 2)`. UVs belong to triangle corners. A seam does not require
  duplicate geometric vertices. Connectivity, vertex order, and UV order are
  part of correspondence, and a topology fingerprint checks compatibility.
- **Coordinates:** all in-memory positions use RAS axes. Mesh readers accept an
  explicit RAS/LPS convention; landmark readers honor file metadata unless
  overridden. Length units are the caller's responsibility and must agree
  across specimens, landmarks, and Blender ray settings.
- **Landmarks:** exact, unique labels identify homologous points. Validation
  reorders each specimen to the first specimen's label order. Rigid alignment
  needs three noncollinear landmarks; the 3-D spline needs at least four
  noncoplanar landmarks. Reflections are never allowed.
- **Textures:** one 8-bit sRGB image per specimen, with native source UVs.
  Atlas UVs occupy a single `[0, 1]` tile. RGB arrays use `[0, 1]`; Lab uses D65.
  Explicit masks identify missing samples. Black is a valid color.
- **Ownership:** validated mesh and color records own read-only numerical arrays.
  `NamedTuple` construction itself is lightweight; numerical record factories
  inside algorithms explicitly call `.validated()`. Pydantic boundary models
  perform these checks automatically before returning an internal record. Inputs
  are not edited. Callers must also treat input files and fitted third-party
  estimators as read-only during concurrent work.
- **Files:** output paths are explicit. Existing outputs raise `FileExistsError`
  unless replacement is requested. A future manager must allocate distinct
  paths to concurrent tasks. Temporary Blender files are private per call.

Mesh input supports triangular OBJ and triangulated PLY, STL, and VTP. OBJ
polygons must be triangulated before loading; OBJ materials are not read.
Landmarks support Slicer Markups JSON, FCSV, and CSV with `label,x,y,z` columns.
Texture loading supports ordinary 8-bit images decoded by Pillow. ICC color
conversion, HDR, UDIM tiles, multiple materials, and source material opacity are
outside the current contract.

## Internal records and boundary serialization

Computation uses named fields such as `mesh.vertices`, `request.resolution`, and
`entry.moments`. Internal records are `NamedTuple`s, including normalization
entries and decoded image pixels. Dictionaries remain only where a key/value
mapping is the actual data structure, such as the edge-to-faces lookup table,
or where a file format/library API requires one.

Use the corresponding Pydantic `*Data` class in `interdeca.serde` at an API or
persistence boundary. Conversion is explicit:

```python
from interdeca.serde import MeshData

# An external JSON object has named fields; the result is an internal Mesh tuple.
incoming_json = '''{
  "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
  "faces": [[0, 1, 2]],
  "uv": null
}'''
mesh = MeshData.model_validate_json(incoming_json).to_internal()

# Encode arrays as nested numeric lists through Pydantic's JSON serializer.
outgoing_json = MeshData.from_internal(mesh).model_dump_json(indent=2)
```

The same pattern supports specimens, landmarks, atlases, transformations,
resampling results, analysis geometry, textures, colors, normalization statistics,
cluster alignments, and population features. Pydantic rejects unknown fields in
InterDeCA-owned formats, nonfinite values, incorrect scalar types, and malformed
numerical records. Its models use named JSON objects rather than the positional
arrays a bare NamedTuple serializer might produce. See the
[Pydantic serialization documentation](https://docs.pydantic.dev/latest/concepts/serialization/).

Markups JSON has dedicated models preserving Slicer's field names. Display and
other unused Slicer metadata are intentionally ignored on input. CSV/FCSV values
are converted from text using Pydantic before entering the core. Image and mesh
files retain their native binary/text encodings, with Pydantic validation at the
adapter boundary.

`PopulationModelData.from_model(model)` serializes a `PopulationModelSnapshot`
containing numerical results and preprocessing parameters. The live fitted
estimator is intentionally outside this JSON contract: loading a snapshot does
not reconstruct an estimator capable of transforming new observations.

## A direct workflow

This example makes the dependencies visible without introducing an execution
manager. Replace the input paths and choose ray distances in your mesh's units.
Eight clusters and two model components are examples, not universal defaults
for a particular dataset; each must be supported by the available variation.

```python
from pathlib import Path

from interdeca import Atlas, Blender, ColorAnalysis, Dataset, Geometry, Population
from interdeca import LandmarkIO, MeshIO

inputs = [
    ("specimen_01", "inputs/01.obj", "inputs/01.mrk.json", "inputs/01.png"),
    ("specimen_02", "inputs/02.obj", "inputs/02.mrk.json", "inputs/02.png"),
    ("specimen_03", "inputs/03.obj", "inputs/03.mrk.json", "inputs/03.png"),
]
output = Path("output")
specimens = Dataset.validate_specimens(
    [Dataset.inspect_specimen(*paths) for paths in inputs],
    require_textures=True,
)
blender = Blender.check_available()

# Each specimen contributes the same vertex/face ordering to the mean atlas.
reference = Atlas.compute_reference(specimens)
contributions = [Atlas.prepare_contribution(item, reference) for item in specimens]
atlas = Atlas.assemble_mean(contributions)
atlas = Blender.prepare_atlas_uv(atlas, blender)
MeshIO.write_obj(atlas.mesh, output / "atlas.obj")
LandmarkIO.write_json(atlas.landmarks, output / "atlas.mrk.json")

# Shared analysis geometry is prepared once, after the final UV map exists.
analysis = Geometry.prepare_analysis(atlas)  # Or sample_count=1000 for large meshes.
aligned = [Atlas.align_specimen(item, atlas.landmarks).specimen for item in specimens]
resampled = [
    Geometry.resample_specimen(item, atlas, mean_landmarks=atlas.landmarks)
    for item in aligned
]
textures = [
    Blender.bake_specimen_texture(
        source,
        target,
        blender,
        output / f"{source.specimen_id}.png",
        resolution=(2048, 2048),
        cage_extrusion=0.01,  # Choose this for the physical scale of your meshes.
    )
    for source, target in zip(aligned, resampled, strict=True)
]
ColorAnalysis.average_textures(textures, output / "mean_texture.png")

# Pooled statistics form a barrier before independent specimen processing.
sampled = [ColorAnalysis.sample_specimen_colors(item, analysis) for item in textures]
normalization = ColorAnalysis.compute_normalization_stats(sampled)
processed = [
    ColorAnalysis.process_specimen_colors(item, normalization=normalization, n_clusters=8)
    for item in sampled
]
matched = Population.align_clusters(processed)
features = Population.assemble_features(matched, analysis, representation="cluster_areas")
model = Population.fit_model(features, method="pca", n_components=2)
print(model.specimen_ids, model.scores)
```

For unclustered color analysis, omit `n_clusters`, skip `align_clusters`, and use
`Population.assemble_features(processed, analysis, representation="colors")`.
For an existing atlas, construct `AtlasModel(mesh, landmarks)` and skip reference
selection and mean construction. Reuse its existing UV map if it is suitable;
regenerating UVs makes previously baked textures incompatible.

## Algorithm choices

Atlas reference selection uses generalized Procrustes alignment and chooses the
specimen closest to the resulting landmark mean. Scaling is off by default.
Contributions use spline-guided nearest-surface projection, then the atlas is an
equal-weight coordinate-wise mean of corresponding vertices and landmarks.
Projection distances are returned as diagnostics; resampling can reject a
caller-defined maximum distance. Invalid or collapsed triangles raise errors.

The spline uses the 3-D radial basis `r` plus an affine term. In SciPy this is
`RBFInterpolator(kernel="linear", degree=1)` (the kernel's minus sign is absorbed
by the fitted coefficients). SciPy's kernel named `thin_plate_spline` uses
`r² log(r)`, so it is not selected here. See the
[SciPy RBFInterpolator documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html).
After projection, barycentric coordinates pull points back onto the original
specimen triangles. This avoids treating a separately fitted reverse spline as
an exact inverse.

Blender Smart UV projection changes only UV coordinates. Diffuse color baking
uses CPU Cycles with illumination disabled. Ray misses, folds, and occlusion
still require inspection; alpha represents image coverage, not correspondence
confidence or source material opacity. No mesh cleanup or topology change is
performed during baking.

Analysis sampling uses face UV centroids with bilinear image interpolation. This
is a point approximation, not integrated face color. Optional smoothing averages
over face adjacency. Optional subsampling uses deterministic graph-distance
farthest-point selection and covers every disconnected component. Cluster-area
features assign each sample the atlas area of its nearest-sample region; with
all faces sampled this becomes the ordinary sum of atlas face areas.

Normalization pools moments over valid sampled colors, rather than background
pixels in whole texture images. Each sample has equal statistical weight. It
matches per-specimen luminance/chroma moments to pooled moments, preserving hue
where hue is defined. Constant channels do not acquire artificial variance.
Clustering uses Lab K-means, optional average-linkage consolidation, and Hungarian
matching to an explicit reference specimen. The common palette averages aligned
centroids equally across specimens. Texture image averaging uses linear-light
RGB; display conversions of normalized Lab may clip out-of-gamut colors.

Population models support PCA, FastICA, and optional UMAP. Feature
standardization is explicit and its fitted parameters are returned. Missing
samples raise by default; `missing="drop"` explicitly uses the same valid sample
intersection for every specimen. Model rank and convergence are checked.

These choices are documented intentionally: the new package is not claimed to
produce bit-for-bit output matching the legacy Slicer implementation.

## Later execution managers

The results establish boundaries for future `list[list[Task]]` workflows:
inspection → reference → contributions → mean atlas → UV/alignment → resampling
and analysis preparation → baking → sampling/statistics → specimen processing
→ cluster matching/features → modeling. A supplied atlas can bypass construction.

Managers will own concurrency, cancellation, resource budgets, persistence,
progress, and Slicer integration. Independent calls here do not share a scene,
global random generator, output directory, or mutable computation cache. Blender
thread counts are explicit so a manager can avoid oversubscribing CPUs when
running several bakes. No managers or task classes are included at this stage.

Testing is deferred as requested. Numerical correctness, legacy comparisons,
Blender versions, large datasets, and future Slicer integration need their own
validation work before this core is relied on for analysis.
