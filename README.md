# InterDeCA core

An independent Python package for landmark-guided surface correspondence,
Blender texture transfer, color analysis, and population models.

The numerical core does not import Slicer, Qt, or the old
`3d_color_package` repository. A separate application worker owns the MVC controller,
state, pipelines, and serial executor. A small WebSocket client and the
**InterDeCANext** Slicer presentation package provide the active ATLAS, Mesh
Selection, and MultiRecolor workflows. See [Slicer setup and parity](docs/slicer.md).
A Click CLI runs the same application session directly in the uv environment.
See [CLI usage and Slicer parity](docs/cli.md). A [pytest end-to-end check](tests/end_to_end/README.md)
compares independent Slicer and CLI runs of the small clams dataset.

To try the workflows manually, use the [small clams tutorial](docs/tutorials/small_clams.md).

## Installation

The worker and core support Python 3.12 or newer in their own environment.
Slicer installs the separate `interdeca-client` package from `client/` and uses
its bundled numerical libraries only for scene snapshots and display.

```sh
cd ~/code/interdeca
git submodule update --init --recursive
uv sync
```

The core now includes a required Rust extension built by Maturin. Source checkout
development needs Rust and the platform linker (Xcode command-line tools on
macOS, MSVC tools on Windows, or a C toolchain on Linux). `just rust-build`
rebuilds and installs the optimized extension into `.venv` after Rust edits;
Python-only edits remain editable. `just rust-check` runs the non-Slicer checks
and `just rust-wheel` creates the combined Python/Rust wheel. See
[native development and numerical contracts](rust/README.md). Installing a
compatible prebuilt wheel does not require a Rust compiler.

UMAP is a required core dependency; no extra installation flag is needed.
Slicer still installs only the independent client, keeping UMAP and its numerical
dependencies in the external worker environment.

Blender is an external application, not a Python dependency. The worker targets
Blender 3.6 and newer; supported-version integration testing remains to be done.
Pass its executable to `Blender.check_available()`, set `BLENDER_EXECUTABLE`, or
let discovery search PATH and conventional installation locations. Core operations
do not install dependencies automatically.
The `interdeca` CLI and `interdeca-worker` host entry points are separate from
those stateless numerical operations.

## MorphoWeave comparison backend

The original atlas implementation remains available and is the default. Select
`atlas --backend morphoweave` in the CLI, or **MorphoWeave (comparison)** in the
Slicer atlas form, to run the new backend through the same application core.
Use separate output directories for side-by-side comparisons:

```sh
uv run interdeca atlas --backend native \
  --models /path/to/ras/models --landmarks /path/to/landmarks \
  --textures /path/to/textures --output /path/to/native-results
uv run interdeca atlas --backend morphoweave \
  --models /path/to/ras/models --landmarks /path/to/landmarks \
  --textures /path/to/textures --output /path/to/morphoweave-results
```

MorphoWeave is an unmodified Git submodule at
`src/interdeca/_vendor/morphoweave`, pinned to
`6fd9e448fb3441b71f7fed2bec9434a9b2190d24`. The wheel and source distribution
include its Atlas Builder and license notices. No Slicer, Qt, or CTK installation
is required in the worker. A private module import context supplies the data
handles, RAS OBJ/Markups I/O, logging, and inert display hooks needed by upstream
`_generateAtlas`, `runMean`, and `runAlign`. It does not replace global Python
modules or construct widgets. Unsupported host APIs and an unexpected upstream
source hash fail explicitly; there is no fallback to the original backend.

Both backends retain source triangle order and per-corner UVs during alignment.
MorphoWeave's TPS projection is requested for **every** atlas vertex; the adapter
attaches the shared atlas faces and UVs unchanged, including collapsed specimen
faces. This differs from upstream's default 2,000-point landmark export.
Blender cleanup, UV generation, baking, color analysis, PCA/UMAP, and presentation
remain shared. Existing-atlas overrides work with either correspondence backend.

This is a comparison phase, not a declaration of numerical equivalence. The
`--normalize-scale` flag controls specimen alignment (default: preserve scale).
MorphoWeave's reference/mean Procrustes calls still normalize scale internally.
After mean construction, the application scales its mesh and landmarks together
to the arithmetic mean **input landmark centroid size**, before Blender cleanup.
Centroid size is the square root of the sum of squared landmark distances from
their centroid; it is measured separately for each original specimen, giving each
specimen equal weight. The conversion uses the actual upstream atlas centroid
size, so it does not assume that floating-point normalization returns exactly one.
It scales about the atlas landmark centroid, preserving its pose and mean shape.
This restores representative input coordinate units in both alignment modes;
individual specimen sizes remain preserved unless `--normalize-scale` is enabled.
All inputs must already use common length units. Supplied atlas overrides retain
their given scale. This conversion does not reproduce the original Atlas mean
calculation or establish numerical equivalence between backends.

MorphoWeave also uses nearest warped vertices for mean construction, float32 in
several outputs, and atlas-to-specimen TPS followed by surface projection. Exact duplicate
mean vertices are cleaned with its own VTK cleanup before establishing atlas
topology. This first phase uses TPS only. Specimen IDs must be unique ignoring
case because upstream pairs filenames that way; output IDs retain original case.

Every completed run writes `atlas_run.json` (schema version 2) with its options,
backend identity, upstream revision/hash where applicable, and projection-distance
convention. `atlas_scale_conversion` records the conversion method, source and
target centroid sizes, factor, and fixed center. A point `p` is mapped to
`center + factor * (p - center)`. The field is null for native atlas construction
and supplied atlas overrides. The baseline commit `e12196a` records the earlier
MorphoWeave behavior without this conversion, using metadata schema version 1.
Native diagnostics measure displacement in common warped space; MorphoWeave
diagnostics use aligned specimen space. They must not be compared as the same
quantity. The pre-migration findings remain in [MATERIAL _DIFFERENCES.md](<MATERIAL _DIFFERENCES.md>).

The serial manager reports progress and accepts cancellation between backend
operations. MorphoWeave's mean construction and high-level alignment are batch
operations; cancellation waits for the active batch, and detailed messages are
logs rather than parsed progress. Temporary staging files are removed on success
or failure. No additional worker process is introduced.

To upgrade the dependency, review the upstream API and numerical changes, update
the submodule pin and `_morphoweave_host.py` revision/source hash together, and run
the host-contract tests plus both small-clams e2e cases. Do not patch the submodule
source or refresh expected scientific results without reviewing the differences.

## Operations

Every operation below is a static method. Calling an operation does its work
immediately; it does not create a scheduled task.

| Module | Namespace | Operations |
| --- | --- | --- |
| `dataset.py` | `Dataset` | `inspect_specimen`, `validate_specimens` |
| `blender.py` | `Blender` | `check_available`, `clean_atlas`, `prepare_atlas_uv`, `bake_specimen_texture` |
| `atlas.py` | `Atlas` | `compute_reference`, `prepare_contribution`, `assemble_mean`, `align_specimen` |
| `morphoweave.py` | `MorphoWeave` | `build_atlas`, `restore_input_scale`, `align_specimens`, `resample_specimen` |
| `geometry.py` | `Geometry` | `prepare_analysis`, `resample_specimen`, `select_region`, `extract_region` |
| `color_analysis.py` | `ColorAnalysis` | `sample_specimen_colors`, `compute_normalization_stats`, `process_specimen_colors`, `average_textures`, `expand_face_colors` |
| `population.py` | `Population` | `align_clusters`, `assemble_features`, `fit_model`, `reconstruct_features` |

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
  unless replacement is requested. Atlas workflows allocate a distinct run
  directory; temporary Blender files are private per call.

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
# Clean before creating final UVs and resampling; choose distance in model units.
atlas = Blender.clean_atlas(atlas, blender, merge_distance=0.0001)
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
caller-defined maximum distance. Native surfaces and analysis atlases reject
collapsed triangles. A nearest-point projection can map distinct atlas vertices
onto the same specimen point or edge, so resampled correspondence meshes retain
those collapsed faces without removing or reordering indices or UVs. They use
`CorrespondenceMeshData` at boundaries; `MeshIO.read(..., allow_degenerate=True)`
explicitly reads such exported meshes. They are used for averaging and baking,
not as source surfaces for barycentric queries or analysis area weights.

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

Population models support PCA, FastICA, and UMAP, all installed by `uv sync`. Feature
standardization is explicit and its fitted parameters are returned. Missing
samples raise by default; `missing="drop"` explicitly uses the same valid sample
intersection for every specimen. Model rank and convergence are checked.

These choices are documented intentionally: the new package is not claimed to
produce bit-for-bit output matching the legacy Slicer implementation.

## MVC and task execution

`workflows.py` constructs `list[list[Task]]` workflows:
inspection → reference → contributions → mean atlas → cleanup → UV/alignment → resampling
and analysis preparation → baking → sampling/statistics → specimen processing
→ cluster matching/features → modeling. A supplied atlas can bypass construction.

`execution.py` defines `Task`, `ResultRef`, `Pipeline`, `ExecutionManager`, and
`EventDispatcher`. Result references must point to earlier stages. Submission
copies input arrays and freezes the stages. Registered operations receive named
input records; task definitions contain no executable callbacks or scene objects.
Every submission requires the run ID reserved by the application worker; the
executor no longer allocates an implicit identifier.

`SerialExecutionManager` runs complete pipelines FIFO on one worker thread.
`ApplicationController` depends only on the execution protocol, not the concrete
class. `ApplicationSession` in `application.py` supplies an asynchronous dispatcher to
deliver notifications on its application thread. It constructs both components
and retains the fitted estimators. The WebSocket worker and in-process Click CLI
both host this same session; `worker.py` contains only the socket host and launch
entry point. Slicer and future standalone hosts send user actions through the
independent `interdeca_client` package and receive presentation updates on their
UI threads. See the [version 1 application protocol](docs/application-protocol.md)
for JSON schemas, request/run identity, capabilities, and numerical file ownership.
`application_models.py` holds immutable session inputs, selections, results, and
run state. User requests are validated by `AtlasRequest`, `ColorRequest`, and
`ModelRequest` in `serde.py` before conversion to those internal records.

Cancellation prevents subsequent tasks from starting. An active operation finishes
before the run becomes cancelled. Failures stop that pipeline and retain completed
task results on the controller; another queued run can still execute. Shutdown is
nonblocking and detaches callbacks. Stopping the worker ends the separate process;
ordinary cancellation remains cooperative. Progress counts stages and tasks, not
elapsed time. No pause/resume or persistent execution queue is provided.

Each session accepts one analysis at a time. Input revisions invalidate dependent
results; display changes do not. Morphospace requests keep only the latest pending
request. The worker also owns slider ranges and conversion to fitted coordinates;
Slicer only sends the selected specimen and slider fractions. Fitted estimators
remain live, read-only session objects and are not
made restorable by the JSON snapshot format.

The [small-clams pytest](tests/end_to_end/README.md) runs the CLI and Slicer as
separate applications and compares numerical outputs and MRML presentation data.
Earlier core, Blender, and Slicer smoke checks are recorded in
[the parity checklist](docs/slicer.md).
Exhaustive numerical comparisons, broader Blender versions, and large datasets
still need separate validation.

The [profiling harness](tools/profiling/README.md) measures both atlas backends
on the same three small clams, using separate baseline, CPU, and allocation
runs. It records the application and Blender process memory, retained arrays,
file conversion costs, and output fingerprints. Profiling dependencies are
installed only with `uv sync --group profiling`.
The [initial profile report](docs/profiling/2026-10-02/index.html) records measured
runtime, allocation peaks, serialization costs, and their scaling implications.
