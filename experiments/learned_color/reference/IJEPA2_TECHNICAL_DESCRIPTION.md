# I-JEPA rendered surface-appearance pipeline: technical description

This document describes pipeline version `surface-appearance-v2.2-group-alignment`,
implemented in
[`ijepa_pipeline_2.py`](ijepa_texture_embeddings/ijepa_pipeline_2.py). The pipeline turns
paired textured meshes into one frozen I-JEPA embedding per specimen through the following
stages:

```text
OBJ mesh + texture
        -> standardized orthographic exterior render
        -> deterministic, unaugmented, all-surface crops
        -> frozen I-JEPA crop vectors
        -> mean raw crop vector, one specimen-level L2 normalization
        -> PCA, UMAP, and cosine-neighbor reports
```

## Scope, terminology, and status

**I-JEPA2** is an internal name for the second repository experiment built around I-JEPA.
It is not the name of a model. The encoder is the original Image-based Joint-Embedding
Predictive Architecture (I-JEPA) checkpoint, normally `facebook/ijepa_vith14_1k`.
I-JEPA and V-JEPA 2 are separate models; manuscript text should call the encoder
**I-JEPA** and, if necessary, define this implementation as the second, patch-based
experiment.

The pipeline is frozen downstream feature extraction. It does not train or fine-tune the
encoder, reproduce masked-block I-JEPA pretraining, use a predictor or target encoder, or
apply a JEPA loss. It also deliberately excludes photometric jitter and horizontal or
vertical reflection augmentation. It uses mean pooling over final spatial tokens, not CLS
pooling.

The representation is a **rendered surface-appearance embedding**, not a texture-only or
geometry-free embedding. Input pixels retain projected texture, local geometry, smooth
shading, foreshortening, and any acquisition or texture-bake artifacts. Strict all-surface
crops remove background and most direct silhouette cues, but they do not remove these
other sources of variation. Crop coordinates are discarded at specimen aggregation, so
the final vector is an unordered bag-of-crops summary rather than an anatomically
corresponded texture field.

This is an offline exploratory analysis, not a ColorAtlas or 3D Slicer interface feature.
Its purpose is to complement the area-weighted color-composition representation with a
descriptor that can respond to local arrangements of rays, bands, growth lines, and other
surface markings. It does not replace ColorAtlas's interpretable palette proportions,
shared atlas coordinates, or inverse mapping.

The active freshwater-mussel profile is
[`ijepa_config_2.yaml`](ijepa_texture_embeddings/ijepa_config_2.yaml). It requests 32
paired specimens and selects `group_pca` orientation with `UF_IZ_438751` as the common
reference. At the time of this documentation update, the artifacts already present under
`outputs_2_1` and `rendered_surface_2_1` still identify themselves as pipeline v2.1. They
are therefore evidence for the preceding feature-flags implementation, not a completed
v2.2 group-aligned run. The v2.2 pipeline version and orientation fingerprint will
invalidate those render and embedding caches on the next complete run.

## Active configuration versus class defaults

Several experimental choices are compatibility flags. The YAML used for the mussel
analysis intentionally differs from the Python model defaults:

| Decision | Active `ijepa_config_2.yaml` | Python default | Consequence |
|---|---:|---:|---|
| Orientation | `group_pca` | `independent_pca` | Resolves target PCA signs against a shared reference |
| Reference specimen | `UF_IZ_438751` | first paired specimen | Anchors the group camera frame explicitly |
| Forced-square normalization | `true` | `false` | Stretches cropped content to fill 2048 × 2048 |
| Larger-context crops | `true` | `false` | Extracts 256-pixel rather than 224-pixel crops |
| PCA feature standardization | `true` | `false` | Standardizes features only for PCA |
| Model revision | unpinned | unpinned | Loads the current Hub revision unless explicitly set |

Other active settings are a 2048-pixel render, 4096-pixel maximum texture edge, 12-pixel
content padding, 64 crops per specimen, maximum crop IoU 0.6, batch size 8, UMAP with 10
neighbors and `min_dist = 0.1`, and five nearest neighbors per specimen
([active configuration](ijepa_texture_embeddings/ijepa_config_2.yaml)). Configuration
files are strict Pydantic mappings: unknown keys are rejected, numeric ranges are
validated, and relative paths are resolved against the YAML file rather than the current
working directory
([configuration model](ijepa_texture_embeddings/ijepa_pipeline_2.py#L121-L354)).

## Manuscript-ready methods

### Frozen I-JEPA representation of rendered surface appearance

We implemented an offline, patch-based analysis of rendered specimen appearance using a
frozen I-JEPA encoder. Meshes and texture images were paired by identical filename stem.
For each specimen, we rendered one exterior view with an orthographic camera, fixed smooth
shading, and a transparent background. Camera frames were aligned across the collection
using a common reference specimen: target mesh principal-axis signs were selected to
match the reference's standardized third-moment asymmetry signature. The active analysis
used `UF_IZ_438751` as the reference. Corresponding-landmark Procrustes alignment is also
implemented as a more strongly constrained alternative when Slicer landmark files are
available.

Each render was cropped to alpha-defined content with 12 pixels of padding and resampled
to 2048 × 2048 pixels. In the active compatibility profile, the cropped content was
stretched to the square canvas; the implementation can instead preserve its aspect ratio
by centering the crop on a transparent square before resizing. We selected 64
deterministic 256 × 256-pixel crops per specimen. Every accepted crop consisted entirely
of pixels classified as rendered surface at alpha tolerance 10, and candidates whose
intersection-over-union with an accepted crop exceeded 0.6 were rejected. No reflection,
brightness, contrast, saturation, or hue augmentation was applied.

Crops were converted to RGB and processed by the model-specific Hugging Face image
processor. For `facebook/ijepa_vith14_1k`, the processor resizes crops to 224 × 224,
rescales 8-bit values by (1/255), and normalizes each channel with mean and standard
deviation 0.5. The frozen ViT-H/14 encoder produces 256 final spatial tokens with 1,280
features per token. We averaged all final tokens to obtain one raw crop vector. For each
specimen, the 64 raw crop vectors were averaged first, and the resulting specimen vector
was L2-normalized once. Crop vectors were not individually normalized before averaging.

For population summaries, PCA was fitted to feature-wise standardized specimen vectors
in the active compatibility profile. UMAP was fitted separately to the unit-normalized
specimen vectors using cosine distance, 10 neighbors, `min_dist = 0.1`, one execution
thread, and random seed 42. Five nearest non-self specimens were ranked by cosine
similarity in the specimen embedding space. These are descriptive fits to the complete
collection, not cross-validated estimates of out-of-sample performance.

This text describes the v2.2 implementation and active protocol. It should not be presented
as a completed v2.2 analysis until the outputs have been regenerated and audited.

## Inputs and pairing

The renderer indexes `*.obj` files and supported texture images, then pairs them by exact
stem. With no `specimen_ids` filter, it uses the sorted stem intersection. With an explicit
filter, every requested specimen must have both inputs. An empty intersection, missing
requested input, or mismatch with a nonzero `expected_pair_count` is an error
([pair discovery](render_clam.py#L163-L190),
[pipeline count validation](ijepa_texture_embeddings/ijepa_pipeline_2.py#L416-L450)).

The active profile points to the original freshwater-mussel models and textures under
`data/All_Clams`; it does not consume ColorAtlas-transferred atlas textures or a shared UV
parameterization. The source texture is converted to RGBA and, when its longest edge
exceeds `texture_max_size`, downsampled with Lanczos interpolation before being passed to
VTK ([texture loading](render_clam.py#L193-L199)).

## Cross-specimen camera alignment

Camera orientation is an explicit configuration choice with three modes:

| Mode | Shared reference | Alignment method | Intended use |
|---|---|---|---|
| `independent_pca` | No | Per-mesh exterior and world-up heuristics | Legacy behavior and datasets already in a consistent frame |
| `group_pca` | Yes | PCA axis roles plus reference-matched skewness signs | Landmark-free group alignment |
| `landmark_group` | Yes | Proper orthogonal Procrustes rotation of corresponding landmarks | Preferred when reliable anatomical landmarks exist |

For either group mode, `orientation_reference_id` must name a paired specimen. If omitted,
the first paired specimen is used. The reference mesh, reference landmark file when
applicable, calculated camera frame, and reference skewness all contribute to the
orientation fingerprint
([orientation context](ijepa_texture_embeddings/ijepa_pipeline_2.py#L576-L636)).

### Reference frame and independent PCA

For a deterministic strided sample of mesh vertices $p_n \in \mathbb{R}^3$, the shared
renderer computes

```text
p_bar = (1 / N) sum_n p_n
C = (1 / N) sum_n (p_n - p_bar) (p_n - p_bar)^T.
```

Eigenvectors are ordered from greatest to least eigenvalue as the major in-plane axis,
minor in-plane axis, and plane normal. Meshes with more than 250,000 points are sampled
with stride `floor(N_all / 250000)` before this calculation. The normal sign is selected
by comparing depth along the candidate normal for the innermost and outermost radial
quartiles. The direction is negated when the central quartile lies behind the rim. The
minor axis is projected perpendicular to the selected direction and signed toward
positive world z to define view-up
([independent orientation](render_clam.py#L260-L305)).

`independent_pca` applies this complete procedure separately to every specimen. It is
deterministic for well-separated principal axes, but it does not guarantee consistent
left-right or exterior signs for a heterogeneous group.

### Group PCA

`group_pca` first obtains the reference direction and view-up using the independent
procedure. It completes the right-handed reference frame

```text
r_ref = unit(direction_ref × up_ref)
F_ref = [r_ref, up_ref, direction_ref].
```

For any frame $F$, vertex coordinates are centered, projected onto its three axes, and
divided axis-wise by their root-mean-square scale. The frame signature is the vector of
standardized third moments

```text
kappa(F) = mean_n(((p_n - p_bar) F / rms_axis)^3).
```

For each target mesh, the second PCA axis is assigned to view-up and the third PCA axis to
view direction. The implementation evaluates all four combinations of their signs,
constructs the corresponding right-handed frame $F$, and maximizes

```text
score(F) = -||kappa(F) - kappa(F_ref)||_2^2
           + 0.01 trace(F^T F_ref).
```

The skewness term provides the primary sign correspondence; world-frame agreement is a
weak tie-breaker. The selected score and target skewness are written to the render
sidecar. This method resolves signs but assumes that PCA axis ordering has the same
anatomical meaning across specimens. It cannot reliably resolve nearly symmetric shapes,
near-equal eigenvalues, or a true permutation of anatomical axes
([group-PCA implementation](ijepa_texture_embeddings/ijepa_pipeline_2.py#L639-L683)).

### Landmark-group alignment

`landmark_group` reads the first markup collection from
`<landmarks_dir>/<specimen_id>.mrk.json`, retains defined control points, and requires at
least three 3D points. Target and reference arrays must have the same shape; their row
order is therefore the anatomical correspondence. Let centered landmark matrices be
$X$ for the target and $Y$ for the reference. With

```text
X^T Y = U S V^T,
R = U V^T,
```

the implementation corrects the last singular vector when necessary so
`det(R) = +1`. This prohibits a reflection. It then transports the reference camera axes
back into target coordinates with $R^T$, followed by re-orthogonalization of view-up
against view direction
([landmark loading and rotation](ijepa_texture_embeddings/ijepa_pipeline_2.py#L509-L573),
[camera transport](ijepa_texture_embeddings/ijepa_pipeline_2.py#L686-L722)).

The sidecar reports a landmark-alignment normalized RMSE:

```text
RMSE_norm = sqrt(mean_n ||X_n R - Y_n||_2^2)
            / sqrt(mean_n ||Y_n||_2^2).
```

Only translation and rotation are removed; target landmarks are not rescaled. The metric
therefore reflects scale and shape differences as well as correspondence or fitting error.
The landmark path and SHA-256 digest are also recorded.

## Rendering and canvas normalization

The selected camera is placed along the exterior direction and aimed at the mesh center.
Parallel projection removes perspective size changes. Parallel scale is computed from the
horizontal and vertical projected extents and multiplied by `render_margin`. Rendering is
off-screen on a square canvas with a transparent background, smooth shading, no mesh
edges, supersampling anti-aliasing, and fixed material coefficients: ambient 0.38, diffuse
0.75, and specular 0.08
([camera and scene](render_clam.py#L308-L377),
[render execution](ijepa_texture_embeddings/ijepa_pipeline_2.py#L824-L853)).

The screenshot is converted to RGBA. Content is the alpha mask above
`crop_alpha_tolerance`; the active value is 8. Its bounding box is expanded by 12 pixels,
subject to image limits. Two mutually exclusive normalization paths then produce the
configured square render:

- With `force_square_normalization: false`, the crop is centered on a transparent square
  whose side equals its longer dimension and is then resized with Lanczos. This preserves
  the projected width-to-height ratio.
- With `force_square_normalization: true`, the rectangular crop is resized directly to
  the square. This is the active compatibility behavior. It equalizes canvas occupancy
  but anisotropically distorts shell shape and projected pattern scale.

See [canvas normalization](ijepa_texture_embeddings/ijepa_pipeline_2.py#L745-L777).
Because the encoder later receives local crops, forced-square normalization does not
inject the entire silhouette into a crop, but it does change the pixels and spatial scale
from which every crop is sampled.

Each render is saved as `<specimen_id>_exterior.png`. A neighboring JSON sidecar records
the selected view vectors, orientation method and reference, method-specific diagnostics,
input paths, pipeline version, and a content fingerprint
([render loop and sidecar](ijepa_texture_embeddings/ijepa_pipeline_2.py#L856-L927)).

## Surface-crop sampling

### Full-content mask

The patch stage operates on the normalized render. For an image with transparency, a
pixel is surface content when alpha exceeds `patch_alpha_tolerance`; the active value is
10. A white-background fallback is used for fully opaque inputs. For patch side $s$ and
binary surface mask $M$, the background fraction at top-left coordinate $(x,y)$ is

```text
b(x, y) = 1 - (1 / s^2)
          sum_{u=0}^{s-1} sum_{v=0}^{s-1} M(x+u, y+v).
```

An integral image computes counts for every candidate window. The pipeline constructs
`PatchSpec` with `require_full_content = true` and `max_background_fraction = 0`, so a
candidate is valid only when $b(x,y)=0$. Alpha values above 10 can include partially
transparent antialiased boundary pixels, although a 256-pixel square must remain entirely
inside the classified mask
([effective patch specification](ijepa_texture_embeddings/ijepa_pipeline_2.py#L310-L329),
[candidate calculation](patch_sampling.py#L83-L116)).

### Deterministic selection and overlap

The active source-crop side is 256 pixels because `use_larger_context_crops` is true. The
model processor subsequently resizes it to 224 pixels. With the flag disabled, the active
source side would be `patch_size_px`, currently 224.

A stable 64-bit specimen seed is the BLAKE2b digest of
`"<patch_sample_seed>:<specimen_id>"`; the active base seed is 20,240,601. All valid
positions are randomly permuted using that seed and visited greedily. For equal square
crops separated by center offsets $(\Delta x,\Delta y)$, the intersection-over-union is

```text
w = max(0, s - |Delta x|)
h = max(0, s - |Delta y|)
I = w h
IoU = I / (2 s^2 - I).
```

A candidate is rejected only when its IoU with an accepted crop is greater than 0.6;
equality is allowed. Minimum center distance is fixed at zero. Sampling stops when 64
positions have been accepted or the candidate list is exhausted
([seed and greedy selection](patch_sampling.py#L77-L80),
[overlap implementation](patch_sampling.py#L119-L170)).

The selected regions are converted directly to RGB. `NO_JITTER` disables every flip and
photometric transformation. A BLAKE2b pixel digest suppresses duplicate crop images
within a specimen. The pipeline rejects the specimen if this de-duplication or a shortage
of valid positions leaves fewer than the requested 64 crops
([unaugmented sampling](ijepa_texture_embeddings/ijepa_pipeline_2.py#L930-L991),
[crop materialization](patch_sampling.py#L219-L249)).

### Manifest contract

Saved crops are named `<specimen_id>_pNNNN.png` under one directory per specimen. The
ordered `patch_manifest.csv` is the handoff to inference and records:

- specimen ID and patch index;
- absolute patch path and patch SHA-256;
- absolute source-render path and source-render SHA-256;
- crop left, top, size, and measured background fraction; and
- source model and texture paths.

Before inference, every referenced patch must exist and its current SHA-256 must match the
manifest. At least two specimens are required for the population-analysis stage
([manifest writing and verification](ijepa_texture_embeddings/ijepa_pipeline_2.py#L962-L1029)).
The sampler does not clear unreferenced files from an existing patch directory; consumers
must use the manifest rather than a directory glob.

## I-JEPA preprocessing and crop embeddings

The model and image processor are loaded through Hugging Face `AutoModel` and
`AutoImageProcessor` at the same requested revision. An explicit `model_revision` is
passed to both. The model is moved to the selected device, switched to evaluation mode,
and executed inside `torch.inference_mode()`
([model loading and inference](ijepa_texture_embeddings/ijepa_pipeline_2.py#L1032-L1071),
[batch loop](ijepa_texture_embeddings/ijepa_pipeline_2.py#L1190-L1219)).

With the default `facebook/ijepa_vith14_1k` checkpoint, the processor transformation is

```text
x_224 = bilinear_resize(x_RGB, 224, 224)
x_01 = x_224 / 255
x_norm[c] = (x_01[c] - 0.5) / 0.5,  c in {R, G, B}.
```

The checkpoint is a ViT-H/14 encoder: 224/14 gives a 16 × 16 grid, so the final hidden
state for crop $j$ of specimen $i$ has shape
$H_{ij} \in \mathbb{R}^{256 \times 1280}$. The model has no prepended CLS token. The
pipeline validates a three-dimensional `[batch, tokens, features]` result and averages all
spatial tokens:

```text
q_ij = (1 / 256) sum_{t=1}^{256} H_ijt.
```

The raw $q_{ij}$ is saved. It is not L2-normalized at crop level
([token pooling](ijepa_texture_embeddings/ijepa_pipeline_2.py#L1074-L1086)).

Automatic device selection uses a CUDA-compatible device, including ROCm builds exposed
through `torch.cuda`, when available. CPU inference occurs only when `allow_cpu: true`;
an explicit CPU request is otherwise rejected. The active configuration does not allow a
silent CPU fallback.

## Specimen aggregation

Let specimen $i$ have $m_i$ accepted crops. The implementation calculates

```text
q_i = (1 / m_i) sum_{j=1}^{m_i} q_ij
z_i = q_i / max(||q_i||_2, 1e-12).
```

Thus, raw crop vectors are averaged and each specimen is normalized exactly once. This is
not equivalent to averaging unit-normalized crop directions: crop-vector magnitude is
retained as an implicit weight until the final normalization. With the active settings,
the sampling stage enforces $m_i=64$ for every specimen
([aggregation](ijepa_texture_embeddings/ijepa_pipeline_2.py#L1167-L1187)).

No crop is weighted by projected pixel area, three-dimensional surface area, anatomical
location, informativeness, or overlap. Because accepted crops may overlap, a rendered
region can influence the mean more than once. Because crop coordinates are discarded,
the same set of local patterns in different anatomical locations can yield similar
specimen vectors.

Both crop and specimen matrices must be nonempty, finite, two-dimensional, and not
effectively constant. The specimen vectors are stored as `float32`.

## Population analysis

Let $Z \in \mathbb{R}^{n \times 1280}$ contain the unit-normalized specimen vectors.

### PCA

When `standardize_pca_features` is false, PCA is fitted directly to $Z$. When true, as
in the active profile, every feature is standardized across specimens first:

```text
Z'_id = (Z_id - mean_i Z_id) / std_i Z_id.
```

PCA is fitted to $Z'$, and two scores are saved. Standardization affects PCA only. The
pipeline does not save the fitted scaler, component loadings, explained-variance ratios,
or a transform for new specimens.

### UMAP

UMAP is always fitted to the original specimen vectors $Z$, not the optionally
standardized PCA input. It uses `metric = "cosine"`, one execution thread, and the
configured random seed. The neighborhood count is bounded by $n-1$. If $n \le 3$,
the implementation copies the PCA coordinates instead of fitting UMAP
([coordinate calculation](ijepa_texture_embeddings/ijepa_pipeline_2.py#L1271-L1300)).

### Nearest neighbors

Vectors are normalized defensively and the similarity matrix is

```text
S = Z Z^T.
```

For each row, self is excluded and the highest-scoring specimens are retained, with the
requested count bounded by $n-1$. The active report saves five neighbors per specimen
([cosine neighbors](ijepa_texture_embeddings/ijepa_pipeline_2.py#L1303-L1323)). The new
pipeline does not include the old DINOv2 overlap comparison or nearest-neighbor contact
sheets.

## Provenance and cache behavior

### Render cache

Each render fingerprint hashes the pipeline version, model and texture contents, render
size, margin, texture cap, crop parameters, square-normalization flag, orientation
fingerprint, projection, shading, and fixed material coefficients. Group orientation also
hashes the reference mesh and, for landmark mode, reference and target landmark files. A
render is reused only when both its PNG and JSON sidecar exist, `force_render` is false,
and the sidecar fingerprint matches
([render fingerprint](ijepa_texture_embeddings/ijepa_pipeline_2.py#L725-L821)).

The sidecar does not store a hash of the output PNG itself. Manual pixel edits with an
unchanged matching sidecar are therefore not detected by the render stage. Re-running
`sample-patches` hashes the pixels that actually become downstream inputs.

### Embedding cache

The embedding fingerprint hashes:

- pipeline version;
- checkpoint identifier and requested revision;
- token pooling and specimen aggregation semantics; and
- the ordered specimen ID, patch index, and SHA-256 of every manifest crop.

Both the crop-level and specimen-level NPZ files must exist and contain the same expected
fingerprint. Otherwise inference is repeated. `force_recompute_embeddings: true` bypasses
the cache. A valid cache hit still regenerates coordinates, neighbors, and the PCA figure
from the cached specimen matrix
([embedding fingerprint and cache](ijepa_texture_embeddings/ijepa_pipeline_2.py#L1101-L1150)).

The NPZ metadata stores the pipeline version, requested and resolved model revisions,
full processor and model configurations, pooling and aggregation labels, input semantics,
square-normalization flag, and effective crop size. When `model_revision` is null, the
resolved revision is recorded after model loading but is not part of the pre-load cache
fingerprint. For archival reproducibility, an immutable revision should therefore be set
explicitly before the final run.

## Outputs

For the active `output_dir`, the pipeline creates:

| Artifact | Contents |
|---|---|
| `tables/patch_manifest.csv` | Ordered crop identities, hashes, coordinates, mask fraction, and source paths |
| `embeddings/ijepa_surface_per_patch_embeddings.npz` | Raw crop vectors, specimen IDs, patch indices and hashes, fingerprint, metadata |
| `embeddings/ijepa_surface_specimen_embeddings.npz` | Unit specimen vectors, sorted specimen IDs, patch counts, fingerprint, metadata |
| `tables/ijepa_surface_coordinates.csv` | PCA and UMAP coordinates plus manifest crop count |
| `tables/ijepa_surface_neighbors.csv` | Ranked cosine neighbors and similarities |
| `figures/ijepa_surface_pca_shells.png` | Standardized exterior thumbnails positioned at PCA coordinates |

The render directory separately contains one `<specimen_id>_exterior.png` and one
`<specimen_id>_exterior.json` sidecar per specimen. The patch directory contains the RGB
crop PNGs. UMAP coordinates are written to CSV, but the pipeline does not create a UMAP
thumbnail figure.

## Command-line interface

Run commands from the repository environment with the desired YAML:

```bash
python learned_color_space/ijepa_texture_embeddings/ijepa_pipeline_2.py \
  run-all --config learned_color_space/ijepa_texture_embeddings/ijepa_config_2.yaml
```

Available subcommands are:

- `write-config`: write a documented YAML template;
- `render`: create or reuse content-matched standardized renders;
- `sample-patches`: resample crops and rewrite the authoritative manifest;
- `embed`: verify crops, create or reuse embeddings, and rewrite population reports; and
- `run-all`: execute render, sampling, and embedding/reporting in dependency order.

The stage commands expose their dependencies deliberately: `sample-patches` fails when a
required render is missing, and `embed` fails when the manifest or any referenced crop is
missing or changed
([CLI](ijepa_texture_embeddings/ijepa_pipeline_2.py#L1432-L1509)).

## Validation coverage

Focused unit tests cover aspect-preserving and forced-square normalization, the effective
crop-size flag, all-token mean pooling and shape validation, raw-mean-then-L2 aggregation,
optional PCA scaling, proper landmark rotation, and group-PCA consistency under a rigid
rotation
([pipeline tests](../tests/unit/learned_color_space/test_ijepa_pipeline_2.py)). Separate
sampler tests cover full-content windows, deterministic selection, duplicate suppression,
distance and count constraints, oversized crops, jitter bookkeeping in the shared sampler,
and transparent-background masking
([sampler tests](../tests/unit/learned_color_space/test_patch_sampling.py)).

These tests exercise core mathematical contracts but are not a full end-to-end render and
model-inference test. In particular, there is no automated regression image for camera
orientation across the complete biological dataset, no checkpoint-download integration
test, and no statistical stability test for PCA or UMAP.

## Legacy artifact audit and migration boundary

The existing `outputs_2_1` artifacts were created by
`surface-appearance-v2.1-feature-flags`. They contain 2,048 unique manifest crops from 32
specimens, with 64 crops per specimen, 256 × 256 crop size, and zero recorded background
fraction. Their raw crop matrix has shape 2,048 × 1,280, and their unit-normalized specimen
matrix has shape 32 × 1,280. These facts can be used to audit the v2.1 run only.

They must not be described as group-aligned v2.2 results. In particular, the v2.1 render
sidecars do not record `orientation_mode`, `orientation_reference_id`, view vectors, or
group-alignment diagnostics. The active v2.2 configuration reuses the same output
directory names, but the version change causes render fingerprints to differ; a
`run-all` invocation will re-render, resample, and recompute embeddings. Manuscript
results, PCA percentages, biological interpretations, and neighbor summaries should be
recalculated from the resulting v2.2 artifacts rather than copied from the previous
technical description.

## Limitations and claim boundaries

- Only one rendered view is analyzed. Interior coloration, self-occluded regions, and any
  surface not visible to the selected camera are absent.
- Group PCA resolves axis signs from coarse geometric asymmetry but does not establish
  anatomical correspondence. Landmark-group alignment is stronger, but its validity
  depends on landmark definition, ordering, coverage, and placement quality.
- Forced-square normalization changes aspect ratio and local pattern scale. It should be
  treated as part of the estimator, not a neutral file-format operation.
- Strict full-content crops reduce direct background and silhouette signals but retain
  local curvature, shading, foreshortening, glare, damage, rasterization, and texture-bake
  artifacts.
- No ICC-profile conversion, color-chart calibration, exposure matching, or physical-light
  normalization is applied. The only fixed numeric normalization is the checkpoint's image
  processor transform described above.
- Greedy random selection is not uniform over three-dimensional surface area. Projected
  regions that admit more full-content windows can be overrepresented, and overlap lets
  some pixels contribute repeatedly.
- The final mean removes anatomical crop location. The embedding cannot reconstruct a
  texture or drive the current inverse-palette morphospace without an additional decoder,
  correspondence scheme, or retrieval method.
- The frozen model was pretrained on natural images rather than the target biological
  collection. Its features may reflect imaging and rendering characteristics as well as
  biologically relevant appearance.
- Feature-wise PCA standardization is estimated from a small population relative to 1,280
  dimensions in the mussel profile. PCA and UMAP are descriptive in-sample views, not
  evidence of generalization.
- No taxonomic labels, expert similarity judgments, controlled ground truth,
  cross-validation, permutation test, or resampling-stability analysis are built into the
  pipeline.

The defensible claim is that the pipeline computes reproducible, provenance-tracked,
frozen I-JEPA summaries of local rendered appearance under the stated renderer, alignment,
sampling, and aggregation choices. It does not by itself establish biological validity,
texture invariance, taxonomic recovery, superiority to ColorAtlas or another encoder, or
generalization to new collections.

## Implementation map

| Topic | Implementation or artifact |
|---|---|
| Active mussel configuration | `ijepa_texture_embeddings/ijepa_config_2.yaml` |
| Pipeline, alignment, inference, reports, and CLI | `ijepa_texture_embeddings/ijepa_pipeline_2.py` |
| Mesh/texture pairing and VTK rendering primitives | `render_clam.py` |
| Surface mask, deterministic crop selection, and shared sampler | `patch_sampling.py` |
| Path resolution, L2 normalization, and thumbnail utilities | `util.py` |
| Pipeline unit tests | `../tests/unit/learned_color_space/test_ijepa_pipeline_2.py` |
| Sampler unit tests | `../tests/unit/learned_color_space/test_patch_sampling.py` |
| v2.1 manifest awaiting v2.2 regeneration | `ijepa_texture_embeddings/outputs_2_1/tables/patch_manifest.csv` |

## I-JEPA reference

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and
Ballas, N. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive
Architecture. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition*, 15619–15629. https://doi.org/10.1109/CVPR52729.2023.01501

## Manuscript-ready section

> **Editorial status:** The following prose is ready for insertion after the group-aligned
> v2.2 pipeline has been rerun and its outputs have been audited. It intentionally contains
> no numerical results from the incompatible v2.1 artifacts.

### Exploratory learned representation of rendered surface appearance

To complement the area-weighted color-composition representation, we performed a separate
offline analysis using frozen I-JEPA features to characterize local arrangements of color
and pattern on the shell exterior. This exploratory analysis used the original paired mesh
and texture image for each of the 32 freshwater mussel specimens rather than textures
transferred to the shared ColorAtlas UV parameterization. It therefore represented
rendered surface appearance and was not a test of atlas transfer accuracy or dense
anatomical correspondence.

Each specimen was rendered once from a standardized exterior viewpoint using an
orthographic camera, smooth surface shading, fixed material coefficients, supersampling
anti-aliasing, and a transparent background. To reduce arbitrary camera reversals across
specimens, we aligned each mesh to the camera frame of reference specimen `UF_IZ_438751`.
The major, minor, and surface-normal axes were estimated by principal component analysis
of mesh vertices. For each target mesh, the signs of the minor and normal axes were chosen
from the four possible combinations by minimizing their difference from the reference
mesh's standardized third-moment asymmetry signature, with weak world-frame agreement
used only to break close ties. This group-PCA procedure resolves principal-axis signs but
does not establish anatomical correspondence.

Source texture images were downsampled with Lanczos interpolation when their longest edge
exceeded 4096 pixels. Each alpha-defined render was cropped with 12 pixels of padding and
resampled to 2048 × 2048 pixels. The active compatibility profile stretched the cropped
content to the square canvas, thereby standardizing image occupancy while altering the
original projected aspect ratio. From each normalized render, we selected 64 deterministic
256 × 256-pixel crops. A crop was eligible only when every pixel was classified as rendered
surface at an alpha threshold of 10. Candidate locations were visited in a seeded,
specimen-specific random order, and a candidate was rejected when its
intersection-over-union with any accepted crop exceeded 0.6. Crops could therefore
overlap, but exact duplicate pixel crops were excluded. No reflection or photometric
augmentation was applied.

Crops were encoded with the frozen `facebook/ijepa_vith14_1k` ViT-H/14 checkpoint. The
model-specific processor resized each RGB crop to 224 × 224 pixels, rescaled channel values
to [0, 1], and normalized each channel with mean and standard deviation 0.5. The encoder
produced a 16 × 16 grid of 256 final-layer spatial tokens with 1,280 features per token.
Because the I-JEPA encoder has no class token, we averaged all spatial tokens to obtain one
raw 1,280-dimensional vector per crop. We then averaged the 64 raw crop vectors for each
specimen and L2-normalized the resulting specimen vector once. Crop vectors were not
normalized individually before aggregation, and no collection-specific training or
fine-tuning was performed.

For descriptive visualization, we standardized each of the 1,280 specimen features across
the collection before fitting two-component principal component analysis. We fitted
two-component Uniform Manifold Approximation and Projection separately to the
unit-normalized specimen vectors using cosine distance, 10 neighbors, `min_dist = 0.1`,
one execution thread, and random seed 42. We also identified the five nearest non-self
specimens by cosine similarity in the original specimen embedding space. These analyses
were fitted to the complete collection and were not used to estimate out-of-sample
performance.

This representation should be interpreted as an unordered summary of local rendered
appearance rather than a texture-only descriptor. Although strict surface-only crops
remove background and direct silhouette boundaries, the encoded pixels retain effects of
local geometry, shading, foreshortening, texture projection, rasterization, and image
acquisition. The aggregation also discards each crop's anatomical location. Accordingly,
the resulting PCA, UMAP, and neighbor relationships provide an exploratory description of
this collection; without independent labels, expert similarity judgments, or held-out
validation, they do not establish taxonomic recovery, biological validity, or superiority
to ColorAtlas or another learned representation.
