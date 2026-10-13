# Fishy experiment reproduction

For Alek's October 5, 2026 scripts mapped to Figures 6–13, use the
[source-pinned paper reproduction runner](PAPER_REPRODUCTION.md). The notes
below document the earlier generation and composition-baseline implementation.

This directory is part of **ericreg/interdeca**, not a separate repository or
independent package. It contains the synthetic cichlid generation and pipeline
steps needed for the paper. The old shared-mesh toy benchmark, browser demos,
and unrelated exploratory experiments are deliberately excluded.

**Current scope:** deterministic cichlid parameter/brush generation, headless
Blender emission baking, an Interdeca atlas/transfer runner, archived-atlas
input support, legacy corner-color extraction and composition-baseline recovery.
The July 16 synthetic-input archive has been inspected: 250 samples, seven
donors, and 116 cheek-positive rows match the draft counts. The paired atlas-output archive is now available and all 250 sample IDs match.
An explicit archived-UV extraction mode handles its verified VTK export layout.
Historical geometry defects and figure-level source artifacts remain unresolved. The input audit found landmark
label problems and a same-seed parameter mismatch; see the provenance notes.
This bundle does **not** claim to reproduce all published numbers or
Figures 7–13. See [provenance and remaining evidence](../../docs/fishy/provenance.md).

See the [verification record](../../docs/fishy/verification.md) for checks passed
and the current Blender runtime limitation.

## Environment

From the repository root:

```sh
uv sync --locked
uv run python -m experiments.fishy --help
```

Use the root `uv.lock`; no `pip install` inside Blender or Slicer is needed.
Blender is an external executable. Supply `--blender /path/to/blender` to generation
or transfer, or use Interdeca's existing executable discovery. Generation uses
CPU Cycles, a BYTE_COLOR corner attribute, emission-only baking and native donor
UVs. Blender's exact version is recorded in each run. Do not upgrade packages and
assume old image hashes will remain identical.

## Input data

Supply the seven **prepared, triangulated, UV-preserving** donor meshes and their
47 labeled landmarks. The example assumes this layout (large files are ignored):

```text
data/fishy/donors/
  Models/Mchenga_m1.obj
  Models/Nimbo_f1.obj
  Models/Nimbo_f2.obj
  Models/yellowhead_m1.obj
  Models/yellowhead_m2.obj
  Models/yellowhead_m4.obj
  Models/yellowhead_m5.obj
  LMS/<matching-name>.mrk.json
```

The paired July 16 archives and their locally measured hashes are documented in
`docs/fishy/provenance.md`; their public redistribution license remains unverified. Original
photogrammetry textures are not needed to paint synthetic colors; the prepared
meshes' native UVs and landmarks are required. Do not replace prepared meshes with
arbitrarily decimated originals and call that the archived experiment.

Edit `configs/cichlid.example.json`. Paths are relative to the **configuration
file**, not the shell's working directory. Verify model units and mesh/landmark
coordinate conventions. Landmark files honor RAS/LPS metadata unless explicitly
overridden. All generated files use RAS. Optional asset `sha256` fields enforce
published hashes; the resolved hashes are always recorded during generation.
Donor order controls seeded donor assignment and must be preserved.

The example combines the checked-in generator's seed/distributions with the
paper's requested sample count and transfer settings. Its profile is deliberately
named `cichlid-rerun-unverified`: it is not a recovered original configuration.

## Import the archived synthetic inputs

Use the downloaded, extracted `cichlid-synth-decimated` folder directly:

```sh
uv run python -m experiments.fishy import-inputs \
  --source /path/to/cichlid-synth-decimated \
  --output data/fishy/imports/july16 --seed 49
```

This hashes all 750 paired assets, decodes every image, validates the CSV and
sample joins, checks repeated-donor asset identity, and compares the archived
parameters against the current sampler. It writes a `manifest.json`, `audit.json`,
unchanged copies of the source CSV and README, and execution provenance.
The manifest references the source files by absolute path; retain that folder.
Reimport to a new output directory when moving it to another machine.

A completion marker means the **import** completed. Inspect `transfer_ready` and
`transfer_errors` in the audit before attempting transfer. The July 16 inputs
currently fail the core's exact landmark correspondence requirements. No labels
are trimmed, deduplicated, renamed, or discarded during import.

The archived CSV is the authoritative parameter table. It lacks belly/tail
mixture assignments, and the current sampler disagrees with its color draws at
seed 49. Consequently `samples.original.csv` is deliberately not an evaluation
labels file: do not attach current-sampler labels or silently infer them with
KMeans. Archived atlas outputs can avoid rebuilding the historical transfer,
using the explicit `--archived-uv` compatibility mode described below. The
source-backed `legacy-kmeans` protocol below permits an explicitly inferred-label
baseline; exact figure provenance remains unavailable. See the output inspection
in the provenance notes.

## Run generation and the current Interdeca pipeline

Each stage requires a **new** output directory. Completed stages have
`complete.json`; partial runs retain logs but never receive that success marker.
No stage overwrites prior results or silently resumes them.

```sh
uv run python -m experiments.fishy validate \
  --config experiments/fishy/configs/cichlid.example.json

uv run python -m experiments.fishy generate \
  --config experiments/fishy/configs/cichlid.example.json \
  --output data/fishy/runs/run01/generated

uv run python -m experiments.fishy transfer \
  --config data/fishy/runs/run01/generated/config.json \
  --manifest data/fishy/runs/run01/generated/manifest.json \
  --output data/fishy/runs/run01/atlas

uv run python -m experiments.fishy extract \
  --manifest data/fishy/runs/run01/atlas/manifest.json \
  --output data/fishy/runs/run01/extracted

uv run python -m experiments.fishy evaluate \
  --extraction data/fishy/runs/run01/extracted \
  --labels data/fishy/runs/run01/generated/ground_truth/samples.csv \
  --config experiments/fishy/configs/evaluation.example.json \
  --output data/fishy/runs/run01/baseline
```

For a small processing check, copy the generator configuration and reduce samples
and resolutions. A small generation run may have too few examples of a factor
for stratified classification; that is a clear error, not a reason to silently
change folds or report a favorable score. The full evaluation profile uses four
repeats of stratified five-fold CV and leave-one-donor-out where applicable.
A single-class held-out donor fold has no two-class balanced-accuracy estimate;
its score is recorded as null with the fold indices retained.

## Reuse an archived ATLAS run without rebuilding its atlas

The `transfer` command runs **Eric's current numerical core**. It is not a wrapper
around legacy ATLAS and does not claim its output is numerically equivalent.
There is no MorphoWeave backend or hidden fallback.

To analyze an existing legacy ATLAS result, write a manifest next to that run
and pass it directly to `extract`, skipping `generate` and `transfer`:

```json
{
  "schema_version": 1,
  "kind": "atlas",
  "backend": "legacy-atlas",
  "atlas_mesh": {"path": "atlasModelUV.obj", "sha256": "<actual 64-character hash>"},
  "atlas_landmarks": {"path": "atlas.mrk.json"},
  "samples": [{
    "sample_id": "Mchenga_m1_fishy_000000",
    "donor_id": "Mchenga_m1",
    "mesh": {"path": "resampled/sample.obj"},
    "landmarks": {"path": "resampled/sample.mrk.json"},
    "texture": {"path": "atlasTextures/sample.png"}
  }]
}
```

Include **all samples**, use actual paths and hashes, and retain the real sample
IDs. Extraction verifies face/vertex order and per-corner UVs against the atlas.
Opaque textures cannot identify failed bakes by alpha; black is retained as valid
pigment. This command does not claim to measure source-referenced bake failures.
For the July 16 VTK archive, use the explicit compatibility mode:

```sh
uv run python -m experiments.fishy extract \
  --manifest data/fishy/outputs/july16/manifest.json \
  --output data/fishy/extracted/july16-archived --archived-uv
```

This is restricted to `legacy-atlas` manifests. It requires sequential expanded
vertex/texture indices and **exact equality** between ordered specimen UVs and
the shared atlas UVs rounded to float32. Reordered faces/corners, changed UVs,
nonfinite coordinates and invalid references fail. It does not use a general
numerical tolerance or repair any source file. The default mode still enforces
valid specimen geometry and exact topology fingerprints.

Colors, face areas and centroids use the original shared atlas. Historical
specimen geometry is inspected only to write `archived_degenerate_faces.npy`
and the per-sample `archive-compatibility.json`. The audit also counts pixel
addresses that would differ if rounded specimen UVs were used. `valid.npy`
remains alpha coverage only, not a geometry-quality or transfer-fidelity mask.
Successful extraction does not establish that a collapsed face transferred
color correctly or that the paper's reported results have been reproduced.

The evaluation CSV requires `sample_id`, `donor_id`, `belly_group`, `tail_group`,
`stripe_count` and `rosy_cheeks_present`. Groups/presence are 0 or 1; stripe count
is 4 or 5. Rows join by ID, never by CSV order. The generator now saves the actual
`make_blobs` mixture assignments. Old CSVs that omitted them must recover them
from the matching original generator/environment; inferring overlapping tail
groups with KMeans is a different labeling protocol.

## Completed archive analysis

The local 250-fish results and limitations are recorded in
[results.md](../../docs/fishy/results.md). The historical branch provides a
specific hue-label protocol: scalar KMeans, two clusters, ten initializations,
seed zero, labels ordered by increasing center. This reproduces the draft's
belly/tail group counts. It is explicitly inferred labeling, **not** recovery of
missing generating mixture IDs. The user does not have the September figure scripts. Their exact source mapping
has not been located; this baseline cannot certify those figures.

The following commands run from the repository root. Use fresh output directories
if these stages already exist. The shared palette is fitted without labels on
native colors; both spaces use that identical palette and identical CV splits.

```sh
uv run python -m experiments.fishy label-archive \
  --csv data/fishy/imports/july16-verified/samples.original.csv \
  --protocol legacy-kmeans --output data/fishy/labels/july16-legacy-kmeans

uv run python -m experiments.fishy native-palette \
  --manifest data/fishy/imports/july16-verified/manifest.json \
  --config experiments/fishy/configs/evaluation.example.json \
  --output data/fishy/evaluated/july16-native-palette

uv run python -m experiments.fishy evaluate \
  --extraction data/fishy/extracted/july16-archived \
  --labels data/fishy/labels/july16-legacy-kmeans/labels.csv \
  --config experiments/fishy/configs/evaluation.example.json \
  --palette data/fishy/evaluated/july16-native-palette/palette_lab.npy \
  --output data/fishy/evaluated/july16-native-palette-atlas

uv run python -m experiments.fishy compare-composition \
  --native data/fishy/evaluated/july16-native-palette \
  --atlas data/fishy/evaluated/july16-native-palette-atlas \
  --labels data/fishy/labels/july16-legacy-kmeans/labels.csv \
  --output data/fishy/evaluated/july16-matched-comparison
```

`compare-composition` checks exact sample/donor order, palette equality, matching
configuration, finite normalized vectors and one-to-one labels before scoring.
Its `comparison.json` contains raw fold scores, split indices and mean observed,
shuffled and null-corrected scores. Native areas and atlas areas differ; this
comparison combines geometric weighting, texture sampling and transfer effects.
It is not a source-referenced transfer-error estimate. Omitting `--palette` from
`evaluate` retains the earlier atlas-fitted baseline as a separate analysis.

## Outputs and interpretation

Generation writes paired `meshes/`, `landmarks/`, `images/`, separate
`ground_truth/samples.csv`, configuration, Blender logs and a hash-pinned manifest.
Transfer writes the atlas, aligned/resampled meshes, projection distances and
atlas-space textures. It keeps every generated specimen's contribution to the
mean rather than deduplicating repeated donors.

Extraction writes `colors.npy` (specimens × faces × RGB uint8), explicit
`valid.npy`, atlas face areas/centroids, sample/donor IDs and a copy of the atlas.
Baseline evaluation writes the pooled Lab palette, L2-normalized area vectors,
PCA coordinates and a `recovery.json` with observed/shuffled/null-corrected fold
scores and exact split indices. It does not infer that visible PCA separation
establishes recovery of all factors.

The palette is fitted to all samples without labels, matching the research
harness's transductive descriptor convention. Logistic feature scaling is fitted
inside each training fold. Repeated random splits and donor-blocked splits answer
different questions; retain both and their own shuffled-label nulls.

Do not use this baseline alone as a replacement for the paper's source-referenced
transfer validation, twelve-descriptor comparison, expert feedback simulations
or reported figure scripts. Those must be linked to verified source artifacts
before the paper reproduction release can be declared complete.
