# July 16 Fishy archive: completed baseline

**Historical October 1–2 record.** The October 5 upstream update supplies the
paper analyses and figure mapping that were missing during this earlier work.
See [the October 6 reproduction verdict](paper-reproduction.md) for the current
execution, numerical comparisons and remaining limits. The earlier baseline
findings below remain separate evidence.

All 250 paired specimens have been imported, audited, extracted and evaluated in
Eric’s InterDeCA repository. This is a completed composition baseline, including
a matched native/atlas comparison. It is **not a complete reproduction of the
paper’s Figures 6–13**. The exact September figure scripts/configuration have not been located, and remaining
methods cannot be certified from the earlier toy-mesh source alone.

## Matched comparison

Mean balanced accuracy; random CV uses four repeats of five stratified folds.
Donor CV holds each of seven donor shapes out once and averages folds equally.
The same native-fitted 24-color Lab palette and exact split indices are used in
both spaces. Supervised standardization is fitted inside each training fold.

| Factor | Native random CV | Atlas random CV | Native donor CV | Atlas donor CV |
| --- | ---: | ---: | ---: | ---: |
| Belly hue | 1.000 | 1.000 | 1.000 | 1.000 |
| Tail hue | 0.986 | 0.983 | 0.985 | 0.980 |
| Stripe count | 0.673 | 0.687 | 0.633 | 0.656 |
| Rosy cheeks | 0.549 | 0.582 | 0.564 | 0.579 |

The matched atlas random-CV shuffled baselines and observed-minus-null scores are:

| Factor | Observed | Shuffled labels | Observed − shuffled |
| --- | ---: | ---: | ---: |
| Belly hue | 1.000 | 0.423 | 0.577 |
| Tail hue | 0.983 | 0.417 | 0.566 |
| Stripe count | 0.687 | 0.428 | 0.259 |
| Rosy cheeks | 0.582 | 0.570 | 0.012 |

Belly and tail recovery is strong with this descriptor and these inferred labels.
Cheek recovery is close to its shuffled baseline; these scores do not establish
reliable cheek recovery. A single shuffled label assignment per factor is only a
diagnostic baseline, not a calibrated significance test. Repeated folds overlap,
so these are not independent replicate experiments.

Native vectors use native face areas and atlas vectors use shared atlas areas.
Differences therefore combine geometry weighting, pixel sampling and transfer.
They cannot be read as a pure transfer-error or fidelity measurement.

## Relationship to the draft

The draft’s Table 4 composition row reports 1.00, 0.98, 0.66 and 0.58 for these
four factors. The matched atlas run gives 1.000, 0.983, 0.687 and 0.582. This is
broad agreement for some factors, not an exact reproduction. The separate
atlas-fitted palette baseline gives 0.999, 0.982, 0.704 and 0.599. Both runs are
retained; no configuration was selected or adjusted to force agreement.

The label protocol is the earlier source’s scalar KMeans rule: two clusters,
ten initializations, seed zero, increasing-center label order. It yields the
draft’s belly 123/127 and tail 121/129 counts exactly. Original stripe and cheek
labels come directly from the archived CSV (198/52 and 134/116). Belly/tail
labels are explicitly inferred groups, not recovered make_blobs assignments.

## Settings and artifacts

- Palette: 24 Lab centers, 200,000 uniformly sampled native face observations,
  KMeans `n_init=5`, seed 0; pooled across all specimens without factor labels.
  This is a transductive palette, not a palette trained only on held-in fish.
- Feature: surface-area histogram, L2 normalized. Legacy nearest-pixel triangle
  corners averaged to uint8, using native UVs or original shared atlas UVs.
- Classifier: standardized logistic regression, `C=1`, balanced class weights,
  `max_iter=2000`; seed 0 for splitting and one shuffled-label assignment.
- Input manifest: `data/fishy/imports/july16-verified/manifest.json`.
- Output manifest: `data/fishy/outputs/july16/manifest.json`.
- Verified extraction: `data/fishy/extracted/july16-archived/`.
- Explicit labels and provenance: `data/fishy/labels/july16-legacy-kmeans/`.
- Atlas-fitted baseline: `data/fishy/evaluated/july16-composition/`.
- Native palette/vectors: `data/fishy/evaluated/july16-native-palette/`.
- Atlas vectors on that palette: `data/fishy/evaluated/july16-native-palette-atlas/`.
- Paired scores, split indices and validation:
  `data/fishy/evaluated/july16-matched-comparison/`.

These local data directories are Git-ignored. Each completed stage contains
input/source hashes, package versions and a completion marker. The supplied ZIPs
and source textures/meshes were retained unchanged. Reproduction commands are in
[the experiment README](../../experiments/fishy/README.md#completed-archive-analysis).

## Validation and remaining limits

All 250 atlas color arrays matched an independent PNG calculation byte-for-byte.
Native vector checks covered all seven donors. All 108 paired folds matched,
donor-blocked folds had disjoint donors, and palette/identity mismatches and
overwrites were rejected. See [verification.md](verification.md).

Remaining requirements for a full paper claim:

1. Figure-to-script mapping and exact settings for the twelve descriptors,
   population multimodality analysis, expert simulations and limit experiments.
2. Numeric pigment masks and closest-point/reference construction settings for
   source-referenced transfer fidelity; qualitative descriptions are insufficient
   to reproduce the reported thresholds and failures.
3. Original generating environment/configuration to explain the seed-49 color
   mismatch, plus working Blender execution for new image-level generation.
4. A documented resolution of archived landmark discrepancies and degenerate
   resampled triangles before claiming a validated current-core transfer rerun.

The archive-specific reader verifies exact ordered float32 UV correspondence
and records 74,511 degenerate face instances; it does not repair those meshes.
Five corner pixel addresses differ under UV rounding at 2048², and the extraction
consistently uses original shared atlas UVs. Opaque alpha coverage is not proof
of a successful bake. See [provenance.md](provenance.md) for archive hashes and
source evidence.
