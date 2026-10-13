# Fishy provenance and reproduction boundaries

**Historical October 1–2 record.** The October 5 upstream update supplies the
paper analyses and figure mapping that were missing during this earlier work.
See [the October 6 reproduction verdict](paper-reproduction.md) for the current
execution, numerical comparisons and remaining limits. The earlier baseline
findings below remain separate evidence.

## Source revisions

- Generator: Alek Spiridonov's `color_fishy_tps.py` from
  `bshi42/3d_color_package`, branch `fishy_experimentation`, commit
  `3db23a1de18eb2a24d073e9a571edda290e5937f`.
- Baseline area features and legacy corner sampling: `fishy_research/src/fishpipe/`
  (`features.area_hist`, `mesh.sample_face_colors`) at the same revision.
- Current integration target: `ericreg/interdeca` starting at
  `8da7f4206795e6f16aa58753833d8ccac27b12bd`.
- Legacy workflow comparison: `3d_color_package:demo` at
  `464edfb77836ee192741391c5fcde0982c1ad0b0`.

Eric's independent repository mirrors the demo workflow but does not record its
original source branch/hash. These are comparison baselines, not a claim of Git
ancestry. The old repository is not imported or modified at runtime.

## Preserved generator behavior

Parameter draws retain NumPy RandomState/MT19937 ordering, including make_blobs
consumption. The canonical frame uses semantic landmark groups, a proper Kabsch
rotation, front/back distance normalization and posterior-origin translation.
Pigment placement uses r² log(r+1e-12), regularization 1e-6, and central-difference
local stretch with step 1e-5. Brushes retain the original positions, widths and
compositing order. Stripes are painted last. Geometry and UV order are preserved.

Changes at the execution boundary are explicit: local RNG rather than resetting
global state; typed records; required semantic anchors; RAS/LPS-aware readers;
validated manifests; named samples and saved true mixture labels; CPU emission
baking rather than a GPU requirement; and no automatic UV creation. Blender mesh
construction uses arrays instead of an OBJ importer that could reorder geometry.
These changes require image-level checks on actual donor data before claiming
pixel-identical generation across Blender/library versions.

## Evidence gaps that prevent a complete paper reproduction claim

The supplied draft describes 250 samples distributed over seven donor shapes,
116 cheek-positive samples and transfer-validation results. The experiment branch
README/config instead describes 250 textures on one toy mesh. Its methods note
identifies the toy dataset as a precursor and says the multi-donor cichlid run is
pending unless its output directory is located. Its cichlid generator defaults
to 25 samples. Archived results from those different datasets are not interchangeable.

### Archived-run leads checked on 2026-10-01

The project Slack now provides evidence of a later decimated cichlid synthetic
run. A [July 17 message](https://app.slack.com/archives/C09D45GJ8RX/p1784252763094329)
links `cichlid-synth-decimated.zip` and
`cichlid-synth-decimated-output.zip` on the project's Georgia Tech SharePoint.
The author reports generating synthetic textures and processing them through
the module. The thread reports a slight bend in the output and suspects the
local atlas implementation; it does not establish the cause.

These are candidate source and atlas archives, not yet verified paper artifacts.
The initial download requests returned HTTP 401. Browser access subsequently
succeeded, and the user supplied the extracted input archive locally. The input
inspection below supersedes that access limitation; the output archive was subsequently supplied and inspected locally, as recorded
below. The message supplies
no sample count, donor distribution, seed, parameter CSV, code revision, or
figure-to-run mapping. In particular, it does not establish a 250-sample run.

The [July 6 request](https://app.slack.com/archives/C09D45GJ8RX/p1783375493659149)
asks for synthetic fishy colors on decimated cichlid meshes. This supports the
purpose of the later run, but does not resolve its settings. The earlier
[February 6 run](https://app.slack.com/archives/C09D45GJ8RX/p1770401432217679)
explicitly has 25 samples and must remain separate.

Preserve both original archives and compute SHA256 checksums. The input ZIP
and extracted files have now been compared; the output ZIP has been hashed and
the analysis assets extracted with ZIP CRC checking. Verify the complete sample list, seven donor IDs,
47 landmark labels, prepared mesh/UV layout, parameter CSV and 116 cheek-positive
count against the paper. Pair every source texture with its atlas result by
sample ID, inspect the reported deformation, and recover the original settings
and figure scripts before marking this run as the paper reference. Store large
inputs under ignored `data/fishy/`; replace private handoff links with a stable,
licensed data release before publication.

### Local input inspection on 2026-10-01

The downloaded ZIP SHA256 is
`0f98285425ebb6d32db51cb8f57f0d5391857ffd7f8e4a2d1798b5b8b9dc2a60`.
All 752 extracted files matched their ZIP members by SHA256. The original CSV
SHA256 is `b038405d6b07e78f27c5116b831bbce157ae85209662cfc53eb6c28db06d4400`.

The supplied `cichlid-synth-decimated` directory contains 250 unique CSV rows,
250 OBJ meshes, 250 landmark JSONs, and 250 decoded 2048×2048 RGB PNGs, paired
exactly by sample tag. There are no missing CSV cells. Donor counts are
Mchenga_m1 40, Nimbo_f1 31, Nimbo_f2 48, yellowhead_m1 37, yellowhead_m2 36,
yellowhead_m4 28, and yellowhead_m5 30. Exactly 116 rows have rosy cheeks;
198 have four stripes and 52 have five. These counts support identifying this
as the candidate paper input population, but do not prove a figure-to-run mapping.

All landmark files contain 47 points and declare LPS coordinates. For
`yellowhead_m2`, the label `" TA"` has a leading space. For `yellowhead_m4`,
`RBWL` appears twice and `RBWR` is absent. Their 64 samples report 46 landmarks
used; the other 186 report 47. The original generator takes exact-label
intersections and the first occurrence of duplicate labels. Correcting these
files before regeneration would change the historical computation. The current
core rejects the duplicate and inconsistent label sets. Anatomical correction
requires evidence; neither dropping labels nor guessing a replacement is part
of the import command.

The source README states seed 49, 250 samples, 2048 resolution, TPS lambda 1e-6,
and CPU emission baking. At seed 49, current donor assignment matches every
CSV row, and base/stripe parameters match to 1e-12. Divergence starts at the
belly hue draw and affects later fields: all 250 belly/tail hues and 122 cheek
presence values differ. The source/environment difference causing this remains
unresolved. Earlier sampler parity checks were same-environment code comparisons,
not a match to this archived CSV. Preserve the CSV and do not recover mixture
labels from the mismatching regenerated sequence.

The reproducible `import-inputs` command writes a hash-pinned manifest and audit
under ignored `data/fishy/`. The original data remain unchanged and the 250
specimens remain separate even where donor geometry is repeated. The original environment/configuration and figure scripts still prevent a complete
reproduction claim. The source-backed hue-label protocol now supports the explicit
baseline described below; it does not establish the September figure protocol.

### Local atlas-output inspection on 2026-10-01

Alek's Slack message explicitly pairs the input ZIP with
`cichlid-synth-decimated-output.zip`. Its SHA256 is
`a3f49bb87098ed6c683800f23b5d56a3dc0796f93ac575676bad56cf6712a55c`.
The ZIP contains 1,757 files (14,521,530,126 uncompressed bytes), rooted at
`2026_07-16_20_37_11`. The local inspection extracted only `colorAnalysis/`
and `ATLAS/alignedLMs/` (754 files), retaining the original ZIP. ZIP reads checked
member CRCs and the extracted files have individually recorded SHA256 hashes.

All 250 atlas textures, resampled UV OBJs, and aligned landmark files match the
input CSV IDs exactly. `average_texture.png` is an additional population image
and is excluded from the manifest. Every specimen texture decoded successfully
as 2048×2048 RGB. A `legacy-atlas` manifest now pairs these assets by sample ID.
This establishes paired archived outputs, not their use in any particular paper figure.

The atlas has 24,999 vertices and 49,997 triangles and passes the current core's
mesh reader. All 250 resampled OBJs instead have 149,991 vertices (three per
triangle) and 49,997 triangles. Their raw face indices do not equal the atlas's;
per-corner UVs differ from the atlas by at most 2.9800415024539006e-8. The legacy
writer's VTK export path is consistent with expanded vertex indexing and
float32 texture coordinates, but this comparison alone does not prove complete
geometric correspondence or transfer fidelity.

All 250 resampled OBJs contain numerically degenerate triangles under the current
core's area criterion: 30–628 per specimen. Independent raw-OBJ inspection
confirmed this is in the archived geometry, rather than introduced by the new
reader. No source meshes were repaired or labels altered. These observations do
not establish the cause of the slight bend reported in Slack.

The default `extract` mode retains its valid-geometry and exact topology/UV
requirements. The explicit `--archived-uv` mode now provides a narrow texture-only
contract for this legacy export: positive sequential expanded vertex/UV indices,
matching triangle counts and exact ordered equality to float32-rounded atlas
UVs. It rejects arbitrary UV changes or face/corner reordering. Specimen geometry
is retained and audited, not repaired or substituted. Colors, areas and centroids
use the valid original shared atlas; degenerate specimen faces are saved in a
separate mask. The compatibility audit records any pixel-address differences
caused by UV rounding. The full 250-sample extraction completed and matched an independent PNG
calculation byte-for-byte. Five corner pixel addresses per texture differ under
UV rounding; extraction consistently uses original atlas coordinates. These
results do not claim historical geometry or transfer fidelity is validated. Figure-specific label definitions, scripts and the historical environment remain unresolved.

The generator source blends deterministic noise **before** applying the opaque
base brush. The draft describes a 10% blend without establishing that placement.
The source cheek radius is 0.035 while the draft mentions 0.037. Preserve source
behavior until the actual run configuration resolves these differences.

The source branch derives hue labels with KMeans on sampled hues. This bundle
retains the true generating mixture assignments. For overlapping hue groups,
those labels can differ; a reproduction must state which labels its archived
scores used. Do not silently substitute either protocol.

The current Interdeca core differs from the paper's legacy code in atlas spline,
color sampling, some clustering and averaging choices. `transfer` is labeled
`interdeca` and requires downstream revalidation. `extract` separately implements
the legacy corner sampling convention. An archived ATLAS manifest supports
repeating downstream work without rerunning atlas construction.

## Completed downstream baseline

A read-only remote-head check still resolves `fishy_experimentation` to
`3db23a1de18eb2a24d073e9a571edda290e5937f` (July 6). Alek's June 15 Slack reply
points to that branch's `fishy_research`; its checked-in experiments are the
precursor toy-mesh work. The later manuscript describes analyses not traceable
to a complete seven-donor figure script there.

`label-archive --protocol legacy-kmeans` ports `fishpipe.data._kmeans_1d_labels`:
KMeans with two clusters, `n_init=10`, `random_state=0`, then increasing-center
label order. On the archived scalar hues it yields belly counts 123/127 and tail
121/129, exactly matching Table 3. This supports using the protocol as an explicit
baseline, but count agreement does not establish the exact figure labels or
recover generating mixture IDs. Labels are saved separately from original data.

Both an atlas-fitted baseline and a matched native-palette comparison completed
for all 250 specimens. See [results.md](results.md) for measured scores, settings,
checks and remaining limits. No settings were adjusted to match paper values.

## Figure-level status

| Paper content | Implemented here | Evidence still needed |
| --- | --- | --- |
| Synthetic dataset / Figure 6 | Generator plus verified archived prepared meshes, landmarks, textures and original 250-row parameter table | Original generating environment/configuration; same-seed color mismatch unresolved |
| Transfer / Figures 7–9 | Paired original ATLAS archive, all 250 textures extracted and independently checked; current-core runner separate | Numeric pigment rules, fidelity/correspondence scripts, historical geometry defects |
| Population / Figure 10 | Completed 250-fish corner extraction, area-vector baseline and PCA coordinates | Exact module settings, GMM/BIC and dip-null figure script |
| Recovery / Figure 11 and Table 4 | Completed native/atlas composition recovery, random and donor-blocked CV, nulls and split exports | Exact twelve-descriptor script/configuration and reference scores |
| Expert feedback / Figure 12 | Not migrated without its figure-to-source mapping | Verified composition/custom representation, budgets, response protocol and restart seeds |
| Limits / Figure 13 | Not claimed reproduced | Exact dimensionality/sample-size/ICA/quantization comparison scripts and outputs |

This status table is intentional: no placeholder figures, invented threshold
rules or borrowed toy benchmark metrics are shipped as paper results. The available source, Slack handoff and archives were checked; no verified September
figure scripts were located. The user confirmed they do not have them.

## Publication gate

Before tagging a paper release, resolve the figure-to-script mapping, archive
licensed data with stable URLs/checksums, pin the actual code/environment and
rerun the supported commands from a clean checkout. Add reference metrics and
citation metadata only after those facts are known. Large data are external
artifacts, not a second code repository. No new browser interface or generic
experiment framework is part of this deliverable.

## Source availability clarification

The broader [source recheck](source-audit.md) confirms substantial precursor
analysis code exists. The missing item is a verified handoff for the exact paper
run (scripts/configuration/commands), not all descriptor or expert-feedback code.
