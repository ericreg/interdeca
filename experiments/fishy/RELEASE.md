# Fish paper reproduction bundle

This directory includes the actual 250 synthetic specimens, July 16 atlas outputs,
all aligned source meshes and atlas landmarks, and seven original donor scan
textures with their decimated meshes. It runs the pinned author's downstream
analyses for Figures 6–13. It does not regenerate synthetic textures or reconstruct
the historical atlas. Historical geometry defects are preserved.

## Data contract

OBJ vertices, triangle order, per-corner UV indices, PNG pixels, landmark JSON and
CSV rows are copied byte-for-byte; no mesh repair, rescaling or reordering occurs.
The atlas has 24,999 vertices and 49,997 triangles. Archived resampled meshes may
have expanded vertex indexing and degenerate triangles; the pinned research
reader preserves these inputs rather than applying current-core mesh guards.
The analysis consumes archived OBJ coordinates directly, without a coordinate
conversion. Absolute physical length units are not established by the available
provenance, so projection distances are reported in archived model units.
All 250 source Markups files and the atlas Markups file declare `LPS`; their
metadata and coordinates are preserved. The reference's historical landmark
handling is retained, without claiming new coordinate-registration validation.

Texture coordinates are triangle-corner UVs in the shared atlas or the specimen's
native texture layout. Reference face-color extraction flips V for image row
indexing, selects nearest pixels at the three UV corners, averages RGB, and stores
uint8, exactly as the pinned paper source. Ground-truth rows join by exact sample
ID; specimen output ordering follows the pinned source's sorted texture names.
Hue labels are the author's two-cluster inference from stored continuous hue
values; original mixture-component labels are unavailable. Seven donor identities
and the 250 specimen IDs are explicit in `specimens.json`.

The bundle is relocatable. Its commands resolve data relative to `reproduce.py`;
no original computer paths, credentials, old virtual environment, or pre-existing
cache are required. Historical documents inside `source/` retain their original
path descriptions for provenance; those paths are not runtime dependencies.
Use the commands below from the bundle directory. Run outputs must be new paths.

## Fresh environment

Install `uv` using its official installation instructions, then:

```sh
uv python install 3.12.8
uv venv --python 3.12.8 .venv
uv pip install --python .venv/bin/python \
  -r source/experiments/fishy/reference/paper_synth/requirements-paper.txt \
  -r source/experiments/fishy/requirements-tooling.txt
export PYTHONNOUSERSITE=1
.venv/bin/python reproduce.py verify
```

On Windows use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.
In PowerShell set `$env:PYTHONNOUSERSITE = "1"` instead of `export`.
Python and every numerical dependency are pinned. The unchanged 29 author
scientific pins are separate from the five integration validation pins in
`requirements-tooling.txt`. Pydantic validates metadata in the runner; archived
numerical scripts retain their original imports and dependencies. Blender, Slicer, the InterDeCA
core's environment and a Rust compiler are unnecessary for these analyses.
Creating the environment needs package-index access unless its wheels are cached.
The datasets themselves are all included. `manifest.sha256` records the manifest
digest; retain that value independently when transferring the bundle.

## Representative smoke

```sh
.venv/bin/python reproduce.py smoke --output runs/smoke
```

This starts with empty caches and uses one actual specimen from each of the seven
donors. It loads original meshes and textures, extracts atlas face colors,
projects 1,024 atlas-face locations per specimen onto the aligned source mesh,
samples the native texture, computes CIEDE2000, independently checks the color
formula, and writes `smoke.json`, fresh NPZ arrays and `smoke-fidelity.png`.
It reports agreement with the included October 6 local result separately from
execution success. A successful smoke is not a full 250-specimen reproduction.
Scientific child processes disable the user site and use fresh writable cache
directories beneath the requested output, independent of home-directory caches.

## Full study and figures

```sh
.venv/bin/python reproduce.py prepare --output runs/full
.venv/bin/python reproduce.py run --output runs/full --stage all
```

Preparation verifies exact 250-ID joins and hashes every input. All commands
validate the bundle inventory before using it. Execution checks the exact source
and scientific package versions. Fresh preparation leaves result/cache folders
empty and moves the committed author JSON to `results_ref` for comparison.

The full run is CPU/memory intensive: the previously measured macOS run took about
40 minutes for the 11 required analyses and eight figures; platform-specific
timings vary. Allow roughly 15 GiB for the bundle, additional space for a fresh
environment/run, and at least 16 GiB RAM. Run one full analysis at a time.
This is an estimate from prior execution, not a promise about another computer.

Stages can run individually, or use `--stage analyses`, `--stage figures`, or
`--stage figures-7-13`. Restarting an upstream stage invalidates dependent success
records. E1/build-cache restarts clear E6's dependent partial checkpoints; direct
E6 recovery retains its same-input checkpoint. A failed or interrupted stage
cannot preserve a full completion marker. Original result files remain inspectable.

| Paper figure | Required analysis stage(s) | Figure stage |
| --- | --- | --- |
| 6: dataset | build_cache and original donor scans | fig1_dataset |
| 7: fidelity | e1_fidelity | fig2_fidelity |
| 8: correspondence | e1b_correspondence | fig2b_correspondence |
| 9: preservation | e2_preservation, e2b_donorblocked, e2c_landmark_baseline | fig3_preservation |
| 10: exploratory analysis | e3_module_eda | fig4_module_eda |
| 11: descriptors | e4_byoa_descriptors | fig5_byoa |
| 12: expert feedback | e5b_expert_repr | fig6_expert |
| 13: limits | e6_negative_results, e6b_limits | fig7_limits |

`build_cache` is required before every analysis. Both independent author E2
verification programs can be run afterward using `--stage verify_e2_preservation`
and `--stage verify_e2_followup`. They are not silently counted as part of `all`.

## Numerical validation

The acceptance tolerance is declared before a fresh run: relative tolerance
**1e-6**, per numeric JSON leaf and maximum array absolute error divided by maximum
reference magnitude. Elapsed-time fields are excluded. Object arrays remain
unverified without enabling pickle; numeric arrays are compared with pickle off.
Comparison failure is recorded and is not converted to an execution pass.

```sh
.venv/bin/python reproduce.py compare --output runs/full --reference local
.venv/bin/python reproduce.py compare --output runs/full --reference author
```

These create separate `comparison-local.json` and `comparison-author.json` reports.
Exit 1 means a missing/differing required result. The local reference is the fresh
October 6 macOS reproduction. The author reference is extracted from pinned Git
commit `bf36233da21bb38e93ebe565fa99666c7ad2f51e`, including original NPZ arrays.
The documented local run **differs from the author** at strict tolerance, so author
comparison failure is expected and must be reported honestly. Cross-platform
differences may also affect comparison with the local reference; do not increase
tolerance after inspecting results to manufacture agreement.

Source settings, dataset counts, means and figure reproduction can be supported
without claiming identity to every historical author number. Original input hashes,
platform/BLAS details and warm-cache provenance are missing. The source paper also
understates composition-based cheek change (+0.061 versus the claimed maximum
0.012), and its ordinary paired t-tests/SEM treat correlated CV folds as independent.
The pinned author code is retained without silently rewriting those analyses.

## Export and inventories

```sh
.venv/bin/python reproduce.py export --output runs/full --destination exports/full
```

Exports include fresh results, figures, logs, metadata, comparison reports and an
output SHA256 inventory. `input-inventory.json` lists all 1,517 actual input assets;
Export refuses incomplete runs or missing required JSON, NPZ or figure files,
and checks the run's author revision and input hashes against this bundle.
`specimens.json` identifies the 250 specimens and seven donors. `manifest.json`
hashes payload files and gives the exact current source-tree digest. The source
snapshot includes uncommitted audit fixes; `source-changes.patch` records tracked
changes relative to the source HEAD. New packaging tools are included directly.

`provenance.json` distinguishes original author outputs, fresh October 6 results,
and future reconstructions. Data redistribution rights and repository-wide code
licensing were not established for this snapshot; this local research bundle does
not grant public redistribution permission. No external upload was performed.

## Rebuilding this package from the research checkout

```sh
python -m experiments.fishy.release build \
  --run /path/to/the/archived/paper-run \
  --upstream-checkout /path/to/3d_color_package \
  --output /path/to/a/new/bundle
```

The builder copies actual hash-pinned inputs, source working files, original Git
result blobs and prior local numerical evidence. It excludes environments, caches,
and unrelated `docs/paper/` manuscript files. `--data-only` supports an early data
transfer; repeat without that flag to finish an interrupted package. Completed
bundles are immutable and require a new output path for rebuilding.

## Metadata compatibility

The integration runner validates version 1 explicitly for `run.json`, `bundle.json`
and bundle `manifest.json`. Missing, boolean, string and unsupported versions fail
before stage/output allocation. Historical unversioned completion, interrupted
build, input inventory, specimen, provenance and export documents have explicit
legacy readers. They are not silently treated as version-one bundle manifests.
Old optional run timestamps, analysis metadata and thread-environment entries
retain their values and field presence when run state is rewritten. Hash checks
remain independent of these schema checks, and the source-list digest uses the
same historical JSON representation.

Existing immutable bundles retain their original sources and environments.
To consume their version-one metadata with the current runner, install the
current checkout's `experiments/fishy/requirements-tooling.txt` beside the bundle's
unchanged author requirements, then invoke the current checkout:

```sh
/path/to/python -m experiments.fishy.release --bundle /path/to/old/bundle verify
```

New builds include the records module, serde and tooling requirement pins in the
inventoried source snapshot. Do not edit a finished bundle to add these files;
build a fresh bundle when distributing the current runner.
