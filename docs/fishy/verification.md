# Fishy verification record

**Historical October 1–2 record.** The October 5 upstream update supplies the
paper analyses and figure mapping that were missing during this earlier work.
See [the October 6 reproduction verdict](paper-reproduction.md) for the current
execution, numerical comparisons and remaining limits. The earlier baseline
findings below remain separate evidence.

Checked on 2026-10-01 using the repository's locked Python 3.13.9 environment.
Per AGENTS.md, no new automated test suite was added; these were focused,
disposable numerical and integration smoke checks.

## Passed

- Ruff checks/formatting for the experiment modules; Python 3.12 grammar parsing.
- All 250 parameter vectors at seed 49 matched the extracted original
  `color_fishy_tps.py` sampler exactly in the same environment.
- Sampling preserved the caller's NumPy global RNG state.
- Canonical coordinates, inverse coordinate round trips and proper rotations
  matched the original generator on two synthetic donor landmark sets.
- TPS evaluation and four warped brush fields matched the original source within
  numerical tolerances (1e-12 for spline positions, 1e-9 for warped colors).
- Four specimens spanning two repeated synthetic donor shapes passed the core's
  reference selection, mean construction, alignment and surface resampling.
  All resampled meshes retained the common topology and UV fingerprint.
- A 40-specimen synthetic image fixture passed corner-color extraction and
  composition-baseline evaluation. Opaque black faces remained valid black.
- Reversed label CSV order still joined correctly by sample ID. All four factors
  exported 20 repeated-stratified folds and five donor-separated folds. Training
  and held-out donor IDs were disjoint in every donor-blocked fold.
- Existing output directories were rejected. A changed image failed its SHA256
  check before a new output directory was allocated.
- CLI help and donor configuration validation succeeded.

These fixtures check implementation contracts; they are not paper datasets or
scientific evidence for reported accuracy values.

## Not verified

The installed Blender 5.2.2 process crashed with SIGSEGV during native Metal/GPU
backend initialization. A separate minimal background command containing only a
Python print statement reproduced the crash before Python started. The crash
stack entered `supports_barycentric_whitelist` / `MTLBackend::metal_is_supported`.
Consequently neither emission baking nor the complete UV/bake stage could be
executed here. No mathematical or renderer fallback was introduced to conceal
that limitation; failed generation retains its log and has no completion marker.

## Archived input verification

The local July 16 input dataset has now been imported through `import-inputs`.
All 752 extracted files matched their ZIP members by SHA256. Disposable failure
checks confirmed duplicate-ID rejection, missing-asset rejection, refusal to write
inside the source, checksum enforcement and refusal to overwrite an existing run.
Ruff checks/formatting, CLI help and Python compilation passed for this addition.
All 750 assets were hashed, all 250 images decoded, sample IDs joined one-to-one,
and repeated donors checked for identical mesh/landmark bytes. Counts match the
draft: 250 samples, seven donors, 116 cheek-positive rows. The import writes an
inspectable audit and preserves the original CSV and README unchanged.

The core's actual readers confirmed that duplicate labels in `yellowhead_m4`
and differing labels in `yellowhead_m2` block current-core transfer. These were
not repaired. The current seed-49 sampler matches donor assignment and the first
seven parameter columns, but differs from the archived color parameters. This
limits the earlier same-environment parity finding; it does not validate
regeneration of the archived run or recovery of its missing hue-mixture labels.

Both archived inputs and paired atlas outputs are now available. All 250 output
textures decoded and matched the input sample IDs, with the separate population
average excluded. The shared atlas loaded successfully. Independent raw-OBJ
inspection of all 250 resampled meshes found expanded vertex indexing, UV
precision differences up to 2.9800415024539006e-8 and 30–628 degenerate triangles
per mesh. The strict core rejects the geometry and exact topology/UV fingerprints also
fail. The separate `--archived-uv` extraction contract handles only the observed
VTK expansion with exact float32 UV correspondence. No strict core guard was
disabled and no archived mesh was repaired. Inspection
scripts and full per-sample results are stored beside the ignored local output
manifest in `data/fishy/outputs/july16/`.

The original environment/configuration and complete figure scripts remain missing.
No full 250-specimen transfer, image-level generation parity or paper figure
reproduction has been verified. See provenance.md for exact findings.

## Archived-UV compatibility checks

Disposable focused fixtures confirmed that float32-rounded ordered atlas UVs
are accepted while face order changes, UV order changes, UV drift, invalid
indices, missing vertices and nonfinite positions are rejected. A collapsed
triangle is recorded in the archive mask while the ordinary core reader still
rejects it. Ruff, Python compilation and whitespace checks passed.

The full July 16 run completed at `data/fishy/extracted/july16-archived/`:
250 specimens × 49,997 faces × 3 uint8 channels. All 250 saved color arrays
matched an independent Pillow/integer-arithmetic extraction byte-for-byte.
Sample order matched the original CSV, all atlas face areas were finite and
positive, and every specimen's saved degenerate-face count matched the earlier
independent raw-OBJ audit (74,511 face instances in total).

Rounding atlas UVs to the archived float32 values changes five corner pixel
addresses per texture at 2048×2048. The output deliberately uses the original
shared atlas coordinates for every specimen; this difference is recorded in the
compatibility audit. All 12,499,250 face observations pass alpha coverage because
the source images are opaque RGB. This does not measure transfer success or
validate collapsed specimen geometry. `verification.json` and the disposable
verification scripts are retained alongside the extraction outputs.

Color extraction and the explicit source-backed recovery baseline have completed.
Full paper figures still require their descriptors, settings and figure scripts;
current sampler labels must not replace the mismatching archived sequence.

## Label and matched-comparison verification

- Executed the original source's `_kmeans_1d_labels` function against the archived
  hues: all 250 belly and tail labels exactly matched the port.
- Reversing the input CSV yielded a byte-identical labels CSV after canonical
  sample-index ordering; no archived source values were changed.
- Independently sampled one native texture from each of the seven donors with
  Pillow and integer corner averaging. Independent SciPy nearest-center assignment
  and area sums matched saved native vectors to 1e-12, as did the fixed-palette
  atlas composition function on those same inputs.
- All four factors exported 20 random and seven donor-held-out folds. All 108
  native/atlas fold pairs had identical observed and null splits. No train/test
  overlap, donor leakage in blocked folds, undefined scores or nonfinite scores
  were found. Independently rerunning atlas scoring exactly reproduced its saved
  fold records.
- Changed palettes and reordered specimen identities were rejected before output
  creation; an existing comparison directory was not overwritten.
- CLI stages, Ruff checks/formatting and Python compilation passed.

The local comparison directory retains `verification.json` and the disposable
`verify_comparison.py` script. These checks establish numerical implementation
consistency, not identity to missing September figure scripts. The one shuffled
label assignment per factor is a baseline diagnostic, not a permutation-test
p-value or uncertainty interval.
