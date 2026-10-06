# Learned-color reproduction: October 6, 2026

**Executed successfully within the available scope.** The documented frozen
I-JEPA method completed on the 47 locally available native mussels, producing
3,008 crop vectors, 47 unit specimen vectors, PCA/UMAP coordinates and cosine
neighbor tables. This is a new, explicitly labeled method rerun. It does not
certify Eric's original 32-specimen experiment or reproduce unavailable figures.

## Source and method identity

- Author source: `bshi42/3d_color_package`, `learned-color-space` at
  `b30c495c701093e456231ec8a035012266ccef74`, October 3.
- Its parent is the requested `demo` head
  `464edfb77836ee192741391c5fcde0982c1ad0b0`.
- Eric's renderer, patch-selection helpers and technical description are
  attributed under `experiments/learned_color/reference/`. The pinned source
  manifest lists the two import/typing adaptations.
- The main integration independently implements the documented v2.2 group-PCA,
  alpha cropping, no-jitter sampling, frozen token pooling and specimen
  aggregation protocol. It is not the absent `ijepa_pipeline_2.py`.
- Encoder: `facebook/ijepa_vith14_1k` at immutable revision
  `f157467ea509bc356ff9f61fd3c0d840eec5e04e`, float32, eager attention, explicit MPS
  device. The Mac device choice is recorded rather than equated to Eric's GPU.

## Executed population

All 47 exact-stem OBJ/texture pairs in the supplied `Mussels` folder were used.
No 32-specimen subset was guessed. Every input mesh and texture was hashed.
The common camera-reference specimen is `UF_IZ_438751`. Each 2048² exterior
render was cropped with 12-pixel padding and square stretching, then sampled
into 64 distinct 256² all-surface patches, with maximum pairwise IoU 0.6.
Frozen 256-token means yielded 1280-dimensional raw crop vectors. Their
specimen means were normalized once.

The first two standardized PCA components explain 16.5532% and 15.2891% of
full-cohort variance. These are descriptive summaries, not taxonomic accuracy
or out-of-sample performance. UMAP used cosine distance, ten neighbors,
`min_dist=0.1`, seed 42 and one execution thread. Five non-self cosine neighbors
were recorded per specimen.

## Independent checks

- All 3,008 saved patches exactly match the corresponding source-render pixels;
  no flips or photometric augmentation occurred. Every patch pixel exceeds the
  documented alpha threshold, all background fractions are zero, and every
  patch SHA256 matches its manifest. Each specimen has exactly 64 patches.
- The maximum pairwise crop IoU is 0.599063, below the 0.6 limit.
- Independently averaging the saved raw vectors in float64 and normalizing them
  matches the 47 saved vectors with maximum absolute difference `7.90e-8`.
  All crop and specimen values are finite, and specimen order joins exactly.
- Independently computed cosine-neighbor rankings match all saved rows.
- Applying the saved scaler and PCA transform independently reproduces coordinate
  values within `1.87e-5`, consistent with float32 fitted transform arithmetic.
- A same-cohort config with expected count 32 is rejected before execution.
- A modified saved patch is rejected before model loading; original artifacts
  were restored and their checksums reverified.
- Ruff checks/formatting, Python compilation, CLI help and Git whitespace checks
  passed for the integration. Per the repository's initial implementation policy,
  no permanent automated test suite was added. Focused disposable checks and
  their numerical receipts are retained beside the local run.
- An exterior render and the exported population figure were visually inspected.
  The export avoids overlapping specimen labels; exact IDs remain in the CSV.

Bounded evidence is checked in under
[`evidence/2026-10-06/`](evidence/2026-10-06/): input/source/environment provenance,
independent checks, coordinates, neighbors, summary and a figure. Native inputs,
rendered specimen images, crops, checkpoint weights and full embedding matrices
remain outside Git in `data/learned-color/`.

## Commands and artifacts

The actual local run used the separate Python 3.12.8 environment:

```sh
data/learned-color/env/bin/python -m experiments.learned_color \
  --config data/learned-color/local-47.json --stage all
MPLBACKEND=Agg data/learned-color/env/bin/python -m experiments.learned_color.plot \
  --run data/learned-color/runs/2026-10-06-47 \
  --output data/learned-color/exports/2026-10-06-47
```

`local-47.json` is an ignored machine-specific configuration. The portable paper
example and environment instructions are in the
[experiment README](../../experiments/learned_color/README.md).
`data/learned-color/runs/2026-10-06-47/` retains complete outputs, model and
processor metadata, exact input hashes and a completion marker. A completion
marker establishes this configured run only.

## Still needed for original-paper certification

1. Eric's referenced original pipeline and active configuration.
2. The exact 32 specimen IDs and corresponding input checksums.
3. Original model revision, output arrays/figures and the figure-to-run mapping.

The source document explicitly distinguishes existing v2.1 outputs from the
v2.2 protocol. Neither missing output set is replaced by this 47-specimen run.
The independent implementation is available for review and execution while the
author handoff is completed.
