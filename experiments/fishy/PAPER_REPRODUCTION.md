# Synthetic paper reproduction

The paper's current synthetic analyses now live **inside InterDeCA**. The
numerical reference is Alek Spiridonov's
[`bf36233`](https://github.com/bshi42/3d_color_package/tree/bf36233da21bb38e93ebe565fa99666c7ad2f51e/fishy_research/paper_synth)
(October 5, 2026), copied under `reference/`. The previous generation and
composition bundle remains available through `python -m experiments.fishy`.

This pipeline recomputes analyses from the fixed July 16 synthetic inputs and
atlas outputs. It does not replace them with a newly built native or MorphoWeave
atlas, and it does not establish regeneration of the original synthetic textures.
Its experiment settings match the author reference; geometry defects in archived
surfaces remain part of those inputs.

## Environment

Use a separate environment to retain the author's exact Python and scientific
package versions without changing the InterDeCA core's environment:

```sh
uv python install 3.12.8
uv venv --python 3.12.8 data/fishy/paper-env
uv pip install --python data/fishy/paper-env/bin/python \
  -r experiments/fishy/reference/paper_synth/requirements-paper.txt
```

Run commands from the repository root. The runner executes the scripts directly,
so a separate `fishy-research` package installation is unnecessary. It verifies
reference hashes and package versions before executing a stage. The snapshot
keeps the original numerical implementation, including its research-specific
types, imports and caching rules, to support meaningful parity comparisons.

## Required data

- `cichlid-synth-decimated/`: 250 native textures, meshes, 47-landmark files and
  the original `samples.csv`.
- `2026_07-16_20_37_11/`: the fixed module-run directory, including
  `colorAnalysis/atlasModelUV.obj`, `atlasLM.mrk.json`, `atlasTextures/`,
  `resampledOBJ_withUV/`, and all 250 `ATLAS/alignedModels/*_align.obj`.
  Extract the aligned models from the supplied output ZIP as well; the earlier
  composition-only import did not need them.
- For Figure 6 only, the original seven donor meshes and scan textures in
  `3D_Fish/Downsampled_decimate/Models/` and `3D_Fish/Texture/`.
  Synthetic textures cannot substitute for original scan textures.

Preparation checks exact sample-ID joins and hashes every consumed asset.
Input and output directories must be disjoint. Keep inputs unchanged; subsequent
stages reject missing, modified or mismatched inputs.

## Run

```sh
data/fishy/paper-env/bin/python -m experiments.fishy.paper prepare \
  --source /path/to/cichlid-synth-decimated \
  --module-run /path/to/2026_07-16_20_37_11 \
  --output data/fishy/paper-runs/my-run

data/fishy/paper-env/bin/python -m experiments.fishy.paper run \
  --output data/fishy/paper-runs/my-run --stage analyses
data/fishy/paper-env/bin/python -m experiments.fishy.paper run \
  --output data/fishy/paper-runs/my-run --stage figures-7-13
```

Supply `--donor-scans /path/to/3D_Fish` during preparation to include original
scan panels. Then `--stage all` runs all analyses and Figures 6–13. Each stage is
also selectable by script name. If original scans become available after the
analyses, attach their validated, hash-pinned pairs between stages with
`attach-donor-scans --output /path/to/run --donor-scans /path/to/3D_Fish`,
then run `--stage fig1_dataset`. Existing run directories are refused during
preparation; failed stages retain logs. A zero exit status records execution;
strict numerical parity is a separate check below.

The runner follows the author's thread policy: e4/e5b receive four explicit BLAS
threads, e2/e6 retain their own defaults, and other scripts use library defaults.
It clears inherited global BLAS settings. Operating system and BLAS implementation
still affect outputs. The upstream README describes UMAP, ICA and logistic-fit
sensitivities; the execution verdict records observed additional discrepancies.

## Figure-to-source mapping

| Figure | Analysis | Rendering |
| --- | --- | --- |
| 6: dataset | Original donor scans and synthetic inputs | `fig1_dataset` |
| 7: photometric fidelity | `e1_fidelity` | `fig2_fidelity` |
| 8: anatomical correspondence | `e1b_correspondence` | `fig2b_correspondence` |
| 9: preservation | `e2_preservation`, `e2b_donorblocked`, `e2c_landmark_baseline` | `fig3_preservation` |
| 10: built-in EDA | `e3_module_eda` | `fig4_module_eda` |
| 11: descriptors | `e4_byoa_descriptors` | `fig5_byoa` |
| 12: expert guidance | `e5b_expert_repr` | `fig6_expert` |
| 13: limits | `e6_negative_results`, `e6b_limits` | `fig7_limits` |

`e3_builtin_eda`, `e5_expert_guided` and their verification scripts are retained
as reference material. These supporting runs are not silently counted as
executed by the figure reproduction.

## Compare and inspect

```sh
data/fishy/paper-env/bin/python -m experiments.fishy.paper_compare \
  --output data/fishy/paper-runs/my-run
```

This compares every JSON leaf for required analysis results and writes
`comparison.json`. Elapsed-time fields are excluded. Exit status 1 means a
required JSON file is missing or differs beyond relative tolerance `1e-6`.
When reference arrays are supplied, missing or differing required array files
also fail the comparison; supporting experiments outside the executed scope
are listed separately. It never
rewrites results to force a match. For array comparison, extract the author's
`paper_synth/results/*.npz` from the pinned upstream commit and add
`--reference-arrays /path/to/results`. Pickled object arrays are marked
unverified; numerical arrays are compared without enabling pickle. Array tolerance uses
maximum absolute error divided by the maximum reference magnitude; JSON
tolerance is assessed at each leaf.

Runs retain staged source, package versions, input SHA256s, stage logs, fresh
result JSON/NPZs and figures in Git-ignored `data/fishy/`. Reference result JSONs
are committed for offline comparison; large arrays and original figures remain
available at the pinned upstream commit. See
[`docs/fishy/paper-reproduction.md`](../../docs/fishy/paper-reproduction.md)
for the actual execution verdict and remaining evidence gaps.
