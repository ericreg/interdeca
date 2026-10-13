# Source availability recheck

## October 6 update

The earlier search below describes the October 1–2 repository state.
`fishy_experimentation` advanced to `bf36233da21bb38e93ebe565fa99666c7ad2f51e`
on October 5. Its `fishy_research/paper_synth/` now contains the Figure 6–13
mapping, seven-donor analyses, fidelity masks, landmark baseline, donor-blocked
evaluation, expert-guided and limit experiments, reference results and a pinned
environment. These files are integrated as a source-pinned reference under
`experiments/fishy/reference/`. See [the current reproduction verdict](paper-reproduction.md).
The historical findings below are retained for provenance; the newly supplied
analysis code is no longer absent.

Checked during the October 1–2, 2026 session after the user asked us to verify
whether code is actually missing before contacting Alek.

## Corrected conclusion

We have **not located the exact scripts/configuration that generated the current
manuscript's seven-donor analyses**. This is not proof that Alek lacks them or that
they were never written. The earlier statement that descriptor, population,
expert-feedback and limits code was missing was too broad: substantial precursor
implementations exist. What remains unverified is their mapping and adaptation to
the manuscript's archived cichlid run.

## Search coverage

- Fresh `git ls-remote --heads --tags` on `bshi42/3d_color_package`: nine branches,
  no advertised tags. Every returned head matches a locally available remote ref.
  `fishy_experimentation` remains `3db23a1de18eb2a24d073e9a571edda290e5937f`.
- All 25 local/remote refs were inventoried. Inspected 328 distinct source/document
  blobs at branch tips, plus filenames across all 163 reachable commits. A second
  targeted content search covered 463 historical source/document blobs. Vendor
  `ModelColors-main/` and `node_modules/` were excluded from that content search.
- GitHub pull-request listing with `--state all --limit 100` returned zero PRs.
- Eric's `ericreg/interdeca` advertises only `main`, at
  `8da7f4206795e6f16aa58753833d8ccac27b12bd`; no alternate analysis branch or tags.
- Both supplied ZIPs were inventoried, including entries not extracted locally.
  The input contains assets, CSV and a README; the output contains atlas assets.
  Neither contains Python/R/MATLAB scripts or notebooks.
- Live Slack: Alek's June 15 reply explicitly points to
  `fishy_experimentation/fishy_research`. A search of all his project-channel
  messages after July 6 returned ten messages, including the July 16 archives,
  but no newer analysis-code link. The September 28 question about who produced
  the fidelity figures has no threaded reply in the available channel export.
- The public repository listing for `bit-wrangler`, the account behind the
  Slack-linked expert demo, showed no identifiable replacement analysis repo.
  This does not enumerate inaccessible private repositories.

## Code that does exist

Paths below refer to revision `3db23a1` in `bshi42/3d_color_package`:

| Area | Existing code |
| --- | --- |
| Synthetic cichlid generation | `color_fishy_tps.py` |
| Basic population analysis | `fishy_research/experiments/exp01_baseline.py`, `src/fishpipe/embed.py`, `metrics.py` |
| Descriptor comparisons | `experiments/exp02_structure_descriptors.py`, `exp30_textons.py`; `src/fishpipe/features.py`, `spectral.py`, `textons.py`, `structure.py` |
| Expert feedback and active selection | `experiments/exp44_recover_grid.py`, `exp52_active_tags.py`, `_tag_harness.py`, `expert-tagging-demo/` |
| Dimensionality, quantization, ICA and other limits | `experiments/exp53_dims_vs_accuracy.py`, `exp54_k_vs_accuracy.py`, `exp55_ica_disentangle.py` and related scripts |

These are real implementations, not empty placeholders. They may be reusable
for further migration, even without an exact figure-to-source mapping.

## Specific unresolved evidence

The central `fishpipe/config.py` still defaults to `/mnt/data/ml_data/fishy_all`,
`fishy1.obj`, `fishy_*.png`, and the earlier `fishy/samples.csv`. That is the shared
mesh toy dataset, not a configured seven-donor archive run. The cichlid methods
note (lines 24–35) explicitly distinguishes the target dataset and describes its
module validation as pending unless matching output is identified. The supplied
July archive resolves the output-availability gap, but it contains no downstream
analysis driver or saved figure configuration.

The paper describes source-referenced closest-point texture fidelity, pigment
masks, a landmark correspondence baseline, matched native/atlas recovery and
leave-one-donor-out evaluations. Searches did not locate their corresponding
paper-run implementation/configuration. CIEDE2000 does exist in the Slicer
application, so absence of that primitive is not the issue. The earlier
`metrics.py` uses five-fold stratification; the manuscript's four repeated runs
and donor-blocked comparison are not identified by that implementation alone.
The published descriptor comparison also uses settings different from the
checked-in `exp02` comparison; an exact match has not been established.

This is evidence of a missing **reproduction handoff** in the inspected sources,
not evidence that every method must be implemented from scratch. Private repos,
unpushed local scripts, deleted/unreachable Git history, or other shared folders
remain outside this verified search coverage.

## Suggested question for Alek

> I found the Fishy analysis code in `fishy_experimentation/fishy_research` and
> the July 16 cichlid input/output archives. Which scripts and configuration
> generated the current paper's Figures 6–13, especially texture-transfer fidelity
> and the seven-donor native/atlas and leave-one-donor-out comparisons? Are those
> runs produced by the existing scripts with different settings, or is there a
> later branch/notebook/local script we should use? A commit/path and run commands
> would let us reproduce the exact results.
