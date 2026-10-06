# Synthetic paper reproduction — October 6, 2026

This branch integrates Alek's October 5 paper analyses into InterDeCA and reruns
those analyses on the 250-specimen July 16 archive. It preserves the upstream
numerical code and settings, saves fresh results, and compares them with the
committed author reference. Execution success and numerical parity are separate
verdicts: the run has observable differences and must not be described as an
exact reproduction of every paper number.

## Execution and strict comparison

All **11 required analysis stages** and **8 figure stages (Figures 6–13)** exited
successfully. Both independent E2 verification programs also completed. All eight
rendered PNGs were visually inspected. Their original author layout is retained,
including legends placed inside data areas; this is not a typesetting redesign.
The numerical comparison exits **1**, correctly reporting differences.

| Result JSON | Strict status | Leaves outside relative tolerance 1e-6 |
| --- | --- | ---: |
| `cache_meta.json` | IDENTICAL | 0 / 4 |
| `e1_fidelity.json` | DIFFERS | 165 / 305 |
| `e1b_correspondence.json` | CLOSE | 0 / 279 |
| `e2_preservation.json` | DIFFERS | 57 / 582 |
| `e2b_donorblocked.json` | DIFFERS | 11 / 188 |
| `e2c_landmark_baseline.json` | DIFFERS | 171 / 515 |
| `e3_module_eda.json` | DIFFERS | 4 / 567 |
| `e4_byoa_descriptors.json` | DIFFERS | 106 / 1313 |
| `e5b_expert_repr.json` | DIFFERS | 25 / 822 |
| `e6_negative_results.json` | DIFFERS | 175 / 2936 |
| `e6b_limits.json` | CLOSE | 0 / 599 |

Elapsed-time fields are excluded. Numerical NPZ fields from all eight required
array files were also compared; full differences and maximum errors are retained
in [comparison.json](evidence/2026-10-06/comparison.json). Two supporting
experiments that do not feed these figures are explicitly excluded. Array error
is normalized by the maximum reference magnitude, while JSON tolerance is
assessed at each leaf. No strict pass is inferred from rounded figure labels.

## Source, environment and inputs

The numerical reference is
[`bf36233da21bb38e93ebe565fa99666c7ad2f51e`](https://github.com/bshi42/3d_color_package/tree/bf36233da21bb38e93ebe565fa99666c7ad2f51e/fishy_research/paper_synth)
from `fishy_experimentation`, copied under `experiments/fishy/reference/` with
individual source hashes. This replaces the earlier missing-source assessment;
October 1–2 baseline reports remain explicitly historical. The integration starts
from InterDeCA `8da7f4206795e6f16aa58753833d8ccac27b12bd`.

Analysis and figure execution use Python **3.12.8** and every exact version in
Alek's `requirements-paper.txt` (including NumPy 2.4.6, SciPy 1.17.1,
scikit-learn 1.8.0 and UMAP 0.5.12), on macOS ARM64. Initial preparation used the
system Python 3.14 interpreter; preparation performs file validation and hashing,
and that interpreter is recorded separately from the analysis interpreter.
Library-version agreement does not establish an identical operating system or
BLAS implementation. The runner retains the author thread policy.

The inputs are the supplied `cichlid-synth-decimated` meshes, textures, landmarks
and original CSV, plus the paired `2026_07-16_20_37_11` module outputs. All 250
aligned source meshes and the atlas landmark file were additionally extracted
from the original output ZIP, with hashes recorded. All six asset groups and the
CSV join by exact specimen ID. Analyses start with empty results and cache
folders; committed author outputs are moved to `results_ref` for comparison.
This recomputes downstream analyses, without rebuilding the archived atlas.

For Figure 6, the original seven scan textures were recovered from
`/Volumes/Seagate/School/3D_Fish/Fish/Texture/`. The corresponding UV-preserving
decimated meshes are already in the synthetic input archive: the updated generator
copies its input OBJ unchanged (`color_fishy_tps.py`, lines 1014–1019). One of
those archived mesh copies per donor was staged with its original scan PNG in
`data/fishy/donor-scans/seagate-2026-10-06/`. File copies were hash-checked; no
mesh, UV or texture transformation was applied. This reconstruction of the donor
folder is documented separately from an original folder handoff.

## Selected numerical findings

The fidelity median is **0.890101** dE2000 versus the reference **0.890116**;
mean **1.837926** versus **1.837725**, p95 **3.638578** versus **3.639335**, and
fraction below 2.3 **0.923551** versus **0.923540**. Aggregate agreement does not
establish per-face parity: 542,840 of 12,499,250 face scores differ by more than
1e-6, with maximum absolute difference about 97.99. Fidelity arrays and hole
classifications remain different.

The E2 headline composition-transfer and spatial-gain balanced accuracies match
the reference exactly for all four factors:

| Factor | Composition transfer change | Best spatial atlas gain | Dimension-matched spatial gain |
| --- | ---: | ---: | ---: |
| Belly | −0.004042 | +0.007808 | −0.005000 |
| Tail | −0.012260 | −0.000997 | −0.000997 |
| Stripe | −0.001600 | +0.173396 | +0.164777 |
| Cheeks | +0.060780 | +0.127491 | +0.045477 |

Other E2 scores differ, including raw native-spatial donor accuracy (**0.96375**
versus **0.99540**). E4's 52 primary classifier scores have maximum absolute
difference **0.001087**. E5's representation ceilings match exactly, its tag
curve values differ by at most **6.39e-6**, and pairwise-warp observations differ
by up to **0.01443**. UMAP coordinates, display silhouettes, some null scores,
landmark projections and exploratory scores also differ. These are documented
observations, rather than values corrected to match the paper.

## Investigation and independent checks

- **Axial direction:** 37 native raw profiles and all 250 atlas raw profiles
  match the author arrays after reversing the longitudinal bins, with residuals
  at most 1.21e-13. All 250 canonical native profiles match directly to 1.03e-13.
  The flip flags differ for exactly 37 specimens. This identifies a sign/direction
  difference in saved arrays; it does not prove the historical environment identity.
- **Closest-point candidate ties:** on the first specimen's 49,997 faces,
  reversing or shuffling source corner order changes 2,005 or 2,084 reference
  samples, mostly at tied six-neighbor boundaries. This demonstrates the upstream
  fidelity construction's order sensitivity. It does not establish that every
  author/local discrepancy has this cause; original author input hashes are absent.
- **Independent color calculation:** original CIEDE2000 agrees with
  `skimage.color.deltaE_ciede2000` on 200 random Lab pairs, maximum error 7.11e-14.
- **Author independent E2 verifier:** exact specimen order and ground-truth join;
  all 40 independently rescored classification values match the local E2 run.
  Pigment areas, raw axial profiles and canonical profiles match independently;
  matched-palette descriptors differ by at most 5.76e-11. Both author E2 verification
  programs exit successfully. Their execution does not imply all historical scores
  match or certify the statistical interpretation of overlapping CV folds.
- **Execution contracts:** disposable fixtures reject existing runs, same-size
  input edits, mismatched exact-ID joins and missing donor pairs. Invalid joins
  allocate no output; invalid attachments leave the run record unchanged. Donor
  attachment preserves prior stage status and refuses replacement.
- **Code checks:** Ruff and formatting for authored experiment modules, Python
  compilation, CLI help in the appropriate environment, and Git whitespace checks.
  No permanent test suite was added under the repository's initial-test policy.

The upstream exploratory SigClust field contains an undefined NaN. The strict
comparison represents undefined observations explicitly and does not replace
missing or undefined evidence with a score of zero.

## Rerun and evidence

See [the runnable instructions](../../experiments/fishy/PAPER_REPRODUCTION.md)
for preparation, exact environments, stage selection, figure mapping and strict
comparison. The local run is `data/fishy/paper-runs/2026-10-06/`, containing staged
source, input hashes, package versions, logs, fresh JSON/NPZs, PNG/PDF figures,
verification scripts and detailed comparison. Large inputs and arrays stay
Git-ignored. The published evidence bundle below contains bounded numerical and
figure outputs for review, with source/result/figure hashes.

### Published artifacts

[Execution summary](evidence/2026-10-06/summary.json),
[portable run and 1,517-asset provenance](evidence/2026-10-06/run-provenance.json),
[artifact SHA256 manifest](evidence/2026-10-06/manifest.json),
[original scan recovery](evidence/2026-10-06/donor-scan-provenance.json),
[headline and independent checks](evidence/2026-10-06/checks/headline-comparisons.json),
[axial-direction diagnostic](evidence/2026-10-06/checks/axial-signs.json), and
[candidate-tie diagnostic](evidence/2026-10-06/checks/fidelity-ties.json).
Fresh author-format result JSONs are retained under `evidence/2026-10-06/results/`.
The bundle preserves undefined NaN values in original upstream-format result
files; the comparison, summary and provenance use strict JSON.

| Paper figure | Regenerated PNG |
| --- | --- |
| 6 — Dataset | [Donor scans and synthetic dataset](evidence/2026-10-06/figures/fig1_dataset.png) |
| 7 — Fidelity | [Photometric fidelity](evidence/2026-10-06/figures/fig2_fidelity.png) |
| 8 — Correspondence | [Anatomical correspondence](evidence/2026-10-06/figures/fig2b_correspondence.png) |
| 9 — Preservation | [Native, landmark and atlas comparisons](evidence/2026-10-06/figures/fig3_preservation.png) |
| 10 — Population | [Built-in EDA](evidence/2026-10-06/figures/fig4_module_eda.png) |
| 11 — Descriptors | [Representation comparisons](evidence/2026-10-06/figures/fig5_byoa.png) |
| 12 — Expert simulation | [Tagging and pairwise feedback](evidence/2026-10-06/figures/fig6_expert.png) |
| 13 — Limits | [Dimensions, sample size and negative results](evidence/2026-10-06/figures/fig7_limits.png) |

## Remaining reproduction boundaries

This run cannot certify regeneration of the original synthetic textures or a
new current-core atlas transfer. Earlier seed-49 parameter divergence, landmark
label discrepancies and archived degenerate resampled triangles remain recorded
in [the historical provenance](provenance.md). No labels or meshes were repaired.
To establish exact historical numerical identity, obtain Alek's original input
hashes, platform/BLAS details and any warm-cache provenance, then resolve the
reported discrepancies against that evidence. The retained source and fresh local
outputs make that comparison possible.
