# Supplemental E2 donor-held-out contrasts

The archived `verify_e2_followup.py` computes `scipy.stats.ttest_rel` and a
`mean ± 1.96 × SD / √20` interval from 20 repeated stratified folds. Those
folds repeatedly reuse the same 250 specimens, so their score differences are
dependent. The reported t statistics, p values and interval endpoints are
historical diagnostic calculations, **not calibrated population inference**.
The [Bengio–Grandvalet variance result](https://www.jmlr.org/papers/v5/grandvalet04a.html)
explains why naive cross-validation variance calculations can understate
uncertainty when training and test sets overlap. The attributed reference and
its saved results are preserved unchanged.

`experiments.fishy.inference` supplies an additional, descriptive check from
the existing E2 descriptor arrays. For each of the nine fixed before/after
comparisons inherited from the reference follow-up, it holds out each of the
seven donor scans once.
Both descriptors use the identical train/test specimen indices in each fold.
A separate `StandardScaler` and balanced logistic classifier are fitted on
training specimens only. A held-out donor must contain both binary label
classes, and the remaining donors must also contain both; otherwise that fold
is marked undefined and the comparison has no aggregate. Every donor's size,
class support, score difference and split-index digests are saved.

The aggregate is the **unweighted mean of the seven donor scores**: each donor
gets one vote, even though the donor sizes range from 28 to 48 specimens.
This is a cohort description, not a p value, confidence interval, or estimate
of variability for a population of donor scans. Even the seven held-out fits
reuse most training specimens. The saved descriptors also predate this check:
some were constructed using the whole 250-specimen cohort (for example the
pre-module axial sign alignment). Only classifier fitting and scaling are
donor-held-out here. It is therefore not an end-to-end prospective test on new
donors. The `post_spectral` comparison also uses a different feature dimension
from its pre-module baseline; the `post_axial` comparisons match dimensions.

Run the bounded CPU check from the repository root using a Python 3.12.8
environment containing both the unchanged author requirements and the separate
`experiments/fishy/requirements-tooling.txt` metadata requirements described
in `experiments/fishy/RELEASE.md`. Use a **new** output path:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.fishy.inference \
  --arrays data/fishy/releases/2026-10-12/reference/local-2026-10-06/results/e2_preservation_arrays.npz \
  --output /tmp/fishy-e2-donor-contrasts.json
```

The command refuses to overwrite an existing result. Its strict JSON records
the input NPZ SHA256, executable source SHA256, and a SHA256 of the protocol and
ordered comparison list. The source file and saved E2 arrays are read only.
The behavioral tests are runnable without meshes, rendering, or a GPU:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests/fishy -p test_inference.py -v
```
