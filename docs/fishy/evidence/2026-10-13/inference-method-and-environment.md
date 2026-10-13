# E2 inference claim verification and supplemental analysis

Reviewer/implementer: `/root/boundary_review_final`. Source worktree: `/Users/leyangloh/dev/interdeca`, HEAD `0d9fe7df1beaa67e82624a4268409fc4d4d33863`. This lane authored only `experiments/fishy/inference.py`, `tests/fishy/test_inference.py`, and `docs/fishy/inference.md`; no attributed reference source, historical arrays, frozen bundle, or old result was edited. The root coordinator owns commits, evidence publication, and broader PR disposition.

## Verified claim

The immutable reference `experiments/fishy/reference/paper_synth/verify_e2_followup.py` (SHA256 `7b3af5cbd30100b212d9e5b989781e0483edca8b69a42e17f8ed9927d30f496f`) computes 20 scores with `RepeatedStratifiedKFold(n_splits=5, n_repeats=4)`, then applies `scipy.stats.ttest_rel` to paired descriptor scores and reports `mean ± 1.96 × sample-SD / √20`. All nine saved t statistics, p values and interval endpoints independently match that arithmetic using n=20 and 19 t degrees of freedom. The numerical calculations are historical diagnostics, but the 20 folds overlap heavily and are not 20 independent experimental units. They cannot establish calibrated population p values or 95% confidence intervals. [Bengio and Grandvalet (JMLR 2004)](https://www.jmlr.org/papers/v5/grandvalet04a.html) show why naive cross-validation variance estimates can understate uncertainty under fold dependence. The reference remains untouched.

## Authored supplement

`inference.py` adds nine fixed comparisons inherited from the reference follow-up using **all seven** donor scans in `LeaveOneGroupOut`. For each donor, before/after classifiers use identical train/test indices; `StandardScaler` is fitted on training rows only. Binary class support is checked separately for train and held-out donor. The JSON records donor sizes, class counts, donor-disjoint train sets, split-index SHA256 digests, per-donor balanced accuracies and differences. If any donor lacks two classes, its score is explicitly undefined and no seven-donor aggregate is reported. Otherwise the aggregate is the unweighted mean of the seven donor scores; no p value or confidence interval is computed.

The saved E2 descriptors are reused unchanged. Some descriptors, including pre-module axial sign canonicalization, were constructed using the full 250-specimen cohort. This is a held-out **classifier/scaler** comparison, not a prospective end-to-end new-donor experiment. The seven fitted classifiers also reuse most training specimens. The seven donors are too few and have no documented probability-sampling design for calibrated population inference. The `post_spectral` contrast has a different descriptor dimension; `post_axial` matches the pre-module axial dimension. These limitations are stated in the executable protocol and `docs/fishy/inference.md`.

## Bounded real-array evidence

Command, from repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /private/tmp/interdeca-research-grade-2026-10-13/venv/bin/python -m experiments.fishy.inference \
  --arrays data/fishy/releases/2026-10-12/reference/local-2026-10-06/results/e2_preservation_arrays.npz \
  --output /Users/leyangloh/dev/.planning/research-grade-review-2026-10-13/inference-real250-r8.json
```

The clean environment combines the 29 author scientific pins with five declared tooling pins. Its observed runtime was Python 3.12.8 (Clang 19.1.6), macOS-26.5-arm64-arm-64bit, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.8.0, Pydantic 2.13.4, and threadpoolctl 3.6.0; OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, and MKL_NUM_THREADS were each set to 1 for the run. The protocol records LogisticRegression C=1.0, l1_ratio=0.0, solver=lbfgs, tol=0.0001, fit_intercept=True, class_weight=balanced, max_iter=4000, and random_state=0. The saved input has 250 specimens and donor sizes 40, 31, 48, 37, 36, 28, 30. All nine contrasts scored all seven donors; train/test label support was present in each fold. The JSON has no p/interval fields and was emitted with `allow_nan=False`.

| Post − pre, factor | Historical 20-fold mean | Seven-donor macro difference |
| --- | ---: | ---: |
| axial − canonical spatial, stripe | +0.164777 | +0.102269 |
| axial − canonical spatial, cheeks | +0.045477 | +0.037211 |
| spectral − canonical spatial, cheeks | +0.127491 | +0.044009 |
| axial − canonical spatial, belly | −0.005000 | +0.000000 |
| axial − canonical spatial, tail | −0.000997 | −0.011364 |
| matched histogram − native histogram, belly | −0.004042 | −0.004202 |
| matched histogram − native histogram, tail | −0.012260 | −0.024116 |
| matched histogram − native histogram, stripe | −0.001600 | +0.028381 |
| matched histogram − native histogram, cheeks | +0.060780 | +0.046202 |

The columns use different held-out designs, so their differences are descriptive sensitivity to evaluation protocol, not tests of one design against the other. The historical column comes directly from `docs/fishy/evidence/2026-10-06/results/verify_e2_followup.json`; the new column is in `inference-real250-r8.json`.

Identity: input NPZ SHA256 `627b7cf52c5a69078e11a6126be806368d8516b96b0bd13d6bc0bb898acc99a3`; authored executable SHA256 `24444763c43729fef977be439ded2f9b6ba53243642fed271949fd087c1f2a14`; protocol/ordered-contrast SHA256 `cbcc619846edaf57d586ec72c65f2de188f55d2f5743acb2765259b87906964d`; output JSON SHA256 `b41522f7f8339f9dbf5c1775a26865d11bb25a2c92cb11da5561e6e3f8a9d454`. The executable hashes the NPZ both before and after reading to reject a changed input. The output path is exclusive-create so historical evidence cannot be overwritten accidentally.

## Behavioral validation

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/interdeca-research-grade-2026-10-13/venv/bin/python -m unittest discover -s tests/fishy -p test_inference.py -v` passed **7/7** tests. They check all-donor paired split coverage and macro interpretation, spy on scaler fits to prove only training rows are used for both descriptors, expose missing class support, reject too few groups, empty text identifiers and nonfinite features before output allocation, decode byte-string donor IDs, and compare two complete runs byte-for-byte. Ruff 0.16.10 lint/format checks, Python 3.12 compilation and `git diff --check` passed. Tests use small synthetic arrays; the bounded real-array run above supplies the separate cohort check. No mesh, rendering, atlas construction, model download, GPU job, or database was used.

Current authored file SHA256: test `86561a30809d19050faf92d08cafe4e6bf128f19e919aba3a2e9c56d32c3acea`; documentation `c20fb94c55c91e7e435c744968025199aabf3502bce54c8f7970240643242f51`. An independent reviewer reconstructed all 63 donor splits, class counts, index digests, paired differences and equal-donor means, and refit one real donor pair (pre 0.59375, post 0.9375). The reviewer repeated the audit against the final formatted source and r8 JSON; all non-source fields match r7 exactly. Coordinator evidence publication is pending.
