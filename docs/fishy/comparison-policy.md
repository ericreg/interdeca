# Scientific comparison policy

The authored integration comparator reports `symmetric-elementwise-v2` separately
from archived comparison reports. It reads inputs without changing scientific
outputs, and it never adjusts values, aligns embeddings, enables pickle loading,
or changes a tolerance after inspecting a result to manufacture parity.

## Finite floating-point values

For each corresponding finite scalar or array element `a` and `b`, acceptance is:

```
abs(a - b) <= atol + rtol * max(abs(a), abs(b))
```

Defaults are `rtol = 1e-6` and `atol = 1e-8`. The relative default retains the
previous engineering setting. The absolute default is an engineering guard for
roundoff near zero in stored units; it is **not a scientifically established
measurement tolerance**, a claim that differences below it are biologically
irrelevant, or a universal bound for all quantities. Heterogeneous quantities
retain their stored units. Applications needing domain-specific tolerances must
predeclare and justify those separately.

The bound is symmetric when reference and current values are exchanged. Arrays
apply it independently at every element: an unrelated large value cannot hide a
small element's error. Complex finite elements use complex magnitude, including
both real and imaginary components. Scaled arithmetic avoids intermediate
subtraction overflow for finite values near the floating-point limit.

Reports identify both tolerances, the policy and criterion. Array diagnostics
include finite/undefined counts, the number outside tolerance, maximum absolute
and symmetric relative error, maximum error divided by its allowed tolerance,
the worst element index, and up to ten failing examples. A ratio above one fails.
`relative_to_reference_magnitude` is retained solely as a labeled historical
**diagnostic**; its global reference norm no longer controls acceptance. Counts
and worst-index diagnostics, rather than a dataset-wide norm alone, expose
near-zero and large-element discrepancies.

## Exact structure and undefined evidence

JSON object/list structure, empty containers, missing fields, list order, strings,
booleans, nulls and integers compare exactly. JSON `1`, `1.0` and `true` have
different stored types. Keys containing punctuation cannot collide with nested
paths. Only the pre-existing elapsed-time fields `seconds`, `elapsed`, `time`,
`runtime_s` and `wall_s` are excluded recursively.

NPZ keys, array shapes and stored dtypes compare exactly. Integer identifiers,
counts, booleans and strings use exact value comparison; large integers never
pass through floating-point conversion. A dtype change is a structural difference
even when the values happen to compare numerically. Array order is preserved.
No vertex, specimen, feature or embedding permutation is inferred.

Large numeric fields use 250,000-element chunks. NPZ decompression materializes
one input field pair; extended-precision calculations and diagnostic arrays are
bounded by the chunk size. Scalar totals and at most ten failure examples persist
between chunks. A forced-chunk regression checks counts, maxima and worst indices
against the direct formula, including Fortran-ordered and complex data; a
500,003-element regression verifies the scratch path never sees a larger chunk.

Matching NaN or signed-infinity patterns are explicitly `UNDEFINED`, with the
finite portion reported separately. They establish agreement about an undefined
stored observation, not complete scientific numerical parity. Replacing an
undefined value by a finite value, switching infinity sign, or changing a finite
component of a partially undefined complex number yields `DIFFERS`.

Object arrays remain `UNVERIFIED_OBJECT_ARRAY`. Files are always loaded with
`allow_pickle=False`; a controlled malicious-pickle regression verifies that
comparison never executes an object payload. Undefined or unsafe-object evidence
prevents the aggregate report from claiming complete parity: it is `INCOMPLETE`
when there are no known differences, or `DIFFERS` when differences also exist.
Both outcomes exit 1. Overall `numerical_parity` is `PASS` only when both every
required JSON file and every required reference array were assessed and agree.
Without `--reference-arrays`, arrays are not assessed, so JSON-only agreement is
reported separately as `json_parity: PASS` while the overall verdict is
`INCOMPLETE` and the CLI exits 1. A known difference in either part is `DIFFERS`.

## Sensitivity and preserved historical evidence

Use the same saved inputs at multiple **predeclared** absolute settings, for
example `atol = 0`, `1e-8` and `1e-7`, keeping `rtol = 1e-6`. Report the changed
counts and statuses for all settings; never choose the loosest successful setting
as the new scientific conclusion. `--atol 0` provides a stricter relative-only
profile. It does not reproduce the defective historical global-array gate.

The old JSON formula already used the symmetric maximum of both scalar
magnitudes, with a `1e-12` denominator floor. Its defect was the absence of an
explicit absolute tolerance near zero. The old array gate divided the single
largest error by the single largest **reference** magnitude: it was asymmetric
and could hide individual failures. Describing both old methods as asymmetric
would be inaccurate. The old flattening/type checks also lost empty containers
and could conflate discrete values. Historical reports and their counts remain
unchanged; updated reports state their new policy rather than relabeling old
results.

The default absolute guard does not make strict author reproduction succeed.
Observed author differences must remain reported, including differences that
could come from random states, library/platform changes, or non-identifiable
embedding coordinates. Procrustes/sign/permutation alignment may be useful as a
separately labeled exploratory analysis of embedding geometry; it is excluded
from strict parity and cannot replace the raw comparison.

## Commands

The existing run interface remains available:

```sh
python -m experiments.fishy.paper_compare --output /path/to/run \
  --reference-arrays /path/to/author/results --rtol 1e-6 --atol 1e-8
```

Compare existing directories without altering either run, writing a new report
outside all scientific input directories:

```sh
python -m experiments.fishy.paper_compare \
  --reference-results /path/to/reference/results \
  --current-results /path/to/current/results \
  --reference-arrays /path/to/reference/results \
  --report /path/to/new/comparison-atol-1e-8.json --rtol 1e-6 --atol 1e-8
```

`Compare.json(reference, current, tolerance)` and
`Compare.arrays(reference, current, tolerance)` retain their positional relative
argument. The new optional keyword is `atol`; `Compare.results(...)` creates a
report in memory without writing files. Finite nonnegative tolerances are required
before any file is read. The standalone behavioral suite uses only disposable
inputs:

```sh
python -m unittest discover -s tests/fishy -p test_comparison.py -v
```
