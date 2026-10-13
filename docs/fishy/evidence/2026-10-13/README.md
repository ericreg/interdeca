# Published clean-run and focused-refactor evidence

This directory publishes previously local records. Publication does not represent
a new full scientific run or a claim of original-paper numerical equivalence.
The run/bundle manifests identify the actual source snapshots, inputs and outputs;
they must not be described as execution of later commits merely because the
receipts are committed alongside those changes.

## Separate campaigns

| Comparison | JSON leaves outside historical tolerance | Array fields marked DIFFERS |
| --- | ---: | ---: |
| October 6 local run versus author | 714 / 8,110 | 38 / 146 |
| October 12 package clean run versus author | 702 / 8,110 | 37 / 146 |
| October 12 package clean run versus October 6 local run | 178 / 8,110 | 9 / 146 |

The October 12 package campaign completed on October 13 UTC. Its historical
comparison policy used relative tolerance 1e-6, with different scalar/array
scaling rules. These are preserved reports, not results from the revised
elementwise comparator. The [summary](summary.json) links each complete report.
The earlier [October 6 evidence](../2026-10-06/comparison.json) remains unchanged.

The [full-clean attestation](full-clean/attestation.json) records all 11 analyses,
8 figures, and 2 supplementary author diagnostics, with execution, parity and
statistical-interpretation limits separated. Its source and input identities are
in [the bundle manifest](full-clean/source-and-input-manifest.json); output
identities are in [the artifact manifest](full-clean/artifact-manifest.json).
Absolute paths in original receipts are historical provenance, not portable
download instructions. Large inputs and arrays are not included here, and public
redistribution rights remain unresolved.

## Focused boundary validation

The [focused report](focused-boundary/scientific-comparison.json) covers a new
seven-donor smoke, cold 250-specimen cache, E1 fidelity and its figure. It does
not cover a fresh execution of all 21 stages. Its source corresponds to branch
head `0d9fe7df1beaa67e82624a4268409fc4d4d33863`, before the subsequent review fixes.

[Read-only rehashing](focused-boundary/fidelity-hashes.json) confirms identical
NPZ bytes (all nine fields) and PNG bytes against the preceding same-host full
run. The raw JSON bytes differ because `seconds` differs. All 305 non-runtime
leaves and the canonical JSON hash excluding declared runtime fields match.
The full source-file digests and copied-source check are retained beside it.

## Revised comparison and inference

The [sensitivity report](sensitivity.json) applies the current symmetric,
elementwise comparator to saved full-clean outputs at fixed rtol 1e-6 and
atol 0, 1e-8, and 1e-7. All six comparisons remain DIFFERS. At the default
atol 1e-8, the author comparison has 660 outside-tolerance JSON leaves and
37 differing array fields; the October 6 local comparison has 141 and 9.
The new JSON denominator is 8,111 because empty containers now count explicitly;
the historical flattening omitted one. Seventeen undefined JSON observations
remain explicit and cannot establish complete numerical parity.

Complete default reports are provided for [author](elementwise-author-atol-1e-08.json)
and [local](elementwise-local-atol-1e-08.json). The sensitivity report records every
setting, input-file hash, comparator hash and runtime environment. This is a
read-only recomparison, not a new execution of the scientific stages. The
[policy and commands](../../comparison-policy.md) explain how to regenerate
the other settings; their complete reports remain in the local validation folder.

The [donor-held-out result](inference-donor-held-out.json) and
[method/environment record](inference-method-and-environment.md) supplement the
archived overlapping-fold diagnostics. All nine fixed inherited contrasts score
all seven donors, using training-only scaling and matched splits. They report
descriptive differences only. Saved descriptors retain their original full-cohort
construction; this is not a prospective end-to-end new-donor test.

## Verify and interpret

From the repository root:

```sh
python tools/verify_research_evidence.py docs/fishy/evidence/2026-10-13
python -m unittest discover -s tests/fishy -v
```

The evidence manifest verifies integrity of these published files. Tests exercise
the current implementation. Neither operation regenerates the archived atlas,
establishes missing author provenance, or proves that repeated CV folds are
independent. See [comparison policy](../../comparison-policy.md) and the
[authored inference supplement](../../inference.md) for the current contracts.
