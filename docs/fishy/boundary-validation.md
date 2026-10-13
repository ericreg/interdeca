# Integration metadata boundaries

The integration layer separates public JSON from immutable internal records.
`Evaluation` validates finite, strict classifier settings in `serde.py`, then
converts once to `EvaluationOptions`. `Analysis.score` returns `FoldScore` and
`SchemeScore` records; scheme/factor registries remain dictionaries. JSON retains
the existing fold field names, arrays, nulls and classifier protocol.

`records.py` contains standard-library NamedTuples. The runner's `serde.py` imports
these records directly, so inspecting metadata or CLI help loads neither NumPy
nor InterDeCA. `models.py` reexports the existing generator `Parameters` type.
`requirements-tooling.txt` declares five integration-only Pydantic dependencies;
the immutable reference's 29 scientific pins remain unchanged. Install both
requirements files for paper/release commands, as shown in the reproduction docs.

Owned run, bundle and bundle-manifest documents require integer `schema_version: 1`.
Unsupported, missing, boolean and string versions fail validation. Historical
unversioned completion, build markers, input inventories, specimens, provenance
and export inventories have explicitly named legacy readers. Optional analysis
metadata and timestamps preserve missing versus explicit null fields, and thread
environment/package registries preserve their original entries. Unknown fixed
fields fail rather than disappearing during a rewrite.

Structural checks do not replace input/content SHA256 verification, exact specimen
joins, dependency invalidation, stage exit statuses or E6 checkpoint invalidation.
The historical source-list digest uses the same JSON representation after records
are serialized. Finished historical bundles and runs are not modified by this
refactor. New bundle builds explicitly include the records module and tooling
requirements in their hashed source snapshot.

The bounded regression checks compare the previous scoring method from an immutable
bundle with the current method in the same environment on actual 250-specimen
composition vectors. Both observed and shuffled predictions, split indices,
omitted single-class folds and serialized scores match exactly across 27 folds
for each of two responses. The matched native/atlas adapter also completes on
archived inputs in a disposable directory. Metadata checks cover real historical
roundtrips, strict validation before output allocation and failed-stage resume
state, including dependent records and E6 partial checkpoints. These checks do not
claim a new full paper reproduction or resolve historical numerical discrepancies.

The subsequent research-repository review promoted behavioral boundary, roundtrip,
stage-invalidation and comparison checks into the [committed test suite](../../tests/README.md).
These are executable fixtures with temporary inputs; they do not need private
data or replicate a full paper run. CI runs them in the pinned CPU environment.

The previous focused real-data validation is now linked in the
[published evidence index](evidence/2026-10-13/README.md). Its nine E1 array fields
and figure PNG match the preceding same-host run exactly. The JSON's 305
non-runtime leaves match exactly; raw JSON bytes differ only in `seconds`, as
the separately published raw and normalized hashes show. The receipts identify
the historical tested source, not subsequent commits.
