# CPU and memory optimization plan

Status: Phases 0–6 complete. B5 is the final accepted application baseline;
Phase 6 adds scaling checks and consolidates the evidence without changing it.

## Acceptance history

The last accepted **performance** baseline is **B5** at
`~/Library/Application Support/InterDeCA/optimization/B5`. Phase 6 uses this final
application baseline and the preserved original B0 reference in the same parent
directory. Phase 5 was compared against its immediate predecessor B4b and B0.
The [final results](docs/profiling/optimization/RESULTS.md),
[complete measurement tables](docs/profiling/optimization/phase6-tables.md), and
[completion audit](docs/profiling/optimization/COMPLETION_AUDIT.md) consolidate the
outcome and its limits.

| Phase | Decision and evidence | Next comparison reference |
| --- | --- | --- |
| 0 | Accepted: 19 comparison tests pass; real native/MorphoWeave Slicer/CLI parity passes (2 cases, 428.65 s); all ten B0 captures compare identically, including metadata. | B0 |
| 1 | Accepted: 89 focused checks and both fresh Slicer/CLI cases pass (512.49 s). All 12 profiling runs succeed and match B0 outputs and metadata. Application RSS falls 36–38% and tracked peak heap falls 54%, with no investigation alerts. | B1 |
| 2 | Accepted: 115 focused tests and both real Slicer/CLI cases pass (616.61 s). All 12 profiling captures succeed and match B1 and B0, including metadata, with no investigation alerts. Application RSS falls 58–64% and tracked peak heap falls 74% relative to B1. | B2 |
| 3 | Accepted: 141 focused tests, both real Slicer/CLI cases (599.60 s), and all 12 profiling captures pass. Outputs and metadata match B2 and B0 with no investigation alerts. MorphoWeave session wall/CPU time falls 6.91%/6.87%; peak heap is essentially unchanged. | B3 |
| 4a | Accepted: 165 focused tests, nine budget trials, two allocation controls, both real Slicer/CLI cases (504.93 s), and all 12 profiling captures pass. Outputs and metadata match B3/B0. MorphoWeave wall/CPU time falls 10.18%/10.38%; the confirmed 91.7 MiB cache cost is an explicit tradeoff. | B4a |
| 4b | Accepted: 182 focused tests, nine budget trials, both real Slicer/CLI cases (433.26 s), and all 12 profiling captures pass. Outputs and metadata match B4a/B0 with no investigation alerts. MorphoWeave wall/CPU time falls 21.39%/23.72%, application RSS falls 8.13%, and tracked heap falls 2.90%. | B4b |
| 5 | Accepted: 212 focused tests, both real Slicer/CLI cases (548.00 s), and all 12 captures pass. Outputs and metadata match B4b/B0 with no investigation alerts. MorphoWeave application RSS falls 8.05% and tracked heap falls 2.20%; lifetime checks verify earlier release. Wall/CPU times remain within observed variation. | B5 |
| 6 | Complete: 15 bounded scaling checks pass; all accepted measurements and both-reference deltas are consolidated. The final application/capture code and lockfile match B5, so its complete profiling/parity suite remains applicable. | B5, final |

Phase 0 was committed as `31fb5d9`; Phase 1 was committed as `0cf6ece`. Its captures
used the Phase 0 revision plus the source changes
identified by all 43 Python file hashes in the
[environment record](docs/profiling/optimization/phase1-environment.json); these
hashes and the dependency lock were checked against the files being committed.
The archive was renamed from `B1-candidate` to `B1` after completion, and the
comparison passed again at its final location. Captures retain their original
paths as provenance; the comparison command resolves archived paths locally.

Each backend has three baseline repeats and separate timer, CPU, and allocation
passes. The [summary](docs/profiling/optimization/phase1-summary.json),
[focused-test log](docs/profiling/optimization/phase1-focused.log), and
[Slicer/CLI log](docs/profiling/optimization/phase1-e2e.log) are retained in Git.
Complete numerical outputs and raw profiles remain in the B1 archive outside Git.

Accepted Phase 1 measurements against B0, which is also the previous accepted
performance baseline:

| Metric | Native B0 → B1 | MorphoWeave B0 → B1 |
| --- | --- | --- |
| Median session wall time | 55.36 → 50.53 s (8.71% lower) | 99.48 → 95.87 s (3.63% lower) |
| Median application CPU | 45.93 → 43.21 s (5.93% lower) | 90.42 → 88.54 s (2.07% lower) |
| Maximum baseline application RSS | 8.18 → 5.06 GiB (38.14% lower) | 8.37 → 5.38 GiB (35.77% lower) |
| Tracked live heap at peak | 6.96 → 3.21 GiB (53.85% lower) | 6.94 → 3.19 GiB (54.00% lower) |

The [B0 comparison](docs/profiling/optimization/phase1-comparison.json) reports
compatible environments, identical outputs/metadata, and no investigation alerts
across all completed captures. All 33 checked output hashes per backend match B0.
Both allocation profiles show one 2 GiB float64 texture representation at peak,
replacing the prior three. Ordinary float-array callers still receive owned,
immutable buffers; only redundant copies inside the trusted boundary were removed.
No mesh, texture-resolution, precision, or upstream algorithm change was made.

The substantial memory reduction is the primary acceptance evidence. Session
times above are observed medians from this machine, not promised speedups on other
workloads. CPU-profiler elapsed times are excluded from these speed measurements;
the CPU captures took 239.7 s (native) and 751.1 s (MorphoWeave).

Phase 2 was committed as `924aa23`. The
captures used parent revision `0cf6ece` plus the source hashes in the
[environment record](docs/profiling/optimization/phase2-environment.json), checked
against the files being committed. Completed captures are preserved at
`~/Library/Application Support/InterDeCA/optimization/B2`. The focused
tests cover supported image modes, corrupt/truncated images, file-handle cleanup,
revalidation between inspection and baking, and the CLI raw-texture presentation.
The CLI also discarded numerical colors solely for validation and now uses the
same operation. Both real Slicer/CLI cases passed in 616.61 s; their combined
elapsed time includes host startup and comparison work and is not the performance
benchmark. CPU captures passed in 234.4 s (native) and 744.4 s (MorphoWeave).

The [summary](docs/profiling/optimization/phase2-summary.json),
[focused-test log](docs/profiling/optimization/phase2-focused.log), and
[Slicer/CLI log](docs/profiling/optimization/phase2-e2e.log) are retained in Git.
As for B1, complete outputs and raw profiles remain outside Git. Byte-identical
large files share filesystem blocks with B0 through independent file clones; their
contents were verified before replacement and rechecked by the comparisons after
promotion from `B2-candidate` to `B2`.

Accepted Phase 2 measurements against the last accepted baseline, B1:

| Metric | Native B1 → B2 | MorphoWeave B1 → B2 |
| --- | --- | --- |
| Median session wall time | 50.53 → 48.54 s (3.95% lower) | 95.87 → 93.77 s (2.19% lower) |
| Median application CPU | 43.21 → 41.34 s (4.32% lower) | 88.54 → 86.60 s (2.19% lower) |
| Maximum baseline application RSS | 5.06 → 1.84 GiB (63.73% lower) | 5.38 → 2.25 GiB (58.22% lower) |
| Tracked live heap at peak | 3.21 → 0.84 GiB (73.90% lower) | 3.19 → 0.83 GiB (74.16% lower) |

The native allocation peak now contains two 256 MiB byte-image buffers plus
geometry allocations. MorphoWeave's peak moved to OBJ parsing and mesh validation.
Both the [B1 comparison](docs/profiling/optimization/phase2-vs-B1.json) and the
[B0 comparison](docs/profiling/optimization/phase2-vs-B0.json) report compatible
environments, identical outputs and metadata, and no investigation alerts. All 33
fingerprints per backend match both references. Against B0, application RSS is
77.56% lower for native and 73.17% lower for MorphoWeave. These figures exclude
CPU-profiler wall time. The memory reduction is the main acceptance evidence;
the small measured time reductions are observations from these repeated runs.
Phase 3 was implemented in `fd5743b` and uses B2 as its comparison reference.
All twelve completed captures are preserved at
`~/Library/Application Support/InterDeCA/optimization/B3`. The new
staging operation writes original RAS meshes/landmarks once, verifies their input
identities on reuse, and supplies separate aligned-output directories for mean
construction and final alignment. Managed task results close before terminal
notifications, including cancellation received during cleanup. Standalone core
calls own their scopes or borrow an explicit one; the core imports no execution
manager or UI libraries. The measured time improvement is accepted; this phase
does not claim a peak-memory reduction.

The evidence includes the [focused-test log](docs/profiling/optimization/phase3-focused.log),
[profiling log](docs/profiling/optimization/phase3-profile.log),
[summary](docs/profiling/optimization/phase3-summary.json), and
[environment record](docs/profiling/optimization/phase3-environment.json).
The captures used B2 commit `924aa23` plus the checkpoint source changes; all 43
recorded Python file hashes and the dependency lock match the checkpoint. The
[B2 comparison](docs/profiling/optimization/phase3-vs-B2.json) and
[B0 comparison](docs/profiling/optimization/phase3-vs-B0.json) report identical
outputs and metadata, compatible environments, and no investigation alerts for
the completed captures. Raw profiles and numerical outputs remain outside Git.

Accepted Phase 3 measurements against the last accepted baseline, B2:

| Metric | Native B2 → B3 | MorphoWeave B2 → B3 |
| --- | --- | --- |
| Median session wall time | 48.54 → 48.84 s (0.62% higher) | 93.77 → 87.29 s (6.91% lower) |
| Median application CPU | 41.34 → 41.38 s (0.09% higher) | 86.60 → 80.66 s (6.87% lower) |
| Maximum baseline application RSS | 1.835 → 1.870 GiB (1.92% higher) | 2.247 → 2.275 GiB (1.24% higher) |
| Tracked live heap at peak | 0.839 → 0.839 GiB | 0.825 → 0.825 GiB |
| OBJ reads / writes | 3 / 7 → 3 / 7 | 16 / 19 → 16 / 16 |

Removing three repeated OBJ writes saves 235,765,927 serialized bytes. These runs
support a time improvement for MorphoWeave, with essentially unchanged tracked
peak heap; they do not establish a memory reduction. The three MorphoWeave baseline
session times are 87.20–88.87 s, compared with B2's 93.54–94.02 s. Native's
48.20–49.10 s range overlaps B2's 48.46–48.71 s range. The small native timing and
both-backend RSS increases fall within observed variation and raise no alerts;
the essentially identical live-heap measurements support unchanged peak storage.
Both real Slicer/CLI end-to-end cases subsequently passed in 599.60 s with Slicer
5.12.4 and Blender 5.2.1; the [test log](docs/profiling/optimization/phase3-e2e.log)
records this result. Both [CPU captures](docs/profiling/optimization/phase3-cpu.log)
passed in 234.6 s (native) and 697.8 s (MorphoWeave); these instrumented elapsed
times are excluded from speed claims.

The CPU captures originally used a separate `B3-cpu-candidate` directory because
checkpoint commit `fd5743b` changed HEAD. Its
[environment record](docs/profiling/optimization/phase3-environment-cpu.json),
input hashes, lockfile, and all 43 measured source hashes exactly match the ten
earlier captures. Both sets match the committed source at `fd5743b`. After
independent CPU-output comparisons against B2 and B0, the two CPU directories
were cloned into the complete B3 archive without rewriting their original paths
or either environment record. The
[capture provenance](docs/profiling/optimization/phase3-capture-provenance.json)
records the association. Complete comparisons passed again after promotion from
`B3-candidate` to `B3`. Phase 4 now uses B3 as its immediate predecessor.

Phase 4a reuses parsed originals only, with a run-local least-recently-used cache.
Each hit still creates owned VTK buffers; disk files, source validation, upstream
calls, and final output reads remain unchanged. File identity and conversion
settings distinguish entries. The array budget and a 256-entry ceiling bound
retention; Python metadata is reported separately. Failure, cancellation, and
successful completion all release cached buffers.

The [budget experiment](docs/profiling/optimization/phase4a-budget-comparison.json)
uses three fresh processes per budget in alternating order, with all nine runs
matching B3 and B0 outputs and metadata:

| Budget | Median CLI session | Median application CPU | Maximum application RSS | Hits / loads | Peak cached arrays |
| --- | ---: | ---: | ---: | ---: | ---: |
| Disabled | 86.99 s | 80.68 s | 2.289 GiB | 0 / 10 | 0 MiB |
| 64 MiB | 84.15 s | 78.05 s | 2.288 GiB | 1 / 10 | 48.09 MiB |
| 96 MiB | 78.60 s | 72.38 s | 2.358 GiB | 4 / 10 | 91.71 MiB |

The selected 96 MiB budget saves 8.39 s of median session time
against disabled reuse, for about 0.069 GiB higher maximum sampled application RSS.
Its three entries use approximately 3.4 KiB of reported Python metadata. All trials
finish with zero cached array bytes. These budget-selection measurements are
separate from the completed both-backend acceptance captures preserved at
`~/Library/Application Support/InterDeCA/optimization/B4a`. Passing data between
saves and loads in memory remains a separate Phase 4b change.

The full profiling batch matches B3/B0 outputs and metadata. Against B3,
MorphoWeave's median session is 10.18% shorter and application CPU is 10.38% lower.
Its sampled application RSS is 4.16% higher, and tracked live heap is 10.85% higher,
which triggers the plan's memory-investigation threshold. The heap increase is
96,152,209 bytes; the cache retains 96,161,280 bytes in nine NumPy buffers, directly
identified in the allocation stacks. Their difference is only 9,071 bytes. This
is a bounded storage cost for reducing repeated parsing, not a memory improvement.
An additional disabled/enabled allocation pair confirms the attribution: the
tracked peaks are 886,250,747 and 982,403,644 bytes. Their 96,152,897-byte difference
matches the retained arrays within 8,383 bytes; both controls match B3/B0 outputs
and metadata exactly. Retain the 96 MiB budget for final validation as an explicit
tradeoff: about 8.88 s saved per median session against B3 for a fixed additional
91.7 MiB of live array storage. The earlier 64 MiB trial saved only about 2.85 s
against disabled reuse. The extra cache storage does not grow beyond its budget
when specimen count grows. Preserve the 10.85% alert and its explanation rather
than changing the comparison thresholds. Both real Slicer/CLI cases pass in
504.93 s with Slicer 5.12.4 and Blender 5.2.1; their
[log](docs/profiling/optimization/phase4a-e2e.log) records the result. CPU captures
also pass in 233.8 s (native) and 591.6 s (MorphoWeave); these instrumented elapsed
times are excluded from speed claims.

Accepted Phase 4a measurements against the last accepted baseline, B3:

| Metric | Native B3 → B4a | MorphoWeave B3 → B4a |
| --- | --- | --- |
| Median session wall time | 48.84 → 48.25 s (1.20% lower) | 87.29 → 78.41 s (10.18% lower) |
| Median application CPU | 41.38 → 41.22 s (0.38% lower) | 80.66 → 72.29 s (10.38% lower) |
| Maximum baseline application RSS | 1.870 → 1.843 GiB (1.44% lower) | 2.275 → 2.370 GiB (4.16% higher) |
| Tracked live heap at peak | 0.839 → 0.838 GiB (0.12% lower) | 0.825 → 0.915 GiB (10.85% higher) |
| OBJ reads / writes | 3 / 7 → 3 / 7 | 16 / 16 → 12 / 16 |

The [B3 comparison](docs/profiling/optimization/phase4a-vs-B3.json) preserves the
memory alert, resolved by the
[allocation investigation](docs/profiling/optimization/phase4a-memory-investigation.json).
The [B0 comparison](docs/profiling/optimization/phase4a-vs-B0.json) reports no
investigation alerts. All 33 checked output fingerprints per backend and all
checked metadata match both references across the twelve captures. Native remains
an unchanged control; its small resource changes are not attributed to the cache.

The [summary](docs/profiling/optimization/phase4a-summary.json),
[environment](docs/profiling/optimization/phase4a-environment.json),
[focused checks](docs/profiling/optimization/phase4a-focused.log), and
[CPU log](docs/profiling/optimization/phase4a-cpu.log) preserve the evidence. Captures
used parent commit `4119b59` plus the source changes in this commit; all 44 recorded
Python file hashes and the lockfile match. Large byte-identical artifacts use
independent filesystem clones of B0 files. The archive was promoted from
`B4a-candidate` to `B4a`, then compared again at its final location. Phase 4b must
use B4a as its immediate predecessor, including its documented memory cost.

Phase 4a was committed as `25399c2`. Phase 4b adds a bounded in-memory
handoff after successful OBJ saves. The candidate still writes every real file
and keeps the upstream workflows unchanged. It retains an owned, validated
float64 mesh matching the ordinary writer/reader round trip; readers check file
versions and conversion settings before reuse. Originals and saved intermediates
share one byte budget. Operation-local entries are forgotten before their
temporary directory is removed, including on failure.

The completed [budget experiment](docs/profiling/optimization/phase4b-budget-comparison.json)
compares disabled, 96 MiB, and 192 MiB reuse in alternating order, with three
fresh processes per budget. All nine captures have compatible environments and
match B4a/B0 outputs and metadata exactly:

| Budget | Median CLI session | Median application CPU | Maximum application RSS |
| --- | ---: | ---: | ---: |
| Disabled | 87.00 s | 80.62 s | 2.252 GiB |
| 96 MiB | 76.72 s | 69.67 s | 2.375 GiB |
| 192 MiB | 62.17 s | 55.66 s | 2.296 GiB |

The 192 MiB budget was selected from these trials: against B4a's accepted medians
of 78.41/72.29 s, the experiment reduces session/CPU time by about 21%/23%.
It serves all 13 adapter loads from memory, with nine successful-save admissions,
no evictions, and a peak of 192,322,560 array bytes (183.41 MiB) across six entries.
Originals and the current operation's aligned outputs share that budget; operation
cleanup releases aligned entries, and run cleanup leaves zero live cache bytes.
The real files remain intact. A higher cache budget is not itself evidence of a
higher application peak: avoiding Python OBJ parsing also removes its temporary
lists and buffers. The allocation pass below quantifies that tradeoff.

The [focused checks](docs/profiling/optimization/phase4b-focused.log) pass all 182
tests, including float32 VTK scratch saves, float64 decimal round trips, signed
zero and extreme finite UV values, invalid inputs, writable aliases, file-version
changes, shared-budget eviction, and partial-staging cleanup. Both-backend
profiling completed six baseline, two timer, two allocation, and two CPU captures.
All twelve match B4a/B0 outputs and metadata exactly, with compatible environments
and no investigation alerts. Accepted measurements against B4a are:

| Metric | Native B4a → B4b | MorphoWeave B4a → B4b |
| --- | --- | --- |
| Median session wall time | 48.25 → 48.09 s (0.34% lower) | 78.41 → 61.63 s (21.39% lower) |
| Median application CPU | 41.22 → 40.89 s (0.80% lower) | 72.29 → 55.14 s (23.72% lower) |
| Maximum baseline application RSS | 1.843 → 1.826 GiB (0.95% lower) | 2.370 → 2.177 GiB (8.13% lower) |
| Tracked live heap at peak | 0.838 → 0.838 GiB | 0.915 → 0.888 GiB (2.90% lower) |
| OBJ reads / writes | 3 / 7 → 3 / 7 | 12 / 16 → 3 / 16 |

MorphoWeave now reads only the three inspected input OBJs: logical read volume
falls from 943,216,055 to 235,765,927 bytes. The timer pass records 5.67 s in OBJ
parsing versus 22.77 s in B4a, while the 16 actual writes remain and take 24.14 s.
Cumulative `Arrays.frozen` copy bytes are unchanged: validated saved snapshots
replace validated parsed snapshots. Peak live heap moves from final-output OBJ
parsing to source TIFF validation during baking, where the two largest allocations
are 256 MiB byte-image buffers. The larger cache therefore reduces CPU without
increasing the measured application peak for these three clams. This does not
establish the same working-set behavior for a larger population.

Both real Slicer/CLI cases passed in 433.26 s with Slicer 5.12.4 and Blender 5.2.1;
their [log](docs/profiling/optimization/phase4b-e2e.log) records that result. Separate
CPU attribution captures also pass in 234.1 s (native) and 370.0 s (MorphoWeave);
these instrumented elapsed times are excluded from speed claims. Native's small
time/RSS differences are treated as control variation; its tracked peak heap,
OBJ traffic, and numerical-copy volume are unchanged.

The [B4a comparison](docs/profiling/optimization/phase4b-vs-B4a.json),
[B0 comparison](docs/profiling/optimization/phase4b-vs-B0.json),
[summary](docs/profiling/optimization/phase4b-summary.json),
[environment](docs/profiling/optimization/phase4b-environment.json),
[profiling log](docs/profiling/optimization/phase4b-profile.log), and
[CPU log](docs/profiling/optimization/phase4b-cpu.log) preserve the evidence.
All 33 output fingerprints per backend and all checked metadata match both
references across the complete batch. Captures used parent revision `25399c2`
plus this commit's source changes; all 44 recorded Python file hashes and the
lockfile were checked against the implementation. Baselines were captured in
three paired rounds using `--modes baseline --repeats N`, with `--resume` for
rounds 2 and 3; timer, allocation, and CPU modes followed with `--resume`.
The nine budget trials use the documented `--cache-budget-mib` override; all
twelve acceptance captures omit it and exercise the production 192 MiB default.

Complete artifacts remain outside Git at
`~/Library/Application Support/InterDeCA/optimization/B4b`. Large byte-identical
files use independent filesystem clones, verified against B0. Final comparisons
passed again after promotion from `B4b-candidate` to `B4b`. Phase 5 must use B4b
as its immediate predecessor.

Phase 4b was committed as `4ffb536`. Phase 5's implementation is checkpoint
`eac68e0`. It closes the MorphoWeave workspace after final alignment, before
publication and texture baking. The workflow explicitly declares that resource
boundary; ordinary managed resources retain their run-wide lifetime so indirect
consumers of a resource or its files keep working. Final aligned meshes and
landmarks own their arrays, and source texture paths refer to external inputs.

The manager counts every nested/repeated result reference and final output before
dropping its own unused entries. Controllers retain partial results for failed,
cancelled, stale, or unsuccessfully published runs; after successful publication,
only useful session results remain. Worker/application threads drop completed
queue items before blocking again, and queued execution callbacks retain their
observer instead of the whole pipeline snapshot. Numerical host scopes clear VTK
handles explicitly without closing a borrowed workspace cache.

The [212 focused tests](docs/profiling/optimization/phase5-focused.log) include
weak-reference checks with no manual garbage collection, indirect resource
forwarding, nested fan-out and final references, queued/active cancellation,
early cleanup failures, host-node cleanup, and repeated four-specimen synthetic
PCA/UMAP fitting and reconstruction. Both real Slicer/CLI cases passed in 548.00 s
with Slicer 5.12.4 and Blender 5.2.1. Separate CPU attribution captures passed in
234.4 s (native) and 370.0 s (MorphoWeave); these instrumented elapsed times are
excluded from speed claims. Task traces retain the same order as B4b: 48 task
executions for native and 44 for MorphoWeave across the complete application session.

Accepted Phase 5 measurements against the last accepted baseline, B4b:

| Metric | Native B4b → B5 | MorphoWeave B4b → B5 |
| --- | --- | --- |
| Median session wall time | 48.088 → 48.136 s | 61.635 → 61.829 s |
| Median application CPU | 40.892 → 40.911 s | 55.143 → 55.568 s |
| Maximum baseline application RSS | 1.826 → 1.827 GiB | 2.177 → 2.002 GiB (8.05% lower) |
| Tracked live heap at peak | 0.838 → 0.838 GiB | 0.888 → 0.869 GiB (2.20% lower) |
| Maximum array bytes retained by manager | 346,832,400 → 203,263,872 | 280,193,064 → 222,580,032 |

Wall/CPU differences are within the observed run-to-run variation; this phase
does not establish a speedup. Native session ranges are 47.720–48.175 s in B4b
and 47.887–48.375 s in B5; MorphoWeave ranges are 61.160–61.904 s and
61.319–62.467 s. OBJ read/write counts, logical bytes, and numerical copy volume
are unchanged. All six MorphoWeave captures close the cache after
`morphoweave_aligned`, releasing 96,161,280 array bytes before baking, and end
with no cache entries or arrays. The production 192 MiB cache budget and
192,322,560-byte array high-water mark are unchanged.

The global tracked peak falls less than that released cache because the peak
moves earlier. B4b peaks during source TIFF validation, with two 256 MiB byte-image
buffers plus live geometry. B5 peaks during OBJ serialization inside upstream
alignment, before the resource release boundary. Its three largest serialization
allocations total 367,045,670 bytes (about 350 MiB). Native's peak is essentially
unchanged. Manager ownership reductions and successful-session cleanup are
separately supported by lifetime tests; a smaller lookup alone does not prove
that controllers or queued callbacks have released the same arrays. RSS includes
allocator retention and is not a measurement of live array size.

The [B4b comparison](docs/profiling/optimization/phase5-vs-B4b.json),
[B0 comparison](docs/profiling/optimization/phase5-vs-B0.json),
[ownership review](docs/profiling/optimization/phase5-ownership-review.json),
[summary](docs/profiling/optimization/phase5-summary.json),
[environment](docs/profiling/optimization/phase5-environment.json),
[profiling log](docs/profiling/optimization/phase5-profile.log),
[CPU log](docs/profiling/optimization/phase5-cpu.log), and
[Slicer/CLI log](docs/profiling/optimization/phase5-e2e.log) preserve the evidence.
All 33 output fingerprints per backend and all checked metadata match both
references across all twelve captures. Both comparisons pass again at the final
archive location, with no investigation alerts. All 44 recorded Python source
hashes, including upstream files, and the lockfile match `eac68e0`.

The full batch was captured in `B5-checkpoint-candidate` and promoted to `B5`;
recorded paths are retained as provenance and the comparator resolves them
locally. Complete outputs, CPU profiles, allocation stacks, and independently
cloned large files remain outside Git at
`~/Library/Application Support/InterDeCA/optimization/B5`. The earlier six
pre-commit baseline captures remain separately in `B5-candidate`, with a record
verifying their source hashes against the same checkpoint; they are not mixed
into the accepted measurements. Phase 5's acceptance was committed as `385815c`.
Phase 6 uses B5 as the final application baseline and consolidates its cumulative
and per-phase comparisons in the final results linked above.

Phase 0 preserved 606 original input/output/profile/source files outside Git and
pytest scratch storage, with an archive SHA-256 manifest. Its evidence is in
[docs/profiling/optimization](docs/profiling/optimization/phase0-comparison.json).
The new command is `uv run --group profiling python -m tools.profiling.compare
--reference REF --candidate CANDIDATE --output REPORT.json`; use it against the
last accepted phase and B0. Numerical/environment failures stop acceptance;
resource deltas and investigation alerts require a separate improvement decision.

## Objective and constraints

Reduce CPU time and peak memory while preserving the existing numerical results
and application behavior. Keep both the original Atlas (`native`) and
MorphoWeave backends available for comparison.

- Keep the pinned MorphoWeave source unchanged and continue using its high-level
  workflows. The host adapter should stay small and the uv worker Slicer-free.
- Preserve float64 numerical precision, source texture resolution, sRGB/alpha
  semantics, vertex order, triangle order, per-corner UVs, and landmark labels.
- Preserve MorphoWeave's normalized mean calculation, restoration to mean input
  centroid size before Blender cleanup, and individual specimen sizes by default.
- Keep Pydantic boundary validation in `serde.py` and internal `NamedTuple`
  records. Core numerical operations remain independent of UI and execution code.
- Continue serial execution. GPU work, multiprocessing, transport replacement,
  algorithm substitution, decimation, and dependency upgrades are outside this plan.
- Use only the three small clams for the real-data acceptance runs. Do not run
  the full clams dataset or fifty real specimens under this plan. The reference
  package and source data remain read-only.

## Measured starting point: B0

The [2 October 2026 report](docs/profiling/2026-10-02/index.html) and
[summary data](docs/profiling/2026-10-02/summary.json) record ten successful runs
against core commit `129c2e218c3eecce6b475a8a19279bf75d1f8b32` and MorphoWeave
revision `6fd9e448fb3441b71f7fed2bec9434a9b2190d24`.

| Measurement | Native Atlas | MorphoWeave |
| --- | ---: | ---: |
| Median complete CLI session | 55.36 s | 99.48 s |
| Median atlas and texture pipeline | 38.24 s | 82.67 s |
| Median application CPU | 45.93 CPU s | 90.42 CPU s |
| Peak application RSS across two baseline runs | 7.52–8.18 GiB | 8.29–8.37 GiB |
| Tracked live heap at peak, separate allocation run | 6.96 GiB | 6.94 GiB |
| Maximum manager-retained NumPy buffers | 330.8 MiB | 267.2 MiB |
| OBJ reads / writes | 3 / 7 | 16 / 19 |
| OBJ codec wall time, separate timer run | 13.51 s | 60.41 s |
| Logical OBJ read + write volume | 550.6 MiB | 2,454.7 MiB |
| Cumulative `Arrays.frozen` copy volume, timer run | 19.49 GiB | 21.12 GiB |

Two mechanisms deserve separate treatment:

1. Both backends temporarily retain three floating-point representations of one
   8192 × 8192 RGBA texture. Each representation occupies 2 GiB. Boundary and
   internal validation account for two additional copies, creating about 6 GiB
   of live texture buffers before decoder memory and masks.
2. MorphoWeave performs repeated OBJ serialization and parsing around upstream
   operations. Approximately 47 extra seconds in these codecs explain the
   measured atlas-pipeline slowdown. Their CPU time is close to wall time;
   logical file volume alone does not establish a physical disk bottleneck.

All 33 checked output files and presentation arrays matched exactly across the
five runs of each backend. Native and MorphoWeave are distinct numerical
baselines; equality between their outputs is not an acceptance requirement.

## Comparison contract for every phase

### Preserve the original baseline and the previous accepted phase

Call the original measurements B0 and each accepted optimization B1, B2, and so
on. Compare every candidate with both B0 and its immediate predecessor. This
reveals cumulative improvement and prevents a later regression from being hidden
by an earlier large gain.

Before editing numerical or adapter code, preserve the profiling harness revision,
input hashes, dependency lock, Blender/Slicer versions, settings, and baseline
output hashes. The report is already in the repository. Phase 0 preserved the
complete original outputs from `/tmp/interdeca-profile.3quIrY/run` in the durable
B0 archive listed above. Preserve at least one complete successful output per
backend for diagnosing a hash mismatch. Keep reference artifacts outside pytest
scratch directories, which pytest may delete.

The profiler compares runs **within its output directory**. Phase 0 added the
separate comparison command for B0 and the last accepted phase; a successful
profiler run by itself is insufficient to accept an optimization.

### Numerical and application checks

- Require the same 33 output hashes within each backend under the same environment,
  both against B0 and the previous accepted phase. Do not replace expected hashes
  with candidate hashes after a failure.
- Separately compare `atlas_run.json` settings, upstream identity, scale conversion,
  presentation metadata, specimen IDs, and action completion. Random session IDs,
  artifact names, and run-specific paths are normalized explicitly. These fields
  are not all covered by the current 33 hashes.
- On a hash mismatch, report the file/field, shape, dtype, and numerical difference.
  Check decoded pixels separately if an image encoding differs. This is diagnosis,
  not permission to weaken the exact-output gate or refresh the reference.
- Run the existing Slicer/CLI pytest for both backends at each completed phase.
  Its geometry tolerances remain `rtol=1e-12, atol=1e-9`; PCA/coordinates/RGB remain
  `rtol=1e-10, atol=1e-10`. Faces, UVs, indices, masks, rendered colors, and decoded
  texture pixels remain exact. Do not sort geometry or align PCA signs to hide a
  change. See the [e2e contract](tests/end_to_end/README.md).
- Check input immutability, validation failures, cancellation, cleanup, and repeated
  use where a phase changes ownership or object lifetime. Host parity alone cannot
  detect the same regression occurring in both hosts.

The small-clams workflow covers PCA. UMAP requires at least four specimens in this
application, so its absence from this fixture is not evidence of UMAP coverage.
If a change reaches generic population/model ownership, add a small deterministic
four-or-more-specimen synthetic check without adding real clams to the benchmark.

### Performance checks

Use fresh sequential processes with the same inputs, bake size, sampling settings,
dependency versions, thread settings, and external executables. Begin each phase
with three baseline repeats per backend, reversing backend order as the harness
already supports. Where a result is close to observed variability, rerun the
previous version and candidate in alternating order on the same machine; expand
to five repeats if needed. Keep reference and candidate environments isolated.

For every accepted phase, record:

| Question | Measurement |
| --- | --- |
| Did the user-visible work get faster? | Baseline session and pipeline wall times, medians and ranges |
| Did the application do less CPU work? | Process CPU, with Blender child CPU reported separately |
| Did peak memory improve? | Application/process-tree sampled RSS, OS high-water RSS, Memray live heap at peak |
| Did the intended mechanism change? | OBJ counts/bytes, copy volume, allocation stacks, retained buffers; cache metrics when introduced |
| Did outputs remain stable? | B0 and previous-phase comparisons, plus Slicer/CLI parity |

Record absolute differences and percentage reductions using
`100 × (reference - candidate) / reference`, for both reference versions.
Never use call-profiler or allocation-profiler wall time as the speed comparison.
Keep CPU, timer, and allocation passes separate from baseline runs.

As investigation thresholds, flag a median wall/CPU increase above 5% or a peak
memory increase above 10% against the previous accepted phase. These are practical
alerts, not statistical confidence limits or an allowance to hide smaller known
regressions. Confirm apparent regressions with repeated controls. Accept a phase
only when its intended resource reduction is demonstrated and any tradeoff is
explicitly justified; an unexplained confirmed regression prevents promotion.
Do not claim a speedup that is smaller than the measured variability.

RSS can retain freed allocator pages and sampled peaks can miss short spikes.
Use allocation stacks and buffer counts to explain changes rather than treating
RSS as live array size. USS was unavailable in B0 and remains an unavailable
metric unless the OS can actually report it. Memray excludes executed Blender
processes and has incomplete native symbols on this Mac.

## Phase 0 — Make baseline comparisons repeatable

**Work:** Preserve B0 and add a report/comparison command that accepts a reference
directory and a candidate directory. Validate input hashes, numerical settings,
upstream revision, and environment identity before claiming a like-for-like
comparison. Compare outputs and metadata under the contract above, then emit
per-backend resource deltas. The implemented comparison command is documented in
the acceptance history above.

**Tests:** Compare B0 with itself and with another successful B0 run. Deliberately
change one array value, a face index/order, a UV corner, a texture pixel, and scale
metadata in disposable copies; require useful failures. Require rejection of
missing outputs, failed runs, and mismatched inputs/settings. Demonstrate that
allowed path/session changes do not trigger false numerical differences.

**Exit:** A documented comparison procedure can reject regressions independently
of Slicer/CLI parity. B0 artifacts are durable and identifiable. No optimization
claim is made in this phase.

## Phase 1 — Remove duplicate texture-validation buffers

**Primary target:** Peak memory in both backends. Secondary target: copy CPU time.

**Work:** Trace `TextureIO.read → TexturePixelsData → TexturePixels.validated` in
[io.py](src/interdeca/io.py), [serde.py](src/interdeca/serde.py), and
[models.py](src/interdeca/models.py). Arrange for boundary type checks and internal
invariant checks to share one owned, validated representation. Start with texture
records; keep a broad redesign of all array validation out of this phase.

Keep copies where external input ownership requires them. A NumPy read-only flag
does not prove exclusive ownership: a writable base array or another alias can
still change the data. Do not globally change `Arrays.frozen` to return every
read-only input unchanged. Preserve float64 conversion arithmetic and accepted
image modes. Consider chunked range/finite checks if temporary masks still make
a material contribution after duplicate copies are removed.

**Tests:** Cover all accepted image modes (RGB, RGBA, P, L, LA), opaque and partial
alpha, PNG/TIFF, invalid dimensions/ranges, nonfinite supplied arrays, and rejected
image encodings. Compare pixel arrays exactly with the old path. Exercise writable
inputs, non-contiguous views, and read-only views with writable bases; changing the
caller-owned input must not change an accepted internal record. Then run the common
output/parity gate and both-backend baseline, timer, and allocation passes.

**Exit:** Memray demonstrates removal of the two duplicate full-image validation
representations at the identified peak; B0's three simultaneous 2 GiB floating
representations no longer occur through that path. Copy volume falls, output
precision is unchanged, and measured end-to-end peaks improve. The resulting
whole-process memory target is measured, not assumed to equal the remaining buffer.

## Phase 2 — Validate source texture files without float64 pixel materialization

**Primary target:** Further shared memory reduction during inspection and baking.

**Work:** Separate source-file validation from numerical pixel loading.
[Dataset.inspect_specimen](src/interdeca/dataset.py) and
[Blender.bake_specimen_texture](src/interdeca/blender.py) currently decode source
images into floating-point arrays whose values are discarded. Use a lightweight
validation operation that checks the supported encoding and actually decodes enough
to detect corruption, retaining at most the required compact image buffers.
Metadata/header checks alone are insufficient. Keep `TextureIO.read` for consumers
that need numerical colors.

Initially retain validation at both call sites so this phase does not silently
change when modified input files are detected. Reusing a validation result is a
separate change requiring file-version identity and invalidation tests. Do not
cache decoded source textures across the population.

**Tests:** Valid image modes and pixels must behave as before. Corrupt/truncated
PNG and TIFF files, unsupported encodings, missing files, and a file modified
between inspection and baking must fail at the applicable boundary. Check file
handles close on success and failure. Then run both-backend output/parity checks
and baseline, timer, and allocation passes.

**Exit:** Profiling shows no full-resolution float64 allocation solely for source
file validation. Corrupt-input detection remains effective, baked textures and
downstream colors remain exact, and both peak allocations and baseline memory
improve relative to Phase 1. Report any new dominant allocation after this peak
is removed.

## Phase 3 — Stage MorphoWeave input geometry once per run

**Primary target:** Eliminate redundant mesh writes with a small adapter change.

**Work:** Replace the two independent `_inputs` staging passes in
[morphoweave.py](src/interdeca/morphoweave.py) with a run-owned staging workspace.
Reuse immutable original model/landmark files between mean construction and final
alignment; keep distinct directories for their different aligned outputs. Preserve
the original transform inputs and the restored atlas used for final alignment.
Continue calling upstream `_generateAtlas`, `runMean`, and `runAlign` as today.

Own workspace lifetime in the application/workflow integration. Numerical core
records must not acquire execution-manager or UI dependencies. Standalone calls
must still own and clean up their required workspace explicitly. A staging directory
must never be a process-global cache or leak between input revisions.

Use an explicitly managed task resource for the shared workspace: its producer
stages the validated originals, and later tasks reference that same workspace.
The execution layer should manage a generic context-manager lifetime, without
importing MorphoWeave or introducing execution dependencies into numerical code.
Release the resource before reporting a terminal run state, including when a
later task fails, a queued/active run is cancelled, or the manager closes. A
standalone core caller should use the same workspace in a `with` scope. Keep
geometry/input identity checks at reuse boundaries, and leave room for the bounded
run-local cache in Phase 4 to share that explicit cleanup lifetime.

**Tests:** Test workspace reuse within a run and isolation between runs, file
pairing/case rules, unchanged staged geometry, and cleanup after success, failure,
and cancellation. Require the same upstream high-level calls and scale restoration.
Run MorphoWeave's focused tests, both-backend output/parity checks, and the profiling
passes, using native as an unchanged control.

**Exit:** The second serialization of the three original meshes disappears. If all
other writes are unchanged, the observed MorphoWeave count falls from 19 to 16.
Require measured count/byte reduction and inspect wall/CPU improvement; do not
claim that this phase removes the entire 47-second gap.

## Phase 4 — Reduce remaining host-side serialization with bounded reuse

**Primary target:** Remaining MorphoWeave OBJ read/write CPU. Proceed only where
Phase 3 profiles still show repeated conversion as a material cost.

**Work:** Audit [Host.loadModel/saveNode](src/interdeca/_morphoweave_host.py) and the
upstream path-discovery calls. First reuse already parsed, validated representations
for repeated loads of an unchanged staged file. Then evaluate passing intermediate
geometry through the host in memory when upstream still sees the same file/node
contract. Introduce these as separate measurable changes. Retain actual files where
upstream discovery or external consumers require them, and preserve exported files.

Use a run-local byte budget with eviction and disk fallback. Compare disabled,
small, and larger budgets on the same three clams; choose the default from measured
CPU/memory tradeoffs. Account separately for NumPy storage, VTK deep copies, cached
objects, and transient materialization. An oversized entry must bypass the cache.
Do not store both every mesh and its serialized text, or every decoded texture.

Cache identity must include the file/geometry version and relevant conversion
settings. Saves invalidate previous loads. A cache hit must give upstream an owned
mutable VTK handle where its scratch-node behavior requires one. Never share a
mutable scratch node across specimens.

Removing an OBJ round trip can change rounding, point dtype, or cleanup behavior.
Establish equivalence to the current writer/reader path before removing it. Keep
the existing file path for any boundary whose exact semantics cannot yet be
preserved; upstream algorithms and output tolerances are not bargaining points.

**Tests:** Check cache hits, eviction, disabled-cache operation, oversized entries,
overwritten files, conversion-setting changes, and cleanup on failed/cancelled runs.
Mutate upstream handles to prove isolation from cached/input arrays. Include UV
seams, triangle order, landmark order, and round-trip precision cases. Compare
disabled versus enabled outputs and B0 hashes. Run both-host parity and full
profiling for the selected design; use smaller targeted experiments to choose
budgets before that acceptance run.

**Exit:** Additional OBJ calls/bytes and codec CPU fall. Record cache high-water
bytes, hits/misses, and evictions; the cache respects its byte limit. Extra memory
is bounded and justified by measured CPU savings. Keep the existing file-backed
behavior when a proposed cache provides no useful benefit.

## Phase 5 — Release completed intermediates when ownership allows it

**Primary target:** Population memory growth and repeated-session memory, after
the large per-texture peak has been addressed.

**Work:** Profile ownership across the serial manager, controllers, host nodes,
and application state. Where objects have no remaining consumers, release them.
For manager task results, count every downstream reference, including fan-out,
nested inputs, final pipeline outputs, and data retained for current failure or
presentation behavior. Preserve completed results required by the application.
Removing an entry from the manager is not evidence that another owner released it.

Keep this change separate from numerical operations and leave the task schedule
serial. Release stage-local host caches and temporary files when their consumers
finish. Avoid manual garbage-collection calls as a substitute for clear ownership.
For managed resources, the last direct `ResultRef` alone does not prove the end
of their lifetime: an operation can forward a resource or return a path into its
directory. Preserve the run-wide lifetime unless the workflow explicitly declares
an earlier boundary and its outputs are proven independent of the closed resource.

**Tests:** Small deterministic task graphs exercise fan-out, reused inputs, final
outputs, failure, cancellation, queued runs, and repeated execution. Verify both
correct results and the lifetime of no-longer-needed buffers. Repeat application
actions in one session to detect unintended retained references. Run both-backend
parity and profiles, plus synthetic population/model checks if generic model
lifetimes are affected.

**Exit:** Measured retained buffers decrease at the intended boundaries without
losing observable results or diagnostics. Explain any gap between live-buffer
reduction and RSS. Keep changes only where ownership or measured resource use
actually improves.

## Phase 6 — Consolidate results and assess scaling limits

**Completed:** The [results](docs/profiling/optimization/RESULTS.md) and
[tables](docs/profiling/optimization/phase6-tables.md) include all requested metrics,
absolute/percentage changes, provenance, and commands. The final B5 suite has
12 captures and both real Slicer/CLI cases; all 44 captured Python source hashes
and the lockfile still match the final application. No measured source changed
in this phase. Fifteen additional [scaling tests](tests/test_resource_scaling.py)
pass, varying synthetic population count, image area, and cache pressure
independently. Their [log](docs/profiling/optimization/phase6-scaling.log) and the
[completion audit](docs/profiling/optimization/COMPLETION_AUDIT.md) preserve the
checks and distinguish logical buffer accounting from full-process memory.

**Work:** Run the complete profiling and parity suite against the final candidate.
Publish a phase-by-phase table against B0 and the immediate predecessor, including
numerical-check status, wall/CPU time, RSS, live heap, retained arrays, OBJ traffic,
copy volume, and cache budget. Retain the profiler commands, output directories,
implementation revisions, environment records, and raw diagnostic artifacts.

Use small synthetic meshes/images to vary specimen count, image dimensions, and
cache pressure independently. These tests establish accounting and eviction
behavior; they are not a fifty-clam performance benchmark. Keep synthetic resource
tests bounded and label them separately from the real three-clam results.

Both current backends load population geometry. Upstream MorphoWeave's `runMean`
holds imported meshes and a correspondence group; adapter caching cannot make
that algorithm stream. The three source meshes contain about 91.7 MiB of raw
arrays, suggesting roughly 1.49 GiB for fifty meshes of the same average density
before copies and other allocations. This is not a predicted total peak.

**Exit:** State which costs were eliminated, what now dominates, and which memory
components grow with population size versus maximum texture dimensions. Propose
a separately authorized larger real-data experiment only if needed. Do not
extrapolate the entire three-clam peak by multiplying it by 50/3.

## Commands and acceptance record

Existing commands, run from the repository root:

```sh
uv sync --group profiling

# Existing focused numerical/adapter checks.
uv run pytest tests/test_correspondence.py tests/test_morphoweave.py -v

# Real Slicer/CLI parity for both backends; use a new scratch directory.
uv run pytest tests/end_to_end/test_small_clams.py -v -s \
  --basetemp /tmp/interdeca-phase-N-e2e

# Three baseline repeats plus timer, CPU, and memory passes for both backends.
# Replace N and choose a new output directory for each candidate.
uv run --group profiling python -m tools.profiling \
  --output /tmp/interdeca-phase-N-profile --repeats 3
uv run --group profiling python -m tools.profiling.summarize \
  /tmp/interdeca-phase-N-profile
```

Add the focused tests described in each phase when that change is implemented.
These are ordinary pytest checks in the uv environment, with no CMake/Slicer test
registration. Keep performance thresholds out of timing-sensitive unit tests;
resource acceptance uses the controlled profiling comparison. Full CPU attribution
can be expensive, so run it after a candidate passes the faster correctness checks.

Each acceptance record must contain the phase and implementation revision, B0 and
previous-phase references, input/settings/environment identity, commands, numerical
and parity outcomes, resource deltas, any explained tradeoffs, and the decision.
If a candidate fails, retain its evidence and revise or revert that change without
replacing either accepted baseline. This preserves a reviewable history of what
each optimization achieved.
