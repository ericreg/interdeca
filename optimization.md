# CPU and memory optimization plan

Status: proposed implementation phases. This document does not implement the
optimizations. Each phase must pass its checks before becoming the baseline for
the next phase.

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
output hashes. The report is already in the repository, but the complete original
outputs currently live at `/tmp/interdeca-profile.3quIrY/run`. Move or copy the
required reference outputs and raw profiles to durable storage outside Git before
temporary cleanup. Preserve at least one complete successful output per backend
for diagnosing a hash mismatch. Keep reference artifacts outside pytest scratch
directories, which pytest may delete.

The existing profiler compares runs **within its output directory**. It does not
yet compare an optimization with B0. Phase 0 adds that explicit comparison; a
successful profiler run by itself is insufficient to accept an optimization.

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
per-backend resource deltas. The exact command/API will be defined in this phase;
it is not part of the current harness.

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
