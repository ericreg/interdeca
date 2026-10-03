# Small-clams CPU and memory profiling

This harness measures the current `native` Atlas and `morphoweave` backends
before optimization. It launches the real Click client, application session,
serial execution manager, numerical operations, and Blender. Instrumentation
is installed only in disposable profiling processes; no production code or
upstream MorphoWeave source is patched on disk.

The [2 October 2026 baseline report](../../docs/profiling/2026-10-02/index.html)
records ten successful runs, the CPU and memory findings, and the supporting
tables, profiles, flame graphs, and numerical output hashes.

Run from the repository root on macOS or Linux:

```sh
uv sync --group profiling
uv run --group profiling python -m tools.profiling \
  --output /tmp/interdeca-profile-new
uv run --group profiling python -m tools.profiling.summarize \
  /tmp/interdeca-profile-new
```

Choose a **new** output directory. The harness refuses to replace previous
results. `--resume` verifies existing outputs and input hashes before continuing
missing runs; incomplete or rejected run directories must be moved aside explicitly.
Resume also checks the dependency lock and source hashes, including uncommitted
code, so an experiment cannot silently combine implementations.

For mesh-reuse budget experiments, use `--backends morphoweave --modes baseline
--repeats 3 --cache-budget-mib 0` (or another nonnegative MiB budget). Use a separate
directory for each budget. Increase `--repeats` with `--resume` to interleave fresh
processes across budgets while keeping existing results. These experiments do not
replace both-backend acceptance captures: omit the budget override for those, so
they exercise the same production default as Slicer. `mesh-cache.jsonl` records
hits, misses, evictions, successful-save admissions/bypasses, array high-water
bytes, bounded Python metadata estimates, and zero live cache bytes after cleanup.
Parsed originals and validated saved intermediates share one budget; actual OBJ
files remain complete and operation-local saved entries are released with their
directory. VTK copies and transient parsing buffers
remain covered by RSS and the separate allocation pass.

Initialize the MorphoWeave submodule first on a fresh checkout, and
install Blender as described in `tests/end_to_end/README.md`. The existing
`BLENDER_EXECUTABLE` and `INTERDECA_SMALL_CLAMS` overrides apply.

Only the three named small-clam specimens are used. The harness reuses the e2e
fixture conversion from LPS to RAS, outside the measurements, and leaves the
reference package untouched. Baking uses 512 × 512 output textures; source
textures retain their original resolution. The full action chain includes
atlas construction, alignment, correspondence, baking, color sampling with
1,000 faces, PCA, reconstruction, display, selection, and export. This profiles
the shared application core through the CLI, not Slicer startup or rendering.

## Measurements and files

The default is ten sequential fresh-process runs:

1. Two baseline runs per backend, reversing backend order on the second repeat.
2. One run per backend with nested function timers only, to measure detailed
   stage costs without a call profiler or allocation tracker.
3. One CPU profiling run per backend.
4. One allocation profiling run per backend.

`--modes baseline`, `--modes spans`, `--modes cpu`, or `--modes memory` selects a subset;
`--repeats` affects only the baseline. No cold filesystem-cache claim is made.
The same backend's atlas files and presentation arrays must have identical
SHA-256 hashes across all measured runs. Failures stop the harness and retain
the logs. Each child has a 40-minute timeout and its own process group for cleanup.

Each run contains:

- `console.log`, the normal CLI events and worker log, and complete outputs.
- `resources.jsonl`: external samples of the application and its descendants,
  including Blender, every 100 ms by default. It records process identity,
  RSS, CPU counters, thread counts, and USS when the OS allows access. USS is
  requested every fifth sample. Unavailable USS is null, never zero.
- `spans.jsonl`: task and pipeline boundaries with wall time, application CPU,
  calling-thread CPU, and CPU used by completed child processes. Task boundaries
  also record RSS and unique NumPy backing-buffer bytes retained by the manager.
- `run.json`: validated configuration, sampled peaks, completion status, and
  output fingerprints. `rusage.json` additionally preserves the OS process
  high-water RSS and CPU totals; macOS reports that RSS in bytes, Linux in KiB.
- CPU runs: Yappi CPU-clock profiles, with thread statistics, per-pipeline
  combined Python-thread `.pstats`, and readable tables. CPU clocks
  are maintained per thread, so waiting is not charged as CPU work. Captures
  containing negative function timings are rejected.
- Spans, CPU, and memory runs: nested function spans for OBJ codecs, array copies,
  geometry validation, host/VTK conversions, upstream methods, and Blender calls.
- Memory runs: `allocations.bin` from Memray, with Python and native allocation
  stacks and allocations live at the tracked heap peak.

The summarizer writes `report.html`, `summary.json`, `memory.png`, `memory.svg`,
and interactive native allocation flame graphs. Raw spans and process samples
remain available for questions not anticipated by the summary.

## Reading the evidence correctly

Use the baseline repeats for runtime and resident-memory comparisons. Detailed
profilers add overhead and have separate runs. Baselines still include lightweight
task timers and a read-only walk of retained result buffers.

RSS measures resident pages, not live numerical arrays. Allocators can retain
freed pages, and the OS can reclaim pages without Python objects changing.
The simultaneous sum of application and child RSS can count shared pages more
than once; it is not the sum of each process's independent peak. Sampling can
miss brief peaks. Private memory (USS) is included only where OS permissions
allow it. See the [psutil memory API](https://psutil.readthedocs.io/stable/#psutil.Process.memory_info).

The retained-array measurement deduplicates NumPy backing arrays reachable from
the execution manager's result table. It excludes VTK, Python containers,
temporary arrays inside the active operation, estimators, and allocator slack.
Multiple references to one result do not become multiple copies in this count.

CPU seconds are processor time, while wall seconds include waiting. Task-level
application CPU includes every thread in the process; completed Blender child
CPU is reported separately. Yappi attributes CPU to Python threads; native
threads created internally by numerical libraries and Blender are accounted for
by the OS counters, not Python function stacks. See the
[Yappi profiling API](https://github.com/sumerc/yappi/blob/master/doc/api.md).
Nested inclusive
function times overlap. Exclusive times subtract other instrumented spans, not
every possible nested function. Never sum inclusive times as a stage budget.

OBJ byte totals describe logical reads and writes, including repeated access to
the same file. They do not measure physical disk transfers or establish that
storage bandwidth is the bottleneck. Array-copy bytes are the sum of arrays
returned by `Arrays.frozen`, not a peak or all allocations in the program.

Memray starts after imports and tracks the application process, including native
allocations; Blender is measured externally. Aggregated captures retain live
allocations at the heap peak but omit allocation chronology and total churn.
Native symbol quality can vary on macOS. These limits follow the
[Memray API](https://bloomberg.github.io/memray/api.html) and
[supported environments](https://bloomberg.github.io/memray/supported_environments.html).

The three-specimen profile does not establish a fifty-specimen peak. Population
retention and per-specimen temporary work scale differently; extrapolating the
entire measured peak linearly would be misleading.

## Compare an optimization with its accepted predecessor

```sh
uv run --group profiling python -m tools.profiling.compare \
  --reference /path/to/last-accepted-profile \
  --candidate /path/to/candidate-profile \
  --output /path/to/comparison.json
```

Run this against the **last accepted baseline**, and again against B0. The command
rehashes prepared inputs and actual outputs, validates capture success, compares
all result presentations and terminal actions, checks atlas scale/settings/upstream
metadata, and compares environments. It checks every run within each backend as
well as across directories. Archived absolute paths are resolved inside the
archive, so the original temporary directory is not needed.

Numerical or environment differences fail the command. Resource deltas and 5%
time/CPU or 10% memory investigation alerts are reported separately; success is
not automatic acceptance of a performance improvement. The optimization plan
defines the required measurements and acceptance decision. Missing instrumented
metrics are reported as unavailable rather than improvements.

An older capture lacking Blender/lockfile metadata needs a `provenance.json`
containing only `blender` and `lock_sha256`, recovered from preserved experiment
evidence. It cannot override a recorded value. B0's durable archive is at
`~/Library/Application Support/InterDeCA/optimization/B0`; it contains the
original inputs, outputs, raw profiles, environment, source/lock snapshots,
and an archive hash manifest. No benchmark artifacts belong in pytest scratch
directories used for later tests.
