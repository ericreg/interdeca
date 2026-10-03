# Optimization plan completion audit

The final application is B5 (`eac68e0`), accepted in `385815c`. Phase 6 changes
only tests and reporting. This audit distinguishes numerical/application evidence,
mechanism checks, and measured resource improvements; none substitutes for another.

| Requirement | Evidence and outcome |
| --- | --- |
| Follow phases in order and compare with the previous accepted baseline | [Acceptance history](../../../optimization.md#acceptance-history) and the [generated tables](phase6-tables.md) preserve B0 → B1 → B2 → B3 → B4a → B4b → B5. The report generator checks reference-value continuity for every metric. |
| Preserve B0 and complete reference artifacts | Durable B0 archive, SHA-256 manifest, environment and source/profiler snapshots; [Phase 0 record](phase0-comparison.json). Complete artifacts remain outside pytest scratch storage. |
| Reject regressions independently of host parity | [19 comparison tests](phase0-comparison-tests.log) and [comparison test source](../../../tests/test_profile_comparison.py) cover changed values, correspondence/order, texture pixels, metadata, missing/failed outputs, settings, and input/environment mismatches. |
| Compare all outputs and metadata with both references | Each accepted phase has its predecessor and B0 comparisons. Final [B4b](phase5-vs-B4b.json) and [B0](phase5-vs-B0.json) checks pass for all 12 captures, with 33 output hashes per run and normalized settings, scale, identities, and application events. |
| Preserve exact correspondence and numerical contracts | Faces, UV corners, index order, decoded texture pixels, and per-backend fingerprints remain exact. [Correspondence tests](../../../tests/test_correspondence.py), adapter/cache tests, and the [real-host contract](../../../tests/end_to_end/README.md) retain the original tolerances. |
| Preserve both algorithms and MorphoWeave scale behavior | Original Atlas is retained; upstream source/submodule is unchanged and pinned. Metadata checks preserve normalized mean construction, input-scale restoration before cleanup, and default individual specimen sizes. |
| Remove duplicate texture buffers safely | [Phase 1 tests](phase1-focused.log) and allocation profiles demonstrate the removed copies; [texture memory tests](../../../tests/test_texture_memory.py) preserve external ownership, finite/range validation, and immutable buffers. |
| Validate source files without discarded float colors | [Phase 2 tests](phase2-focused.log) cover image modes, PNG/TIFF, corruption/truncation, unsupported encoding, file cleanup, revalidation, and CLI behavior. [Validation tests](../../../tests/test_texture_validation.py) and allocation stacks support the changed mechanism. |
| Reuse staging while preserving validation and cleanup | [Phase 3 tests](phase3-focused.log), real upstream adapter tests, resource tests, and all 12 comparisons pass. Writes drop from 19 to 16; wall/CPU improvements are measured against B2. |
| Bound mesh reuse and preserve OBJ semantics | [Mesh-cache tests](../../../tests/test_mesh_cache.py) cover budget, eviction, file versions, settings, failed writes, aliases, exact rounding/order/UVs, mutable VTK handles, and cleanup. B4a and B4b each have nine budget trials, full profiles, and host parity. Actual OBJ files remain complete. |
| Investigate tradeoffs instead of hiding regressions | B4a's 10.85% heap alert is retained and explained by [two cache allocation controls](phase4a-memory-investigation.json). Its bounded cost was explicitly accepted for measured CPU savings; no memory improvement was claimed. |
| Release intermediates only when ownership permits | [Phase 5 lifetime tests](../../../tests/test_execution_lifetimes.py), [controller tests](../../../tests/test_controller_lifetimes.py), and [ownership review](phase5-ownership-review.json) verify fan-out, final/nested references, indirect resources, idle queues, publication failure, cancellation, and actual buffer lifetimes without manual GC. |
| Preserve useful model state and UMAP behavior | Repeated four-specimen synthetic PCA/UMAP fitting/reconstruction matches direct core results; previous estimators are released while current session results remain valid. The three real clams are not presented as UMAP coverage. |
| Demonstrate Phase 5 improvement against B4b | All 12 captures match outputs/metadata; MorphoWeave application RSS decreases 8.05% and tracked heap 2.20%. Cache cleanup occurs after alignment in all six MorphoWeave captures. CPU differences remain within observed variation. |
| Complete the final profiling and real-host suite | [12 captures](phase5-summary.json), [both Slicer/CLI cases](phase5-e2e.log), and 212 focused checks pass. Current application/capture source and lockfile match all 44 recorded B5 hashes. Phase 6 makes no measured application change, so this is the final suite. |
| Assess specimen count, image area, and cache pressure separately | [15 bounded pytest cases](phase6-scaling.log) vary 1/4/16 synthetic meshes, 0/1/2/4/8-mesh capacity, 1/4/16 sequential files, and 128–1,024-pixel images. They measure accounting/eviction and preserve bytes; they are not large-dataset timing predictions. |
| Report every requested measurement and comparison | [Tables](phase6-tables.md) include absolute costs, absolute/percentage reductions against predecessor/B0, wall/CPU ranges, child CPU, application/tree/OS RSS, tracked heap, manager arrays, OBJ counts/bytes, copy volume, and cache budgets. [Results](RESULTS.md) distinguish scopes, uncertainty, and remaining costs. |
| Preserve reproduction and diagnostic evidence | Per-phase commits, environment/source records, commands, complete archives, CPU tables/pstats, allocation binaries/flame graphs, and raw resource/task/cache records are identified in [Results](RESULTS.md#evidence-and-reproduction). The figure was visually checked. |
| Honor scope restrictions | Only three named real clams; reference data/package read-only; no GPU, multiprocessing, upstream edits, transport substitution, decimation, precision reduction, or numerical dependency upgrades. No CMake/Slicer test registration. |

The [final validation log](phase6-validation.log) records the archive/capture count,
comparison-chain checks, source/lock identity, unchanged upstream tree, relative
documentation links, and static checks. The 15 new tests passed in 3.63 s; their
elapsed time is not a performance benchmark. All required plan deliverables are
present. A larger real-data benchmark or a streaming OBJ writer would be separate
work; neither is needed to claim completion of this bounded plan.
