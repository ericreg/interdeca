# Parry spatial-search experiment

Seconds for all specimens in each workload; median of repeat totals.

| Workload | Method | Build | Query | Build + query | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| morphoweave | vtk_existing | 1.343 | 26.552 | 27.895 | 1.00× |
| morphoweave | vtk_reuse | 1.339 | 26.095 | 27.434 | 1.02× |
| morphoweave | parry_serial | 4.319 | 8.165 | 12.484 | 2.23× |
| morphoweave | parry_parallel | 4.310 | 1.192 | 5.501 | 5.07× |
| native | vtk_existing | 1.383 | 35.288 | 36.671 | 1.00× |
| native | vtk_reuse | 1.365 | 34.632 | 35.997 | 1.02× |
| native | parry_serial | 4.317 | 10.841 | 15.157 | 2.42× |
| native | parry_parallel | 4.334 | 1.562 | 5.888 | 6.23× |

Build/query medians are calculated independently; their sum can differ slightly from the median combined time.

| Workload | Specimens | Queries per repeat | Different triangle IDs | Max point error | Max distance error | Points beyond tolerance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| morphoweave | 47 | 4,092,854 | 346,802 | 9.96523e-18 | 9.75782e-18 | 0 |
| native | 47 | 5,310,013 | 497,758 | 4.91133e-18 | 4.84503e-18 | 0 |

Repeats: 3. Parallel Parry workers: 8. Serial and parallel results bitwise equal: True.

Tolerance: 1e-9 × max(surface bounding-box diagonal, 1), in input length units. Different closest points can still have equal minimum distances at surface ties. No exact triangle-ID equivalence is assumed.

Maximum recovered Native-coordinate difference: 4.18436e-15. MorphoWeave-workload projected points that differ after float32 rounding: 0.

The VTK baseline reproduces the Native query loop on both query workloads. vtk_reuse reuses its scratch objects. These are not timings of complete MorphoWeave correspondence or its vtkPoints marshalling loop.

Build includes mesh copying/conversion, bounds checks in Parry, index construction, and thread-pool creation for parallel Parry. Query includes snapshots, finite checks, searches, and NumPy result construction. OBJ reading, spline warping, verification, garbage collection, and index destruction are excluded. No barycentric reconstruction or mesh validation after projection is timed.

VTK uses the Sequential SMP backend. Parry index construction is serial; only its explicitly labelled parallel query mode uses multiple workers. Each observation rebuilds the index. Meshes are processed sequentially, with three repeated observations by default and method order rotated by specimen and repeat. These are warm-process microbenchmarks, not fresh-process pipeline runs.

At measurement time (revision `c272472`), production InterDeCA and vendored MorphoWeave were unchanged, and Parry was an experimental optional extension. The later production integration is separate from these saved prototype measurements.

Platform: macOS-26.6.2-arm64-arm-64bit-Mach-O; arm; physical/logical CPUs: 14/14.

See results.json for per-specimen timings, CPU time, versions, input hashes, source hashes, invocation, and agreement measurements.
