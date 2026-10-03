# Accepted measurements

Generated from the linked comparison records. GiB = 2³⁰ bytes; MiB = 2²⁰ bytes.

| Backend | Baseline | Session median (s) | Application CPU median (s) | Application RSS maximum (GiB) | Tracked heap peak (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| native | B0 | 55.359 | 45.932 | 8.178 | 6.964 |
| native | B1 | 50.534 | 43.207 | 5.059 | 3.214 |
| native | B2 | 48.537 | 41.343 | 1.835 | 0.839 |
| native | B3 | 48.840 | 41.380 | 1.870 | 0.839 |
| native | B4a | 48.252 | 41.221 | 1.843 | 0.838 |
| native | B4b | 48.088 | 40.892 | 1.826 | 0.838 |
| native | B5 | 48.136 | 40.911 | 1.827 | 0.838 |
| morphoweave | B0 | 99.479 | 90.415 | 8.374 | 6.943 |
| morphoweave | B1 | 95.872 | 88.541 | 5.379 | 3.194 |
| morphoweave | B2 | 93.771 | 86.603 | 2.247 | 0.825 |
| morphoweave | B3 | 87.292 | 80.656 | 2.275 | 0.825 |
| morphoweave | B4a | 78.408 | 72.285 | 2.370 | 0.915 |
| morphoweave | B4b | 61.635 | 55.143 | 2.177 | 0.888 |
| morphoweave | B5 | 61.829 | 55.568 | 2.002 | 0.869 |

## Reductions against both references

Each cell is **previous accepted baseline / B0**, in percent. Negative means an increase.

| Backend | Candidate | Session time | Application CPU | Application RSS | Tracked heap | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| native | B1 | 8.71% / 8.71% | 5.93% / 5.93% | 38.14% / 38.14% | 53.85% / 53.85% | [comparison](phase1-comparison.json) |
| native | B2 | 3.95% / 12.32% | 4.32% / 9.99% | 63.73% / 77.56% | 73.90% / 87.95% | [comparison](phase2-vs-B1.json) |
| native | B3 | -0.62% / 11.78% | -0.09% / 9.91% | -1.92% / 77.13% | -0.00% / 87.95% | [comparison](phase3-vs-B2.json) |
| native | B4a | 1.20% / 12.84% | 0.38% / 10.26% | 1.44% / 77.46% | 0.12% / 87.97% | [comparison](phase4a-vs-B3.json) |
| native | B4b | 0.34% / 13.13% | 0.80% / 10.97% | 0.95% / 77.67% | 0.00% / 87.97% | [comparison](phase4b-vs-B4a.json) |
| native | B5 | -0.10% / 13.05% | -0.05% / 10.93% | -0.04% / 77.66% | 0.00% / 87.97% | [comparison](phase5-vs-B4b.json) |
| morphoweave | B1 | 3.63% / 3.63% | 2.07% / 2.07% | 35.77% / 35.77% | 54.00% / 54.00% | [comparison](phase1-comparison.json) |
| morphoweave | B2 | 2.19% / 5.74% | 2.19% / 4.22% | 58.22% / 73.17% | 74.16% / 88.11% | [comparison](phase2-vs-B1.json) |
| morphoweave | B3 | 6.91% / 12.25% | 6.87% / 10.79% | -1.24% / 72.84% | -0.00% / 88.11% | [comparison](phase3-vs-B2.json) |
| morphoweave | B4a | 10.18% / 21.18% | 10.38% / 20.05% | -4.16% / 71.70% | -10.85% / 86.82% | [comparison](phase4a-vs-B3.json) |
| morphoweave | B4b | 21.39% / 38.04% | 23.72% / 39.01% | 8.13% / 74.00% | 2.90% / 87.20% | [comparison](phase4b-vs-B4a.json) |
| morphoweave | B5 | -0.32% / 37.85% | -0.77% / 38.54% | 8.05% / 76.10% | 2.20% / 87.49% | [comparison](phase5-vs-B4b.json) |

B4a's 10.85% MorphoWeave heap increase was investigated and accepted as a bounded cache tradeoff; see [the controls](phase4a-memory-investigation.json). Small timing/RSS fluctuations are not speedup claims.

## Absolute reductions against both references

Each cell is **previous accepted baseline / B0**. Time differences are seconds; memory differences are GiB. Negative means an increase; the JSON records retain unrounded bytes and seconds.

| Backend | Candidate | Session time | Application CPU | Application RSS | Tracked heap |
| --- | --- | ---: | ---: | ---: | ---: |
| native | B1 | 4.824 / 4.824 | 2.724 / 2.724 | 3.119 / 3.119 | 3.750 / 3.750 |
| native | B2 | 1.998 / 6.822 | 1.865 / 4.589 | 3.224 / 6.343 | 2.375 / 6.125 |
| native | B3 | -0.303 / 6.519 | -0.037 / 4.552 | -0.035 / 6.307 | -0.000 / 6.125 |
| native | B4a | 0.588 / 7.107 | 0.159 / 4.711 | 0.027 / 6.334 | 0.001 / 6.126 |
| native | B4b | 0.164 / 7.271 | 0.328 / 5.039 | 0.017 / 6.352 | 0.000 / 6.126 |
| native | B5 | -0.048 / 7.223 | -0.019 / 5.021 | -0.001 / 6.351 | 0.000 / 6.126 |
| morphoweave | B1 | 3.608 / 3.608 | 1.874 / 1.874 | 2.996 / 2.996 | 3.749 / 3.749 |
| morphoweave | B2 | 2.101 / 5.709 | 1.938 / 3.812 | 3.132 / 6.127 | 2.368 / 6.117 |
| morphoweave | B3 | 6.479 / 12.188 | 5.948 / 9.760 | -0.028 / 6.100 | -0.000 / 6.117 |
| morphoweave | B4a | 8.884 / 21.071 | 8.370 / 18.130 | -0.095 / 6.005 | -0.090 / 6.028 |
| morphoweave | B4b | 16.774 / 37.845 | 17.143 / 35.273 | 0.193 / 6.197 | 0.027 / 6.054 |
| morphoweave | B5 | -0.195 / 37.650 | -0.426 / 34.847 | 0.175 / 6.373 | 0.020 / 6.074 |

## Mechanism measurements

OBJ traffic and array-copy volume come from separate timer runs. Retained arrays are the maximum in the manager's lookup, not total live memory.

| Backend | Baseline | OBJ reads / writes | Logical OBJ read / write (GiB) | Array-copy volume (GiB) | Manager arrays (MiB) | Cache array peak / budget (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| native | B0 | 3 / 7 | 0.220 / 0.318 | 19.489 | 330.77 | — |
| native | B1 | 3 / 7 | 0.220 / 0.318 | 1.294 | 330.77 | — |
| native | B2 | 3 / 7 | 0.220 / 0.318 | 1.294 | 330.77 | — |
| native | B3 | 3 / 7 | 0.220 / 0.318 | 1.294 | 330.77 | — |
| native | B4a | 3 / 7 | 0.220 / 0.318 | 1.294 | 330.77 | — |
| native | B4b | 3 / 7 | 0.220 / 0.318 | 1.294 | 330.77 | — |
| native | B5 | 3 / 7 | 0.220 / 0.318 | 1.294 | 193.85 | — |
| morphoweave | B0 | 16 / 19 | 1.203 / 1.194 | 21.117 | 267.21 | — |
| morphoweave | B1 | 16 / 19 | 1.203 / 1.194 | 2.922 | 267.21 | — |
| morphoweave | B2 | 16 / 19 | 1.203 / 1.194 | 2.922 | 267.21 | — |
| morphoweave | B3 | 16 / 16 | 1.203 / 0.974 | 2.743 | 267.21 | — |
| morphoweave | B4a | 12 / 16 | 0.878 / 0.974 | 2.479 | 267.21 | 91.71 / 96 |
| morphoweave | B4b | 3 / 16 | 0.220 / 0.974 | 2.479 | 267.21 | 183.41 / 192 |
| morphoweave | B5 | 3 / 16 | 0.220 / 0.974 | 2.479 | 212.27 | 183.41 / 192 |

Cache values are validated in the accepted [B4a](../../../optimization.md#acceptance-history), [B4b](phase4b-budget-comparison.json), and [B5](phase5-ownership-review.json) records. Cache metadata, parsing transients, VTK copies, and caller-owned buffers are outside the cache's array budget.

## Ranges and process accounting

Ranges are observed minimum–maximum, not confidence intervals. Child CPU is separate from application CPU. Process-tree RSS sums pages and can double-count shared memory. OS high-water RSS below is application-only; unavailable USS is not treated as zero.

| Backend | Baseline | Repeats | Session range (s) | Application CPU range (s) | Blender CPU median (s) | Tree RSS max (GiB) | OS application RSS max (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| native | B0 | 2 | 55.343–55.375 | 45.493–46.370 | 7.250 | 8.178 | 8.259 |
| native | B1 | 3 | 49.354–51.288 | 41.996–44.027 | 7.087 | 5.919 | 5.059 |
| native | B2 | 3 | 48.464–48.706 | 40.727–41.366 | 7.048 | 2.694 | 1.836 |
| native | B3 | 3 | 48.195–49.099 | 40.199–41.468 | 7.072 | 2.735 | 1.870 |
| native | B4a | 3 | 48.141–48.323 | 41.124–41.319 | 7.044 | 2.705 | 1.846 |
| native | B4b | 3 | 47.720–48.175 | 40.837–41.169 | 7.041 | 2.686 | 1.828 |
| native | B5 | 3 | 47.887–48.375 | 40.637–40.955 | 7.081 | 2.695 | 1.831 |
| morphoweave | B0 | 2 | 98.763–100.196 | 89.777–91.054 | 7.280 | 8.374 | 8.375 |
| morphoweave | B1 | 3 | 95.285–96.096 | 88.135–88.914 | 7.167 | 5.896 | 5.470 |
| morphoweave | B2 | 3 | 93.540–94.016 | 86.541–86.830 | 7.065 | 3.019 | 2.256 |
| morphoweave | B3 | 3 | 87.197–88.868 | 79.808–80.932 | 7.031 | 2.890 | 2.290 |
| morphoweave | B4a | 3 | 78.176–78.991 | 71.811–72.649 | 7.023 | 3.116 | 2.441 |
| morphoweave | B4b | 3 | 61.160–61.904 | 55.114–55.532 | 6.950 | 2.982 | 2.459 |
| morphoweave | B5 | 3 | 61.319–62.467 | 54.906–55.859 | 7.036 | 2.862 | 2.192 |
