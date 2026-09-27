# Gemma: full-head MTP depth results

Both arms retain the full vocabulary. Fixed control K=5; native adaptive and learned K=1..5.
Initial prompt: 4,037 tokens. Eight paired repetitions; all selected runs passed the offline audit.

| Arm | Pooled request E2E tok/s |
| --- | ---: |
| Fixed K=5 | 175.325 |
| Native adaptive | 188.080 |
| Instrumented native adaptive | 187.557 |
| Learned | 188.889 |

| Baseline | Learned pooled change | Median paired change | Wins |
| --- | ---: | ---: | ---: |
| Fixed K=5 | +7.74% | +7.91% | 8/8 |
| Native adaptive | +0.43% | +0.36% | 6/8 |
| Instrumented native adaptive | +0.71% | +0.40% | 6/8 |

| Seed | Fixed tok/s | Native adaptive tok/s | Learned tok/s | Learned vs fixed | Learned vs adaptive |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261300 | 175.259 | 189.253 | 189.735 | +8.26% | +0.25% |
| 20261301 | 175.271 | 188.049 | 189.131 | +7.91% | +0.58% |
| 20261302 | 173.778 | 187.236 | 187.852 | +8.10% | +0.33% |
| 20261303 | 175.388 | 188.813 | 187.972 | +7.17% | -0.45% |
| 20261304 | 177.102 | 187.330 | 192.382 | +8.63% | +2.70% |
| 20261305 | 174.753 | 188.103 | 186.571 | +6.76% | -0.81% |
| 20261306 | 175.660 | 187.528 | 188.251 | +7.17% | +0.39% |
| 20261307 | 175.405 | 188.337 | 189.295 | +7.92% | +0.51% |

The paired sets retain their original seeds and order. Selected receipt directories are named in
`gemma-selected-sets.json`; `gemma-analysis.json` contains audited counts and round statistics.

Binary SHA-256: `cf80a950a38a7cf528231b203704eedeb3f8bb3c378a0343521407d9b032237a`.
Workload SHA-256: `37f02420635cee6379f54e5d56b7f4e59390d73319a12a522c9e2f23cbb728c2`.
Original local runtime commit: `a37e30a6dd9637ece727c040f12a115dd6d96517`. The exact runtime patch and blob manifest are published alongside this report; see `SOURCE.md`.
The exact runner is retained inside every receipt directory and bound by its identity hash.
