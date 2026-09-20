# Qwen: full-head MTP depth results

Both arms retain the full vocabulary. Fixed control K=3; native adaptive and learned K=1..7.
Initial prompt: 3,703 tokens. Eight paired repetitions; all selected runs passed the offline audit.

| Arm | Pooled request E2E tok/s |
| --- | ---: |
| Fixed K=3 | 96.750 |
| Native adaptive | 97.714 |
| Instrumented native adaptive | 97.609 |
| Learned | 94.377 |

| Baseline | Learned pooled change | Median paired change | Wins |
| --- | ---: | ---: | ---: |
| Fixed K=3 | -2.45% | -2.61% | 1/8 |
| Native adaptive | -3.42% | -3.30% | 0/8 |
| Instrumented native adaptive | -3.31% | -3.17% | 0/8 |

| Seed | Fixed tok/s | Native adaptive tok/s | Learned tok/s | Learned vs fixed | Learned vs adaptive |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20261200 | 96.500 | 97.207 | 93.489 | -3.12% | -3.82% |
| 20261201 | 99.146 | 97.675 | 94.964 | -4.22% | -2.77% |
| 20261202 | 96.699 | 99.180 | 93.418 | -3.39% | -5.81% |
| 20261203 | 94.368 | 97.003 | 94.809 | +0.47% | -2.26% |
| 20261204 | 96.024 | 96.051 | 95.399 | -0.65% | -0.68% |
| 20261205 | 96.949 | 98.226 | 96.086 | -0.89% | -2.18% |
| 20261206 | 94.801 | 98.284 | 92.805 | -2.10% | -5.57% |
| 20261207 | 99.768 | 98.152 | 94.136 | -5.64% | -4.09% |

The paired sets retain their original seeds and order. Selected receipt directories are named in
`qwen-selected-sets.json`; `qwen-analysis.json` contains audited counts and round statistics.

Binary SHA-256: `4e20676ee9a6ef00f642728b93446cb948d261b077e59b2e503e24a3f7b3b6aa`.
Workload SHA-256: `37f02420635cee6379f54e5d56b7f4e59390d73319a12a522c9e2f23cbb728c2`.
Original local runtime commit: `a37e30a6dd9637ece727c040f12a115dd6d96517`. The exact runtime patch and blob manifest are published alongside this report; see `SOURCE.md`.
The exact runner is retained inside every receipt directory and bound by its identity hash.
