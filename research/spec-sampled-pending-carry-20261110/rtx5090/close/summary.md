| mode | class | request | finish | prompt / cached / completion tokens | text | rounds base | cand | close ms base | cand |
|---|---|---|---|---|---|---:|---:|---:|---:|
| close | greedy | eos | stop | 27 / 0 / 52 | equal | 19 | 19 | 52.9 | 52.44 |
| close | greedy | budget | length | 34 / 0 / 96 | equal | 43 | 43 | 0.16 | 0.08 |
| close | greedy | stop | stop | 34 / 0 / 36 | equal | 16 | 16 | 0.11 | 0.12 |
| close | greedy | turn1 | length | 1286 / 0 / 96 | equal | 45 | 45 | 0.09 | 0.09 |
| close | greedy | turn2 | stop | 1414 / 1248 / 85 | equal | 38 | 38 | 0.07 | 0.07 |
| close | greedy | park1 | length | 220 / 0 / 64 | equal | 23 | 23 | 0.07 | 0.1 |
| close | greedy | park2 | length | 284 / 192 / 96 | equal | 29 | 29 | 0.08 | 0.07 |
| close | greedy | other | stop | 24 / 0 / 85 | equal | 38 | 38 | 0.08 | 0.14 |
| close | greedy | cancelled | None | hung up after 40 chunks | not judged | None | None | 0.05 | 0.06 |
| close | greedy | after-cancel | length | 34 / 0 / 160 | equal | 69 | 69 | 0.5 | 0.08 |
| close | t1 | eos | stop | 27 / 0 / 46 | equal | 20 | 21 | 36.15 | 0.08 |
| close | t1 | budget | length | 33 / 0 / 96 | equal | 40 | 41 | 33.87 | 0.08 |
| close | t1 | stop | stop | 33 / 0 / 38 | equal | 17 | 17 | 36.17 | 0.08 |
| close | t1 | turn1 | length | 1285 / 0 / 96 | equal | 46 | 47 | 34.24 | 0.08 |
| close | t1 | turn2 | length | 1413 / 1248 / 96 | equal | 49 | 50 | 34.62 | 0.06 |
| close | t1 | park1 | length | 227 / 0 / 64 | equal | 30 | 30 | 34.23 | 0.11 |
| close | t1 | park2 | length | 291 / 192 / 96 | equal | 34 | 35 | 34.17 | 0.14 |
| close | t1 | other | stop | 24 / 0 / 61 | equal | 28 | 27 | 89.6 | 53.5 |
| close | t1 | cancelled | None | hung up after 40 chunks | not judged | None | None | 0.03 | 0.03 |
| close | t1 | after-cancel | length | 33 / 0 / 160 | equal | 69 | 71 | 33.95 | 0.12 |
| close | chat | eos | stop | 27 / 0 / 58 | equal | 26 | 26 | 36.63 | 0.07 |
| close | chat | budget | length | 33 / 0 / 96 | equal | 42 | 44 | 34.28 | 0.07 |
| close | chat | stop | stop | 33 / 0 / 36 | equal | 18 | 19 | 36.15 | 0.08 |
| close | chat | turn1 | length | 1285 / 0 / 96 | equal | 38 | 39 | 34.54 | 0.11 |
| close | chat | turn2 | stop | 1413 / 1248 / 81 | equal | 36 | 38 | 34.64 | 0.06 |
| close | chat | park1 | length | 227 / 0 / 64 | equal | 25 | 24 | 34.51 | 0.08 |
| close | chat | park2 | length | 291 / 192 / 96 | equal | 26 | 26 | 34.45 | 0.12 |
| close | chat | other | stop | 24 / 0 / 59 | equal | 26 | 26 | 36.45 | 0.14 |
| close | chat | cancelled | None | hung up after 40 chunks | not judged | None | None | 0.04 | 0.03 |
| close | chat | after-cancel | length | 33 / 0 / 160 | equal | 64 | 65 | 34.33 | 0.09 |
| close | pen | eos | stop | 27 / 0 / 59 | equal | 27 | 27 | 35.88 | 0.14 |
| close | pen | budget | length | 34 / 0 / 96 | equal | 40 | 40 | 34.27 | 0.07 |
| close | pen | stop | stop | 34 / 0 / 43 | equal | 16 | 16 | 36.66 | 0.13 |
| close | pen | turn1 | length | 1239 / 0 / 96 | equal | 40 | 40 | 34.63 | 0.1 |
| close | pen | turn2 | stop | 1367 / 1216 / 92 | equal | 45 | 47 | 34.49 | 0.13 |
| close | pen | park1 | length | 219 / 0 / 64 | equal | 25 | 25 | 34.43 | 0.08 |
| close | pen | park2 | length | 283 / 192 / 96 | equal | 31 | 32 | 34.64 | 0.12 |
| close | pen | other | stop | 25 / 0 / 72 | equal | 33 | 33 | 34.26 | 0.11 |
| close | pen | cancelled | None | hung up after 40 chunks | not judged | None | None | 0.03 | 0.04 |
| close | pen | after-cancel | length | 33 / 0 / 160 | equal | 71 | 72 | 34.13 | 0.07 |

rows: 40, with a judged difference: 0
