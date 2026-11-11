| mode | class | request | finish | prompt / cached / completion tokens | text | rounds base | cand | close ms base | cand |
|---|---|---|---|---|---|---:|---:|---:|---:|
| close | greedy | eos | stop | 27 / 0 / 59 | equal | 22 | 22 | 0.09 | 0.14 |
| close | greedy | budget | length | 34 / 0 / 96 | equal | 43 | 43 | 0.06 | 0.13 |
| close | greedy | stop | stop | 34 / 0 / 36 | equal | 16 | 16 | 0.08 | 0.11 |
| close | greedy | turn1 | length | 1286 / 0 / 96 | equal | 50 | 50 | 0.13 | 0.17 |
| close | greedy | turn2 | stop | 1414 / 1248 / 84 | equal | 46 | 46 | 0.14 | 0.15 |
| close | greedy | park1 | length | 220 / 0 / 64 | equal | 23 | 23 | 0.09 | 0.11 |
| close | greedy | park2 | length | 284 / 192 / 96 | equal | 31 | 31 | 0.16 | 0.18 |
| close | greedy | other | stop | 24 / 0 / 83 | equal | 33 | 33 | 0.07 | 0.11 |
| close | greedy | cancelled | None | hung up after 40 chunks | not judged | None | None | 0.14 | 0.14 |
| close | greedy | after-cancel | length | 34 / 0 / 160 | equal | 69 | 69 | 0.1 | 0.13 |
| close | t1 | eos | stop | 27 / 0 / 52 | equal | 22 | 22 | 76.79 | 0.11 |
| close | t1 | budget | length | 33 / 0 / 96 | equal | 38 | 39 | 75.31 | 8.91 |
| close | t1 | stop | stop | 33 / 0 / 37 | equal | 13 | 13 | 75.79 | 0.09 |
| close | t1 | turn1 | length | 1285 / 0 / 96 | equal | 47 | 48 | 75.72 | 0.17 |
| close | t1 | turn2 | length | 1413 / 1248 / 96 | equal | 41 | 43 | 75.25 | 0.16 |
| close | t1 | park1 | length | 227 / 0 / 64 | equal | 31 | 32 | 75.24 | 0.1 |
| close | t1 | park2 | length | 291 / 192 / 96 | equal | 44 | 43 | 75.74 | 0.17 |
| close | t1 | other | stop | 24 / 0 / 66 | equal | 28 | 27 | 77.09 | 0.11 |
| close | t1 | cancelled | None | hung up after 40 chunks | not judged | None | None | 0.1 | 0.08 |
| close | t1 | after-cancel | length | 33 / 0 / 160 | equal | 67 | 69 | 74.58 | 0.11 |
| close | chat | eos | stop | 27 / 0 / 58 | equal | 27 | 27 | 180.71 | 104.65 |
| close | chat | budget | length | 33 / 0 / 96 | equal | 39 | 40 | 75.52 | 0.24 |
| close | chat | stop | stop | 33 / 0 / 33 | equal | 12 | 13 | 78.58 | 0.09 |
| close | chat | turn1 | length | 1285 / 0 / 96 | equal | 38 | 38 | 75.85 | 0.15 |
| close | chat | turn2 | stop | 1413 / 1248 / 78 | equal | 34 | 35 | 75.49 | 0.16 |
| close | chat | park1 | length | 227 / 0 / 64 | equal | 25 | 25 | 75.75 | 0.12 |
| close | chat | park2 | length | 291 / 192 / 96 | equal | 26 | 28 | 76.27 | 0.16 |
| close | chat | other | stop | 24 / 0 / 65 | equal | 26 | 26 | 77.47 | 0.12 |
| close | chat | cancelled | None | hung up after 40 chunks | not judged | None | None | 0.14 | 0.13 |
| close | chat | after-cancel | length | 33 / 0 / 160 | equal | 61 | 64 | 75.49 | 0.1 |
| close | pen | eos | stop | 27 / 0 / 61 | equal | 29 | 29 | 76.45 | 0.13 |
| close | pen | budget | length | 34 / 0 / 96 | equal | 43 | 44 | 74.9 | 0.13 |
| close | pen | stop | stop | 34 / 0 / 43 | equal | 17 | 17 | 76.78 | 0.21 |
| close | pen | turn1 | length | 1239 / 0 / 96 | equal | 41 | 42 | 76.21 | 0.21 |
| close | pen | turn2 | length | 1367 / 1216 / 96 | equal | 38 | 39 | 75.06 | 0.28 |
| close | pen | park1 | length | 219 / 0 / 64 | equal | 22 | 22 | 76.44 | 0.14 |
| close | pen | park2 | length | 283 / 192 / 96 | equal | 30 | 30 | 76.08 | 0.19 |
| close | pen | other | stop | 25 / 0 / 72 | equal | 33 | 33 | 74.87 | 0.14 |
| close | pen | cancelled | None | hung up after 40 chunks | not judged | None | None | 0.08 | 0.11 |
| close | pen | after-cancel | length | 33 / 0 / 160 | equal | 73 | 75 | 75.15 | 0.15 |

rows: 40, with a judged difference: 0
