## cfix2-two: steady medians a run, ms (calls)

| call | run1-A | run2-B | run3-B | run4-A | run5-B | run6-A | run7-A | run8-B |
|---|---|---|---|---|---|---|---|---|
| 4 rows | 8.501 (9) | 8.588 (9) | 8.590 (9) | 8.505 (9) | 8.598 (9) | 8.505 (9) | 8.516 (9) | 8.595 (9) |
| 8 rows | 11.554 (9) | 11.256 (9) | 11.263 (9) | 11.559 (9) | 11.272 (9) | 11.592 (9) | 11.644 (9) | 11.268 (9) |
| 9 rows | 12.696 (9) | 12.486 (9) | 12.495 (9) | 12.700 (9) | 12.500 (9) | 12.733 (9) | 12.741 (9) | 12.500 (9) |
| 12 rows | 13.972 (10) | 13.674 (9) | 13.678 (9) | 13.998 (10) | 13.693 (9) | 14.036 (10) | 14.023 (10) | 13.675 (9) |
| 16 rows | 16.312 (10) | 15.589 (10) | 15.599 (10) | 16.359 (10) | 15.603 (10) | 16.419 (10) | 16.454 (10) | 15.603 (10) |
| decode c1 | 6.582 (240) | 6.249 (240) | 6.264 (240) | 6.589 (240) | 6.256 (240) | 6.580 (240) | 6.594 (240) | 6.256 (240) |

Arms (median of the run medians) and each adjacent A/B pair as B / A:

| call | A | B | B / A | run1-A + run2-B | run3-B + run4-A | run5-B + run6-A | run7-A + run8-B |
|---|---|---|---|---|---|---|---|
| 4 rows | 8.505 | 8.593 | 1.010 | 1.010 | 1.010 | 1.011 | 1.009 |
| 8 rows | 11.575 | 11.266 | 0.973 | 0.974 | 0.974 | 0.972 | 0.968 |
| 9 rows | 12.717 | 12.497 | 0.983 | 0.983 | 0.984 | 0.982 | 0.981 |
| 12 rows | 14.010 | 13.677 | 0.976 | 0.979 | 0.977 | 0.976 | 0.975 |
| 16 rows | 16.389 | 15.601 | 0.952 | 0.956 | 0.954 | 0.950 | 0.948 |
| decode c1 | 6.585 | 6.256 | 0.950 | 0.950 | 0.951 | 0.951 | 0.949 |

Decode token tapes (gen_sha256 of each round's tokens, by cell):

- decode-c1: e16ea3b78575aa90... in 4 runs (arms A)
- decode-c1: 8ae2b9d6271154a9... in 4 runs (arms B)

Telemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):

- run1-A (44 lines, 2.8 s): card 0: 180 MHz, 13 W, 51 C, thermal 0%; card 1: 180 MHz, 7 W, 39 C, thermal 0%; card 2: 2235 MHz, 267 W, 68 C, thermal 0%; card 3: 2265 MHz, 262 W, 62 C, thermal 0%
- run2-B (40 lines, 2.7 s): card 0: 180 MHz, 14 W, 50 C, thermal 0%; card 1: 180 MHz, 9 W, 39 C, thermal 0%; card 2: 2254 MHz, 246 W, 68 C, thermal 0%; card 3: 2257 MHz, 229 W, 62 C, thermal 0%
- run3-B (40 lines, 2.7 s): card 0: 180 MHz, 15 W, 50 C, thermal 0%; card 1: 180 MHz, 12 W, 38 C, thermal 0%; card 2: 2257 MHz, 252 W, 68 C, thermal 0%; card 3: 2272 MHz, 217 W, 63 C, thermal 0%
- run4-A (44 lines, 2.9 s): card 0: 180 MHz, 13 W, 49 C, thermal 0%; card 1: 180 MHz, 10 W, 38 C, thermal 0%; card 2: 2227 MHz, 279 W, 70 C, thermal 0%; card 3: 2265 MHz, 257 W, 64 C, thermal 0%
- run5-B (40 lines, 2.7 s): card 0: 180 MHz, 16 W, 49 C, thermal 0%; card 1: 180 MHz, 12 W, 37 C, thermal 0%; card 2: 2242 MHz, 257 W, 69 C, thermal 0%; card 3: 2272 MHz, 239 W, 64 C, thermal 0%
- run6-A (40 lines, 2.9 s): card 0: 180 MHz, 13 W, 48 C, thermal 0%; card 1: 180 MHz, 8 W, 37 C, thermal 0%; card 2: 2190 MHz, 280 W, 70 C, thermal 0%; card 3: 2242 MHz, 268 W, 65 C, thermal 0%
- run7-A (40 lines, 2.9 s): card 0: 180 MHz, 13 W, 48 C, thermal 0%; card 1: 180 MHz, 8 W, 37 C, thermal 0%; card 2: 2220 MHz, 279 W, 71 C, thermal 0%; card 3: 2227 MHz, 268 W, 66 C, thermal 0%
- run8-B (44 lines, 2.7 s): card 0: 180 MHz, 12 W, 48 C, thermal 0%; card 1: 180 MHz, 7 W, 37 C, thermal 0%; card 2: 2235 MHz, 267 W, 71 C, thermal 0%; card 3: 2265 MHz, 254 W, 65 C, thermal 0%
