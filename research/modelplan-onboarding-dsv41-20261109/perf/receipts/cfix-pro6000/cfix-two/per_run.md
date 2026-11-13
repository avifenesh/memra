## cfix-two: steady medians a run, ms (calls)

| call | run1-A | run2-B | run3-B | run4-A |
|---|---|---|---|---|
| 4 rows | 8.591 (9) | 8.721 (9) | 8.714 (9) | 8.772 (9) |
| 8 rows | 11.806 (9) | 11.547 (9) | 11.553 (9) | 12.230 (9) |
| 9 rows | 12.567 (10) | 12.416 (9) | 12.417 (9) | 13.000 (10) |
| 12 rows | 13.926 (10) | 13.593 (9) | 13.595 (9) | 14.383 (10) |
| 16 rows | 16.319 (10) | 15.464 (10) | 15.474 (10) | 16.970 (10) |
| decode c1 | 6.657 (240) | 6.358 (240) | 6.355 (240) | 6.664 (240) |

Arms (median of the run medians) and each adjacent A/B pair as B / A:

| call | A | B | B / A | run1-A + run2-B | run3-B + run4-A |
|---|---|---|---|---|---|
| 4 rows | 8.682 | 8.717 | 1.004 | 1.015 | 0.993 |
| 8 rows | 12.018 | 11.550 | 0.961 | 0.978 | 0.945 |
| 9 rows | 12.783 | 12.416 | 0.971 | 0.988 | 0.955 |
| 12 rows | 14.155 | 13.594 | 0.960 | 0.976 | 0.945 |
| 16 rows | 16.645 | 15.469 | 0.929 | 0.948 | 0.912 |
| decode c1 | 6.660 | 6.356 | 0.954 | 0.955 | 0.954 |

Decode token tapes (gen_sha256 of each round's tokens, by cell):

- decode-c1: e16ea3b78575aa90... in 2 runs (arms A)
- decode-c1: 8ae2b9d6271154a9... in 2 runs (arms B)

Telemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):

- run1-A (44 lines, 2.9 s): card 0: 1590 MHz, 250 W, 86 C, thermal 0%; card 1: 1642 MHz, 250 W, 82 C, thermal 0%; card 2: 2220 MHz, 277 W, 71 C, thermal 0%; card 3: 2227 MHz, 253 W, 55 C, thermal 0%
- run2-B (44 lines, 2.8 s): card 0: 1590 MHz, 250 W, 87 C, thermal 0%; card 1: 1650 MHz, 250 W, 82 C, thermal 0%; card 2: 2242 MHz, 265 W, 73 C, thermal 0%; card 3: 2280 MHz, 243 W, 57 C, thermal 0%
- run3-B (44 lines, 2.8 s): card 0: 1590 MHz, 250 W, 87 C, thermal 0%; card 1: 1650 MHz, 250 W, 83 C, thermal 0%; card 2: 2242 MHz, 268 W, 75 C, thermal 0%; card 3: 2272 MHz, 244 W, 58 C, thermal 0%
- run4-A (46 lines, 2.9 s): card 0: 1582 MHz, 250 W, 87 C, thermal 0%; card 1: 1650 MHz, 250 W, 83 C, thermal 0%; card 2: 2235 MHz, 273 W, 77 C, thermal 0%; card 3: 2242 MHz, 252 W, 61 C, thermal 0%
