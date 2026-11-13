## cfix-tok23: steady medians a run, ms (calls)

| call | run1-A | run2-B |
|---|---|---|
| 8 rows | 12.604 (9) | 11.541 (9) |
| decode c1 | 6.911 (228) | 6.349 (222) |

Arms (median of the run medians) and each adjacent A/B pair as B / A:

| call | A | B | B / A | run1-A + run2-B |
|---|---|---|---|---|
| 8 rows | 12.604 | 11.541 | 0.916 | 0.916 |
| decode c1 | 6.911 | 6.349 | 0.919 | 0.919 |

Decode token tapes (gen_sha256 of each round's tokens, by cell):

- decode-c1: 8ae2b9d6271154a9... in 2 runs (arms A, B)

Telemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):

- run1-A (36 lines, 2.3 s): card 0: 1575 MHz, 250 W, 87 C, thermal 0%; card 1: 1657 MHz, 250 W, 82 C, thermal 0%; card 2: 2235 MHz, 224 W, 68 C, thermal 0%; card 3: 2257 MHz, 209 W, 53 C, thermal 0%
- run2-B (32 lines, 2.2 s): card 0: 1605 MHz, 250 W, 86 C, thermal 0%; card 1: 1658 MHz, 250 W, 82 C, thermal 0%; card 2: 2254 MHz, 229 W, 71 C, thermal 0%; card 3: 2272 MHz, 211 W, 56 C, thermal 0%
