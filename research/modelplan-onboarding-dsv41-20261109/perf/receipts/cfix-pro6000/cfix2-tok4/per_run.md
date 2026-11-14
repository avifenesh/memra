## cfix2-tok4: steady medians a run, ms (calls)

| call | run1-A | run2-B |
|---|---|---|
| 8 rows | 19.082 (11) | 16.883 (11) |
| decode c1 | 11.041 (249) | 9.705 (243) |

Arms (median of the run medians) and each adjacent A/B pair as B / A:

| call | A | B | B / A | run1-A + run2-B |
|---|---|---|---|---|
| 8 rows | 19.082 | 16.883 | 0.885 | 0.885 |
| decode c1 | 11.041 | 9.705 | 0.879 | 0.879 |

Decode token tapes (gen_sha256 of each round's tokens, by cell):

- decode-c1: cd38e1b772b75d5a... in 2 runs (arms A, B)

Telemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):

- run1-A (60 lines, 3.7 s): card 0: 2295 MHz, 197 W, 55 C, thermal 0%; card 1: 2242 MHz, 181 W, 43 C, thermal 0%; card 2: 2265 MHz, 216 W, 70 C, thermal 0%; card 3: 2295 MHz, 203 W, 65 C, thermal 0%
- run2-B (52 lines, 3.3 s): card 0: 2302 MHz, 212 W, 59 C, thermal 0%; card 1: 2257 MHz, 193 W, 47 C, thermal 0%; card 2: 2280 MHz, 233 W, 70 C, thermal 0%; card 3: 2302 MHz, 217 W, 65 C, thermal 0%
