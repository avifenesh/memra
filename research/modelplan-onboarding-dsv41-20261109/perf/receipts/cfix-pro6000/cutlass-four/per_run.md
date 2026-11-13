## cutlass-four: steady medians a run, ms (calls)

| call | run1-A | run2-B | run3-B | run4-A | run5-B | run6-A | run7-A | run8-B |
|---|---|---|---|---|---|---|---|---|
| 4 rows | 12.941 (11) | 14.481 (11) | 14.474 (11) | 12.949 (11) | 14.479 (11) | 12.945 (11) | 12.945 (11) | 14.490 (11) |
| 8 rows | 17.313 (11) | 19.042 (11) | 19.067 (11) | 17.311 (11) | 19.072 (11) | 17.326 (11) | 17.316 (11) | 19.056 (11) |
| decode c1 | 10.021 (246) | 11.041 (251) | 11.046 (251) | 10.011 (246) | 11.054 (251) | 10.015 (246) | 10.021 (249) | 11.040 (250) |

Arms (median of the run medians) and each adjacent A/B pair as B / A:

| call | A | B | B / A | run1-A + run2-B | run3-B + run4-A | run5-B + run6-A | run7-A + run8-B |
|---|---|---|---|---|---|---|---|
| 4 rows | 12.945 | 14.480 | 1.119 | 1.119 | 1.118 | 1.118 | 1.119 |
| 8 rows | 17.315 | 19.062 | 1.101 | 1.100 | 1.101 | 1.101 | 1.100 |
| decode c1 | 10.018 | 11.043 | 1.102 | 1.102 | 1.103 | 1.104 | 1.102 |

Decode token tapes (gen_sha256 of each round's tokens, by cell):

- decode-c1: cd38e1b772b75d5a... in 8 runs (arms A, B)

Telemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):

- run1-A (56 lines, 3.5 s): card 0: 2310 MHz, 215 W, 66 C, thermal 0%; card 1: 2265 MHz, 190 W, 50 C, thermal 0%; card 2: 2280 MHz, 227 W, 70 C, thermal 0%; card 3: 2302 MHz, 214 W, 65 C, thermal 0%
- run2-B (60 lines, 3.9 s): card 0: 2302 MHz, 207 W, 68 C, thermal 0%; card 1: 2250 MHz, 186 W, 52 C, thermal 0%; card 2: 2280 MHz, 215 W, 70 C, thermal 0%; card 3: 2295 MHz, 205 W, 64 C, thermal 0%
- run3-B (60 lines, 3.9 s): card 0: 2302 MHz, 211 W, 69 C, thermal 0%; card 1: 2250 MHz, 187 W, 54 C, thermal 0%; card 2: 2280 MHz, 215 W, 69 C, thermal 0%; card 3: 2295 MHz, 204 W, 64 C, thermal 0%
- run4-A (56 lines, 3.5 s): card 0: 2317 MHz, 225 W, 72 C, thermal 0%; card 1: 2265 MHz, 198 W, 55 C, thermal 0%; card 2: 2287 MHz, 228 W, 70 C, thermal 0%; card 3: 2302 MHz, 211 W, 64 C, thermal 0%
- run5-B (60 lines, 3.9 s): card 0: 2295 MHz, 214 W, 72 C, thermal 0%; card 1: 2250 MHz, 189 W, 57 C, thermal 0%; card 2: 2280 MHz, 216 W, 70 C, thermal 0%; card 3: 2287 MHz, 202 W, 64 C, thermal 0%
- run6-A (56 lines, 3.5 s): card 0: 2314 MHz, 227 W, 73 C, thermal 0%; card 1: 2261 MHz, 200 W, 58 C, thermal 0%; card 2: 2287 MHz, 229 W, 70 C, thermal 0%; card 3: 2302 MHz, 213 W, 64 C, thermal 0%
- run7-A (56 lines, 3.6 s): card 0: 2310 MHz, 229 W, 73 C, thermal 0%; card 1: 2265 MHz, 202 W, 60 C, thermal 0%; card 2: 2287 MHz, 229 W, 71 C, thermal 0%; card 3: 2302 MHz, 211 W, 64 C, thermal 0%
- run8-B (60 lines, 3.9 s): card 0: 2302 MHz, 217 W, 73 C, thermal 0%; card 1: 2250 MHz, 195 W, 61 C, thermal 0%; card 2: 2272 MHz, 218 W, 71 C, thermal 0%; card 3: 2295 MHz, 203 W, 64 C, thermal 0%
