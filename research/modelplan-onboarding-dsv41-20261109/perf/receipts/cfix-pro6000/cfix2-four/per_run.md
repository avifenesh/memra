## cfix2-four: steady medians a run, ms (calls)

| call | run1-A | run2-B | run3-B | run4-A | run5-B | run6-A | run7-A | run8-B |
|---|---|---|---|---|---|---|---|---|
| 4 rows | 12.931 (11) | 13.063 (11) | 13.055 (11) | 12.945 (11) | 13.063 (11) | 12.935 (11) | 12.948 (11) | 13.039 (11) |
| 8 rows | 16.921 (11) | 16.840 (11) | 16.842 (11) | 16.930 (11) | 16.841 (11) | 16.937 (11) | 16.921 (11) | 16.832 (11) |
| 9 rows | 19.000 (11) | 19.185 (11) | 19.172 (11) | 19.009 (11) | 19.172 (11) | 19.025 (11) | 19.009 (11) | 19.187 (11) |
| 12 rows | 20.678 (11) | 22.113 (11) | 22.086 (11) | 20.688 (11) | 22.097 (11) | 20.683 (11) | 20.703 (11) | 22.060 (11) |
| 16 rows | 23.970 (11) | 24.860 (11) | 24.851 (11) | 24.022 (11) | 24.870 (11) | 23.977 (11) | 24.027 (11) | 24.854 (11) |
| decode c1 | 9.893 (251) | 9.713 (251) | 9.711 (251) | 9.893 (251) | 9.718 (251) | 9.903 (251) | 9.889 (251) | 9.699 (251) |

Arms (median of the run medians) and each adjacent A/B pair as B / A:

| call | A | B | B / A | run1-A + run2-B | run3-B + run4-A | run5-B + run6-A | run7-A + run8-B |
|---|---|---|---|---|---|---|---|
| 4 rows | 12.940 | 13.059 | 1.009 | 1.010 | 1.009 | 1.010 | 1.007 |
| 8 rows | 16.926 | 16.840 | 0.995 | 0.995 | 0.995 | 0.994 | 0.995 |
| 9 rows | 19.009 | 19.179 | 1.009 | 1.010 | 1.009 | 1.008 | 1.009 |
| 12 rows | 20.686 | 22.091 | 1.068 | 1.069 | 1.068 | 1.068 | 1.066 |
| 16 rows | 24.000 | 24.857 | 1.036 | 1.037 | 1.035 | 1.037 | 1.034 |
| decode c1 | 9.893 | 9.712 | 0.982 | 0.982 | 0.982 | 0.981 | 0.981 |

Decode token tapes (gen_sha256 of each round's tokens, by cell):

- decode-c1: cd38e1b772b75d5a... in 8 runs (arms A, B)

Telemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):

- run1-A (64 lines, 4.3 s): card 0: 2310 MHz, 227 W, 68 C, thermal 0%; card 1: 2250 MHz, 206 W, 57 C, thermal 0%; card 2: 2276 MHz, 232 W, 67 C, thermal 0%; card 3: 2294 MHz, 209 W, 53 C, thermal 0%
- run2-B (64 lines, 4.2 s): card 0: 2302 MHz, 230 W, 69 C, thermal 0%; card 1: 2257 MHz, 208 W, 59 C, thermal 0%; card 2: 2280 MHz, 234 W, 69 C, thermal 0%; card 3: 2295 MHz, 212 W, 55 C, thermal 0%
- run3-B (64 lines, 4.3 s): card 0: 2306 MHz, 230 W, 71 C, thermal 0%; card 1: 2254 MHz, 210 W, 60 C, thermal 0%; card 2: 2280 MHz, 237 W, 70 C, thermal 0%; card 3: 2298 MHz, 214 W, 57 C, thermal 0%
- run4-A (64 lines, 4.3 s): card 0: 2306 MHz, 233 W, 73 C, thermal 0%; card 1: 2250 MHz, 209 W, 62 C, thermal 0%; card 2: 2287 MHz, 238 W, 71 C, thermal 0%; card 3: 2294 MHz, 212 W, 59 C, thermal 0%
- run5-B (68 lines, 4.2 s): card 0: 2302 MHz, 234 W, 73 C, thermal 0%; card 1: 2250 MHz, 213 W, 63 C, thermal 0%; card 2: 2280 MHz, 241 W, 72 C, thermal 0%; card 3: 2295 MHz, 218 W, 61 C, thermal 0%
- run6-A (64 lines, 4.3 s): card 0: 2302 MHz, 235 W, 73 C, thermal 0%; card 1: 2265 MHz, 210 W, 64 C, thermal 0%; card 2: 2280 MHz, 241 W, 73 C, thermal 0%; card 3: 2298 MHz, 217 W, 62 C, thermal 0%
- run7-A (68 lines, 4.3 s): card 0: 2317 MHz, 237 W, 74 C, thermal 0%; card 1: 2265 MHz, 214 W, 65 C, thermal 0%; card 2: 2280 MHz, 244 W, 74 C, thermal 0%; card 3: 2302 MHz, 219 W, 63 C, thermal 0%
- run8-B (68 lines, 4.2 s): card 0: 2302 MHz, 238 W, 74 C, thermal 0%; card 1: 2250 MHz, 216 W, 65 C, thermal 0%; card 2: 2280 MHz, 244 W, 74 C, thermal 0%; card 3: 2302 MHz, 222 W, 64 C, thermal 0%
