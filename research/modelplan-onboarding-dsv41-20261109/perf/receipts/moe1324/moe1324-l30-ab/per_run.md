## l30-1337-four: steady medians a run, ms (calls)

| call | run1-A | run2-B | run3-B | run4-A | run5-B | run6-A | run7-A | run8-B |
|---|---|---|---|---|---|---|---|---|
| 4 rows | 13.079 (11) | 11.904 (11) | 11.903 (11) | 13.069 (11) | 11.903 (11) | 13.076 (11) | 13.080 (11) | 11.892 (11) |
| 8 rows | 16.616 (11) | 14.551 (11) | 14.569 (11) | 16.631 (11) | 14.558 (11) | 16.629 (11) | 16.637 (11) | 14.542 (11) |
| 9 rows | 18.896 (11) | 16.815 (11) | 16.816 (11) | 18.892 (11) | 16.792 (11) | 18.916 (11) | 18.908 (11) | 16.775 (11) |
| 10 rows | 19.615 (11) | 17.497 (11) | 17.510 (11) | 19.601 (11) | 17.505 (11) | 19.619 (11) | 19.569 (11) | 17.524 (11) |
| 12 rows | 20.644 (11) | 18.533 (11) | 18.552 (11) | 20.610 (11) | 18.527 (11) | 20.609 (11) | 20.615 (11) | 18.499 (11) |
| 16 rows | 23.585 (11) | 21.252 (11) | 21.170 (11) | 23.600 (11) | 21.202 (11) | 23.600 (11) | 23.567 (11) | 21.240 (11) |
| decode c1 | 9.626 (252) | 9.229 (249) | 9.224 (249) | 9.617 (253) | 9.231 (249) | 9.624 (253) | 9.624 (253) | 9.214 (249) |

Arms (median of the run medians) and each adjacent A/B pair as B / A:

| call | A | B | B / A | run1-A + run2-B | run3-B + run4-A | run5-B + run6-A | run7-A + run8-B |
|---|---|---|---|---|---|---|---|
| 4 rows | 13.078 | 11.903 | 0.910 | 0.910 | 0.911 | 0.910 | 0.909 |
| 8 rows | 16.630 | 14.555 | 0.875 | 0.876 | 0.876 | 0.875 | 0.874 |
| 9 rows | 18.902 | 16.804 | 0.889 | 0.890 | 0.890 | 0.888 | 0.887 |
| 10 rows | 19.608 | 17.508 | 0.893 | 0.892 | 0.893 | 0.892 | 0.896 |
| 12 rows | 20.613 | 18.530 | 0.899 | 0.898 | 0.900 | 0.899 | 0.897 |
| 16 rows | 23.592 | 21.221 | 0.899 | 0.901 | 0.897 | 0.898 | 0.901 |
| decode c1 | 9.624 | 9.227 | 0.959 | 0.959 | 0.959 | 0.959 | 0.957 |

Decode token tapes (gen_sha256 of each round's tokens, by cell):

- decode-c1: cd38e1b772b75d5a... in 8 runs (arms A, B)

Telemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):

- run1-A (68 lines, 4.4 s): card 0: 2310 MHz, 218 W, 64 C, thermal 0%; card 1: 2257 MHz, 214 W, 59 C, thermal 0%; card 2: 2280 MHz, 233 W, 66 C, thermal 0%; card 3: 2302 MHz, 218 W, 60 C, thermal 0%
- run2-B (64 lines, 4.1 s): card 0: 2317 MHz, 210 W, 66 C, thermal 0%; card 1: 2257 MHz, 204 W, 60 C, thermal 0%; card 2: 2287 MHz, 226 W, 67 C, thermal 0%; card 3: 2302 MHz, 210 W, 60 C, thermal 0%
- run3-B (68 lines, 4.1 s): card 0: 2317 MHz, 212 W, 67 C, thermal 0%; card 1: 2265 MHz, 205 W, 61 C, thermal 0%; card 2: 2287 MHz, 230 W, 69 C, thermal 0%; card 3: 2302 MHz, 211 W, 62 C, thermal 0%
- run4-A (72 lines, 4.4 s): card 0: 2317 MHz, 225 W, 69 C, thermal 0%; card 1: 2254 MHz, 216 W, 62 C, thermal 0%; card 2: 2280 MHz, 240 W, 70 C, thermal 0%; card 3: 2295 MHz, 222 W, 63 C, thermal 0%
- run5-B (64 lines, 4.1 s): card 0: 2314 MHz, 218 W, 70 C, thermal 0%; card 1: 2261 MHz, 210 W, 63 C, thermal 0%; card 2: 2287 MHz, 233 W, 71 C, thermal 0%; card 3: 2302 MHz, 213 W, 63 C, thermal 0%
- run6-A (68 lines, 4.4 s): card 0: 2317 MHz, 225 W, 71 C, thermal 0%; card 1: 2250 MHz, 218 W, 64 C, thermal 0%; card 2: 2280 MHz, 243 W, 72 C, thermal 0%; card 3: 2295 MHz, 223 W, 64 C, thermal 0%
- run7-A (72 lines, 4.4 s): card 0: 2310 MHz, 227 W, 72 C, thermal 0%; card 1: 2257 MHz, 218 W, 65 C, thermal 0%; card 2: 2280 MHz, 244 W, 73 C, thermal 0%; card 3: 2295 MHz, 224 W, 65 C, thermal 0%
- run8-B (64 lines, 4.1 s): card 0: 2321 MHz, 220 W, 72 C, thermal 0%; card 1: 2257 MHz, 210 W, 64 C, thermal 0%; card 2: 2284 MHz, 237 W, 73 C, thermal 0%; card 3: 2302 MHz, 216 W, 65 C, thermal 0%
