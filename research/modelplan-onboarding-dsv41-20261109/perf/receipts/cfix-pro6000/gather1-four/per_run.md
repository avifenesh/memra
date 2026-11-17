## gather1-four: steady medians a run, ms (calls)

| call | run1-A | run2-B | run3-B | run4-A | run5-B | run6-A | run7-A | run8-B |
|---|---|---|---|---|---|---|---|---|
| 4 rows | 13.052 (11) | 13.061 (11) | 13.047 (11) | 13.052 (11) | 13.057 (11) | 13.055 (11) | 13.068 (11) | 13.037 (11) |
| 8 rows | 16.868 (11) | 16.874 (11) | 16.866 (11) | 16.864 (11) | 16.856 (11) | 16.864 (11) | 16.860 (11) | 16.878 (11) |
| 9 rows | 19.179 (11) | 19.370 (11) | 19.400 (11) | 19.173 (11) | 19.445 (11) | 19.191 (11) | 19.188 (11) | 19.404 (11) |
| 10 rows | 20.874 (11) | 19.848 (11) | 19.787 (11) | 20.894 (11) | 19.811 (11) | 20.863 (11) | 20.869 (11) | 19.801 (11) |
| 12 rows | 21.929 (11) | 20.762 (11) | 20.750 (11) | 22.012 (11) | 20.790 (11) | 21.990 (11) | 21.937 (11) | 20.772 (11) |
| 16 rows | 24.811 (11) | 23.760 (11) | 23.700 (11) | 24.821 (11) | 23.706 (11) | 24.813 (11) | 24.821 (11) | 23.699 (11) |
| decode c1 | 9.692 (258) | 9.711 (253) | 9.708 (253) | 9.703 (254) | 9.715 (253) | 9.715 (253) | 9.720 (253) | 9.719 (253) |

Arms (median of the run medians) and each adjacent A/B pair as B / A:

| call | A | B | B / A | run1-A + run2-B | run3-B + run4-A | run5-B + run6-A | run7-A + run8-B |
|---|---|---|---|---|---|---|---|
| 4 rows | 13.053 | 13.052 | 1.000 | 1.001 | 1.000 | 1.000 | 0.998 |
| 8 rows | 16.864 | 16.870 | 1.000 | 1.000 | 1.000 | 1.000 | 1.001 |
| 9 rows | 19.183 | 19.402 | 1.011 | 1.010 | 1.012 | 1.013 | 1.011 |
| 10 rows | 20.871 | 19.806 | 0.949 | 0.951 | 0.947 | 0.950 | 0.949 |
| 12 rows | 21.963 | 20.767 | 0.946 | 0.947 | 0.943 | 0.945 | 0.947 |
| 16 rows | 24.817 | 23.703 | 0.955 | 0.958 | 0.955 | 0.955 | 0.955 |
| decode c1 | 9.709 | 9.713 | 1.000 | 1.002 | 1.001 | 1.000 | 1.000 |

Decode token tapes (gen_sha256 of each round's tokens, by cell):

- decode-c1: cd38e1b772b75d5a... in 8 runs (arms A, B)

Telemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):

- run1-A (72 lines, 4.5 s): card 0: 2295 MHz, 217 W, 59 C, thermal 0%; card 1: 2250 MHz, 201 W, 49 C, thermal 0%; card 2: 2280 MHz, 220 W, 57 C, thermal 0%; card 3: 2302 MHz, 209 W, 51 C, thermal 0%
- run2-B (72 lines, 4.4 s): card 0: 2295 MHz, 220 W, 62 C, thermal 0%; card 1: 2250 MHz, 204 W, 51 C, thermal 0%; card 2: 2287 MHz, 225 W, 60 C, thermal 0%; card 3: 2291 MHz, 212 W, 53 C, thermal 0%
- run3-B (72 lines, 4.4 s): card 0: 2298 MHz, 224 W, 65 C, thermal 0%; card 1: 2250 MHz, 205 W, 53 C, thermal 0%; card 2: 2280 MHz, 228 W, 63 C, thermal 0%; card 3: 2302 MHz, 214 W, 56 C, thermal 0%
- run4-A (72 lines, 4.5 s): card 0: 2302 MHz, 226 W, 67 C, thermal 0%; card 1: 2257 MHz, 206 W, 55 C, thermal 0%; card 2: 2287 MHz, 231 W, 65 C, thermal 0%; card 3: 2295 MHz, 215 W, 57 C, thermal 0%
- run5-B (68 lines, 4.4 s): card 0: 2310 MHz, 230 W, 68 C, thermal 0%; card 1: 2257 MHz, 211 W, 57 C, thermal 0%; card 2: 2287 MHz, 236 W, 67 C, thermal 0%; card 3: 2295 MHz, 218 W, 59 C, thermal 0%
- run6-A (72 lines, 4.5 s): card 0: 2302 MHz, 230 W, 70 C, thermal 0%; card 1: 2250 MHz, 209 W, 58 C, thermal 0%; card 2: 2280 MHz, 235 W, 69 C, thermal 0%; card 3: 2302 MHz, 217 W, 60 C, thermal 0%
- run7-A (72 lines, 4.5 s): card 0: 2302 MHz, 230 W, 71 C, thermal 0%; card 1: 2250 MHz, 211 W, 59 C, thermal 0%; card 2: 2280 MHz, 238 W, 70 C, thermal 0%; card 3: 2302 MHz, 220 W, 61 C, thermal 0%
- run8-B (72 lines, 4.5 s): card 0: 2302 MHz, 233 W, 72 C, thermal 0%; card 1: 2250 MHz, 212 W, 60 C, thermal 0%; card 2: 2280 MHz, 239 W, 71 C, thermal 0%; card 3: 2295 MHz, 222 W, 62 C, thermal 0%
