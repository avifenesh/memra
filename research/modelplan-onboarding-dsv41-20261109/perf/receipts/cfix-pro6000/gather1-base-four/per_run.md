## gather1-base-four: steady medians a run, ms (calls)

| call | run1-A | run2-B | run3-B | run4-A | run5-B | run6-A | run7-A | run8-B |
|---|---|---|---|---|---|---|---|---|
| 4 rows | 13.032 (11) | 13.045 (11) | 13.041 (11) | 13.038 (11) | 13.049 (11) | 13.002 (11) | 13.012 (11) | 13.058 (11) |
| 8 rows | 16.945 (11) | 16.869 (11) | 16.871 (11) | 16.954 (11) | 16.880 (11) | 16.899 (11) | 16.913 (11) | 16.850 (11) |
| 9 rows | 18.990 (11) | 19.401 (11) | 19.365 (11) | 19.001 (11) | 19.399 (11) | 18.980 (11) | 18.968 (11) | 19.399 (11) |
| 10 rows | 19.643 (11) | 19.822 (11) | 19.804 (11) | 19.647 (11) | 19.835 (11) | 19.603 (11) | 19.568 (11) | 19.771 (11) |
| 12 rows | 20.741 (11) | 20.754 (11) | 20.766 (11) | 20.813 (11) | 20.752 (11) | 20.724 (11) | 20.725 (11) | 20.762 (11) |
| 16 rows | 24.028 (11) | 23.736 (11) | 23.720 (11) | 24.035 (11) | 23.736 (11) | 24.027 (11) | 24.038 (11) | 23.706 (11) |
| decode c1 | 9.923 (254) | 9.719 (253) | 9.714 (253) | 9.908 (254) | 9.716 (253) | 9.903 (254) | 9.902 (254) | 9.700 (253) |

Arms (median of the run medians) and each adjacent A/B pair as B / A:

| call | A | B | B / A | run1-A + run2-B | run3-B + run4-A | run5-B + run6-A | run7-A + run8-B |
|---|---|---|---|---|---|---|---|
| 4 rows | 13.022 | 13.047 | 1.002 | 1.001 | 1.000 | 1.004 | 1.004 |
| 8 rows | 16.929 | 16.870 | 0.997 | 0.996 | 0.995 | 0.999 | 0.996 |
| 9 rows | 18.985 | 19.399 | 1.022 | 1.022 | 1.019 | 1.022 | 1.023 |
| 10 rows | 19.623 | 19.813 | 1.010 | 1.009 | 1.008 | 1.012 | 1.010 |
| 12 rows | 20.733 | 20.758 | 1.001 | 1.001 | 0.998 | 1.001 | 1.002 |
| 16 rows | 24.032 | 23.728 | 0.987 | 0.988 | 0.987 | 0.988 | 0.986 |
| decode c1 | 9.905 | 9.715 | 0.981 | 0.979 | 0.980 | 0.981 | 0.980 |

Decode token tapes (gen_sha256 of each round's tokens, by cell):

- decode-c1: cd38e1b772b75d5a... in 8 runs (arms A, B)

Telemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):

- run1-A (72 lines, 4.5 s): card 0: 2280 MHz, 234 W, 73 C, thermal 0%; card 1: 2257 MHz, 211 W, 62 C, thermal 0%; card 2: 2287 MHz, 240 W, 72 C, thermal 0%; card 3: 2287 MHz, 222 W, 64 C, thermal 0%
- run2-B (68 lines, 4.4 s): card 0: 2302 MHz, 237 W, 73 C, thermal 0%; card 1: 2250 MHz, 216 W, 63 C, thermal 0%; card 2: 2280 MHz, 243 W, 72 C, thermal 0%; card 3: 2302 MHz, 223 W, 64 C, thermal 0%
- run3-B (72 lines, 4.4 s): card 0: 2295 MHz, 238 W, 74 C, thermal 0%; card 1: 2242 MHz, 216 W, 64 C, thermal 0%; card 2: 2280 MHz, 244 W, 72 C, thermal 0%; card 3: 2302 MHz, 225 W, 65 C, thermal 0%
- run4-A (72 lines, 4.5 s): card 0: 2302 MHz, 239 W, 75 C, thermal 0%; card 1: 2242 MHz, 215 W, 65 C, thermal 0%; card 2: 2280 MHz, 243 W, 73 C, thermal 0%; card 3: 2295 MHz, 224 W, 66 C, thermal 0%
- run5-B (72 lines, 4.4 s): card 0: 2302 MHz, 240 W, 75 C, thermal 0%; card 1: 2250 MHz, 218 W, 65 C, thermal 0%; card 2: 2284 MHz, 245 W, 74 C, thermal 0%; card 3: 2295 MHz, 225 W, 66 C, thermal 0%
- run6-A (72 lines, 4.5 s): card 0: 2317 MHz, 238 W, 76 C, thermal 0%; card 1: 2272 MHz, 217 W, 66 C, thermal 0%; card 2: 2280 MHz, 246 W, 75 C, thermal 0%; card 3: 2302 MHz, 224 W, 67 C, thermal 0%
- run7-A (69 lines, 4.5 s): card 0: 2295 MHz, 243 W, 76 C, thermal 0%; card 1: 2250 MHz, 219 W, 66 C, thermal 0%; card 2: 2272 MHz, 247 W, 75 C, thermal 0%; card 3: 2295 MHz, 226 W, 67 C, thermal 0%
- run8-B (68 lines, 4.4 s): card 0: 2302 MHz, 241 W, 76 C, thermal 0%; card 1: 2250 MHz, 220 W, 66 C, thermal 0%; card 2: 2280 MHz, 246 W, 75 C, thermal 0%; card 3: 2302 MHz, 227 W, 67 C, thermal 0%
