## moe1324-ab: steady medians a run, ms (calls)

| call | run1-A | run2-B | run3-B | run4-A | run5-B | run6-A | run7-A | run8-B |
|---|---|---|---|---|---|---|---|---|
| 4 rows | 13.048 (11) | 11.829 (11) | 11.838 (11) | 13.037 (11) | 11.841 (11) | 13.036 (11) | 13.036 (11) | 11.824 (11) |
| 8 rows | 16.857 (11) | 14.716 (11) | 14.713 (11) | 16.838 (11) | 14.711 (11) | 16.840 (11) | 16.835 (11) | 14.713 (11) |
| 9 rows | 19.195 (11) | 16.997 (11) | 16.968 (11) | 19.157 (11) | 16.937 (11) | 19.131 (11) | 19.116 (11) | 16.952 (11) |
| 10 rows | 20.829 (11) | 18.695 (11) | 18.695 (11) | 20.829 (11) | 18.760 (11) | 20.809 (11) | 20.816 (11) | 18.721 (11) |
| 12 rows | 21.987 (11) | 19.780 (11) | 19.733 (11) | 21.936 (11) | 19.808 (11) | 21.941 (11) | 21.974 (11) | 19.767 (11) |
| 16 rows | 24.876 (11) | 22.318 (11) | 22.264 (11) | 24.827 (11) | 22.344 (11) | 24.779 (11) | 24.764 (11) | 22.296 (11) |
| decode c1 | 9.715 (254) | 9.322 (250) | 9.325 (250) | 9.691 (253) | 9.319 (250) | 9.686 (253) | 9.699 (253) | 9.314 (250) |

Arms (median of the run medians) and each adjacent A/B pair as B / A:

| call | A | B | B / A | run1-A + run2-B | run3-B + run4-A | run5-B + run6-A | run7-A + run8-B |
|---|---|---|---|---|---|---|---|
| 4 rows | 13.037 | 11.834 | 0.908 | 0.907 | 0.908 | 0.908 | 0.907 |
| 8 rows | 16.839 | 14.713 | 0.874 | 0.873 | 0.874 | 0.874 | 0.874 |
| 9 rows | 19.144 | 16.960 | 0.886 | 0.886 | 0.886 | 0.885 | 0.887 |
| 10 rows | 20.822 | 18.708 | 0.898 | 0.898 | 0.898 | 0.902 | 0.899 |
| 12 rows | 21.957 | 19.773 | 0.901 | 0.900 | 0.900 | 0.903 | 0.900 |
| 16 rows | 24.803 | 22.307 | 0.899 | 0.897 | 0.897 | 0.902 | 0.900 |
| decode c1 | 9.695 | 9.320 | 0.961 | 0.960 | 0.962 | 0.962 | 0.960 |

Decode token tapes (gen_sha256 of each round's tokens, by cell):

- decode-c1: cd38e1b772b75d5a... in 8 runs (arms A, B)

Telemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):

- run1-A (72 lines, 4.5 s): card 0: 2310 MHz, 210 W, 50 C, thermal 0%; card 1: 2254 MHz, 197 W, 43 C, thermal 0%; card 2: 2280 MHz, 212 W, 49 C, thermal 0%; card 3: 2287 MHz, 202 W, 44 C, thermal 0%
- run2-B (64 lines, 4.2 s): card 0: 2310 MHz, 204 W, 53 C, thermal 0%; card 1: 2257 MHz, 189 W, 46 C, thermal 0%; card 2: 2284 MHz, 205 W, 51 C, thermal 0%; card 3: 2302 MHz, 194 W, 47 C, thermal 0%
- run3-B (64 lines, 4.2 s): card 0: 2310 MHz, 208 W, 56 C, thermal 0%; card 1: 2268 MHz, 192 W, 48 C, thermal 0%; card 2: 2287 MHz, 210 W, 54 C, thermal 0%; card 3: 2295 MHz, 198 W, 48 C, thermal 0%
- run4-A (68 lines, 4.5 s): card 0: 2302 MHz, 217 W, 58 C, thermal 0%; card 1: 2265 MHz, 201 W, 50 C, thermal 0%; card 2: 2280 MHz, 220 W, 57 C, thermal 0%; card 3: 2295 MHz, 207 W, 51 C, thermal 0%
- run5-B (68 lines, 4.2 s): card 0: 2310 MHz, 211 W, 60 C, thermal 0%; card 1: 2272 MHz, 193 W, 51 C, thermal 0%; card 2: 2287 MHz, 214 W, 59 C, thermal 0%; card 3: 2302 MHz, 199 W, 52 C, thermal 0%
- run6-A (72 lines, 4.5 s): card 0: 2302 MHz, 223 W, 63 C, thermal 0%; card 1: 2257 MHz, 205 W, 53 C, thermal 0%; card 2: 2280 MHz, 226 W, 61 C, thermal 0%; card 3: 2302 MHz, 210 W, 54 C, thermal 0%
- run7-A (72 lines, 4.5 s): card 0: 2310 MHz, 224 W, 65 C, thermal 0%; card 1: 2257 MHz, 206 W, 54 C, thermal 0%; card 2: 2284 MHz, 228 W, 64 C, thermal 0%; card 3: 2295 MHz, 211 W, 55 C, thermal 0%
- run8-B (68 lines, 4.2 s): card 0: 2310 MHz, 220 W, 66 C, thermal 0%; card 1: 2265 MHz, 198 W, 55 C, thermal 0%; card 2: 2280 MHz, 222 W, 65 C, thermal 0%; card 3: 2302 MHz, 203 W, 56 C, thermal 0%
