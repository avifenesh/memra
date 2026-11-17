## moe1324-ab-base: steady medians a run, ms (calls)

| call | run1-A | run2-B | run3-B | run4-A | run5-B | run6-A | run7-A | run8-B |
|---|---|---|---|---|---|---|---|---|
| 4 rows | 13.008 (11) | 11.832 (11) | 11.824 (11) | 13.034 (11) | 11.834 (11) | 13.022 (11) | 13.033 (11) | 11.823 (11) |
| 8 rows | 16.880 (11) | 14.709 (11) | 14.706 (11) | 16.882 (11) | 14.722 (11) | 16.891 (11) | 16.896 (11) | 14.728 (11) |
| 9 rows | 18.951 (11) | 16.932 (11) | 16.966 (11) | 18.968 (11) | 16.960 (11) | 18.982 (11) | 18.988 (11) | 16.946 (11) |
| 10 rows | 19.571 (11) | 18.684 (11) | 18.712 (11) | 19.588 (11) | 18.729 (11) | 19.542 (11) | 19.577 (11) | 18.741 (11) |
| 12 rows | 20.640 (11) | 19.751 (11) | 19.737 (11) | 20.658 (11) | 19.789 (11) | 20.644 (11) | 20.696 (11) | 19.770 (11) |
| 16 rows | 23.980 (11) | 22.278 (11) | 22.282 (11) | 23.980 (11) | 22.300 (11) | 23.945 (11) | 23.969 (11) | 22.311 (11) |
| decode c1 | 9.919 (254) | 9.315 (250) | 9.320 (250) | 9.908 (253) | 9.310 (251) | 9.914 (254) | 9.917 (253) | 9.309 (250) |

Arms (median of the run medians) and each adjacent A/B pair as B / A:

| call | A | B | B / A | run1-A + run2-B | run3-B + run4-A | run5-B + run6-A | run7-A + run8-B |
|---|---|---|---|---|---|---|---|
| 4 rows | 13.027 | 11.828 | 0.908 | 0.910 | 0.907 | 0.909 | 0.907 |
| 8 rows | 16.886 | 14.716 | 0.871 | 0.871 | 0.871 | 0.872 | 0.872 |
| 9 rows | 18.975 | 16.953 | 0.893 | 0.893 | 0.894 | 0.893 | 0.892 |
| 10 rows | 19.574 | 18.721 | 0.956 | 0.955 | 0.955 | 0.958 | 0.957 |
| 12 rows | 20.651 | 19.761 | 0.957 | 0.957 | 0.955 | 0.959 | 0.955 |
| 16 rows | 23.975 | 22.291 | 0.930 | 0.929 | 0.929 | 0.931 | 0.931 |
| decode c1 | 9.915 | 9.313 | 0.939 | 0.939 | 0.941 | 0.939 | 0.939 |

Decode token tapes (gen_sha256 of each round's tokens, by cell):

- decode-c1: cd38e1b772b75d5a... in 8 runs (arms A, B)

Telemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):

- run1-A (72 lines, 4.5 s): card 0: 2295 MHz, 223 W, 65 C, thermal 0%; card 1: 2265 MHz, 203 W, 53 C, thermal 0%; card 2: 2280 MHz, 227 W, 64 C, thermal 0%; card 3: 2295 MHz, 209 W, 55 C, thermal 0%
- run2-B (64 lines, 4.2 s): card 0: 2318 MHz, 218 W, 66 C, thermal 0%; card 1: 2265 MHz, 197 W, 54 C, thermal 0%; card 2: 2287 MHz, 223 W, 65 C, thermal 0%; card 3: 2302 MHz, 202 W, 55 C, thermal 0%
- run3-B (64 lines, 4.2 s): card 0: 2317 MHz, 220 W, 68 C, thermal 0%; card 1: 2264 MHz, 198 W, 55 C, thermal 0%; card 2: 2288 MHz, 225 W, 67 C, thermal 0%; card 3: 2302 MHz, 205 W, 57 C, thermal 0%
- run4-A (72 lines, 4.5 s): card 0: 2317 MHz, 231 W, 70 C, thermal 0%; card 1: 2257 MHz, 207 W, 57 C, thermal 0%; card 2: 2287 MHz, 235 W, 69 C, thermal 0%; card 3: 2287 MHz, 215 W, 59 C, thermal 0%
- run5-B (68 lines, 4.2 s): card 0: 2310 MHz, 225 W, 70 C, thermal 0%; card 1: 2272 MHz, 201 W, 58 C, thermal 0%; card 2: 2287 MHz, 230 W, 69 C, thermal 0%; card 3: 2302 MHz, 208 W, 59 C, thermal 0%
- run6-A (71 lines, 4.5 s): card 0: 2317 MHz, 234 W, 71 C, thermal 0%; card 1: 2254 MHz, 209 W, 59 C, thermal 0%; card 2: 2284 MHz, 239 W, 70 C, thermal 0%; card 3: 2288 MHz, 217 W, 60 C, thermal 0%
- run7-A (68 lines, 4.5 s): card 0: 2317 MHz, 236 W, 72 C, thermal 0%; card 1: 2257 MHz, 210 W, 60 C, thermal 0%; card 2: 2287 MHz, 240 W, 71 C, thermal 0%; card 3: 2295 MHz, 216 W, 61 C, thermal 0%
- run8-B (64 lines, 4.2 s): card 0: 2317 MHz, 227 W, 72 C, thermal 0%; card 1: 2264 MHz, 202 W, 60 C, thermal 0%; card 2: 2288 MHz, 233 W, 71 C, thermal 0%; card 3: 2302 MHz, 210 W, 62 C, thermal 0%
