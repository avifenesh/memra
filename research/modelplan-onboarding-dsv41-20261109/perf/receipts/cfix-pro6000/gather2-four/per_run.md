## gather2-four: steady medians a run, ms (calls)

| call | run1-A | run2-B | run3-B | run4-A | run5-B | run6-A | run7-A | run8-B |
|---|---|---|---|---|---|---|---|---|
| 4 rows | 13.052 (11) | 13.028 (11) | 13.050 (11) | 13.040 (11) | 13.035 (11) | 13.031 (11) | 13.046 (11) | 13.063 (11) |
| 8 rows | 16.838 (11) | 16.803 (11) | 16.796 (11) | 16.828 (11) | 16.799 (11) | 16.825 (11) | 16.826 (11) | 16.826 (11) |
| 9 rows | 19.151 (11) | 19.151 (11) | 19.094 (11) | 19.146 (11) | 19.135 (11) | 19.125 (11) | 19.146 (11) | 19.178 (11) |
| 10 rows | 20.810 (11) | 19.767 (11) | 19.723 (11) | 20.822 (11) | 19.723 (11) | 20.777 (11) | 20.856 (11) | 19.802 (11) |
| 12 rows | 21.955 (11) | 20.738 (11) | 20.732 (11) | 21.982 (11) | 20.729 (11) | 21.949 (11) | 21.973 (11) | 20.760 (11) |
| 16 rows | 24.798 (11) | 23.674 (11) | 23.620 (11) | 24.752 (11) | 23.661 (11) | 24.788 (11) | 24.820 (11) | 23.644 (11) |
| decode c1 | 9.707 (257) | 9.674 (253) | 9.674 (253) | 9.692 (253) | 9.682 (253) | 9.688 (253) | 9.699 (253) | 9.689 (253) |

Arms (median of the run medians) and each adjacent A/B pair as B / A:

| call | A | B | B / A | run1-A + run2-B | run3-B + run4-A | run5-B + run6-A | run7-A + run8-B |
|---|---|---|---|---|---|---|---|
| 4 rows | 13.043 | 13.043 | 1.000 | 0.998 | 1.001 | 1.000 | 1.001 |
| 8 rows | 16.827 | 16.801 | 0.998 | 0.998 | 0.998 | 0.998 | 1.000 |
| 9 rows | 19.146 | 19.143 | 1.000 | 1.000 | 0.997 | 1.001 | 1.002 |
| 10 rows | 20.816 | 19.745 | 0.949 | 0.950 | 0.947 | 0.949 | 0.949 |
| 12 rows | 21.964 | 20.735 | 0.944 | 0.945 | 0.943 | 0.944 | 0.945 |
| 16 rows | 24.793 | 23.653 | 0.954 | 0.955 | 0.954 | 0.955 | 0.953 |
| decode c1 | 9.695 | 9.678 | 0.998 | 0.997 | 0.998 | 0.999 | 0.999 |

Decode token tapes (gen_sha256 of each round's tokens, by cell):

- decode-c1: cd38e1b772b75d5a... in 8 runs (arms A, B)

Telemetry inside each run (per card: median SM MHz / median W / max C / thermal-reason share):

- run1-A (72 lines, 4.5 s): card 0: 2302 MHz, 211 W, 52 C, thermal 0%; card 1: 2257 MHz, 199 W, 47 C, thermal 0%; card 2: 2280 MHz, 213 W, 51 C, thermal 0%; card 3: 2302 MHz, 204 W, 48 C, thermal 0%
- run2-B (68 lines, 4.4 s): card 0: 2310 MHz, 216 W, 55 C, thermal 0%; card 1: 2257 MHz, 203 W, 49 C, thermal 0%; card 2: 2280 MHz, 218 W, 54 C, thermal 0%; card 3: 2295 MHz, 208 W, 50 C, thermal 0%
- run3-B (68 lines, 4.4 s): card 0: 2310 MHz, 218 W, 58 C, thermal 0%; card 1: 2257 MHz, 204 W, 52 C, thermal 0%; card 2: 2287 MHz, 222 W, 57 C, thermal 0%; card 3: 2287 MHz, 210 W, 53 C, thermal 0%
- run4-A (72 lines, 4.5 s): card 0: 2302 MHz, 221 W, 61 C, thermal 0%; card 1: 2265 MHz, 204 W, 53 C, thermal 0%; card 2: 2287 MHz, 223 W, 60 C, thermal 0%; card 3: 2295 MHz, 211 W, 55 C, thermal 0%
- run5-B (68 lines, 4.4 s): card 0: 2302 MHz, 225 W, 64 C, thermal 0%; card 1: 2257 MHz, 207 W, 55 C, thermal 0%; card 2: 2280 MHz, 228 W, 62 C, thermal 0%; card 3: 2302 MHz, 214 W, 57 C, thermal 0%
- run6-A (68 lines, 4.5 s): card 0: 2310 MHz, 226 W, 66 C, thermal 0%; card 1: 2257 MHz, 208 W, 57 C, thermal 0%; card 2: 2280 MHz, 231 W, 65 C, thermal 0%; card 3: 2302 MHz, 215 W, 58 C, thermal 0%
- run7-A (72 lines, 4.5 s): card 0: 2310 MHz, 229 W, 68 C, thermal 0%; card 1: 2257 MHz, 209 W, 59 C, thermal 0%; card 2: 2280 MHz, 233 W, 67 C, thermal 0%; card 3: 2302 MHz, 216 W, 60 C, thermal 0%
- run8-B (72 lines, 4.4 s): card 0: 2310 MHz, 232 W, 69 C, thermal 0%; card 1: 2257 MHz, 213 W, 60 C, thermal 0%; card 2: 2280 MHz, 237 W, 69 C, thermal 0%; card 3: 2302 MHz, 218 W, 61 C, thermal 0%
