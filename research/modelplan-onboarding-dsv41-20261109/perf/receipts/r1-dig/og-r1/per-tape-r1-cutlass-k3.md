Per tape, KL(vendor || x) of three realizations (k3 = the current route; cutlass = 498473abd8, which passed;
r1 = the K-segment order of the fp32 family) and the vendor's second program (floor). Sources: dev desk
og-all.json (r1 vs k3) and og-c-all.json (cutlass vs k3), by-tape tables of the order gate.

| tape | steps | k3 | cutlass | r1 | floor | r1 highest of the three |
|---|---|---|---|---|---|---|
| capital | 2 | 6.386e-07 | 3.035e-05 | 5.346e-06 | 3.341e-06 | no |
| gsm | 20 | 1.780e-02 | 3.240e-02 | 3.420e-02 | 3.395e-03 | yes |
| code | 42 | 2.369e-03 | 5.374e-03 | 2.263e-03 | 3.039e-04 | no |
| copy | 30 | 3.093e-09 | 6.003e-09 | 2.565e-09 | 5.154e-09 | no |
| hebrew | 3 | 3.334e-07 | 7.349e-08 | 9.308e-08 | 4.202e-08 | no |
| gsm_think | 50 | 4.023e-03 | 4.373e-03 | 4.541e-03 | 3.314e-03 | yes |
| story | 138 | 3.293e-02 | 3.526e-02 | 4.180e-02 | 3.417e-02 | yes |
| hebrew_para | 155 | 1.568e-02 | 1.344e-02 | 1.760e-02 | 1.497e-02 | yes |
| code_pal | 53 | 2.756e-03 | 4.155e-03 | 6.158e-03 | 2.191e-03 | yes |
| sys_json | 15 | 1.734e-03 | 4.102e-04 | 2.261e-04 | 4.228e-05 | no |
| multiturn | 3 | 3.393e-05 | 2.682e-06 | 2.510e-05 | 2.884e-04 | no |
| needle | 3 | 7.519e-07 | 9.372e-07 | 7.529e-07 | 3.680e-07 | no |
| needle1200 | 3 | 8.713e-07 | 1.099e-06 | 1.409e-06 | 3.403e-06 | yes |
| decode32k_he_summary | 96 | 3.472e-02 | 3.276e-02 | 4.267e-02 | 4.130e-02 | yes |
| decode64k_en_continue | 92 | 1.441e-02 | 2.440e-02 | 2.463e-02 | 2.125e-02 | yes |
| needle4800 | 3 | 1.505e-07 | 6.337e-07 | 5.620e-07 | 4.089e-07 | no |
| needle8400 | 3 | 4.800e-05 | 4.570e-05 | 1.711e-05 | 3.449e-05 | no |
| needle16800 | 3 | 1.062e-04 | 9.964e-04 | 4.107e-03 | 3.643e-05 | yes |

Tapes of 20 steps or more: 9; r1 the highest of the three on 7.
