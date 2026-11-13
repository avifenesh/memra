| group | tapes | steps | KL cur | KL new | KL floor | d = new - cur: mean | d by-tape 95% (the pass line) | d per-step 95% | floor's half-width | top-1 cur / new / floor | steps with byte-equal top-32 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 13 | 517 | 1.509e-02 | 1.902e-02 | 1.432e-02 | +3.93e-03 | [+8.44e-04, +7.09e-03] = +5.6% to +47.0% of KL cur | [-2.88e-04, +8.56e-03] | 4.25e-03 | 96.9% / 95.9% / 97.5% | 0 | FAIL: d above zero, top-1 |
| all | 13 | 517 | 1.509e-02 | 1.902e-02 | 1.432e-02 | +3.93e-03 | [+8.44e-04, +7.09e-03] = +5.6% to +47.0% of KL cur | [-2.88e-04, +8.56e-03] | 4.25e-03 | 96.9% / 95.9% / 97.5% | 0 | FAIL: d above zero, top-1 |

router, the new order against the current route: 918920 routed rows (20160 of them decode steps) x top-6
| | rows | of all |
|---|---|---|
| the same six experts | 391838 | 42.6411% |
| another sixth expert (its least-weight choice swapped) | 136834 | 14.8907% |
| any other difference | 390248 | 42.4681% |
largest weight difference on a shared expert: 1.496e+00 (median 1.748e-02)
the MoE output of a decode step, relative difference against the current route, median / max: another sixth expert 2.05e-01 / 1.39e+00 (2865 steps); other differences 2.65e-01 / 3.53e+00 (2798); the same experts 9.89e-02 / 1.06e+00 (14497)
| layer | rows | another sixth | other |
|---|---|---|---|
| 1 | 22973 | 477 | 310 |
| 2 | 22973 | 2216 | 1924 |
| 3 | 22973 | 2879 | 3200 |
| 4 | 22973 | 3395 | 3819 |
| 5 | 22973 | 3445 | 5030 |
| 6 | 22973 | 3795 | 7576 |
| 7 | 22973 | 4157 | 7265 |
| 8 | 22973 | 4121 | 5199 |
| 9 | 22973 | 4435 | 5776 |
| 10 | 22973 | 4263 | 5980 |
| 11 | 22973 | 4492 | 6596 |
| 12 | 22973 | 4061 | 8941 |
| 13 | 22973 | 3682 | 8498 |
| 14 | 22973 | 4024 | 8217 |
| 15 | 22973 | 4381 | 4873 |
| 16 | 22973 | 4340 | 9103 |
| 17 | 22973 | 4045 | 10699 |
| 18 | 22973 | 4154 | 9463 |
| 19 | 22973 | 3501 | 8025 |
| 20 | 22973 | 4150 | 8680 |
| 21 | 22973 | 4117 | 11606 |
| 22 | 22973 | 3456 | 13893 |
| 23 | 22973 | 3475 | 14200 |
| 24 | 22973 | 3270 | 15019 |
| 25 | 22973 | 2758 | 16173 |
| 26 | 22973 | 2493 | 16269 |
| 27 | 22973 | 2653 | 16905 |
| 28 | 22973 | 2795 | 15797 |
| 29 | 22973 | 2852 | 15673 |
| 30 | 22973 | 3469 | 13780 |
| 31 | 22973 | 3204 | 14026 |
| 32 | 22973 | 3185 | 14769 |
| 33 | 22973 | 2833 | 14912 |
| 34 | 22973 | 3307 | 14107 |
| 35 | 22973 | 3361 | 12886 |
| 36 | 22973 | 3133 | 13035 |
| 37 | 22973 | 3784 | 9633 |
| 38 | 22973 | 4237 | 9448 |
| 39 | 22973 | 4439 | 8943 |

router at EQUAL INPUTS (the new order's gate against the one-chain gate on the same block inputs): 918920 routed rows x top-6
| | rows | of all |
|---|---|---|
| the same six experts | 918914 | 99.9993% |
| another sixth expert (its least-weight choice swapped) | 5 | 0.0005% |
| any other difference | 1 | 0.0001% |
largest weight difference on a shared expert: 1.323e-03 (median 8.941e-08)
| layer | rows | another sixth | other |
|---|---|---|---|
| 3 | 22973 | 1 | 0 |
| 7 | 22973 | 0 | 1 |
| 22 | 22973 | 1 | 0 |
| 24 | 22973 | 1 | 0 |
| 29 | 22973 | 1 | 0 |
| 39 | 22973 | 1 | 0 |
