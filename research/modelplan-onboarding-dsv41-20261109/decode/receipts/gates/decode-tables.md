Teacher-forced, pooled per group (KL lower bound from the shared top-32 plus a rest bucket):

| group | subset | steps | KL memra | KL floor | ratio | top-1 memra | top-1 floor | band | pass |
|---|---|---|---|---|---|---|---|---|---|
| chat | all | 517 | 1.63e-02 | 1.43e-02 | 1.14 | 97.3% | 97.5% | KL <= 2.86e-02, top-1 >= 96.1% | PASS |
| chat | entropy_gt_1nat | 22 | 9.84e-02 | 9.50e-02 | 1.04 | 77.3% | 81.8% | KL <= 1.90e-01, top-1 >= 65.4% | PASS |
| long | all | 194 | 3.68e-02 | 3.05e-02 | 1.21 | 93.8% | 96.4% | KL <= 6.10e-02, top-1 >= 93.7% | PASS |
| long | entropy_gt_1nat | 9 | 3.96e-01 | 3.83e-01 | 1.03 | 33.3% | 44.4% | KL <= 7.66e-01, top-1 >= 11.3% | PASS |
| n257k | all | 3 | 1.62e-04 | 3.64e-05 | 4.44 | 100.0% | 100.0% | KL <= 7.29e-05, top-1 >= 99.0% | FAIL |

Full-logit steps (exact KL against the tape's full logits):

| group | steps | KL memra mean / max | KL floor mean / max |
|---|---|---|---|
| chat | 78 | 1.36e-02 / 3.77e-01 | 2.39e-03 / 6.69e-02 |
| long | 22 | 8.87e-03 / 1.32e-01 | 1.43e-02 / 1.55e-01 |
| n257k | 3 | 1.80e-04 / 5.31e-04 | 5.89e-05 / 1.38e-04 |

Per case (teacher-forced):

| case | prompt | steps | ref = tape | KL memra mean / max | KL floor mean / max | top-1 memra | top-1 floor | first miss memra / floor |
|---|---|---|---|---|---|---|---|---|
| capital | 16 | 2 | yes | 1.36e-05 / 2.72e-05 | 3.34e-06 / 6.68e-06 | 2/2 | 2/2 | - / - |
| gsm | 42 | 20 | yes | 2.50e-02 / 3.77e-01 | 3.39e-03 / 3.63e-02 | 19/20 | 20/20 | 5 / - |
| code | 24 | 42 | yes | 1.95e-03 / 5.05e-02 | 3.04e-04 / 1.27e-02 | 42/42 | 42/42 | - / - |
| copy | 41 | 30 | yes | 4.42e-09 / 2.97e-08 | 5.15e-09 / 5.92e-08 | 30/30 | 30/30 | - / - |
| hebrew | 16 | 3 | yes | 2.37e-07 / 4.07e-07 | 4.20e-08 / 1.22e-07 | 3/3 | 3/3 | - / - |
| gsm_think | 68 | 50 | yes | 2.64e-03 / 2.47e-02 | 3.31e-03 / 3.99e-02 | 49/50 | 50/50 | 3 / - |
| story | 22 | 138 | yes | 3.19e-02 / 1.01e+00 | 3.42e-02 / 6.56e-01 | 132/138 | 132/138 | 19 / 19 |
| hebrew_para | 27 | 155 | yes | 1.89e-02 / 4.69e-01 | 1.50e-02 / 3.23e-01 | 150/155 | 148/155 | 23 / 13 |
| code_pal | 33 | 53 | yes | 7.03e-03 / 3.02e-01 | 2.19e-03 / 6.69e-02 | 52/53 | 53/53 | 19 / - |
| sys_json | 31 | 15 | yes | 7.61e-04 / 1.09e-02 | 4.23e-05 / 3.73e-04 | 15/15 | 15/15 | - / - |
| multiturn | 45 | 3 | yes | 1.40e-04 / 4.21e-04 | 2.88e-04 / 8.65e-04 | 3/3 | 3/3 | - / - |
| needle | 3,761 | 3 | yes | 2.87e-07 / 7.31e-07 | 3.68e-07 / 8.96e-07 | 3/3 | 3/3 | - / - |
| needle1200 | 18,343 | 3 | yes | 1.28e-06 / 3.38e-06 | 3.40e-06 / 9.87e-06 | 3/3 | 3/3 | - / - |
| decode32k_he_summary | 41,098 | 96 | yes | 5.52e-02 / 2.29e+00 | 4.13e-02 / 1.28e+00 | 90/96 | 92/96 | 13 / 13 |
| decode64k_en_continue | 64,443 | 92 | yes | 1.99e-02 / 4.78e-01 | 2.12e-02 / 3.10e-01 | 86/92 | 89/92 | 4 / 4 |
| needle4800 | 73,520 | 3 | yes | 2.11e-07 / 4.32e-07 | 4.09e-07 / 1.02e-06 | 3/3 | 3/3 | - / - |
| needle8400 | 128,745 | 3 | yes | 6.74e-05 / 1.98e-04 | 3.45e-05 / 1.02e-04 | 3/3 | 3/3 | - / - |
| needle16800 | 257,185 | 3 | yes | 1.62e-04 / 4.78e-04 | 3.64e-05 / 7.09e-05 | 3/3 | 3/3 | - / - |

Greedy generation against the vendor's tokens:

| case | vendor steps | memra steps | exact prefix | parts at | vendor margin (logit) | vendor p1 / p2 | memra's token, vendor rank | floor picks | CPU picks | near-tie | prefix steps = forced run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| capital | 2 | 2 | 2 | - | - | - | - | - | - | - | 2 steps identical |
| gsm | 20 | 20 | 5 | 5 | 0.665 | 0.564 / 0.290 | 2143, #2 | vendor's | memra's | yes | 6 steps identical |
| code | 42 | 42 | 42 | - | - | - | - | - | - | - | 42 steps identical |
| copy | 30 | 30 | 30 | - | - | - | - | - | - | - | 30 steps identical |
| hebrew | 3 | 3 | 3 | - | - | - | - | - | - | - | 3 steps identical |
| gsm_think | 50 | 47 | 3 | 3 | 0.083 | 0.401 / 0.369 | 16, #2 | vendor's | - | yes | 4 steps identical |
| story | 138 | 126 | 19 | 19 | 1.138 | 0.634 / 0.203 | 1928, #2 | memra's | - | yes | 20 steps identical |
| hebrew_para | 155 | 175 | 23 | 23 | 0.006 | 0.495 / 0.492 | 117664, #2 | memra's | - | yes | 24 steps identical |
| code_pal | 53 | 54 | 19 | 19 | 0.502 | 0.622 / 0.377 | 14, #2 | vendor's | - | yes | 20 steps identical |
| sys_json | 15 | 15 | 15 | - | - | - | - | - | - | - | 15 steps identical |
| multiturn | 3 | 3 | 3 | - | - | - | - | - | - | - | 3 steps identical |
| needle | 3 | 3 | 3 | - | - | - | - | - | - | - | 3 steps identical |
| needle1200 | 3 | 3 | 3 | - | - | - | - | - | - | - | 3 steps identical |

chat: the vendor's top-2 margin moves between its two realizations by 0.388 logits at the median step, 2.234 at p99 and 3.374 at most (517 steps).

| decode32k_he_summary | 96 | 96 | 13 | 13 | 0.485 | 0.462 / 0.285 | 5964, #2 | memra's | - | yes | 14 steps identical |
| decode64k_en_continue | 92 | 76 | 4 | 4 | 0.146 | 0.362 / 0.313 | 4496, #2 | memra's | - | yes | 5 steps identical |
| needle4800 | 3 | 3 | 3 | - | - | - | - | - | - | - | 3 steps identical |
| needle8400 | 3 | 3 | 3 | - | - | - | - | - | - | - | 3 steps identical |

long: the vendor's top-2 margin moves between its two realizations by 0.625 logits at the median step, 3.565 at p99 and 4.374 at most (194 steps).

| needle16800 | 3 | 3 | 3 | - | - | - | - | - | - | - | 3 steps identical |

n257k: the vendor's top-2 margin moves between its two realizations by 1.046 logits at the median step, 1.353 at p99 and 1.359 at most (3 steps).
