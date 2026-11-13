| group | tapes | steps | KL cur | KL new | KL floor | d = new - cur: mean | d by-tape 95% (the pass line) | d per-step 95% | floor's half-width | top-1 cur / new / floor | steps with byte-equal top-32 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 48 | 3528 | 1.900e-02 | 1.830e-02 | 1.588e-02 | -7.02e-04 | [-3.91e-03, +2.17e-03] = -20.6% to +11.4% of KL cur | [-3.59e-03, +2.09e-03] | 2.43e-03 | 97.3% / 97.5% / 97.7% | 0 | PASS |
| long | 4 | 194 | 2.402e-02 | 2.778e-02 | 3.051e-02 | +3.77e-03 | [-1.94e-03, +9.88e-03] = -8.1% to +41.1% of KL cur | [-1.01e-02, +1.66e-02] | 1.72e-02 | 95.9% / 94.3% / 96.4% | 0 | PASS |
| needle 257k | 1 | 3 | 1.062e-04 | 9.964e-04 | 3.643e-05 | +8.90e-04 | [+8.90e-04, +8.90e-04] = +838.1% to +838.1% of KL cur | [-1.40e-05, +2.68e-03] | 3.54e-05 | 100.0% / 100.0% / 100.0% | 0 | DECIDE: d above zero (current route over the band: by d alone) |
| all | 53 | 3725 | 1.925e-02 | 1.878e-02 | 1.662e-02 | -4.68e-04 | [-3.54e-03, +2.34e-03] = -18.4% to +12.2% of KL cur | [-3.24e-03, +2.19e-03] | 2.49e-03 | 97.2% / 97.4% / 97.7% | 0 | PASS |
