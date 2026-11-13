| group | tapes | steps | KL cur | KL new | KL floor | d = new - cur: mean | d by-tape 95% (the pass line) | d per-step 95% | floor's half-width | top-1 cur / new / floor | steps with byte-equal top-32 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 13 | 517 | 1.509e-02 | 1.599e-02 | 1.432e-02 | +9.00e-04 | [-1.05e-03, +3.71e-03] = -7.0% to +24.6% of KL cur | [-2.90e-03, +4.71e-03] | 4.25e-03 | 96.9% / 96.7% / 97.5% | 0 | PASS |
| long | 4 | 194 | 2.402e-02 | 2.778e-02 | 3.051e-02 | +3.77e-03 | [-1.94e-03, +9.88e-03] = -8.1% to +41.1% of KL cur | [-1.01e-02, +1.66e-02] | 1.72e-02 | 95.9% / 94.3% / 96.4% | 0 | PASS |
| needle 257k | 1 | 3 | 1.062e-04 | 9.964e-04 | 3.643e-05 | +8.90e-04 | [+8.90e-04, +8.90e-04] = +838.1% to +838.1% of KL cur | [-1.40e-05, +2.68e-03] | 3.54e-05 | 100.0% / 100.0% / 100.0% | 0 | DECIDE: d above zero (current route over the band: by d alone) |
| all | 18 | 714 | 1.745e-02 | 1.913e-02 | 1.866e-02 | +1.68e-03 | [-8.39e-04, +5.21e-03] = -4.8% to +29.8% of KL cur | [-3.01e-03, +6.07e-03] | 5.52e-03 | 96.6% / 96.1% / 97.2% | 0 | PASS |
