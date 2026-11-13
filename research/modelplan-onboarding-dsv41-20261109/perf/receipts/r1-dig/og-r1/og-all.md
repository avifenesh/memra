| group | tapes | steps | KL cur | KL new | KL floor | d = new - cur: mean | d by-tape 95% (the pass line) | d per-step 95% | floor's half-width | top-1 cur / new / floor | steps with byte-equal top-32 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chat | 13 | 517 | 1.509e-02 | 1.902e-02 | 1.432e-02 | +3.93e-03 | [+8.44e-04, +7.09e-03] = +5.6% to +47.0% of KL cur | [-2.88e-04, +8.56e-03] | 4.25e-03 | 96.9% / 95.9% / 97.5% | 0 | FAIL: d above zero, top-1 |
| long | 4 | 194 | 2.402e-02 | 3.280e-02 | 3.051e-02 | +8.78e-03 | [-1.52e-05, +1.01e-02] = -0.1% to +42.1% of KL cur | [-1.06e-02, +2.86e-02] | 1.72e-02 | 95.9% / 94.8% / 96.4% | 0 | PASS |
| needle 257k | 1 | 3 | 1.062e-04 | 4.107e-03 | 3.643e-05 | +4.00e-03 | [+4.00e-03, +4.00e-03] = +3766.5% to +3766.5% of KL cur | [-4.64e-06, +1.20e-02] | 3.54e-05 | 100.0% / 100.0% / 100.0% | 0 | DECIDE: d above zero (current route over the band: by d alone) |
| all | 18 | 714 | 1.745e-02 | 2.270e-02 | 1.866e-02 | +5.25e-03 | [+2.26e-03, +7.82e-03] = +13.0% to +44.8% of KL cur | [-7.60e-04, +1.13e-02] | 5.52e-03 | 96.6% / 95.7% / 97.2% | 0 | FAIL: d above zero, top-1 |
