| group | steps | KL cur | KL new | KL floor | d = new - cur: mean [95% interval] | floor's half-width | top-1 cur / new / floor | steps with byte-equal top-32 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| chat | 517 | 1.631e-02 | 1.509e-02 | 1.432e-02 | -1.22e-03 [-5.55e-03, +2.70e-03] = -34.0% to +16.6% of KL cur | 4.22e-03 | 97.3% / 96.9% / 97.5% | 0 | PASS |
| long | 194 | 3.679e-02 | 2.402e-02 | 3.051e-02 | -1.28e-02 [-3.05e-02, +2.24e-03] = -83.0% to +6.1% of KL cur | 1.72e-02 | 93.8% / 95.9% / 96.4% | 0 | PASS |
| needle 257k | 3 | 1.618e-04 | 1.062e-04 | 3.643e-05 | -5.56e-05 [-1.74e-04, +6.95e-06] = -107.3% to +4.3% of KL cur | 3.54e-05 | 100.0% / 100.0% / 100.0% | 0 | PASS (current route over the band: by d alone) |
| all | 714 | 2.181e-02 | 1.745e-02 | 1.866e-02 | -4.35e-03 [-1.01e-02, +6.43e-04] = -46.2% to +2.9% of KL cur | 5.61e-03 | 96.4% / 96.6% / 97.2% | 0 | PASS |
