# w15 served A/B: MEMRA_DSV41_BATCH_ROWS 8 (A) against 16 (B), four ranks

memra-server sha256 (every run): 1d01fad3deedb7b3be4e5aa0f56e49c99627bf0030e4af8a7fdc06b2fb498d5f

| run | arm | workload | agg out tok/s | per-request decode tok/s (median, min) | TTFT median / max s | tokens | wall s | errors | power mean max W | SM mean min MHz | temp max C | clock events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run1-A | A (MEMRA_DSV41_BATCH_ROWS=8) | conc32 | 384.61 | 13.18, 12.41 | 3.4649 / 7.2811 | 16384 | 42.599 | 0 | 264.8 | 2238 | 83.0 | 0x404 |
| run1-A | A (MEMRA_DSV41_BATCH_ROWS=8) | dconc16 | 370.71 | 25.25, 23.69 | 1.7515 / 3.436 | 8192 | 22.098 | 0 | 274.4 | 2235 | 85.0 | 0x4 |
| run2-B | B (MEMRA_DSV41_BATCH_ROWS=16) | conc32 | 481.69 | 16.9, 15.57 | 3.4748 / 7.1788 | 16384 | 34.014 | 0 | 274.0 | 2197 | 85.0 | 0x4 |
| run2-B | B (MEMRA_DSV41_BATCH_ROWS=16) | dconc16 | 471.51 | 32.915, 30.25 | 1.7401 / 3.3659 | 8192 | 17.374 | 0 | 277.8 | 2146 | 87.0 | 0x4 |
| run3-B | B (MEMRA_DSV41_BATCH_ROWS=16) | conc32 | 481.12 | 16.885, 15.56 | 3.4912 / 7.1952 | 16384 | 34.054 | 0 | 274.9 | 2193 | 85.0 | 0x404 |
| run3-B | B (MEMRA_DSV41_BATCH_ROWS=16) | dconc16 | 474.02 | 33.105, 30.42 | 1.7347 / 3.3657 | 8192 | 17.282 | 0 | 277.9 | 2144 | 87.0 | 0x4 |
| run4-A | A (MEMRA_DSV41_BATCH_ROWS=8) | conc32 | 381.69 | 13.095, 12.32 | 3.5454 / 7.4715 | 16384 | 42.924 | 0 | 271.6 | 2235 | 86.0 | 0x404 |
| run4-A | A (MEMRA_DSV41_BATCH_ROWS=8) | dconc16 | 372.97 | 25.4, 23.85 | 1.7339 / 3.4101 | 8192 | 21.964 | 0 | 278.0 | 2215 | 88.0 | 0x4 |
| run5-A | A (MEMRA_DSV41_BATCH_ROWS=8) | conc32 | 381.41 | 13.09, 12.31 | 3.564 / 7.4803 | 16384 | 42.956 | 0 | 272.0 | 2235 | 86.0 | 0x404 |
| run5-A | A (MEMRA_DSV41_BATCH_ROWS=8) | dconc16 | 373.88 | 25.47, 23.9 | 1.7389 / 3.4134 | 8192 | 21.911 | 0 | 277.9 | 2218 | 88.0 | 0x404 |
| run6-B | B (MEMRA_DSV41_BATCH_ROWS=16) | conc32 | 483.36 | 16.975, 15.63 | 3.4907 / 7.1898 | 16384 | 33.896 | 0 | 275.1 | 2194 | 85.0 | 0x4 |
| run6-B | B (MEMRA_DSV41_BATCH_ROWS=16) | dconc16 | 472.33 | 32.965, 30.3 | 1.7325 / 3.3602 | 8192 | 17.344 | 0 | 278.0 | 2138 | 87.0 | 0x4 |

## Arm medians (N = 3 runs an arm, one sample a run)

| workload | metric | A (8) median [min, max] | B (16) median [min, max] | B/A |
|---|---|---|---|---|
| conc32 | agg out tok/s | 381.69 [381.41, 384.61] | 481.69 [481.12, 483.36] | 1.262 |
| conc32 | per-request decode tok/s | 13.095 [13.09, 13.18] | 16.9 [16.885, 16.975] | 1.291 |
| conc32 | TTFT median s | 3.5454 [3.4649, 3.564] | 3.4907 [3.4748, 3.4912] | 0.985 |
| conc32 | TTFT max s | 7.4715 [7.2811, 7.4803] | 7.1898 [7.1788, 7.1952] | 0.962 |
| dconc16 | agg out tok/s | 372.97 [370.71, 373.88] | 472.33 [471.51, 474.02] | 1.266 |
| dconc16 | per-request decode tok/s | 25.4 [25.25, 25.47] | 32.965 [32.915, 33.105] | 1.298 |
| dconc16 | TTFT median s | 1.7389 [1.7339, 1.7515] | 1.7347 [1.7325, 1.7401] | 0.998 |
| dconc16 | TTFT max s | 3.4134 [3.4101, 3.436] | 3.3657 [3.3602, 3.3659] | 0.986 |
