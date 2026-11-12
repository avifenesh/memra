## gb10/plain

`python3 tools/summ_plain.py gb10/plain`

| long streams | class | plain class | arm | boots | plain reqs | TTFT p50 / max s | gap p50 / p95 / max s | E2E p50 / max s | plain tok/s live | long tok/s live | long tok/s idle | long longest gap s | solo gap p50 s | solo E2E s |
|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | greedy | pen | change | 2 | 12 | 0.264 / 0.289 | 0.101 / 0.102 / 0.106 | 2.68 / 2.70 | 8.99 | 23.05 | 25.41 | 0.294 | 0.072 | 1.86 1.86 1.85 1.85 |
| 1 | greedy | pen | parent | 2 | 4 | 1.577 / 1.586 | 1.361 / 1.667 / 1.771 | 34.56 / 34.92 | 0.70 | 23.76 | 26.55 | 0.294 | 0.073 | 1.89 1.89 1.88 1.87 |
| 1 | greedy | t1json | change | 2 | 12 | 1.562 / 1.867 | 0.172 / 0.174 / 0.179 | 5.70 / 6.01 | 4.06 | 17.29 | 24.51 | 0.291 | 0.072 | 1.87 1.86 1.85 1.85 |
| 1 | greedy | t1json | parent | 2 | 4 | 2.831 / 2.948 | 1.361 / 1.669 / 1.773 | 35.66 / 36.00 | 0.67 | 24.08 | 24.61 | 0.292 | 0.073 | 1.89 1.89 1.87 1.88 |
| 2 | greedy | pen | change | 2 | 12 | 0.272 / 0.334 | 0.114 / 0.116 / 0.120 | 2.99 / 3.05 | 8.03 | 40.47 | 40.88 | 0.623 | 0.072 | 1.86 1.85 1.86 1.85 |
| 2 | greedy | pen | parent | 2 | 4 | 1.807 / 1.943 | 1.605 / 1.841 / 1.943 | 40.55 / 40.76 | 0.59 | 40.78 | 41.87 | 0.600 | 0.073 | 1.89 1.89 1.88 1.87 |
| 2 | greedy | t1json | change | 2 | 12 | 1.916 / 2.054 | 0.187 / 0.189 / 0.192 | 6.39 / 6.53 | 3.78 | 31.04 | 41.95 | 0.404 | 0.072 | 1.86 1.86 1.85 1.85 |
| 2 | greedy | t1json | parent | 2 | 4 | 3.493 / 3.551 | 1.613 / 1.875 / 1.960 | 42.10 / 42.94 | 0.57 | 40.86 | 40.19 | 0.791 | 0.073 | 1.90 1.89 1.92 1.92 |
| 2 | t1 | pen | change | 2 | 12 | 0.308 / 0.346 | 0.115 / 0.116 / 0.120 | 3.03 / 3.09 | 7.94 | 39.81 | 36.84 | 0.521 | 0.072 | 1.86 1.86 1.85 1.85 |
| 2 | t1 | pen | parent | 2 | 4 | 1.933 / 1.942 | 1.738 / 2.013 / 2.167 | 44.47 / 44.78 | 0.55 | 37.79 | 38.31 | 0.789 | 0.073 | 1.89 1.89 1.88 1.88 |
| 3 | greedy | pen | change | 2 | 12 | 0.354 / 0.383 | 0.132 / 0.135 / 0.138 | 3.47 / 3.52 | 6.97 | 52.02 | 52.98 | 0.682 | 0.072 | 1.86 1.86 1.85 1.85 |
| 3 | greedy | pen | parent | 2 | 4 | 2.328 / 2.358 | 1.844 / 2.133 / 2.201 | 47.57 / 47.90 | 0.51 | 52.75 | 51.77 | 1.024 | 0.073 | 1.89 1.89 1.87 1.87 |

plain texts: 36 (cell, n) keys over 144 requests (beside streams and solo, all arms): IDENTICAL
long texts: 44 texts over 11 (cell, stream) keys: PREFIX PASS

## gb10/t1yield

`python3 tools/summ_plain.py gb10/t1yield`

| long streams | class | plain class | arm | boots | plain reqs | TTFT p50 / max s | gap p50 / p95 / max s | E2E p50 / max s | plain tok/s live | long tok/s live | long tok/s idle | long longest gap s | solo gap p50 s | solo E2E s |
|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | t1 | t1json | change | 2 | 12 | 1.639 / 1.972 | 0.173 / 0.185 / 0.200 | 5.82 / 6.11 | 3.99 | 15.82 | 22.39 | 0.310 | 0.072 | 1.86 1.86 1.90 1.89 |
| 1 | t1 | t1json | parent | 2 | 4 | 2.957 / 3.042 | 1.447 / 1.743 / 2.035 | 39.00 / 39.13 | 0.62 | 22.38 | 21.84 | 0.291 | 0.072 | 1.86 1.86 1.86 1.86 |
| 2 | t1 | t1json | change | 2 | 12 | 1.927 / 2.346 | 0.188 / 0.191 / 0.196 | 6.46 / 6.86 | 3.59 | 28.87 | 39.16 | 0.700 | 0.072 | 1.86 1.86 1.86 1.86 |
| 2 | t1 | t1json | parent | 2 | 4 | 3.329 / 3.342 | 1.728 / 1.993 / 2.141 | 45.32 / 45.45 | 0.54 | 38.38 | 36.84 | 0.780 | 0.072 | 1.87 1.86 1.86 1.87 |

plain texts: 12 (cell, n) keys over 48 requests (beside streams and solo, all arms): IDENTICAL
long texts: 12 texts over 3 (cell, stream) keys: PREFIX PASS

## gb10/plain2

`python3 tools/summ_plain.py gb10/plain2`

| long streams | class | plain class | arm | boots | plain reqs | TTFT p50 / max s | gap p50 / p95 / max s | E2E p50 / max s | plain tok/s live | long tok/s live | long tok/s idle | long longest gap s | solo gap p50 s | solo E2E s |
|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | greedy | pen | change2 | 2 | 12 | 0.284 / 0.312 | 0.100 / 0.102 / 0.104 | 2.69 / 2.71 | 8.97 | 23.11 | 25.38 | 0.294 | 0.072 | 1.84 1.84 1.85 1.85 |
| 1 | greedy | pen | parent2 | 2 | 4 | 1.439 / 1.469 | 1.337 / 1.642 / 1.921 | 34.16 / 34.18 | 0.71 | 24.23 | 26.52 | 0.289 | 0.072 | 1.85 1.85 1.85 1.85 |
| 1 | t1 | t1json | change2 | 2 | 12 | 1.631 / 1.980 | 0.172 / 0.173 / 0.175 | 5.76 / 6.10 | 4.02 | 16.02 | 22.95 | 0.291 | 0.071 | 1.85 1.84 1.85 1.85 |
| 1 | t1 | t1json | parent2 | 2 | 4 | 3.002 / 3.112 | 1.441 / 1.742 / 2.023 | 39.01 / 39.07 | 0.62 | 22.41 | 22.36 | 0.290 | 0.072 | 1.86 1.85 1.85 1.85 |
| 2 | greedy | pen | change2 | 2 | 12 | 0.309 / 0.337 | 0.114 / 0.116 / 0.121 | 2.99 / 3.07 | 7.99 | 39.92 | 41.23 | 0.722 | 0.072 | 1.84 1.84 1.85 1.85 |
| 2 | greedy | pen | parent2 | 2 | 4 | 1.852 / 1.862 | 1.586 / 1.808 / 1.923 | 40.14 / 40.16 | 0.60 | 41.26 | 42.11 | 0.594 | 0.072 | 1.84 1.85 1.85 1.85 |
| 2 | greedy | t1json | change2 | 2 | 12 | 1.905 / 2.054 | 0.187 / 0.188 / 0.190 | 6.37 / 6.53 | 3.79 | 30.94 | 42.30 | 0.406 | 0.071 | 1.84 1.84 1.85 1.85 |
| 2 | greedy | t1json | parent2 | 2 | 4 | 3.446 / 3.462 | 1.590 / 1.876 / 1.938 | 41.77 / 42.12 | 0.58 | 41.28 | 41.05 | 0.678 | 0.072 | 1.85 1.85 1.85 1.85 |
| 2 | t1 | t1json | change2 | 2 | 12 | 1.959 / 2.015 | 0.187 / 0.189 / 0.192 | 6.44 / 6.51 | 3.76 | 29.12 | 38.97 | 0.496 | 0.071 | 1.84 1.84 1.85 1.85 |
| 2 | t1 | t1json | parent2 | 2 | 4 | 3.322 / 3.331 | 1.710 / 2.088 / 2.131 | 45.47 / 45.50 | 0.54 | 38.48 | 36.22 | 0.775 | 0.072 | 1.85 1.85 1.85 1.85 |
| 3 | greedy | pen | change2 | 2 | 12 | 0.344 / 0.365 | 0.132 / 0.135 / 0.137 | 3.49 / 3.51 | 6.97 | 52.27 | 52.64 | 0.685 | 0.072 | 1.86 1.84 1.85 1.85 |
| 3 | greedy | pen | parent2 | 2 | 4 | 2.317 / 2.325 | 1.834 / 2.111 / 2.189 | 47.36 / 47.41 | 0.51 | 53.10 | 52.17 | 1.008 | 0.072 | 1.85 1.85 1.85 1.85 |

plain texts: 36 (cell, n) keys over 144 requests (beside streams and solo, all arms): IDENTICAL
long texts: 44 texts over 11 (cell, stream) keys: PREFIX PASS

## gb10/plain3

`python3 tools/summ_plain.py gb10/plain3`

| long streams | class | plain class | arm | boots | plain reqs | TTFT p50 / max s | gap p50 / p95 / max s | E2E p50 / max s | plain tok/s live | long tok/s live | long tok/s idle | long longest gap s | solo gap p50 s | solo E2E s |
|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 2 | greedy | pen | change3 | 1 | 4 | 1.790 / 1.807 | 0.116 / 0.117 / 0.119 | 4.55 / 4.57 | 5.47 | 41.56 | 39.93 | 0.509 | 0.073 | 1.88 1.88 |
| 2 | greedy | pen | parent3 | 1 | 2 | 1.812 / 1.812 | 1.601 / 1.841 / 1.852 | 40.58 / 40.58 | 0.60 | 40.95 | 41.56 | 0.603 | 0.073 | 1.88 1.88 |
| 2 | greedy | pen+hit | change3 | 1 | 8 | 1.732 / 1.947 | 0.115 / 0.117 / 0.121 | 4.45 / 4.70 | 5.38 | 40.64 | 40.68 | 1.103 | 0.073 | 1.90 1.89 |
| 2 | greedy | pen+hit | parent3 | 1 | 4 | 1.831 / 1.931 | 1.532 / 1.837 / 2.021 | 40.14 / 40.72 | 0.61 | 42.09 | 43.50 | 0.787 | 0.073 | 1.88 1.88 |
| 2 | greedy | t1json | change3 | 1 | 4 | 3.404 / 3.580 | 0.189 / 0.191 / 0.192 | 7.92 / 8.11 | 2.82 | 33.56 | 42.92 | 0.500 | 0.073 | 1.89 1.88 |
| 2 | greedy | t1json | parent3 | 1 | 2 | 3.578 / 3.578 | 1.609 / 1.846 / 1.964 | 42.21 / 42.21 | 0.57 | 40.92 | 40.13 | 0.682 | 0.073 | 1.89 1.88 |

plain texts: 12 (cell, n) keys over 36 requests (beside streams and solo, all arms): IDENTICAL
long texts: 12 texts over 6 (cell, stream) keys: PREFIX PASS

## gb10/plain5

`python3 tools/summ_plain.py gb10/plain5`

| long streams | class | plain class | arm | boots | plain reqs | TTFT p50 / max s | gap p50 / p95 / max s | E2E p50 / max s | plain tok/s live | long tok/s live | long tok/s idle | long longest gap s | solo gap p50 s | solo E2E s |
|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 2 | greedy | pen | change5 | 1 | 4 | 3.110 / 3.140 | 0.115 / 0.117 / 0.119 | 5.84 / 5.87 | 4.15 | 33.35 | 40.53 | 1.204 | 0.072 | 2.95 2.95 |
| 2 | greedy | pen | parent3 | 1 | 2 | 5.848 / 5.848 | 1.608 / 1.857 / 2.001 | 44.96 / 44.96 | 0.54 | 40.15 | 39.78 | 1.108 | 0.073 | 2.97 2.96 |
| 2 | greedy | pen+hit | change5 | 1 | 8 | 2.898 / 3.293 | 0.115 / 0.117 / 0.118 | 5.59 / 6.02 | 5.56 | 35.86 | 41.33 | 1.298 | 0.072 | 2.96 2.96 |
| 2 | greedy | pen+hit | parent3 | 1 | 4 | 6.104 / 6.200 | 1.516 / 1.822 / 1.957 | 42.79 / 44.37 | 0.59 | 41.99 | 41.89 | 1.294 | 0.073 | 2.96 2.96 |
| 2 | greedy | t1json | change5 | 1 | 4 | 4.718 / 4.920 | 0.188 / 0.191 / 0.191 | 9.24 / 9.44 | 2.60 | 29.37 | 41.46 | 1.108 | 0.072 | 2.97 2.97 |
| 2 | greedy | t1json | parent3 | 1 | 2 | 7.646 / 7.646 | 1.617 / 1.900 / 1.962 | 46.83 / 46.83 | 0.52 | 39.87 | 42.04 | 1.204 | 0.072 | 2.97 2.96 |

| long streams | arm | boot | request | TTFT s | streams tok/s until its first token | alone TTFT s |
|---:|---|---|---:|---:|---:|---|
| 2 | change5 | change5-2 | 0 | 3.11 | 27.65 | 1.21 1.21 |
| 2 | change5 | change5-2 | 1 | 3.01 | 27.22 | 1.21 1.21 |
| 2 | change5 | change5-2 | 2 | 3.14 | 26.11 | 1.21 1.21 |
| 2 | change5 | change5-2 | 3 | 2.92 | 28.41 | 1.21 1.21 |
| 2 | change5 | change5-2 | 0 | 4.92 | 30.89 | 1.22 1.23 |
| 2 | change5 | change5-2 | 1 | 4.70 | 30.86 | 1.22 1.23 |
| 2 | change5 | change5-2 | 2 | 4.46 | 32.99 | 1.22 1.23 |
| 2 | change5 | change5-2 | 3 | 4.72 | 30.10 | 1.22 1.23 |
| 2 | change5 | change5-2 | 0 | 3.29 | 24.60 | 1.21 1.22 |
| 2 | change5 | change5-2 | 0 | 0.07 | 53.56 | 1.21 1.22 |
| 2 | change5 | change5-2 | 1 | 3.03 | 25.07 | 1.21 1.22 |
| 2 | change5 | change5-2 | 1 | 0.13 | 29.72 | 1.21 1.22 |
| 2 | change5 | change5-2 | 2 | 2.90 | 27.95 | 1.21 1.22 |
| 2 | change5 | change5-2 | 2 | 0.10 | 49.85 | 1.21 1.22 |
| 2 | change5 | change5-2 | 3 | 3.05 | 27.18 | 1.21 1.22 |
| 2 | change5 | change5-2 | 3 | 0.12 | 60.22 | 1.21 1.22 |
| 2 | parent3 | parent3-1 | 0 | 5.85 | 35.40 | 1.22 1.21 |
| 2 | parent3 | parent3-1 | 1 | 5.64 | 35.82 | 1.22 1.21 |
| 2 | parent3 | parent3-1 | 0 | 7.65 | 35.70 | 1.22 1.22 |
| 2 | parent3 | parent3-1 | 1 | 7.14 | 38.91 | 1.22 1.22 |
| 2 | parent3 | parent3-1 | 0 | 6.20 | 33.55 | 1.21 1.21 |
| 2 | parent3 | parent3-1 | 0 | 1.57 | 45.89 | 1.21 1.21 |
| 2 | parent3 | parent3-1 | 1 | 6.10 | 32.77 | 1.21 1.21 |
| 2 | parent3 | parent3-1 | 1 | 1.61 | 45.98 | 1.21 1.21 |

plain texts: 18 (cell, n) keys over 36 requests (beside streams and solo, all arms): IDENTICAL
long texts: 12 texts over 6 (cell, stream) keys: PREFIX PASS

## gb10/longprime

`python3 tools/summ_plain.py gb10/longprime`

| long streams | class | plain class | arm | boots | plain reqs | TTFT p50 / max s | gap p50 / p95 / max s | E2E p50 / max s | plain tok/s live | long tok/s live | long tok/s idle | long longest gap s | solo gap p50 s | solo E2E s |
|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 2 | greedy | pen | change | 2 | 4 | 55.811 / 55.887 | 0.119 / 0.121 / 0.122 | 56.75 / 56.83 | 0.14 | 26.37 | 40.26 | 1.529 | 0.077 | 20.10 20.06 |
| 2 | greedy | pen | change2 | 2 | 4 | 55.627 / 55.843 | 0.118 / 0.120 / 0.121 | 56.57 / 56.78 | 0.14 | 26.44 | 40.70 | 1.513 | 0.076 | 20.03 19.96 |
| 2 | greedy | pen | parent | 2 | 4 | 59.507 / 59.807 | 1.592 / 1.895 / 1.901 | 71.77 / 72.11 | 0.11 | 29.51 | 43.62 | 1.615 | 0.076 | 20.17 20.04 |
| 2 | greedy | pen | parent2 | 2 | 4 | 59.332 / 59.822 | 1.590 / 1.883 / 1.886 | 71.60 / 72.08 | 0.11 | 29.50 | 43.72 | 1.611 | 0.076 | 20.06 20.06 |

| long streams | arm | boot | request | TTFT s | streams tok/s until its first token | alone TTFT s |
|---:|---|---|---:|---:|---:|---|
| 2 | change | change-2 | 0 | 55.22 | 26.53 | 19.49 |
| 2 | change | change-2 | 1 | 55.89 | 25.98 | 19.49 |
| 2 | change | change-3 | 0 | 55.27 | 26.51 | 19.45 |
| 2 | change | change-3 | 1 | 55.81 | 26.02 | 19.45 |
| 2 | change2 | change2-2 | 0 | 54.99 | 26.64 | 19.42 |
| 2 | change2 | change2-2 | 1 | 55.63 | 26.10 | 19.42 |
| 2 | change2 | change2-3 | 0 | 54.96 | 26.56 | 19.36 |
| 2 | change2 | change2-3 | 1 | 55.84 | 26.00 | 19.36 |
| 2 | parent | parent-1 | 0 | 57.73 | 27.38 | 19.56 |
| 2 | parent | parent-1 | 1 | 59.51 | 26.53 | 19.56 |
| 2 | parent | parent-4 | 0 | 58.03 | 27.24 | 19.43 |
| 2 | parent | parent-4 | 1 | 59.81 | 26.40 | 19.43 |
| 2 | parent2 | parent2-1 | 0 | 57.60 | 27.45 | 19.45 |
| 2 | parent2 | parent2-1 | 1 | 59.33 | 26.61 | 19.45 |
| 2 | parent2 | parent2-5 | 0 | 58.32 | 27.11 | 19.45 |
| 2 | parent2 | parent2-5 | 1 | 59.82 | 26.39 | 19.45 |

plain texts: 3 (cell, n) keys over 24 requests (beside streams and solo, all arms): IDENTICAL
long texts: 16 texts over 2 (cell, stream) keys: PREFIX PASS

## rtx5090/plain

`python3 tools/summ_plain.py rtx5090/plain`

| long streams | class | plain class | arm | boots | plain reqs | TTFT p50 / max s | gap p50 / p95 / max s | E2E p50 / max s | plain tok/s live | long tok/s live | long tok/s idle | long longest gap s | solo gap p50 s | solo E2E s |
|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | greedy | pen | change | 2 | 12 | 0.168 / 0.186 | 0.059 / 0.060 / 0.063 | 1.56 / 1.60 | 15.41 | 41.73 | 72.71 | 0.182 | 0.026 | 0.74 0.73 0.74 0.74 |
| 1 | greedy | pen | parent | 2 | 4 | 0.642 / 0.680 | 0.457 / 0.561 / 0.581 | 11.74 / 12.04 | 2.06 | 70.73 | 71.79 | 0.174 | 0.026 | 0.74 0.78 0.74 0.73 |
| 1 | greedy | t1json | change | 2 | 12 | 0.698 / 1.116 | 0.060 / 0.061 / 0.063 | 2.06 / 2.16 | 10.91 | 48.99 | 72.88 | 0.187 | 0.025 | 0.50 0.73 0.50 0.73 |
| 1 | greedy | t1json | parent | 2 | 4 | 1.488 / 1.533 | 0.470 / 0.578 / 0.647 | 12.53 / 12.68 | 1.86 | 69.16 | 72.80 | 0.189 | 0.025 | 0.50 0.74 0.50 0.73 |
| 2 | greedy | pen | change | 2 | 12 | 0.201 / 0.219 | 0.073 / 0.080 / 0.087 | 1.94 / 2.09 | 12.38 | 62.23 | 96.07 | 0.318 | 0.026 | 0.73 0.73 0.73 0.73 |
| 2 | greedy | pen | parent | 2 | 4 | 0.836 / 0.921 | 0.720 / 0.846 / 0.912 | 18.40 / 18.57 | 1.32 | 91.14 | 93.55 | 0.357 | 0.026 | 0.73 0.73 0.75 0.75 |

plain texts: 18 (cell, n) keys over 72 requests (beside streams and solo, all arms): IDENTICAL
long texts: 16 texts over 4 (cell, stream) keys: PREFIX PASS

## rtx5090/plain2

`python3 tools/summ_plain.py rtx5090/plain2`

| long streams | class | plain class | arm | boots | plain reqs | TTFT p50 / max s | gap p50 / p95 / max s | E2E p50 / max s | plain tok/s live | long tok/s live | long tok/s idle | long longest gap s | solo gap p50 s | solo E2E s |
|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | greedy | pen | change2 | 1 | 6 | 0.175 / 0.181 | 0.059 / 0.060 / 0.061 | 1.56 / 1.58 | 15.40 | 41.60 | 73.07 | 0.174 | 0.026 | 0.73 0.73 |
| 1 | greedy | pen | parent2 | 1 | 2 | 0.671 / 0.671 | 0.463 / 0.561 / 0.577 | 11.96 / 11.96 | 2.03 | 69.95 | 71.63 | 0.180 | 0.026 | 0.73 0.73 |
| 1 | greedy | t1json | change2 | 1 | 6 | 0.683 / 1.099 | 0.060 / 0.062 / 0.062 | 2.10 / 2.17 | 10.81 | 48.44 | 72.72 | 0.183 | 0.025 | 0.50 0.73 |
| 1 | greedy | t1json | parent2 | 1 | 2 | 1.506 / 1.506 | 0.476 / 0.619 / 0.620 | 12.70 / 12.70 | 1.85 | 68.56 | 72.05 | 0.188 | 0.025 | 0.50 0.73 |
| 2 | greedy | pen | change2 | 1 | 6 | 0.201 / 0.210 | 0.076 / 0.078 / 0.090 | 1.97 / 2.00 | 12.23 | 61.09 | 94.87 | 0.213 | 0.026 | 0.73 0.73 |
| 2 | greedy | pen | parent2 | 1 | 2 | 0.879 / 0.879 | 0.717 / 0.828 / 0.880 | 18.11 / 18.11 | 1.33 | 91.74 | 94.44 | 0.331 | 0.026 | 0.73 0.73 |

plain texts: 18 (cell, n) keys over 36 requests (beside streams and solo, all arms): IDENTICAL
long texts: 8 texts over 4 (cell, stream) keys: PREFIX PASS

## rtx5090/plain5

`python3 tools/summ_plain.py rtx5090/plain5`

| long streams | class | plain class | arm | boots | plain reqs | TTFT p50 / max s | gap p50 / p95 / max s | E2E p50 / max s | plain tok/s live | long tok/s live | long tok/s idle | long longest gap s | solo gap p50 s | solo E2E s |
|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | greedy | pen | change5 | 1 | 4 | 1.539 / 1.577 | 0.054 / 0.056 / 0.056 | 2.82 / 2.84 | 8.59 | 36.67 | 76.20 | 0.757 | 0.022 | 1.52 1.59 |
| 1 | greedy | pen | parent3 | 1 | 2 | 2.812 / 2.812 | 0.500 / 0.596 / 0.633 | 15.31 / 15.31 | 1.64 | 61.24 | 69.86 | 0.994 | 0.025 | 1.82 1.90 |
| 1 | greedy | t1json | change5 | 1 | 4 | 2.142 / 2.406 | 0.057 / 0.059 / 0.060 | 3.49 / 3.75 | 6.84 | 39.96 | 76.03 | 0.844 | 0.023 | 1.66 1.71 |
| 1 | greedy | t1json | parent3 | 1 | 2 | 3.662 / 3.662 | 0.523 / 0.612 / 0.693 | 16.29 / 16.29 | 1.50 | 58.70 | 70.68 | 1.032 | 0.025 | 1.78 1.92 |
| 2 | greedy | pen | change5 | 1 | 4 | 1.949 / 1.976 | 0.071 / 0.075 / 0.076 | 3.61 / 3.63 | 6.72 | 53.32 | 98.66 | 0.921 | 0.023 | 1.58 1.67 |
| 2 | greedy | pen | parent3 | 1 | 2 | 3.575 / 3.575 | 0.746 / 0.885 / 0.926 | 21.52 / 21.52 | 1.12 | 83.86 | 89.51 | 1.000 | 0.024 | 1.71 1.80 |
| 2 | greedy | pen+hit | change5 | 1 | 8 | 1.930 / 2.064 | 0.075 / 0.078 / 0.081 | 3.67 / 3.85 | 8.45 | 54.93 | 94.63 | 0.935 | 0.024 | 1.66 1.73 |
| 2 | greedy | pen+hit | parent3 | 1 | 4 | 3.391 / 3.498 | 0.729 / 0.837 / 0.878 | 20.49 / 21.03 | 1.22 | 87.81 | 80.42 | 1.070 | 0.023 | 1.60 1.71 |

| long streams | arm | boot | request | TTFT s | streams tok/s until its first token | alone TTFT s |
|---:|---|---|---:|---:|---:|---|
| 1 | change5 | change5-1 | 0 | 1.52 | 26.94 | 0.99 1.05 |
| 1 | change5 | change5-1 | 1 | 1.46 | 27.33 | 0.99 1.05 |
| 1 | change5 | change5-1 | 2 | 1.58 | 25.37 | 0.99 1.05 |
| 1 | change5 | change5-1 | 3 | 1.54 | 24.68 | 0.99 1.05 |
| 1 | change5 | change5-1 | 0 | 2.41 | 42.81 | 1.10 1.15 |
| 1 | change5 | change5-1 | 1 | 1.95 | 38.04 | 1.10 1.15 |
| 1 | change5 | change5-1 | 2 | 2.14 | 35.02 | 1.10 1.15 |
| 1 | change5 | change5-1 | 3 | 2.11 | 36.02 | 1.10 1.15 |
| 1 | parent3 | parent3-2 | 0 | 2.50 | 41.21 | 1.23 1.30 |
| 1 | parent3 | parent3-2 | 1 | 2.81 | 35.91 | 1.23 1.30 |
| 1 | parent3 | parent3-2 | 0 | 3.66 | 45.61 | 1.18 1.33 |
| 1 | parent3 | parent3-2 | 1 | 3.57 | 37.54 | 1.18 1.33 |
| 2 | change5 | change5-1 | 0 | 1.89 | 42.42 | 1.04 1.12 |
| 2 | change5 | change5-1 | 1 | 1.98 | 42.01 | 1.04 1.12 |
| 2 | change5 | change5-1 | 2 | 1.88 | 45.84 | 1.04 1.12 |
| 2 | change5 | change5-1 | 3 | 1.95 | 45.14 | 1.04 1.12 |
| 2 | change5 | change5-1 | 0 | 2.00 | 40.49 | 1.10 1.16 |
| 2 | change5 | change5-1 | 0 | 0.19 | 57.43 | 1.10 1.16 |
| 2 | change5 | change5-1 | 1 | 1.93 | 42.48 | 1.10 1.16 |
| 2 | change5 | change5-1 | 1 | 0.23 | 66.06 | 1.10 1.16 |
| 2 | change5 | change5-1 | 2 | 2.06 | 41.66 | 1.10 1.16 |
| 2 | change5 | change5-1 | 2 | 0.22 | 36.32 | 1.10 1.16 |
| 2 | change5 | change5-1 | 3 | 2.05 | 42.43 | 1.10 1.16 |
| 2 | change5 | change5-1 | 3 | 0.22 | 40.66 | 1.10 1.16 |
| 2 | parent3 | parent3-2 | 0 | 3.58 | 56.78 | 1.14 1.23 |
| 2 | parent3 | parent3-2 | 1 | 3.47 | 59.09 | 1.14 1.23 |
| 2 | parent3 | parent3-2 | 0 | 3.50 | 58.03 | 1.04 1.15 |
| 2 | parent3 | parent3-2 | 0 | 0.86 | 86.29 | 1.04 1.15 |
| 2 | parent3 | parent3-2 | 1 | 3.39 | 59.87 | 1.04 1.15 |
| 2 | parent3 | parent3-2 | 1 | 0.82 | 86.70 | 1.04 1.15 |

plain texts: 24 (cell, n) keys over 46 requests (beside streams and solo, all arms): IDENTICAL
long texts: 12 texts over 6 (cell, stream) keys: PREFIX PASS

## gb10/riders

`python3 tools/summ_riders.py gb10/riders`

| long streams | riders | class | arm | boots | window s | streams tok/s before | streams tok/s live | riders tok/s live | all tok/s live | rider gap p50 / p95 s | rider TTFT p50 s | rider E2E max s | riders alone tok/s |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2 | 4 | pen | change3 | 1 | 8.8 | 43.65 | 33.28 | 29.07 | 62.35 | 0.133 / 0.136 | 0.430 | 8.81 | 44.76 |
| 2 | 4 | pen | parent3 | 1 | 106.2 | 44.64 | 40.32 | 2.41 | 42.73 | 1.615 / 1.849 | 2.084 | 106.21 | 45.60 |
| 2 | 4 | vendor | change3 | 1 | 6.7 | 43.82 | 20.48 | 37.99 | 58.47 | 0.143 / 0.164 | 0.676 | 6.74 | 46.08 |
| 2 | 4 | vendor | parent3 | 1 | 6.7 | 43.82 | 20.59 | 38.20 | 58.79 | 0.143 / 0.162 | 0.650 | 6.70 | 46.58 |
| 2 | 8 | vendor | change3 | 1 | 9.3 | 43.64 | 14.64 | 55.11 | 69.75 | 0.135 / 0.139 | 0.676 | 9.29 | 57.78 |
| 2 | 8 | vendor | parent3 | 1 | 9.3 | 44.64 | 14.27 | 54.92 | 69.18 | 0.135 / 0.139 | 0.730 | 9.32 | 57.02 |

plain texts: 16 (riders, class, n) keys over 64 requests (beside streams and alone, all arms): DIFFER {(8, 'vendor', 0): {('9c9afb2481b792e7', 64), ('0f85961509321c8b', 64), ('c6688e3acd19f017', 64)}, (8, 'vendor', 1): {('d3e2d8e5f4e0fee3', 64), ('7cbc433d84785da1', 64)}}
long texts: 12 texts over 6 keys: PREFIX PASS

## gb10/riders9

`python3 tools/summ_riders.py gb10/riders9`

| long streams | riders | class | arm | boots | window s | streams tok/s before | streams tok/s live | riders tok/s live | all tok/s live | rider gap p50 / p95 s | rider TTFT p50 s | rider E2E max s | riders alone tok/s |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2 | 6 | pen | change5 | 1 | 6.7 | 44.65 | 19.95 | 57.59 | 77.53 | 0.096 / 0.099 | 0.571 | 6.67 | 61.15 |
| 2 | 6 | pen | parent3 | 1 | 6.7 | 44.65 | 19.93 | 57.55 | 77.48 | 0.095 / 0.099 | 0.574 | 6.67 | 61.13 |
| 2 | 8 | pen | change5 | 1 | 7.5 | 43.82 | 18.19 | 68.49 | 86.68 | 0.106 / 0.110 | 0.721 | 7.48 | 74.03 |
| 2 | 8 | pen | parent3 | 1 | 7.5 | 43.82 | 18.22 | 68.59 | 86.81 | 0.106 / 0.110 | 0.717 | 7.46 | 74.24 |

plain texts: 14 (riders, class, n) keys over 56 requests (beside streams and alone, all arms): IDENTICAL
long texts: 8 texts over 4 keys: PREFIX PASS

## gb10/riders10

`python3 tools/summ_riders.py gb10/riders10`

| long streams | riders | class | arm | boots | window s | streams tok/s before | streams tok/s live | riders tok/s live | all tok/s live | rider gap p50 / p95 s | rider TTFT p50 s | rider E2E max s | riders alone tok/s |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2 | 4 | pen | change5 | 1 | 8.8 | 43.80 | 33.37 | 29.16 | 62.53 | 0.132 / 0.135 | 0.459 | 8.78 | 45.57 |
| 2 | 4 | pen | parent3 | 1 | 106.3 | 43.82 | 40.34 | 2.41 | 42.74 | 1.607 / 1.883 | 2.061 | 106.33 | 44.80 |
| 2 | 5 | pen | change5 | 1 | 6.5 | 44.65 | 20.51 | 49.35 | 69.86 | 0.093 / 0.096 | 0.553 | 6.48 | 54.63 |
| 2 | 5 | pen | parent3 | 1 | 6.5 | 44.64 | 20.56 | 49.46 | 70.02 | 0.093 / 0.096 | 0.547 | 6.47 | 54.74 |

plain texts: 9 (riders, class, n) keys over 36 requests (beside streams and alone, all arms): IDENTICAL
long texts: 8 texts over 4 keys: PREFIX PASS

## rtx5090/riders5

`python3 tools/summ_riders.py rtx5090/riders5`

| long streams | riders | class | arm | boots | window s | streams tok/s before | streams tok/s live | riders tok/s live | all tok/s live | rider gap p50 / p95 s | rider TTFT p50 s | rider E2E max s | riders alone tok/s |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 2 | 4 | pen | change5 | 1 | 2.7 | 107.30 | 50.39 | 95.56 | 145.95 | 0.037 / 0.039 | 0.287 | 2.68 | 121.09 |
| 2 | 4 | pen | parent3 | 1 | 2.6 | 108.14 | 50.91 | 97.26 | 148.18 | 0.037 / 0.039 | 0.253 | 2.63 | 119.22 |
| 2 | 4 | vendor | change5 | 1 | 3.6 | 106.32 | 36.34 | 70.48 | 106.83 | 0.053 / 0.054 | 0.284 | 3.63 | 81.73 |
| 2 | 4 | vendor | parent3 | 1 | 3.6 | 107.14 | 36.25 | 70.84 | 107.08 | 0.052 / 0.054 | 0.290 | 3.61 | 82.68 |
| 2 | 8 | pen | change5 | 1 | 3.9 | 106.64 | 33.99 | 129.88 | 163.88 | 0.055 / 0.058 | 0.428 | 3.94 | 148.55 |
| 2 | 8 | pen | parent3 | 1 | 3.9 | 106.63 | 34.60 | 132.20 | 166.80 | 0.055 / 0.058 | 0.388 | 3.87 | 150.61 |
| 2 | 8 | vendor | change5 | 1 | 5.9 | 107.15 | 22.42 | 86.96 | 109.38 | 0.085 / 0.088 | 0.471 | 5.89 | 97.61 |
| 2 | 8 | vendor | parent3 | 1 | 5.9 | 106.65 | 22.60 | 86.37 | 108.97 | 0.086 / 0.089 | 0.452 | 5.93 | 94.51 |

plain texts: 24 (riders, class, n) keys over 96 requests (beside streams and alone, all arms): DIFFER {(4, 'vendor', 0): {('6e1005c63acf10c5', 64), ('b41d0c67ade4dbbd', 64)}}
long texts: 16 texts over 8 keys: PREFIX PASS

## gb10/new

`python3 tools/summ_newcomer.py gb10/new`

| long streams | class (long/new) | arm | boots | newcomers | TTFT p50 s | TTFT p95 s | TTFT max s | 1st to 2nd token p50 s | newcomer E2E p50 s | long tok/s (sum), per boot | all tok/s (long + newcomers), per boot | long longest gap s, max | long ITL p50 / p99 ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| 0 | greedy/greedy | change | 2 | 24 | 0.211 | 0.224 | 0.251 | 0.100 | 0.712 | - | 7.71, 7.75 | - | - |
| 0 | greedy/greedy | parent | 2 | 24 | 0.211 | 0.224 | 0.255 | 0.100 | 0.710 | - | 7.72, 7.75 | - | - |
| 1 | greedy/greedy | change | 2 | 24 | 0.269 | 0.304 | 0.305 | 0.005 | 0.926 | 21.34, 21.37 | 28.48, 28.53 | 0.409 | 0.0 / 399.2 |
| 1 | greedy/greedy | parent | 2 | 24 | 0.282 | 0.305 | 0.308 | 0.005 | 0.931 | 21.35, 21.44 | 28.48, 28.58 | 0.412 | 0.0 / 400.0 |
| 2 | greedy/greedy | change | 2 | 24 | 0.271 | 0.313 | 0.315 | 0.007 | 1.014 | 36.13, 36.67 | 42.96, 43.48 | 0.615 | 0.0 / 418.6 |
| 2 | greedy/greedy | parent | 2 | 24 | 0.266 | 0.308 | 0.318 | 0.007 | 1.021 | 36.16, 36.62 | 42.99, 43.46 | 0.623 | 0.0 / 418.3 |
| 2 | t1/t1 | change | 2 | 24 | 0.271 | 0.317 | 0.319 | 0.007 | 1.100 | 35.84, 36.12 | 42.53, 42.84 | 0.608 | 0.0 / 420.2 |
| 2 | t1/t1 | parent | 2 | 24 | 0.257 | 0.314 | 0.318 | 0.007 | 1.070 | 35.8, 36.07 | 42.51, 42.83 | 0.617 | 0.0 / 418.8 |
| 3 | greedy/greedy | change | 2 | 24 | 0.283 | 0.335 | 0.336 | 0.009 | 1.151 | 48.16, 47.96 | 54.55, 54.39 | 0.927 | 0.0 / 445.0 |
| 3 | greedy/greedy | parent | 2 | 24 | 0.280 | 0.336 | 0.337 | 0.009 | 1.154 | 47.8, 48.2 | 54.24, 54.65 | 1.044 | 0.0 / 442.9 |

prefix check: 32 long texts, 240 newcomer texts, 0 mismatches -> PASS

## gb10/new2

`python3 tools/summ_newcomer.py gb10/new2`

| long streams | class (long/new) | arm | boots | newcomers | TTFT p50 s | TTFT p95 s | TTFT max s | 1st to 2nd token p50 s | newcomer E2E p50 s | long tok/s (sum), per boot | all tok/s (long + newcomers), per boot | long longest gap s, max | long ITL p50 / p99 ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| 0 | greedy/greedy | change2 | 1 | 12 | 0.212 | 0.222 | 0.222 | 0.100 | 0.717 | - | 7.74 | - | - |
| 0 | greedy/greedy | parent2 | 1 | 12 | 0.212 | 0.221 | 0.221 | 0.099 | 0.715 | - | 7.74 | - | - |
| 1 | greedy/greedy | change2 | 1 | 12 | 0.267 | 0.301 | 0.301 | 0.005 | 0.912 | 21.4 | 28.59 | 0.409 | 0.0 / 397.0 |
| 1 | greedy/greedy | parent2 | 1 | 12 | 0.263 | 0.301 | 0.301 | 0.005 | 0.915 | 21.44 | 28.65 | 0.407 | 0.0 / 397.1 |
| 2 | greedy/greedy | change2 | 1 | 12 | 0.271 | 0.316 | 0.316 | 0.007 | 1.018 | 36.62 | 43.44 | 0.614 | 0.0 / 417.0 |
| 2 | greedy/greedy | parent2 | 1 | 12 | 0.268 | 0.314 | 0.314 | 0.007 | 1.035 | 36.92 | 43.75 | 0.611 | 0.1 / 416.2 |
| 2 | t1/t1 | change2 | 1 | 12 | 0.262 | 0.310 | 0.310 | 0.007 | 1.101 | 35.97 | 42.72 | 0.514 | 0.0 / 417.4 |
| 2 | t1/t1 | parent2 | 1 | 12 | 0.266 | 0.317 | 0.317 | 0.007 | 1.099 | 36.13 | 42.84 | 0.61 | 0.0 / 416.2 |
| 3 | greedy/greedy | change2 | 1 | 12 | 0.280 | 0.337 | 0.337 | 0.009 | 1.160 | 48.15 | 54.59 | 1.035 | 0.0 / 440.7 |
| 3 | greedy/greedy | parent2 | 1 | 12 | 0.286 | 0.332 | 0.332 | 0.009 | 1.154 | 48.36 | 54.78 | 0.754 | 0.0 / 448.2 |

prefix check: 16 long texts, 120 newcomer texts, 0 mismatches -> PASS

## rtx5090/new

`python3 tools/summ_newcomer.py rtx5090/new`

| long streams | class (long/new) | arm | boots | newcomers | TTFT p50 s | TTFT p95 s | TTFT max s | 1st to 2nd token p50 s | newcomer E2E p50 s | long tok/s (sum), per boot | all tok/s (long + newcomers), per boot | long longest gap s, max | long ITL p50 / p99 ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| 0 | greedy/greedy | change | 2 | 24 | 0.141 | 0.155 | 0.162 | 0.035 | 0.320 | - | 9.66, 9.67 | - | - |
| 0 | greedy/greedy | parent | 2 | 24 | 0.141 | 0.151 | 0.160 | 0.035 | 0.320 | - | 9.67, 9.67 | - | - |
| 1 | greedy/greedy | change | 2 | 24 | 0.159 | 0.182 | 0.189 | 0.002 | 0.439 | 63.23, 63.04 | 72.34, 72.08 | 0.235 | 0.0 / 51.9 |
| 1 | greedy/greedy | parent | 2 | 24 | 0.160 | 0.175 | 0.175 | 0.001 | 0.439 | 63.09, 63.21 | 72.16, 72.29 | 0.219 | 0.0 / 51.9 |
| 2 | greedy/greedy | change | 2 | 24 | 0.202 | 0.235 | 0.253 | 0.034 | 1.333 | 79.6, 78.66 | 85.66, 84.64 | 0.436 | 0.0 / 187.8 |
| 2 | greedy/greedy | parent | 2 | 24 | 0.842 | 0.912 | 0.916 | 0.035 | 2.001 | 79.28, 83.63 | 84.07, 88.53 | 0.393 | 0.0 / 193.9 |
| 2 | t1/t1 | change | 2 | 24 | 0.214 | 0.229 | 0.232 | 0.035 | 1.429 | 68.05, 63.14 | 73.95, 68.91 | 0.373 | 0.0 / 211.2 |
| 2 | t1/t1 | parent | 2 | 24 | 0.896 | 1.029 | 1.041 | 0.035 | 2.070 | 71.14, 71.58 | 75.86, 76.29 | 0.405 | 0.0 / 200.7 |
| 3 | greedy/greedy | change | 2 | 24 | 0.141 | 0.151 | 0.152 | 0.031 | 0.638 | 97.84, 97.49 | 106.01, 105.67 | 0.155 | 28.1 / 151.9 |
| 3 | greedy/greedy | parent | 2 | 24 | 0.142 | 0.155 | 0.183 | 0.031 | 0.640 | 96.8, 96.47 | 104.97, 104.65 | 0.187 | 28.2 / 153.9 |

prefix check: 32 long texts, 240 newcomer texts, 0 mismatches -> PASS

## rtx5090/new5

`python3 tools/summ_newcomer.py rtx5090/new5`

| long streams | class (long/new) | arm | boots | newcomers | TTFT p50 s | TTFT p95 s | TTFT max s | 1st to 2nd token p50 s | newcomer E2E p50 s | long tok/s (sum), per boot | all tok/s (long + newcomers), per boot | long longest gap s, max | long ITL p50 / p99 ms |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| 1 | greedy/greedy | change5 | 1 | 12 | 0.170 | 0.197 | 0.197 | 0.002 | 0.480 | 63.24 | 72.17 | 0.242 | 0.0 / 59.5 |
| 1 | greedy/greedy | parent3 | 1 | 12 | 0.159 | 0.176 | 0.176 | 0.001 | 0.461 | 65.2 | 74.25 | 0.232 | 0.0 / 55.7 |
| 2 | greedy/greedy | change5 | 1 | 12 | 0.212 | 0.238 | 0.238 | 0.032 | 1.381 | 74.66 | 80.61 | 0.347 | 0.0 / 201.9 |
| 2 | greedy/greedy | parent3 | 1 | 12 | 0.860 | 0.966 | 0.966 | 0.032 | 2.050 | 77.55 | 82.31 | 0.437 | 0.0 / 195.1 |
| 2 | t1/t1 | change5 | 1 | 12 | 0.207 | 0.227 | 0.227 | 0.032 | 1.383 | 66.46 | 72.38 | 0.341 | 0.0 / 202.4 |
| 2 | t1/t1 | parent3 | 1 | 12 | 0.999 | 1.178 | 1.178 | 0.034 | 2.287 | 65.91 | 70.37 | 0.427 | 0.0 / 211.6 |

prefix check: 10 long texts, 72 newcomer texts, 0 mismatches -> PASS
