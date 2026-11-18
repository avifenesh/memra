land27-four (01:51 take, main26 vs land27, both nvcc 13.4) recomputed 2026-11-18 by recompute.py.
Metric: ttft_s per prefill call, round 0 dropped (11 a run); decode-c1 step_s pooled, round 0 dropped.
All 8 runs reproduce steady.md exactly (0.9489 / 0.9439 / 0.9531 at 10/12/16 rows), so the method matches.
Disturbance: the landing build ran until 01:54:10, i.e. through run1-A (01:53:20-01:53:48) and the first 22 s of
run2-B (01:53:48-01:54:18). run1-A is visibly slow (+0.3 to +0.6% at every call); run2-B is in line with the other B runs.

| call | criterion | all 8 | drop run1-A (A 3 runs, B 4) | drop run1-A + run2-B (3 + 3) |
|---|---|---|---|---|
| 4 rows | 1.000 +/- 0.003 | 1.0021 | 1.0021 | 1.0022 |
| 8 rows | 1.000 +/- 0.003 | 1.0004 | 1.0008 | 1.0007 |
| 9 rows | 1.000 +/- 0.003 | 0.9994 | 0.9999 | 0.9997 |
| 10 rows | <= 0.953 | 0.9489 | 0.9495 | 0.9491 |
| 12 rows | <= 0.950 | 0.9439 | 0.9443 | 0.9443 |
| 16 rows | <= 0.959 | 0.9531 | 0.9537 | 0.9535 |
| decode c1 | 1.000 +/- 0.003 | 0.9976 | 0.9990 | 0.9989 |

Every criterion holds in both variants. Files: recompute-all.txt, recompute-drop1.txt, recompute-drop12.txt (per-run medians).
