# OWED 26 G3: recorded rows that carry ring-busy mmap fallbacks

Source: `census.py` over every committed file under `research/` with a `[spill-pread]` totals
line (`census.json`, `census.txt`). 261 directories carry a totals line; 67 show fallbacks. Every
fallback reason quoted in these logs reads `worker read ring is busy` (the engine quotes the first
three per process). Nothing below is rescored or rewritten; this is where the mixed worker-plus-mmap
program sits in the record.

| Rows | Directories | Fallbacks | Max per visit |
|---|---|---|---|
| F, BOX27 B3 smoke, cold, cold window 2, warm, bounded: `worker2`, every visit | 41 | 32,410 | 1,615 |
| F, 5090 capped: `worker2`, every visit | 10 | 5,939 | 746 |
| F, 5090 capped: `worker16` | 3 | 16 | 6 |
| F, 5090 capped: `direct16` (refused by the registered gate) | 2 | 6 | 4 |
| F, 5090 capped smoke and anon-peak sizing (never scored) | 4 | 1,394 | 510 |
| F, OWED 17 smoke and door-off diagnostic (never scored) | 4 | 15,625 | 8,287 |
| per-expert-quant `local-5090-sota-20260719` | 1 | 43 | 43 |
| per-expert-quant `local-5090-next3-20260722` | 1 | 1 | 1 |
| per-expert-quant `local-5090-plain-arm-20260725` | 1 | 79 | 10 |

Covered above: 67 of 67 directories.

What it changes in the record:

- BOX27 `worker2` rows (all regimes) measure the mixed program; their correction note in
  `../box27/RESULTS.md` stays as written. No BOX27 verdict names `worker2` a winner, and the
  scored bounded verdict (worker16 over every challenger) is unaffected: worker16, pread16 and
  direct16 had zero fallbacks on BOX27.
- 5090 capped is unscored either way; its descriptive `worker2` and two `worker16` rows are partly
  mmap (`../rtx5090/RESULTS.md`).
- Three per-expert-quant evidence directories from 2026-07-19 to 2026-07-25 (the local 5090,
  worker path) show the same mechanism in some of their runs (1 of 100, 1 of 47 and 20 of 26 totals
  lines). Which of their rows back the depth-2 local-safe default in `per-expert-quant/README.md`
  is not established here; they are flagged for the owning lane, not relabelled.

