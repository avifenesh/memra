# WP-C day 48 (2026-09-24): the MoE slot cache door, improvements I8 (the validate memo) and I5 (the trace off the hot path)

`OWED.md` C1 step (b). Two small costs from day 40's reading, each its own rung and receipt in one cell. Written before
any code for either; tree at start: `f184e7b83` (I6, I9, the fill, I1, I2).

## 1. Pre-registration

**I8, the validate memo.** Day 40: `validate` 0.649 ms per window token over 471 admissions per token (GPU hits
included), about 1.4 us each: a proxy call through the thread-local registry, two `BTreeMap` lookups and a layout
fetch, to learn that the catalog's record for `(layer, proj, expert)` has the dispatched byte count. The catalog is
immutable for the door's life, so the answer for an `(id, bytes)` pair never changes.
- Design: the cache keeps a map of `BlockId -> bytes` it has validated through the proxy. An admission whose id is in
  the map with the same byte count skips the proxy call; any other admission (a new id, or a known id with a
  different byte count) calls `validate` as today, and only a successful `validate` enters the map. A refusal is
  never memoized. The map is dropped with the cache.
- Correctness: the check still fires for every `(id, bytes)` pair the proxy has not accepted; a lying caller (a
  changed byte count) reaches the proxy and is refused as before. A CPU census pins the memo's only insertion after
  a successful `bank.validate`.

**I5, the trace off the hot path.** Day 40: `trace` 0.265 ms per window token, one unbuffered `eprintln!` per host
demand (about 2.9 us each).
- Design: `TracedDispatch` appends each `[expert-host-slru]` line, byte for byte as today, to an in-memory buffer and
  writes the buffer to stderr in one call whenever it passes 64 KiB, cut at a line boundary, and at close; nothing
  else changes. The trace stays complete and in order; the line stamper's times for trace lines become the chunk
  times (no reader uses them; the verifiers count and parse the lines, which are unchanged).
- Correctness: the day-43 integrity clause (trace lines equal host demands, `physical_reads` equals `hit=false`
  lines) must hold on the new rung.

**The cell `small` (RTX 5090 first).** Day 47's shape and budget; four arms, every door arm with
`--expert-bank-stages`: OFF (the I5 binary, no door); I2 (the day-47 binary); I8 (I2 plus I8); I5 (I2 plus I8 plus
I5). Order 1 (OFF, I2, I8, I5) x 5, order 2 reversed x 5, one collector hold. Integrity as day 47's.

**Clauses.** `noise` as before.
- (i) I8's `validate` per window token below 0.1 of I2's; (ii) I8 does not regress: `median(I8 window) - median(I2
  window) <= noise` pooled and both orders.
- (iii) I5's `trace` per window token below 0.1 of I8's; (iv) I5 does not regress against I8 the same way.
- A reading, direction registered: the window door cost falls from I2 to I8 to I5.
Each rung stays if its two clauses hold.

**What each card can decide.** The RTX 5090 decides the four clauses here; the target card reads them in the ladder.

## 2. Results, cell `small` (RTX 5090 Laptop GPU, `rtx5090-day48/small/`)

One collector hold, 23:59:59Z to 00:09:53Z, 40 runs, tree `305ca1ca1`, binaries `run-gen-i2` `974a7e4b...`,
`run-gen-i8` `308dad3a...`, `run-gen-i5` `2b5a738e...`, the approved artifact, the runner under the 1200% cap. Regime
(`regime.log`, 250 ms, N=2311): SM 180 to 2782 MHz, power 10.1 to 158.1 W, 58 to 71 C. Collector `--validate` rc=0.

Verbatim (`small/reading.log`):

`DAY48 SMALL CHECKS rig=rtx5090 runs=40 integrity=ok`

`DAY48 ARM i2 window_door_ms_per_token=0.91 window_s median=0.360 iqr=0.002 | per window token: gpu_misses=92.3 host_hits=92.3 validate=0.435 trace=0.243 demand=0.800 enqueue=0.176 retire=0.054 finish=0.002 miss_total=1.360`

`DAY48 ARM i8 window_door_ms_per_token=0.70 window_s median=0.353 iqr=0.002 | per window token: gpu_misses=92.3 host_hits=92.3 validate=0.131 trace=0.240 demand=0.855 enqueue=0.178 retire=0.055 finish=0.002 miss_total=1.410`

`DAY48 ARM i5 window_door_ms_per_token=0.69 window_s median=0.353 iqr=0.002 | per window token: gpu_misses=92.3 host_hits=92.3 validate=0.134 trace=0.033 demand=0.656 enqueue=0.179 retire=0.053 finish=0.002 miss_total=1.215`

`DAY48 CLAUSE (i) validate per window token i2=0.435 i8=0.131 rule i8 < 0.1 x i2 -> FAIL`

`DAY48 CLAUSE (ii) i8_minus_i2 window pooled=-0.007 o1=-0.007 o2=-0.006 noise=0.002 rule <=noise -> PASS`

`DAY48 CLAUSE (iii) trace per window token i8=0.240 i5=0.033 rule i5 < 0.1 x i8 -> FAIL`

`DAY48 CLAUSE (iv) i5_minus_i8 window pooled=-0.001 o1=+0.000 o2=-0.001 noise=0.002 rule <=noise -> PASS`

`DAY48 READING window i2=0.360 i8=0.353 i5=0.353 -> falls`

`DAY48 SMALL rig=rtx5090 integrity=ok i8=FAIL i5=FAIL`

Both rungs fail their stage clause and hold their no-regression clause: I8 took the validate stage down 3.3 times and
the window by 7 ms per 32 tokens, I5 took the trace down 7.3 times; neither reached the registered tenth. The rule
stands; both are fixed before the final tree is named (`DAY58.md`, registered before the fix code).
