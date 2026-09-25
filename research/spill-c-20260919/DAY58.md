# WP-C day 58 (2026-09-25): fixing I8 and I5 after their clauses failed, registered before the fix code

`OWED.md` C1 step (b). Day 48's cell (`DAY48.md` section 2) read `DAY48 SMALL rig=rtx5090 integrity=ok i8=FAIL
i5=FAIL`: I8's `validate` per window token fell 0.435 to 0.131 ms, where clause (i) asks for below a tenth (0.0435);
I5's `trace` fell 0.240 to 0.033 ms, where clause (iii) asks for below a tenth (0.024). Both no-regression clauses
held, and both rungs moved the window the right way. `DAY48.md` says each rung stays only if its two clauses hold,
and `DAY51.md` says a rung that fails its own clauses is fixed or reverted before the final tree is named. They are
fixed here, not reverted; the clauses are day 48's, unchanged. Tree at start: the lane tip after day 48's receipts.

## 0. What is left in each stage, from source

- `validate` brackets the memo check: two `Instant::now` calls, a `HashMap<BlockId, usize>` lookup with the default
  SipHash over a 5-byte key, 471 times per window token (every admission, hits included). The proxy call is gone
  for every known pair; what remains is the hash lookup and the clock.
- `trace` brackets the line's assembly: `writeln!` through the formatting machinery, plus a `format!` allocation for
  the victim field on every line, about 92 lines per window token; the stderr write is already one call per 64 KiB.

## 1. Pre-registration

**The fixes.**

- **I8f, a dense memo.** The `(id, bytes)` pairs the proxy accepted are held in a table indexed by layer, projection
  and expert (grown on the first insertion that needs it, `0` meaning not validated), not a hashed map. Same rule:
  a pair enters only after `bank.validate` accepted it; any pair not in the table with the same byte count reaches
  the proxy; nothing is removed while the cache lives. Both call sites (the demand and the door's prefetch) use it.
- **I5f, the line written directly.** The `[expert-host-slru]` line is assembled with direct pushes: the fixed text,
  the integers through a small decimal writer, the victim written in place (no `format!`, no allocation). Byte for
  byte the line day 48 printed. A CPU test compares the new writer with the old `writeln!` format over edge values and
  a randomized set of keys, sizes, slots, hits and victims.

**Correctness.** CPU tests: the dense memo against a `HashMap` oracle over a randomized sequence of validations
(same accept/skip decision every step), and the byte-identity test above; the day-48 census updated to pin the new
guard at both sites and the new writer as the only trace formatter.

**The cell `smallfix` (RTX 5090 first; the target card runs it in its final phase).** Day 48's shape and budget
(16 GiB, the stage clock on every door arm), four arms from one tree, the tip with both fixes:
- OFF: the tip, no door;
- NOMEMO: the tip with the memo check removed at both sites (`day58-nomemo.patch`: every admission calls
  `bank.validate`), I2's program for this stage;
- UNBUF: the tip with the trace printed by one unbuffered `eprintln!` per line (`day58-unbuf.patch`), I8's program
  for this stage;
- TIP: the tip.
The two patched arms are built from the same commit with the patch applied (the patches committed and checked to
apply and compile before the cell). Order 1 (OFF, NOMEMO, UNBUF, TIP) x 5, order 2 reversed x 5, one collector hold.
Integrity as day 48's.

**Clauses** (day 48's, arms renamed; `noise` the larger window IQR of the pair):
- (i) `validate(TIP) < 0.1 x validate(NOMEMO)` per window token; (ii) `median(TIP window) - median(NOMEMO window)
  <= noise` pooled and in both orders.
- (iii) `trace(TIP) < 0.1 x trace(UNBUF)` per window token; (iv) `median(TIP window) - median(UNBUF window) <= noise`
  pooled and in both orders.
I8 (as I8f) stays if (i) and (ii) hold, I5 (as I5f) if (iii) and (iv) hold; a failure is fixed again or the rung
reverted, before `DAY51.md` section 2 names the final tree.

**What each card can decide.** The RTX 5090 decides here; the target card reads the clauses in its final phase.

## 1a. Scripts, before the fixes' cells run

`day58-cell.sh` (cell `smallfix`), reader `day58-smallfix.py` (day 48's reader with the arms renamed and the same
clauses), the arm patches `day58-nomemo.patch` and `day58-unbuf.patch` (made from `7ea765687`, checked with
`git apply --check`). The fixes landed in `7ea765687`. The box build takes a label with a patch
(`day58-<label>.patch` applied to the commit, the tree restored after the build) and builds only the labels whose
binary is missing, so the final phase adds `tip`, `nomemo`, `unbuf` and `final` without rebuilding the rest; the
target card runs `smallfix` in its final phase after `residfix`.

## 2. Results, cell `smallfix` (RTX 5090 Laptop GPU, `rtx5090-day58/smallfix/`)

One collector hold, 01:16:34Z to 01:24:51Z, 40 runs, tree `97bed06d3`, binaries `run-gen-tip` `0fbf6328...` (the
crates of `7ea765687`), `run-gen-nomemo` `d411bcf1...` and `run-gen-unbuf` `4f2d8275...` (the same commit with each
patch), the approved artifact, the runner under the 1200% cap. Regime (`regime.log`, 250 ms, N=1948): SM 180 to 2610
MHz, power 11.9 to 166.8 W, 62 to 79 C. Collector `--validate` rc=0.

Verbatim (`smallfix/reading.log`):

`DAY58 SMALLFIX CHECKS rig=rtx5090 runs=40 integrity=ok`

`DAY58 ARM nomemo window_door_ms_per_token=-0.39 window_s median=0.320 iqr=0.004 | per window token: gpu_misses=92.3 host_hits=92.3 validate=0.430 trace=0.021 demand=0.032 enqueue=0.008 retire=0.055 finish=0.000 miss_total=0.058`

`DAY58 ARM unbuf window_door_ms_per_token=-0.56 window_s median=0.314 iqr=0.003 | per window token: gpu_misses=92.3 host_hits=92.3 validate=0.039 trace=0.255 demand=0.038 enqueue=0.008 retire=0.055 finish=0.000 miss_total=0.061`

`DAY58 ARM tip window_door_ms_per_token=-0.59 window_s median=0.313 iqr=0.003 | per window token: gpu_misses=92.3 host_hits=92.3 validate=0.038 trace=0.021 demand=0.032 enqueue=0.008 retire=0.055 finish=0.000 miss_total=0.056`

`DAY58 CLAUSE (i) validate per window token nomemo=0.430 tip=0.038 rule tip < 0.1 x nomemo -> PASS`

`DAY58 CLAUSE (ii) tip_minus_nomemo window pooled=-0.007 o1=-0.010 o2=-0.006 noise=0.004 rule <=noise -> PASS`

`DAY58 CLAUSE (iii) trace per window token unbuf=0.255 tip=0.021 rule tip < 0.1 x unbuf -> PASS`

`DAY58 CLAUSE (iv) tip_minus_unbuf window pooled=-0.001 o1=-0.003 o2=+0.000 noise=0.003 rule <=noise -> PASS`

`DAY58 SMALLFIX rig=rtx5090 integrity=ok i8f=PASS i5f=PASS`

Both rungs now hold day 48's clauses: the dense memo takes `validate` to 0.038 ms per token (a ninth of its bound,
11.3 times below the no-memo arm), the direct writer takes `trace` to 0.021 ms (12.1 times below the unbuffered
print). I8 and I5 stay, as I8f and I5f.
