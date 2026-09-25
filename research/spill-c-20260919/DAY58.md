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
