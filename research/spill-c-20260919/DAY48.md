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
