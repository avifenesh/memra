# WP-C day 57 (2026-09-24): the MoE slot cache door, improvement I10: the fill completes inside the install

`OWED.md` C1 step (b). Found while preparing the deciding cell, from the day-50 dry check of the final binary
(`day50-cpu/dry-check-i4.log`, one run under the 5090 lock outside the collector, not a receipt): the door's decode
still races its own host fill. The fill starts at install and its eight threads are still reading when the argmax
gate and the first generated tokens run: between the `gate` and `generate` stage lines the door took 448 host misses
(`host_misses` 2,877 to 3,325), each a positioned read and a SHA-256 verify on the owner thread (`pread_ns` +29.2 ms,
`verify_ns` +52.9 ms, `alloc_ns` +10.6 ms, `demand_ns` +121.6 ms over the 32 generated tokens), while the window after
it took none (`demand_ns` +0.9 ms). The legacy program makes its 15 GB pinned host copy inside the load, before any
decode; the door makes the same size of copy (the fill into the pinned pool) after the install, overlapping decode.
Written before any I10 code; tree at start: `979537710`.

## 1. Pre-registration

**The design.**

- (a) **The installer waits for the fill.** After starting the fill and before registering the owner, the installer
  admits every finished fill on the installing (owner) thread with the same admission code as the demand path's
  (`drain_fill`'s body, factored into one function both call) until every fill worker has exited (the channel
  disconnects: jobs exhausted, the tier full, or a read error). A typed line after it: `[experts-via-tier] fill
  complete before decode in <ms> ms: <the fill counts line>`.
- (b) **Bounded.** If no fill completes for 10 s, the wait stops, prints `[experts-via-tier] fill wait stopped: no
  completion in 10 s (<counts>)`, and the rest of the fill continues as before (admitted at demands).
- (c) Unchanged: the fill's reads, checksums, admission rule, pool and buffers, the demand path (its `drain_fill`
  stays, and is empty after a completed fill), the gate's teardown.

**Correctness.** The same records are admitted by the same function against the same catalog digests; only when. A
CPU test drives the factored admission from both callers over one fixture and asserts the same outcomes and counts.

**The cell `fillwait` (RTX 5090 first).** Day 50's shape and budget; arms OFF (the I10 binary, no door), I4 (the
day-50 binary), I10 (the I10 binary), both door arms with `--expert-bank-stages`; order 1 (OFF, I4, I10) x 5, order 2
reversed x 5, one collector hold. Integrity as day 50's, plus on every I10 run the `fill complete before decode`
line with `fill_refused=0`.

**Clauses.** `noise` as before.
- (i) **No decode demand reads storage**: `physical_reads=0` on every I10 run.
- (ii) **I10 beats I4 in gen-only decode**: `median(I10 gen) < median(I4 gen) - noise` in both orders.
- (iii) **No window regression**: `median(I10 window) - median(I4 window) <= noise` pooled and both orders.
- A reading, direction registered: I10's `install_s` above I4's by about the fill's wall time (the copy moved into
  the install, as the legacy's is inside its load); reported beside the clauses, with the fill line's ms.
I10 stays if (i), (ii) and (iii) hold.

**What each card can decide.** The RTX 5090 decides the clauses here; the target card reads them in the ladder
(`DAY52.md` grows an `i10` arm and a day-57 view before the sitting runs). The deciding cell (`DAY51.md`) runs on
the tree after this rung's verdict, like the others.

## 2. Results, cell `fillwait` (RTX 5090 Laptop GPU, `rtx5090-day57/fillwait/`)

One collector hold, 00:22:26Z to 00:28:27Z, 30 runs, tree `2f86e3078`, binaries `run-gen-i4` `f14df194...` and
`run-gen-i10` `90c0496d...`, the approved artifact, the runner under the 1200% cap. Regime (`regime.log`, 250 ms,
N=1421): SM 1582 to 2775 MHz, power 28.5 to 167.1 W, 60 to 73 C. Collector `--validate` rc=0.

Verbatim (`fillwait/reading.log`):

`DAY57 FILLWAIT CHECKS rig=rtx5090 runs=30 integrity=ok`

`DAY57 ARM i4 window_door_ms_per_token=-0.56 gen_door_ms_per_token=2.69 window_s median=0.313 iqr=0.002 | per window token: gpu_misses=92.3 host_hits=92.3 prefetches=88.6 prefetch_hits=88.6 demand=0.027 enqueue=0.008 copy_gpu=0.095 wait=0.000 retire=0.054 miss_total=0.051`

`DAY57 ARM i10 window_door_ms_per_token=-0.59 gen_door_ms_per_token=-1.25 window_s median=0.312 iqr=0.001 | per window token: gpu_misses=92.3 host_hits=92.3 prefetches=88.6 prefetch_hits=88.6 demand=0.033 enqueue=0.008 copy_gpu=0.095 wait=0.000 retire=0.054 miss_total=0.056`

`DAY57 CLAUSE (i) i10 physical_reads per run=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] rule all 0 -> PASS`

`DAY57 CLAUSE (ii) i10_minus_i4 gen o1=-0.127 o2=-0.125 noise=0.008 rule < -noise both orders -> PASS`

`DAY57 CLAUSE (iii) i10_minus_i4 window pooled=-0.001 o1=+0.000 o2=-0.001 noise=0.002 rule <= noise pooled and both orders -> PASS`

`DAY57 READING install_s median i4=9.85 i10=10.86 (i10 - i4 +1.00) fill_complete_ms median=1038.3 gen_s median off=0.402 i4=0.487 i10=0.361`

`DAY57 FILLWAIT rig=rtx5090 integrity=ok clause_i=PASS clause_ii=PASS clause_iii=PASS`

I10 stays. No decode demand reads storage, gen-only decode falls 0.487 to 0.361 s (below this cell's legacy 0.402),
and the install carries the fill's 1.04 s instead (9.85 to 10.86 s), the direction registered.
