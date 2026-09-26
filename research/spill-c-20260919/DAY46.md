# WP-C day 46 (2026-09-24): the MoE slot cache door, improvement I1: leases retired on events, no drain

`OWED.md` C1 step (b). Day 40 read the banked branch's compute-stream drain at 2.07 ms and `sync2` at 0.02 ms per
window token on the RTX 5090, and the drain does more harm than its own time: while the owner thread waits for the
GPU to finish everything queued before a copy, the GPU then idles while the owner does the next miss's host work
(demand, read, verify), so host and device never overlap on the miss path; the legacy miss is an asynchronous DMA the
host runs ahead of. Written before any I1 code; tree at start: `ee9083e07` (I6, I9, the fill).

## 1. Pre-registration

**The design.**

- (a) **No drain.** On a GPU miss `admit_banked` still demands the lease, borrows its bytes on the owner thread and
  enqueues the H2D on the compute stream (`stage_expert`), then records one event on the compute stream after the
  copy and publishes the GPU slot. It does not synchronize the stream, before or after. The copy and every consumer
  of the slot are on the compute stream, so stream order is the consumer fence, as on the legacy path.
- (b) **Retire on the copy's event, in order.** The lease and its event go on an in-flight queue in the cache. Before
  every admission the cache retires, from the front, every lease whose event has completed (`is_complete`), calling
  `finish` for each; events on one stream complete in order, so the front is always the oldest. At most `N = 32`
  leases are in flight: when the queue is full, the cache waits on the front event (`synchronize` on that event,
  not on the stream) and retires it before demanding. The host bytes of an in-flight lease are never released
  before its event completes (the lease holds them; `finish` is the only release).
- (c) **The publication rule of `d0acf6f03`, kept.** A slot is published only after the copy's enqueue and its event
  record both succeeded. If either fails, the existing drain path runs: synchronize the stream; on success release
  the reserved slot and finish the lease; on failure mark the stream unknown and keep the slot outside every table
  (the lease is retained, never finished).
- (d) **Capacity.** The owner registry's pending bound, the bank's ticket limit and the governor's in-flight and
  staging dimensions rise from 1 to `N + 1`, so an in-flight lease never blocks the next demand's ticket.
- (e) **Teardown.** `Drop` synchronizes the stream once, then finishes every in-flight lease in order; if the
  synchronize fails every lease is retained (leaked with its bytes), as today's single-lease rule.
- (f) The stage clock keeps its brackets; `drain` and `sync2` read 0 when nothing drains, and it gains `retire_ns`
  (the front-of-queue retirement) and `wait_ns` (the full-queue wait on the front event).

**Correctness.** The same bytes reach the same kernels; only the host's waiting changes. The CPU census pins that
`admit_banked` holds no `stream().synchronize()` on its success path and that `finish` is reached only through the
in-flight queue's event check or the teardown drain. The cells' identity clauses (one tape, `MATCH`) stay.

**The cell `nodrain` (RTX 5090 first).** Day 40's shape; three arms, every door arm with `--expert-bank-stages` and
the full-bank budget (`--expert-bank-host-bytes=17179869184`): OFF (the I1 binary, no door); FILL (the day-45 binary);
I1 (the I1 binary). Order 1 (OFF, FILL, I1) x 5, order 2 reversed x 5, one collector hold. Integrity as day 45's.

**Clauses.** `noise` as before (the larger window IQR of the two arms compared).
- (i) **I1 beats FILL in the window**: `median(I1 window) < median(FILL window) - noise` in both orders.
- (ii) **Nothing drains**: I1's `drain` and `sync2` per window token below 0.05 ms, and its in-flight waits
  (`wait_ns`) below 1.0 ms per window token.
I1 stays if (i) and (ii) hold; if (ii) holds and (i) fails, the drain was not what the window paid, recorded.

**What each card can decide.** The RTX 5090 decides (i) and (ii) here; the target card reads them in the ladder.

## 2. Results, cell `nodrain` (RTX 5090 Laptop GPU, `rtx5090-day46/nodrain/`)

One collector hold, 23:37:31Z to 23:43:40Z, 30 runs, tree `1a8c7ff15`, binaries `run-gen-fill` `19bcfd59...` and
`run-gen-i1` `0d90e124...`, the approved artifact, the runner under the 1200% cap. Regime (`regime.log`, 250 ms,
N=1469): SM 180 to 2790 MHz, power 18.1 to 158.9 W, 60 to 73 C. Collector `--validate` rc=0.

Verbatim (`nodrain/reading.log`):

`DAY46 NODRAIN CHECKS rig=rtx5090 runs=30 integrity=ok`

`DAY46 ARM fill window_door_ms_per_token=5.03 gen_door_ms_per_token=13.16 window_s median=0.492 iqr=0.008 | per window token: gpu_misses=92.3 host_hits=92.3 demand=0.863 drain=2.067 sync2=0.018 wait=0.000 retire=0.055 finish=0.085 enqueue=3.175 miss_total=6.669`

`DAY46 ARM i1 window_door_ms_per_token=3.31 gen_door_ms_per_token=12.34 window_s median=0.437 iqr=0.009 | per window token: gpu_misses=92.3 host_hits=92.3 demand=0.892 drain=0.000 sync2=0.000 wait=0.000 retire=0.062 finish=0.002 enqueue=3.273 miss_total=4.579`

`DAY46 CLAUSE (i) i1_minus_fill window o1=-0.048 o2=-0.058 noise=0.009 rule < -noise both orders -> PASS`

`DAY46 CLAUSE (ii) i1 drain+sync2=0.000 wait=0.000 ms per window token rule <0.05 and <1.0 -> PASS`

`DAY46 NODRAIN rig=rtx5090 integrity=ok clause_i=PASS clause_ii=PASS`

I1 stays: no drain on the miss path, the window door cost 5.03 to 3.31 ms per token; the remaining miss cost is the
enqueue (3.27 ms per token, the pageable H2D, I2's).
