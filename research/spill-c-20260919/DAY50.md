# WP-C day 50 (2026-09-24): the MoE slot cache door, improvement I4: prefetch through the owner

`OWED.md` C1 step (b), the last item on day 40's list. With the fill, I1 and I2 a GPU miss is a host-resident lease and
a pinned DMA enqueued on the compute stream just before its consumer, so every copy still serializes with compute.
The legacy cache overlaps the next routed expert's copy with the current expert's kernels on the copy stream, but only
under the experimental `MEMRA_MOE_PREFETCH=1` (off in the naked default and in every cell so far), and under the
door `prefetch_source` returns `false` so that no detached prefetch bypasses the owner. Written before any I4 code;
tree at start: `49fb2c37f` (through I7).

## 1. Pre-registration

**The design.**

- (a) **The call site.** In the per-token expert loop (`hybrid_forward.rs`, the branch that calls
  `moe_prefetch_expert` for `sel[j + 1]`), the condition `moe_prefetch_enabled()` becomes `moe_prefetch_enabled() ||
  e.expert_bank_prefetch()`, where `expert_bank_prefetch` is an engine flag the door's installer sets. Without the
  door it is false, so the legacy program and every legacy default are unchanged.
- (b) **A prefetch through the owner.** Under the door `prefetch_source` no longer returns `false` unconditionally:
  for a block that is not resident, not pending and whose record is resident in the host tier (a new proxy query,
  `host_resident`; a host miss is left to the demand path, so a prefetch never reads storage on the owner thread),
  it takes a lease through the proxy exactly as a demand does, reserves a GPU slot that is not one of the current
  expert's (`keep`), and enqueues the H2D on the copy stream after an event that orders every earlier compute-stream
  reader of the slot (the legacy `stage_on_copy_stream`). The block goes into the pending map with its copy event and
  its lease; it is not in the table.
- (c) **Consumption.** `admit_banked` looks in the pending map before demanding: a pending block gets a compute-stream
  wait on its copy event, is published, and its lease moves to the in-flight queue under that event (retired when it
  completes). No block is consumed without the wait, so no consumer reads a slot before its copy.
- (d) **Bounds.** Prefetched and in-flight leases together never exceed I1's 32; a prefetch that would exceed it, a
  host-tier miss, or no evictable slot outside `keep` returns `false` and the demand path serves the block.
- (e) **Teardown.** The gate's retirement and the cache's `Drop` drain both streams before finishing pending
  prefetched leases; an unknown copy stream keeps them open.
- (f) Counts: the stage clock gains `prefetches` (leases taken by prefetch) and `prefetch_hits` (misses served from a
  pending prefetch).

**Correctness.** Same bytes, same kernels, the copy ordered before the consumer by an explicit wait, as the legacy
prefetch. A CPU census pins that `admit_banked` consumes a pending block only after `compute_wait`, and that the only
new `prefetch_source` branch takes its lease through the proxy.

**The cell `prefetch` (RTX 5090 first).** Day 49's shape and budget; arms OFF (the I4 binary, no door), I7 (the day-49
binary), I4 (the I4 binary), door arms with `--expert-bank-stages`; order 1 (OFF, I7, I4) x 5, order 2 reversed x 5,
one collector hold. Integrity as day 49's.

**Clauses.** (i) I4 beats I7 in the window by more than `noise` in both orders; (ii) at least half of I4's GPU misses
per window token are `prefetch_hits`. I4 stays if both hold.

**What each card can decide.** The RTX 5090 decides (i) and (ii) here; the target card reads them in the ladder.
Because the door's prefetch has a legacy counterpart that is off by default, the deciding cell (next) carries the
legacy with `MEMRA_MOE_PREFETCH=1` as a reference arm beside the naked-default baseline.

## 2. Results, cell `prefetch` (RTX 5090 Laptop GPU, `rtx5090-day50/prefetch/`)

One collector hold, 00:16:24Z to 00:22:24Z, 30 runs, tree `4653eba9f`, binaries `run-gen-i7` `3eab35aa...` and
`run-gen-i4` `f14df194...`, the approved artifact, the runner under the 1200% cap. Regime (`regime.log`, 250 ms,
N=1422): SM 180 to 2782 MHz, power 25.9 to 162.9 W, 60 to 73 C. Collector `--validate` rc=0.

Verbatim (`prefetch/reading.log`):

`DAY50 PREFETCH CHECKS rig=rtx5090 runs=30 integrity=ok`

`DAY50 ARM i7 window_door_ms_per_token=0.69 gen_door_ms_per_token=4.92 window_s median=0.354 iqr=0.001 | per window token: gpu_misses=92.3 host_hits=92.3 prefetches=0.0 prefetch_hits=0.0 demand=0.661 enqueue=0.179 copy_gpu=2.191 wait=0.000 retire=0.054 miss_total=1.226`

`DAY50 ARM i4 window_door_ms_per_token=-0.63 gen_door_ms_per_token=2.75 window_s median=0.312 iqr=0.002 | per window token: gpu_misses=92.3 host_hits=92.3 prefetches=88.6 prefetch_hits=88.6 demand=0.027 enqueue=0.008 copy_gpu=0.095 wait=0.000 retire=0.054 miss_total=0.051`

`DAY50 CLAUSE (i) i4_minus_i7 window o1=-0.041 o2=-0.041 noise=0.002 rule < -noise both orders -> PASS`

`DAY50 CLAUSE (ii) i4 prefetch_hits/gpu_misses per window token median=0.959 rule >=0.5 -> PASS`

`DAY50 PREFETCH rig=rtx5090 integrity=ok clause_i=PASS clause_ii=PASS`

I4 stays. 88.6 of 92.3 GPU misses per window token are served from a copy-stream prefetch, the miss path falls to
0.05 ms per token, and the door's window runs 0.63 ms per token faster than the legacy naked default in this cell
(`off` is the legacy with no prefetch; the deciding cell's REF arm is the legacy with its own prefetch). Gen-only
decode still pays 2.75 ms per token over the legacy, the fill race I10 addresses.
