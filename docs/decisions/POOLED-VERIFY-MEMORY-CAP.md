# A pooled MTP verify forward fits the device; the largest member's round is funded before the cohorts step (2026-11-13)

Date: 2026-11-13. Issue memra#1266, branch `fix/1266-pooled-verify-oom`, receipts
`research/pooled-verify-memcap-20261113/`. Cards: one RTX 5090 Laptop GPU (24 GB; enforced power
limit 150 W, recorded per run and not changed) and one DGX Spark (GB10). Model: Qwen3.8-27B
NVFP4 + Q5_K with its masked 32,768-row MTP head.

## Problem

On the RTX 5090 Laptop the 29-turn agentic replay at c = 2 on the MTP route (ctx 32,768, 8
sessions, prefix cache 4,000 MB, no host tier, `MEMRA_PREFIX_STABLE_BOUNDARY=1`, thinking on,
temperature 0, seed 1234, 768 tokens) ended two requests with a CUDA out-of-memory in the
cohort's pooled verify, at the same point in every main boot: the warm admission of the
3,905-token prompt (k=19) beside the decoding k=21. Both sessions had streamed tokens, so
neither parked, and both requests failed while 2,226 MiB of unleased prefix entries and two
parked spec sessions stayed resident until the teardown.

A diagnostic arm (main plus logging only) named the failing allocation and the memory around it:

| step | driver free | pool cached | allocatable (sum) |
|---|---:|---:|---:|
| admission gate (charged cost 900 MiB + reserve 1,536 MiB = 2,436) | 515 MiB | 2,143 MiB | 2,658 MiB |
| after the driver-headroom rung (released 1,241 MiB of cached blocks) | 1,699 MiB | 516 MiB | 2,215 MiB |
| after the newcomer's 417-row suffix prime | 1,539 MiB | 376 MiB | 1,915 MiB |
| the cohort's first round, before its verify | 99 MiB | 1,517 MiB | 1,616 MiB |
| the failure | 67 MiB | 128 MiB | 195 MiB |

The failing allocation was 15,728,640 B, the second member's GDN ssm checkpoint slab (5 columns
of 3 MiB) at layer 61 of 64, in `pooled_linear_layer`. The forward pooled two K=5 rounds (12
rows) and had allocated 1,453 MiB of checkpoint slabs by then. Each member's verify keeps its
replay-free checkpoint, `rows - 1` columns of every linear layer's GDN state (156.9 MB a column
on this model), until its round finishes, so a pooled forward holds every member's checkpoint
at once: about 1.5 GB for K=5 + K=5.

Admission held its own arithmetic: after the newcomer's charge the allocatable reading stayed
above the 1,536 MiB transient floor. That floor funds one serial round. Nothing charged or
checked the extra members of a pooled forward. The DFlash2 cohort has such a check
(`pooled_verify_headroom`, below); the MTP lockstep (`generate_spec_sessions_unified`) had none.

Graph replay suspension was not the cause. On this dense model the MTP verify-graph pool is off
(`vgraph_family_default` needs linear layers and a routed MoE), so every cohort round is
poolable and runs the eager pooled verify whether or not replay is suspended; the suspension
(driver free 99 MiB while 1,517 MiB sat in the pool's cache) switched only the draft arms.

## Chosen

- **The memory cap.** The lockstep's verify chunk walk reads the allocatable memory once per
  forward (`pooled_verify_allocatable`: the driver's free bytes, or on unified memory the larger
  of them and `MemAvailable`, plus the async pool's cached blocks; an unreadable device reads 0,
  so nothing pools) and takes its members from a pure function, `pooled_verify_chunk`. A round
  after the first joins only while the forward's members (`pooled_verify_round_bytes`: each
  member's checkpoint columns plus two logits stacks), the FA partial pool's growth for the
  forward's rows and the verify margin fit the reading. The growth is priced launch by launch
  (`fa_rows_growth`, from the pool's current lengths): the layer launches once per run of
  consecutive rows on one split rung, members first and then riders, each launch sized by its own
  rows and deepest row, and a launch the pool cannot hold grows it to twice its size, keeping the
  old buffers. A pooled forward is wider than any member's serial verify, and one launch at the
  deepest row's rung would underprice it: across the 16,384 rung a c = 3 cohort (K=5 at 15,990,
  K=3 at 16,100, K=1 at 16,500) priced 0 there against 108.7 MiB of real growth, and pooled all
  three at a reading the forward exceeds. The first round always
  joins: alone it is the serial round. The walk stops at the first round that does not fit and
  tries no smaller later one; that round leads the next chunk, after this one's rounds have
  finished and freed their checkpoints. Plain riders join within the same reading
  (`pooled_rider_take`). A capped member verifies after the forward. Pooled and serial verifies
  are bitwise equal (`decode-batch-bench --pooled`, every shape the cap splits), and the
  forced-cap arm, which verifies every member alone, writes the same texts as the default arm in
  every cell, so the cap moves no token. A member held back pays the serial price: +14% for a
  K5+K5 round, up to +66% for four K2 rounds; in the agentic replay at c = 2, about 0.2 s of a
  4-minute run. `[spec-pool] pooled MTP verify capped by memory: ...`
  (the first 16, then every 256th); `/metrics` `spec_pool_capped_forwards_total` and
  `spec_pool_capped_rider_rows_total`.
- **The verify margin** (`POOLED_VERIFY_MARGIN`, 256 MiB) is its own constant: the part of the
  reading the pool could not deliver at the failure (195 MiB: 67 MiB driver free and 128 MiB of
  cached fragments could not serve a 15 MiB slab, after the pool had grown in one 64 MiB step),
  rounded up to whole 64 MiB steps. One measurement (N = 1, the diagnostic boot), so it carries
  no spread. A test pins it above the measurement.
- **The pre-cohort reclaim.** After the newcomers primed and before the cohorts form, the
  reading must fund the cohort's first verify (`pre_cohort_need`): the largest member's round
  (`spec_round_bytes`: its checkpoint columns, logits and the FA pool's growth for its rows,
  priced run by run as the pooled layer launches them, a safe overestimate for the serial
  verify),
  every member's burst-entry snapshot (`spec_burst_snapshot_bytes`: each burst takes a fresh copy
  of the session's GDN state at its start and holds it to its end, one column, 149.6 MiB here;
  the diagnostic boot's reading fell from 1,915 to 1,616 MiB between the cohort's entry and its
  first verify, two members' snapshots) and the margin. When it cannot, read again after a fence,
  the ladder (`spec_round_reclaim_ladder`, a pure function over injected readings and rungs)
  evicts unleased prefix entries first, oldest first, until the shortfall is freed and settled
  (`prefix_yield_to_reserve`, on unified memory too), then parked sessions, globally oldest
  first, one at a time; under on-demand KV planes each parked eviction reaps the graveyard
  (`vmm_reap_for`) before the next reading, or the evicted planes would not show and every parked
  session would go (a census pins that reap: it does work only under on-demand KV planes, the
  RTX PRO 6000 default, so no cell here exercises it). The reading leaves out the part of the
  admitted sessions' unbuilt booking (memra#1255's `unbuilt_session_booking`, single device
  only) that overlaps the cohort: those sessions build their state after the cohorts step, once
  the rounds' checkpoints and snapshots are freed, so only the FA pool's growth, which the pool
  keeps, is live beside them (`booking_overlap`). That growth is priced on the cohort's members
  stacked in one pooled forward (`spec_cohort_fa_growth_bytes`, the larger of the stack and any
  round alone): across a split rung a forward grows the pool where no single round does. The
  prefix rung covers the need on top of the bounded booking. An
  unreadable device does nothing; a pass the fence alone funds evicts nothing and counts in
  `spec_cohort_fence_funded_total`. `[spec-pool] pre-cohort reclaim: ...`;
  a pass that stays short with nothing evictable prints `[spec-pool] pre-cohort reclaim still
  short: ...` (the first 16, then every 256th); `/metrics` `spec_cohort_reclaims_total` and
  `spec_cohort_short_total`; the prefix evictions also count in `prefix_headroom_evictions`.
- **The band past admission's floor.** At K = 5 the need is 1,314 MiB at c = 2, 1,464 at c = 3,
  1,614 at c = 4 and 2,212 at c = 8 (the FA growth aside): from four members it passes
  admission's 1,536 MiB transient floor, so a cohort admitted to the floor reclaims at its first
  pass or stays short and counts it. The c = 4 cells read `spec_cohort_short_total`.
- **The need-pad door.** `MEMRA_SPEC_COHORT_NEED_PAD_MB=N`, a test door, adds N MiB to the need,
  so the ladder and its counters run on hardware where the real need is funded; the receipts
  compare its texts with main's.
- **The forced-cap arm.** `MEMRA_SPEC_POOL_SERIAL=1`, a test door, reads the allocatable memory
  as 0 on every forward, so every member verifies alone and no rider joins. The receipts compare
  it text for text with the default arm, which proves a capped request writes its uncapped text.
- **A mid-forward OOM still takes the honest error.** By then the forward has appended KV rows
  and advanced each member's GDN state in place for the layers it ran, so the round cannot be
  run again on that state, and a streamed session cannot replay. The fix keeps the forward from
  starting what it cannot finish.

## Rejected

- **DFlash2's rule for the MTP cohort** (`pooled_verify_headroom`: the extra members fit the
  reading less the 1,536 MiB transient floor, `sum - max <= free + cached - floor`). It keeps the
  floor unspent during the forward, but the forward's own allocations are all the cap prices:
  the newcomers primed before it and the prefix publications run after the tick's serving work,
  and the FA pool's growth inside the forward is part of the need. In this
  replay's two-member pooled rounds the allocatable reading was median 2,135, 10th percentile
  1,723 and minimum 1,294 MiB (578 rounds); at 1,723 MiB the DFlash2 rule would serialize a
  K=5 + K=2 round (extra about 320 MB against 187 MiB above the floor) that fits with about
  600 MiB to spare. It is also short of what the cohort holds besides its checkpoints (the burst
  snapshots and the FA pool's growth, above). The DFlash2 cohort keeps its rule; whether it should read the allocatable
  figure on unified memory and drop the floor is that route's call. (2026-11-14, memra#1307: it
  reads the allocatable figure, through this cap's `pooled_verify_reading`, and keeps the floor;
  [UNIFIED-MEMORY-READERS.md](UNIFIED-MEMORY-READERS.md).)
- **A per-session admission charge for the pooled transient.** The checkpoint of a pooled
  forward is a per-forward transient, not a per-session resident; charging each MTP session
  K columns (about 785 MB at K=5) at admission would cut c = 8 capacity on the 24 GB card to pay
  for an opportunistic speedup the cap already makes safe.
- **Evicting prefix entries so the cohort stays pooled.** It trades later cache hits for verify
  speed with no measurement behind the trade. The reclaim funds the serial round only.
- **Retrying the step after the OOM.** The state is already advanced (above).

## Measured

See `research/pooled-verify-memcap-20261113/README.md` for every table, binary and run.
