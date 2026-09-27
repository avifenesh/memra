# WP-B day 28: memra#476, the graph growth term (arithmetic, cell, verdict) and the park door's decide-by

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, merged with `origin/main` `3df055601` (#626, day 26 on
main) at `88552c5ef`, clean (`tools/check-conflict-markers.sh` OK). Every push today whose range touches engine source
is refused `UNQUALIFIED` by the #589 hook on a plain push and goes out as `MEMRA_RELEASE_QUALIFICATION_MODE=development
git push` (`UNQUALIFIED DEVELOPMENT ... no GPU qualification claimed`, logged in `.git/memra-gate-skips.log`). **No
qualification is claimed anywhere in this record**; every GPU cell is `executed-not-qualified`; no timing is compared
across cards; no default changes today.

Rig discipline as on every prior day: every GPU command on the local RTX 5090 Laptop GPU runs under `flock -w` on
`/tmp/memra-5090.lock`, CPU-heavy work under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`,
`nvidia-smi --query-compute-apps` read before and after every cell, no process this lane did not start is touched
(identification by cwd only).

## 1. memra#476, the graph growth term: what a new shape allocates, where, and how it is booked (written before any run)

Day 24 booked `W` (prefill workspace) and `D` (per-session draft-graph state) into the predictive book and left one
term named and unmeasured: "graph growth of a NEW decode shape ... is in neither book". This section is the census of
that term on tree `88552c5ef`, as arithmetic, before the cell runs. Anchors are `file:line` on that tree.

### 1.1 What "graph capture" reaches the served path on these models

- **Draft-chain graphs (spec sessions).** Captured at a session's first burst through `capture_graph_retained`
  (`crates/memra-engine/src/lib.rs:32775`, warmup allocations kept alive in a keeper), keyed by `SampledGraphKey`
  (`spec.rs:1988`: seed, temp, k, top_k, top_p, min_p, penalties) and by the mask shape (`spec.rs:10333`). A new key on
  the same session drops the parked graph and recaptures (`spec.rs:10674-10690`). Its device state is measured per
  session as `observed = max(parked delta, capture-time pool peak)` across the capture bracket (`spec.rs:10355-10385`
  open, `:11010-11045` close) and recorded as a model high-water, `record_draft_state_bytes` (`hybrid.rs:4076`); the
  admission charge `D = draft_session_admission_bytes()` (`hybrid.rs:4089`) is that high-water, charged per
  spec-capable admission (`worker.rs:18385-18405`) and, since day 24, read into the predictive book by
  `RequestCharge::from_physical_cost` (`admit_predict.rs:564`). Before the first observation the capture gate uses
  `draft_capture_bootstrap_estimate` (`spec.rs:569`), the boot probe normally supplies the first observation.
- **Verify-graph pool (`[spec-vg]`).** The one graph pool that GROWS with new keys: `DsparkVgraphs::graphs` keyed
  `(segment start, vt)` (`spec.rs:3086-3190`), backed by the driver's per-device graph memory pool
  (`device_graph_mem_reserved`, `lib.rs:3837`, `cuDeviceGetGraphMemAttribute RESERVED_MEM_CURRENT`; never released).
  Charged on the physical side as `vg_debt` added to `reserve` (`dspark_vg_admission_debt` `dflash.rs:5635`,
  `dspark_vg_debt_projection` `spec.rs:485`, `worker.rs:18611-18634`), and NOT on the predictive side (the shadow budget
  subtracts the calibrated `floor` once, `vg_debt` is not in it). Default: `spec_verify_graph_env().unwrap_or_else(||
  vgraph_family_default())` (`spec.rs:10010`) = linear layers AND routed MoE. **Qwen3.5-9B and Qwen3.8-27B are dense**:
  the pool never engages on either card (0 `[spec-vg]` lines in every day-24 and day-27 server log). Named gap for the
  GatedDeltaNet + MoE family, not measurable on this lane's models.
- **Step-wise decode graphs (`GraphSession`, `decode.rs:28`, bucket-keyed `(fa_vec, n_splits)` recaptures).** Not
  referenced by `crates/memra-server/src/worker.rs` (no `graph_session_new` / `graph_session_from_cache` call): the
  served solo plain path is not the graph-session path. Not a serving term.
- **The batched plain trunk** (`decode_step_batch_sampled_lean_masked`, `decode_batch.rs:1055`, called at
  `worker.rs:20934/20943`, width cap 8 `decode_batch.rs:602`): no capture (`decode_batch.rs` has no `begin_capture`).
  Its transients are `e.uninit(b_n * ...)` per step (`:1473`, `:1516`, `:3165-3502`), freed each step into the async
  pool. Per row on the 9B (n_embd 4096, n_ff 12288, 16 heads x 256, GDN inner 4096): about 10 x n_embd + 3 x n_ff +
  attention q/k/v/o + GDN q/k/v/o/beta/g, roughly (40,960 + 36,864 + 16,384 + 24,576) x 4 B = 475 KB per row, 3.8 MB at
  B = 8. Pool-cached between steps; inside the floor by two orders of magnitude.

### 1.2 The one shape-keyed, process-lifetime, never-freed allocation: the FA partial pool

`Engine::fa_part_pool` (`lib.rs:1863`) holds three f32 buffers `(o, m, l)` read and written by every FA decode and
verify launch in the process. It is **grow-only, retire-on-grow, never freed** (`fa_part_pool_grow` `lib.rs:32331`:
captured graphs bake its addresses, #68 root cause, so the old buffer is pushed to `fa_part_retired` `lib.rs:1869` and
kept for the process lifetime), and each grow allocates at least double (`o_len.max(2 * co)`, `ml_len.max(2 * cm)`).
The demand per launch (`lib.rs:30320-30345`, mirrored at every FA site):

```text
sp        = fa_split_keys(t_kv, n_head_kv)                       (lib.rs:1589; on this 82-SM rig with n_head_kv = 4 the
                                                                  day-24 ladder reads sp = 64: 65 splits at t_kv 4160)
n_splits  = ceil(t_kv / sp)
o_demand  = n_head x n_splits x head_dim        (f32)             eager per-row FA (the path this rig takes)
ml_demand = n_head x n_splits                   (f32, x2: m and l)
rows path (fa_decode_dcw_rows, lib.rs:32434, requires fa_sm_count() >= 128, the target card class only):
o_demand  = t x n_head x max_ns x head_dim;  ml_demand = t x n_head x max_ns        (t = batch rows)
grow bytes  g_i = 4 x (o_i + 2 x m_i)   with  o_i = max(o_demand, 2 x o_{i-1}),  m_i = max(ml_demand, 2 x m_{i-1})
G_fa(t)     = sum of g_i over every `[fa-pool] grow` line stamped at or before t      (retired buffers are kept)
ladder bound: G_fa(final) <= 4 x (2 x o_final + 4 x m_final)      (each grow at least doubles: geometric sum)
```

Booking today: **neither book carries it.** The physical gate charges `cost(r) = ctx(C) + W + A + D` per request and
`reserve = floor` once; the predictive book charges `ctx(P + L + 8) + A + W + D` per request against a budget that
subtracts `floor` once. `G_fa` is a device-level term (shared by every session, grown by whichever request first
presents the shape), so it is covered only insofar as the boot-calibrated `floor` (`run_boot_calibration`
`worker.rs:29703`: one B = 1 spec-shaped 4096-row probe, `RESERVED` high-water minus rest minus the charged classes,
`calibration_transient_floor` `worker.rs:29677`, floored at `SPEC_SHRINK_RESERVE` = 1,536 MiB) exceeds what later
shapes add. The probe itself grows the pool three times (day-24 log: grows #0 to #2 during the probe, #3 and #4 at the
13.5k request).

**Arithmetic on the two cards, from the ladders already on tape (verbatim `[fa-pool] grow` lines):**

```text
local 9B  (rtx5090-day24/after/server.log, n_head 16, head_dim 256, n_head_kv 4, 33 layers, 8 full-attention):
  o_len 0 -> 266240 -> 532480 -> 1064960 -> 2129920 -> 4259840 ;  ml_len 0 -> 1040 -> 2080 -> 4160 -> 8320 -> 16640
  G_fa = 4 x (266240 + 532480 + 1064960 + 2129920 + 4259840) + 8 x (1040 + 2080 + 4160 + 8320 + 16640)
       = 33,013,760 + 257,920 = 33,271,680 B  (33.3 MB)
  ladder bound 4 x (2 x 4259840 + 4 x 16640) = 34,344,960 B  (holds)
  o_demand at the served MEMRA_CTX = 65,536: 16 x ceil(65536 / 64) x 256 = 4,194,304 f32 <= 4,259,840 already held,
  so on the eager path NO further grow is possible at any t_kv this shape can reach; at 262,144: 16,777,216 f32
  (67.1 MB) -> two more doublings, G_fa would end near 4 x (8519680 + 17039360) + ml = 102 MB.
target 27B (pro-single-day27/cells/after-ab/server.log):
  o_len 0 -> 399360 -> 798720 -> 1597440 -> 12582912 ;  ml_len 0 -> 1560 -> 3120 -> 6240 -> 49152
  G_fa = 4 x (399360 + 798720 + 1597440 + 12582912) + 8 x (1560 + 3120 + 6240 + 49152)
       = 61,513,728 + 480,576 = 61,994,304 B  (62.0 MB)
floor on both cards: 1,536 MiB = 1,610,612,736 B.  G_fa / floor = 2.1 % (local), 3.8 % (target).
```

So the never-freed shape growth on these two models is two orders of magnitude inside the floor that both books already
subtract once. What day 24's assertion B caught at `inflight 0` (2,550 MB device delta against a 0 book) is not this
term: it is the async pool's CACHED free blocks (`RELEASE_THRESHOLD` pinned to `u64::MAX` at `Engine::new`,
`lib.rs:3699-3711`, `pool_cached_bytes` `lib.rs:3823`), which `effective_free_bytes` (`worker.rs:22739` = driver free +
pool cached) already hands back to the admission gate; `nvidia-smi memory.used` cannot see the difference, the gate can.

### 1.3 The per-session term that a new shape can move: `D` after the probe

`D` is a model high-water. When a session captures a shape larger than every earlier capture (a wider `k`, the filtered
in-graph sampler, the draft mask node), `record_draft_state_bytes` raises the high-water by `delta_D` and every FUTURE
spec admission is charged the new value in both books; the session that raised it was booked the OLD value. That is the
one per-request under-booking a new capture shape can produce, and it is bounded by the flip itself (`[spec]
draft-session state high-water:` prints old and new). On this rig the probe observed 41 MB and serving charged 44 MB on
every day-24 admission; the walk below records every flip.

### 1.4 Pre-registered cell: the shape walk (local RTX 5090 Laptop GPU, 9B, `MEMRA_CTX=65536`, shadow mode)

Server: the merged tree's `target/release/memra-server`, `MEMRA_COMPAT=openai`, one model `q9`
(`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`), `MEMRA_CTX=65536` (the day-26/27 local shape; the served-context budget of day 27
does not change the walk), `MEMRA_ADMIT_PREDICT_SHADOW=1` (log only, nothing refused, so before and after would run the
identical sequence), `MEMRA_TTFT_TRACE=1`. Client (`day28-client.py`), fixed sequence after `/readyz` 200: one 64-token
warm request; idle 2 s (baseline); **S1** one request of about 2k prompt tokens alone (B = 1, spec, K = 3: the first
draft-chain capture of a serving session); **S2** two concurrent 2k requests (both spec at `LOW=2`); **S4** four
concurrent (projected active 4 > LOW: plain, the batched trunk at B = 4); **S8** eight concurrent (B = 8, the trunk's
cap); **L** one request of about 23k prompt tokens alone (a t_kv the walk has not presented: 360 splits against the
probe's 65 and S1's about 35, plus a longer prime chunk sequence); **S8b** eight concurrent 2k again (control: every
shape already presented, no growth expected). `max_tokens` 96 everywhere, temperature 0. Sampler every 250 ms:
`nvidia-smi memory.used` and `/metrics` `admission_booked_bytes` (sum), `cuda_driver_free_bytes`,
`cuda_pool_reserved_bytes`, `cuda_pool_used_bytes`, `cuda_pool_cached_bytes` (published every 32nd tick or at any
retire, `worker.rs:21820-21830`, so it is a coarse second column; the books are reconstructed from the stamped admit
and `[ttft]` lines as on day 24). Server log parsed for every `[fa-pool] grow`, `[spec] draft-session state
high-water`, `[admission] request cost`, `[admission] per-session draft-state charge`, `[admit-predict]`, `[ttft]`,
`[admit-cal]`, `[spec-vg]`, `Overloaded`, `out of memory` line.

Rows (`day28-parse.py`, verbatim): every grow line with `g_i` and `G_fa` after it; every high-water flip with
`delta_D`; per request `tag P L C path kv_hat booked_real cost_exact residual` (day 24's A); per 250 ms sample `t used
delta real_book shadow_book under_real under_shadow inflight growth(t)` where `growth(t) = G_fa(t) - G_fa(ready) +
sum of delta_D flips after ready` (the growth attributable to shapes the WALK presented, the probe's grows excluded
because the floor was measured after them); the device step across each grow (the sample right after minus the sample
right before, MiB grain).

Pre-registered assertions, fixed here before the run, not fitted:

- **G1 (the headline): `growth(end) <= 10 % of floor = 161,061,273 B`.** Reason: section 1.2 predicts at most ONE more
  grow on this shape (the pool already holds 4,259,840 f32 after the day-24 ladder; a fresh process re-walks the
  ladder, grows #0 to #2 in the probe, and the walk adds the grows the 23k prompt demands: `ceil(23000 / 64) = 360`
  splits x 4096 = 1,474,560 f32 -> covered by the 2,129,920 rung, so grows #3 (and possibly #4 by the doubling rule)
  of about 8.5 MB and 17 MB) plus any `delta_D` flip. If `growth(end)` exceeds the 10 % line, a booking is warranted
  and the fix lands today (the predictor reads `G_fa` demand from the same `n_head x n_splits x head_dim` formula the
  pool uses); if it does not, the floor covers it and section 3 proposes the booking point instead of forcing a
  per-request charge on a shared device term.
- **G2: while `inflight > 0`, `max_underbook_real <= floor`** (day 24's labelled in-flight row, recorded before and, if
  a fix lands, after). Reason: `reserve` is the one physical term not booked per request and the floor is its bound.
- **G3: on every admit, `booked_real - kv_hat == ctx(C) - ctx(P + L + 8)`** (tolerance 0 where the cost is exact, the
  1 MB line grain otherwise) on the B = 2, 4, 8 shapes as well as B = 1: day 24 proved it on B = 1 and B = 4 only.
- **G4: zero `Overloaded`, zero OOM lines.**

If the cell does not run (lock never free within the bounded wait, model absent), the arithmetic above stands and the
stop is recorded.

## 2. The `MEMRA_KV_PARK_COMPACT` door: its `decide-by` and the cell that decides it

The door-hygiene rule (`CLAUDE.md`, 2026-09-05) puts a `decide-by:` date on every default-OFF door; the row had none
(recorded on day 27 and on #539). Set today in `docs/FLAGS.md`: **decide-by 2026-10-06**, 14 days after the door's first
serving receipt on the target card class (`DAY27.md` 2.6). The date is the lane's, the verdict is the owner's; the
default does not change. The decision input, from the day-27 receipts, is written in full in
`KV-RESIDENCY-DESIGN.md` (day-28 addendum) and in the row; in brief:

- **What the receipts say.** Spec path (the served path, both cards): the door writes nothing (`park-compact lines: 0`),
  retained bytes identical across arms (39,090,913,280 B target, 7,449,083,904 B local), 0 hits in both pools, digests
  equal 45 of 45. Plain path (`MEMRA_SERVE_SPEC=0`, target card, labelled extra): the door engages on every retirement
  (46 lines, `2071 of 262144 rows retained in 2.1ms` up to `6459 of 262144`), retention 39,191,576,576 ->
  22,649,241,600 B, pools still 0 hits, digests equal 45 of 45 across arms and across paths.
- **Promotion would change** the plain pool's retained bytes only (a parked entry from `cache.max_ctx x bpt`,
  7,784,628,224 B on the 27B at the served context, to `fed x bpt`, about 61 MB at 2071 rows) and make a plain-pool
  resume a D2D copy into a request-cap cache instead of an in-place adoption. Spec-path deployments see nothing.
- **Deletion would lose** the only mechanism that bounds a parked plain cache below the served context, and nothing
  measured: 0 continuation-pool hits on every tape this lane holds (days 14, 20, 26, 27); pressure reclaim and LRU on a
  later park would be the whole policy.
- **The spec-pool equivalent** is a different mechanism: a spec session parks live engine state whose captured
  draft-chain graphs bake the cache's plane addresses, so compaction there means copying the trunk rows AND dropping the
  graphs for a recapture on resume (one `D`-class capture, 41 to 44 MB on the 9B); the address-preserving policies are a
  TTL on `parked_at` or a per-pool byte cap. That is the owner's product switch on #539.
- **The deciding cell** (pre-registered in the design note, pointed at by the row): plain path, both cards, default vs
  `=1`, both orders, N>=5 per arm per length, one binary: (i) the compacted-park resume byte-identity gate on both
  resume shapes (exact-extension continuation resuming a compacted entry, against a plain-parked resume and a cold
  prime; digests equal, `plain-affinity` hit lines present, which needs the continuation sent as the parked session's
  committed sequence plus the suffix, the `DAY20.md` shape, not a re-rendered chat), (ii) the step-OOM adjacency replay
  (`retire_may_park(_, true)` refuses the park, no line, no entry), (iii) the park-time copy cost per park on both
  cards. About 1 agent-day on the day-26/27 harness.

## 3. The shape-walk cell: both cards, one binary (`3c9bdfeaa`), `executed-not-qualified`

Two attempts did not produce data and are kept as labelled failures. Local attempt 1
(`rtx5090-day28/before-attempt1-oom-cotenant/`): the lock was acquired at once but the card was already at 23,806 of
24,463 MiB, two processes of another session held 20,778 + 1,604 MiB outside the canonical lock (`compute-apps-before.csv`,
identified by cwd only, never signalled); the server died at boot, quoted: `[server] FATAL: worker init failed:
Engine::new failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`; the cell was re-queued behind
`wait-then-run.sh` (memory.used below 2,500 MiB on two consecutive 30 s polls, 90 min bound; the first waiter, this
lane's own process, was stopped by pid and restarted at 2,500 MiB when a 1.6 GB `kernel-check` of the other session was
still on the card; `before.wait.log`). Target-card attempt 1 (`pro-single-day28/cell-before-attempt1-env-run/`,
`cells/before-attempt1-env-run/`): a harness bug, `env ... run "$BIN"` cannot call the shell function `run`
(`env: 'run': No such file or directory`), fixed in `3c9bdfeaa`; no GPU data. The runs below are on the fixed runner.

### 3.1 Target card: one RTX PRO 6000 Blackwell, 27B, `MEMRA_CTX` unset (`pro-single-day28/cells/before/`)

Collector cell (`cell-before/command.capture.json`: `status executed-not-qualified`, `qualification false`,
`exit_code 0`), tree `3c9bdfeaa`, binary sha `59d74cd2...`, 25 requests, `non-200=0`, `compute-apps` empty before and
after, regime `32 C, 32.42 W` before / `45 C, 55.07 W` after, `power.limit 600.00 W`, server up 03:37:44Z to
03:38:40Z, 191 samples at 250 ms. Boot, verbatim: `[admit-cal] boot calibration done: model="q38" route=mtp transient
floor 2194MB (static was 1536MB; measured 2194MB; probe kv charge 127MB, draft-state 46MB, drafted 63 accepted 42;
[dev0 peak-mapped 18976MB rest-mapped 16608MB charged 173MB -> 2194MB]; 1.7s)`, so `floor = 2,300,575,744 B`;
`[admit-predict] shadow armed: budget_bytes=65881157328 budget_src=derived(effective_free_bytes=84065415872 -
prefix_cache_budget_bytes=15883042816 - admission_reserve_bytes=2301215728) ... enforce=false`. Paths as predicted:
`S1-0=spec S2-0=spec S2-1=spec`, every S4, S8, S8b request `plain` (the batched trunk), `L-0=spec`.

The growth term, verbatim (`REPORT.txt`):

```text
1790048269787 [fa-pool] grow #0 dev=0 o_len 0 -> 399360 ml_len 0 -> 1560 (retired kept, zero=false)  g_i=1609920 G_fa=1609920
1790048269802 [fa-pool] grow #1 dev=0 o_len 399360 -> 798720 ml_len 1560 -> 3120 (retired kept, zero=false)  g_i=3219840 G_fa=4829760
1790048269847 [fa-pool] grow #2 dev=0 o_len 798720 -> 1597440 ml_len 3120 -> 6240 (retired kept, zero=false)  g_i=6439680 G_fa=11269440
1790048294509 [fa-pool] grow #3 dev=0 o_len 1597440 -> 3194880 ml_len 6240 -> 12480 (retired kept, zero=false)  g_i=12879360 G_fa=24148800
1790048305305 [fa-pool] grow #4 dev=0 o_len 3194880 -> 6389760 ml_len 12480 -> 24960 (retired kept, zero=false)  g_i=25758720 G_fa=49907520
1790048269804 [spec] draft-session state high-water: 46MB (max of parked delta 46MB and capture-time pool peak 12MB; charged per spec admission and gating future captures)
[spec-vg] lines: 0
G_fa at ready (the probe's grows, inside the floor's measurement) = 11269440; grows after ready = 2
grow at 1790048294509: used 26805 -> 26805 MiB; g_i=12879360 (12.3 MiB); inflight=8
grow at 1790048305305: used 26805 -> 26805 MiB; g_i=25758720 (24.6 MiB); inflight=1
G3: exact rows max |residual| = 0 (tolerance 0); line~ rows max |residual| = 804952 (tolerance 2e6 at the 1 MB grain)
G1: growth(end) = 38638080 (G_fa after ready 38638080 + delta_D 0) <= 10% floor = 230057574: PASS
G2: while inflight > 0, max_underbook_real = 7443968680 <= floor = 2300575744: FAIL
G4: Overloaded/OOM lines = 0: PASS
```

Grows #0 to #2 are the probe's (before ready, inside the floor's measurement). Grow #3 fired at S8 with 8 in flight
(the batched trunk's FA at `t = 8` rows, the B-scaled demand of section 1.2), grow #4 at L with 1 in flight (the 22,600-
token t_kv). The device did not step at either grow at the 250 ms grain (`26805 -> 26805 MiB`): the allocations landed
in the pool's already-reserved blocks. `delta_D = 0`: the probe's 46 MB high-water held through every serving capture.
G3 holds on every shape: 17 exact rows read `residual = 0` (for example `S8-0 P=2136 L=96 C=2296 path=plain
kv_hat=1402714900 cost_exact=1404377876 bracket=1662976`, and `1404377876 - 1402714900 = 1662976 = 29696 x 56`), the 8
MB-rounded rows sit inside the line's grain; at the eighth concurrent admit `booked_bytes=10856373644
booked_real=10868014476`, a difference of `11,640,832 = 7 x 1,662,976`, the seven brackets and nothing else.

G2 reads FAIL, and the reading is the day-24 one, not a graph term. The maximum in-flight under-booking (7,443,968,680 B
at the `S8b-0` admit, one request booked) is the device delta the walk left behind between bursts: pool reserved rose
18,589,155,328 -> 30,970,740,736 B over the walk, of which 4,460,933,712 B was cached free blocks at that sample
(`RELEASE_THRESHOLD = u64::MAX`, handed back to the gate by `effective_free_bytes`), the rest the intentional retention
(25 prefix-cache entries of the walk's prompts under the 15.9 GB budget, the last parked spec sessions, `L`'s at 22,760 x
31,552 B = 718 MB). Subtracting `growth(t)` moves the number by 38.6 MB (`7405330600`). The graph growth's whole share
of the in-flight under-booking is 0.5 %.

### 3.2 Local RTX 5090 Laptop GPU, 9B, `MEMRA_CTX=65536` (`rtx5090-day28/before/`)

Lock acquired 03:38:08Z (`lock.txt`), `before.exit` 0, 25 requests, `non-200=0`, one foreign long-lived tenant on the
card throughout (a 1,390 MiB process of another project, present in every `compute-apps` snapshot of the day, not this
lane's, not touched); regime `68 C, 11.05 W` before / `75 C, 24.28 W` after (`power.limit [N/A]`), server up 03:38:18Z to
03:39:05Z, 164 samples. Boot: `transient floor 1536MB (static was 1536MB; measured 1266MB; probe kv charge 67MB,
draft-state 41MB ...)`, `floor = 1,610,612,736 B`; prefix budget `2052MB ... at served ctx 65536 (from MEMRA_CTX)`.
Paths: `S1-0=spec S2-0=spec S2-1=spec`, S4 all `plain`; in the eight-bursts the first arrivals were admitted with at most
two projected active and took the spec path (`S8-0=spec`, `S8b-1=spec`), the other seven of each burst `plain`;
`L-0=spec`.

```text
1790048301945 [fa-pool] grow #0 dev=0 o_len 0 -> 266240 ml_len 0 -> 1040 (retired kept, zero=false)  g_i=1073280 G_fa=1073280
1790048301956 [fa-pool] grow #1 dev=0 o_len 266240 -> 532480 ml_len 1040 -> 2080 (retired kept, zero=false)  g_i=2146560 G_fa=3219840
1790048302001 [fa-pool] grow #2 dev=0 o_len 532480 -> 1064960 ml_len 2080 -> 4160 (retired kept, zero=false)  g_i=4293120 G_fa=7512960
1790048323315 [fa-pool] grow #3 dev=0 o_len 1064960 -> 2129920 ml_len 4160 -> 8320 (retired kept, zero=false)  g_i=8586240 G_fa=16099200
1790048332795 [fa-pool] grow #4 dev=0 o_len 2129920 -> 4259840 ml_len 8320 -> 16640 (retired kept, zero=false)  g_i=17172480 G_fa=33271680
1790048301960 [spec] draft-session state high-water: 41MB (max of parked delta 41MB and capture-time pool peak 9MB; charged per spec admission and gating future captures)
[spec-vg] lines: 0
G_fa at ready (the probe's grows, inside the floor's measurement) = 7512960; grows after ready = 2
grow at 1790048323315: used 13198 -> 13262 MiB; g_i=8586240 (8.2 MiB); inflight=8
grow at 1790048332795: used 14062 -> 14062 MiB; g_i=17172480 (16.4 MiB); inflight=1
G3: exact rows max |residual| = 0 (tolerance 0); line~ rows max |residual| = 482808 (tolerance 2e6 at the 1 MB grain)
G1: growth(end) = 25758720 (G_fa after ready 25758720 + delta_D 0) <= 10% floor = 161061273: PASS
G2: while inflight > 0, max_underbook_real = 3144845816 <= floor = 1610612736: FAIL
G4: Overloaded/OOM lines = 0: PASS
```

The same ladder day 24 produced (`G_fa = 33,271,680 B`, section 1.2's arithmetic to the byte), the same two post-ready
triggers (S8 at 8 in flight, L at 1), `delta_D = 0`, G3 `residual = 0` on 17 exact rows including the spec rows inside
the bursts (`S8-0`, `S8b-1`: `cost_exact - kv_hat = 935424 = 16704 x 56`). G2 FAIL for the same reason as the target
card (3,144,845,816 B at the `S8b-1` admit with one request booked: pool reserved 8,791,261,184 -> 13,287,555,072 B
over the walk, 1,545,726,224 B cached at that sample, the prefix entries under the 2,052 MB budget, the parked spec
sessions); minus `growth(t)` it reads `3119087096`, the graph term's share 0.8 %.

### 3.3 Verdict on the fix, against the pre-registration

G1 PASS on both cards: the growth attributable to the shapes the walk presented is 38,638,080 B (target) and
25,758,720 B (local), 1.7 % and 1.6 % of the respective floors, against a 10 % line. Per section 1.4, **no per-request
booking lands today**: the floor covers the term by a factor of 60, and a per-request charge for a shared, grow-only
device pool would be charged N times for one allocation (the same over-booking direction day 24 named for `W`'s
`call_row_bytes` term). G2 is FAIL on both cards before, as on day 24, and the fix that would move it is not a booking:
the in-flight device delta the books do not carry is the async pool's cached blocks plus the retention the budget
intends (prefix entries, parked sessions), which `effective_free_bytes` already hands the gate and `nvidia-smi` cannot
see. No tolerance is moved; the FAIL stands as measured.

**The booking point, proposed (not done).** The term is predictable from the request only in the sense that its
DEMAND is (`n_head x ceil(t_kv / sp) x head_dim`, and `t x` that on the rows path), but the ALLOCATION is lazy, shared
and monotone: it happens once per process per rung, at whichever request first presents the shape, and is never
returned. The right booking point is therefore the boot, not the admission: `fa_dcw_pool_ensure(head_dim, n_head,
n_head_kv, served_ctx)` (`lib.rs:32380`, exists, idempotent, callable from outside any capture) run for each loaded
model inside `run_boot_calibration` BEFORE the probe's `pool_high_water_reset`, with the rows-path multiplier
`decode_batch_cap()` on a `fa_sm_count() >= 128` card, so every rung the served context and the batch cap can reach is
grown before ready and the calibrated floor is measured with the pool at its final size; the two `[fa-pool] grow` lines
this walk produced after ready would then not exist. Cost: about 0.3 agent-day (one call site, one CPU test on the
demand arithmetic, the walk re-run on both cards as the after cell with `grows after ready = 0` as the assertion). It
changes no numeric program (the pool's contents are per-launch scratch; only its size moves) and needs no flag; it
does change the boot's memory footprint by the final rung (67 MB on the 9B at 262,144, more on the rows path), which
is why it is proposed here for the owner rather than landed. The `[spec-vg]` pool (MoE + linear families) is the one
graph pool whose growth is per new key and charged on the physical side only (`vg_debt`), so on those families the
predictive book has a real gap this lane's models cannot measure; named for the lead.

## 4. Checks actually run

| Check | Result |
| --- | --- |
| `git merge --no-ff origin/main` (`3df055601`) | clean, `check-conflict-markers: OK` |
| `cargo build --release -p memra-server --offline`, local (CPU quota) and target card | `exit=0` both (`rtx5090-day28/build-local.log`, `pro-single-day28/build.log`) |
| `cargo fmt --all -- --check`, clippy `-D warnings` | no Rust source changed today (no engine change); not re-run |
| `python3 -m py_compile` on the client and parser; `bash -n` on the runner, waiter and chain | OK |
| Target-card cell through the collector (`cell-before`) | `status executed-not-qualified`, `qualification false`, `exit_code 0`; 25 of 25 requests 200 |
| Local cell under `flock /tmp/memra-5090.lock` (`before`) | `before.exit` 0; 25 of 25 requests 200; attempt 1 kept as `before-attempt1-oom-cotenant/` |
| Full GPU exactness battery (`kernel-check`, `run-gen`, `run-spec`) | NOT RUN (no engine change; nothing is qualified today) |
| `git diff --check`, `cargo fmt --all -- --check` (CPU quota), `tools/check-flags.sh`, `tools/check-conflict-markers.sh`, `python3 tools/check-public-boundary.py check` | clean; PASS; `check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)`; `check-conflict-markers: OK`; `public-boundary: 582 matches (582 grandfathered, 0 new)` |
| em-dash census on every file written today | 0 |

## 5. Boundaries and record

- No engine change, no new `MEMRA_*` read, no default changed (the park door's row gains a date, not a value), no format
  substitution, no `unsafe`, no third lock name, no bare GPU run (the target-card cell under the collector's
  `/tmp/memra-gpu.lock`; the local cell under `flock` on the canonical 5090 lock; the OOM'd attempt held the lock too),
  no `--no-verify`, no touch of `/root/artifacts`, `/root/memra-spill` or other lanes' worktrees or processes (the
  co-tenants of the local card were read from `nvidia-smi` and left alone; the one process this lane stopped was its
  own waiter, by pid), no host, id, location or cost in a tracked file, no cross-box timing; every cell
  `executed-not-qualified`; every verdict line verbatim; no tolerance moved after a result.
- Receipts: `pro-single-day28/` (mirror of the target card's `b-day28`: `cell-before/` collector capture, `cells/before/`
  server.log stamped, client.jsonl, samples.csv, REPORT.txt, gpu and compute-apps before and after; `build.log`,
  `binary.sha256`, `source.txt`, `chain.sh`, `chain.log`, `chain-attempt1.log`, the attempt-1 dirs),
  `rtx5090-day28/` (`before/`, `before-attempt1-oom-cotenant/`, `build-local.log`, `wait-then-run.sh`,
  `before.wait.log`, `before.launch.log`).
