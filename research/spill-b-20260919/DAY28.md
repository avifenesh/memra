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
