# memra#536 census: what the CUDA owner thread runs inside the tick (day 16, lane A)

Tree `21307b636` (lane `lane/spill-a-20260919` after the merge of `origin/main` `653c997f4`); every line
number below is at that tree. The issue's claim: one worker thread owns the CUDA context and runs prime,
decode, prefix snapshot capture, hit restore (D2D), host demotion (D2H), promotion (H2D) and pool trim
synchronously, so each of them stalls every other request's decode tick. This file is the code reading,
class by class: the call site, whether it runs on the owner thread, whether it synchronizes the device,
and whether the host-contracts door (`MEMRA_KV_HOST_CONTRACTS=1`) already moves it onto a `TransferEngine`
stream (which side, which stream, where publication still waits). No engine change in this task.

## The thread and the tick

`memra_server::worker::run` (`worker.rs:15956`) is the one thread that binds the CUDA context for every
loaded model; every class below executes inside its loop body. The loop's order per iteration: command
drain and admission (`17440` on), the parked `TrimPools` replies (`17456`), then "3. The tick"
(`19141`): the disconnect sweep (`19160`, `s.tx.is_closed()` per active session, `abort_log`), phase (a)
spec bursts, phase (b) one `prefill_tick` per prefilling session (`20183`; the dark-lane twin `20623`),
phase (c) batched decode, then the DFlash publish and drains (`20654`, `20666`), then retirement
(`20700` on). Nothing in the loop hands work to another thread except the constraint compiler
(`resolve_constraint_compiles`, CPU) and the HTTP layer's channels.

## Census

| class | call site (tree `21307b636`) | owner thread | device synchronize | door (`MEMRA_KV_HOST_CONTRACTS=1`) |
|---|---|---|---|---|
| prime | `prefill_tick` `26167`; one call per prefilling session per tick at `20183` (interactive, budget `interactive_prefill_budget` `183`: `PREFILL_TICK_T = 1024` `55`, widened to `SOLO_PREFILL_TICK_T = 8192` `60` for a sole fresh session, the WHOLE queue for the monolithic class `monolithic_prime_model` `26552`) and `20623` (dark lane, one chunk per tick); the batched fresh wave `prime_cache_batch` `20022` (dark twin `20555`); the hyper walker through `prime_service.advance` `26355` (one chunk per tick under `MEMRA_PRIME_YIELD=1`, else the whole take). Engine entry `HybridModel::prime_cache_overlaid` (`hybrid_forward.rs:5501`) whose inner driver splits the take into internal chunks (the walks at `5722` GEMM loop, `5936` serial, `2400` single-engine hyper, `2365` hyper ranges, and the pipelined `4794`, `6081`, `6243`), each stamping `progress::note_prime_rows` when its logits are host-side | yes | per internal chunk: the chunk's logits are a `Vec<f32>`, a D2H that drains the owner stream (`progress.rs:15-22`, "the device finished that chunk's work and the host read the answer back") | not touched by the door. Cancellation point: only the tick-top sweep (`19160`), that is once per `prefill_tick` call (1024 tokens; 8192 solo; the whole prompt for E4B); none inside the engine call at the internal chunk boundary (memra#536 item 2) |
| decode | phase (c) `decode_step_batch_sampled_lean_masked*` over survivors (`20384`, `20393`); eager-only `step_session` (`20245`) | yes | per step: the sampled logits or ids come back to the host | not touched |
| prefix snapshot capture (D2D) | `prefix_snapshot` `13150`: per layer `engine.alloc_u8` + `engine.copy_u8_into` (`13236-13243`, D2D on the owner stream), `clone_dtod` for the recurrent states (`13257-13258`); callers `prefix_insert_from_session` `14284` from the LCP-split boundary `26475`, the grid seed `26490`/`26500` (`maybe_prefix_seed` `14418`) inside `prefill_tick`, and the DFlash/glm5 drains `19320`/`19325`/`20654`/`20666` | yes | no explicit synchronize: stream-ordered copies; the next step's own D2H drains them. The insert that follows can evict, and an eviction into the host tier is the demote class below, synchronous inside the same call | not touched (the door binds identity to the captured planes after the copy; no copy program change, `HOSTPREFIX-DOOR.md`) |
| hit restore (D2D) | `prefix_restore_at` `13368` (`prefix_restore` `13578`): per layer `engine.copy_u8_into` (`13550`, `13553`), `set_i32_one`, `copy_into` for the recurrent states (`13559-13560`); called from admission for a device hit and after a host promote (`host_promote_prefix_hit` `11596` publishes, then the restore copies into the session cache) | yes | no explicit synchronize; stream-ordered | not touched |
| host demotion (D2H) | OFF: `host_entry_from_device` `10700` reads every KV plane with `Engine::dtoh_u8_into_pinned` (`9111`, `9116`; `lib.rs:13559`: `memcpy_dtoh` then `stream().synchronize()` PER PLANE) and every f32 plane with `host_glm::read_f32` (`host_glm.rs:13-15`: `clone_dtoh` then `synchronize`). ON (Option B, pageable tier): `host_kv_planes_through_contract` `9260`: ONE `TransferEngine` batch. Callers: the evicting insert (`insert_pinned_demoting`, `evict_all_demoting` `10988`) reached from the capture publish inside `prefill_tick` and from the pause-demote sweep after retirement (the `[prefix-host] pause armed` arm, `MEMRA_KV_PAUSE_DEMOTE`), and `host_demote_prefix_entry` `10963` | yes (`CudaTransfers::check_thread`, `tier_transfer.rs:376`, refuses any other thread with `WrongOwner`) | OFF: one host-blocking `synchronize` per plane (2 per layer, plus draft and f32 planes). ON: `t.synchronize(&ticket)` `9564` = a blocking `cuEventSynchronize` per item (`tier_transfer.rs:861-869`), then `owner_stream().synchronize()` `9674` before `retire`/`acknowledge` | ON side moves the D2H onto the engine's contract but the `CudaTransfers` stream IS the worker's owner stream (`CudaTransfers::new(owner, ..)` `tier_transfer.rs:355`, `stream: owner`; constructed in `host_tier_context` with the worker's stream). Publication (`take_plane` back into the entry's slots, `bind_tier_image`) waits on the owner thread inside the same call. The door changes ownership and receipts, not the stream or the wait |
| host promotion (H2D) | OFF: `plane_up` through `Engine::htod_u8_into` (`lib.rs:6875`: `memcpy_htod` from the pinned lease on the owner stream, no host synchronize; the optional `MEMRA_KV_HOST_VERIFY` digest is a D2H readback). ON (Option C): `host_kv_planes_from_contract` `9930`: one `TransferEngine` batch. Caller `host_promote_prefix_hit` `11596` from admission (tick top), then the D2D restore above | yes (same `check_thread`) | OFF: none explicit (stream-ordered; the following restore and step drain it). ON: `t.synchronize(&ticket)` `10316` (event waits per item) then `owner_stream().synchronize()` `10466` before `retire_source`/`retire`/`acknowledge` | ON side is on the owner stream too; `Completion::require` against the D2H receipts and `ready_view` per item happen before the caller publishes through `insert_pinned_demoting`. The ON promote therefore ADDS two host waits the OFF promote does not have (the day-15 pair on this card: promote minus inline demote 5.8 against 4.4 ms; demote 82 against 6.1 ms steady) |
| pool trim | `TrimPools` executed at the tick top `17456-17472`: `synchronize_model_devices` (`29016-29020`: `owner.stream().synchronize()` per device), pool eviction (`px.evict_all`, reuse pools cleared), `trim_model_device_pools` `28966` (`owner.stream().synchronize()` `28980` then `pool_trim_to_zero`, one report per device, in the SAME call). Also from the admission drain `18616` and the OOM teardown `21069` | yes | yes, twice (before the drop and before the trim) | not touched. The issue's "does not return pool reservation until a second call" is not what this tree's handler reads (the device trim and its report are in one call at `17472`); not measured here |

Two facts the table makes plain. (1) Under the door the D2H and the H2D leave the OFF copy calls for the
engine's contract, but they do not leave the owner stream or the owner thread: `CudaTransfers` is built
on the worker's stream and refuses any other thread. What the door buys is ownership, receipts and the
typed unwind, not concurrency; the synchronous waits (`synchronize(&ticket)`, `owner_stream().synchronize()`)
are the publication's ordering, so the tick still absorbs the copy. (2) The only cancellation point of a
prime is the tick-top sweep, so a disconnected client's prime runs to the end of the CURRENT `prefill_tick`
take (1024 tokens; 8192 for a sole fresh session; the whole prompt for the monolithic E4B class) and the
engine's internal chunk boundary, where the odometer already stamps, has no check. A failed or aborted
prefill never parks (`20936` requires `prefill_done`; `retire_may_park` refuses an aborted session), so what
was primed returns to the pool at retire, and the capture sites (`26475`, `26490`, `26500`) run only after
the prime call returned `Ok`, so an interrupted prime publishes nothing.

## Pre-registration of the stall cell (written and committed before any GPU run)

**Question.** By how much does a concurrent tenant's decode tick stretch while another request runs (a) a
cold prime of a fixed length, (b) a host demotion, (c) a host promotion, each against an idle control, on
the target card class, door OFF and ON for (b) and (c).

**Rig and regime.** BOX3, one RTX PRO 6000 Blackwell Server Edition at its 600 W limit (the card class of
day 15). `memra-server` built on the box from this lane's tree after the merge (engine source equal to
`main`'s at `653c997f4`), the Qwen3.8-27B NVFP4-Q5K artifact with its embedded MTP drafter, the day-15
boot shape (`MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_KV_HOST_MB=8192`) plus `MEMRA_SERVE_SPEC=0` in
EVERY boot so the tenant emits exactly one token per tick and its inter-token time IS the tick (under the
default spec environment a step emits a burst and the client-side gaps inside a burst are zero; that
shape is not measured today and is stated as such). Every cell runs through `tools/tier-battery.py --rig
pro-single` (one `/tmp/memra-gpu.lock` hold per boot pair, 250 ms telemetry, the lock proof); the
collector's regime (temperature, power under the 600 W limit, SM clock) is reported per cell. Lane B may
hold the card: the runner retries a busy lock boundedly and never signals a holder.

**Tenant.** One streaming `/v1/completions` request (`stream: true`, `temperature: 0`, `max_tokens: 160`)
with a 20-token prompt (below `PREFIX_CACHE_MIN_TOKENS = 64`, so the tenant never captures, never evicts
and never contends for the prefix cache). The client records the monotonic arrival time of every SSE
token event; ITL sample `i` is the gap between token `i` and token `i+1` for `i` in 1..159 (TTFT excluded).
The intruder fires when the tenant's 24th token arrives, so at least 130 tenant ticks follow it.

**Intruders (one request each, `max_tokens: 1`, `temperature: 0`).**
- (a) `prime`: a fresh 4096-token prompt, distinct per run (a run counter in its first line so no prefix
  hit is possible), in a boot with `MEMRA_PREFIX_CACHE_MB=0` so its prefill-done publishes nothing (no
  capture, no eviction, no demote: the prime alone). Under a concurrent tenant `sole_unfinished` is false,
  so the prime takes `PREFILL_TICK_T = 1024` tokens per tick: the expected shape is four stretched ticks.
- (b) `demote`: a fresh 64-token prompt, distinct per run, in a boot with `MEMRA_PREFIX_CACHE_MB=256`
  (one 64-token entry of this model, about 160 MB by the day-15 receipts, fits; the second does not): its
  prefill-done seed inserts and evicts the previous entry into the host tier, one D2H demote. The window
  contains that one-tick 64-token prime as well; the server's own `[prefix-host] demote: .. ms` line for
  the same run is read beside the tenant's ITL so the composition is on the record, not inferred.
- (c) `promote`: the day-15 pair's alternation: with `E_A` and `E_B` seeded once per boot (not timed), each
  run's intruder is the prompt whose entry is on the host: a host hit that promotes (H2D) and whose insert
  evicts the other entry (the inline demote of C's harness), then the D2D restore. The server's `promote:`
  and `demote:` lines of the run are read beside the ITL.
- Door arms: (b) and (c) run in an OFF boot and in an ON boot (`MEMRA_KV_HOST_CONTRACTS=1`); (a) runs in
  the cache-off boot only (the door has no arm in a prime).

**Design.** Per class-arm one cell: order 1 = (`idle`, `arm`) x 5, order 2 = (`arm`, `idle`) x 5, so N=5 per
arm per order, N=10 pooled, where `idle` is the tenant alone. Five class-arms (`prime`, `demote-off`,
`demote-on`, `promote-off`, `promote-on`); the two OFF arms share one boot and the two ON arms another,
so the sitting has three boots (cache-off, OFF, ON) inside one collector session per boot. The tenant's
wall clock per run and the intruder's own latency are recorded too.

**What is reported, per class-arm, verbatim from the harness's one rule line.** The idle tenant's ITL
p50/p95/p99 and max, pooled over its 10 runs; the arm's ITL p50/p95/p99 and max, pooled; and the stall,
defined before the run as `stall_ms = max ITL of the arm run minus the p50 ITL of the same run` (the
tick that absorbed the intruder against the run's own steady tick), reported per run and as the median
over the 10 runs with min and max; plus the server-side `demote:`/`promote:` ms of the same runs. No
threshold and no verdict clause: the cell measures a stall per class, the issue's acceptance ("decode ITL
p99 for peers unchanged") is the target of the design note's cells, not of this census. What the numbers
are and are not: one card class, one host, one artifact, `MEMRA_SERVE_SPEC=0`, a 64-token entry size (the
issue's 3 GB entry is a 135k-token GLM shape; not measured here). No number here is divided into a number
from another box. Every cell is `executed-not-qualified`.
