# WP-C day 40 (2026-09-24): the MoE slot cache door, step (a): where the 60 ms per decode token goes

Lane `lane/spill-c-20260919`, `OWED.md` item C1 step (a). Tree at start: `origin/main` `d61012658` (#716), the lane
fast-forwarded to it (day 39 was already in main through integ48), then `33d6f5cef` (`OWED.md`). Every cell here is
`executed-not-qualified` development evidence. Nothing here moves a default or claims a support state. No number is
compared across cards.

## 0. What is being attributed

`DAY18.md` cell `overlap`, target card, verbatim: `OVERLAP-PAIR rule decode_off_s=0.408 decode_on_s=2.343
decode_ratio=5.743 ... door_cost_ms_per_decode_token=60.47 steady_mb_per_token=43.5 ... misses_off=[8211]
misses_on=[18195] reads_on=[22077] evictions_on=[12091] install_on_s=74.84 ... -> sync_miss_path_slower`. The same
shape on a rented RTX 5090 (day 8) read `generated 32 tokens in 0.303s` OFF and `2.885s` ON.

Two counts in that line are reconciled here from source before any run, so the attribution does not chase them:

- `misses_on - misses_off = 9984`, about the 9986 slots. `force_admit` (the one-shot `prewarm_layer` fill) counts
  no miss on the legacy path (`moe_cache.rs` `force_admit` then `admit`, neither touches `misses`) and one miss per
  block under the door (`force_admit` goes to `admit_banked`, which counts). The routed traffic is the same in both
  arms, which the identical STEADY-STATE line (`hit-rate=90.4% | 43.5 MB/decode-token`) already said.
- `reads_on - misses_on = 3882`: the steady-state window in `run_gen.rs` (the re-prime of 35 decode steps plus 32
  measured steps) runs after the cumulative line is printed; its host reads are in `physical_reads`, its misses
  are not in `misses_on`.

## 1. The two programs, from source (tree `d61012658`)

Per GPU-slot miss on the decode path (`hybrid_forward.rs` `moe_cached_gemm_q8` and siblings call
`MoeSlotCache::dispatch_source`):

- **Legacy (door absent).** `dispatch_source` -> `admit` -> `admit_native`: `reserve_slot` (free pop or SLRU evict),
  `Engine::stage_expert` = `stream().memcpy_htod(&[u8], ..)` from the loader's pinned host slab (`HostBuf::Pinned`,
  `MEMRA_MOE_PINNED` on by default under the cache), then `publish`. cudarc 0.19.8's `HostSlice for [T]` returns
  `SyncOnDrop::Sync(None)`, which synchronizes nothing, so the copy is an asynchronous DMA ordered on the compute
  stream and the host thread runs ahead. The legacy copy-stream prefetch (`prefetch_source`) runs only under
  `MEMRA_MOE_PREFETCH=1` or the pread worker (`moe_prefetch_enabled`), neither of which the day-18 shape sets, so
  the day-18 OFF arm had no prefetch either. (The comment on `ExactPinnedPrefix` in `moe_cache.rs` says the raw
  slice "waits for the whole stream before returning"; on cudarc 0.19.8 it does not. Recorded, not changed today.)
- **Door (`--experts-via-tier`).** `dispatch_source` -> `admit_banked`: `validate`; table hit or
  `ExpertBankProxy::demand` -> `TracedDispatch::demand` (SLRU residency check, the `[expert-host-slru]` eprintln)
  -> `SlruExpertDispatch::demand` -> `BankService::stage` (`plan_reads`, governor reservations, `ReadWork::new`
  allocating and zero-filling the record's output `Vec`), `progress` (`ReadWork::step`: a second zero-filled slot
  `Vec`, `FileReader::read_exact` = `pread` from the artifact inode, a `copy_from_slice` into the output; then the
  SHA-256 `checksum` of the record and the zero-tail check), `publish` (lease, host SLRU); then
  `with_bytes` -> `admit_native`: `reserve_slot`, `stage_expert` from the lease's pageable `Vec<u8>`, then the
  banked branch's `stream().synchronize()`; back in `admit_banked` a second `stream().synchronize()`, then
  `finish` (`finish_host_use`, `retire`, `acknowledge`, `collect_evicted`). One lease at a time (`max_pending = 1`,
  `items: 1`, `tickets: 1`), a 16-record host tier (the 256 MiB qualification ceiling clamps to 16 records).

So the door differs from the legacy miss in: a compute-stream drain per miss (two syncs), a host-tier read per miss
at this budget (16 host records against a working set of thousands), two zero-filled allocations, one assembly copy,
one SHA-256, one trace line, the bank bookkeeping, and a pageable instead of pinned H2D source.

## 2. The instrument (log-only; lands before any cell)

A stage clock behind a new gate-only CLI flag of the door, `--expert-bank-stages` (no value; requires
`--experts-via-tier`; a repeat or a value is a usage error, the same exact-key parser). No `MEMRA_*` read, no
default, no numeric change: it adds `Instant` reads and, per miss, two timing-enabled CUDA events on the compute
stream around the H2D, and prints lines. Without the flag the door's program is today's, byte for byte in
behaviour. It is an explanatory diagnostic of the door and shares the door's decide-by (2026-10-04, recorded in
`MOE-SLOT-CACHE-DOOR.md`): it goes when the door goes.

Timed sites (every bracket is exact code; nothing is estimated):

| Clock | Name | Bracket |
|---|---|---|
| cache (`MoeSlotCache`, CUDA thread) | `admits`, `gpu_hits`, `gpu_misses` | counts in `admit_banked` |
| | `validate` | `bank.validate(..)` |
| | `demand` | `bank.demand(..)` (the whole owner-side demand below) |
| | `reserve` | `reserve_slot(..)` in `admit_native` (free pop or SLRU eviction) |
| | `enqueue` | `e.stage_expert(payload, ..)` host wall |
| | `copy_gpu` | event E0 recorded on the compute stream before the memcpy, E1 after; `E0.elapsed_ms(E1)` read after the drain (both complete then) |
| | `drain` | the banked branch's `e.stream().synchronize()` in `admit_native` |
| | `sync2` | `admit_banked`'s own `e.stream().synchronize()` |
| | `finish` | `bank.finish(..)` |
| | `miss_total` | `admit_banked` entry to return on the miss path |
| owner (`TracedDispatch`) | `host_hits`, `host_misses` | the SLRU residency check before the inner demand |
| | `inner_demand` | `SlruExpertDispatch::demand` |
| | `trace` | the `[expert-host-slru]` eprintln |
| owner (`FileReader`) | `pread`, `reads` | `read_exact_at` |
| owner (`BankService`) | `stage`, `alloc` | `stage()` whole; `ReadWork::new` inside it |
| | `step` | `work.step(..)` (slot allocation and zero-fill, the reader call, the assembly copy) |
| | `verify` | the per-segment `checksum` and zero-tail loop in `progress` |
| | `publish` | `publish()` whole |
| | `retire`, `collect` | `finish_host_use` + `retire` + `acknowledge`; `collect_evicted` |
| installer | `sha`, `catalog`, `records`, `setup` | the artifact SHA-256 pass; `plan_catalog`; every `bank_projection` (the byte compare against the loaded `HostExps` and the per-record checksum); governor, bank, SLRU, dispatch, owner, cache install |

Output. One line per phase from `run-gen` when the flag is on: `[experts-via-tier] stages phase=<p> <name>=<count or
ns> ...`, cumulative since install, at `phase=gate` (after the argmax verify, before `generate_with`),
`phase=generate` (after `generate_with`), `phase=warm` (after the steady window's re-prime, where the window's
counters are reset) and `phase=window` (after the 32 measured steps); the installer prints `[experts-via-tier] install
sha_ns= catalog_ns= records_ns= setup_ns=` with the flag. For EVERY arm (door or not, flag or not), `run-gen` prints
one new line after the measured window: `MoE cache STEADY-STATE window: 32 decode steps in <s>s` (host wall around
the 32 `decode_step` calls; each returns host logits, so no sync is added). The existing lines are unchanged.

## 3. The cell `attrib` (pre-registered here; the RTX 5090 first, the target card in the next sitting)

**Shape.** The day-18 `overlap` environment unchanged: `MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986`,
prompt `55 88 13`, the approved artifact (`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`,
`unsloth/Qwen3.6-35B-A3B-MTP-GGUF@5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`, SHA-256 `df27a780...7adf`, checked
before the hold). Three arms: OFF (no door), ON (`--experts-via-tier`), ONS (`--experts-via-tier
--expert-bank-stages`). Order 1: (OFF, ON, ONS) x 5; order 2: (ONS, ON, OFF) x 5; N=5 per arm per order, N=10
pooled, 30 runs in ONE collector hold (`tools/tier-battery.py --rig rtx5090 --external-lock`, lock
`/tmp/memra-5090.lock`, 250 ms telemetry; on the target card `--rig pro-single`, `/tmp/memra-gpu.lock`). Binary
`run-gen` built from the instrument's commit, its SHA-256 in `ev/binary.sha256`; every run's output through the
day-18 line stamper, exit status kept.

**Integrity (a failure voids the reading, the clause named).** All 30 runs exit 0 and print `MATCH`; one `tokens:`
tape across the 30; `slots=9986` in every cache line; one STEADY-STATE hit-rate and MB line across the 30; ON and ONS
print `installed` and `physical_reads=`; ONS prints four stage lines and one install line, ON and OFF none; OFF prints
no door line.

**Readings (computed by `day40-attrib.py` from the stamped logs; no threshold moves after a run).**

- R1, the door's cost where it was measured: `door_ms_per_token = (median ON gen-only - median OFF gen-only) / 32`,
  pooled and per order (the day-18 observation), and the same on the steady window's wall
  (`window_door_ms_per_token`).
- R2, per stage per decode token, ONS, the steady window only: `(value at phase=window - value at phase=warm) / 32`,
  milliseconds, per run, median over the 10 ONS runs; counts per token the same way (GPU misses, host misses,
  reads).
- R3, coverage: `serial_ms = validate + demand + reserve + enqueue + drain + sync2 + finish` per token (the owner
  thread's serialized door work on the miss path; `drain` includes the GPU finishing the work queued before the
  copy, which the legacy program overlaps), against `window_door_ms_per_token`. Reported, and `miss_total` against
  the sum of its parts (the residual is the unbracketed code between the timers).
- R4, the split inside `demand`: `inner_demand`, `trace`; inside `inner_demand`: `stage` (of it `alloc`), `step` (of
  it `pread`), `verify`, `publish`; `finish` into `retire` and `collect`.
- R5, the instrument's own cost: `instrument_ms_per_token = (median ONS window - median ON window) / 32`. **Clause:**
  if it exceeds 5 percent of `window_door_ms_per_token` or 2.0 ms per token, the attribution is labelled
  `perturbed`, and the stage clock is revised and re-run before any improvement cites it.
- R6, install: `sha`, `catalog`, `records`, `setup` medians over the ONS runs, against the ON runs' `install_s` (the
  `installed` line minus the `[q8rp] split-plane decode mirrors built` line, day 18's definition).

**Decision rule.** The attribution decides no default and no verdict on the door. It orders the improvement list
below by R2's per-token cost on each card (largest first) and gives each improvement its pre-registered expected
stage movement. The owner order stands: every improvement on the list is designed, landed and measured whatever its
rank; a stage R2 shows at zero is still measured once and recorded as such.

**What each card can decide.** The RTX 5090 cell attributes the 5090's door cost and orders the 5090's list; it
cannot attribute the 60.47 ms, which is the target card's. The target card's `attrib` cell, run with the same script
and the same binary tree in its next sitting, attributes that figure. Neither card's number is a denominator for the
other.

## 4. The improvement list the attribution orders (each gets its own day, pre-registration and receipt)

- I1. **No compute-stream drain per lease.** The H2D rides the copy stream after an event that protects the reused
  GPU slot's earlier readers; the compute stream waits on the copy's event before the consumer; the slot is published
  only after the copy's submission and its event record succeed and the compute-stream wait is enqueued (the
  `d0acf6f03` rule: an unknown submission result leaves the slot outside every table); the host lease retires when
  its event is observed complete (several leases in flight, retired in order; `max_pending` > 1). Expected: `drain`
  and `sync2` to near zero.
- I2. **Bounded pinned host tier slots.** Preallocated cached pinned buffers (cached, not write-combined: the owner
  hashes them), the positioned read straight into the slot, no per-read allocation, zero-fill or assembly copy, the
  H2D a true DMA. Expected: `alloc`, the non-read part of `step`, and `copy_gpu` down.
- I3. **Read and verify off the owner thread.** A reader worker does the `pread` and the SHA-256 of demanded and
  known-next records; the owner thread keeps the H2D and the publication. Expected: `pread` and `verify` leave the
  owner's serial path.
- I4. **Bank-routed prefetch of the known-next experts.** The router's selection is on the host when the layer
  dispatches; the next experts' records are demanded as `OptionalPrefetch` and their H2D submitted on the copy stream
  while the current expert computes (the legacy program's shape under `MEMRA_MOE_PREFETCH=1`, with the bank's bytes).
- I5. **The complete host-demand trace off the hot path.** Buffered and written whole at close, not one unbuffered
  stderr line per demand. Expected: `trace` to near zero, the trace itself unchanged line for line.
- I6. **Host residency sized to the working set.** The 256 MiB / 16-record ceiling re-derived from the machine's host
  budget through the governor, so a GPU eviction re-demanded later is a host hit (no read, no verify).
- I7. **The installer.** The record compare reads the loaded `HostExps`, which live in write-combined pinned memory
  (`HostBuf::Pinned`, cudarc's `alloc_pinned`); the per-record verification is serial. Expected: `records` down.

## 5. The RTX 5090 cell, as it ran (`rtx5090-day40/attrib/`)

**The instrument landed first** (`08210a291`): the stage clock, the CPU tests (`day40-cpu/tier-bank-tests.log`: tier
bank `64 passed` after the fixture re-pin below; `day40-cpu/engine-lib-tests.log`: engine lib `546 passed`), clippy
`-D warnings` clean over both crates and all targets (`day40-cpu/clippy.log`). The day-4 SLRU fixture pins the SHA-256
of `moe_cache.rs`; the first tier run failed on that pin alone (`left: "072cad5d..."`, the old digest), the file's
diff adds clock statements only and binds `stage_expert`'s result to a name before the same `if let Err`, so the
recorded SLRU trace stands and the digest was re-pinned (`d0acf6f03`'s precedent). A dry check (`day40-cpu/dry-check*`,
one ONS run under the lock outside the collector, not a receipt) confirmed every stage line prints.

**The hold.** Lane B's queue held the card first (`waits.log`: 8 bounded waits at 120 s; my first runner was stopped
inside its wait and restarted with longer bounds, `chore` commit, before any cell ran). Then one collector hold,
16:46:38Z to 17:12:10Z, 30 runs, `attrib.exit` 0, collector `--validate` rc=0 (`attrib-validate.log`). Binary
`run-gen` `b73bb4d3...` built from `08210a291` (`ev/binary.sha256`); the cell's tree is the next commit (the runner's
wait bounds only; `git diff 08210a291 <tree> -- crates` empty). The artifact `df27a780...7adf`, fetched today from its
pinned revision and checked (`ARTIFACT_SHA256_MATCH`). Regime over the hold (`regime.log`, 6,087 samples at 250 ms):
53 to 70 C (median 56), at most 147 W, SM clock up to 2775 MHz (the median 180 MHz is the CPU-bound install and SHA
phases). No compute app of another process at any run boundary (`ev/*.snap`).

**The reading, verbatim** (`reading.log`, `day40-attrib.py`):

```
DAY40 ATTRIB CHECKS rig=rtx5090 runs=30 integrity=ok
DAY40 R1 door_ms_per_token gen: pooled=39.25 o1=40.31 o2=38.75 | window: pooled=18.25 o1=18.47 o2=18.16 | medians gen off=0.401 on=1.657 ons=1.692 window off=0.331 on=0.915 ons=0.936 (N=10 per arm)
DAY40 R2 per_token_ms validate=0.649 demand=16.364 reserve=0.022 enqueue=1.075 copy_gpu=2.834 drain=2.071 sync2=0.019 finish=0.264 miss_total=20.300 inner_demand=15.993 trace=0.265 pread=3.457 stage=1.286 alloc=0.754 step=5.036 verify=9.382 publish=0.231 retire=0.094 collect=0.135 (N=10 ONS runs)
DAY40 R2 per_token_counts admits=471.0 gpu_hits=378.7 gpu_misses=92.3 host_hits=0.0 host_misses=92.3 reads=92.3 stages=92.3 steps=92.3 verified=92.3 copy_events=92.3 event_errors=0.0
DAY40 R2 serial_rank demand=16.364 > drain=2.071 > enqueue=1.075 > validate=0.649 > finish=0.264 > reserve=0.022 > sync2=0.019
DAY40 R3 serial_ms_per_token=20.465 window_door_ms_per_token=18.250 coverage=1.121 miss_total=20.300 miss_parts_excl_validate=19.816 residual=0.484
DAY40 R4 demand=16.364 = inner_demand 15.993 + trace 0.265 + other 0.106; inner_demand: stage 1.286 (alloc 0.754) step 5.036 (pread 3.457) verify 9.382 publish 0.231; finish=0.264: retire 0.094 collect 0.135
DAY40 R5 instrument_ms_per_token=0.656 bound=0.913 (min of 5% of 18.250 and 2.0) -> within_bound
DAY40 R6 install_s ONS sha=5.08 catalog=0.00 records=55.77 setup=0.11 | ON install_s=60.66 (installed minus q8rp line, N=10)
DAY40 ATTRIB rig=rtx5090 integrity=ok window_door_ms_per_token=18.25 serial_ms_per_token=20.47 instrument=within_bound top=demand
```

**What it says on this card (the RTX 5090; nothing here attributes the target card's 60.47 ms).**

1. The door costs 39.25 ms per decode token on the gen-only span and 18.25 on the steady window; both orders agree
   within 1.6 ms. The window's miss traffic is 92.3 GPU misses per token, every one a host-tier miss and a physical
   read (host hits 0.0 per token at the 16-record tier).
2. The owner thread's serialized door work in the window is 20.47 ms per token, of which the host demand is 16.36:
   the per-record SHA-256 verify 9.38, the read step 5.04 (the `pread` itself 3.46; the slot allocation, zero-fill and
   assembly copy the other 1.58), `stage` 1.29 (the output allocation and zero-fill 0.75), the trace print 0.27, the
   publish 0.23. The compute-stream drain is 2.07, the pageable enqueue 1.08, `validate` 0.65 (471 admissions per
   token, hits included), `finish` 0.26. The H2D itself takes 2.83 ms of GPU time per token (92.3 copies of the
   pageable lease, about 30 us each).
3. Coverage 1.12: the serialized work exceeds the door's window cost because the drain includes GPU compute the
   legacy program also runs (it overlaps it instead of waiting for it). The instrument costs 0.66 ms per token,
   within its bound, so the stage lines are usable as registered.
4. The install is 60.7 s per process on this host, 55.8 s of it the record pass (the byte compare against the loaded
   `HostExps`, which live in write-combined pinned memory, and the per-record SHA-256), 5.1 s the artifact SHA pass.

**The order this sets for the improvement list (the decision rule).** `demand` first: I6 (host residency sized for
the bank), then the host fill that serves first touches before they are demanded, then I2 (pinned slots, the read
straight into the slot), then I1 (the drain), then I8 (a `validate` memo; a new item: 0.65 ms per token spent
validating hits, the ids and lengths never change), I5 (the trace), I7 (the install), I4 (prefetch). One reading
outside the rule shapes I6, recorded here: a stack-distance profile of one ONS run's host-demand trace
(`o1-ons-r1.log`, 22,077 demands, 17,688 distinct records) gives an LRU host tier a window hit fraction of 0.000 at
every budget up to 2 GiB, 0.039 at 4 GiB and 0.439 at 8 and 16 GiB, and 1,659 of the window's 2,955 demands are the
first demand of their record in the process: a GPU miss is for a record evicted long ago or never loaded, so a host
tier helps only past the GPU cache's reach, and first touches need the fill.

**The target card.** The same cell (`day40-cell.sh attrib`, `--rig pro-single`, `/tmp/memra-gpu.lock`) is owed there
to attribute the 60.47 ms; it runs as rung 0 of the target-card sitting that measures the improvement ladder (the
sitting's shape goes to the lead when the ladder's 5090 rungs are in).

## 6. Commits

`33d6f5cef` (`OWED.md`), `a47d950d4` (this file's sections 0 to 4, before any code), `08210a291` (the instrument, the
cell, runner and reader), the runner's wait-bounds commit, and the receipts commit that carries this section.
