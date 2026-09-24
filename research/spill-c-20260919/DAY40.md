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

## 5. Commits, receipts

Filled in as they land: the instrument commit, the local build, the CPU tests, the RTX 5090 cell under
`rtx5090-day40/attrib/`, the reading, and the target-card sitting's shape for the lead.
