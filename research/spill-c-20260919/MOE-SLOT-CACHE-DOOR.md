# MoeSlotCache door: experts-via-tier owner proxy

Status: landed, default OFF, CLI door. Decide-by: **2026-10-04** (lead ruling 8,
`research/spill-lead-20260919/HANDOVER-20260920.md`: a CLI door needs no
`docs/FLAGS.md` row but carries its decide-by in its design doc; 14 days after the
day-nine seal on 2026-09-20). Every cell behind it is `executed-not-qualified`
development evidence on one card class; nothing here is a support state.

## What is landed

Door name: `--experts-via-tier` on `run-gen` and `run-spec`, with the gate-only
budgets `--expert-bank-host-bytes=N` and `--expert-bank-gpu-bytes=N`. Absent, the
legacy SLRU slot cache is byte-for-byte the pre-lane program. No new `MEMRA_*` read.
Landing commits (oldest first): `e21870438`, `82d75d9cd`, `ec1c356de`,
`3d9e28b07`, `e962407df`, `d0acf6f03`, `18b2f092f`, `44f87f181`, `85677e50c`,
`748903f73`, `79353d53d` (typed budgets, GPU slot refusal, native refusal
token), `1de17d41f` (day-ten cell driver and verifier), `6db8ac122` (installer catalog from
the model plan and tensor contract, typed catalog refusals), `76f78c569` (exact flag keys,
shared slot tail pad, gate helpers off the crate root), `6defcd604` (day-eleven driver),
`69905776f` (day twelve: lease token identity, `bank::dispatch_id`, `admit_banked` record assertion).

decide-by: 2026-10-04 (covers the door, both budget flags and the stage clock; CLI doors
carry their decide-by here, not in `docs/FLAGS.md`). Since day 60 it also covers `run-gen
--moe-dispatch-clock` (log only, both the legacy slot cache and the door: the dispatch and prefetch
entry points bracketed, `DAY60.md`), and since day 67 `run-gen --cpu-probe` (log only: a compute chain and L1, L2 and
DRAM dependent-load chases on the main thread after every timed phase, `DAY67.md`) and since day 68 `run-gen
--cpu-probe-phases` (log only: a 2^20-step compute chain at the start and at each stage-line point, `DAY68.md`).

Day 40 (`DAY40.md` section 2): `--expert-bank-stages` (no value, requires the door) installs
the door's log-only stage clock, an explanatory diagnostic: `Instant` brackets around every
step of `admit_banked`, the banked branch of `admit_native`, the owner's demand (`TracedDispatch`,
`FileReader`) and the host bank lifecycle (`BankService::with_stage_clock`,
`BankStageTimes`), plus two timing events per GPU miss around the H2D; `run-gen` prints
`[experts-via-tier] stages phase=<gate|generate|warm|window>` lines and the gate prints
`phase=close` and an `[experts-via-tier] install sha_ns= catalog_ns= records_ns= setup_ns=`
line. No `MEMRA_*` read, no decision changes. It goes with the door. `run-gen` also prints,
for every arm, `MoE cache STEADY-STATE window: <n> decode steps in <s>s`.

| Surface | File | What it does |
|---|---|---|
| Owner registry | `crates/memra-tier/src/bank/owner_proxy.rs` | `ExpertBankOwner` (`!Send` via `PhantomData<Rc<()>>`) registers a `Box<dyn ExpertDispatchBank>` in a thread-local `OWNERS` map. `ExpertBankProxy` (`Clone`, `Send + Sync`: a `ThreadId` and a `u64`) and `ExpertLeaseToken` (`Send + Sync`, no pointer: owner and lease numbers plus the identity the registry holds for the lease, `record()` the `(layer, proj, expert)` key derived from the leased `BankId` through `bank::dispatch_id`, `artifact()` the record's artifact digest, `epochs()` the staging ticket's epochs; day twelve) are the only things that leave the owner thread. `demand` refuses a bank that leases another record than the one demanded (`ProgramMismatch`, retiring the lease first); `with_bytes` / `finish` refuse a token whose identity does not match the pending lease (`ForeignLease`). `access` refuses `WrongOwner` off-thread; `close` refuses `Busy` with an open lease; `Entry::drop` forgets pending backing on unknown completion instead of releasing it. Not RPC: a migrated caller refuses. |
| Bank contract | `crates/memra-tier/src/bank/expert_dispatch.rs` | `ExpertDispatchBank { validate, demand, finish }` and `SlruExpertDispatch` over `BankService`. |
| Cache fields | `crates/memra-engine/src/moe_cache.rs` `MoeSlotCache` | `banked: Option<ExpertBankProxy>`, `banked_pending: Option<ExpertLeaseToken>`, `banked_evictions: u64`. No service, `Rc`, or byte backing is stored in the cache. |
| Install | `moe_cache.rs::install_banked` | Refuses unless the cache is an untouched default SLRU: no resident table, no pending or worker reads, no LFU, not frozen, no pread backend, zero staged bytes, no prior bank. |
| Admission | `moe_cache.rs::admit_banked` | `validate` then hit check, else `demand` one lease, assert `token.record()` names the demanded `(layer, proj, expert)` (retire and refuse otherwise; day twelve), `with_bytes` borrows the payload on the owner and runs the unchanged `admit_native` H2D, `stream().synchronize()`, `finish`. One pending lease at a time; a second demand or a frozen cache refuses. `admit_native` publishes a banked slot only after an explicit stream sync; an unknown result leaves the slot outside every table and queue. |
| Guards | `moe_cache.rs` `admit`, `dispatch_source`, `force_admit`, `prefetch_source`, `restage_block`, `remove_occupant`, `Drop` | Admit and dispatch route through the bank; prefetch returns `false` (no detached legacy prefetch may bypass the owner); freeze-profile restage refuses; evictions under the bank are counted; `Drop` finishes a pending lease only after a successful stream drain and otherwise retains it. |
| Compile assertion | `moe_cache.rs`, `const _: fn()` after `impl Drop` | Ordinary library build (not a test) asserts `Engine: Send + Sync`, `MoeSlotCache: Send`, `ExpertBankProxy: Send + Sync`, `ExpertLeaseToken: Send + Sync`. The cache is `Send` only because its pread receiver is `!Sync`; `Engine` holds it in `Mutex<Option<MoeSlotCache>>` (`lib.rs`), which is what makes `Engine: Sync` hold. Putting an owner-only bank or lease into the cache fails here, before the scoped PP-worker spawns cascade. |
| Native installer | `crates/memra-engine/src/banked_residency/native.rs` `Engine::install_expert_bank_gate(model, gguf, budget: ExpertBankBudget)` | Takes the typed CLI budget as a parameter (no argv or env scan). Hash-locks the already-open inode to `APPROVED_SHA` `df27a780…7adf` (Qwen3.6-35B-A3B-UD-IQ4_XS). Requires one GGUF shard, `MEMRA_MOE_CACHE` enabled, at most one MTP head. Takes its expert catalog from the compiled `ModelPlan` and the model pack's GGUF `TensorContract` bound against the artifact census (`memra_gguf::expert_banks::expert_bank_catalog`, day eleven): semantic ids `LayerTensor::MoeExpert{Gate,Up,Down}Bank`, accepted names, required shapes, quant layouts; no checkpoint name is spelled in the installer and no architecture name is consulted. Typed catalog refusals (`REFUSED: experts-via-tier expert catalog refused: <detail>`, exit 2): no MoE expert projections in the plan; a contract entry missing, duplicated, ambiguous, shape- or layout-incompatible (the contract's own verdict text); a scale plane the artifact carries for a bank (`.scale`, `.input_scale` census rows) or on the loaded `HostExps` (`macros`, `fp8_blk`); a plan/model disagreement (layer count, dense vs routed, MTP head without a plan block). Plain failures (`Error:`, exit 1) stay for shards/cache/MTP-count, the SHA lock, and `dev_exps`, `step_ep`, `step_tp`, `glm5_ep`, `glm5_tp_split` bypasses. Every retained record is still compared byte-for-byte with the loaded `HostExps`. Prints `[experts-via-tier] catalog blocks= banked= projections= catalog_sha256= records= records_sha256=` (identity of the bound catalog and chained record checksums) before `installed`. Builds Governor, `BankService` (per-record catalog), `SlruPolicy`, `SlruExpertDispatch`, a `TracedDispatch` that prints `[expert-host-slru] key= bytes= slot= hit= victim=`, and registers the owner with `max_pending = 1`. Returns `BankedExpertGate` (`!Send`, held on the CUDA thread for the run); its `Drop` prints `[expert-gpu-slru] slots= allocated_bytes= evictions=` and `[experts-via-tier] physical_reads= owner_close=`. |
| Catalog | `crates/memra-gguf/src/expert_banks.rs` `expert_bank_catalog(plan, contract, census)`, `ExpertBankCatalog { projections }`, `ExpertBankProjection`, `ExpertBankBlock::{Trunk, Mtp}`, `ExpertBankCatalogError` | Plan-derived, GGUF-only, no name spelled: for every MoE block of the plan (trunk by position, then MTP by depth) it selects the three bank requirements by semantic id, carries their declared `QuantAux` requirements along, and binds the sub-contract against exactly the census rows those names match through `TensorContract::bind`, so a duplicated name is `DuplicateCensusName` and never a first match. A bound auxiliary is `ScalePlanes { names }`. `identity()` is the hashed text (one line per projection: block, layer, projection, name, shape, storage, bytes). |
| Budgets | `crates/memra-engine/src/banked_residency.rs`: `ExpertBankBudget { host_bytes, gpu_bytes }`, `expert_bank_cli`, `host_bank_plan` / `host_bank_budget` / `host_bank_ceiling` (day 43), `SLOT_TAIL_PAD_BYTES`, `gpu_slot_bytes`, `gpu_bank_slots(bytes, max_record, hard_bytes)` / `gpu_bank_budget`, `ExpertBankRefusal`, `refusal_reason` | Host (`--expert-bank-host-bytes=N`, default 256 MiB): since day 43 planned into one SLRU class per exact record size, slots proportional to each size's records and capped at them; refuses below one record of the largest size and above three quarters of the host's `MemAvailable` (the 256 MiB ceiling and the 16-record clamp are gone; `DAY43.md`). GPU (`--expert-bank-gpu-bytes=N`, default unset = native sizing untouched): exact slot count `N / (max_record + SLOT_TAIL_PAD_BYTES)`, the one eight-byte constant `moe_cache.rs` imports for its own sizing and allocation (day eleven; every replaced literal was 8); refuses below eight slots, above the machine hard ceiling (`moe_cache::hard_slot_bytes`, the same value the native constructor applies), on checked overflow, and when `MEMRA_MOE_SLOTS` is also set (a conflict, never a silent loser). Both refusals are the typed `ExpertBankRefusal`, raised before any bank, CUDA slot or source read; the exact count reaches the cache through `MoeSlotCache::with_exact_slots` / `Engine::build_moe_cache_exact`, which refuse an already built cache or a count below eight instead of clamping. The `MEMRA_MOE_SLOTS` clamp itself is untouched (lead ruling 2, option A of `BUDGET-REFUSAL.md`). |
| CLI | `run_gen.rs` (`expert_bank_cli` after the path, install after load), `run_spec.rs` (same) | Parses the door and its budgets once; keys match exactly (`--expert-bank-host-bytes-x=1` is `unknown expert bank flag`, `--experts-via-tier=1` is `takes no value`); a budget flag without the door, a bare flag, junk, or a repeat is a usage error (`Error:`, exit 1). The helpers are reached through the `#[doc(hidden)] pub mod banked_residency` path; the crate root re-exports nothing from the gate. Installs the gate after load, before the first forward; run-gen refuses the door without the approved artifact. Refusal token contract: only the typed `ExpertBankRefusal` becomes the final stderr line `REFUSED: <reason>` with exit 2; every other installer error stays `Error:` exit 1. `pressure-refusal.py` is a red arm that must see that native token. |

Tests that exist: `crates/memra-tier/tests/bank/owner_proxy.rs` (four tests:
Send+Sync assertions plus `WrongOwner` from a spawned thread; pending bound,
foreign token, failed `finish` retained; dropping the owner invalidates the proxy
without auto-finishing open DMA; the token identity equals the fixture lease),
`crates/memra-tier/src/bank/owner_proxy.rs` unit tests (identity carried, a lying bank
refused and retired, a record without a dispatch id refused, three forged tokens foreign,
`dispatch_id` boundaries including the `u16::MAX` MTP key), `crates/memra-tier/tests/bank/day4.rs:413-422`
(`host_bank_plan` boundaries since day 43), `crates/memra-tier/tests/bank/day10.rs`
(`gpu_bank_slots` boundaries and overflow, typed refusal text and downcast, argv
parse including the exact-key red arms), `crates/memra-gguf/src/expert_banks.rs`
(plan-derived catalog equals the literal day-ten spelling on a qwen3_5_moe plan with an
MTP block; dense plan, missing tensor, duplicated tensor, shape mismatch, scale plane,
missing or duplicated contract entry, non-GGUF dialect each refuse), the compile
assertion above, and the receipt verifiers `verify-day8.py` / `verify-day9.py` /
`verify-day10.py` / `verify-day10-budget.py` / `verify-day11.py` with their red arms. No CUDA-free
test can construct `MoeSlotCache`, so `install_banked` and `admit_banked`
refusals are exercised only by native cells.

Evidence: `DAY18.md` (day-18 inputs for items 1, 3, 4 and 6, verbatim lines under each item below),
`DAY8.md` (rented 5090, bank budgets 8 GiB and 4 GiB, refusal cell),
`DAY9.md` (target card: default and 8 GiB, gen `MATCH`, spec K1..8
`SELF-CONSISTENCY PASS`, ON/OFF tapes identical, 12,091 / 63,996 GPU evictions).

## Days 40 to 58: the tuned program (each with its own pre-registration, commit and cell)

The door's program after the tuning of `OWED.md` C1 (days 40 to 50). Every row is `executed-not-qualified` until its
cell reads; the cells are listed in each day file and the deciding cell in `DAY51.md`.

| Day | Change | Where | Commit |
|---|---|---|---|
| 40 | Stage clock `--expert-bank-stages` (log only) and the attribution cell | `moe_cache.rs`, `native.rs`, `residency.rs` | `08210a291` |
| 43 | I6: the host tier planned per record size (no 16-record clamp, no 256 MiB ceiling; refused above 3/4 of `MemAvailable`), an O(1) host SLRU decision-identical to the VecDeque oracle, `collect_evicted` over a candidate list, the governor's release no longer O(outstanding charges) | `banked_residency.rs`, `slru.rs`, `residency.rs`, `governor.rs` | `ed3c8c5b4`, `51467de12` |
| 44 | I9: under the door the loader keeps expert banks as views of the artifact's mapping (no pinned write-combined duplicate); `Engine::set_expert_host_mapped`, `GgufFile::shard_mmap` | `model.rs`, `lib.rs`, `memra-gguf` | `c2d78fb9c` |
| 45 | I3, the host fill: `min(8, cores/2)` threads read and checksum every record off the owner thread; the owner admits each against the catalog digest into a free slot (`BankService::admit_filled`, `SlruPolicy::reserve_free`) | `native.rs`, `residency.rs`, `slru.rs` | `ef7db702e` |
| 46 | I1: no compute-stream drain on the miss path; leases retired in order on their copy events, at most 32 open | `moe_cache.rs`, `native.rs` | `a6258a8f0` |
| 47 | I2: one cached pinned pool per install, the tier's `HostBuffer` hook, reads straight into pooled buffers, the H2D a DMA from cached pinned memory | `host_buffer.rs`, `rows.rs`, `residency.rs`, `native.rs` | `10a1c30df` |
| 48 | I8: a memo of `(id, bytes)` pairs `validate` accepted; I5: the host-demand trace written in 64 KiB line-aligned chunks | `moe_cache.rs`, `native.rs` | `cd49c8bcb`, `ae5237e6c` |
| 49 | I7: the installer's per-expert compare and checksum on scoped threads after the serial SHA lock | `native.rs` | `14b2b9970` |
| 50 | I4: prefetch of the next routed expert through the owner (host-resident records only, copy stream, consumption after a compute-stream wait) | `moe_cache.rs`, `hybrid_forward.rs`, `lib.rs`, `native.rs` | `6745fd062` |
| 57 | I10: the installer admits the host fill to completion before decode (bounded) | `native.rs` | `70d6633f5` |
| 58 | I8f and I5f (the day-48 clauses failed): the validate memo as a dense table, the trace line written directly | `moe_cache.rs`, `native.rs` | `7ea765687` |
| 60 | `run-gen --moe-dispatch-clock` (log only, both programs): the dispatch and prefetch entry points bracketed, for the gap to REF | `moe_cache.rs`, `lib.rs`, `run_gen.rs` | `fec3c582f` |
| 61 | I11, the host-hit lease's repeated work removed: the catalog's memoized ticket allowance, one catalog and one host-cache read in `stage`, one SLRU lookup in `publish`, one id lookup in `demand`, the reused host slot (5370 to 3620 ns per host-hit prefetch cycle on the local CPU; a sixth change, one owner call, read flat and was reverted) | `types.rs`, `residency.rs`, `expert_dispatch.rs`, `native.rs` | `6116cddc2` to `59a5875d4`, `a068ee37d` |
| 61 | I12: finished leases retire where a lease is taken, not on every admission | `moe_cache.rs` | `117302725` |
| 63 | I13: the governor reserves and releases without temporaries, one body per `BankLease`, the retire side in one pending lookup, the SLRU's maps on a deterministic Fx hasher (3569 to 2940 ns per host-hit prefetch cycle on the local CPU); target card `i13=improves`, the door still `loses` to REF (0.264 against 0.255 s gen-only, 0.232 against 0.226 window), `DAY63.md` section 4 | `contracts.rs`, `governor.rs`, `residency.rs`, `expert_dispatch.rs`, `slru.rs` | `9bbab60ed` to `c9379c051` |
| 64 | I14: the catalog and the host cache found through an Fx-hashed index (orders, answers and refusals kept); I15: one ticket per prefetched expert (the door's three blocks reserved in order, leased by one owner call, staged in order, finished once); 2995 to 1997 ns per host-hit prefetched block on the local CPU; the first card cell (a 9950X host) read `flat`, `flat`, `matches` with noise set by the door's per-boot host-CPU bimodality (`DAY64.md` section 4), the rerun on the 285K class with an admissibility clause pending | `types.rs`, `residency.rs`, `fx.rs`, `expert_dispatch.rs`, `owner_proxy.rs`, `native.rs`, `moe_cache.rs`, `hybrid_forward.rs` | `8e7faf4ec`, `83f03d9b7`, `2243b1fe2` |

Every rung's RTX 5090 verdict is in its DAY file (I6's default-budget regression fixed on the tuned tree, `DAY43
RESIDFIX ... no_regression=PASS`; I8 and I5 pass as I8f and I5f, `DAY58 SMALLFIX ... i8f=PASS i5f=PASS`), and the
target card read the whole ladder (`DAY52.md` section 11). **The deciding cell** (`DAY51.md`), on the final tree
`62e848b1f`: on the target card `DAY51 VERDICT rig=pro-single integrity=ok -> door_wins` (gen-only decode 0.311 to
0.277 s, window 0.251 to 0.240 s against the naked legacy; REF, the legacy with its own default-OFF prefetch
`MEMRA_MOE_PREFETCH=1`, 0.255 and 0.226, faster than the door on both; install 9.91 s); the first run of the cell
was void on a trace term I10 contradicts (`DAY51.md` sections 3 and 1c). The RTX 5090's `decide-b` is queued behind
the card's reset. The door's decide-by is unchanged (2026-10-04); a target-card win is the owner's promotion call,
with `OWED.md` C2 as the promotion work. What the pending items below say about the synchronous miss path (item 4)
describes the program before day 46.

Lead integ60 owed items (`DAY59.md` to `DAY61.md`, all registered before code or cells): REF's own deciding cell
(`MEMRA_MOE_PREFETCH=1`, its `docs/FLAGS.md` row now carries decide-by 2026-10-04); the gap to REF attributed on both
programs with the day-60 clock; I11 and I12 against the door they tune and against REF (cell `i11`). On the target
card (BOX12, 2026-09-25, `pro-single-day61/`): REF qualifies as that card's naked default (`DAY59 VERDICT
rig=pro-single shape=pftime integrity=ok -> pf_wins`, gates PASS, `pfnaked -> pf_flat`); the gap is on the door's CPU
side, its prefetch path's owner demand the largest part (`DAY60 GAP rig=pro-single integrity=ok window:
wall_gap=+0.437 cpu_gap=+0.699 top=prefetch_ns cpu_side; ...`); `DAY61 VERDICT rig=pro-single integrity=ok
i11=improves i12=flat door=i12 vs_ref=loses (window: i11=improves i12=flat vs_ref=loses)`: the tuned door at 0.266 s
gen-only and 0.234 s window against REF's 0.255 and 0.226. The RTX 5090's cells are queued behind its reset (queue
v6); the next improvement registers in `DAY63.md`.

## What is pending before the door can sit behind the tiered materializer

Listed, not implemented. None is a "small default-OFF step with no new numeric
program": each either needs an `Engine` (CUDA) to test, awaits a lead decision,
or is a transport, so this document records them rather than speculating in code.

1. **Owner placement under PP.** The registry is thread-local and the proxy
   refuses `WrongOwner` off its thread. `Engine::with_moe_cache` can be entered
   from any thread that takes the Mutex, so a PP worker admitting through the
   bank refuses today. Usable behind the materializer needs one of: a bank owner
   registered per CUDA owner thread (per stage), or a typed owner-thread hand-off
   (channel or RPC) that keeps `ExpertBankOwner` `!Send`. Neither is small; the
   choice depends on the PP placement design the lead owns. Day 18 input (`DAY18.md`): the
   census of `with_moe_cache` callers (the MoE layer functions in `hybrid_forward.rs`, run on
   whichever thread walks the layer; under the stage-split scopes that is a stage thread) and the
   CPU proof `cargo test -p memra-tier --test bank owner_proxy` (4 passed, `WrongOwner` from a
   spawned thread for `validate`, `demand`, `with_bytes`, `finish`); no native cell exists because
   no PP gate binary carries the installer. Landed behavior under a stage split: a typed refusal
   before any demand or H2D.
2. **GPU-side budget refusal.** Landed on day ten as option A (lead ruling 2):
   `--expert-bank-gpu-bytes=N` (Budgets row above), `MEMRA_MOE_SLOTS` untouched.
   Cells and receipts: `DAY10.md`. Still open inside this item: the door budget
   is uniform-layout only (mixed layouts need per-class minima, option C's
   shape) and the refusal reads free VRAM at install time, so a cache built by
   an earlier forward would be refused rather than resized. Both stay as they
   are unless the door wins its decide-by.
3. **Installer generality.** First half landed on day eleven (`6db8ac122`, `DAY11.md`):
   `install_expert_bank_gate` takes its catalog from the compiled model plan and the
   model pack's GGUF tensor contract bound against the artifact census
   (`memra_gguf::expert_banks`, Catalog row above); it spells no checkpoint name and
   consults no architecture name, and a plan without MoE projections, a missing,
   ambiguous or shape-incompatible contract entry, and a scale plane the artifact
   carries are typed refusals. The bank for the approved artifact is byte-identical to
   day ten (catalog parity replayed by `verify-day11.py`, gen tape equal to the day-nine
   control). Still pending inside this item: the hash lock to one artifact, and **scale
   admission**: the installer refuses `.scale` / `.input_scale` census rows and loaded
   `macros` / `fp8_blk` planes (`experts-via-tier expert catalog refused: artifact
   carries expert scale planes the consumer does not declare: <names>` / `loaded bank
   <name> carries scale planes (macro or block scales) the native installer does not
   consume`) instead of banking scale-bearing records for the Hy3/Step ladder
   (`banked_residency.rs` already checksums scale planes on the tier side). Each new
   artifact and family is its own census, gates and receipts; the shared loader proves
   nothing. Day 18 input (`DAY18.md`, cell `hashlock`, both cards): the hash lock refuses a
   non-approved artifact with `Error: "experts-via-tier artifact SHA256 mismatch"` (exit 1, no door
   line) and the same artifact runs natively without the door (`MATCH`), verbatim `-> hash_lock_refuses`
   on the RTX 5090 (Qwen3.5-9B NVFP4 MTP) and on the target card (Qwen3.8-27B NVFP4 Q5K MTP); the
   scale-admission refusal has its CPU proof (`cargo test -p memra-gguf --lib expert_banks`, 10 passed,
   `a_scale_plane_on_a_bank_is_refused_by_name`) and no native cell (no scale-bearing artifact on
   either rig; shape pre-registered).
4. **Overlap.** `max_pending = 1`, `items: 1`, `tickets: 1`: every miss is
   synchronous (demand, H2D, full `stream().synchronize()`, finish), and
   `prefetch_source` returns `false`. Speed behind the materializer needs several
   leases in flight retired on copy-stream events instead of a compute-stream
   drain. Performance item; needs interleaved receipts on both rig classes before
   any default, and it must not reintroduce the publish-before-completion hole
   `d0acf6f03` closed. Day 18 input (`DAY18.md`, cell `overlap`, target card, N=5 per arm per
   order, both orders, one lock hold, the day-nine 8 GiB pressure shape): verbatim
   `... decode_off_s=0.408 decode_on_s=2.343 decode_ratio=5.743 ratio_o1=5.694 ratio_o2=5.833 ...
   door_cost_ms_per_decode_token=60.47 ... door_cost_ms_per_staged_MB=1.390 ... install_on_s=74.84
   ... -> sync_miss_path_slower` (tape identical, `DAY18 REPLAY overlap: PASS (9 checks)`). The item
   is confirmed as the promotion blocker at this budget; the synchronous path is the door's only
   miss path, so there is no separate arm to delete for it.
5. **Refused arms.** `install_banked` requires the untouched default SLRU, so
   LFU, size-aware classes, pread backends, and frozen residency all refuse with
   the bank installed. Hy3 mixed layouts need size-aware classes; that
   combination is refused today and stays refused until a receipt says otherwise.
6. **Serving shape.** The door exists in `run-gen` and `run-spec` only;
   `memra-server` has no installer and no serving-shape gate. The one numeric
   program per request rule requires a serving-shape bit-identity gate (banked vs
   native, solo vs batched) before any promotion. Day 18 input (`DAY18.md`, cell `serverdoor`, both
   cards): `memra-server` booted with `--experts-via-tier` on its argv serves the native program and
   prints no door line and no refusal (`flag_silently_accepted=true`), verbatim
   `-> door_unreachable_in_serving`; the server consults argv for `--version` and the key-lifecycle
   flags only. The serving-shape identity gate this item requires cannot exist before a serving
   installer does; the silent acceptance is a hygiene finding for the review.

## Decision at decide-by

Either the pending items 1 to 3 are landed with their gates and the door is
promoted per the flags doctrine, or the door is deleted in one PR: the `banked*`
fields, `install_banked`, `admit_banked`, the guards, `banked_residency/`,
`host_bank_plan` and the whole budget layer (`ExpertBankBudget`,
`expert_bank_cli`, `gpu_bank_slots`, `ExpertBankRefusal`, `refusal_reason`,
`MoeSlotCache::with_exact_slots`, `Engine::build_moe_cache_exact`), the CLI
arms and their refusal mapping, the owner registry if nothing else uses it,
`memra_gguf::expert_banks` if the installer is still its only consumer,
their tests, and a "Removed doors" ledger row pointing at `DAY8.md`, `DAY9.md`,
`DAY10.md` and `DAY11.md`. `SLOT_TAIL_PAD_BYTES` moves back into `moe_cache.rs` as
the naked constant it names. The compile assertion on `Engine: Send + Sync` stays
either way.
