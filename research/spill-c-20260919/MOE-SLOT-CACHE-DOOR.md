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
token), `1de17d41f` (day-ten cell driver and verifier).

decide-by: 2026-10-04 (covers the door and both budget flags; CLI doors carry
their decide-by here, not in `docs/FLAGS.md`).

| Surface | File | What it does |
|---|---|---|
| Owner registry | `crates/memra-tier/src/bank/owner_proxy.rs` | `ExpertBankOwner` (`!Send` via `PhantomData<Rc<()>>`) registers a `Box<dyn ExpertDispatchBank>` in a thread-local `OWNERS` map. `ExpertBankProxy` (`Clone`, `Send + Sync`: a `ThreadId` and a `u64`) and `ExpertLeaseToken` (`Send + Sync`, two `u64`s, no pointer) are the only things that leave the owner thread. `access` refuses `WrongOwner` off-thread; `close` refuses `Busy` with an open lease; `Entry::drop` forgets pending backing on unknown completion instead of releasing it. Not RPC: a migrated caller refuses. |
| Bank contract | `crates/memra-tier/src/bank/expert_dispatch.rs` | `ExpertDispatchBank { validate, demand, finish }` and `SlruExpertDispatch` over `BankService`. |
| Cache fields | `crates/memra-engine/src/moe_cache.rs` `MoeSlotCache` | `banked: Option<ExpertBankProxy>`, `banked_pending: Option<ExpertLeaseToken>`, `banked_evictions: u64`. No service, `Rc`, or byte backing is stored in the cache. |
| Install | `moe_cache.rs::install_banked` | Refuses unless the cache is an untouched default SLRU: no resident table, no pending or worker reads, no LFU, not frozen, no pread backend, zero staged bytes, no prior bank. |
| Admission | `moe_cache.rs::admit_banked` | `validate` then hit check, else `demand` one lease, `with_bytes` borrows the payload on the owner and runs the unchanged `admit_native` H2D, `stream().synchronize()`, `finish`. One pending lease at a time; a second demand or a frozen cache refuses. `admit_native` publishes a banked slot only after an explicit stream sync; an unknown result leaves the slot outside every table and queue. |
| Guards | `moe_cache.rs` `admit`, `dispatch_source`, `force_admit`, `prefetch_source`, `restage_block`, `remove_occupant`, `Drop` | Admit and dispatch route through the bank; prefetch returns `false` (no detached legacy prefetch may bypass the owner); freeze-profile restage refuses; evictions under the bank are counted; `Drop` finishes a pending lease only after a successful stream drain and otherwise retains it. |
| Compile assertion | `moe_cache.rs`, `const _: fn()` after `impl Drop` | Ordinary library build (not a test) asserts `Engine: Send + Sync`, `MoeSlotCache: Send`, `ExpertBankProxy: Send + Sync`, `ExpertLeaseToken: Send + Sync`. The cache is `Send` only because its pread receiver is `!Sync`; `Engine` holds it in `Mutex<Option<MoeSlotCache>>` (`lib.rs`), which is what makes `Engine: Sync` hold. Putting an owner-only bank or lease into the cache fails here, before the scoped PP-worker spawns cascade. |
| Native installer | `crates/memra-engine/src/banked_residency/native.rs` `Engine::install_expert_bank_gate(model, gguf, budget: ExpertBankBudget)` | Takes the typed CLI budget as a parameter (no argv or env scan). Hash-locks the already-open inode to `APPROVED_SHA` `df27a780…7adf` (Qwen3.6-35B-A3B-UD-IQ4_XS). Requires one GGUF shard, `MEMRA_MOE_CACHE` enabled, at most one MTP head. Refuses `dev_exps`, `step_ep`, `step_tp`, `glm5_ep`, `glm5_tp_split` and any scale plane (`macros`, `fp8_blk`). Builds Governor, `BankService` (per-record catalog), `SlruPolicy`, `SlruExpertDispatch`, a `TracedDispatch` that prints `[expert-host-slru] key= bytes= slot= hit= victim=`, and registers the owner with `max_pending = 1`. Returns `BankedExpertGate` (`!Send`, held on the CUDA thread for the run); its `Drop` prints `[expert-gpu-slru] slots= allocated_bytes= evictions=` and `[experts-via-tier] physical_reads= owner_close=`. |
| Budgets | `crates/memra-engine/src/banked_residency.rs`: `ExpertBankBudget { host_bytes, gpu_bytes }`, `expert_bank_cli`, `host_bank_slots` / `host_bank_budget`, `gpu_bank_slots(bytes, max_record, hard_bytes)` / `gpu_bank_budget`, `ExpertBankRefusal`, `refusal_reason` | Host (`--expert-bank-host-bytes=N`, default 256 MiB): refuses below one record and above the 256 MiB qualification ceiling, clamps to 16 records. GPU (`--expert-bank-gpu-bytes=N`, default unset = native sizing untouched): exact slot count `N / (max_record + 8)`; refuses below eight slots, above the machine hard ceiling (`moe_cache::hard_slot_bytes`, the same value the native constructor applies), on checked overflow, and when `MEMRA_MOE_SLOTS` is also set (a conflict, never a silent loser). Both refusals are the typed `ExpertBankRefusal`, raised before any bank, CUDA slot or source read; the exact count reaches the cache through `MoeSlotCache::with_exact_slots` / `Engine::build_moe_cache_exact`, which refuse an already built cache or a count below eight instead of clamping. The `MEMRA_MOE_SLOTS` clamp itself is untouched (lead ruling 2, option A of `BUDGET-REFUSAL.md`). |
| CLI | `run_gen.rs` (`expert_bank_cli` after the path, install after load), `run_spec.rs` (same) | Parses the door and its budgets once; a budget flag without the door, a bare flag, junk, or a repeat is a usage error (`Error:`, exit 1). Installs the gate after load, before the first forward; run-gen refuses the door without the approved artifact. Refusal token contract: only the typed `ExpertBankRefusal` becomes the final stderr line `REFUSED: <reason>` with exit 2; every other installer error stays `Error:` exit 1. `pressure-refusal.py` is a red arm that must see that native token. |

Tests that exist: `crates/memra-tier/tests/bank/owner_proxy.rs` (three tests:
Send+Sync assertions plus `WrongOwner` from a spawned thread; pending bound,
foreign token, failed `finish` retained; dropping the owner invalidates the proxy
without auto-finishing open DMA), `crates/memra-tier/tests/bank/day4.rs:413-422`
(`host_bank_slots` boundaries), `crates/memra-tier/tests/bank/day10.rs`
(`gpu_bank_slots` boundaries and overflow, typed refusal text and downcast, argv
parse), the compile assertion above, and the receipt verifiers `verify-day8.py`
/ `verify-day9.py` / `verify-day10.py` / `verify-day10-budget.py` with their red
arms. No CUDA-free
test can construct `MoeSlotCache`, so `install_banked` and `admit_banked`
refusals are exercised only by native cells.

Evidence: `DAY8.md` (rented 5090, bank budgets 8 GiB and 4 GiB, refusal cell),
`DAY9.md` (target card: default and 8 GiB, gen `MATCH`, spec K1..8
`SELF-CONSISTENCY PASS`, ON/OFF tapes identical, 12,091 / 63,996 GPU evictions).

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
   choice depends on the PP placement design the lead owns.
2. **GPU-side budget refusal.** Landed on day ten as option A (lead ruling 2):
   `--expert-bank-gpu-bytes=N` (Budgets row above), `MEMRA_MOE_SLOTS` untouched.
   Cells and receipts: `DAY10.md`. Still open inside this item: the door budget
   is uniform-layout only (mixed layouts need per-class minima, option C's
   shape) and the refusal reads free VRAM at install time, so a cache built by
   an earlier forward would be refused rather than resized. Both stay as they
   are unless the door wins its decide-by.
3. **Installer generality.** `install_expert_bank_gate` is hash-locked to one
   artifact, hard-codes `blk.N.ffn_{gate,up,down}_exps.weight`, and refuses scale
   planes. Behind the materializer it must take its catalog from the model plan
   and `tensor_contract` (semantic tensor ids, required shapes, quant layouts)
   and admit scale-bearing records for the Hy3/Step ladder (`banked_residency.rs`
   already checksums scale planes on the tier side; the native installer does
   not consume them). Each new artifact and family is its own census, gates and
   receipts; the shared loader proves nothing.
4. **Overlap.** `max_pending = 1`, `items: 1`, `tickets: 1`: every miss is
   synchronous (demand, H2D, full `stream().synchronize()`, finish), and
   `prefetch_source` returns `false`. Speed behind the materializer needs several
   leases in flight retired on copy-stream events instead of a compute-stream
   drain. Performance item; needs interleaved receipts on both rig classes before
   any default, and it must not reintroduce the publish-before-completion hole
   `d0acf6f03` closed.
5. **Refused arms.** `install_banked` requires the untouched default SLRU, so
   LFU, size-aware classes, pread backends, and frozen residency all refuse with
   the bank installed. Hy3 mixed layouts need size-aware classes; that
   combination is refused today and stays refused until a receipt says otherwise.
6. **Serving shape.** The door exists in `run-gen` and `run-spec` only;
   `memra-server` has no installer and no serving-shape gate. The one numeric
   program per request rule requires a serving-shape bit-identity gate (banked vs
   native, solo vs batched) before any promotion.

## Decision at decide-by

Either the pending items 1 to 3 are landed with their gates and the door is
promoted per the flags doctrine, or the door is deleted in one PR: the `banked*`
fields, `install_banked`, `admit_banked`, the guards, `banked_residency/`,
`host_bank_slots` and the whole budget layer (`ExpertBankBudget`,
`expert_bank_cli`, `gpu_bank_slots`, `ExpertBankRefusal`, `refusal_reason`,
`MoeSlotCache::with_exact_slots`, `Engine::build_moe_cache_exact`), the CLI
arms and their refusal mapping, the owner registry if nothing else uses it,
their tests, and a "Removed doors" ledger row pointing at `DAY8.md`, `DAY9.md`
and `DAY10.md`. The compile assertion on `Engine: Send + Sync` stays either way.
