# integ6 self-review (lead, 2026-09-20)

Read in full: the 17-file crate diff vs `main` `847168642` (A `tier_transfer.rs`, `tier_transfer_gate.rs`; B
`memra-kv/lib.rs`, `plane.rs`, `kv_tier_gate.rs`, `active.rs`; C `banked_residency.rs`, `native.rs`, `moe_cache.rs`,
`lib.rs`, `run_gen.rs`, `run_spec.rs`; tier tests). Docs and receipts spot-checked against the verifiers.

## Findings
1. **Serving path untouched.** `CudaTransfers`, `KvAllocator::Vmm` and `Cache::new_with_allocator` have no caller in
   memra-server; every existing `Cache` constructor passes `KvAllocator::Pooled`; `MoeSlotCache::new` is
   `build(e, max, None)` with the old sizing logic; `hard_slot_bytes` is the same formula moved into one function.
2. **A, per-side retention.** `Retention { graph_pins, consumer_event }` for source and destination; `retire_source`
   needs the source side idle, `retire` needs both. `pin_graph` (legacy) pins both sides. `release_device` and
   `take_plane` now refuse (`Busy`) while any live ticket binds the allocation, without a stream wait (the comment
   explains the deadlock it avoids). D2H requires an exclusive host allocation at submit and at execution
   (`Rc::get_mut`), so a taken destination cannot be re-used as a mutable target; H2D may read it immutably. The taken
   host lease shares one `PinnedAllocation` with the entry until `acknowledge`, which removes the entry
   (`tier_transfer.rs:1081`) and makes the caller's lease exclusive again; the native gate asserts the post-ack
   `write()` succeeds. Frozen schedules unchanged; `pin_source_graph` refuses after source retirement.
3. **B, allocator policy.** `KvDev::alloc_vmm_u8` defaults to `REFUSED:` (no pooled substitute; unit-tested
   `allocator_dispatch_has_no_bootstrap_or_fallback`); Engine implements it with `KvPlane::vmm`. The gate builds the
   cache directly with the policy (no empty-plane swap) and writes `allocation-construction.txt`.
   `probe_demoted_va_release` unmaps retained chunks without releasing handles, frees and re-reserves the VA at the
   original address (refuses and frees the stray reservation otherwise), remaps; `Mapping.reserved` guards `Drop`.
   Gate rule tightened: `reclaimed` requires `residual == 0`, so a classified nonzero residual is reported but never
   labelled PASS. Stricter than criterion (d); recorded, not relaxed (ruling 5 says when it may be revisited).
4. **C, typed budgets.** `install_expert_bank_gate(model, gguf, budget)` reads no argv or env for the budget;
   `expert_bank_cli` fails closed on a bare flag, a repeat, junk, or a budget without the door. Only
   `ExpertBankRefusal` becomes `REFUSED: <reason>` exit 2 in both gate binaries; all other errors stay `Error:` exit 1.
   `build_moe_cache_exact` refuses if a cache exists, `with_exact_slots` refuses below 8 and never clamps. New env read
   `MEMRA_MOE_SLOTS` (conflict refusal) is an existing FLAGS row; census green. `TracedDispatch` unwraps are gone.
5. **Nits, not blocking.** `expert_bank_cli` matches by `starts_with`, so `--expert-bank-host-bytesX=1` is reported as
   the host flag's usage error (still fails closed). `gpu_slot_bytes` hard-codes the 8-byte tail pad the cache uses;
   a shared constant would keep them from drifting. `pub use banked_residency::{expert_bank_cli, …}` exposes gate CLI
   parsing from the engine library; acceptable because the gate binaries live in the same crate.

## Verification
`INTEGRATION-DAY10.md` §integ6: CPU battery green; local 5090 correctness battery green on every gate that has its
model on this rig (kernel-check 109 cells, decode-batch, graph, serve-smoke, serve-stress; memra-server 513 tests),
run-gen/run-spec/VERIFY-GATE/prime/accept SKIPPED for absent models, hit-gate r3/g2 = main's #379 regression. Target-card
evidence per lane in their `DAY*.md`. BOX3 global `--validate`: 32 cells, qualification=false, exit 0.

## Tool halves
revuto and Bugbot as available on the PR; merging on green CI with this self-review as the review (owner ruling 2026-09-16).
