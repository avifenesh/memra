# WP-D day 11: active-KV fault arms on the frozen tier contracts (#552, criterion 2)

Repository: **memra**, branch `lane/spill-d-20260919`. Merged `origin/lane/spill-d-20260919`
(`a3aff2ae0`) and `origin/main` (`ea08bc7f8`) into `9872c1469`; the merge tree is byte-identical to
main's tree. The push of that merge was refused by the pre-push perf-ci arm ("engine files touched
after the last perf-ci battery", merge-base `a3aff2ae0`, nineteen engine files inherited from other
lanes). No `--no-verify`, no skip variable: the lane stays local and the lead pushes. Local commits
on top of the merge: `bda5d3c91` (the arms), `688b40468` (clippy fix), `b2962bac0` (TESTING rows,
verifier) and the data commit that carries this file.

## What the arms are

`kv-tier-gate --case active --same-program --kv-allocator pooled --fault <arm>` runs the unchanged
active roundtrip (`kv_tier_gate/active.rs`: the same tokenwise `decode_step_h` program, the same
`demote`/`restore` contract calls) and injects exactly one fault at one documented contract call on
the first full-history K plane (layer 3 of Qwen3.8-27B, 8,773,632 valid bytes = 8,064 tokens x 1,088
B, allocation 8,912,904 B; 16 full-attention layers, 239,468,544 B of whole-state pinned bytes).
The contract's answers are the evidence: each arm records `check / expected / observed` rows
(`fault-checks.tsv`) and the pure rule `fault_contract::verdict` prints `FAULT-ARM PASS <arm>` only
when every required check was recorded and held. A differing answer or a missing check is a failed
cell; the two arms whose expectation has no seam in the contracts end in a typed refusal, never
PASS. A passing arm whose cache is whole and bit-identical runs the same 128-token continuation; an
arm whose cache is incomplete writes no token (a layer whose K plane did not come back never
re-enters the cache). Design notes and the per-arm rows: `docs/TESTING.md`, kv-tier-gate section.

Two roundtrip changes apply to every run, not only the arms: a whole-state admission through the
governor's own `reserve` before any layer is taken (`active::admit_whole_state`, the
`TierStore::admit` rule "reserve before moving bytes"), and `StateBundle::verify` as the restore
integrity check (typed `Corrupt` before any device call, replacing a string comparison). No new
`MEMRA_*` read, no numeric-program change, no `unsafe`, no second cancellation or integrity
mechanism: `TransferEngine::cancel`, `CudaPinnedLease::write`, `StateBundle::verify`,
`take_destination` and `BudgetGovernor::reserve` are the contracts' own.

## Cells on one RTX PRO 6000 Blackwell (N=1 each, `pro-single`, `/tmp/memra-gpu.lock`, 600/600 W)

Collector: `tools/tier-battery.py --rig pro-single --timeout 1800 --out <cell> --execute env
NVIDIA_TF32_OVERRIDE=0 <bin> --artifact <Qwen3.8-27B NVFP4/Q5K GGUF> --case active --context 8192
--tiers host --same-program --kv-allocator pooled --fault <arm> --out <receipt>`, driven by
`run-fault-arms.sh` (bounded lock retry; a launched cell is never rerun). Every attempt launched on
its first try (`launched-attempt` = 0). Receipts: native `/root/spill-receipts/d-day11/`, mirrored
here under `pro-single-day11/<arm>/` with each cell's `--validate` output (`validate.json`) and the
root validation (`pro-single-day11/validate.json`: `cells 7, failed_commands 0, refused_commands 2,
qualification false`). Telemetry: 3,680 samples at 250 ms, GPU 32 to 50 C, power draw 33 to 362 W
under the 600/600 W envelope; each cell took 126 to 135 s (collector scope, not a timing claim).
Compute-app snapshots before and after every cell show no co-tenant.

| Arm | Injection point (contract call) | Expected | Printed last line (verbatim) | Receipt |
| --- | --- | --- | --- | --- |
| `cancel-demote` | `TransferEngine::cancel` after the D2H `synchronize`, before `take_destination` | intact-resident; cancel revokes, take refuses, budget zero, source returns, continuation matches | `FAULT-ARM PASS cancel-demote committed=8192 generated=128` | `pro-single-day11/cancel-demote/` |
| `cancel-restore` | `TransferEngine::cancel` after the H2D `synchronize`, before `ready_view` | typed refusal: the contract has no seam to recover the H2D source | `REFUSED: cancel-restore revoked publication, but the transfer contract has no seam to recover the H2D source after cancellation; no tokens, budget drained` | `pro-single-day11/cancel-restore/` |
| `corrupt-host` | `CudaPinnedLease::write` (Busy under the live D2H ticket; Ok after its legal drain), then `restore` | restore refuses `Corrupt` before any device call; no publish; no token | `FAULT-ARM PASS corrupt-host committed=8064 generated=0` | `pro-single-day11/corrupt-host/` |
| `missing-host` | untaken D2H ticket retired and acknowledged; `take_destination` at restore | `UnknownTicket`; pinned released; no publish; no token | `FAULT-ARM PASS missing-host committed=8064 generated=0` | `pro-single-day11/missing-host/` |
| `host-budget-short` | governor pinned capacity 239,468,543 B (whole state minus 1); `admit_whole_state` | `Capacity` before any layer is taken; nothing copied; continuation matches | `FAULT-ARM PASS host-budget-short committed=8192 generated=128` | `pro-single-day11/host-budget-short/` |
| `device-short` | competing tenant reserves 2,138,570,745 B of the governor's device dimension (headroom 8,912,903 B); restore's `alloc_device` | `Capacity`; host copy verifies; no publish; competitor released; continuation matches | `FAULT-ARM PASS device-short committed=8192 generated=128` | `pro-single-day11/device-short/` |
| `require-resident` | `Cache::ensure_usable` asked while all 16 planes are demoted | typed refusal: no continuation-time required-resident contract | `REFUSED: require-resident has no contract today: Cache::ensure_usable accepts a suspended cache, decode_step_h unwraps a suspended layer, and tier RestoreDecision::RequireState is a load-versus-recompute rule` | `pro-single-day11/require-resident/` |

Every recorded check held (`checks_failed` empty in all seven `FAULT-ARM.txt`). Verbatim answers
worth keeping: `cancel` `Ok(PublicationRevoked)` on both the D2H and the H2D ticket;
`take_destination` / `ready_view` / `with_destination` after cancel `Err(Cancelled)`;
`CudaPinnedLease::write` under the live D2H ticket `Err(Busy)`, after the drain `Ok(())`;
`restore` on the flipped copy `Corrupt` (byte index 4,386,816); `take_destination` on the removed
copy `Err(UnknownTicket)`; on the acknowledged-but-taken D2H twin `Err(AlreadyReleased)`;
`admit_whole_state` and the restore `alloc_device` under the short budgets `Err(Capacity)`;
`Cache::ensure_usable` on the suspended cache `Ok(())`; the transfer budget after every drain
`TierBudget::zero(1)`. Accounting in `cancel-restore`: pinned stayed at 239,468,544 B after the
H2D `retire` (the D2H entry still owned the copy) and fell to 230,694,912 B only after the D2H
`acknowledge`; the copy was never reachable in between.

Continuation identity: the three continuing arms match the frozen target-card 8k bundle
(`research/spill-b-20260919/BOX3-BASELINES.json`, source `8ef562b12`) on all seven surfaces
(`prefix-state.tsv`, `final-state.tsv`, `tokens.u32le`, `logits.tsv`, `final-logits.f32le`,
`prompt.u32le`, `plan.debug`), and every arm's suspended prefix manifest equals the bundle's
`ce48492d782d34cb4562cd11ca6d8b610f43efd50137f36da50f93f08b49dace`. The four arms that restore the
cache report `restored-identical true`. Offline replays: `verify-day11.py`
(`day11/verify-day11.log`: `DAY11 FAULT-ARM REPLAY MATCH: 7 arms; 5 PASS, 2 typed refusals; N=1
each; NOT qualification`) and the Rust harness `crates/memra-tier/tests/reclaim/fault.rs`
(`committed_target_card_receipts_replay_their_verdicts`, 30 tests in the `reclaim` target).

## Native build receipts

`pro-single-day11/build/` (first attempt): build 0, **clippy 101** (`large_enum_variant` on the
fault `Slot` enum, 816 B), tests 0. Retained as the record of the failure. `pro-single-day11/build-2/`
(after `688b40468` boxed the variant): build 0, clippy 0, tests 0 (295 memra-kv and memra-tier tests),
binary `644d2baa29435c4db3cb0a2931384bd75b459a92db1126ee87b672e03268fe59`, clean tree. The single-PRO
clone built branch `d-day11` = `origin/main` plus the two lane patches applied with `git am` (its commit
ids differ from the local lane because the local lane sits on the merge commit; the tree ids are
identical, checked with `git rev-parse HEAD^{tree}` on both sides). All seven cells bind to build-2's
binary and source; `verify-day11.py` refuses a cell that binds to a non-green build.

## Findings for the lead (contract seams missing)

1. **Cancellation recovery.** `TransferEngine::cancel` revokes publication and `retire` releases the
   owned inputs, but the H2D source handle (the demoted copy) is consumed at submission and there is
   no seam to take it back; the D2H entry's twin is take-once (`AlreadyReleased`) and lives only
   until its acknowledgement. A cancelled restore therefore leaves the copy physically alive and
   charged but unreachable; the only legal outcome is a drain. A tier store that holds its own host
   handle (or a `Rejected`-style return of the inputs on cancel) would close this.
2. **Device capacity.** The governor has no post-construction capacity seam (`Governor::capacity` is
   private; no setter), so "shrink the device pool for the restore" was exercised through the
   governor's own admission with a second tenant. That is the contract's seam for exhaustion; a
   device-pool shrink of the CUDA pool itself does not exist and was not invented.
3. **Required state.** No continuation-time required-resident contract exists. `Cache` has only the
   one-way, pipeline-scoped taint flag (`ensure_usable` returned `Ok(())` on the fully suspended
   cache, observed); `decode_step_h` reaches `cache.kv[il].as_mut().unwrap()`
   (`crates/memra-engine/src/hybrid_forward.rs`), a panic rather than a refusal; the tier's
   `RestoreDecision::RequireState` / `PrefetchDecision::Refuse` (`memra-kv/src/tiered/policy.rs`) decide
   load versus recompute for mandatory state, not whether a continuation may run. The gate's "no
   token during the interval" is a driver invariant. Not added here, as ordered.
4. **Whole-state admission** is new behaviour in the shared roundtrip (lane B's series path included):
   a pinned budget below the demoted bytes now refuses before any copy instead of failing at the
   plane that overflows. With the 2 GiB gate capacity it never fires outside the arm.

## Checks actually run (all exit 0)

Local, under `systemd-run --scope -p CPUQuota=1200% -p MemoryMax=28G`: `cargo test -p memra-tier
--test reclaim` (30 tests, 12 new: door, red arm per arm, receipt replay), `cargo fmt --all --
--check`, `git diff --check`, `python3 -m py_compile` (verifier, collector), Python battery suite
(85 tests), `bash -n` and `shellcheck` on both new scripts, `tools/check-flags.sh` (864 names, none
uncovered), `tools/docs-registry-census.sh`, `tools/check-public-boundary.py check` on the staged
tree. On the clone: the build-2 receipt above. Not run: engine or server exactness batteries, any
timed comparison, the 32k context, the VMM door (refused by the CLI for these arms by design), a
second card class. The local RTX 5090 was busy with the lead's perf battery and took no cell.

## Scope

Seven N=1 cells on one target-class card, 8k context, pooled allocator, development evidence
(`executed-not-qualified` or `refused`); no G-gate advanced, no default or support state changed,
no cross-box timing. About 3.5 agent-hours against the 8-hour budget.
