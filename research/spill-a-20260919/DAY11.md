# WP-A day 11: lane D's fault-arm findings become contracts (lead ruling 9)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Start: tip `d9d424398`, merged
`origin/main` `ea08bc7f8` (`1a1cbd0d8`) and `origin/lane/spill-d-20260919` `15bd53152`
(`be43012e7`). Rule commits: `42e53f3a9` (rule 1), `e52635715` (rule 2); the receipts commit that
adds this file is the branch tip. Raw receipts: `day11/`.

Lead ruling 9, verbatim: "Fault-arm findings become contracts, not patches. Findings (1) and (3)
are contract gaps in `memra_tier`: a cancelled restore must be recoverable or refused before the
H2D source is consumed, and a continuation over a suspended layer must be refused at
`ensure_usable`. Lane A (contracts owner) day 11: write both as typed contract rules with red arms
in the frozen-schedule style (schedules unchanged; new rules only), then D reruns `cancel-restore`
and `require-resident` for the PASS lines. No `decode_step_h` change until the contract exists."

## Push

`git push origin lane/spill-a-20260919` after the two merges was refused by the pre-push perf-ci
gate: `pre-push: engine files touched after the last perf-ci battery.` with
`base (merge-base with refs/remotes/origin/lane/spill-a-20260919): d9d4243985e68113f984c8224b12b18a18778fbd`
and thirteen `crates/memra-engine` files, all arriving through the two merges (`banked_residency.rs`,
`banked_residency/native.rs`, the six `bin/kv_tier_gate*` files, `bin/run_gen.rs`, `bin/run_spec.rs`,
`lib.rs`, `moe_cache.rs`, `spec/prime.rs`). No `--no-verify`, no skip variable; the lead pushes.
This lane's own engine changes (`tier_transfer.rs`, `bin/tier_transfer_gate.rs`,
`hybrid_forward.rs`, one struct literal) would be refused the same way.

## Where the rules live

`crates/memra-tier/src/conformance/recovery.rs`, re-exported from `memra_tier::conformance`
beside the frozen schedules, unversioned: the task allowed v1.4 "only if the frozen-schedule format
requires a version for new rules"; the format is one module per revision plus `pub use`, so a rule
module beside them needs no version, `WIRE_VERSION` stays 1, and the freeze ledger entry is the
lead's. The three frozen revision files are byte-identical before and after
(`git rev-parse <sha>:crates/memra-tier/src/conformance/revision_v1{1,2,3}.rs` at `be43012e7` and
at `e52635715`: `cd144f3b1c06261f9fd91ffe9e26328d8dbd43b8`, `981bcc811603d0d357674daebe466302f740757e`,
`50928b759262d2a83762157c9fd026c41fa633d6`); `conformance/mod.rs` gained only the five-line
include. The v1 schedules in `mod.rs` are untouched.

## Rule 1: cancelled restore

Rule, verbatim from the task: a restore (H2D) that is cancelled before its consumer event completes
must either hand the untouched host source back to the caller as a typed lease (recoverable) or
refuse the cancel with a typed error while the source is still intact; consuming the source and then
cancelling is forbidden by the rule.

D's finding, verbatim: "No H2D-source recovery after `cancel`: the demoted copy's only caller
handle is consumed at submission and released by `retire`; the D2H twin is take-once
(`AlreadyReleased`). A cancelled restore can only be drained."

Seam (`TransferEngine::recover_source(&mut self, ticket, item) -> Result<Self::Host>`, default
body `Err(Error::Unsupported)`, the v1.3 `retire_source` shape) and the answers the rule fixes:

| Call | State | Answer |
|---|---|---|
| `recover_source` | publication not revoked (live or published) | `NotReady` (live: the source is the ticket's) / `AlreadyReleased` (published: consumed) |
| `recover_source` | cancelled, producer pending or observation unknown | `Busy` / `Quarantined`, ownership and charge retained |
| `recover_source` | cancelled, producer observed, source consumer or source graph pin live | `Busy` |
| `recover_source` | cancelled, producer observed, source side idle | `Ok(host)`, exactly once; the lease keeps its own pinned charge |
| `recover_source` | second call, or after the source left any other way | `AlreadyReleased` |
| `retire`, `retire_source` | cancelled and an H2D source not yet recovered | `Busy`: the engine never drains a cancelled restore |
| `cancel` | a source already left the ticket (`retire_source`, `retire`, recovery) | `AlreadyReleased`: consuming then cancelling cannot happen |
| `recover_source` | D2H item, rejected item, unknown ticket | `Unsupported`, `Rejected`, `UnknownTicket` |

A backend without the seam keeps the default refusal and must then refuse `cancel` on an
unpublished H2D (`Unsupported`) while the source is intact, never revoke and drain it. The
production CPU transport (`io::transfer::CpuTransfers`) has no H2D route at all, so it keeps the
default; `services.rs` asserts `recover_source` answers `Unsupported` without a charge mutation,
beside the v1.3 `retire_source` assertion.

Native: `CudaTransfers::recover_source` (`crates/memra-engine/src/tier_transfer.rs`) takes the
`Item.host` lease out and marks `source_retired`, after `progress` (observed producer, quarantine
on a lost observation) and `source.idle()` (source graph pins, source consumer event);
`holds_cancelled_source` guards `retire` and `retire_source`; `cancel` refuses once any item's
source is retired or the ticket is retired. No new numeric program, no new allocation, no change
to acknowledged or retired tickets, no `unsafe`.

Schedules: `transfer_cancel_recovers_source` (recoverable outcome; producer pending, live source
consumer and source graph pin; the five refusal steps, the hold, the once-only hand-back, the
refused second cancel, normal retirement; returns the lease so the binding checks bytes and charge)
and `transfer_cancel_refused_after_source_consumed` (the forbidden order).

Red arm, before and after. The CPU fake transport (`tests/contracts/transfer.rs`) gained a
`legacy` switch that is the transport before the rule: no recovery, drains at `retire`, grants a
revocation after the source left. `day11_red_arm_legacy_transport_drains_a_cancelled_restore`
runs the schedule against it under `catch_unwind` and requires the panic, then replays the finding
literally: `cancel` `Ok(PublicationRevoked)`, `recover_source` `Err(Unsupported)`, `retire`
`Ok(())`, `acknowledge` `Ok(())`, and the host's pinned charge is gone with the entry (nothing
left to hand back). With the seam (`legacy = false`) `day11_cancelled_restore_recovers_its_source`
and `day11_cancel_is_refused_after_the_source_left` pass, the recovered lease reads the untouched
bytes `[3, 20, 37]`, still holds its charge (`release` `Err(Busy)` until the lease drops) and the
governor returns to zero. All three: `ok` in `day11/gates/test-tier-kv.log`.

First battery run on the working tree, before the rule commits, verbatim: `test result: FAILED. 58
passed; 2 failed` in the `contracts` target, both `assertion failed: !engine.retired(&ticket).unwrap()`
(`recovery.rs:98:5` and `:133:5`). Cause: the fake derives `retired()` from its completion flags,
so once every flag is set it reports retired without a `retire` call; the native `retired` is a
state `retire` sets. Fix on the fixture side: the fake's `retired()` honours the cancelled-source
hold (whole retirement cannot have happened while the caller's source is held), and the
forbidden-order schedule dropped its one `!retired` assertion after `retire_source` (v1.3's
`transfer_source_retirement` already holds that invariant). The first run's log was overwritten
by the rerun on the committed source; the two panic lines above are the record.

Native bindings (`crates/memra-engine/src/bin/tier_transfer_gate.rs`): new
`cancelled_restore_recovers_source` (producer held by the native host function, `pin_source_graph`,
a real `record_source_consumer` event; prints `PASS rule cancelled-restore-recovers-source native
CUDA`) and `cancel_refused_after_source_consumed` (prints `PASS rule cancel-refused-after-source-consumed
native CUDA`). The v1 `transfer_cancel` completion hook, the v1.1 `transfer_complete_cancel`
drain, the `transfer_lifetime` `Graph` step and the acceptance batch drain now assert `retire`
`Err(Busy)`, recover every accepted item (the rejected sibling answers `Err(Rejected)`), then
retire; the schedule functions they call are unchanged. Not run natively today (no GPU cell; lane B
holds the card): the day-9 lines remain the last native evidence, and these two lines are
unrecorded until the gate reruns on a card.

## Rule 2: required-resident continuation

Rule, verbatim from the task: a continuation (decode or prime) over a cache with any suspended
layer must be refused at `Cache::ensure_usable` (`crates/memra-kv`) with a typed error naming the
suspended layers, unless the caller restores first; `decode_step_h` is NOT changed (ruling 9), it
simply never sees a suspended layer once `ensure_usable` refuses.

D's finding, verbatim: "`Cache::ensure_usable` returned `Ok(())` on the fully suspended cache
(observed), `decode_step_h` unwraps a suspended layer (`hybrid_forward.rs`),
`RestoreDecision::RequireState` is load-vs-recompute only."

Seam (`crates/memra-kv/src/lib.rs`): `Cache.suspended: SuspendedLayers`, a typed register
(`BTreeSet<usize>` behind `is_empty`, `contains`, `layers`, `FromIterator`); `Cache::suspend_layer(il)
-> Result<KvLayer, SuspendError>` takes the layer out and registers it (`NotResident`,
`AlreadySuspended`); `Cache::resume_layer(il, layer) -> Result<(), SuspendError>` puts it back and
clears the entry (`NotSuspended`, `Occupied`); `Cache::ensure_usable` returns
`Box<ContinuationRefused { path, layers }>` (ascending) while the register is non-empty, after the
existing taint check, signature unchanged so all 30 callers compile. `suspend_layer` drops the
three captured graph pools (`glm5_decode_graph`, `glm5_tp_sym_graph`, `qwen_prime_graph`) because a
restored plane may land at another address and the cache's own rule is to drop them in any seam
that replaces a state buffer; the bytes are untouched, no numeric program changes. `decode_step_h`
(`decode.rs:779`) is unchanged. The pp split (`hybrid_forward.rs` `PrimeCacheStages::new`) gives each
stage cache the parent's entries for its layers, so a stage gate refuses like the parent's would.

Schedule: `required_resident_continuation<F: ContinuationGateFixture>(f, layers)`: a whole cache
continues; one suspended layer refuses and is named; asking is not restoring; every suspended layer
is named ascending; a partial restore still refuses, naming what is left; a restored cache
continues, twice. The fixture trait (`suspend`, `restore`, `continuation -> Result<(), Vec<u32>>`)
keeps `memra-tier` free of `memra-kv` types (the dependency runs the other way).

Red arm, before and after. `tests/contracts/resident_bindings.rs`: `TaintOnly` is the gate before
the rule (a one-way pipeline taint flag that knows nothing about residency);
`day11_red_arm_taint_flag_lets_a_suspended_cache_continue` requires the schedule to panic on it and
then records the observed answer, `Ok(())` on a suspended cache. `Register` (a `BTreeSet`) passes
`day11_suspended_layers_refuse_continuation_until_restored`. On the real `Cache`
(`memra-kv` `continuation_gate_tests`, CPU-built, no device): `day11_ensure_usable_refuses_a_suspended_cache_until_restored`
binds the schedule with sixteen layers, suspending 3, 0 and 15, and downcasts every refusal to
`ContinuationRefused` with `path == "decode_step_h"`; `day11_refusal_is_typed_and_names_the_layers`
checks `layers == [1, 2]`, the exact message `prime_cache: continuation refused: layers [1, 2] are
suspended on a tier and must be restored first`, and taint precedence;
`day11_seam_refusals_on_a_cache_without_resident_planes` checks `NotResident` (layer 0, layer 9),
`AlreadySuspended`, `layers()`, `contains` and the stage filter. All five: `ok`. A `KvLayer` cannot be
built without a device, so the positive `suspend_layer`/`resume_layer` roundtrip is native: the
kv-tier-gate under D's rerun.

The kv-tier-gate's existing roundtrip (restore, then continue) stays green: `active::roundtrip` and
`fault::run` detach with a raw `cache.kv[i].take()` and reattach with `cache.kv[i] = Some(layer)`,
which never touches the register, so their `decode_step_h` continuation is gated exactly as before.
That is also why they cannot show the new refusal until they use the seam (below).

### `ensure_usable` callers (memra-engine, memra-server)

30 call sites (34 with the four in the new `memra-kv` tests) in `memra-engine` and `memra-kv`, none in `memra-server` (grep). Every
continuation entry `memra-server/src/worker.rs` calls reaches the gate through its entry function:
`decode_step` (via `decode_step_h`), `decode_step_h`, `decode_step_chain`,
`decode_step_glm5_tp_device_logits`, `prime_cache` and `prime_cache_overlaid` (via
`prime_cache_overlaid_inner`), `prime_cache_batch` (via `prime_cache_batch_inner`),
`decode_step_batch_sampled_lean_masked` and `_scheduled` (via
`decode_step_batch_sampled_lean_masked_schedule`, which also fronts `decode_step_batch_hyper` and
`step35_decode_batch_layers`), `generate_spec_session_*` (prime through `prime_cache_*`, step through
`spec_target_step_h` / `decode_step_t_h_emb_dev`, restore through `spec_session_from_restored_deferred`,
`spec_flush_pending`), `mtp_prime_start` (and its `mtp_prime_walker` after it), the glm5 prime source
(`glm5_prime.rs` `new`), `pp::restore_cache_checkpoint` (target and source), `Cache::snapshot`,
`snapshot_refresh`, `rollback`, and the kv-tier-gate's `capture`.

Paths that continue without calling it, all outside the server:

1. `GraphSession::step` and `graph_session_recapture` (`decode.rs:59`, `:2505`): the session's
   cache is gated once at `graph_session_from_cache_masked`; a layer suspended after creation would
   replay over its baked pointers. Reached only by the bins `graph_warmup_stress` and the graph
   benches; `worker.rs` has no `graph_session` use.
2. `prime_graph_run` (`prime_graph.rs:140`): bin `prime_graph_gate` only.
3. DSV4 (`dsv4_gpu.rs` `decode_step*`, `spec_*`, served by `dsv4_serve.rs`) and qwen4exp
   (`qwen4exp_gpu.rs`) run on their own `DecodeState`, not `memra_kv::Cache`; the register does not
   exist there. A tier program for those families needs its own continuation gate; not added.
4. `PrimeCacheStages` (`hybrid_forward.rs:233`) moves layers between the parent and its stage
   caches with raw `take()`: a move, not a suspension; stage caches now copy the parent's register.
5. The kv-tier-gate's `active::roundtrip` and `fault::run` (above): raw `take()`, the register
   stays empty, the finding's `Ok(())` reproduces until they use `suspend_layer`/`resume_layer`.

No server code changed; the list is a report.

## What D's reruns need (D's files, not changed here)

`cancel-restore` (`fault.rs` `cancel_restore`): after the three `Cancelled` checks, check `retire`
`Err(Busy)` (the hold), `recover_source(&ticket, 0)` `Ok(host)` with `bundle.verify(&[host.bytes()])`
`Ok(())` (source intact), then `retire` and `acknowledge` `Ok(())`, rebuild the `HostPlane` with the
recovered `host` and the D2H `ticket`, and restore it through `active::restore` so the layer is whole
again (`restores_cache` true, `restored-identical`, continuation). `retake-demoted-copy` on the D2H
twin still answers `AlreadyReleased` and is no longer the outcome. `fault_contract::Arm::refusal`
loses the `CancelRestore` line once the required checks are updated.

`require-resident` (`fault.rs` `run`): detach every plane's layer with `cache.suspend_layer(i)`
instead of `cache.kv[i].take()`, reattach with `cache.resume_layer(i, layer)` instead of
`cache.kv[i] = Some(layer)`; the check `continuation-gate-on-suspended-cache` expects
`Err(ContinuationRefused { layers: [<the 16 full-attention layers>] })` (downcast, or its message)
while suspended and `Ok(())` after the restore. `active::roundtrip` can take the same two calls so
lane B's series carries the contract too; that is a shared-program change for the lead to place.

## Gates (all under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`, runner `day11/run-gates.sh`, logs `day11/gates/`)

Run on the committed source `e52635715` (a first run on the working tree had the two fixture
failures recorded above).

| gate | exit | verbatim tail |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | (no output) |
| `cargo test -p memra-tier -p memra-kv --offline` | 0 | 11 suites, `passed` sums to 303, `failed` to 0; the eight `day11_*` tests `ok` |
| `DOCS_RS=1 cargo clippy -p memra-engine -p memra-tier -p memra-kv --offline --all-targets -- -D warnings` | 0 | ``Finished `dev` profile [unoptimized + debuginfo] target(s) in 12.05s`` (second run, warm) |
| `bash tools/check-flags.sh` | 0 | `check-flags: no uncovered runtime names` |
| `git diff --check` | 0 | (no output) |

No new `MEMRA_*` read, no `.cu` change, no flag row.

## BOX3 native build check (build only, no GPU cell; lane B holds the card)

`/root/wt-a` synced by git bundle (`1adf2be3..lane/spill-a-20260919`, 142 commits, 13,492,555 bytes,
streamed over the existing ControlMaster socket, bundle deleted after the fetch) to `e52635715`
on branch `lane-a-day11`, `git status --short` empty. Under `nice -n 19`, `-j 8`, `--offline`:
`cargo build --release -p memra-engine --bin kv-tier-gate` exit 0,
``Finished `release` profile [optimized] target(s) in 3m 00s`` (a first attempt exited 127,
`nice: 'cargo': No such file or directory`, the non-interactive PATH; rerun through
`/root/.cargo/env`); `--bin tier-transfer-gate` exit 0 (bin relink only, the library was built).
Binaries: `kv-tier-gate` sha256 `fc5c8ad10889820634c9331b78bec76b04bd5878e9347b12ad6901c655b3387f`,
`tier-transfer-gate` sha256 `63393dbc6d2a267cdb42e9ef4e3ce2788465d88e1c775130acd71beb5ecd098a`
(`MEMRA_CUDA_ARCH auto-detected 120a`). Neither binary was run. Receipt
`day11/box3-build/build.log`. `/root/memra-spill` and the artifacts were not touched.

## Scope

Contract rules and CPU red arms only. Not done today: any native run of the two new gate lines or
of D's two arms (D's task; no GPU cell, lane B holds the card), the shared `active::roundtrip` move
to the seam (lead's placement), a continuation gate for the `DecodeState` families, the freeze
ledger entry (lead). About 5 agent-hours against the 6-hour budget.
