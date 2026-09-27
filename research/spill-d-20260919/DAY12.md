# WP-D day 12: the two refused arms through lane A's rule seams (#552, criterion 2)

Repository: **memra**, branch `lane/spill-d-20260919`. Start: tip `15bd53152` (= origin). Merges,
in order: `origin/main` `a1bd0be62` (`ffffd84b1`), `origin/lane/spill-a-20260919` `432816926`
(`5a7eb8517`, the two rule seams), `origin/lane/spill-integ9-20260921` `10c1d05da` (`c0655f249`,
the lead's holed-cache taint on my day-11 drop site), then `origin/main` `5ecfd262c` after #584
merged (`4c94a123e`, the replay tolerance for extra evidence rows). Code commit `55f242e98`; the
data commit carries this file. Both pushes were refused by the pre-push perf-ci arm, verbatim
`pre-push: engine files touched after the last perf-ci battery.` with `base (merge-base with
refs/remotes/origin/lane/spill-d-20260919): 15bd53152721e1a84bb7c240621ea77c76218155` and nine
engine files (my three `bin/kv_tier_gate/` files plus `tier_transfer_gate.rs`, `dsv4_graph.rs`,
`hybrid_forward.rs`, `lib.rs`, `model_memory.rs`, `tier_transfer.rs` inherited through the
merges). No `--no-verify`, no skip variable: the lane stays local and the lead pushes.

## What changed in the contract vocabulary

Day 11 had three outcomes for a fault arm: `FAULT-ARM PASS <arm>`, a failed cell (`fault arm <arm>
did not prove its contract; missing=[..] failed=[..]`), and a typed refusal for the two arms whose
expectation had no seam in the contracts (`Arm::refusal`). Day 12 removes the third: lane A's day-11
rules (`crates/memra-tier/src/conformance/recovery.rs`, unversioned, frozen schedules byte-identical)
gave both arms their seam, so the arms drive the seams and a backend without one fails the seam rows.
`fault_contract::Arm::refusal` and the `REFUSED` branch of `verdict` are gone; `restores_cache` is
now false only for the two arms that lose a K plane (`corrupt-host`, `missing-host`) and `continues`
equals `restores_cache`. The old refusal texts survive in three places only, all red arms or records:
the day-11 receipts (`pro-single-day11/cancel-restore/`, `pro-single-day11/require-resident/`,
untouched), `verify-day11.py` (frozen day-11 vocabulary) and the reclaim test
`day11_refusal_receipts_are_failed_cells_under_the_day12_rule`, which replays those receipts under
the day-12 rule and requires `missing=[recover-source, ..]` / `missing=[continuation-gate-on-suspended-cache, ..]`
with every recorded row still holding: the absence of a seam is now a failure the receipt names.

Required rows added (`fault_contract::Arm::required_checks`):

| Arm | Seam | New required rows (expected) |
| --- | --- | --- |
| `cancel-restore` | rule 1, `TransferEngine::recover_source` (native `CudaTransfers::recover_source`, `holds_cancelled_source`) | `retire-holds-source` `Err(Busy)`; `retire-source-holds` `Err(Busy)`; `recover-source` `Ok(host)`; `recovered-source-checksum` (hex of `checksum(host.bytes())` equals the bundle's sealed checksum); `recovered-source-intact` (`StateBundle::verify` `Ok(())`); `recover-source-once` `Err(AlreadyReleased)`; `cancel-after-recovery` `Err(AlreadyReleased)`; `pinned-held-by-recovered-lease` (pinned bytes after `retire`+`acknowledge` equal the bytes before the hold); `restored-identical` `true`. Kept: `cancel`, the three `Err(Cancelled)` rows, `retire`, `acknowledge`, `retake-demoted-copy` `Err(AlreadyReleased)`, `budget-zero` |
| `require-resident` | rule 2, `Cache::suspend_layer` / `resume_layer`, `SuspendedLayers`, `ContinuationRefused` | `suspended-register` (ascending ids of the taken layers equal `cache.suspended.layers()`); `continuation-gate-on-suspended-cache` and `continuation-gate-asked-twice` `Err(ContinuationRefused { path: "kv-tier-gate continuation", layers: [<all 16>] })`; `continuation-gate-after-partial-resume` (after the first `resume_layer`, the other 15); `register-empty-after-resume` `true`; `continuation-gate-after-resume` `Ok(())`. Kept: `budget-zero`, `restored-identical` |

The rule-1 rows are one generic sequence over `TransferEngine`
(`fault_contract::cancel_restore_revoke`, `cancel_restore_recover`, `cancel_restore_retire`),
recorded by the native arm between its backend-only rows and by a new CPU binding in the contracts
harness (`crates/memra-tier/tests/contracts/fault_arm_bindings.rs`) against lane A's fake transport:
`legacy = false` records every row green and the recovered lease reads the untouched bytes
`[3, 20, 37]`; `legacy = true` is the red arm (`retire-holds-source` observed `Ok(())`,
`recover-source` observed `Err(Unsupported)`, verdict `fault arm cancel-restore did not prove its
contract; ...`, never `REFUSED`, never PASS). To reuse that fixture rather than copy it, three of its
items became `pub(super)` (`pending_restore`, `finish`, `owner`); A's rule text and schedules are
untouched.

## How the arms drive the seams (`kv_tier_gate/fault.rs`, `active.rs`)

`cancel-restore`: `HostPlane::detach_source` splits the demoted copy from its bookkeeping
(`DetachedPlane`, `active.rs`); the arm submits the H2D itself, observes completion, cancels before
`ready_view`, records the three `Err(Cancelled)` rows, then the hold (`retire`, `retire_source`
`Busy`), `recover_source(ticket, 0)` once, the checksum and `StateBundle::verify` over the recovered
bytes, the once-only and cancel-after-recovery refusals, `retire` and `acknowledge`, the pinned
accounting, the D2H twin's `AlreadyReleased`, releases the unpublished destination, and
`DetachedPlane::reattach(host)` gives the same demoted plane back to the roundtrip's own
`active::restore`. The layer is whole again, the driver captures `restored-prefix` and the arm
continues under the same tokenwise program. When `recover_source` does not answer `Ok`, the arm
returns `None`, the layer stays out (holed, tainted) and the verdict fails on the seam rows: that is
the shape of the red arm in the binary, never a refusal.

`require-resident` and every other arm: the demote loop takes each layer with
`cache.suspend_layer(i)` (was `cache.kv[i].take()`) and the restore loop returns it with
`cache.resume_layer(i, layer)` (was `cache.kv[i] = Some(layer)`), so the typed register is the
state for the whole interval and `ensure_usable` is the continuation gate; the holed arms keep the
day-11 `mark_tainted` and the `holed-cache-refuses-continuation` row (the register also still
names the lost layer, recorded as `observation.holed_register_layers`). The `require-resident`
arm records the register, asks the gate twice while everything is suspended, once after the first
resume and once after the last. `capture` refuses any captured graph pool, and the day-11 prefix
capture passed, so `suspend_layer`'s pool drop is a no-op in this gate and `restored-identical` is
unaffected.

The shared `active::roundtrip` (lane B's series path) keeps raw `take()`. Switching it was left for
the lead's placement, as A's DAY11.md says: `suspend_layer` drops captured graph pools, and that
roundtrip's receipt is a free-memory series (`reclaim-cycles.tsv`, the ruling-6 label) whose frozen
32k receipts were produced with raw `take()`; the fault arms have no such surface, so they moved.

## Checks run locally (all exit 0, under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`)

`DOCS_RS=1 cargo clippy -p memra-engine -p memra-tier -p memra-kv --offline --all-targets -- -D warnings`
(``Finished `dev` profile [unoptimized + debuginfo] target(s) in 17.50s``); `cargo test -p memra-tier
--offline --test reclaim --test contracts` (`test result: ok. 32 passed` and `test result: ok. 68
passed`, the new `day12_*` bindings and `day11_refusal_receipts_are_failed_cells_under_the_day12_rule`
among them); `cargo fmt --all -- --check`; `git diff --check`; `tools/check-flags.sh`
(`check-flags: no uncovered runtime names`); `bash -n` and `shellcheck` on the three new scripts;
`python3 -m py_compile verify-day12.py`. No new `MEMRA_*` read, no `.cu` change, no `unsafe`, no
numeric-program change, no frozen schedule or rule text touched. No local GPU cell (the local RTX
5090 carried the lead's jobs).

## Cells on one RTX PRO 6000 Blackwell (N=1 each, `pro-single`, `/tmp/memra-gpu.lock`, 600/600 W)

Driver `run-day12.sh` (build receipt, then `run-fault-arms.sh` for all seven arms, then
`run-transfer-gate.sh`, then the root validation), detached under tmux on the single-PRO clone at
`55f242e98` (branch `lane-d-day12`, synced by git bundle because the push is blocked, clean tree).
Build receipt `pro-single-day12/build/`: build 0, clippy 0 (`--release --all-targets -D warnings`
over engine, tier, kv), tests 0 (kv and tier CPU suites), `kv-tier-gate`
`02e72db9b54323109609e2f94948a33f04d4073470dd9a623262890a1c334264`, `tier-transfer-gate`
`7fd972dbd5ea89482a4b024a791b84df39eb63f4b56864e811d6d345cac89f5a`. Every cell binds to that
source and those binaries (`verify-day12.py` refuses a cell bound to a non-green build). Collector:
`tools/tier-battery.py --rig pro-single --timeout 1800 --out <cell> --execute env
NVIDIA_TF32_OVERRIDE=0 <kv-tier-gate> --artifact <Qwen3.8-27B NVFP4/Q5K GGUF> --case active --context
8192 --tiers host --same-program --kv-allocator pooled --fault <arm> --out <receipt>` per arm and
`--timeout 700 --execute <tier-transfer-gate> <case>` per transfer case. Every cell launched on its
first attempt (`launched-attempt` 0); compute-app snapshots before and after every cell show no
co-tenant; no D job, tmux session or lock remains on the clone. Receipts: native
`/root/spill-receipts/d-day12/`, mirrored here under `pro-single-day12/` with each cell's `--validate`
output (`validate.json`), the root validation (`pro-single-day12/validate.json`: `cells 9,
failed_commands 0, refused_commands 0, qualification false`) and the local re-validation of the mirror
(`day12/validate-mirror.json`, identical counts). Telemetry: 3,752 samples at 250 ms, GPU 32 to 51 C,
power draw 33 to 362 W under the 600/600 W envelope. Collector scope per cell (not a timing claim):
arms 126 to 135 s, `conformance` 0.3 s, `roundtrip` 13.6 s. The five continuing arms' `final-logits.f32le`
are stored gzip-compressed, as on day 11.

| Arm | Injection point (contract call) | Expected | Printed last line (verbatim) | Checks | Receipt |
| --- | --- | --- | --- | --- | --- |
| `cancel-demote` | `TransferEngine::cancel` after the D2H `synchronize`, before `take_destination` | intact-resident; take refuses, budget zero, source returns, continuation matches | `FAULT-ARM PASS cancel-demote committed=8192 generated=128` | 8 | `pro-single-day12/cancel-demote/` |
| `cancel-restore` | `TransferEngine::cancel` after the H2D `synchronize`, before `ready_view`; rule 1 from there | hold, once-only hand-back of the demoted copy, clean retirement, same plane restored, continuation matches | `FAULT-ARM PASS cancel-restore committed=8192 generated=128` | 17 | `pro-single-day12/cancel-restore/` |
| `corrupt-host` | `CudaPinnedLease::write` (Busy under the live D2H ticket; Ok after its drain), then `restore` | `Corrupt` before any device call; no publish; no token; holed cache refuses | `FAULT-ARM PASS corrupt-host committed=8064 generated=0` | 6 | `pro-single-day12/corrupt-host/` |
| `missing-host` | untaken D2H ticket retired and acknowledged; `take_destination` at restore | `UnknownTicket`; pinned released; no publish; no token; holed cache refuses | `FAULT-ARM PASS missing-host committed=8064 generated=0` | 6 | `pro-single-day12/missing-host/` |
| `host-budget-short` | governor pinned capacity 239,468,543 B (whole state minus 1); `admit_whole_state` | `Capacity` before any layer is taken; nothing copied; continuation matches | `FAULT-ARM PASS host-budget-short committed=8192 generated=128` | 5 | `pro-single-day12/host-budget-short/` |
| `device-short` | competing tenant reserves the governor's device dimension down to one byte short; restore's `alloc_device` | `Capacity`; host copy verifies; competitor released; continuation matches | `FAULT-ARM PASS device-short committed=8192 generated=128` | 5 | `pro-single-day12/device-short/` |
| `require-resident` | every plane through `Cache::suspend_layer`; `ensure_usable` asked while suspended, again, after the first `resume_layer`, after the last; rule 2 | typed refusal naming exactly the suspended layers until every layer is back; `Ok(())` after; continuation matches | `FAULT-ARM PASS require-resident committed=8192 generated=128` | 8 | `pro-single-day12/require-resident/` |

Every recorded row held (`checks_failed` empty in all seven `FAULT-ARM.txt`; the two holing arms carry
the extra `holed-cache-refuses-continuation` `Err` row). Verbatim answers worth keeping, from
`cancel-restore/receipt/fault-checks.tsv`: after `cancel` `Ok(PublicationRevoked)` and the three
`Err(Cancelled)` publication rows, `retire` `Err(Busy)`, `retire_source` `Err(Busy)`,
`recover_source(ticket, 0)` `Ok(host)`, the recovered bytes' checksum
`1058a34a5f763dea085ce1aa280fabb82820006e6312111e38cad69413a3312e` equal to the bundle's sealed
checksum (8,773,632 valid bytes, layer 3 K), `StateBundle::verify` `Ok(())`, a second
`recover_source` `Err(AlreadyReleased)`, `cancel` after the hand-back `Err(AlreadyReleased)`,
`retire` and `acknowledge` `Ok(())`, pinned 239,468,544 B before the hold and after the
acknowledgement (the charge stayed with the recovered lease), `take_destination` on the D2H twin
`Err(AlreadyReleased)`, budget `TierBudget::zero(1)` after the roundtrip's own `restore` drained the
D2H entry. From `require-resident/receipt/fault-checks.tsv`: the register `[3, 7, 11, 15, 19, 23, 27,
31, 35, 39, 43, 47, 51, 55, 59, 63]`, `ensure_usable("kv-tier-gate continuation")` twice
`Err(ContinuationRefused { path: "kv-tier-gate continuation", layers: [3, 7, .., 63] })`, after the
first `resume_layer` the same error naming the other fifteen (`[7, 11, .., 63]`), after the last
`register-empty-after-resume` `true` and `Ok(())`. The day-11 observation on the same cache and the
same call was `Ok(())`.

Continuation identity: the five continuing arms match the frozen target-card 8k bundle
(`research/spill-b-20260919/BOX3-BASELINES.json`, source `8ef562b12`) on all seven surfaces
(`prefix-state.tsv`, `final-state.tsv`, `tokens.u32le`, `logits.tsv`, `final-logits.f32le`,
`prompt.u32le`, `plan.debug`); every arm's suspended prefix manifest equals the bundle's; the five
restoring arms report `restored-identical true`, `cancel-restore` now among them.

### `tier-transfer-gate` (lane A's native bindings, first native run of the two day-11 rule lines)

`pro-single-day12/transfer-gate/conformance/`: 13 `PASS` lines, exit 0, among them verbatim
`PASS rule cancelled-restore-recovers-source native CUDA` and
`PASS rule cancel-refused-after-source-consumed native CUDA`, ending on
`PASS native governor zero after controlled drain`. `pro-single-day12/transfer-gate/roundtrip/`:
6 lines, every one `PASS native D2H-H2D roundtrip bytes=<n> N=1 ... byte_exact=true
source_freed_host_live=true handback_no_copy=true governor_zero=true` (4 KiB to 256 MiB), the
last verbatim `PASS native D2H-H2D roundtrip bytes=268435456 N=1
expected_sha256=6020d865b68ed263bbee3a2556e6c064906410f5e220a584827034fdbc72191b
actual_sha256=6020d865b68ed263bbee3a2556e6c064906410f5e220a584827034fdbc72191b byte_exact=true
source_freed_host_live=true handback_no_copy=true governor_zero=true`. These two cells are the
native evidence A's DAY11.md said was missing; the day-9 lines are no longer the last.

## Offline replays (all exit 0)

`verify-day12.py` (`day12/verify-day12.log`): `DAY12 FAULT-ARM REPLAY MATCH: 7 arms; 7 PASS, 0
refusals; N=1 each; NOT qualification` and `DAY12 TIER-TRANSFER-GATE REPLAY MATCH: 2 cells; both
day-11 rule lines printed natively; NOT qualification`. `cargo test -p memra-tier --test reclaim`
after the mirror: `test result: ok. 32 passed`, with `committed_day12_receipts_replay_their_pass_lines`
now replaying all seven day-12 receipts, `committed_day11_receipts_replay_their_pass_lines` the five
unchanged arms from day 11, and `day11_refusal_receipts_are_failed_cells_under_the_day12_rule` the two
day-11 refusals. `verify-day11.py` is unchanged and still replays day 11 under its own vocabulary.

## Notes for the lead

1. `active::roundtrip` (lane B's series path) still detaches with raw `take()`; the seam move there
   is the shared-program placement A left to the lead (reasoning above). Until it moves, the series
   path's `decode_step_h` continuation is gated exactly as before (register empty).
2. The holing arms now leave both a register entry and the taint. A lost K plane is permanent, so
   the one-way taint is the right state; the register entry is recorded
   (`observation.holed_register_layers=[3]`) and `ensure_usable` answers the taint text first.
3. Three items of lane A's CPU fixture (`tests/contracts/transfer.rs`) became `pub(super)` so the
   day-12 binding could reuse it instead of copying it: `pending_restore`, `finish`, `owner`. No
   rule text, schedule or behaviour changed.
4. All seven arms were rerun, not only the three named, because the detach path they share moved
   onto `suspend_layer`/`resume_layer`; the four other arms' vocabulary is unchanged and their
   day-11 receipts still replay.

## Scope

Nine N=1 cells on one target-class card, 8k context, pooled allocator, development evidence
(`executed-not-qualified`); no G-gate advanced, no default, flag or support state changed, no
cross-box timing, no local 5090 cell. About 2 agent-hours against the 6-hour budget.
