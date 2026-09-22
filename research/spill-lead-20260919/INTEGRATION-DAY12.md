# Integration: day 12 (`lane/spill-integ9-20260921`, from `main` `ea08bc7f8` = #579 merged)

## Owner order (2026-09-21 ~00:00 local)
"fix it, fix it, there's open issues that are the pre-DS work, need to do them first. red need fix." Read as: the
open pre-DeepSeek issues (the generic tier/KV/serving backlog) come before any V4.1 work, and the red on this rig
gets fixed, not overridden.

## The red, fixed (`lane/perfci-main-20260921`, PR #583)
- The hit-gate regression (#379, r3/g2) was already fixed on `main` by #578 (21:58Z). `tools/local-ci.sh --perf` on
  `main` `ea08bc7f8` then stopped at the GPU-only lib suite: `test result: FAILED. 21 passed; 3 failed`, the three
  being pair-only tests (`dsv4_graph::tests::replay_rust_partial_submission_and_capture_cleanup`,
  `model_memory::native_tests::glm_peer_admission_materialization_trim_and_refill`,
  `glm_indexed_mla_kda_state_materialization`) that need two CUDA devices and fail with
  `DriverError(CUDA_ERROR_INVALID_DEVICE, "invalid device ordinal")` on this one-card rig
  (`integration-day11/perfci-main/local-ci-perf-attempt1-pair-tests-red.log`).
- Fix: `memra_engine::test_support::skip_unless_native_pair` prints `SKIP-PAIR <test> needs 2 CUDA devices, found N`
  and returns on a one-device rig (unchanged on the pair box); local-ci runs the stage with `--show-output` and prints
  `local-ci: SKIP 3 pair-only GPU test(s) on this rig (need 2 CUDA devices):` with the names. Second run then failed
  only `26b-spec-d1736: FAIL (no reading)`: the 26B MTP drafter is not staged here and the runner discarded
  gemma-gate's stderr. Fix: a spec cell without its drafter or ranks file is `SKIP (no draft at ...)`; a cell with no
  reading prints its last output. Third run: correctness GREEN, serve-smoke 0 failed, `SPEC-ON-CACHE-HIT GATE: ALL
  GREEN (qwen)`, lib suite 24 passed with the three explicit skips, `perf stage: 0 fail, 0 warn`, `rc=0`. Rows
  appended to `research/tune-data/perf-ci.jsonl` (26b-plain-short 208.85 and 208.70 tok/s, qwen9b-plain-short 139.09
  and 138.88 tok/s, `window_clean:true`). The perf-ci freshness gate on this rig is satisfied again; no further
  `MEMRA_SKIP_PERF_CI=1` after #583 lands and the worktrees pull it.

## Pre-DS issue triage (open issues, tier/KV/serving; the lead's queue)
Done: #574 closed (lane A day 10, #579). In flight: #445 §1, #346, #523 item 4 (lane B day 13: eviction must move
driver free; serving-shape gate plus fix); #552 criterion 2 (lane D day 11, below). Queued under the two-agent cap:
#552 criterion 1 (native HostPrefix and bank dispatch owners through the shared contracts), #523 items 1 and 3
(newest turn fits; 8-turn twin gate), #545 (CI executes tier, KV and onboarding CLI suites), #384 and #385 (host tier
cap evaporation; pinned arena startup), #536 (copy engine off the tick; largest), #577 and #562 (MoE lockstep
divergence, gemma MoE cuBLAS m-dependence; MoE program correctness, adjacent to the other Claude session's #565).

## Lane D day 11 (`15bd53152`, pushed by the lead with the logged override; #552 criterion 2)
Seven fault arms on `kv-tier-gate --case active --same-program --fault <arm>` (usage error anywhere else; unknown
arm `REFUSED`), each injected at one documented contract call, each ending in `FAULT-ARM PASS <arm>` or a typed
`REFUSED:`; never tokens from a wrong state. Target card, N=1, 8k, pooled, collector-locked, 600/600 W
(`research/spill-d-20260919/pro-single-day11/<arm>/`):
```text
cancel-demote      FAULT-ARM PASS cancel-demote committed=8192 generated=128
cancel-restore     REFUSED: cancel-restore revoked publication, but the transfer contract has no seam to recover the H2D source after cancellation; no tokens, budget drained
corrupt-host       FAULT-ARM PASS corrupt-host committed=8064 generated=0
missing-host       FAULT-ARM PASS missing-host committed=8064 generated=0
host-budget-short  FAULT-ARM PASS host-budget-short committed=8192 generated=128
device-short       FAULT-ARM PASS device-short committed=8192 generated=128
require-resident   REFUSED: require-resident has no contract today: Cache::ensure_usable accepts a suspended cache, decode_step_h unwraps a suspended layer, and tier RestoreDecision::RequireState is a load-versus-recompute rule
--validate: cells 7, failed_commands 0, refused_commands 2
verify-day11.py: DAY11 FAULT-ARM REPLAY MATCH: 7 arms; 5 PASS, 2 typed refusals; N=1 each; NOT qualification
```
The three continuing arms match the frozen 8k target-card bundle on all seven surfaces; every arm's suspended prefix
equals the bundle's. Findings for the lead (seams missing, recorded not added): (1) no H2D-source recovery after
`cancel` (the demoted copy's caller handle is consumed at submission, the D2H twin is take-once); (2) the governor has
no post-construction capacity seam (device exhaustion exercised through a second tenant); (3) no continuation-time
required-resident contract (`ensure_usable` returns Ok on a fully suspended cache, `decode_step_h` unwraps a
suspended layer); (4) for B: the shared roundtrip now admits the whole state via `governor.reserve` before any layer
is taken, and restore uses `StateBundle::verify` (typed `Corrupt`).

## Lead rulings, day 12
9. **Fault-arm findings become contracts, not patches.** Findings (1) and (3) are contract gaps in
   `memra_tier`: a cancelled restore must be recoverable or refused before the H2D source is consumed, and a
   continuation over a suspended layer must be refused at `ensure_usable`. Lane A (contracts owner) day 11: write
   both as typed contract rules with red arms in the frozen-schedule style (schedules unchanged; new rules only),
   then D reruns `cancel-restore` and `require-resident` for the PASS lines. No `decode_step_h` change until the
   contract exists.
10. **Pre-DS order.** The queue above is worked in the listed order under the two-agent cap; V4.1 work waits.

## Batteries on integ9 (`integration-day12/integ9-cpu-battery/`)
fmt, `cargo test -p memra-tier -p memra-kv -p memra-gguf` (all rc=0), clippy `-D warnings`, check-flags, publish
census, docs registry census, collector pytest, perf board, `git diff --check`: all rc=0. D's own: 30 Rust + 85 Python
tests, native build receipts on BOX3 (clippy clean, 295 tests, binary hash bound), seven cells with `--validate`.

## Lane B day 13 (`1fef60006`, pushed by the lead without an override once #583 landed; #445 §1, #346, #523 item 4)
`tools/prefix-evict-reclaim-gate.py` (serving shape, real `memra-server`, plain path, 48k-token prompt seeds a
1,592,160,256 B prefix entry, ballast child makes the card short, P2 arrives beside a busy peer). Target card, 600 W,
N=1, verbatim:
```text
base main ea08bc7f8: [admit-oom] reclaim-on-defer: evicted 2 prefix entries + ... effective free 6603MB -> 6603MB
                     PREFIX-EVICT-RECLAIM: entry_bytes=1592160256 reclaim_credit_bytes=0 driver_free_delta_bytes=none trim_released_bytes=none pool_retained_bytes=none p2=admit-same-tick busy_overlap_s=21.655 identity=aa6cc3291b981646 V1=FAIL V2=ok V3=FAIL V4=ok -> FAIL
fix f4350c241:       [admit-oom] reclaim settle (reclaim-on-defer): dev0 evicted_prefix_bytes=1751161856 pool_cached_gain_bytes=1751161856 trim_released_bytes=1610612736 driver_free_bytes 4827971584 -> 6438584320 pool_reserved_bytes 21709717504 -> 20099104768 pool_used_bytes 21685840360 -> 19934678504 pool_retained_bytes=140549120 (...)
                     effective free 4852MB -> 6603MB
                     PREFIX-EVICT-RECLAIM: entry_bytes=1592160256 reclaim_credit_bytes=1751000000 driver_free_delta_bytes=1610612736 trim_released_bytes=1610612736 pool_retained_bytes=140549120 p2=admit-same-tick busy_overlap_s=21.659 identity=aa6cc3291b981646 V1=ok V2=ok V3=ok V4=ok -> PASS
```
Fix (`crates/memra-server/src/worker.rs`): `settle_reclaimed_prefix_bytes` after an eviction fences the model-owned
streams, trims each pool to `used + cached_before` (only the eviction's gain leaves the pool), re-reads driver free;
bytes the pool cannot release (a chunk shared with a live neighbour) are printed as `pool_retained_bytes` and count as
pool-cached headroom, never as driver free. Tokens byte-identical across all four boots (`aa6cc3291b98...`). B's
finding for the lead: on this card the pool `used` counter fell at the free call, so the pool-side gate admitted P2
same-tick even on `main`; the #346 `X -> X` text is the line's capture point plus the driver never getting the bytes.
Two refused calibration cells and one over-tight first V3 clause kept in the receipts. Target card `serve-smoke: 0
failed`, `cache-meter-gate: 0 failed` on the fix. Issues #346, #445, #523 commented by the lane.

## Lane A day 11 (`432816926`, pushed by the lead without an override; ruling 9)
Rule 1 (verbatim): a restore (H2D) that is cancelled before its consumer event completes must either hand the
untouched host source back to the caller as a typed lease (recoverable) or refuse the cancel with a typed error while
the source is still intact; consuming the source and then cancelling is forbidden by the rule. Seam
`TransferEngine::recover_source(ticket, item)`; `CudaTransfers` holds a cancelled H2D source for the caller,
`retire`/`retire_source` answer `Busy` until recovered, once-only, `cancel` answers `AlreadyReleased` once a source left
the ticket. Schedules `transfer_cancel_recovers_source`, `transfer_cancel_refused_after_source_consumed`
(`conformance/recovery.rs`); red arm = the CPU fake with `legacy = true`, which replays D's finding literally. Frozen
v1.1/v1.2/v1.3 blobs byte-identical (`cd144f3b`, `981bcc81`, `50928b75`). Native bindings added to
`tier-transfer-gate`; not run natively today (D day 12).
Rule 2 (verbatim): a continuation (decode or prime) over a cache with any suspended layer must be refused at
`Cache::ensure_usable` with a typed error naming the suspended layers, unless the caller restores first;
`decode_step_h` is NOT changed. Seam `Cache::suspend_layer(il)` / `resume_layer(il, layer)` over the typed
`SuspendedLayers` register; `ensure_usable` returns `ContinuationRefused { path, layers }` while non-empty; signature
unchanged, 30 callers compile, none in `memra-server` continues without the gate (caller census in A's DAY11.md;
ungated paths are bins and the DSV4/qwen4exp `DecodeState` families, which do not use `memra_kv::Cache`). Gates:
tier+kv 303 tests, clippy, fmt, flags, diff-check exit 0; BOX3 release builds of both gates exit 0.

## integ10 (`lane/spill-integ10-20260921`): B day 13 + A day 11
Batteries (`integration-day12/integ10-cpu-battery/`): fmt; tier+kv+gguf 600 tests; memra-server 734 tests; clippy
`-D warnings`; check-flags; publish census; docs registry census; collector pytest 85 (rerun; attempt 1 hit `REFUSED:
[Errno 11] Resource temporarily unavailable` in 20 collector tests while the lead's serve-smoke held
`/tmp/memra-5090.lock`, kept as `pytest-battery-attempt1-rig-lock-held-by-smoke.log`); perf board; diff-check: all
rc=0. Local 5090 `tools/serve-smoke.sh` on the merged tree (`integ10-serve-smoke-5090/`): `serve-smoke: 0 failed`. Full
`tools/local-ci.sh --perf` on the integ10 tree (`integ10-local-ci-perf/`): correctness GREEN, serve-smoke 0 failed,
`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, 3 pair-only skips, `perf stage: 0 fail, 0 warn`, rc=0; rows appended to
`perf-ci.jsonl`, so this push needed no override.
Lead finding: `crates/memra-engine/build.rs` read `DOCS_RS` without `cargo:rerun-if-env-changed=DOCS_RS`, so a
`DOCS_RS=1` clippy pass left stub artifacts that the next `cargo test -p memra-server` linked against (`undefined
symbol: memra_dsv4_c4_recent_write`); one line added in integ10, the two failed attempts kept. The lead's battery
script now sets `DOCS_RS=1` for clippy only.

## Lead rulings, day 12 (continued)
11. **#346 closes with integ10; #445 and #523 stay open.** #346's shape (eviction leaves effective free unchanged) is
    fixed and gated; #445 sections 2 and 3 (gemma spec-route prefix capture, spec under concurrency) and #523 items 1
    to 3 (newest-turn invariant, policy re-decision, 8-turn twin gate) are separate work items in the queue.
12. **Collector pytest lock path.** The tier battery's Python tests take the real rig lock path; they must take a
    private lock path under test (lane F or D, small), so a serving job on the rig cannot redden a CPU suite.

## Lane C day 12 (`7efedd13d`, pushed by the lead with the logged override: `moe_cache.rs` after the row; #552 criterion 1)
`research/spill-c-20260919/HOSTPREFIX-CONTRACT-CENSUS.md` (179 lines, every cell file:line). Bank owner: carried are the
artifact digest on every `BankId` and lease, record identity validated against the semantic `RecordId`, byte-for-byte
record check plus chained checksums, CUDA ownership typed as a `ThreadId` with `WrongOwner`, scale planes refused at
three places; dropped were the identity across the proxy seam (two `u64`s), tenant (= artifact digest by
construction), epoch (constant `{0,0,0}`), an owner-thread check at `install_banked`, a `TransferEngine` (completion is
a compute-stream drain), GPU slots outside the governor, the positional MTP key `layer = u16::MAX`. HostPrefix owner,
the decisive finding: `HostTierContext` (`worker.rs:7869`) has no constructor anywhere in the crate; B's sidecar route
(`tier_charge`, `bind_tier_image`, insert/promote identity leases) is compiled and unit-tested but never executed in
any boot; the server holds no `ProgramIdentity` for a loaded model; the D2H is `memcpy_dtoh` + `synchronize` or the
arena's async copy + fence with no ticket, producer fence, epochs or `Completion`; `host_demote_prefix_ref(&PrefixEntry)`
conflicts with `CudaTransfers::register_device`'s owned `KvPlane`; `PinnedHostBuf` is not `CudaPinnedLease`;
`tier_charge` refuses whenever the startup arena exists; the contract checksum and the `MEMRA_KV_HOST_VERIFY` digest
are two hash programs. Eight risks listed (host-destination conversion as a second byte owner; promote is the restore
program; three copy programs in one entry; digest versus checksum boundary; one tenant derivation; epoch and
generation checks are refusals; MTP key collision at depth > 0; scale planes payload-only would change dequant).
Landed on the bank side (`69905776f`): `bank::dispatch_id(&RecordId)` as the single mapping; `ExpertLeaseToken` carries
`owner, lease, record, artifact, epochs` (still `Send + Sync`); `demand` refuses a bank leasing another record
(`ProgramMismatch`); `with_bytes`/`finish` refuse a foreign token; `admit_banked` asserts the record identity before
the H2D; `slru-synthetic.json` re-pinned with 2013 identical rows; 6 new tests; tier 223 tests, clippy, fmt, flags,
docs, diff-check clean. No byte or H2D program change; no server change.

## Lead rulings, day 12 (continued)
13. **Tenant salt has one owner.** `memra-kv` gets one helper deriving `tenant_salt` from the scope namespace string;
    the server passes the same string it feeds `auth::meter_key`. No second derivation anywhere.
14. **Borrow discipline for the contract-routed demote.** Planes move out of `PrefixEntry` as owned `KvPlane`s for the
    D2H; no borrowed-source seam in `CudaTransfers` (that would be a second ownership program). No v1.4.
15. **HostPrefix through the contracts, in C's order.** Option A first: construct `HostTierContext` behind a
    default-OFF door with a decide-by (build `ProgramIdentity` at model load, salt per ruling 13, inject the server's
    governor); OFF must be byte-identical by construction, ON must leave `serve-smoke.sh`, `cache-meter-gate.py`, B's
    `prefix-evict-reclaim-gate.py` (`V1..V4 -> PASS`) and `kv-host-spill-{identity,failure}-gate.sh` lines unchanged,
    with `MEMRA_KV_HOST_VERIFY=1` `verify ok` on every promote and equal `[prefix-host] demote:` byte counts. The
    startup arena path is out of scope for the first two slices. Option B (D2H through `TransferEngine` on the
    pageable path) only after A's receipts; C (promotion) after B's. C day 13 owns Option A.

## Lane D day 12 (`b3324c262`, pushed by the lead with the logged override; the two refusals become PASS through A's seams)
All seven fault arms now print `FAULT-ARM PASS <arm>` on the target card (N=1, 8k, pooled, gate source `55f242e98`,
`pro-single-day12/`): `cancel-restore` through `recover_source` (`retire`/`retire_source` `Err(Busy)` first,
`recover_source` `Ok(host)`, recovered checksum equal to the bundle's, `StateBundle::verify` `Ok(())`, second recover
and post-recovery cancel `Err(AlreadyReleased)`, pinned 239,468,544 B held by the lease, then retire, restore,
continue); `require-resident` through `suspend_layer`/`resume_layer` (`Err(ContinuationRefused { path: "kv-tier-gate
continuation", layers: [3, 7, ..., 63] })` twice while suspended, `Ok(())` after the last resume). Root validation
`cells 9, failed_commands 0, refused_commands 0, qualification false`; the five continuing arms match the frozen 8k
bundle on all seven surfaces. First native run of A's rule lines (`tier-transfer-gate`, 13 PASS lines, exit 0):
```text
PASS rule cancelled-restore-recovers-source native CUDA
PASS rule cancel-refused-after-source-consumed native CUDA
```
Contract vocabulary: `Arm::refusal` and the REFUSED branch are gone; a backend without a seam is a failed cell; rule-1
rows are generic over `TransferEngine` and shared with a CPU binding (`tests/contracts/fault_arm_bindings.rs`, legacy
fake = red arm); all arms detach through `suspend_layer`/`resume_layer`; `active::roundtrip` (B's series path) keeps
raw `take()` because its frozen 32k receipts were produced that way. Replays: `verify-day12.py` MATCH, reclaim 32 and
contracts 68 tests, clippy, fmt, flags, boundary, docs clean. Revuto on integ10 had found exactly this gap (A's `retire`
refusal versus the day-11 `cancel-restore` sequence still on main); D's day 12 is the fix and rides in integ10.
#552 criterion 2 is now met natively on the target card (fault arms with cancellation, corrupt and missing state,
exhaustion, required state), still executed-not-qualified.

## Lane C day 13 (`31ba74550`, pushed by the lead; ruling 15 Option A landed)
Door `MEMRA_KV_HOST_CONTRACTS` (default OFF, FLAGS row, decide-by 2026-10-05, `research/spill-c-20260919/HOSTPREFIX-DOOR.md`):
the server constructs `HostTierContext` at model load with a `ProgramIdentity` per loaded model (artifact = streaming
sha256 of the GGUF; serialized plan = the plan's debug form, the kv_tier_gate convention; numeric class
`server-prefix-entry-v5-kv-q8_0-<K>B-q5_1-<V>B` from `PREFIX_ENTRY_LAYOUT_VERSION` and `kv_blk_bytes()`; stream = the
single worker owner thread; tokenizer = the artifact; template = the GGUF chat template text or the ChatML fallback;
adapter none; modality text; position prefix-from-token-zero; `tenant_salt` per pool key through the one
`memra_kv::tiered::hostprefix::tenant_salt` helper, ruling 13), injects a server-owned governor
(`shared_governor`, 2x host and 2x device prefix budgets, rationale in C's DAY13.md). The door parses `1`/`0` only
(junk refuses at boot), refuses at boot with the startup pinned arena (`MEMRA_GLM5_TP_KV_HOST=1`), with a vision tower,
or with a directory checkpoint; with no host tier (`MEMRA_KV_HOST_MB=0`) it constructs nothing and prints one line.
OFF/ON on the target card (600 W, binary `9ed02495...`, N=1, `verify-day13.py` PASS, `pro-single-day13/`): serve-smoke
plain + cache-metering 33 verdict lines byte-equal, `serve-smoke: 0 failed` both; identity gate with
`MEMRA_SERVE_SPEC=0` `ALL GREEN` both, demote bytes equal, one promote = one `verify ok`, texts identical; failure gate
with `MEMRA_SERVE_SPEC=0` the same pre-existing `1 FAILURE(S)` on both (tenant cap pre-empts `skip demote`); B's reclaim
gate line and digest identical OFF/ON (red on that tree, B's fix was not merged there yet). Under the default spec
env ON refuses draft-bearing boundary entries by name (identity gate `5 FAILURE(S)`, failure gate `6`): the door's
surface is plain-only today, byte identity holds. `tenant_salt("")` derives (the empty namespace is every no-keyring
server's default; refusing would break the equal-count gate), tested and documented instead of the lead's "empty
refuses" wording. Gate input on this artifact needs `MEMRA_HOSTGATE_CACHE_MB=256` (attempt at 128 kept). Local
battery on the lane: fmt, 739 + 67 + 2 tests, clippy, flags 867 names, diff-check, docs-registry, boundary 0 new.

## Lead rulings, day 12 (continued)
16. **The host-contracts door stays plain-only until its surface grows.** ON is asserted under `MEMRA_SERVE_SPEC=0`;
    under the default spec env the refusal-by-name of draft-bearing entries is the correct fail-closed answer, not a
    bug. Growing `bind_tier_image` to draft planes is the next C slice (before Option B), with the identity and
    failure gates equal OFF/ON under the default env as its exit criterion.
17. **`tenant_salt("")` derives.** C's reading stands: the empty namespace is the default single-tenant namespace,
    one derivation, distinct from any keyed namespace; the lead's "empty refuses" wording is withdrawn.

## Receipt hygiene finding (lead, integ10 and integ11)
The root `.gitignore` rule `build/` silently dropped two lanes' native build receipts (`pro-single-day12/build/`,
`pro-single-day13/build/`); both verifiers (`verify-day12.py`, `verify-day13.py`) bind cells to that receipt and
refused on the merged tree while passing in the lane worktrees (day 11's had been force-added). Both receipts are
now tracked, and `.gitignore` gains `!research/**/build/` so a lane's build receipt is never build output. Lanes: no
`git add -f`; the rule now says what the repo means.

## integ11 (`lane/spill-integ11-20260921`): C day 13
Batteries (`integration-day12/integ11-cpu-battery/`): fmt; tier+kv+gguf 619 tests; memra-server 740 tests; clippy
`-D warnings`; check-flags; publish census; docs registry census; collector pytest 85; `verify-day13.py` `DAY13
REPLAY: PASS` (after the build receipt was tracked); perf board; diff-check: all rc=0. Local 5090 `tools/serve-smoke.sh`
with the door unset (`integ11-serve-smoke-5090/`): `serve-smoke: 0 failed`. Full `tools/local-ci.sh --perf`
(`integ11-local-ci-perf/`): correctness GREEN, serve-smoke 0 failed, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, 3
pair-only skips; then the perf stage hit a contended window: another session's Python process (1390 MiB) joined the
card mid-cell, the battery waited its 600 s, latched the co-resident as persistent and recorded both cells with
`window_clean=false`: `26b-plain-short: 189.60 tok/s [FAIL] (-9.15% vs median 208.70)`, `qwen9b-plain-short: 125.73
tok/s [FAIL] (-9.47% vs median 138.88)`, `perf stage: 2 fail`, rc=1. Read per the battery's own tripwire text: a
uniform drop across cells with correctness green in a contended window is machine state, not the diff; the cells run
`run-gen`, which no file in this PR touches (server door, `memra-kv` helper, tier bank identity). Rows appended
(`window_clean:false`), so the freshness gate is satisfied honestly; a clean-window rerun of the two cells is owed when
the card is free and is noted, not claimed. The co-resident was not touched (another session's lane).

## Lane D day 13 (`e78bf933b`, pushed by the lead with the logged override; #545 and ruling 12)
#545: hosted CI job `portable-suites` (no CUDA install) runs `tools/portable-suites.sh`: `skip-census.py verify` over
memra-tier, memra-kv and memra-cli, then `cargo test -p memra-tier -p memra-kv -p memra-cli --offline --no-fail-fast`
under `MEMRA_PORTABLE_SKIP_BUDGET=0` with `--min-passed 300` (325 passed, 0 skipped, 14 binaries, 22 s warm), then
`tools/test_portable_suites.sh` (teeth: planted tier retirement, kv hierarchy and CLI onboarding-receipt failures in a
temp copy red the wrapper and are named; an undeclared SKIP reds it before cargo runs; 15 ok), then the collector
suite with both rig lock paths held. `tools/local-ci.sh` `cpu_chain()` runs the wrapper fatally. Step names and the
wrapper's last line say "NOT GPU qualification". `fault.rs` committed-receipt replays are strict (no silent `.exists()`
skip). Teeth finding: a shared target dir let cargo reuse the copy's planted test binary on the real tree; the copy
now builds in `target/portable-suites-teeth`.
Ruling 12: `MEMRA_TIER_BATTERY_LOCK_DIR` seam in `tools/tier-battery.py` and `tools/tier-rig-bootstrap.sh` (FLAGS row);
`tests/battery/private_lock.py` `PrivateLockMixin` in every lock-taking class; with both rig locks held for the whole
suite inside a private `/tmp` (`bwrap`): before `24 failed, 64 passed`, after `86 passed, 32 subtests passed`, rc=0; a
receipt written under the seam refuses to validate without it (`REFUSED: missing/noncanonical collector lock`).
Boundary: two raw cargo logs matched `live_fingerprint` on cargo's own test-binary path (`memra-<hash>`); pinned as
false positives. Revuto on integ12 found two gaps, both fixed on the lane (`0e9e30b31`): the static skip census scanned `src/` only
(now `tests/` too; memra-tokenizer's four `llama_parity` skips declared) and the private lock seam was honoured from the
environment alone (now refused without `--private-lock-dir-for-tests`, loud startup line, seam in `lock.json`, validate
refuses foreign seams). D's note for the lead: `tools/ci-change-class.sh` classes `research/**` as docs-only while ten
test-time `research/` reads now exist across lanes (list in D's DAY13.md); queued.

## The integ11 tripwire settled (clean window, `main` `30e433c4c`)
Rerun of `tools/local-ci.sh --perf` with no co-resident (`integration-day12/perfci-clean-window/`): correctness GREEN,
serve-smoke 0 failed, hit gate ALL GREEN, `26b-plain-short: 208.52 tok/s [OK]`, `qwen9b-plain-short: 138.80 tok/s [OK]`,
`perf stage: 0 fail, 0 warn`, rc=0, rows `window_clean:true`. The two `[FAIL]` rows of the integ11 run were the
contended window (another session's process on the card), as the record said; the diff carried no tok/s change.

## Lane C day 14 (`fc7375817`, pushed by the lane; ruling 16 exit criterion met)
Census first (`HOSTPREFIX-DOOR.md` "Draft planes"): a spec-served entry carries the MTP draft-scratch K/V rows
(`PrefixEntry.draft`, bytes owned by `memra_kv::KvLayer` in the trunk's own q8_0/q5_1 encodings, copied by the same
`host_plane_from_device`/`plane_up` programs as the trunk), the boundary hidden `last_h` (already `Role::Hidden`), and
under DSPARK the DFlash tail (no artifact identity derivable from a GGUF digest: stays refused by name). The
`MEMRA_KV_HOST_VERIFY` digest is blind to the draft plane (a later slice). Code (`62f48ecec`, `worker.rs`):
`host_tier_entry_class` (pure `Plain` / `MtpDraft`; refuses GLM TP/latent planes and the DFlash tail by name, same
function at demote, bind, insert, promote); `host_tier_draft_program` folds the head's source, plan and draft encodings
into `artifact`, `serialized_plan` and `numeric` (length-framed), so a spec entry and a plain entry of one prompt never
share an identity; `bind_tier_image` binds `Role::Draft` K/V (`mtp-draft-q8_0`/`mtp-draft-q5_1`) with checksums, the
trunk geometry rule and shape blob `host-prefix-shape-v2`; the pinned charge includes the draft plane. Packed bytes
and copy programs untouched; every changed statement is inside a tier-Some block. Target card, DEFAULT spec env, OFF
then ON, N=1, 600 W, `MEMRA_HOSTGATE_CACHE_MB=256`: identity gate `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`
both, 13 verdict lines equal, demote bytes 160.7/160.6/161.1/161.1 MB equal, one promote and two `verify ok` equal,
23-event prefix sequence and r1/r3 texts equal, ON refusal lines 0 (day 13: 5); failure gate the same pre-existing `1
FAILURE(S)` and identical per-cell tier events (day 13 ON: 6); plain pairs and serve-smoke 33 lines identical.
`verify-day14.py`: `DAY14 REPLAY: PASS` (76 ok). Findings: the draft plane adds about 0.2 MB per entry; the second
promote is declined by the device protected-share rule identically OFF and ON (not a door effect); the verify digest
attests the trunk only. Open for the lead: Option B next; the DFlash tail slice needs an export-dir manifest identity.

## integ13 (`lane/spill-integ13-20260921`): C day 14
Batteries (`integration-day12/integ13-cpu-battery/`): fmt; `tools/portable-suites.sh`; memra-server suite; clippy
`-D warnings`; check-flags; publish census; docs registry census; collector pytest; `verify-day14.py` PASS; perf board;
diff-check: all rc=0. Local 5090 `tools/serve-smoke.sh` with the door unset (`integ13-serve-smoke-5090/`): `serve-smoke: 0 failed`.

## Lane B day 14 (`25d252c6f`, pushed by the lane; #523 items 1 and 3)
First sitting stopped by the lead after four hours without a receipt (driver refusals kept verbatim under
`pro-single-day14/refused-sitting1/`: `REFUSED: [Errno 17] File exists: '.../gate-main'` and `REFUSED: --external-lock
requires exactly one @COLLECTOR_LOCK_FD@ argument`; its binaries were also the wrong arms). Second sitting finished in
0.8 agent-hours. Fix (`f42d921db`, `worker.rs`): the SLRU insert never selects itself as victim (`room_victim_with(slru,
keep)`); when the newest turn does not fit it evicts from the protected segment oldest first; an entry larger than the
budget refuses with the typed line `[prefix-cache] insert refused: entry <bytes> exceeds budget <bytes> (...)`; an
entry that cannot fit beside leased bytes refuses with its own line. Victim selection, the preflight's reclaimable set
and the refusal lines only; `prefix_snapshot` and restore untouched. Unit tests: the incident shape (211 turns beside a
protected cohort, cohort evicted oldest first, never itself), the oversized refusal string, fitting inserts keep the
same victims in the same order, the leased boundary. Gate `tools/prefix-newest-turn-fits-gate.py` (8-turn twin on a
live `memra-server`, pre-seeded protected cohort from a second tenant, cache-off calibration boot, per-turn cold-versus-
restored digest, V3 state identity after every turn). Target card, N=1, 600 W, base `be07f2d36` versus fix, verbatim:
```text
base: PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=737943552 turns=8 cold_turns_after_1=7 cached_ok=0/7 lines_ok=0/8 evictions=0 protected_evictions=0 refused_or_skipped=1 effective_free_ok=8/8 V1=FAIL V2=FAIL V3=ok V4=FAIL -> FAIL
fix:  PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=737943552 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 protected_evictions=2 refused_or_skipped=0 effective_free_ok=8/8 V1=ok V2=ok V3=ok V4=ok -> PASS
```
Per turn cached_tokens, base then fix: 0/0, 0/9200, 0/9500, 0/9800, 0/10100, 0/10400, 0/10700, 0/11000. Completion digests
identical across binaries on all 8 turns and 6 cohort sends, and identical to the cache-off boot's cold completions.
`serve-smoke: 0 failed`, `cache-meter-gate: 0 failed` on the fix; `verify-day14.py` `DAY14 REPLAY OK`. Two earlier
gate rounds are kept as failed cells (V3 first asserted consumed == grew without a footprint control, then a
calibration that missed the cohort phase's retained growth); round 3 states V3 on states. Recorded observation, cause
not inferred: the request path retains device memory growing with the longest prompt seen (67,200 B per token on turns
3 to 8), identical across binaries and with the cache idle; not this gate's subject. #523 item 2 (policy re-decision,
interleaved A/B N >= 5) stays open. Revuto on integ14 found two gaps, both fixed on the lane (`9655b9142`): SERVING.md
and the protected-share FLAGS row still stated the old "probation before protected" guarantee (now the real rule with
the scan-resistance trade-off named; lead ruling 18 below), and the refusal lines had lost their one-shot guard (now a
shape-keyed throttle, counters unchanged).

## integ14 (`lane/spill-integ14-20260921`): B day 14
Batteries (`integration-day12/integ14-cpu-battery/`): fmt; `tools/portable-suites.sh`; memra-server 748 tests; clippy `-D warnings`; check-flags; publish census; docs registry census; collector pytest; B's `verify-day14.py` OK; perf board; diff-check: all rc=0. Local 5090 `tools/serve-smoke.sh` (`integ14-serve-smoke-5090/`): `serve-smoke: 0 failed`.

## Lead rulings, day 12 (continued)
18. **The newest turn fits, and the docs say what that costs.** #523 item 1's rule stands (an insert never evicts
    itself; protected oldest first when the free share is short; typed refusal only when the entry exceeds the budget
    or cannot fit beside leases). The consequence that a growing one-hit tenant can remove another tenant's promoted
    cohort is the documented trade-off, not a bug; the default policy is re-decided under #523 item 2 by interleaved
    A/B on the incident shape, N >= 5, both orders, before any promotion or demotion of SLRU as the naked default.
## Lane A day 12 (`f122468a5`, pushed by the lane; #384 fixed and gated, #385 harness)
#384: `HostPrefixCache::reclaim_tenant_share` (`worker.rs`): at the tenant share cap the demoting tenant's own unleased
host entries are evicted oldest first (the twin spared, leased entries skipped through the new `IdentitySlot::leased`
in memra-kv) until the demotion fits, one retry, else today's bounded refusal with nothing evicted; hook order is
reclaim-then-refuse; `/metrics` gains `prefix_host_tenant_reclaims`. Gate `tools/kv-host-tenant-reclaim-gate.sh` on the
target card (600 W, N=1, `executed-not-qualified`), base `be07f2d36` versus fix `405466cf7`, verbatim:
base: [prefix-host] demote evaporated at the tenant share cap before the D2H copy: 93 tokens, 160.8MB (38% of 1074MB, MEMRA_KV_HOST_TENANT_PCT; model gate, ns "t:acme\u{1f}s1")   (three cap events)
fix:  [prefix-host] evict (tenant share): 89 tokens, 160.7MB of tenant "t:acme"'s own entries for its 160.8MB demotion (row now 160.6MB / 408MB share = 38% of 1074MB, model gate, ns "t:acme\u{1f}s1")
      [prefix-host] demote: 93 tokens, 160.8MB in 6.3ms (host resident 482.0MB / 1074MB, model gate, ns "t:acme\u{1f}s1")   (four reclaims, each followed by its demote)
Cross-arm (`verify-day12.py` `DAY12 PASS`): every image base evaporated, fix demoted at the same byte count; all eight
texts byte-identical; the other tenant's demote, promote and `cached_tokens` equal; the reclaimed entry serves on the
fix arm (`cached_tokens` 95 versus 0); `/metrics` rejects 3 versus 0, reclaims absent versus 4; collector `--validate`
exit 0 on all three evidence cells; three failed attempts kept and named. CPU: memra-server 748, memra-kv 73 tests,
clippy, fmt, flags, docs, diff-check clean. Open and stated: the handoff import (`insert` direct) still evaporates at
the cap; #385's decision cell is the 2x B200 pair. Revuto on integ15 found two gaps, both fixed on the lane
(`8b29b2aa3`): the reclaim evicted before the image existed (five later failure paths could waste the tenant's warm
row; now a pure plan before the D2H, the reclaim after `bind_tier_image` and before `insert`, and a
`prefix_host_tenant_reclaims_wasted` counter with one WASTED line), and FLAGS/SERVING still promised the old evaporation
rule (now the reclaim rule and both counters). Fix arm rerun on the card: PASS, reclaims 4, wasted 0.
#385 harness (`research/spill-a-20260919/HOST-ARENA-STARTUP.md`, `tools/pinned-host-reserve-bench.py`): BOX3 only, 21.87 GiB
(75 percent of MemFree), single versus 8 chunks, N=5 pairs per order both orders, GPU idle 36 to 40 C at 600 W:
single median 3598.8 ms, chunked median 3583.8 ms, 6.1 GiB/s both; per-chunk completions step by about 450 ms, so this
driver serializes concurrent `cuMemHostAlloc`. A harness receipt for this box; not the decision.
Push note: A's first push was refused by the perf-ci gate with the lane's upstream range (six engine files arriving
through the merge of main); A unset the upstream so `tools/push-range.sh` took the merge-base with `origin/main`, and
the hook found no engine file in range (the documented range correction after a merge of main; no skip variable).
## integ15 (`lane/spill-integ15-20260921`): A day 12
Batteries (`integration-day12/integ15-cpu-battery/`): fmt; `tools/portable-suites.sh`; memra-server 748 tests; clippy `-D warnings`; check-flags; publish census; docs registry census; collector pytest; A's `verify-day12.py` `DAY12 PASS`; perf board; diff-check: all rc=0. Local 5090 `tools/serve-smoke.sh` (`integ15-serve-smoke-5090/`): `serve-smoke: 0 failed`.

## Lane B day 15 (`bc16ae4c2`, pushed by the lane; #523 item 2 decided on the target card)
Pre-registered before any run (`9466b8912`): primary metric total computed tokens over the replay (lower wins),
secondary the cohort tenants' `cached_tokens` on their returns (stated either way), a policy wins only if better on the
primary at every pair in both orders, digest identity across arms, runs and the cache-off boot as the precondition.
Shape: the 2026-09-05 incident's ratios at a 2048 MiB budget on `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`: four reused cohort
tenants (74 percent of the budget), one loop growing 27,300 to 30,600 ids over 12 turns, cohort returns every third turn
and after the loop, 28 requests per run, `AB-0, BA-0, ..., AB-4, BA-4`, 20 runs, one lock hold, one binary. Target card,
600 W, 46 to 50 C at run boundaries, 6,124 telemetry samples, verbatim:
```text
PREFIX-POLICY-AB: ... digests_identical=28/28 computed_tokens slru_median=132300 lru_median=122700 (N=10 each) pairs_slru_better=0/10 pairs_lru_better=10/10 ties=0/10 return_cached slru_median=0 lru_median=8700 loop_cold_after_1 slru_max=0 lru_max=0 refusals slru=0 lru=0 temp_c=46.0..50.0 power_limit_w=600.0 -> WINNER=lru
```
Every pair, both orders: 132,300 versus 122,700 computed tokens (+9,600, 7.3 percent), deterministic; cohort return
cached 0 versus 8,700; loop cold 0 in both; TTFT loop p50 158.6 versus 157.8 ms. Mechanism from the server's own lines:
SLRU evicts the loop's newest entry after each cohort return and protects the loop's dead last entry over the last
tenant's fresh one. Landed (`87d9e5963`): plain LRU is the only policy; `MEMRA_PREFIX_CACHE_POLICY`,
`MEMRA_PREFIX_CACHE_PROTECTED_PCT`, the segments, promotion and demotion, the SLRU tests and the A/B harness deleted
(harness source at `9466b891` in history); refusals, throttle, leased preflight and the pressure-relief order kept; the
twin gate's V4 is now "cohort eviction, never self-eviction"; FLAGS "Removed doors, 2026-09-21", SERVING, TESTING,
`docs/decisions/PREFIX-CACHE-POLICY.md`. Landed binary on the card: twin gate `-> PASS` with day-14 digests 8/8,
`serve-smoke: 0 failed`, `cache-meter-gate: 0 failed`; `verify-day15.py` `DAY15 REPLAY OK`. Reconciled with #597
(`a63739afb`): A's tenant-share code intact (40 references), 754 server tests. One lock-busy refusal kept
(`refused-lockbusy/`, lane C's cell held the card).

## Lead rulings, day 12 (continued)
19. **Plain LRU is the prefix-cache policy; the SLRU door is gone.** The primary metric is a function of bytes and
    policy, not of the card (a mechanism result: SLRU's protection displaces the growing tenant's newest entry), so
    the one-card verdict lands the naked default and deletes the losing arm per door hygiene; the owner's one-rig rule
    is stated in the decision record and the local 5090 replay of the same harness (from history) is owed as a
    confirmation cell, not as a blocker. The scan-resistance property SLRU advertised is documented as removed with
    the door; a future segmented policy comes back only with its own A/B on this shape.

## Main moved under the integs (09:47Z): #590
Another session merged codex PR #590 "ci: execute portable suites and dispatch pinned GPU qualification" (`34ed99dfc`):
`tools/ci-portable.sh`, `tools/gpu-ci.py`, `.github/workflows/gpu-ci.yml`, `docs/CI.md`, ci.yml and local-ci.sh lines.
It overlaps D's #592 (`tools/portable-suites.sh`, the `portable-suites` job, the skip census with teeth): main now
carries two entry points for the same suites. Not untangled here; flagged for the owner and queued (one entry point,
the teeth kept). integ16 and integ17 merged `34ed99dfc` after their batteries (ci, tools and docs only; the battery
trees' crate content is unchanged).

## integ16 (`lane/spill-integ16-20260921`): B day 15
Batteries (`integration-day12/integ16-cpu-battery/`): fmt; `tools/portable-suites.sh`; memra-server 754 tests; clippy
`-D warnings`; check-flags; publish census; docs registry census; collector pytest; B's `verify-day15.py` OK; perf board;
diff-check: all rc=0. Local 5090 `tools/serve-smoke.sh` (`integ16-serve-smoke-5090/`): `serve-smoke: 0 failed`.

## Lane B day 16 (`0fae7e715`, pushed by the lane; the owed 5090 confirmation cell stopped on its precondition)
Scaled shape (budget 1024 MiB, byte shares preserved, ctx 16384), pre-registered prediction matched the card's byte
arithmetic to the token (slru 31,700 versus lru 29,550 computed tokens at both pairs, 56/56 rows per arm), but the
precondition failed: `digests_identical=26/28`, one restored-suffix request per arm lineage differs from the cache-off
boot, deterministic across runs; and the admission reclaim ladder (`[admit-oom] reclaim-on-defer`) evicted 12 prefix
entries per run under VRAM pressure on this 24 GB card, which the target card never did. By the pre-registered rule the
day produced no verdict (`-> DIGEST-FAIL`) and the 20-run cell was not run; `verify-day16.py` `DAY16 REPLAY OK: receipts
consistent, precondition FAILED on this card, no verdict`; the decision record carries the contrary result as a
paragraph, the decision text untouched (target-card receipts). Lead reading: the restored-versus-cold divergence on the
5090 is a correctness question of its own (one numeric program per request), not a policy question; B day 17 probes it
with the day-14 twin gate on that card. B also found two stale `|||||||` diff3 markers in `research/INDEX.md` on main
(from another session's merge at #587); removed in integ18.
## Main red since 09:47Z: the duplicate workflow key (fixed by #600)
#590 (codex, merged by another session) and #592 (D day 13) each added a ci.yml job named `portable-suites`; a duplicate
mapping key makes the workflow invalid, GitHub ran zero jobs (run 35585228365 on main, "workflow file issue"), and every
push and PR since read as failure. PyYAML `safe_load` keeps the last key silently, so the day-13 YAML-load step could
not have caught it; #590 was green because its merge ref predated #592 (the stale-merge-ref class is closed only by the
branch-protection "require branch up to date" setting; owner decision). Lead hotfix #600 (`9ef2f04d6`): #590's
duplicate block removed, its one unique step (the gpu-ci orchestration tests) kept in the surviving job under the
unittest floor. CI green again on the first run after it.
## Lane D day 14 (`2d2b70c55`, pushed by the lane; one entry point for the portable suites)
Census (D's DAY14.md §1): `tools/portable-suites.sh` (#592: `--offline --no-fail-fast`, static and run-side skip census at
budget 0, floor 300, banked raw log, teeth, `needs: changes`) versus `tools/ci-portable.sh` (#590: `cargo test --release
--locked`, no census, no floor, no teeth, ungated); `gpu-ci.yml` runs no CPU suite, and its dispatch prerequisites are
still in draft #566, so every dispatch refuses as unconfigured by design. Fold: `portable-suites.sh` is the one executor
(`--locked` folded in), `ci-portable.sh` is a forward that nothing tracked calls, one ci.yml job, one `local-ci.sh`
call (`CARGO_BUILD_JOBS=8 RUST_TEST_THREADS=8`), `gpu-ci.yml` untouched, `docs/CI.md` and TESTING name the one entry
point. New guard: `tools/check-workflow-keys.py` (a strict loader that raises on duplicate mapping keys) in the `gates`
job and as an unconditional pre-push arm with no skip switch (a ci.yml step cannot protect ci.yml from itself; the push
can), teeth `tools/test_workflow_keys.sh`; proven on main's own broken file (`duplicate mapping key 'portable-suites' at
line 401 column 3 (first at line 206)` while `safe_load` exits 0). Teeth verbatim: `test_portable_suites: 22 ok, 0 FAIL`
(new arm 3: exactly one job, no live cargo test on the three crates outside the wrapper, the forward runs no cargo),
`test_workflow_keys: 9 ok, 0 FAIL`, `check-workflow-keys: OK: 5 workflow files, no duplicate mapping keys`, wrapper
`skip-census: 335 passed, 0 skipped (budget 0)`, `unittest-floor: OK: ran 11 tests (floor 9) for tools (test_gpu_ci.py)`,
collector pytest 87 with both rig locks held. Post-hotfix merge clean (one job, one gpu-ci step). Findings kept:
`research/**` read at test time versus the docs-only classifier (day 13); an untracked day-11 `build/` leftover in D's
receipts (now un-ignored by the repo rule; left for D to add or remove).
## integ18 (`lane/spill-integ18-20260921`): D day 14
Batteries (`integration-day12/integ18-cpu-battery/`): fmt; `tools/portable-suites.sh`; memra-server suite; clippy
`-D warnings`; check-flags; publish census; docs registry census; collector pytest; `check-workflow-keys.py`;
`test_portable_suites.sh`; `test_workflow_keys.sh`; the gpu-ci tests under the floor; perf board; diff-check: all rc=0.
Also in integ18: two stale `|||||||` diff3 markers removed from `research/INDEX.md` (left by another session's merge).
## Lane C day 15 (`8ed05bfc6`, pushed by the lane; ruling 15 Option B behind the door)
Census before code (`HOSTPREFIX-DOOR.md` "Option B"): the pageable demote was twelve steps by reference with per-plane
`memcpy_dtoh` plus `synchronize`, no ticket, fence or checksum; the single point where owned planes can leave the entry
is `Option::take` on the entry's slots under `&mut PrefixEntry` (no placeholder, no device byte; a `CudaSlice::clone`
would be a D2D copy). Code (`5f8d327e2`, `worker.rs`, door ON only): `host_kv_planes_through_contract` runs the
`kv_tier_gate/active.rs` sequence on the pageable tier (pre-checks, `alloc_host` per plane in the OFF order, admission
probe on the same ledger, `take()` the planes, `register_device`, `retain_device` twins, `record_producer`, one
`submit_batch` with K and V per plane and the draft last, `synchronize`, `poll`, `take_destination`,
`Completion::require`, `record_consumer`, `retire_source`, `take_plane` and `into_pooled` back into the same slots,
`release_producer`, `retire`, `acknowledge`); each host plane is `HostPlaneBytes::Contract { lease, receipt }`;
`bind_tier_image` refuses when its bundle checksum differs from the receipt; a quarantined completion is typed
`SourceQuarantined` and latches the tier off; `HostTierLedger` adapts the server's `Arc<Mutex<..>>` governor to the
engine's `Rc<RefCell<dyn BudgetGovernor>>` (one ledger, two handles); `inflight = 2 x layers + 2`. Promote, arena and
OFF untouched. Target card, default spec env, OFF then ON, N=1, 600 W (`verify-day15.py` `DAY15 REPLAY: PASS`, 134
checks): identity gate `ALL GREEN (teeth=0)` both with equal demote bytes and `verify ok`; identity plain the same;
failure gate the same pre-existing `1 FAILURE(S)` and, under ON, the digest fault cell prints the receipt, `FAULT`, the
named injected difference and `VERIFY FAILED`; lane A's tenant-reclaim fix arm `PASS` with 8 equal demotes and 8
receipts; serve-smoke 33 lines equal; lane B's two gates identical. Receipt per ON demote, verbatim:
[prefix-host] contracts door D2H receipt: ticket issuer=2 seq=1 epochs=0/1/1 items=34 (16 KV planes, draft) complete=34 require=ok checksums_sha256=435f0da4...d8c73ab9 retired acknowledged
Finding for the decide-by review (N=1, no claim): the contract's destinations are write-combined (`cudarc alloc_pinned`
is `CU_MEMHOSTALLOC_WRITECOMBINED`), so the bind hash runs at WC speed; demote 149 versus 281 ms and promote 267 versus
398 ms OFF versus ON on single observations; the promote delta is unexplained and is Option C's first cell; the
allocation flag is engine territory (`tier_transfer.rs`), untouched. C merged lane A's day 12 from its lane branch
before #597 landed; the same commits are now on `main`. Revuto on integ17 found two real bugs in the unwind, both fixed on
the lane (`30704905f`): pre-submit refusals escalated to quarantine because `originals` still held a lease when the
planes came back (now dropped first; typed `Refused`, tier on), and the abort retired straight after `record_consumer`
with the result discarded (a leaked in-flight charge would have made every later demote refuse `Capacity`; now drained,
never discarded, `TicketLeaked` latches the tier). One-shot faults `contract-presubmit`/`contract-postpublish` and
`tools/kv-host-contract-fault-gate.sh` (`ALL GREEN` on the card) prove both. Reconciled with #598 by the lane.
## integ17 (`lane/spill-integ17-20260921`): C day 15
Batteries (`integration-day12/integ17-cpu-battery/`): fmt; `tools/portable-suites.sh`; memra-server 761 tests; clippy
`-D warnings`; check-flags; publish census; docs registry census; collector pytest; C's `verify-day15.py` PASS; perf board;
diff-check: all rc=0. Local 5090 `tools/serve-smoke.sh` with the door unset (`integ17-serve-smoke-5090/`): `serve-smoke:
0 failed`. Main `34ed99dfc` (#590) merged after the battery (ci, tools and docs only).

## Lane B days 17 to 19 (`ca062f170`, pushed by the lane; #602 found, named and fixed on both capture sites)
Day 17 (probe, no code): twin gate on the local 5090, day-16 shape, identity 11/12: turn 10 (12,350 = 12,200 restored +
150 queued) restored `"_\t\t\"\t\t\"\t"` versus cold `"_\n"`, first differing generated token 2. Two programs named
from the server's own `[primeseg]` lines: cold prime calls on the 32-token GDN WY-chunk grid (`grid_off=0`) versus a
restore at the prompt-end seed boundary plus one off-grid prime call (`start=12200 take=150 grid_off=8`), which the
engine's `align_prime_ranges_to_gdn` law says is not bit-identical. Probe E: on-grid restore points identical, off-grid
ones flip to the same alternative stream (3/5); `MEMRA_GDN_CHUNKED=0` identical 12/12; tokenwise identical; not the
admission reclaim. What the #379 gate's cells lacked: a suffix-fed restored completion compared with the cold render
of the same prompt at an off-grid boundary. Issue #602 opened with the repro; the target card's earlier 28/28 and 8/8
were near-ties that did not flip, and the decision record says so.
Day 18 (fix, first half): `seed_capture_boundary(prompt_len, hit_len)` and `Session::seed_at`: the prompt-end seed
publishes at the largest grid-aligned length under the prompt end leaving at least `PRIME_MIN_T` prompt tokens, through
`prefill_tick`'s existing stop; on-grid prompts publish whole; an aligned length under 64 refuses with a typed
`[prefix-cache] seed REFUSED (grid)` line and the counter `prefix_cache_seed_grid_refusals`; `cached_tokens` reports the
aligned length; no prime program moved. The twin gate gained V5 (identity) and V6 (grid) as verdict clauses; new
`tools/prefix-restore-identity-gate.py`. Both green on both cards, but `tools/spec-on-cache-hit-gate.sh qwen` went red on
the fix (`r3`, `g2`): the spec session still captured at the off-grid prompt end, so spec-on and spec-off restored
different lengths. Not integrated; ruling 21.
Day 19 (fix, second half): the cold spec session's `capture_at` is the seed's grid boundary with its own prime stop; a
restored spec session republishes at the render-stable boundary ahead of what it restored or at the seed's grid
boundary; the engine's prompt-end republish (`spec.rs`) fires only on a grid multiple. `spec-on-cache-hit-gate.sh qwen`:
fix `ALL GREEN (qwen)` (61 ok; seven accounting clauses restated to the aligned lengths, e.g. `cached == cap(prompt) ==
64`, `g3 ... 96 of 119`, with the reason beside each; a new on-grid full-cover pair keeps the `restore-full-cover` site
exercised), base `2 FAILURE(S)` on those accounting clauses; the identity law `spec-on text == spec-off text` untouched
and green on both arms. Both cards: twin gate `identity_ok=12/12 grid_ok=31/31 off_grid_calls=0 ... -> PASS`, restore gate
`identical=5/5 grid_ok=5/5 -> PASS`, `serve-smoke: 0 failed`, `cache-meter-gate: 0 failed`; lane A's tenant gate PASS
with equalities reading the leader's `capture_len`; the day-13 evict-reclaim gate's V3 fixed in the gate (it had not
counted `pool_retained_bytes` as headroom; base covered the retention by evicting a second seed), both arms PASS.
`tools/local-ci.sh` correctness exit 0 and `--perf` exit 0 on the lane (two perf-ci rows). Open, inspection only:
`prefix_fanout_groups` takes the raw in-batch LCP as its capture length. Incident, recorded by the lane: it SIGTERMed
the lead's integ19 perf battery (PID identified by name, not by cwd); rerun, 35 minutes lost; rule added to the lane's
STATE.md (never signal a process you did not start; identify by cwd first).

## Lead rulings, day 12 (continued)
21. **One capture law for every prefix-cache capture site.** A restore point is always on the GDN grid, so a restore
    plus its cold-primed suffix is the cold program; the twin gate's identity is a verdict clause on both cards; the
    #379 gate's accounting clauses read the aligned lengths, its identity law is untouched. Day 18 alone was refused
    integration because it turned the #379 gate red; day 19 completes it. #602 closes with integ20.

## Lane A day 13 (`446252336`, integrated from the lane worktree; the pinned-kind seam and the cached-versus-WC A/B)
Census (`PINNED-FLAGS.md`): one allocation site, `tier_transfer.rs:225` via cudarc `alloc_pinned` = `cuMemHostAlloc(..,
WRITECOMBINED)`, the only pinned surface in the engine that did not choose its flags (the other six use raw
`malloc_host` with 0 or PORTABLE); WC readers under the door: the engine's completion hash, the bind hash, Option C's
H2D-source hash (the verify digest reads the device entry, not the pinned image). Seam (`33405a186`): `PinnedKind {
WriteCombined, Cached }`, default `WriteCombined` (flag bits 4); `alloc_host` unchanged and delegating to `alloc_host_kind`;
engine-owned `PinnedBacking` through the existing `malloc_host` FFI; copy sites untouched; no env read; cells: flag bits,
one `malloc_host` site, `cuMemHostGetFlags` per arm on the card. Pre-registered rule committed before the runs: cached wins
iff byte exact 22/22, driver flags honoured, cached bind hash below WC at every pair (10/10), cached D2H no worse at every
pair, medians hold in both orders. Decision cell `pro-single-day13/pinned-ab-160m-s2` (160 MiB, N=5 per arm per order,
one lock hold, 35 to 36 C, 87 to 91 W under 600 W, SM 2347 to 2362 MHz): bind hash median 1711.12 ms WC versus 77.58 ms
cached (pooled N=10; per order 1711.22/1692.42 versus 77.58/77.58); engine hash 1710.20 versus 77.61; D2H 3.01 versus
3.00; H2D 2.98 versus 2.98; alloc 24.33 versus 29.31; per pair bind hash cached below WC 10/10, D2H 10/10 on both readings,
byte exact 22/22; `wc-ab.py` `WC AB REPLAY: PASS (15 checks)`. Verdict for this card, verbatim: `cached_arm=wins-on-this-card`.
Sitting 1 of the same cell kept beside it (inconclusive on the per-pair D2H clause by 1 to 4 us on a 3.0 ms DMA, and a
failed flags equality of A's own). Finding for the review: C's 136 to 175 ms whole-demote line cannot hold two WC
hashes of 160.7 MB at the rate measured (98 to 123 MB/s WC, 2.16 GB/s cached); the door hashes the committed prefix
(`p.len x row`, about 1.75 MB at 89 tokens) while the line prints the 160.7 MB capacity; the door's cost scales with the
committed length (about 3.4 s WC against 0.16 s cached for the two demote-side hashes of a full 8192-token entry, by
rate times bytes, not served). No default moved.

## Lead rulings, day 12 (continued)
22. **Cached pinned destinations for the contract path on the target card class, pending the 5090 cell.** The rule was
    pre-registered and met 10/10 in both orders with byte exactness; the effect is a mechanism (write-combined memory is
    slow to read back on the host, and the door reads every image back) and 22x. Per the one-rig rule the default
    flips for the RTX PRO 6000 class first: lane A day 14 runs the same cell on the local 5090 and lands the per-device
    default (`PinnedKind::Cached` where the card class has a receipt, `WriteCombined` elsewhere) with the FLAGS and
    decision records; the door's decide-by review reads the door's cost again after that.

## integ20 (`lane/spill-integ20-20260921`): B days 17 to 19 and A day 13
Batteries (`integration-day12/integ20-cpu-battery/` for B alone, `-2/` with A): fmt; portable suites; memra-server 758
tests; memra-engine CPU lib tests; clippy `-D warnings`; censuses; collector pytest; `verify-day19.py` OK; A's `wc-ab.py`
decision-cell replay `WC AB REPLAY: PASS (15 checks, 0 failed)`; perf board; diff-check: all rc=0. Full
`tools/local-ci.sh --perf` on the combined tree (`integ20-local-ci-perf/`): correctness GREEN, serve-smoke 0 failed,
`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (the corrected gate, on the fix), `26b-plain-short: 208.81 tok/s [OK]`,
`qwen9b-plain-short: 138.91 tok/s [OK]`, `perf stage: 0 fail, 0 warn`, rc=0. #602 closes with this PR.
## Lane C day 16 (`36fa5262b`, integrated from the lane worktree; Option C behind the door)
Census before code (`HOSTPREFIX-DOOR.md` §"Option C"). Code (`25891baa9`): `host_kv_planes_from_contract` promotes an entry
whose KV planes are `HostPlaneBytes::Contract` through the transfer contract's H2D (fresh destinations from the OFF
allocator, `register_device` and `retain_device` twins, source twins through the one engine addition
`CudaTransfers::retain_host` (a second owned handle on one pinned allocation, the host mirror of `retain_device`; while a
twin lives, `write` and a D2H into either refuse `Busy`), `record_producer`, one `submit_batch`, `synchronize`, `poll`,
`Completion::require` against the D2H receipts before publication, `ready_view`, consumer fence, drain, `retire_source`,
`release_producer`, `retire`, `acknowledge`, `take_plane` into `PrefixPlane`s; publication stays
`insert_pinned_demoting`; the verify digest stays the OFF check). Typed `HostPromoteFailure { Failed, Refused,
ReceiptMismatch (cancel, `recover_source` per rule 1, caller drops the entry), Latched }`; fault sides per route (the
day-15 review lessons applied: originals dropped before the unwind, no discarded retire, drain before retire); device
ledger at 3x. OFF untouched (`plane_up` intact). Target card, N=1, `DAY16 REPLAY: PASS` (203 checks): GPU cells 6 passed;
`tools/kv-host-contract-fault-gate.sh` `ALL GREEN` on four cells (a first sitting `2 FAILURE(S)` was a matcher, kept);
identity default and plain `ALL GREEN` both arms with one H2D receipt per ON promote carrying the same digest as its
D2H; failure `1 FAILURE(S)` both arms (pre-existing), the flip named at both ends; lane A's tenant fix arm `PASS` both
arms (8 D2H and 2 H2D receipts); serve-smoke 34 lines equal; lane B's gates identical.
WC pair (`WC-DESTINATIONS.md`, `wc-pair.py`, target card, N=5 per arm per order, orders agree within 0.5 ms, one lock hold,
37 to 51 C, max 492 W under the 600 W limit): demote median 37.8 ms OFF versus 169.2 ms ON; promote line 12.2 versus 171.7
ms; promote minus inline demote 4.5 versus 33.2; steady-state demote 6 to 8 versus 136 to 140 ms; a first-touch step of
about 35 ms on the first three demotes in both arms. The day-15 promote delta was the inline demote inside the promote's
window. Not a verdict: the first cell of the decide-by review. C's push was refused by the perf-ci gate (one engine file,
`tier_transfer.rs`); the lead integrates from the lane worktree and runs the full `local-ci.sh --perf` on integ19. Revuto
on integ19 found two more unwind bugs, both fixed on the lane (`4467131f6`): partial acceptance asked `recover_source`
for rejected slots (now accepted items only; `Refused`, not `Latched`), and `published` was inferred from the item
index (now the engine's answer through `cancel`); two new faults, six fault-gate cells `ALL GREEN` on the card.
## Main moved again (14:00Z): #589, the release qualification gate
Codex PR #589 (merged by another session, `435a57a75`) retires the perf-ci freshness arm and refuses any engine-source
push as `UNQUALIFIED` unless a content-bound qualification receipt exists or the push is announced as development
(`MEMRA_RELEASE_QUALIFICATION_MODE=development`, logged). integ19 and integ20 push in that mode and claim no
qualification; every cell stays executed-not-qualified. Lanes' own pushes of engine changes now stop at that arm; the
lead pushes the integ trees.
20. **The host-contracts door cannot be promoted at write-combined cost.** A 4x demote and a 14x promote wall-time
    penalty on the target card (N=5 both orders) is the door's dominant open item before its 2026-10-05 decide-by.
    The allocation flag is engine territory: lane A day 13 adds a typed pinned-kind seam with today's default unchanged
    and runs the cached-versus-write-combined A/B through the transfer gates on the target card (pre-registered rule,
    byte exactness in every cell); the default moves only on that verdict and, per the one-rig rule, for that card
    class first. Option C stays behind the door meanwhile.
## integ19 (`lane/spill-integ19-20260921`): C day 16
Batteries (`integration-day12/integ19-cpu-battery/`): fmt; portable suites; memra-server (one order-dependent flake on the
first run, `same_effort_value_resolves_identically_on_every_surface` read 429 where 200 was expected; 758/758 on the
rerun and 3/3 alone; earlier integ batteries green; unrelated to this diff); memra-engine CPU lib tests; clippy;
censuses; collector pytest; `verify-day16.py` PASS; perf board; diff-check: all rc=0. Full `tools/local-ci.sh --perf`
(`integ19-local-ci-perf/`): attempt 1 SIGTERMed by lane B at 13:06:50Z (`rc=143`, no rows; recorded by the lane with the
rule); attempt 2 correctness GREEN, serve-smoke 0 failed, hit gate ALL GREEN, then `26b-plain-short: 23.15 tok/s [FAIL]
(-88.90% vs median 208.65)` beside `qwen9b-plain-short: 138.58 tok/s [OK]`, no co-resident sampled at start or end;
attempt 3 (the two cells, card idle at 56 C, the 26B model fully page-cached): `26b-plain-short: 208.71 tok/s [OK]`,
`qwen9b-plain-short: 138.97 tok/s [OK]`, `perf stage: 0 fail, 0 warn`, rc=0. Read per the tripwire text: a one-cell drop
with the other cell fine and correctness green is machine state, not the diff (the cells run `run-gen`, untouched here);
the attempt-2 rows stay in the log as measured. Second CPU battery after the review round (`integ19-cpu-battery-2/`): all
rc=0, 758 server tests, `DAY16 REVIEW REPLAY: PASS`; local 5090 serve-smoke with the door unset: `serve-smoke: 0 failed`.

## Lane A day 14 (`9a4a3acef`, pushed by the lane in development mode; the 5090 pinned cell and the per-device default)
5090 cell (`rtx5090-day14/pinned-ab-160m`, 160 MiB, N=5 per arm per order, both orders, 22 roundtrips, 55 to 57 C, 28 to
29 W with no power limit reported, SM 1590 to 1627 MHz), verbatim: `PINNED-AB rule byte_exact_all=true
driver_flags_honoured=true bind_hash_cached_below_wc=10/10 engine_hash_cached_below_wc=10/10 d2h_cached_not_above_wc=5/10
d2h_cached_strictly_below_wc=5/10 h2d_cached_not_above_wc=6/10 medians_both_orders=false cached_arm=inconclusive
cached_arm_strict_d2h_reading=inconclusive`. Bind hash cached 37.5 versus 1449 ms write-combined (10/10, 39x); D2H 7.34
versus 7.27 ms with cached above in both orders' medians (the 16 MiB context cell 0/10 with disjoint ranges); byte exact
22/22; driver flags honoured. Two card-class verdicts, no cross-box timing. Per-device default landed (`4488d83fe`):
`PinnedKind::for_device(name)` returns `Cached` for the `HardwareTarget::RtxPro6000Blackwell` class (day 13 receipt) and
`WriteCombined` for the `Rtx5090` class and every unrecognized name, keyed on the device name through the engine's
existing per-device key (compute capability cannot separate the two classes, both 12.0); resolved once in
`CudaTransfers::new`; `alloc_host` uses it; `alloc_host_kind` stays the gate's arm; no env read; flag bits unchanged;
cells for the class resolution, the default bits and the single resolution site; the gate prints `PINNED-DEFAULT
device=".." kind=.. flags=..`. Both cards through the new default: target card `kind=cached flags=0`, conformance 13
PASS, roundtrip `byte_exact=true` at six sizes with `driver_flags=2`; 5090 `kind=write-combined flags=4`, conformance 13
PASS, roundtrip byte exact at six sizes with `driver_flags=6`. `docs/decisions/PINNED-DESTINATIONS.md` (question, both
cards' measurements with N and regime, the per-device rule, what would reverse it). Push note: a docs-only push without
the development variable was refused `UNQUALIFIED: source inputs changed` (the hook compares tree inputs, not the
range); rerun once with the variable.

## Lead rulings, day 12 (continued)
23. **Per-device pinned kind stands as landed.** The target card's rule was met 10/10; the 5090's was not (the D2H
    clause lost by 70 us on a 7.3 ms DMA while the host read gained 39x), so the 5090 keeps write-combined until a
    rule that weighs the door's actual byte mix is pre-registered and met there. The door's decide-by review reads its
    cost again on the target card with cached destinations (C's WC pair is superseded on that card class).

## integ21 (`lane/spill-integ21-20260921`): A day 14
Batteries (`integration-day12/integ21-cpu-battery/`): fmt; portable suites; memra-server suite; memra-engine CPU lib
tests; clippy `-D warnings`; censuses; collector pytest; A's 5090 replay (`WC AB REPLAY: FAIL (15 checks, 2 failed)`
by design: the two rule clauses read inconclusive, all 13 integrity checks ok, `replay agrees with the binary's
verdict: inconclusive`); perf board; diff-check. Local 5090 `tools/serve-smoke.sh` (`integ21-serve-smoke-5090/`):
`serve-smoke: 0 failed`. Pushed in the announced development mode (engine source in range).

## integ22 (`lane/spill-integ22-20260921`): B day 20, the LRU decision confirmed on the second card class
B redid the day-16 confirmation on the local RTX 5090 with both capture sites on the 32-token grid (#602 fixed
the day-16 digest failure). Same harness and two-arm binary, the shape moved onto the grid with the target's byte
shares preserved, `AB-0, BA-0, ..., AB-4, BA-4`, 20 runs, one lock hold, 250 ms telemetry, verbatim:
`PREFIX-POLICY-AB: budget_bytes=1073741824 cohort_tenants=4 cohort_bytes=792920064 turns=12 start_tokens=10912
grow=160 return_every=3 pairs_per_order=5 runs=20 requests_per_run=28 digests_identical=28/28 computed_tokens
slru_median=31776 lru_median=29600 (N=10 each) pairs_slru_better=0/10 pairs_lru_better=10/10 ties=0/10
return_cached slru_median=0 lru_median=1696 loop_cold_after_1 slru_max=0 lru_max=0 refusals slru=0 lru=0
temp_c=66.0..74.0 power_limit_w=None -> WINNER=lru`. Same three mechanisms as the target card, row for row;
every `cached_tokens` row equals the offline prediction (280/280 per arm). The one deviation from day 16, stated in
the pre-registration: the scored cell ran with `MEMRA_REUSE_POOL=0`, because the continuation pool's parked
sessions were the VRAM the admission reclaim ladder took from the prefix cache on that 24 GB card (11 events per
run at the default pool; 0 in all 21 boots with the pool off; the default-pool smoke on the same binary read the
same rows and 28/28 digests, so the pool moved residency, not bytes). No timing compared across boxes. Lead
reading: agreement with the target card, the decision in `docs/decisions/PREFIX-CACHE-POLICY.md` holds on both
card classes; B replaced the day-16 no-verdict paragraph with the day-20 result and the Status line no longer says
the 5090 cell is owed. #523 item 2 is done (B's comment); the issue stays open for its other items.

Lane tip merged: B `952e0f87c` (clean merge, no engine source in the range; the only tracked file outside
`research/` is the decision doc). Batteries (`integration-day12/integ22-cpu-battery/`): `git diff --check`, perf
board check, flags census, public-boundary `check` (0 new), workflow-key census, B's three replays
(`verify-day20.py` on `ab-full-retry1`, `ab-smoke-pool0`, `ab-smoke-default`: each `replayed rule on the primary:
WINNER=lru`, harness outcomes `WINNER=lru`, `SMOKE`, `SMOKE`), em-dash scan on the prose: all rc=0 on the merged
tree. The first pass ran while `origin/main` moved under it (integ21 merged at 15:41Z) and read `rc=2` on
`git diff --check` and on my mis-typed invocations of the boundary scan and the replay; the summary keeps both
passes.

**Correction carried in this integ.** That diff-check complaint (`research/INDEX.md:563: leftover conflict marker`)
exposed a real defect: #604's `research/INDEX.md` (main `c53d0b9f7`) carried a stray diff3 base marker
(`||||||| parent of cdb3b49a1 ...`) above its own row `lockstep-cpu-rows-exact-20260921`. My integ21 merge of that
main hit a conflict in INDEX.md and the union resolver read the marker as a hunk base, so it kept every other row
and dropped the lockstep row together with the marker. Main after #607 has no marker and no lockstep row.
integ22 restores the row verbatim from `c53d0b9f7` (appended after `spill-a-20260919/day14`); a set difference of
the two INDEX files shows it was the only line lost. Ruling 24: a union resolve is followed by a set-difference
check of every conflicted file against both parents, and the marker grep covers `|||||||` too.

Push mode. The first plain push was refused `UNQUALIFIED: source inputs changed: crates/memra-engine/src/bin/
cpu_native_check.rs, ...` although `git diff origin/main HEAD -- crates` is empty and no commit in the range touches
`crates/`: the #589 hook compares the pushed tree's engine source with the committed qualification pointer, not
with the push range, and main's engine tree has moved past the last receipt (#604, #607). Pushed in the announced
development mode; the receipt is the release lane's to renew, nothing here claims qualification.

## Lead slices between integs (2026-09-21, after integ22)
- #608 (build.rs `DOCS_RS` declaration sat past the docs early return): fixed in #610, merged `a51e29abb`, issue closed.
- D's day-13 note (the docs-only classifier calls `research/**` documentation while crates include research files
  at compile time): `tools/ci-change-class.sh` now derives the included set from the head tree
  (`include_str!`/`include_bytes!` literals under `crates/`, resolved to repo paths) and classifies a change to one of
  them as `compile-input:<path>`; any failure to derive the set is code. Revuto round 1 on #611: the first census was
  line-based and missed the multi-line `include_str!(` form (ep_map, the ornith template); the census now reads each
  file as one record (`grep -z`) and exposes `ci-change-class.sh census <rev>`. Real tree today: six included paths
  (recompute-load and two C fixtures, the ep-placement example map, the qwen38 and ornith15 chat templates); teeth arm
  18 asserts them on this repository. Teeth arms 15 and 16 in `tools/test_ci_change_class.sh`.

## integ23 (`lane/spill-integ23-20260921`): C day 17 and B day 21
Lane tips merged: C `a6b62c998`, B `c5384d53c` (clean merges on main `1097d7450`). Files outside `research/`: C's
`tools/memra_cpu_experts.cpp` (one hunk), `tools/memra_cpu_expert_prefetch_test.cpp`, `tools/test_cpu_expert_prefetch.sh`,
a `ci.yml` step in the engine-tests job, a `docs/TESTING.md` paragraph. No `crates/` change; the push still goes in the
announced development mode (the qualification pointer is behind main, see integ22).

**C day 17.** memra#586: confirmed from source, the mirrored projection queues two `IoJob`s and charges
`prefetch_inflight` once while `worker_loop` released per job; the public stats function clamps the negative to zero.
Fixture first (`tools/test_cpu_expert_prefetch.sh`, cells `barrier`, `failure`, `parity`; it compiles the production
translation unit with its one `pread` renamed so a cell can hold a half, and reads the signed counter): before,
verbatim, `FAIL: REGRESSION memra#586 at barrier: primaries complete, alternates held: inflight_signed=0 expected 3`
and `... at failure: primary half complete, alternate half held: inflight_signed=0 expected 1`, summary
`cpu expert prefetch accounting tests: 9 FAILURE(S)`; after the one-hunk fix (the `fetch_sub` moved inside the
final-half branch, released once per projection, success or failure): `ALL GREEN` locally and on the target rig's
host (`pro-single-day17/cpubank/`, collector validate rc=0); `memra native CPU quant check: ALL GREEN` on both.
Pre-existing and stated, not fixed: the companion's submit-side exception path leaves earlier projections' charges
behind. Arena pair cell (`pro-single-day17/arena-pair/`, pre-registered and pushed before the run, N=5 per arm per
order, both orders, one 50 s lock hold, 33 to 51 C, 493 W peak under the 600 W limit), verbatim:
`ARENA-PAIR rule first_touch_page_o1=35.4 first_touch_page_o2=35.8 first_touch_arena_o1=0.0 first_touch_arena_o2=0.0
steady_demote_page=6.1 steady_demote_arena=6.2 demote_page=38.0 demote_arena=6.2 promote_page=11.3
promote_arena=6.6 promote_excl_page=4.5 promote_excl_arena=0.4 N=5/arm/order pooled=10 orders=2 window=50s
temp_c=33..51 power_max_w=493 power_limit_w=600.00 W identity=ok integrity=ok -> arena_first_touch_absent;
arena_not_slower`, replay `ARENA PAIR REPLAY: PASS (18 checks)`. Reading: the day-16 first-touch step was the fresh
pinned region, moved to a 1.3 s boot reserve under the arena; steady-state demote equal; the promote-share
difference is a census item, not a mechanism claim. WC item closed by pointer to `PINNED-DESTINATIONS.md` (ruling
23); the door's review still owes its cost with cached destinations on the target card (A day 15 task 2).

**B day 21.** memra#523 map: (1) newest turn fits (#596, twin gate V2/V4); (2) default re-decided, LRU naked, SLRU
door deleted (#598, #609); (3) the 8-turn twin gate held literally today on both cards, verbatim
`PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7
lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8
identity_ok=8/8 grid_ok=21/21 grid=32 off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` (the clause is
equality to the previous published entry rather than `>= prompt_tokens`, because of the capture law); (4) reclaim
truth (#588, evict-reclaim gate V1/V3). Lead reading: every item is held by a landed mechanism and a gate; #523
closes with this integ. memra#427: reproduced on the tip on both cards with byte-identical digests
(`9280 + 16: logits_sha=35bd15f063bfd5ba DIFFERS`; tails 17, 31, 32, 48 to 208 `ok`); the two capture-site fixes
did not change it; existing rollback seams keep the difference; the chunk arms classify it: under
`MEMRA_PRIME_CHUNK=32` the one-call prime's own 16-row tail chunk digests to `35bd15f063bfd5ba`, the default run's
16-row suffix digest, so the restore is exact and a prime chunk of exactly `PRIME_MIN_T` rows is its own numeric
program, in cold primes whose schedule ends in one as well. The kernel is not yet named (B day 22, running). Not
fixed: both fix shapes move bytes outside the restore path and owe batteries (DAY21 2.4). memra#372: already landed
by #377 (`628521007`, v0.137.0: `AdmissionRestoreRoute::Dflash`, `cost_after_prefix_restore` with the min-prime
workspace rule; the cited exclusion is gone); three CPU plan tests pass on the merged tree; the bounded serving cell
is pre-registered and not run (a 429 reproduction needs a pre-#377 binary). Lead reading: #372 closes on #377's record
with this integ.

Battery (`integration-day12/integ23-cpu-battery/`, merged tree `b2d995870`, CPUQuota 1200 percent): fmt, portable
suites, memra-server suite, flags, publish, docs-registry and workflow-key censuses, the classifier teeth (18 arms),
public-boundary `check` (0 new), collector pytest, C's #586 fixture (`cpu expert prefetch accounting tests: ALL
GREEN`), C's arena replay (`ARENA PAIR REPLAY: PASS (18 checks)`), perf board, `git diff --check`, em-dash scan:
14 steps rc=0. No serve smoke (no engine change).

Revuto round 1 on #612 (two findings, both real, fixed by the lead in the integ): (1) the submit side had no
symmetric release: a later projection's `fstat`, `resize` or mirror `resolve` could throw after earlier projections
took their charge and annex claim and before `pool.submit`, leaving phantom in-flight work for the life of the
process (C had stated it as pre-existing). A `SubmitGuard` in `memra_cpu_expert_prefetch_v2` releases exactly what
the call took and is disarmed before the jobs are handed to the pool. Fixture cell `submit-throw` (second projection
on an unopenable fd): without the guard `FAIL: REGRESSION memra#586 at submit-throw: after the failed call:
inflight_signed=1 expected 0` (`cpu-prefetch-586-round1-red.log`), with it `submit-throw: PASS` and
`cpu expert prefetch accounting tests: ALL GREEN` (`cpu-prefetch-586-round1-green.log`). (2) the barrier cell's
diagnostic printf read two hook counters without the hook mutex; snapshot under the lock. `docs/TESTING.md` and the
script header name the fourth cell.
Revuto round 2: the round-1 guard released claims from `state->runtimes`, but the claim of the projection whose own
resize or resolve throws is taken before its runtime is pushed, so that key stayed `speculated` for the life of the
process. The guard now releases from a `claimed` vector filled the moment `begin_read` succeeds. Fixture cell
`submit-throw-claim` (O_DIRECT with a mirror map keyed on the mirror fixture's inode, so the source is absent):
round-1 tree `submit-throw-claim: key claimable after the failed call=0 expected=1` (`cpu-prefetch-586-round2-red.log`),
fixed tree `=1 expected=1`, `ALL GREEN` (`cpu-prefetch-586-round2-green.log`). Five cells now.

## integ24 (`lane/spill-integ24-20260921`): A day 15
Lane tip merged: A `b9eb3638b` (clean; no `crates/` change; the only tracked file outside `research/` is
`tools/pinned-host-reserve-bench.py`, extended with the chunked and THP arms, `cuMemHostGetFlags` read-back and the
huge-page fraction from smaps). Pushed in the announced development mode (qualification pointer behind main).

**memra#385 on the target card class** (one RTX PRO 6000 Blackwell Server Edition at 600 W, driver 580.178.04; host
30 CPUs, one NUMA node, 88.4 GiB, no swap, THP `madvise`, no hugetlb pool). Reserve size 53.0 GiB (`MemAvailable`
minus the engine's own 32 GiB `check_headroom` margin, read at each cell's start): the largest arena the engine admits
on this box; the 288 GiB figure from the B200 pair does not fit. Two collector cells, one lock hold each, N=5 per arm
per order, both orders, a correctness pass per arm at full size (byte-exact roundtrip, `cuMemHostGetFlags` = 3 in every
arm, write-combined bit absent). Regime: GPU 32 to 36 C, 32 to 107 W (idle card), loadavg 0.09 to 1.21. Verbatim:
`ARENA-RESERVE rule cell=arena-chunked candidate=chunked bytes=56916705280 n_per_order=5 pooled=10 alloc_ok=true
roundtrip_exact_single=true roundtrip_exact_candidate=true wc_bit_absent=true flags_read_ok=true portable_bit_set=true
hugepage_fraction=na hugepage_ok=na cand_below_single_pairs=7/10 medians_both_orders=true pooled_single_ms=8870.374
pooled_candidate_ms=8831.379 ratio=1.0044 floor=1.10 materiality=false candidate_arm=inconclusive` (8 threads, chunk =
bytes/8; the eight calls complete 1.1 s apart in every row: the driver serializes pinned allocation) and
`ARENA-RESERVE rule cell=arena-thp candidate=thp bytes=56969134080 n_per_order=5 pooled=10 alloc_ok=true
roundtrip_exact_single=true roundtrip_exact_candidate=true wc_bit_absent=true flags_read_ok=true portable_bit_set=true
hugepage_fraction=0.6361 hugepage_ok=false cand_below_single_pairs=10/10 medians_both_orders=true
pooled_single_ms=8876.902 pooled_candidate_ms=2885.637 ratio=3.0762 floor=1.10 materiality=true candidate_arm=void
(hugepage-requested-not-granted)`. Void under the pre-registered huge-page clause, applied as written: where the kernel
granted 100 percent huge pages (9 of 10 timed rows) the register call pinned 53 GiB in 2.87 to 2.91 s (3.1x today's
call); where it did not, 18.3 s at 64 percent (the first THP pin, page cache full of other lanes' artifacts) and 6.6 s
at 84 percent. Both replays `ARENA AB REPLAY: PASS`. Lead reading: no engine change; `chunked` is flat, `thp` is a real
3x when huge pages are granted and void by the rule because they are not always granted. A landing needs a hugetlb
pool or a pre-registered page-cache regime, `reserve` plus `Drop` plus a fall-back policy, and the 2x B200 decision cell
with the same harness. `cuMemHostAlloc(PORTABLE)` stays. #385 stays open with the receipt (A's comment).

**The HOSTPREFIX door's review input (task 2, C's harness on this tree):** `WC PAIR REPLAY: PASS (12 checks)`; demote
OFF 37.8 against ON 113.6 ms pooled (N=10; steady-state r5..r7 6.1 to 6.9 against 81.8 to 83.0), promote 11.4 against
88.7, promote minus inline demote 4.4 against 5.8; orders agree within 1.2 ms; texts identical across all four boots;
36 to 51 C, 491 W peak. Banked for the 2026-10-05 review, no verdict. Side effect stated by A: the sitting took the
box's page cache from 57 to 17 GiB.

Battery (`integration-day12/integ24-cpu-battery/`, docs battery): diff-check, perf board, flags census, docs-registry
census, workflow-key census, public-boundary `check` (0 new), the bench tool's `py_compile` and `--help`, A's replays on
both arena cells (`ARENA AB REPLAY: PASS`, rule clauses restating the verdict), C's `wc-pair.py` on A's pair cell
(`WC PAIR REPLAY: PASS (12 checks)`), em-dash scan: 12 steps rc=0.

## integ25 (`lane/spill-integ25-20260921`): B day 22 (and day 23 when it lands), the memra#427 fix
Lane tip merged: B `99af46bf2` (clean, on main `d6401ff70`). Engine source changes: `crates/memra-engine/src/lib.rs`
(a `prefill_rows` AtomicBool on `Engine`, `prefill_rows_scope()` RAII on the existing `ExactScope` guard,
`batched_tier_admits() = verify_exact_on() || !prefill_rows_on()` added as a conjunct to the batched weight-resident
MMVQ tier's admission `(2..=16).contains(&m) && fast && ...` in both `matmul` and `matmul_pre`),
`hybrid_forward.rs` (the scope armed at the top of `HybridModel::prime_layers`), the continuation gate no longer
exempting a 16-row tail, and a new diagnostic binary `qwen-a4-width-walk` (per-operation digests of a 16-row versus a
17-row chunk under the prime's own program). No new `MEMRA_*` read, no kernel, no third program: a prime chunk of
exactly `PRIME_MIN_T` rows now takes the `grid.y = m` dp4a program every wider chunk takes; decode and verify keep the
tier (the exact-16 batched-decode tier and the K = 15 verify ride it by the decode-parity law; `verify_exact` keeps
precedence). Pushed in the announced development mode; nothing here claims qualification.

**B day 22, the kernel named and the fix gated on the local 5090** (verbatim, `rtx5090-day22/`): (a) width walk
`WIDTH WALK scope=prime width 16 vs 17: 0 of 497 tensors differ`, `scope=prime width 48 vs 17: 0 of 497`, and the
bare control `scope=bare width 16 vs 17: 96 of 497 tensors differ; tensor names: {"ssm_alpha", "ssm_beta"}`; the
named site `scope=prime layer 00 ssm_beta qtype=NVFP4 in_f=5120 out_f=48 width 16 vs 17: rows_differ=0/16
maxabs=0.000e0 ref_sha=e40f0aeaaec76762 sha=e40f0aeaaec76762 same` (bare `sha=0a984de6e2ed5fd0 DIFFERS`): the b16
batched-MMVQ tier at m = 16 on the `out_f < 128` GDN projections. (b) continuation gate, seven arms, every split `ok`,
`A4 CONTINUATION GATE: PASS` each: 9296 one call `14ab5f8b365dbd71`, `9280 + 16: logits_sha=14ab5f8b365dbd71 ok`,
48..208 ok; `MEMRA_PRIME_CHUNK=32` one call `14ab5f8b365dbd71` (was `35bd15f063bfd5ba`); `MEMRA_NO_BATCHED=1`
identical to default. (c) `kernel-check` both manifests `ALL GREEN (109 cells, 10 skipped)`. (d)
`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (drafter absent on this rig, WARNING branch as day 19). (e) twin gate
`PREFIX-NEWEST-TURN-FITS: ... V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS`. (f) base tree (the three source hunks
reverted) versus fix: default 9296 `14ab5f8b365dbd71` on both, 9297 `e641952cac19e526` on both; `MEMRA_PRIME_CHUNK=32`
base `35bd15f063bfd5ba` versus fix `14ab5f8b365dbd71` (the wide-chunk digest); base `9280 + 16: 35bd15f063bfd5ba
DIFFERS`. The fix moved only the 16-row chunk program. Stated by B: day 21 used one artifact (the served mint); the
calibrated A4 artifact of #427's table is on neither rig. Target card (`pro-single-day22/`): the continuation gate's
seven arms digest-identical to the local table; `kernel-check` 362 cells OK, 0 FAIL, 15 SKIP but
`MISSING REQUIRED CELL DUAL-BATCHED-AUX` (the 27b manifest's one cell needs the 9B artifact, absent on the box), so
the target-card kernel-check line is owed. Also owed before main and assigned to B day 23: `run-spec` K=1..8 with the
MTP drafter on the target card, `run-gen` argmax on 16-token and 16-mod-4096 prompts on both cards, the
`docs/TESTING.md` sentence, and the scope gap (`prime_layers_gemma` and `step35_prime_cache_batch` do not take the
scope: stated or armed). memra#445 map posted by B (section 1 closed by #588; section 2 open behind #151, gemma prefix
snapshot refused by the SWA flat-history layout; section 3 open on the gemma side, needs a pre-registered c=1/4/8
ladder on the target card). Lead ruling 25: the fix does not reach main until every owed gate above is green and
verbatim in this record; a red one keeps the lane as `wip:` and the integ waits.

**Ruling 26 (2026-09-21 19:5xZ).** While B's fix was on the lane, PR #614 (`653c997f4`, another session, merged
19:46Z, record `research/prime-tail16-20260921/`) fixed the same defect on main by a different mechanism:
`Engine::small_m_tier_max()` is 16 inside the verify-exact scope and `PRIME_MIN_T - 1` otherwise, and both general
entries `matmul` and `matmul_pre` use it as the batched small-m tier's ceiling; `matmul_decode_exact*` keep their own
`2..=16` tiers (they are the decode class). Its gate: the continuation gate on the 9B NVFP4 and on the served mint,
every split `ok`, the one-call digest unchanged; a `MEMRA_CI_CONTGATE` arm in `tools/local-ci.sh`. It covers every
caller of the general entries, so the scope gap B named (`prime_layers_gemma`, `step35_prime_cache_batch`) is closed by
construction. B's `prefill_rows` scope is a second mechanism for the same behaviour and covers fewer sites: main's
mechanism stands, the scope is dropped in the merge (lib.rs, hybrid_forward.rs, the gate binary take main's side), the
width-walk diagnostic stays without its scope arm, and B's gate receipts (width walk, continuation gate, kernel-check,
hit gate, twin gate, base-versus-fix identity, the target-card table) become the evidence that guards #614's mechanism,
re-run on the merged tree. The integ25 CPU battery and smoke run on B's tree (`f35bed45e`, 13 steps rc=0 except the
memra-server suite's known scheduler-sensitive `darklane::tests::stop_mode_full_cycle_launch_yield_resume_shutdown`,
`left: 1 right: 2` on the yield counter under the battery's load, 3 of 3 green alone; `serve-smoke: 0 failed`) are
superseded by the battery on the merged tree and kept as receipts.

**C day 18 (tip `29fe66640`, merged into integ25): the MoE slot cache door's decide-by inputs and the HOSTPREFIX
review table.** New diagnostic binary `hash-micro` (`crates/memra-engine/src/bin/hash_micro.rs`: the engine's
`memra_tier::contracts::checksum` over cached pinned, write-combined pinned and heap bytes; no engine path, no flag).
Item 4 (overlap, target card, N=5 per arm per order, both orders, one 976 s hold, 36 to 40 C, 189 W peak, tape
identical), verbatim: `OVERLAP-PAIR rule decode_off_s=0.408 decode_on_s=2.343 decode_ratio=5.743 ratio_o1=5.694
ratio_o2=5.833 off_range=0.407..0.409 on_range=2.301..2.528 door_cost_ms_per_decode_token=60.47
steady_mb_per_token=43.5 door_cost_ms_per_staged_MB=1.390 misses_off=[8211] misses_on=[18195] reads_on=[22077]
evictions_on=[12091] install_on_s=74.84 forward_off_s=0.65 forward_on_s=6.28 forward_ratio=9.627 N=5/arm/order
pooled=10 orders=2 temp_c=36..40 power_max_w=189 power_limit_w=600.00 W identity=ok integrity=ok ->
sync_miss_path_slower`; replay `DAY18 REPLAY overlap: PASS (9 checks)`. Reading: the synchronous miss path is the
door's only miss path at the 8 GiB budget (decode 5.7x slower with the door ON, install 75 s per process at this
artifact); it is the promotion blocker, and a delete decision at the decide-by deletes the whole door. Item 3 hash
lock, both cards: `HASHLOCK rule door_exit=1 sha_mismatch_line=True door_lines=0 control_exit=0 control_match=True
... -> hash_lock_refuses` (`PASS (7 checks)` each); scale admission is a CPU proof only (`memra-gguf expert_banks` 10
passed), no scale-bearing artifact on either rig, native cell pre-registered. Item 6, both cards: `SERVERDOOR rule
ready=True request_ok=True door_lines=0 flag_refused=False flag_silently_accepted=True ->
door_unreachable_in_serving`; hygiene finding, queued for a lane: `memra-server` rejects no unknown argument at all.
Item 1: census (every `with_moe_cache` caller is a MoE layer function on whichever thread walks the layer; no PP gate
binary carries the installer) plus the CPU proof `memra-tier owner_proxy` 4 passed (`WrongOwner` from a spawned
thread). Hash micro-cell, both cards (`PASS (8 checks)`): target host `cached_ms=77.922 wc_ms=1698.063 heap_ms=77.990
... wc_over_cached=21.792`; local host `cached_ms=37.339 wc_ms=1431.613 ... wc_over_cached=38.340`. HOSTPREFIX review
table appended to `HOSTPREFIX-DOOR.md`: banked (identity and failure gates, contract fault gate 40 ok, the review-round
cells, A's tenant arm, A's cached pair, the hash micro-cell, the arena pair), one arithmetic reading (one cached hash
pass, 77.9 ms, equals A's steady demote delta, 76 ms, so the ticket lifecycle is below resolution), two census questions
named and not resolved, and the missing list (arena lease handoff, DFlash tail slice, verify digest v3, the pool-full
line, an RTX 5090-class pair). Replay correction stated by C: v1 of `day18-replay.py` counted a bare substring the
refusal line carries; corrected to the pre-registered bracketed tags before any other result was read, no threshold
moved. `pro-single-day18/.gitattributes` marks receipt logs `-whitespace` (precedent `research/ttft-20260808/`).

**B day 23 (tip `5d9e61d3f`, ruling 26 executed): the gates re-run on main's mechanism, both cards, verbatim.** Lane
merged `origin/main 653c997f4`; `lib.rs` is main's `small_m_tier_max()` with the lane's `prefill_rows` field, scope,
`batched_tier_admits` and both admission conjuncts removed; `hybrid_forward.rs` arm removed; the continuation gate is
main's; `qwen-a4-width-walk` kept as a one-arm plain-program diagnostic. Against main the lane's non-research diff is
that binary plus one `docs/TESTING.md` pointer sentence. Local RTX 5090 (`rtx5090-day23/`): `WIDTH WALK width 16 vs 17:
0 of 497 tensors differ; tensor names: {}; sites: []` (48 vs 17 also 0); seven arms every split `ok`,
`A4 CONTINUATION GATE: PASS` x7, `9280 + 16: logits_sha=14ab5f8b365dbd71 ok`, chunk-32 one-call `14ab5f8b365dbd71`,
digest for digest the day-22 table; `kernel-check` both manifests `ALL GREEN (109 cells, 10 skipped)`;
`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`; twin gate `... V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS`; run-gen
`prefill argmax=271 decode argmax=271 ... MATCH` (std), probe 90 tokens MATCH plus batched-prime MATCH, p16 (16 tokens)
`23314 ... MATCH` plus batched-prime MATCH, p4112 (4112 tokens) `84728 ... MATCH` plus batched-prime MATCH; run-spec
K=1..8 (embedded NextN drafter, `nextn=1`) `=== SELF-CONSISTENCY PASS ===` on probe, p16, p4112, every K
`self-consistency: PASS (identical to plain target)`, acceptance above 0. Target card (`pro-single-day23/`): step35
manifest alone `ALL GREEN (109 cells, 6 skipped)`; both manifests `ALL GREEN (109 cells, 6 skipped)` with
`DUAL-BATCHED-AUX [NVFP4 rp] out=48 m=3: bit-bad=0/0 OK` (the 9B staged from the verified local copy, SHA-256
`52c9cceb...` equal to five repo receipts, into the lane's own dir; `/root/artifacts`, manifests and checker untouched);
run-gen and run-spec argmaxes, accepted and drafted counts and verdicts identical to the local card; the seven-arm
table digest for digest. Scope gap: closed by construction by #614; B's site reading in `PRIME-MIN-T-DECISION.md`
"Superseded by #614": `step35_prime_cache_batch` passes `m = total = sum(ts)` (16 at B=1) with `attn_gate` at `out_f`
64 or 96 (below the GEMM floor, so it could reach the tier), `prime_layers_gemma` passes `m = t` but every gemma
quantized projection has `out_f >= 128` (width-safe by census only); no Step-3.7-Flash GGUF on either rig, so the
step35 split cell stays owed as a confirmation. Lead reading: every gate ruling 25 named is green and verbatim above;
memra#427 closes with this integ on #614's mechanism and these receipts, the step35 split confirmation stays as a note
on the issue's close.

**Lead slices in integ25.** (1) `research/INDEX.md` on main carried a second stray diff3 base marker
(`||||||| parent of 8faa37ca4 ...`, from #614's rebase); removed, every row of every parent present (set difference
against main, B and C: 0 missing). (2) A conflict-marker census, `tools/check-conflict-markers.sh` (all four marker
kinds over tracked source, docs and data; receipt logs and raw dirs excluded; no skip switch), wired into the pre-push
hook after the docs-registry census and into the CI gates job, teeth `tools/test_conflict_markers.sh` (7 arms). Its
first run found a third marker on main: `research/tune-data/perf-ci.jsonl:1147` (a non-JSON line in the append-only log
the perf gate parses, from the #604 rebase); removed, every remaining line parses. Ruling 27: a marker line in a
tracked file is a push refusal from now on, and a hand-resolved merge is followed by this census before its commit.

Final battery on the integ25 tree (`integration-day12/integ25-cpu-battery/`, tree `5a14f5188`, CPUQuota 1200 percent):
fmt, portable suites (335 passed, 0 skipped), memra-server suite (green), clippy, flags, publish and docs-registry
censuses, collector pytest (87 passed), memra-engine CPU lib tests (518 passed), engine clippy `-D warnings`, perf
board, diff-check, then the marker census and its teeth, workflow keys, em-dash scan: 16 steps rc=0. Local 5090
`tools/serve-smoke.sh` (`integ25-serve-smoke-5090/`): `serve-smoke: 0 failed` (gemma4 and Q35 arms SKIP, models absent
on this rig; a lane's continuation-gate process shared the card during the window, recorded in `window.txt`).

## integ26 (`lane/spill-integ26-20260921`): A day 16, memra#536 census, stall cell, prime cancellation point
Lane tip merged: A `a71bd8db9` on main `e2e9e294a` (#616). INDEX.md conflicted because A had inherited #614's stray
marker: A's day-16 row kept, the marker dropped, every row of both parents present, the census clean. Engine source:
`crates/memra-engine/src/progress.rs` (`PrimeCancelScope`, a thread-local predicate installed for one prime call and
restored on drop; typed `PrimeCancelled { chunk, rows_done, rows_total }`; `prime_cancel_point` answering `Ok` with no
scope or at the last chunk), `hybrid_forward.rs` (the check at the chunk boundary of the three sequential walks: the
serial chunk walk, the GEMM chunk loop of `step35_prime_cache_batch`, the single-engine hyper range walk; the
pipelined PP walks and `prime_cache_batch` keep the tick-top sweep, stated in the module note), `memra-server
worker.rs` (the scope installed around the one prime call with `EventSender::is_closed` as the predicate;
`prime_cancelled_abort` downcasts the typed error, prints one `[prime] cancelled at chunk ...` receipt line and
retires the session as aborted: no park, no publish, the half-primed cache returns to the pool; `abort_log` gains
`fed`), `tools/prime-cancel-gate.sh`. No flag, no new `MEMRA_*` read, no numeric change: the check either lets the walk
continue exactly as before or returns after a completed chunk and before the next; a cancelled prime returns no logits,
so the capture sites (after an `Ok` prime only) are never reached.

**Census** (`OWNER-THREAD-CENSUS.md`): every class (prime, decode, D2D capture, D2D restore, D2H demote, H2D promote,
trim) runs on the one worker thread inside the tick. Prime: one `prefill_tick` per tick (1024 rows; 8192 for a sole
fresh request; the whole prompt for the monolithic class), per-internal-chunk D2H of logits, cancellation only at the
tick-top sweep (before today). Demote OFF: host-blocking synchronize per plane. Promote OFF: no host wait, owner
stream. Under the host-contracts door `CudaTransfers::new(owner, ..)` keeps the worker's owner stream as its only stream
and pins the thread; both routes `synchronize(&ticket)` plus owner drain before `retire`: the door moved ownership and
receipts, not the copy. Trim: two synchronizes, evictions and device trim in one `TrimPools` call.

**Stall cell** (one RTX PRO 6000 Blackwell at 600 W, `MEMRA_SERVE_SPEC=0`, N=5 per arm per order, both orders, replays
PASS, zero errors), verbatim fragments: `stall-prime ... idle_p50=13.5 idle_p99=14.9 ... arm_p99=295.7 arm_max=315.9
stall_median=301.5 stall_min=285.6 stall_max=302.5` (a 5122-token prime beside a decoding tenant: five stretched ticks
of 286 to 316 ms per run, one per 1024-row chunk); `stall-demote-off ... arm_p99=87.6 arm_max=132.0 stall_median=117.5`
(server demote 37 to 43 ms inside a 131 ms tick); `stall-demote-on ... arm_max=207.6 stall_median=193.5` (113 to 118 ms
inside a 207 ms tick); `stall-promote-off ... arm_max=133.6 stall_median=85.0`; `stall-promote-on ... arm_max=211.2
stall_median=162.8`. Unattributed and stated: a second 88 ms tick per demote intruder, present without a demote too.

**Cancellation gate**, verbatim: target card `[prime] cancelled at chunk 2 (768 of 7488 rows of this take primed; fed 0,
queued 20, prompt 7508, model "gate")` and `PRIME-CANCEL GATE: PASS (disconnect_ms=300 words=6000
cold=f243df4517b99525 warm=f243df4517b99525)`; local 5090 (Qwen3.5-9B) cancel at chunk 4 (1280 of 7488 rows),
`PRIME-CANCEL GATE: PASS`. One program on the changed binary, both cards: `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`
(61 ok) and `A4 CONTINUATION GATE: PASS`. Two false starts A kept as records: the hit gate has no `--external-lock`
arm, and its stop matches the server by the name `memra-server`, so a renamed binary left A's own server on the port on
both rigs (stopped by pid; reruns used the canonical name). Design note `OWNER-THREAD-OFFLOAD.md`: Move 1 (the door's
D2H and H2D on a second stream, tick-top polls, `Demoting`/`promoting` states, about 2 agent-days) and Move 2 (a D2D
contract, about 4 agent-days after Move 1), each with its decision cells; nothing started. #536 comment posted, issue
open.

## integ27 (`lane/spill-integ27-20260922`): C day 19 (memra#617) and B day 24 (memra#476, #524 gap)
Lane tips merged: C `cb53b91b0`, B `6b9574473`, on main `a18c936a3` (#618). INDEX.md conflicted twice on the shared
marker deletion; both sides' rows kept, every parent row present, census clean. Engine-crate changes are in
`memra-server` only: `argv.rs` (new), `lib.rs` (the validator as the first statement of `serve_with`),
`admit_predict.rs` (`RequestCharge`), `worker.rs` (the two predictor sites), `tests/argv_boot.rs`; docs
`SERVING.md` (one sentence), `FLAGS.md` (the two admission door rows describe the fuller charge; no new read).

**C day 19, memra#617.** `argv::validate(&[String]) -> Result<(), String>` admits exactly the documented command line
(`--version | -V | --gen-key <tenant> [--lane ...] [--rate-limit N] [--keys <path>] | --revoke-key <prefix> [--keys
<path>]`), refuses any other token with `[server] FATAL: unknown argument "<token>" refused at boot; accepted
arguments: ...` and exit 2 before the version print, the first environment read and any device work; a key modifier
without a key command is refused too; a missing value is left to `run_cli` as before. Tests: `argv::tests` 7 and
`tests/argv_boot.rs` 7 booting the real binary. Lead check: no tracked launcher (`tools/`, `docs/`) passes
`memra-server` any argument outside that set. Serverdoor cell on the fixed tree (local 5090, `serverdoor19b`,
`PASS (16 checks)`), verbatim: `SERVERDOOR19 rule flag_exit=2 flag_refused=True flag_booted=False flag_ready=False
flag_door_lines=0 envon_ready=True envon_request_ok=True envon_door_line=True envon_exit=0 envbad_exit=1
envbad_refused=True envbad_ready=False -> flag_refused; env_door_documented`; the first cell (`serverdoor19`, the
day-18 shape without `MEMRA_KV_HOST_MB`) is kept as `FAIL` verbatim (the door printed its documented no-tier line), rule
untouched, the b cell pre-registered before its run. Not run on the target card (argv admission precedes any device
statement). Arena lease handoff: scoped wider than one bounded change (an arena-backed lease type in `tier_transfer.rs`,
worker charge-once accounting, and a lead ruling on one budget or two); nothing implemented; the DFlash tail slice
pre-registered only. Ruling 28: the arena handoff stays scoped until the HOSTPREFIX decide-by review; the budget
question (one pinned budget or two) is decided there with the door.

**B day 24, memra#476.** Arithmetic first (DAY24 section 1): the physical gate charges `cost = ctx(C) + W(P) + A + D`
(`W(P - R)` after a retained-prefix plan), the predictive book charged `kv_hat = ctx(P + L + 8) + A`, so `cost - kv_hat =
[ctx(C) - ctx(P+L+8)] + W(P) + D`: the bracket is by design, `W` and `D` were in the real book and not the predictive
one. Fix: `RequestCharge::from_physical_cost(cost, ctx(C), A, D, ctx(P+L+8)).total()` at both predictor sites (verdict
reads the cold cost; booking takes the final restore- or eager-adjusted cost the real book takes); no new numeric
program, no flag; the lock tests unchanged and green. Before (local 5090, Qwen3.5-9B NVFP4, shadow mode), verbatim:
fourth concurrent admit `kv_hat=217983412 booked_bytes=615855840 booked_real=5156718304 inflight=3` (8.37x); `a0 P=6006
cost_exact=1702520120 kv_hat=193525048 real/shadow=8.80x`; `A: max |residual~| = 1672223244 bytes`. After, same script
and tolerances: `a0 cost_exact=1702520120 kv_hat=1701213496` (difference `1306624 = 14848 x 88`, the bracket to the
byte); fourth admit `booked_bytes=5152798432 booked_real=5156718304`; `A: max |residual~| = 479880 bytes` PASS (inside
the 1 MB line grain); `C: Overloaded/OOM lines = 0: PASS`. Pre-registered `B: max_underbook_real=2550136832 <=
floor=1610612736: FAIL` before and after: the pool keeps freed blocks mapped, so at inflight 0 the device delta is the
pool high-water while both books are 0 (in-flight only 822879944, inside the floor); recorded, not relaxed; the fix does
not touch the real book. CPU tests `16 passed`. Named, not done: outstanding-only `W` release at prime completion; the
physical book charges the shared prime slab per request (over-books 4-way by about 5.6 GB, conservative); the enforcing
door on the fuller charge needs its own cell; the retained-prefix arm did not trigger on this card. memra#524:
`run_boot_calibration` precedes `ready_tx.send(Ok` and `mark_ready`, so on the armed path readiness already waits for
the one warmup the server runs (ready to first completion 1437 and 1426 ms at a 500 ms poll); the gap is the
probe-skipped paths (`MEMRA_ADMIT_CALIBRATE=0`, plain-only serving, `MEMRA_ADMIT_RESERVE_MB`, probe failure) and shapes
the B=1 probe never walks, plus no `phase=warming`; a source-order CPU test lands; `tools/health-fault-gate.sh` arms (a)
to (f) pre-registered as a two-day lane. Both issues stay open.

Battery on the combined tree (`integration-day12/integ27-cpu-battery/`, tree `826386f28`, CPUQuota 1200 percent): fmt,
portable suites, memra-server suite (773 passed, the 7 `argv_boot` real-binary tests among them), clippy, censuses,
collector pytest, engine CPU lib tests, server clippy `-D warnings`, marker census, workflow keys, perf board: rc=0;
`git diff --check` tripped on trailing whitespace in the archived C-only summary (stripped). Local 5090
`tools/serve-smoke.sh`: `serve-smoke: 0 failed`. The C-only tree's earlier run is kept under `-ctree` (all rc=0,
smoke `0 failed`).

Revuto round 1 on #619, both findings real, fixed by the lead in the integ: (1) the argv refusal sat in `serve_with`,
the entrypoint deployment-owned binaries call with their own command line, so a deployment flag would have been
`exit(2)` before wiring was consulted; the refusal now lives in the stock `serve_main` only, `serve_with` is
argv-agnostic again, `argv::validate` stays public for a deployment binary that wants the stock set (module header,
`docs/SERVING.md` say so); the 7 real-binary tests boot the stock binary and still hold. (2) The `RequestCharge` comment
and test claimed the eager arm's cost is draft-adjusted; the worker never subtracts the draft state on that arm (it
books it, conservative, as the real book does); the comment, the module docs and the test's name and note now say that.
Server clippy `-D warnings` and the memra-server suite (773 passed, 7 boot tests) green on the round-1 tree.
While round 1 was up, main took #615 (`81d75c457`, the unknown or retired `MEMRA_*` boot refusal) and the PR turned
`CONFLICTING` on INDEX.md, which is why GitHub created no `pull_request` run for two pushes (a conflicting PR gets no
merge ref and no Actions run; nothing is reported). Merged main into the integ (INDEX.md both rows, census clean) and
re-ran on the merged tree (`integ27-cpu-battery-merged615/`): fmt, memra-server suite (773 passed), server clippy
`-D warnings`, engine CPU lib tests, flags census, marker census, portable suites: rc=0; `git diff --check` tripped
only on the battery's own summary whitespace (stripped); local 5090 serve-smoke `serve-smoke: 0 failed`
(`integ27-serve-smoke-5090-merged615/`).

## integ28 (`lane/spill-integ28-20260922`): C day 20, memra#365 (the DFlash prefill tap bounded)
Lane tip merged: C `f41ccd874` on main `8e94a480b` (#619), clean. Engine change in `crates/memra-engine/src/dflash.rs`
(the standalone qwen entry `generate_spec_dspark` primes through the existing serving walker `prime_dflash_taps` with
its one 4,096-row chunk sink plus the 256-row carry, instead of allocating a whole-prompt tap sink and holding the
monolithic prime's whole-prompt hiddens; the carry copy's `expect` becomes a typed refusal when a row would be read
before the trunk wrote it or beyond the chunk the trunk wrote; one CPU test on the chunk bookkeeping over thirteen
prompt lengths) and a receipt line in `dspark_q38_gate.rs`. No kernel, value, flag, dependency or serving-path change;
the serving cold and resume paths were already bounded by #370.

**Census (DAY20, before any change).** Tap = `n_taps x hidden` f32 per prompt token = 102,400 bytes on the 27B with
the five-tap q38 DFlash2 export; `131,070 x 102,400 = 13,421,568,000`, the issue's number exactly; written per tapped
layer per prime chunk by `HybridModel::dflash_tap`, read once by drafter ingestion in 256-row windows in prompt order,
freed after ingestion. The whole-prompt sink survived only in the standalone entry points (`generate_spec_dspark`,
and `generate_spec_dflash` capped at 2,048 rows by its window assert); the qwen one also held the monolithic prime's
whole-prompt hiddens (2.68 GB at 131k).

**Before, verbatim** (local RTX 5090 Laptop GPU, `tapladder20`, `dspark_q38_gate`, 27B NVFP4/Q5K plus q38 DFlash2,
ngen 32, `MEMRA_ALLOC_TRACE=1`): 8,194 tokens `EXACT`, `[alloc-trace] 839065600 bytes from
crates/memra-engine/src/dflash.rs:3977`, `acceptance 28/28`, `spec_sha256=72cf1985...3162ef22`, peak 21,607 MiB;
16,382 `EXACT`, tap `1677516800`, `28/28`, `0d6449b2...f047492e`, peak 22,759; 32,759: tap `3354521600` allocated,
then `[alloc-trace] 100663296 bytes from crates/memra-engine/src/lib.rs:33856` and `Error:
DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")` (the GDN scan transient starved by the resident tap);
65,507: `[alloc-trace] 6707916800 bytes from crates/memra-engine/src/dflash.rs:3977` then the same OOM (the tap
itself); 131,004: the gate's plain control refused first (`hybrid_forward.rs:6578`), before any DFlash allocation.

**After, verbatim** (`tapladder20b`, same rungs, `DAY20 REPLAY tapladder20b: PASS (9 checks) -> identity; bounded`):
shared rungs same `spec_sha256`, `spec_len=32`, `acceptance 28/28`, `EXACT`; `[dflash-taps] base=0 rows=8194
max_chunk_rows=4098 chunk_tap_bytes=419635200 carry_bytes=26214400 former_full_tap_bytes=839065600`, `rows=16382 ...
chunk_tap_bytes=419430400 ... former_full_tap_bytes=1677516800`; 32,759 and 65,507 now `EXACT`
(`chunk_tap_bytes=419430400 carry_bytes=26214400`, former `3354521600` and `6707916800`), peaks 22,215 and 23,609 MiB
where the before died at 23,961 and 23,958; 131,004 refuses in the plain control as before (the named resource moved,
as pre-registered). Gates on the changed binary: `test result: ok. 57 passed`, clippy `-D warnings`,
`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, `A4 CONTINUATION GATE: PASS`. Lead reading on the one-program law: the
standalone entry moved from the monolithic prime to the chunked serving walker, and the two shared rungs read the
same `spec_sha256` and acceptance on both sides, which is the identity proof the law asks for at those lengths; the
gate's identical digests are the receipt. Named, not done: the 131k rung needs a control that fits this card; no
before-side `[dflash-oracle]` digest existed for the standalone path (today's are the baseline); not run on the target
card (device-independent bookkeeping). memra#365 comment posted; the issue's serving 128k and 262k cells stay with
#370 and #377; the issue stays open for those.

Battery (`integration-day12/integ28-cpu-battery/`, CPUQuota 1200 percent): fmt, portable suites, memra-server suite,
clippy, censuses, collector pytest, engine CPU lib tests, engine clippy `-D warnings`, marker census, workflow keys,
perf board, diff-check: rc=0; C's replay first ran without its arguments (rc=1, my invocation), then with the
documented before and after evidence dirs and the five rung word counts: `DAY20 REPLAY tapladder20b: PASS (9 checks)
-> identity; bounded`. Local 5090 `tools/serve-smoke.sh`: `serve-smoke: 0 failed`.

## integ29 (`lane/spill-integ29-20260922`): B day 25, the health, readiness and lifecycle fault gate (memra#524, #526)
Lane tip merged: B `fe622cb5b` on main `124dacc76` (#620), clean. Changes in `memra-server` (`health.rs`:
`PHASE_WARMING` between LOADING and IDLE, `mark_warming()` by `compare_exchange` from LOADING only so a serving worker
is never demoted, `live()` names it; `worker.rs`: `run_boot_calibration(.., &health)` marks warming after every skip
return and before the probe generation, `mark_ready` ends it; a CPU test on the order and the day-24 source-order test
extended), `tools/health-fault-gate.sh` (new), `tools/local-ci.sh` (the gate after the smoke, `MEMRA_CI_HEALTH_FAULT=0`
skips), `docs/FLAGS.md` (the skip row; `MEMRA_PANIC_AFTER` and `MEMRA_ADMIT_CALIBRATE` rows say what the gate uses
them for), `docs/SERVING.md` (phase table), `docs/TESTING.md`. No new fault hook anywhere: arm (d) uses the existing
`MEMRA_PANIC_AFTER` door, (e) a PATH-shadowed `nvidia-smi` in the server's environment with the documented
`MEMRA_GPU_WATCH_S` and `MEMRA_GPU_PROBE_TIMEOUT_S`, (f) SIGTERM. There is no `/healthz` route; the gate asserts
`/health` (the same handler as `/livez`) and says so.

**Verdicts, verbatim, target card** (one RTX PRO 6000 Blackwell, collector, 34.9 s; the two local 5090 runs read the
same in every asserted field): `HFG (a) readiness-before-probe: probe_done_line=30 listening_line=33
pre_ready_samples=17 pre_ready_codes={000:17,503:0} ready_samples_not_ready=0 ready_while_warming=0
first_ready_phase=idle -> PASS`; `HFG (b) ready-then-first-request: readyz_before=200/ready request_http=200
finish_reason=length completion_tokens=32 request_ms=143 ready_to_completion_ms=229 readyz_after=200/ready (N=1, not
a timing claim) -> PASS`; `HFG (c) probe-skipped c-calibrate0 [MEMRA_ADMIT_CALIBRATE=0]: skip_line=16
probe_done_line=0 warming_samples=0 ready_ms=1095 first_request_http=200 first_request_ms=166 (N=1) -> DOCUMENTED
(ready without warmup; the first request pays the cold route)` (and the same DOCUMENTED reading for `c-spec0` and
`c-reserve`); `HFG (d) panic-respawn-truthful-health: trigger_http=200 health_503_quoted_after_ms=192
readyz_while_dead=503/not_ready/dead untruthful_samples=0 recovered_after_ms=3469 generation_after=1
request_after_http=200 health_after=200/ok log_panic_line=44 log_respawn_line=45 -> PASS`; `HFG (a2)
warming-phase-on-respawn: not_ready_phase_sequence=dead>loading>warming loading_samples=5 warming_samples=4
loading_then_warming=true -> PASS`; `HFG (e) fatal-fault-not-cleared-by-timeout: watch_on_line=5
control_200_samples=3/3 latched_after_ms=2727 shim_answering_again_for_ms=8157 health_after=503/unhealthy
readyz_after=503/not_ready health_200_after_clear=0 critical_lines=1 request_http_while_latched=200 (observed, bounded,
not asserted) -> PASS`; `HFG (f) sigterm-drains-and-flips-readiness-first: stream_frames_at_sigterm=5
readyz_during_drain=503/not_ready retry_after_s=60 health_during_drain=200/draining new_request_http=503
new_request_code=draining new_request_retry_after_s=60 stream_curl_rc=0 stream_frames=258 stream_done=1
stream_finish_reason=length stream_finished_after_sigterm_ms=886 exit_code=0 exit_after_sigterm_ms=1074
drain_complete_line=48 deadline_hit_line=0 -> PASS`; `health-fault-gate: arms=a,b,c,d,e,f pass=6 documented=3
fail=0`. Two harness attempts kept as labelled failures (`attempt1-harness-clock-bug`: uutils `date +%s%3N` prints
nanoseconds on this rig; `attempt2-harness-jget-bug`); no assertion changed after a result. Pre-registered only: the
step-OOM arm (`MEMRA_STEP_OOM_FAULT`) and the client-disconnect arm. Owner question, recorded, not asserted: the request
path under a latched gpu fault served 200. A first boot binds after the probe, so `warming` is observable over HTTP on
the respawn window only (by design, unchanged). Comments on #524 and #526; both stay open (the release battery decides
what it requires).

Battery (`integration-day12/integ29-cpu-battery/`, CPUQuota 1200 percent): fmt, portable suites, memra-server suite,
clippy, censuses, collector pytest, engine CPU lib tests, server clippy `-D warnings`, marker census, workflow keys,
perf board: rc=0; `git diff --check` tripped on B's HTTP header receipts ending in the protocol blank line, marked
`-whitespace` by a `.gitattributes` in the receipt dirs (the bytes stay). Local 5090 `tools/serve-smoke.sh`:
`serve-smoke: 0 failed`; the health fault gate on this tree (`integ29-health-fault-gate-5090/`):
`health-fault-gate: arms=a,b,c,d,e,f pass=6 documented=3 fail=0`, receipts copied in.
Revuto round 1 on #621: the gate's default port 8186 was `serve-gemma4-batch-gate.sh`'s, the collision class the
2026-08-19 gate-integrity audit removed; moved to 8189 after a census of every port literal under `tools/` (8189 unused),
the port guard unchanged; the gate re-run on this tree at 8189 (`health-fault-gate-port8189.log`).

## integ30 (`lane/spill-integ30-20260922`): A day 17, memra#536 Move 1 first slice under the host-contracts door
Lane tip merged: A `b967b8d30` on main `dc192cd95` (#621); INDEX.md conflicted on the inherited stray marker, both rows
kept, every parent row present, census clean. Engine and worker changes, all reachable only with
`MEMRA_KV_HOST_CONTRACTS=1` (default OFF, decide-by 2026-10-05): `crates/memra-engine/src/tier_transfer.rs` (an
optional second CUDA stream owned by the same owner thread, `new_with_copy_stream`, `copy_stream()`, the D2H items of a
batch issued on it behind the producer fence's event with their completion event recorded there and no owner-stream
wait installed at submit, `synchronize_copy_stream` in `release_device` and `take_plane`); `crates/memra-server/src/
worker.rs` and `worker/host_glm.rs` (the demote route split into `host_kv_planes_submit_contract` and
`host_kv_planes_settle_contract` with the blocking wrapper kept; `PendingContractDemote`, `HostDemoteOutcome::Demoting`,
`host_demote_publish` at the tick-top poll as the run loop's first statement after `require`, planes back, retire,
acknowledge and `bind_tier_image`; settle-first in `host_demote_prefix_ref`, `host_promote_prefix_hit` and
`purge_tenant`; a hit on a `Demoting` entry is a cold prime; a tier latched off drops it unpublished; failures publish
nothing with the day-15 typed outcomes; `apply_flip_demote_fault` for the fault gate); five CPU tests; `docs/FLAGS.md`
door row describes the split. No new flag, no numeric change; `Promoting` named, not introduced.

**Gate lines, verbatim.** Target card (binary `ceaf238f...`): `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`
default and plain, door OFF and ON; `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (62 ok); `SPEC-ON-CACHE-HIT GATE: ALL GREEN
(qwen)` OFF and ON; twin gate `... V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` OFF and ON; failure gate `KV-HOST-SPILL
FAILURE GATE: 1 FAILURE(S)` both arms (the pre-existing `FAIL: pool-full refusal is LOUD and named`, carried since C's
day 16); unit cells `8 passed`. Local RTX 5090 (9B, binary `3a92efab...`): identity `ALL GREEN (teeth=0)` x4; hit gate
`ALL GREEN (qwen)` OFF and ON; failure gate `1 FAILURE(S)` x2 (the same line); fault gate `KV-HOST-CONTRACT-FAULT GATE: 5
FAILURE(S)`, all in `promote-reject` (the gate hardcodes `1 of 34 items` from the 27B; the 9B prints 18; gate unchanged,
stated); twin gate on the 9B `REFUSED: cohort promotion did not happen for 2800 tokens ...` both arms (the gate's cohort
shape), on the 27B artifact `-> PASS` OFF and ON. Lead reading: both reds on the 5090 are gate-shape items of the 9B
artifact and the pool-full line is a pre-existing red on both cards; neither is A's change; both are assigned to C
day 21 (running) with the rule that a gate match may move only to follow a cited deliberate message change.

**Stall verdict, verbatim** (one RTX PRO 6000 Blackwell, N=5 per arm per order, both orders, one lock hold, replay
PASS): `STALL rule cell=stall-demote-on arm=demote n_per_order=5 pooled=10 ... arm_p99=130.9 arm_max=163.4
stall_median=149.6 stall_min=75.9 stall_max=150.0 ... tenant_text_identical=True errors=0`; the pre-registered rule
against day 16 (ON 193.5, OFF 117.5) reads `toward_off`: the stretched tick is 163 ms against 208 ON and 132 OFF, the
copy time left the tick, the two receipt hashes remain. Two false starts A kept as receipts (a submission line carrying
the door's refusal marker text, reworded; a driver misreading the twin gate's `REFUSED: --out must be a new directory`
as a busy lock). #536 has A's receipt and correction comments; the issue stays open (the promote's turn and Move 2).

Battery (`integration-day12/integ30-cpu-battery/`, CPUQuota 1200 percent): fmt, portable suites, memra-server suite,
clippy, censuses, collector pytest, engine CPU lib tests, engine and server clippy `-D warnings`, marker census,
workflow keys, perf board, diff-check: 15 steps rc=0. Local 5090 `tools/serve-smoke.sh` (door OFF, the default):
`serve-smoke: 0 failed`. C day 21 (running while this integ was built; lands as integ31) found both 5090 reds to be
stale gates and re-read failure, fault and identity gates ALL GREEN on both cards with this slice merged.

Revuto round 1 on #622, one finding, real, fixed by the lead in the integ: in `host_demote_settle_with` the arm for
a pending demote missing its shell or its ticket took both fields before matching, so a SUBMITTED ticket without its
shell was dropped without retire, acknowledge or abort: the ledger's one in-flight charge and the registered planes
leaked silently and the tier would never demote again (fail open). The arm now fails closed: a submitted ticket is
settled and its sources retired through the engine where it is reachable and the tier latches off
(`SourceQuarantined`, the planes cannot come back without their shell); a shell without a ticket drops whole (`Failed`,
nothing was submitted). New CPU test `a_pending_demote_missing_its_shell_or_ticket_fails_closed` (both arms); the
six state-machine tests, the memra-server suite and server clippy `-D warnings` green on the round-1 tree.
Round 2, two findings, both real, fixed: (1) the settle-first call in `host_demote_prefix_ref` and in
`host_promote_prefix_hit` can latch the tier off, and each function's `armed()` gate had already been passed, so a
demote or a promote continued on a latched-off tier; both now re-check `armed()` after the settle (`Off` and a miss).
(2) the latched-off drop exit, the flip-fault exit and the fail-closed arm returned without
`waste_pending_reclaim(..)`, unlike every other demote exit, leaving a reclaim booked pending forever; all three book
it wasted now (the image carries the key and tokens when the shell is gone); the CPU test asserts it. Server clippy
`-D warnings` and the memra-server suite (780 passed) green, gated before the commit this time.
## integ31 (`lane/spill-integ31-20260922`): C day 21 (two stale gates) and B day 26 (memra#539 census, cell, design)
Lane tips merged: C `304e8235c` (carrying A's day 17, already in integ30), B `1ea29941e`. Non-research changes are
the two gate scripts only: `tools/kv-host-spill-failure-gate.sh` and `tools/kv-host-contract-fault-gate.sh`.
**C day 21, both reds were stale gates, not the server.** (1) The failure gate matched the insert-path line
`[prefix-host] skip demote: entry X MB > host budget B MB` (`HostPrefixCache::insert`), reachable only at
`MEMRA_KV_HOST_TENANT_PCT=100`; under the server default 50 (`49d1d6f65`, "50 BY DESIGN") the demote path refuses
before the D2H copy with `demote evaporated at the tenant share cap before the D2H copy: N tokens, X MB (50% of B MB,
...); reclaim refused: the image alone exceeds the share (X MB > B MB); nothing evicted` (memra#384, `405466cf7`);
lane D's day 8 was green only with `TENANT_PCT=100` set out of band. The gate now mirrors the server's parse of
`TENANT_PCT`, asserts the full anchored shape with bytes and budget for the arm in force, and adds
`prefix_host_tenant_rejects >= 1` under the cap; stricter, not looser, verified against banked A and D logs before any
run. (2) The fault gate hardcoded `1 of 34 items`; it now reads `items=N` from the server's first `contracts door D2H
receipt` and asserts the refusal's `1 of M` equals N, all 12 prior assertions kept, two added; finding: the plain
environment reads `items=32` on the 27B (no draft planes), so the hardcoded 34 would have failed the plain arm on the
target card too. Verdicts, verbatim, tree `9be3f7373` (A's slice merged): target card `KV-HOST-SPILL FAILURE GATE: ALL
GREEN` (default-off, default-on, plain-off, plain-on; 15 ok each), `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (default
`items=34`, plain `items=32`; 64 ok each), `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (all four arms); local
RTX 5090 (9B) identical verdict lines in all ten cells, reject cell `items=18` default and `items=16` plain. The
whole-budget arm (`TENANT_PCT=100`) was asserted by pattern against D's banked log only, not run. `HOSTPREFIX-DOOR.md`
review rows moved from pre-existing red to their verdicts. Ruling 29: a gate that reads red on both cards and both arms
is a lane item the day it is seen, never a "pre-existing" note carried across days.
**B day 26, memra#539.** Census (anchors on `1c66ff10e`): the cap rule at `worker.rs:21705-21745`: `max_tokens` bounded
gives `prompt + max_tokens + 8`; omitted gives the server context (`MEMRA_CTX`, else the checkpoint's 262,144) unless
`MEMRA_ADMIT_BY_MEMORY=1` (default OFF, decide-by 2026-09-23) gives `prompt + 8192 + 8`; registry deployments never
reach the open arm. Every session cache is born at `ctx_cap` (`Cache::new_inner` allocates K and V planes of `max_ctx`
rows per full-attention layer, never grown or trimmed); a hit D2D-copies `restore_len` rows into a fresh cap-sized cache
and shares nothing; retired caches park in the reuse pools at `cache.max_ctx` with no TTL. Bytes per token: 1,856 B per
full-attention layer (q8_0 K plus q5_1 V); Qwen3.5-9B 14,848 plain and 16,704 spec; the 27B mint 29,696 and 31,552,
confirmed by the server's own `[admission] request cost` lines. Cell (N=5 per arm per order, both orders; P and G
identical across orders on 44 of 45 requests per card), allocated over used, verbatim medians: target card (27B,
`MEMRA_CTX` unset) arm (i) `max_tokens` omitted `median=129.26 / 71.90 / 41.61` (8,271,167,488 B allocated per
request), arm (ii) bounded `1.00` on every row, arm (iii) warm `134.02 / 72.12 / 41.15` with `cached=0` in the
deferred shape (43 of 45 spec-boundary entries LRU-evicted inside an 800 MB budget whose derivation prints
`at MEMRA_CTX=8192`) and `cached=1440/3104/5760` in the warm cell; concurrency at 262,144 from 82.76 GB idle free: L0
`8 -> 49`, L1 `7 -> 35`, L2 `7 -> 27`. Local RTX 5090 (9B, `MEMRA_CTX=65536`): (i) `11.30 / 10.73 / 6.62`, (ii) `1.00`,
(iii) `11.79 / 10.62 / 6.35`; concurrency `8 -> 18`, `6 -> 11`, `5 -> 8`. Recorded, not fixed (B day 27, running):
the prefix-cache budget derived at `MEMRA_CTX=8192` when unset; about 30 GB retained on the target card after 45
sequential requests with none active (four parked whole-session entries at 8.27 GB). Design note
`KV-RESIDENCY-DESIGN.md`: recommended order (a) the existing admission-by-memory door as the bounded open default (0.5
agent-day, policy only), (b) grow-on-demand VMM planes (3 to 4 agent-days; receipts on both cards from the tier lane's
day 10, addresses never move so the one-program law holds by construction), (c) paged KV (10 to 15), (d) copy-on-write
sharing (3 to 5) only on shared-prefix workload evidence; reason: the open-arm ratio is the cap rule, removable with no
numeric change, the bounded arm is already 1.00, layout options buy the prefix copy and preemption, not the ratio. B
could not find G0 and G3 to G7 spelled out in any tracked doc and maps onto G1, G2 and #552 criterion 4, saying so.
Owner decisions flagged: the `MEMRA_ADMIT_BY_MEMORY` door's decide-by is 2026-09-23 with B's receipt as its evidence;
the park policy. Comments on #539 and #552; both open.
Lead error, recorded: the integ30 round-1 commit (`913199b4b`) went out with server clippy `-D warnings` red on an
unused import because the check's result was not gated before the commit (the same heredoc-chain trap as twice
earlier today); the memra-server suite was green. Corrected in `340e8a474` with the note on the PR. Ruling 30: every
gate in a lead chain runs in its own `if ! ...; then exit; fi` line before the commit; a chain never carries a gate's
result across a heredoc.

Battery (`integration-day12/integ31-cpu-battery/`, run on the pre-merge tree that carried A's day 17 through C, then the
final-tree checks after merging main `a19631f9d`): fmt, portable suites, memra-server suite, clippy, censuses, collector
pytest, engine CPU lib tests, engine and server clippy `-D warnings`, marker census, workflow keys, perf board: rc=0;
shellcheck on the two gates clean once `SC1091` (the sourced port guard) is excluded; `git diff --check` tripped only
on battery summary whitespace. The final tree's engine equals main (only the two gate scripts differ). Local 5090
serve-smoke NOT RUN: the 5090 lock was held from 00:31Z by another session's `memra-server` (cwd `wt-525`, not a spill
lane) for the whole window; integ30 smoked the identical engine minutes earlier (`serve-smoke: 0 failed`); my waiter
was stopped (own process, identified by cwd), the other session's untouched. Lead error, recorded: the first attempt
matched the waiter by name and killed my own shell (exit 144), the trap the memory already names.
Revuto round 1 on #626, one finding, real: the fault gate's new self-consistency check (`1 of M` equals the
receipt's `items=N`) had no floor, so a regression registering one plane would read `1 of 1 items` and pass a cell
whose purpose is a partially accepted batch. Floor added: `items=N >= 2`. Every banked run reads far above it (27B 34
default and 32 plain, 9B 18 and 16); a re-run of the fault gate with the floor on both cards is assigned to C day 22.

## integ32 (`lane/spill-integ32-20260922`): A day 18, memra#536 Move 1 promote half under the host-contracts door
Lane tip merged: A `614c53f71` on main `3df055601` (#626). `worker.rs` conflicted at the two sites where the round-2
armed re-checks (#622) met A's day 18: at the demote site both settle-firsts (the `Demoting` entry, then the
`Promoting` entry's contract half) now precede the armed re-check; the promote hook keeps A's day-18 shape (the settles
moved behind the candidate check, so a request that submits nothing waits on nothing) and the armed re-check sits right
after its two settles; A's admission probe re-decides through `host_promote_probe_decision`, which gates on `armed()`,
after its own settles. INDEX.md both rows. Every new path is behind `MEMRA_KV_HOST_CONTRACTS=1` (default OFF, decide-by
2026-10-05); no new flag, no numeric change.

**Contract in brief (A's pre-registration).** The H2D promote issues on the day-17 copy stream behind the producer
fence's event; the engine keeps the submit-time owner-stream wait for an H2D (its consumer is the owner stream, so no
restore, prime or decode issued after the submit reads a fresh plane before its copy landed). The promote decision moved
from the admission body to the admission loop before `admit(..)`, sharing the body's predicates; a host hit submits and
the request parks on the requeue; one `Promoting` entry per worker; a tick-top poll requires the receipt, publishes
through the day-16 tail and holds the insertion pin one tick; the parked request re-admits to a device hit; the same
prompt parks again; another promote, any demote route and a tenant purge settle it first; failures keep the day-16
typed lines and set a one-tick memo so the request serves cold once; a pending promote missing its shell or ticket
fails closed (the #622 ruling mirrored). Six CPU tests, three censuses moved; `docs/FLAGS.md` door row updated.

**Gate lines, target card** (one RTX PRO 6000 Blackwell, run 2 on A's fixed tree, door OFF and ON, verbatim):
`KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` default and plain (12 ok each arm); `KV-HOST-SPILL FAILURE GATE: ALL
GREEN` (15 ok each arm); `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (64 ok); `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61
ok each arm); `PREFIX-NEWEST-TURN-FITS: ... V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` both arms; unit cells `8 passed`.
Local RTX 5090: not run, the canonical lock held by another session's server (`wt-525`) for the whole sitting; A's
resume driver waits detached in bounded retries and its receipts land uncommitted in A's worktree when the card frees
(stated in A's DAY18 and STATE). **Stall verdict, verbatim (run 2):** `STALL rule cell=stall-promote-on arm=promote
n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8 idle_max=14.9 arm_runs=10 arm_p50=13.4
arm_p95=14.8 arm_p99=92.6 arm_max=131.1 stall_median=81.9 stall_min=81.5 stall_max=117.7 server_demote_ms=[207.0,
208.1, 172.7, 172.0, 172.4, 172.1, 172.2, 172.1, 172.7, 172.0] server_promote_ms=[60.8, 61.9, 26.4, 26.1, 26.1, 26.1,
25.9, 26.1, 26.0, 25.9] intruder_prompt_tokens=[89, 86, 89, 86, 89, 86, 89, 86, 89, 86] tenant_text_identical=True
errors=0`; against day 16 (ON 162.8, OFF 85.0) the pre-registered rule reads `at_off` (3.1 ms under OFF); against day
17's 86.4 `promote_half_flat`. Run 1 (before A's hook fix) read `stall_median=157.8`: the hook settled a pending demote
on every admission, so the parked request's re-admission paid the demote copy synchronously (ten `settled
synchronously by a promote` lines); fixed to settle only before its own submission; run 2 has none. Demote arm
unchanged (149.7). Same-box cross-sitting reading, not a same-window A/B. Open for the lead (A): the settle-time owner
wait for an H2D (first owed item) needs the tier crate's conformance before the engine's `consumer_fenced` semantics
move. Lead reading: Move 1 is whole under the door on the target card; the door's 2026-10-05 review now has both halves'
stall receipts (demote 193.5 to 149.6, promote 162.8 to 81.9 ms, both toward or at the OFF arm). The 5090 door gates on
this tree are owed with the lock (C day 22 runs the door gates on main plus A's tip on the target card now).

Battery (`integration-day12/integ32-cpu-battery/`, merged tree `aba193fc8`, CPUQuota 1200 percent): fmt, portable
suites, memra-server suite (786 tests), clippy, censuses, collector pytest, engine CPU lib tests, engine and server
clippy `-D warnings`, marker census, workflow keys, perf board: rc=0; `git diff --check` tripped on A's cargo receipt
logs (blank line at EOF, marked `-whitespace`). Local 5090 `tools/serve-smoke.sh` (door OFF): `serve-smoke: 0 failed`,
after waiting behind the `wt-525` session's server and then lane B's day-27 cell for the lock.

Revuto round 1 on #627, two findings, both real, fixed by the lead in the integ: (1) in `host_promote_park_probe` the
cold memo written after `host_promote_prepare` refused was read at `entries[..][hi]`, but the stale-generation arm
`swap_remove`s the candidate, so the memo could name an unrelated, still promotable entry and make every request whose
candidate it is serve cold for the tick; the refused entry's tokens are captured before the call now. (2) a tenant
purge dropped the revoked tenant's `Promoting` entry but left its one-tick cold memo (prompt token ids) and the
worker's one-tick insertion pin on its just-published device entry, so the revoked tenant's device bytes would survive
the purge as "pinned entries left to in-flight sessions" with the worker as the only lease; `purge_tenant` clears the
memo when it names the tenant, and the worker releases the pin (`release_promoted_pin_for_tenant`) before the device
purge. CPU test `host_purge_clears_the_tenants_cold_memo_and_releases_its_promoted_pin` (another tenant's memo and
pin stay). Server clippy `-D warnings` and the memra-server suite (787 passed) green, gated before the commit.
Round 2, one finding, real, fixed: the idle block's 2 ms cap for a `Promoting` entry sits inside `active.is_empty()
&& queue.is_empty()`, but the off-tick promote keeps its parked request on the queue, so the cap never fired and the
loop spun the CUDA owner thread through park-and-requeue ticks for the whole copy (about 26 ms per promote in A's
receipt). The admission pass now counts the requests it parks on the promote; when nothing is active, every queued
request is parked and the `Promoting` entry is not ready, the loop waits on the command channel for the same bounded
2 ms before the tick (commands stay responsive; the tick top then polls the transfer). Source census test
`the_run_loop_waits_boundedly_when_the_queue_is_only_requests_parked_on_a_promote`. Server clippy `-D warnings` and the
memra-server suite (788 passed) green, gated before the commit. Owed to the door review: the promote stall cell on
this tree (A's day-18 receipt was taken with the spin; the wait changes timing, not bytes).

## integ33 (`lane/spill-integ33-20260922`): B days 27 and 28, C day 22
Lane tips merged: C `a04a9acd2` (its worker.rs resolution of A's day 18 met main's own; main's side taken on every code
conflict, no C-specific code was in the file), B `358441080`. Against main `0713c1a79` (#627) the non-research changes
are B's day-27 fix in `crates/memra-server/src/worker.rs` and two `docs/FLAGS.md` rows.

**B day 27, the prefix-cache budget derived at a literal 8192.** `init_prefix_cache_budget` sized `min(2 x entry(ctx),
boot_free - 1.5 GiB)` with `ctx = MEMRA_CTX` when set, else the literal `PREFIX_CACHE_CTX_FALLBACK = 8192`, while the
cap rule served the checkpoint's context when unset. Target card, 27B, unset: entry `8192 x 29,696 + 156,893,184 =
400,162,816 B`, budget `800,325,632 B` beside `262,144 x 31,552 = 8,271,167,488 B` sessions; day 26's deferred shape
inserted 45 spec-boundary entries and LRU-evicted 43, `cached=0` on 15 of 15. Fix: `prefix_budget_ctx(env_ctx,
model_ctx)` is the cap rule's own `resolve_ctx`, applied per loaded model; `PrefixCacheBudget::Derived` names its
`ctx_source`; the boot line prints it; an unresolvable context contributes no entry and prints a WARNING. The two-entry
count and the boot-free clamp (the product decision) are unchanged; only the context term moved. No new flag. CPU test
`derived_prefix_budget_context_term_is_the_served_context_set_or_unset`. After, target card, verbatim: `[prefix-cache]
on: budget 15883MB (15883042816 B, derived: 2 x 7941521408 B max entry for model "q38" at served ctx 262144 (from
checkpoint; MEMRA_CTX unset), requested 15883042816 B; boot driver free 86519709696 B, post-reserve clamp 84909096960
B)`; warm arm `arm=iii L0 N=5 ... cached=[1440, 1440, 1440, 1440, 1440]`, `L1 ... cached=[3104 x5]`, `L2 ... cached=[5760
x5]` where day 26 read `[0 x5]` on all three; `prefix_cache_entries=45 prefix_cache_evictions=0`; outputs equal to day 26
on 45 of 45 (`P_G_chars_equal=45`), digests equal 45/45 across today's boots; idle retention 30.1 to 39.1 GB (the
entries now stay). Local card (`MEMRA_CTX=65536`): a no-op by construction, `cached=[0 x5]` as day 26, digests equal
45/45. **Park cell** (default versus `MEMRA_KV_PARK_COMPACT=1`, both orders, N=5 per arm per length): census correction
first (a spec session parks in the spec pool only: `continuation_pool_entries=0`, `spec_pool_entries=2`); the door is
plain-pool only and cannot reach what this mix retains: target card all four runs `retained_by_process=39090913280 ...
continuation_pool_hits=0 spec_pool_hits=0 ... park-compact lines: 0`, digests equal 45/45 across arms and orders; local
card the same shape at `7449083904`; the labelled plain-path pair (`MEMRA_SERVE_SPEC=0`, target) engages the door
(`46 [kv-reuse] park-compact` lines, `2071 of 262144 rows retained` at L0, retained `39191576576 -> 22649241600` B,
pools still 0 hits, digests equal 45/45). No default changed.

**B day 28, memra#476 graph growth and the park door's decide-by.** On these dense models the only shape-keyed,
never-freed allocation on the served path is the FA partial pool (`Engine::fa_part_pool`, retire-on-grow); neither book
carries it; the boot-calibrated floor covers it; the verify-graph pool never engages on dense models and is charged on
the physical side only where it does (a named predictive gap for MoE and linear families). Shape-walk cell, one
binary, both cards, verbatim: target `G1: growth(end) = 38638080 (G_fa after ready 38638080 + delta_D 0) <= 10% floor =
230057574: PASS`, `G2: while inflight > 0, max_underbook_real = 7443968680 <= floor = 2300575744: FAIL`, `G3: exact rows
max |residual| = 0 (tolerance 0); line~ rows max |residual| = 804952 (tolerance 2e6 at the 1 MB grain)`, `G4:
Overloaded/OOM lines = 0: PASS`; local `G1: growth(end) = 25758720 ... <= 10% floor = 161061273: PASS`, `G2 ...
3144845816 <= floor = 1610612736: FAIL`, `G3 ... 0 ... 482808`, `G4 ... PASS`. Per the pre-registration G1's pass means
no per-request booking landed; G2 is the day-24 reading (async pool cached blocks plus the retention the prefix budget
intends), the graph term's share 0.5 and 0.8 percent, recorded not relaxed. Proposed, not landed: a boot pre-grow
through `fa_dcw_pool_ensure` for the served context and batch cap before the probe's watermark reset (about 0.3
agent-day; moves the boot footprint; the owner's call). `MEMRA_KV_PARK_COMPACT` row now reads `decide-by: 2026-10-06`
with its deciding cell (plain path, both cards, both orders, N>=5: compacted-park resume byte identity on both resume
shapes, the step-OOM adjacency replay, the copy-cost pair); promotion moves only plain-pool retained bytes, deletion
loses nothing measured (0 hits on every tape), the spec pool is a separate owner decision.

**C day 22.** With Move 1 whole and the fault gate's floor: target card `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`
default (`items=34`) and plain (`items=32`), 65 ok each, floor line `ok: promote-reject: the entry carries at least two
planes, so the reject is partial (items=N >= 2)`; `KV-HOST-SPILL FAILURE GATE: ALL GREEN` all four arms; `KV-HOST-SPILL
IDENTITY GATE: ALL GREEN (teeth=0)` all four arms (ON logs carry A's `promote submitted off the tick ... request parked`
and `promote published off the tick: ticket complete after 1 poll(s), 5.5ms`); local 5090 fault gate `ALL GREEN`
default (`items=18`) and plain (`items=16`). The `TENANT_PCT=100` arm has its own receipt now (`pool-full refusal arm:
whole host budget ...`, `KV-HOST-SPILL FAILURE GATE: ALL GREEN`, 14 ok each door arm; server line `[prefix-host] skip
demote: entry 159.9MB > host budget 1MB`). Owner note recorded by C, not acted on: under the door at `TENANT_PCT=100` the
pool-full demote runs the whole Move 1 contract (a 160 MB copy) and only then refuses at insert; deliberate per ruling
15, reachable only with the share cap disarmed. Review table rows updated; the DFlash tail slice question survives day
20 (the standalone tap sink is bounded; the host tier's draft-tail image identity input is unbound).

Battery (`integration-day12/integ33-cpu-battery/`, merged tree `be554e587`, CPUQuota 1200 percent): fmt, portable suites,
memra-server suite, clippy, censuses, collector pytest, engine CPU lib tests, server clippy `-D warnings`, marker census,
workflow keys, perf board, diff-check: 13 steps rc=0. Local 5090 `tools/serve-smoke.sh`: `serve-smoke: 0 failed`.

## integ34 (`lane/spill-integ34-20260922`): A day 19 (tier conformance rule 3, the H2D reader wait at the settle; Move 2 pre-registration) and C day 23 (promote stall on the fixed tree; the door review table)
Lane tips merged: A `494c5adc6`, C `eafcd9462`, clean on main `da1f59bf6` (#631). Engine and tier changes, all behind
`MEMRA_KV_HOST_CONTRACTS=1`: `crates/memra-tier/src/conformance/reader_fence.rs` (rule 3: `consumer_fenced` is the
installed wait on the reader's stream, never a flag and never the copy's landing; schedules `h2d_reader_fence(fixture,
ReaderWaitInstall::{AtSubmit, AtSettle})` and `h2d_reader_issued_before_its_wait_is_unordered`) with
`tests/contracts/reader_fence_bindings.rs` (`day19_h2d_reader_fence_at_submit_is_the_engines_day18_program` ok,
`day19_red_arm_settle_time_flag_without_a_reader_wait_fails_the_schedule` ok under `catch_unwind`,
`day19_h2d_reader_fence_at_settle_with_a_wait_on_the_reader_stream` ok; contracts target 71 passed);
`crates/memra-engine/src/tier_transfer.rs` (`submit_batch` no longer fences an off-owner H2D at submit;
`install_consumer_wait(ticket)` installs the owner-stream wait on each H2D item's event and records the consumer fence,
fail-closed on a quarantined or eventless item, `unknown` on a stream error); `crates/memra-server/src/worker.rs`
(`host_kv_planes_settle_promote` calls it under `Poll` and `Block` before the receipt check and `ready_view`, aborting
through `host_promote_contract_abort` on refusal; the engine and worker censuses pin the order); `docs/FLAGS.md` door
row. No new flag, no numeric change.

**A day 19.** Day-18 5090 receipts settled first (A's detached driver): identity default ON, plain OFF, plain ON `KV-
HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`; failure OFF and ON `ALL GREEN`; fault `ALL GREEN` (64 ok); hit OFF and
ON `ALL GREEN (qwen)`; unit cells `8 passed`; identity default OFF `REFUSED: canonical GPU lock busy` (not run, 30
lock-busy retries); the 27B twin gate on the 5090 read, in both door arms identically, `... evictions=1
cohort_evictions=1 ... effective_free_ok=2/8 ... V3=FAIL ... -> FAIL` with a constant `-410352980` B effective-free
error on turns 2 to 7 while day 17 local and today's target card read `evictions=9 -> PASS`: a local-card reading, not a
door delta, cause not established, assigned to B day 29 (running). Engine wait moved as above; target card on the moved
tree (binary `bf353fc7...`): identity `ALL GREEN (teeth=0)` default and plain, OFF and ON; failure `ALL GREEN` both
arms; fault `ALL GREEN` (65 ok); twin `-> PASS` both arms; hit `ALL GREEN (qwen)` both arms; unit cells `8 passed`; ON
arms carry `promote published off the tick ... 7.6ms` and `H2D receipt ... require=ok`, zero `reader wait refused`.
Move 2 pre-registered in `OWNER-THREAD-OFFLOAD.md`; first slice named (the capture on the copy stream with an
event-ordered publication, `Capturing` state, `TransferOp::D2d(ContiguousCopy)`, the recurrent-state `clone_dtod` kept
at the boundary on the owner stream, schedule `d2d_capture_publish` first; 1.5 agent-days) and started as A day 20
(running). Owed, stated by A: the 5090 door gates on this tree, the 27B twin repro, the same-window A/B of both Move 1
classes, the receipt hashes off the tick, the by-reference routes.

**C day 23, the promote stall on the tree with the parked-only wait** (one RTX PRO 6000 Blackwell at 600 W, one
collector hold, OFF boot then ON boot, N=5 per arm per order, both orders, replay PASS x4), verbatim: `STALL rule
cell=stall-promote-on arm=promote n_per_order=5 pooled=10 idle_runs=10 idle_p50=13.4 idle_p95=14.7 idle_p99=14.8
idle_max=14.9 arm_runs=10 arm_p50=13.4 arm_p95=14.8 arm_p99=92.5 arm_max=131.0 stall_median=81.9 stall_min=81.4
stall_max=117.6 server_demote_ms=[207.7, 207.8, 172.6, 171.8, 172.3, 172.0, 172.4, 171.8, 172.5, 171.9]
server_promote_ms=[61.3, 61.6, 26.4, 25.8, 25.9, 26.0, 25.9, 26.1, 25.9, 26.0] intruder_prompt_tokens=[89, 86, 89, 86,
89, 86, 89, 86, 89, 86] tenant_text_identical=True errors=0`; `stall-promote-off ... arm_max=134.2 stall_median=85.2
stall_min=84.4 stall_max=120.8 server_demote_ms=[40.6, 42.1, 6.8, 6.3, 6.1, 6.1, 6.0, 6.1, 6.1, 6.0]
server_promote_ms=[45.0, 46.5, 11.2, 10.6, 10.5, 10.4, 10.4, 10.5, 10.5, 10.3] ...`; `stall-demote-on ... arm_max=163.1
stall_median=149.5 ... server_demote_ms=[127.6, 132.6, 132.5, 132.2, 132.4, 132.0, 132.5, 132.2, 131.9]`; demote OFF
`stall_median=117.5`. Pre-registered reading as printed: `admissible=True P1=at_or_under_day18 P2=within_wait
P3=on_at_off P4=off_stable P5=demote_half_unchanged P6=off_stable`: the parked-only wait moved neither the tenant's
stall nor the promote's window (publish `1 poll(s), 19.5` to `19.7ms` x10, zero `settled synchronously by a promote`),
and P3 is the lane's one same-window OFF/ON pair of the promote arm. Door gates on the same tree: target card twelve
cells all `ALL GREEN` (fault 65 ok x2 with the floor, failure 15 ok x4 and whole-budget 14 ok x2, identity 12 ok x4);
local RTX 5090 (9B) ten cells `ALL GREEN`. `HOSTPREFIX-DOOR.md` review table final for 2026-10-05: 15 owed cells all
banked with tree, receipt path and verbatim verdict; the cost table (demote stall 117.5 OFF against 193.5 on-tick and
149.5 after Move 1; promote 85.2 OFF against 162.8 on-tick and 81.9 after Move 1; cached pair demote 37.8 against
113.6; hash pass 77.9 ms cached against 1698 WC on the target host; lifecycle share below resolution; arena pair
`arena_first_touch_absent; arena_not_slower`); the correctness table green both arms on both cards; the missing list
(arena handoff per ruling 28, DFlash tail slice, verify digest v3, the 5090-class pair, two census questions, the 5090
whole-budget arm); the decision question stated, not answered. Lead reading: the door's review input is complete on
the target card; the owner decides on 2026-10-05.

Battery (`integration-day12/integ34-cpu-battery/`, merged tree `720820350`, CPUQuota 1200 percent): fmt, portable
suites, memra-server suite, clippy, censuses, collector pytest, engine CPU lib tests, tier tests, engine, server and tier
clippy `-D warnings`, marker census, workflow keys, perf board: rc=0; `git diff --check` tripped on A's cargo receipt logs
(blank line at EOF, marked `-whitespace`). Local 5090 `tools/serve-smoke.sh` (door OFF): `serve-smoke: 0 failed`.

## integ35 (`lane/spill-integ35-20260922`): B day 29, the 5090 twin gate's `V3=FAIL` classified; the door gates on the 5090
Lane tip merged: B `184edb5bd` on main `226abab0e` (#632), clean. The only non-research change is
`tools/prefix-newest-turn-fits-gate.py`.

**The `V3=FAIL`, reproduced three times, gate unchanged, door OFF, verbatim (identical):** `PREFIX-NEWEST-TURN-FITS:
budget_bytes=1073741824 cohort_bytes=736755712 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=1
cohort_evictions=1 self_evictions=0 refused_or_skipped=0 effective_free_ok=2/8 identity_ok=8/8 grid_ok=21/21 grid=32
off_grid_calls=0 V1=ok V2=ok V3=FAIL V4=ok V5=ok V6=ok -> FAIL`, A's day-18 table to the byte, with a 1390 MiB
co-tenant (another session's python, never touched) on the card on every 1 Hz sample of all three runs. Named cause:
`410352980 = 3272 x 29696 + 313187668`, one parked plain session of the cohort's last shape (`ctx_cap 3272`, the
admission line's `313MB fixed`); the cache-on boot's turn-2 `[admit-oom] reclaim-on-defer: evicted 2 prefix entries + 1
plain ...; effective free 3744MB -> 4651MB` released it while the cache-off boot kept its counterpart until its own
turn-3 reclaim took a different one: `required` is about 4830 MB, day 17 had 5207 MB effective free at that admission,
day 18 3744 MB (the co-tenant's 1.46 GB less, and the cache-on boot carries 0.93 GB of cache), so the two boots crossed
the floor on different turns. V3's premise (equal retained parked-session state in both boots) failed, not the
accounting (every settle line balances); not a door delta; the day-27 budget change is not involved (the gate sets
`MEMRA_PREFIX_CACHE_MB=1024` and `MEMRA_CTX=16384` itself). The clean-card run arrived when the co-tenant left
mid-battery: `evictions=9 ... -> PASS`, day 17's values to the byte.

**What changed, gate only:** V3's clause, form and 64 MiB slack unchanged; when the two boots' `reclaim-on-defer`
parked-session releases differ on any window the gate refuses with `REFUSED: V3 premise: ...` (exit 2) naming the
windows, the releases per boot, the budget and the card at each boot (driver free and compute-apps now sampled into the
receipts), keeping the would-be line in `summary.json` as `verdict_under_broken_premise`. CPU test `test-day29.py` 13 ok
(rerun in this battery). Red arm on the card under one collector hold with a self-owned 1390 MiB `cudaMalloc` child:
`REFUSED: V3 premise: ... 6 window(s) (turn 2: calibration released plain/spec/dspark 0/0/0, measured 1/0/0; ...)`, exit
2. Ruling 31: a gate whose verdict depends on a premise about the card's state names the premise, samples the card into
its receipt, and refuses typed when the premise fails; it never prints a FAIL that a co-tenant produced.

**The 5090 door gates on `1dce64df0`, both arms, verbatim:** `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` default
OFF (the arm A never got), default ON, plain OFF, plain ON (12 ok each); `KV-HOST-SPILL FAILURE GATE: ALL GREEN` OFF
and ON (15 ok); `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (65 ok); `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` OFF and ON
(61 ok); unit cells `8 passed`; 9B twin `REFUSED: cohort promotion did not happen for 2800 tokens ...` both arms (A's
day-17 shape fact); 27B twin `... evictions=9 cohort_evictions=3 ... effective_free_ok=8/8 ... V3=ok ... -> PASS` both
arms on a clean card. Target card, 27B twin `-> PASS` both arms on the patched gate, premise rows equal. Lead reading:
the door's correctness table is now green on both cards for every arm, including the one arm the 5090 lacked.

Battery (`integration-day12/integ35-cpu-battery/`, docs and tool): the gate's `py_compile` and `--help`, B's
`test-day29.py` (`OK`), flags census, marker census, public-boundary `check` (0 new), perf board, workflow keys, em-dash
scan: rc=0; `git diff --check` tripped on a raw cargo receipt log's blank line at EOF (marked `-whitespace`). No engine
change, no smoke.
Revuto round 1 on #633, two findings, both real, fixed by the lead: (1) the refusal was unconditional, so a broken
premise would have masked a real V1, V2, V4, V5 or V6 failure as exit 2; the exit rule is now `gate_outcome`: a failed
non-V3 clause is the verdict FAIL (exit 1) with a premise note beside it, and a broken or unreadable premise refuses
(exit 2) only when every other clause holds. (2) a reclaim line the detailed shape did not parse read as "released
nothing" and let the premise hold falsely (engine wording drift would have reprinted the unclassified FAIL); such a
line now makes the premise unreadable and the gate refuses typed (`REFUSED: V3 premise unreadable: ...`). Two CPU tests
added to `test-day29.py` (15 ok); `docs/TESTING.md` carries the exit rule.

## Lanes
- D day 11 sealed and pushed (`15bd53152`); merged into integ9.
- B day 13 sealed and pushed (`1fef60006`); merged into integ10.
- A day 11 sealed and pushed (`432816926`); merged into integ10. D day 12 sealed and pushed (`b3324c262`); merged into integ10. C day 12 sealed and pushed (`7efedd13d`); C day 13 merged (#591); C day 14 merged (#594); C day 15 merged (#599); C day 16 sealed on the lane (`36fa5262b`, push refused by the perf-ci gate, integrated from the worktree); integ19. A day 13 running (pinned-kind A/B). B day 19 running (spec capture alignment, #379 gate must be green). B day 14 sealed and pushed (`25d252c6f`); merged into integ14. A day 12 running (#384, #385 harness). E, F idle.
