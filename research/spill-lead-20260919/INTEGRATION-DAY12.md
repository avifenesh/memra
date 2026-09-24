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

## integ36 (`lane/spill-integ36-20260922`): A day 20 (Move 2 slice 1, the capture off the tick) and B day 30 (memra#423 lives in darklanes)
Lane tips merged: A `62aa92279`, B `e0a8afeeb`, clean on main `4bb2afb63` (#633). Engine, tier and server changes, all
behind `MEMRA_KV_HOST_CONTRACTS=1` (default OFF, decide-by 2026-10-05), plus B's one-paragraph header fix in
`memra-server/src/metering.rs`. No new flag, no numeric change.

**A day 20, Move 2 slice 1.** Design finding, recorded before any cell ran: the pre-registered op shape
`TransferOp::D2d(ContiguousCopy)` takes owned `DeviceLease`s on both sides and the engine's registry admits only moved
buffers, but a capture's source is the live session cache's plane that the decoding session keeps; so the capture is a
borrowed-source op into an owned registered destination (`D2dCapture`, `CudaTransfers::submit_d2d_capture`), not a
`TransferOp` variant; slice 2's restore has the mirror problem (its destination is the session's cache). What landed:
`crates/memra-tier/src/conformance/d2d_capture.rs` (`d2d_capture_publish`: not landed until every completion event,
publish before the event is a schedule failure, `retire(None)`, `acknowledge`, destinations back; slice-1 receipt clause:
no witnessed checksum so `Completion::require` refuses `Corrupt`) with bindings
`day20_d2d_capture_publishes_only_after_every_items_event` ok, `day20_red_arm_publish_before_the_event_fails_the_schedule`
ok, `day20_d2d_item_without_a_witnessed_checksum_is_refused_by_the_host_contract_gate` ok (contracts target 74 passed;
frozen schedules untouched); engine `CopyDirection::DeviceToDevice`, `D2dCapture`, `submit_d2d_capture` (requires the
copy stream; the copy stream waits on the producer event, `memcpy_dtod`, completion event on the copy stream, fenced at
submit like a D2H, no owner wait), `capture_landed`, census `d2d_capture_rules_are_as_stated`, GPU cell
`d2d_capture_lands_on_the_copy_stream_and_publishes_only_after_its_event`; worker `HostPrefixCache::capturing`,
`PendingCapture`, `prefix_capture_off_tick` (called by `prefix_insert_from_session` after `prepare_snapshot`, before the
tick program; recurrent state `clone_dtod` on the owner stream at the boundary; fresh planes registered with twins;
producer fence; one batch), `host_kv_planes_settle_capture`, `host_capture_settle_*`, `host_capture_publish`,
`host_capture_latch`, `host_capture_drain_at_shutdown`; call sites at the tick top after the promote poll, both idle
waits, `purge_tenant`, admission reclaim, three device trims, the run-loop exit; the ledger's in-flight term is
`2 x (2 x max layers + 2)`. Five CPU tests (the poll keeps the state and done reaches ready once; a missing shell or
ticket fails closed; a latched settle drops the entry and takes the tick program; a purge drops the purged tenant's
entry and keeps another's; every path settles or ignores as stated). Server lib 798 passed; clippy `-D warnings` on
all three crates. Target card (tree `c7a6d3a5b`, binary `263fe777...`), door OFF and ON, verbatim: `KV-HOST-SPILL
IDENTITY GATE: ALL GREEN (teeth=0)` default and plain both arms; `KV-HOST-SPILL FAILURE GATE: ALL GREEN` both arms;
`KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`; `PREFIX-NEWEST-TURN-FITS: ... V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` both
arms; `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` both arms; unit cells `8 passed` plus the engine capture cell `1 passed`;
the route engaged in the plain ON arm (`capture submitted off the tick (seed): 64 tokens, 32 planes (158.8MB) ...`,
`capture published off the tick (seed): ... after 1 poll(s)`), zero refusal or latch lines. Scope fact: spec-boundary
publishes (`prefix_insert_from_spec_boundary`, the MTP draft plane) still take the tick program. Capture stall cell
(one hold, OFF/ON/ON/OFF, N=5 per arm per order, every re-post hit at `cached_tokens=5088`), verbatim: OFF pass 1
`stall_median=355.4`, ON pass 1 `355.4`, OFF pass 2 `353.9`, ON pass 2 `354.0`; C1 `flat` both passes (bound 6.0); ON
`server_capture_ms` median 227.9 (submission to the observing poll). The intruder's 5120-token prime dominates both
arms; the capture's own share is under the cell's resolution; decides nothing about the door; a capture-isolating
cell is C day 24 (running). Not run: the local 5090 door gates on that tree (C day 24). Still owed: the spec-boundary
capture route, Move 2 slices 2 (A day 21, running) and 3, Move 1's receipt hashes, by-reference routes, same-window A/B.

**B day 30, memra#423 and #464.** Key finding: the budget journal, `balance_after_micro`, the boot backfill and the
`/admin` API are not in memra (extracted 2026-08-29, `docs/FLAGS.md` rows 304 to 306); they live in darklanes
`serving/darklanes-metering/src/ledger.rs` behind memra's `Metering` and `Receipt` seam; the memra half is the seam
contract, which promises no cross-request ordering, correctly. Rows are causally ordered (one `Mutex<BudgetState>`, the
journal file inside it); the chain broke because `admit` subtracts each request's worst-case hold with no row and settle
stamps `current + refund`, so with A and B in flight row A stamps `B0 - a - R_B` and row B `B0 - a - b`: the breaks
cancel at quiescence and the live gap equals the outstanding holds; the issue's "stamps a balance it read before the
others landed" describes a race the code does not have. Cells (local 5090, darklanes-serve on the 9B, synthetic tenant,
N=16 plus one credit in flight), verbatim: before `JOURNAL-ORDER CELL arm=before N=16 rows=17 chain_violations=17
order_violations=0 conservation=ok cancel_sum=0 -> FAIL`; after `JOURNAL-ORDER CELL arm=after N=16 rows=17
chain_violations=0 order_violations=0 conservation=ok cancel_sum=0 -> PASS` (twice). Fix, in darklanes, bounded, no
schema change: per-tenant `tenant_reserved` at admit, settle and unsettled drop; every row stamps
`balance_after + tenant_reserved_after` (the settled balance); four CPU tests from the diagram (three red before), suite
92 of 92, clippy clean; on branch `lane/budget-journal-order-20260922` at `20f8140e8`, pushed, not merged (the lead
opens its PR). Stated for the owner, not taken: source-file top-ups still rowless (a second chain break); audits cut at
the first fixed boot; pre-existing fmt drift in darklanes `admin.rs` and `capture.rs` untouched. memra#464 gap: the
backfill's only evidence is journal membership and a carried row is byte-identical to an unbilled one; the #423 fix
repairs the seed arithmetic and adds no id to the guard; two guard-only seeds offered (about 0.5 agent-day) for the
owner. In memra: the stale `metering.rs` header now says the stock binary wires no implementation (the reference ledger
moved out on 2026-08-29). Ruling 32: a lane brief that names a memra seam checks first where the implementation lives
today (the 2026-08-29 extraction moved billing to darklanes); fixes to darklanes go through their own PR under the
owner's merge law, never through a memra integ.
The darklanes PR for B's journal fix is avifenesh/darklanes#1093 (lead-opened, self-review comment posted; merges under
the owner's merge law there).

Battery (`integration-day12/integ36-cpu-battery/`, merged tree `bdd3b8993`, CPUQuota 1200 percent): fmt, portable
suites, memra-server suite, clippy, censuses, collector pytest, engine CPU lib tests, tier tests, engine, server and tier
clippy `-D warnings`, marker census, workflow keys, perf board: rc=0; `git diff --check` tripped on receipt logs (marked
`-whitespace`). Local 5090 `tools/serve-smoke.sh` (door OFF): `serve-smoke: 0 failed`.

Revuto round 1 on #634, two findings, both real, fixed by the lead: (1) the capture's source is a borrowed slice of
the live session's KV plane and the engine retains none of it after `submit_d2d_capture`; a retiring session's cache is
dropped on the owner stream (not ordered against the copy stream) or parked for a later request to rewrite, and the tick
top's settle is a `Poll`, so a capture in flight across the retire seam would publish whatever bytes the copy stream
happened to see, with no checksum term in slice 1 to catch it. Fix: before any session leaves `active`, a pending
`Capturing` entry settles with `ContractWait::Block` ("a session retire"), so no plane is freed or rewritten under an
in-flight read (one capture per worker). (2) when `submit_d2d_capture` refused, nothing was submitted, the producer
event could still be pending, `release_producer` refused `Busy` and the `let _ =` dropped it, leaving the fence in the
engine's producer table for the boot; the refusal path now drains the owner stream, releases the fence, and latches the
route off (typed) if the fence still will not release. Source census test
`a_session_retire_settles_a_pending_capture_before_the_cache_moves_and_a_refused_submission_releases_its_fence`; the
admission-book lock test kept the retire seam intact (the settle sits before the loop). Server clippy `-D warnings` and
the memra-server suite (800 passed) green, gated before the commit. Owed: the retire-seam settle's cost in the
capture-isolating cell (C day 24 measures on the tree it has; the door review reads both).

**C day 24 (tip `01725be2a`, merged into integ36).** The 5090 door gates on the slice-1 tree (9B; 27B for the twin),
verbatim: fault default `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (65 ok); fault plain `KV-HOST-CONTRACT-FAULT GATE: 2
FAILURE(S)` (63 ok), reproduced in a rerun, the two lines `FAIL: presubmit: the next demote completes with a D2H
contract receipt after the refusal` and the postpublish twin; failure x4 `ALL GREEN`; identity x4 `ALL GREEN (teeth=0)`;
hit OFF and ON `ALL GREEN (qwen)`; twin27 OFF and ON `... V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` (no premise
refusal); unit cells `8 passed` and the engine capture cell `1 passed`. C's finding, verified by the lead on the banked
logs: the gate's clause matched the demote receipt by the literal `seq=1` (presubmit) or `seq=2` (postpublish); on the
slice-1 tree the plain arm's seeds submit capture tickets on the same issuer and each consumes a sequence, so the demote
after the refusal completes as `seq=3` or `seq=4` with `require=ok` and the literal missed it; the default (spec) arm
passed because its seed publishes through `prefix_insert_from_spec_boundary` (tick program, no ticket); A's day-20
target-card fault gate ran the default arm only. Gate fix by the lead in this integ (`tools/kv-host-contract-fault-gate.sh`):
the receipt clause matches any `seq`, and a new clause `receipt_seq_accounts` keeps the accounting the literal carried
(a refused presubmit consumed no sequence, a refused postpublish consumed one) as `expected + the capture tickets
submitted before the demote's own submission line` (the sequence is consumed at submission; a capture submitted between
the demote's submission and its receipt does not precede it; trees without the submission line count up to the receipt),
and asserts the receipt's seq equals the submitted demote's. Replayed on the banked logs: slice-1 plain presubmit `seq=3
expected 1 + 2 = 3`, postpublish `seq=4 expected 2 + 2 = 4` (and the rerun), slice-1 default `1 + 0`, `2 + 0`, the day-22
pre-slice tree `1 + 0`: all rc=0. A live rerun of the fault gate plain arm on the fixed gate is owed (C day 25).
Capture-isolating cell: stopped at the pre-registration with the arithmetic: no publish class captures generated rows
(publishes are `seed`, `lcp-split`, `spec-boundary`, `dspark-boundary`, `glm5-boundary`); the nearest shape (a warm hit
deepened by exactly 64 on-grid tokens) moves 154 MB of KV planes at the shape and 243 MB at ctx 8192, 1.5 to 2.4 ms even at
100 GB/s, under C1's 6 ms threshold and the harness's resolution while both arms share tens of ms; the claim would read
`flat` by construction; no run. Slice-3 census in `HOSTPREFIX-DOOR.md` item 10: still on the tick are every
`prefix_insert_from_spec_boundary` publish (MTP drain sweep, dspark, glm5), the fanout leader and pause sweep snapshots,
and slice 1's own `OnTick` refusals; bytes on the 27B at ctx 8192: trunk 243 MB, MTP draft plane about 15 MB, dspark
tail about 85 MB fixed; an off-tick route needs a second borrowed span (the draft scratch owned by `SpecSession`), a
producer event at the drain sweep, and settle-before-drop across a retiring or parking spec session.

## integ37 (`lane/spill-integ37-20260922`): A day 21, Move 2 slice 2, the hit restore off the tick
Lane tip merged: A `ff3f0f6f5` on main `5df11152f` (#634), clean. All behind `MEMRA_KV_HOST_CONTRACTS=1` (default OFF,
decide-by 2026-10-05); no new flag, no numeric change.

**Design finding, pre-registered before code.** The restore's destination is the request's fresh session cache and its
source is a published entry's `PrefixPlane`; neither is registered with the engine, and registering the source would
take planes out of an entry that must stay servable. So the op is `D2dRestore` / `CudaTransfers::submit_d2d_restore`,
borrowed on both sides (source slice, destination view), the device LRU's pin as the producer-side guarantee (a pinned
entry is out of every eviction index by construction), rule 3's owner-stream wait installed at the settle as the reader
fence (items unfenced at submit; `install_consumer_wait` now fences every unfenced non-D2H item). `Restoring` is a state
of the parked request (one per worker), not of the entry. The recurrent f32 state keeps the owner stream in this slice
(about 157 MB of every 27B entry; stated so the stall reading is bounded correctly).

**What landed.** `crates/memra-tier/src/conformance/d2d_restore.rs` (`d2d_restore_ready`,
`d2d_restore_primed_before_its_wait_is_unordered`) with bindings
`day21_d2d_restore_is_ready_only_after_the_landing_and_the_installed_reader_wait` ok,
`day21_red_arm_prime_before_the_reader_wait_fails_the_schedule` ok (fails the schedule under `catch_unwind`),
`day21_d2d_restore_item_without_a_witnessed_checksum_is_refused_by_the_host_contract_gate` ok (contracts target 77
passed); engine `D2dRestore`, `submit_d2d_restore`, `restore_landed`, census `d2d_restore_rules_are_as_stated`, GPU cell
`d2d_restore_lands_on_the_copy_stream_and_is_ready_only_after_the_installed_wait`; worker `HostPrefixCache::restoring`,
`host_restore_park_probe` (in the admission loop after the promote probe: the OFF validation split out as
`prefix_restore_validate`, fresh cache, pin, recurrent state and `len` on the owner stream, one batch, park on the
requeue), the tick-top poll installing the wait plus a three-tick orphan expiry, `host_restore_take_ready` at admit's hit
site (pin carried over), settle-first at the reclaim, the three trims, the tenant purge (worker level, before either
index purges) and shutdown, fail-closed arms, `Latched` forgets the cache and keeps the pin, a refused submission drains
the owner stream and releases the fence or latches typed (the #634 shape mirrored), the ledger's third in-flight term;
CPU tests `restore_probe_decision_parks_its_own_request_and_names_an_orphan`,
`a_pending_restore_missing_its_cache_or_ticket_fails_closed`, census
`every_path_that_meets_a_restoring_request_settles_or_ignores_it_as_stated`; server lib 803 passed; clippy `-D
warnings` on tier, engine, server. Stated limit: the pending-to-ready path needs a device cache, so CPU tests do not
reach it; the card does. `docs/FLAGS.md` door row day-21 sentence.

**Gates, target card** (tree `1350f118b`, binary `ddde2083...`), verbatim per arm: identity default OFF, default ON,
plain OFF, plain ON `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`; failure OFF and ON `KV-HOST-SPILL FAILURE GATE:
ALL GREEN`; fault `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`; twin OFF and ON `PREFIX-NEWEST-TURN-FITS: ... evictions=9
cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 identity_ok=8/8 grid_ok=21/21 grid=32
off_grid_calls=0 V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS`; hit OFF and ON `SPEC-ON-CACHE-HIT GATE: ALL GREEN
(qwen)`; unit cells `8 passed` and engine `2 passed`. Engagement: the plain identity ON arm carries two `restore
submitted off the tick: 64 tokens, 32 planes (158.8MB), ticket seq=6 ...; request parked` and `restore landed off the
tick: ... complete after 1 poll(s), 75.4ms` pairs under `teeth=0`, zero refused, dropped or disabled restore lines. The
hit gate's entries are all `insert (spec-boundary)` (draft-bearing), which the route refuses by name, so its 14 hits kept
the tick program in both arms: the one-program proof of this slice today is the plain identity arm; the draft-bearing
restore is owed. **Restore stall**, verbatim rule lines: `STALL rule cell=stall-restore-off arm=restore n_per_order=5
pooled=10 ... arm_p99=15.5 arm_max=104.8 stall_median=88.1 stall_min=87.8 stall_max=91.3 ... intruder_cached_tokens=[5088
x10] intruder_prompt_tokens=[5121 x10] tenant_text_identical=True errors=0`, `... cell=stall-restore-on ... arm_p99=22.1
arm_max=93.7 stall_median=80.1 stall_min=80.1 stall_max=80.2 ... server_restore_ms=[14.5 x9, 14.4]` (pass 1); pass 2 OFF
87.9, ON 80.2; C1 `on_under_off` both passes (bound 6.0); the pre-registered expectation was `flat`; A's reading: the
intruder's admission work splits across two ticks (ON p99 22.1 against OFF 15.5, about 8 ms moved), not a removal; the
rows' D2D share is under resolution; `server_restore_ms` 14.5 is the tick period; decides nothing about the door. Owed
by Move 2: slice 3 (the receipt term with the `d2d-delay` fault; A day 22, running), the f32 state off the tick, the
spec-boundary capture route and the draft-bearing restore, an isolating stall cell; the 5090 door gates on this tree.

**C day 25 (tip `8c96ca6b5`, merged into integ37).** The fixed fault gate live, door ON, verbatim on both cards, both
arms, 67 ok each, 0 FAIL: local RTX 5090 (9B) fault-default `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` with `receipt seq=1
expected 1 + 0 capture ticket(s) submitted before it = 1` and `receipt seq=2 expected 2 + 0 ... = 2`; fault-plain `ALL
GREEN` with `receipt seq=3 expected 1 + 2 capture ticket(s) submitted before it = 3` and `receipt seq=4 expected 2 + 2 ...
= 4`; target card (27B) the identical four accounting lines and `ALL GREEN` both arms; identity on the target card over
the retire-seam settle: all four arms `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`. The retire-seam settle's cost,
pre-registered before the run, two arms per boot (`plain`; `coincide`, a victim stream aborted right after the intruder
is posted so another session retires in the seed's iteration), OFF/ON/ON/OFF, N=5 per arm per order, both orders, 37
to 58 C, 32.6 to 503.1 W, replay PASS x8, verbatim: `DAY25 RETIRE VERDICT: admissible=True plain-R1-pass1=within_bound
plain-R2-pass1=within_bound plain-R3-pass1=within_bound plain-R1-pass2=within_bound plain-R2-pass2=within_bound
plain-R3-pass2=within_bound coincide-R1-pass1=within_bound coincide-R2-pass1=within_bound coincide-R3-pass1=within_bound
coincide-R1-pass2=within_bound coincide-R2-pass2=within_bound coincide-R3-pass2=within_bound seam-pass1=exercised
seam-pass2=exercised -> HOLDS (R1, R2, R3 within 3.0 ms in both arms and both passes; the seam exercised in both
passes)`; every ON minus OFF delta 0.1 to 0.7 ms; `settled synchronously by a session retire` 10 of 10 in both ON boots.
A first attempt failed at the harness's prompt calibration (kept as a labelled failure, fixed, rerun; no rule moved).

Battery (`integration-day12/integ37-cpu-battery/`, tree `83f9be5e5`, CPUQuota 1200 percent): fmt, portable suites,
memra-server suite, clippy, censuses, collector pytest, engine CPU lib tests, tier tests, engine, server and tier clippy
`-D warnings`, marker census, workflow keys, perf board: rc=0; `git diff --check` tripped on receipt logs (marked
`-whitespace`). Local 5090 `tools/serve-smoke.sh` (door OFF): `serve-smoke: 0 failed`.

Revuto round 1 on #638, one finding, real, fixed by the lead: `host_restore_take_ready`'s fail-closed arm could never
release the source entry's pin because the `let`-else scrutinee `(r.cache.take(), r.pin.take())` had already moved the
pin into a tuple the refutation arm cannot reach, so a ready restore without a cache left its entry pinned for the
boot (out of every eviction index, `pinned_left` on every purge). The shape is decided before anything moves; a missing
cache releases the pin and drops the record. CPU test `a_ready_restore_without_a_cache_releases_its_pin_at_take_ready`
(the entry's pin count returns to 0). Server clippy `-D warnings` and the memra-server suite (804 passed) green, gated
before the commit. Round 2, two findings, both real, fixed: (1) the restore park counted into the parked-only wait's
counter but the wait was still guarded on a not-ready `Promoting` entry alone, so a whole-entry hit on an idle box parked
its request and spun the owner thread through park-and-requeue ticks (the #627 shape again); the guard now reads a
not-ready `Promoting` entry OR a not-ready `Restoring` request, and the census test pins it. (2) the probe called any
other request's READY restore an orphan and dropped it on sight, so the three-tick grace was reachable only from the
tick top's expiry and the first other request through the probe killed the state (the owner request can be requeued by
an earlier admission gate without reaching the probe); the probe now drops it only past `RESTORE_READY_TICKS`, within the
grace it goes through and the state waits for its owner; the CPU test covers both readings. Server clippy `-D warnings`
and the memra-server suite (804 passed) green, gated before the commit.

## integ38 (`lane/spill-integ38-20260922`): A day 22 (Move 2 slice 3, the D2D receipt term), C day 26 (5090 door gates on the slice-2 tree; the hit gate's unarmed ON arm) and C day 27 (the hit gate's ON arm armed), plus one lead fix
Lane tips merged, in order: C `a76d669e4` (day 26), C `24bd36edf` (day 27), A `b94bee809` (day 22), all on main `58b814abe` (#638); no conflicts. Lead fixes on top: `269ef2cec` (the Block wait covers the receipt event), `9d4b17761` (the DOCS_RS stub emits the receipt fatbin; receipt-log attributes), `cc754b476` (the oracle cell orders its upload).

**C day 26.** The 5090 door gates on the slice-2 tree (9B, 27B for the twin), attempt 1 lost twelve cells to a
foreign 22 GB hold outside the canonical lock (`[server] FATAL: worker init failed: load gate:
DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`, kept as `attempt1-oom/`, cause quoted with the card's
snapshot; nothing foreign touched), then the battery waited bounded for an idle card and ran 07:51 to 08:01Z, verbatim:
fault-default `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (67 ok, `receipt seq=1 expected 1 + 0 ... = 1`, `seq=2 ... = 2`);
fault-plain `ALL GREEN` (67 ok, `seq=3 expected 1 + 2 ... = 3`, `seq=4 expected 2 + 2 ... = 4`); failure x4 `ALL GREEN`;
identity x4 `ALL GREEN (teeth=0)`; hit OFF and ON `ALL GREEN (qwen)`; twin27 OFF and ON `... -> PASS` (no `REFUSED`
line); unit cells `8 passed`, engine D2D cells `2 passed`. The restore route engaged on this card in two arms
(`identity-plain-on`: `restore submitted off the tick: 64 tokens, 16 planes (53.6MB), ticket seq=6 on the contracts
door's copy stream; recurrent state copied on the owner stream; request parked` then `restore landed off the tick: 64
tokens (53.6MB) complete after 1 poll(s) ...`, and `seq=7`; `fault-plain`: one restore per promote cell), zero refused,
dropped or disabled lines. The restore-exercising cell was skipped by its pre-registered condition (the identity gate's
plain arm is the smallest engaging shape: a 64-token promoted entry, the `lookup` floor is 64 on every model). Census
item 11 (`HOSTPREFIX-DOOR.md`): the restore route reaches its class check past six silent gates and then refuses by
name and silently (TP shards, latent planes, `e.draft` for every `insert (spec-boundary)` entry, `e.dspark_draft`,
`pos != toks.len()`, empty boundary logits, no KV plane); a draft plane's off-tick restore needs the `MtpScratch`
allocated at the probe and owned by the `Restoring` state, a borrowed destination view, the source under the trunk's
pin, the producer fence on the owner stream, rule 3's wait before the deferred prime's first draft-head read, the
geometry checks moved to the probe, and slice 3's receipt term over the draft role; 11 of 14 hit-gate entries and 12 of
16 hits are draft-bearing on both rigs. **Finding, lead-confirmed from the receipts:** `tools/spec-on-cache-hit-gate.sh`
boots with no `MEMRA_KV_HOST_MB` (zero `[prefix-host]` lines), so the door is off at `hpx.armed()` before any class
check and every "hit gate ALL GREEN OFF and ON" line from A's days 17 to 21 and C's days 22 to 26 exercised the same
unarmed program in both arms; the identity clause it carries is the spec-on-hit identity with the tier off. Ruling 33:
a gate's "door ON" arm asserts that the door engaged (an arming line and, where the entries allow, a route line) or it
is not a door arm; C day 27 (running) arms the hit gate's ON arm and re-reads it on both cards, and the review table
marks the earlier lines as unarmed receipts.

**C day 27.** `tools/spec-on-cache-hit-gate.sh` with `MEMRA_KV_HOST_CONTRACTS=1` in its environment now exports
`MEMRA_KV_HOST_MB=8192` (the identity gate's budget; an exported value is respected) to both boots and asserts per boot
`[prefix-host] on: budget`, `[prefix-host] contracts door ON`, none of `TIER DISABLED` / `CAPTURE OFF-TICK DISABLED` /
`RESTORE OFF-TICK DISABLED`, and across the two boots at least one `(capture|restore|demote|promote) submitted off the
tick`; both arms print an entry-class census (`insert (spec-boundary)` against `insert (seed)`, hit shapes, the class of
the identity clause's own namespaces). The door arm is qwen-only (gemma refuses, exit 2); OFF arm and identity clause
unchanged; no new `MEMRA_*` read. Dry-run before the commit: the day-26 hit-on log reads two FAILs under the new
assertions. Runs on tree `b1e9c75b6`, verbatim: local RTX 5090, 9B, `hit-off` `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`
61 ok, `hit-on` `ALL GREEN (qwen)` 68 ok; target card, 27B, the identical two lines, 61 and 68 ok. ON-arm engagement on
both cards and both boots: `[prefix-host] on: budget 8590MB pinned cacheable host RAM (MEMRA_KV_HOST_MB, startup budget
policy)`, `[prefix-host] contracts door ON (MEMRA_KV_HOST_CONTRACTS=1): 1 model program identities ... host tier armed`,
`ok: door arm: 7 route submission(s) across the two boots`; the spec-off twin `capture submitted off the tick (seed): 64
tokens, 32 planes (158.8MB)` x2 and `restore submitted off the tick: 64 tokens, 32 planes (158.8MB), ticket seq=2` /
`seq=3` / `seq=5` (its r2, r3, g2 hits), each `restore landed ... after 1 poll(s), 2.1ms`; the spec-on boot one capture and
one restore (the np hit); the 9B the same at `16 planes (53.6MB)`; zero refused, dropped or latched lines. What the identity
clause covered: spec-on side rows on draft-bearing entries under the tick program against spec-off side rows restored
through the door's off-tick route, byte-identical: the first hit-gate receipt on any card where the clause covers the
route (plain hits; the draft-bearing restore stays owed). Review-table corrections (`HOSTPREFIX-DOOR.md`): every earlier
door-ON hit-gate log on both lanes has zero `[prefix-host]` lines and the server's own `[kv-host-contracts]
MEMRA_KV_HOST_CONTRACTS=1 with no host tier on this boot (MEMRA_KV_HOST_MB=0): nothing to route, no program identity
built` line (A target card days 17, 18 twice, 19, 20, 21 and 5090 days 17 twice, 18; C 5090 days 24 and 26); the
day-26 row carries the note, two new rows carry the armed runs per card, section A gains an owed-cell row. Extra finding
fixed by C: the gate's `stop()` was a blanket `pkill -x memra-server` and its EXIT trap ran after the lock was released;
on the shared box A's day-22 server appeared right after C's cells (C's after-snapshot shows `0 MiB`, so nothing was on
the card when the trap fired); `stop()` now kills only the child of its own `flock` wrapper (`c1454a5a4`), re-run ALL
GREEN on both cards. Owed and stated: the hit gate has no `--external-lock`.

**A day 22 (Move 2 slice 3, the receipt term; memra#536 comment posted, issue open).** Under the door:
`memra_tier::conformance::d2d_receipt_witnessed` (rules 1 and 2) and `d2d_receipt_refused` (rules 3 and 4) over
`D2dReceiptFixture`, with the program `receipt_digest` (four wrapping u64 lanes of `mix64(w_j + (j + 1) * C_l)` over LE
words, byte count folded; order-independent, so block and atomic order cannot move the value); four bindings, contracts
81 passed (77 before). Engine: `cu/tier_receipt.cu` (`d2d_receipt_digest`, `tier_delay_spin`) as its own fatbin
(`MEMRA_TIER_RECEIPT_FATBIN`), loaded only by `new_with_copy_stream`; per D2D batch a fenced `ReceiptScratch` (lanes
zero-filled on the owner stream, the copy stream waits on that event); on the copy stream, behind the producer fence:
source digest, the copy, destination digest, one D2H of the lanes, the receipt event; `progress` lands a D2D item only
with its lanes (destination digest = checksum, source digest = expectation; `Completion::require`'s clause compares);
`d2d_receipt`, `inject_d2d_early_reader`. Worker: receipt lines naming both digests for both classes
(`contracts door D2D capture receipt: ... source_digests_sha256=... destination_digests_sha256=... require=ok`);
`ReceiptMismatch` arms: capture retires and acknowledges the ticket, takes every fresh plane back and drops it, publishes
nothing, latches the tier and the capture route; restore drops the destination cache (the copy landed, so the free is
safe), releases the source pin, primes nothing, latches the tier and the restore route (`host_restore_latch_landed`).
Fault values `d2d-delay-capture` / `d2d-delay-restore` (one-shot, taken by their own class only; a Move 1 `take_fault`
leaves them armed): the first capture or restore of the boot delays its copy 200 ms on the copy stream and takes its
destination digest from an unordered early reader on the owner stream. Two CPU tests (the settle arm, the fault
classes), FLAGS.md rows (fault values, door sentence, fatbin plumbing), KERNELS.md section. No new numeric program for the
KV bytes; `unsafe` at the two documented launches and the pinned zero-fill. Fault gate's new cells, verbatim: `capture
receipt refused: source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=1462aed093ef5fee..`, `restore
receipt refused: source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=1462aed093ef5fee..`, every clause of
`d2d-capture` and `d2d-restore` `ok`, `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`. Target card, final tree `bb1a2b212`,
verbatim per arm: identity `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` default and plain, OFF and ON; failure
`KV-HOST-SPILL FAILURE GATE: ALL GREEN` both arms; twin `V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` both arms; hit
`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` both arms (A's runs predate C day 27: unarmed, marked so in the table); unit
`8 passed` (server), `5 passed` (engine `d2d_*`); the green arm live in plain identity ON: 4 receipts `require=ok`, equal
digests. Finding 1 (first sitting, tree `ac1ac91c5`): engine cells `3 passed; 2 failed`; cudarc's implicit per-slice
event tracking, which the engine disables at `Engine::new` (`MEMRA_EVT=1` keeps it), stayed on in the plain test context
and ordered the fault's early reader behind the memcpy; the price cell failed on timing-disabled default events; fixed in
`bb1a2b212` (lanes zero-fill fenced explicitly, fixture disables tracking as the engine does, timing events), every gate
re-run. Finding 2, cell (v), verbatim: `copy_median=0.158 digest_median=0.168 pair_median=0.335 pair_over_copy=2.12` and
`copy_median=0.156 digest_median=0.167 pair_median=0.335 pair_over_copy=2.15` (one 158 MiB span, N=5 per order, both
orders, event-timed on the copy stream). By the day-19 rule ("unless the digest's cost on the copy stream exceeds the
copy's own time") the pair exceeds the copy, the clause under which the `Unwitnessed` arm was to be the receipt; A
reports it for the door review and relaxes nothing: the receipt as landed proves the bytes (the two red arms are the
evidence `Unwitnessed` could not give), the cost sits on the copy stream off the tick, 0.34 ms against a 14 ms tick.
**Lead reading, for the door review and not decided here:** the day-19 rule compared digest to copy without weighing the
stream; the review weighs (a) as landed against (b) with that number, and the 5090 price cell (this battery's GPU step)
joins it. Finding 3: the #638 take-and-match audit found one instance (the lead's); slice 3 adds no pending or ready
state. Budget 4.6 agent-hours against 4 (the second card sitting after the cudarc finding).

**Lead review of A day 22 (finding, fixed in this integ, `269ef2cec`).** `CudaTransfers::synchronize`, the host wait
behind every `ContractWait::Block` settle (the retire seam, a session retire, shutdown), waited on the items' events and
returned; the receipt's D2H is recorded on the copy stream AFTER the last item's event, and `progress` lands a D2D item
only with its lanes. In that window `capture_landed` reads false on a landed copy and the settle's Block arm reports
`capture did not land after a host wait on every item's event` (or the restore twin) and latches the tier for the boot:
a spurious fail-closed. The wait now covers the receipt event (between the item waits and the `unknown` reset, the same
`Quarantined` reading when the seal failed); the census test
`integ38_synchronize_waits_on_the_receipt_event_after_the_items` pins the order and that exactly two recorded-event
waits exist. Nothing else in the diff moved; A's arms and receipts read as stated (the mismatch arms, the pin release, the
fresh-plane take-back, the one-shot fault classes, the oracle against the kernel, the zero-fill fence). Ruling 34: a
batch's landing includes every event the settle reads, and a host wait that claims "every event" names all of them; a
schedule that adds an event to the landing adds it to the Block wait in the same commit.

**Battery attempt 1 (tree `269ef2cec`, receipts `integ38-cpu-battery-attempt1/`, `integ38-serve-smoke-5090-attempt1/`)
read two CPU reds and one GPU red, all three fixed in this integ.** (1) The cross-target GPU-less clippy
(`DOCS_RS=1`, the pass CI's runners take) failed with `environment variable MEMRA_TIER_RECEIPT_FATBIN not defined at
compile time`: `build.rs` added `cu/tier_receipt.cu` to the nvcc table but not to the DOCS_RS branch that writes an
empty placeholder per fatbin; `9d4b17761` adds the stem and the env row, the pass is `Finished` on the fixed tree. Every
other CPU step was rc=0 (fmt, portable suites, memra-server suite, censuses, collector pytest, engine CPU lib, tier
suite, engine/server/tier clippy `-D warnings` in 156 s, markers, workflow keys, perf board). (2) `git diff --check
origin/main HEAD` rc=2 on the lanes' raw cargo-test logs (trailing whitespace after `... ` in a test name line, blank
line at EOF): `.gitattributes` `*.log -whitespace` in `pro-single-day22/` and `rtx5090-day26/`, as every earlier receipt
dir. (3) The engine GPU cells under the 5090 lock: `d2d_receipt_digest_matches_the_cpu_oracle ... FAILED` at
`n=8388608: the device digest is the oracle's` (four of five cells ok, the price cell included). Not the kernel: the
test uploaded the pattern on the owner stream and digested it on the copy stream with nothing between them, and with
cudarc's tracking off (as the engine runs) the 8 MiB span read mid-upload; the target card's upload won the race in
both of A's sittings. `cc754b476` synchronizes the upload (the engine's route waits on the producer fence); the cell is
green 4 of 4 on the 5090 under the lock. The 5090 serve-smoke on the attempt-1 tree: `serve-smoke: 0 failed` (gemma4 and
Q35 arms SKIP for absent models, as on every rig receipt). The final battery on `cc754b476` is in
`integ38-cpu-battery/` and `integ38-serve-smoke-5090/` (its window and rc lines quoted in the self-review).

**Revuto round 1 on #639 (real, fixed).** The `d2d-delay-*` fault's spin was launched inside the per-item loop of both
D2D submits, so the documented 200 ms multiplied by the item count (`items=32` on the refused capture in A's target-card
receipt: 6.4 s of copy-stream spin, linear in the layer count) and any Block settle of that ticket held the owner thread
for the whole window. The spin is now issued once per batch, gated on the first item (the copy stream is serial, so
every item's copy waits behind it; the owner-stream early readers still win); the contract doc says "ONCE"; the census
test pins the gate and one spin site per class. Re-run on the local 5090 under the lock: engine `d2d_*` cells `5 passed`;
`tools/kv-host-contract-fault-gate.sh` with the door ON (`MEMRA_HOSTGATE_CACHE_MB=64`, 9B) `KV-HOST-CONTRACT-FAULT GATE:
ALL GREEN`, both D2D cells refusing by receipt (`integ38-fault-gate-5090-revuto/`).

**Revuto round 2 on #639 (real, fixed).** The engine took its one-shot `early_reader` arm AFTER every fallible admission
step of both D2D submits (empty batch, overflow, per-op layout/epoch/owner/fence checks, `receipt_scratch`), while the
worker had already spent its door value at the arming site; a refused submit left the engine's class-agnostic arm live for
the next batch of either class, so a refused capture submit could refuse the next restore's receipt and latch the restore
route, contradicting the "taken by their own class only" invariant and mis-attributing a gate cell's refusal. The arm is
now taken first, right after the thread check, in both submits; the census test pins the take ahead of the first refusal
per class. Engine `d2d_*` GPU cells `5 passed` on the 5090 under the lock.

## integ39 (`lane/spill-integ39-20260922`): A day 23 (the draft-bearing restore through the door, whole) and C day 28 (the hit gate's `--external-lock`, the isolating stall cell, the 5090 receipt price)
Lane tips merged, in order: A `0ebdde372` (day 23), C `efd9f6574` (day 28), on main `f80438799` (#639). One conflict,
`HOSTPREFIX-DOOR.md`, resolved by union; the two lines the union reported as absent were each the OLDER variant of a line
the other lane rewrote (A's item 11 gained its day-23 sentence over C's day-27 tail; C's section-E clause gained the
day-28 numbers over A's older text), verified against both parents before the commit (ruling 27's set-diff).

**A day 23 (Move 2 owed item 2, the restore half; memra#536 comment posted).** Under the door, pre-registered before
code (`0717ab470`): `memra_tier::conformance::d2d_restore_draft_ready` with its red arm
`d2d_restore_draft_read_before_its_wait_is_unordered` and CPU bindings (contracts 84 passed). Engine:
`RestoredDraftScratch` (the `MtpScratch` plus `pos`, `k_bytes`, `v_bytes`; `destinations()` lends two exact
`CudaViewMut<u8>` spans; `set_len` and the OFF path's `copy_from_entry`), `alloc_restored_draft_scratch` (every OFF
geometry check moved here: MTP head, ring-backed scratch, layout, capacity, truncated plane), and the two constructors
sharing one tail `spec_session_from_restored_scratch`: `spec_session_from_restored_deferred` (OFF: alloc, copy on the owner
stream, `set_len`, tail) and `spec_session_from_restored_ready` (ON: takes the filled scratch, checks `draft.pos`,
`kv.len == pos`, copies nothing). Worker: the probe decides the draft at the probe (`host_restore_draft_decision`: the
admission estimate specs the request, no DFlash drafter owns the model, no grammar, and `spec_restore_refusal` with the
load guard and the penalty window evaluated there), allocates the scratch and sets its length before the submit, and the
batch carries the draft K and V spans as two more `D2dRestore` ops under the SAME producer fence and the SAME pin as the
trunk (`items=34` on the 27B against 32 plain); the scratch is owned by the pending `Restoring` record: forgotten with
the cache under a `Latched` settle (never a free under a running copy), dropped after a `ReceiptMismatch` (the copy
landed) and at a landed drop, handed over only at `take_ready` beside the cache and the pin; the consumer wait installed at
the settle fences the draft items with the trunk's, so the deferred prime's first draft-head read is behind rule 3's
wait; a declined draft is typed (`draft plane not submitted (why)`) and admission then takes the OFF copy on the owner
stream from the pinned entry; a probe/admission disagreement is a typed census line in both directions (neither fired).
Two CPU tests (the path census, the decision table). Target card, tree `2b850b2b0`, one sitting 09:54Z to 10:08Z,
verbatim: identity `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` x4 (default and plain, OFF and ON); failure
`ALL GREEN` x2; fault `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (93 ok; `d2d-restore` still refuses: `restore receipt
refused: source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=1462aed093ef5fee..`); twin `V1=ok ... V6=ok
-> PASS` x2; hit OFF `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok); hit ON, armed, `ALL GREEN (qwen)` (68 ok, 0
FAIL), `ok: door arm: 19 route submission(s) across the two boots`; unit `8 passed` (server), `5 passed` (engine `d2d_*`).
Hit ON spec-on boot: 13 `restore submitted off the tick`, 12 with `; draft plane N rows (X KB) in the batch` (8 x `64
tokens, 34 planes (158.9MB) ... draft plane 64 rows (118.8KB)`, 3 x 96 tokens, 1 x 128 tokens), 13 `D2D restore receipt
... require=ok` (12 `items=34`, 1 `items=32`), 12 `restore landed ... after 1 poll(s), 2.3 to 2.4ms ...; draft plane
ready`, 12 `spec restore: ... + draft plane from cache`; zero refused, disabled, declined-at-probe or disagreement lines;
every `r1/r2/r3/g1/g2 spec==plain byte identity` ok. The route engaged on every draft-bearing hit the gate serves (12 of
16 routed hits, item 11's count): the first hit-gate receipt on any card where the identity clause covers the route for
draft-bearing hits. The receipt's byte counts price the draft plane at 118,784 B per 64-token entry on the 27B (1,856
B/token, item 10's estimate). Cell (v) on this tree: `pair_over_copy=2.11` / `2.13` (day 22: 2.12 to 2.15), reported.
A's day-22 CPU census window (a fixed 500 chars of the `ReceiptMismatch` arm) now ends at the arm's real end, assertions
unchanged. Owed and stated: the 5090 door gates on this tree (run by the lead below); Move 2 still owes the recurrent
f32 state, the spec-boundary capture route, the isolating stall cell's capture share and the review's reading of cell
(v). Budget 1.6 agent-hours against 5.

**C day 28.** (1) `tools/spec-on-cache-hit-gate.sh [--external-lock FD]` in the identity gate's shape: under the collector
the per-boot `flock -w 300` wrapper is absent, the FD is verified by `tier-lock-proof.py` into `LOCK.json`, `stop()`
addresses `$SERVER_PID` only while its comm reads `memra-server`; without the flag unchanged; teeth `--lock-self-test
FILE` plus `tools/test_spec_on_cache_hit_gate_lock.sh`, verbatim `SPEC-ON-CACHE-HIT-GATE LOCK TEETH: ALL GREEN (7 ok)`
(default arm `probe=held`, external `probe=free` with inode and mtime unchanged, a non-numeric FD `REFUSED` exit 2), wired
into `ci.yml`'s gate-teeth step; `docs/TESTING.md` bullet; no new `MEMRA_*` read; not yet run under the collector on a
card (C day 29, running). (2) The isolating stall cell (Move 2 owed item 3), target card, one hold 09:35Z to 09:46Z, ten
boots, N=5 per arm per order, both orders, two passes, 33 to 60 C, 33 to 501 W, ten `STALL REPLAY: PASS`, verbatim:
`DAY28 ISOLATION VERDICT: capture pass1 on-off +0.7 (unc 0.2) isolated; restore pass1 on-off +0.2 (unc 0.2) isolated;
capture pass2 on-off +0.7 (unc 0.2) isolated; restore pass2 on-off +0.2 (unc 0.1) isolated; admissible=True`. The
restore class isolated by a whole-entry zero-suffix hit intruder (no prime of its own, `hit: 5152 of 5152` every run):
`stall_median=8.9` / `9.0` OFF against `9.1` ON, the 9 ms being the allocation plus the recurrent f32 copies that stay on
the owner stream in both arms (owed item 1); the rows' move changed the tenant's stall by nothing resolvable. Capture
class: `283.7` against `284.4`, nothing evicting at the 8192 MB cache (named before the run). Finding, not tuned: the
prime-only boot (cache off) read `301.5`, 17 ms above the same prime inside the cache-on boots, so the pre-registered
subtraction reads `share=-17.8` with the wrong sign and the capture arm's own share stays unread. (3) The receipt's price
on the local RTX 5090 under the lock (58 C, P8 before, no compute app), verbatim: `order=copy-first ...
copy_median=0.394 digest_median=0.221 pair_median=0.442 pair_over_copy=1.12` and `order=digest-first ... copy_median=0.395
digest_median=0.221 pair_median=0.441 pair_over_copy=1.12`; rows in the door table's cost section per card, never across
cards. Process slip fixed and recorded: a `;` in a commit chain let `0ae22b577` push with trailing whitespace despite a red
`git diff --check`; `1791a4f40` corrected it (ruling 30 restated in C's day-29 brief). Budget 3.0 against 4.

**Lead review of A day 23.** Read the engine refactor (the OFF path's alloc, copy, `set_len`, tail order is
census-pinned; the ready constructor allocates and copies nothing), the worker's ownership of the scratch at every
settle and drop path (Latched forgets, ReceiptMismatch and landed drops free, `take_ready` hands over under `ready` only,
`host_restore_drop` settles a pending contract with Block before the record drops), the shared pin and producer fence for
the draft items, and the installed wait's coverage of every unfenced non-D2H item. No finding. The probe-side decision
duplicates admission's inputs by construction; its disagreement arms are typed and the gate read neither.

**Battery (tree `6a1909924`, receipts `integ39-cpu-battery/`, `integ39-serve-smoke-5090/`, `integ39-hit-gate-5090/`).**
All fifteen CPU steps rc=0 (fmt, portable suites, memra-server suite, cross-target `DOCS_RS=1` clippy, check-flags,
publish and docs-registry censuses, collector pytest, engine CPU lib, tier suite, engine/server/tier clippy `-D
warnings`, markers, workflow keys, perf board, `git diff --check`). Local RTX 5090: `serve-smoke: 0 failed` (gemma4 and
Q35 arms SKIP, absent models), engine `d2d_*` GPU cells `5 passed` under the lock, and the door gates A stated as owed
on this tree, the hit gate OFF and ON with the tier armed (9B, gate-internal lock), verbatim: `SPEC-ON-CACHE-HIT GATE:
ALL GREEN (qwen)` 61 ok OFF, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 68 ok ON; the spec-on boot 13 `restore submitted
off the tick`, 12 with a draft plane in the batch (8 x `64 rows (118.8KB)`, 3 x `96 rows (178.2KB)`, 1 x `128 rows
(237.6KB)`; the 9B's draft plane is the same 1,856 B/token), 12 `draft plane ready`, 12 `draft plane from cache`, zero
refused, disabled, declined-at-probe or disagreement lines. The draft-bearing route now has the hit gate's identity
clause over it on both cards.

## integ40 (`lane/spill-integ40-20260922`): A day 24 (the spec-boundary capture route, whole) and C day 29 (decision cell (i): the two Move 1 programs in one window; the hit gate under the collector)
Lane tips merged, in order: C `89a6aa8d6` (day 29), A `d5708266f` (day 24), on main `ebe3fe17d` (#642). One conflict,
`HOSTPREFIX-DOOR.md` (C's tip against main), resolved by union; the flagged lines were the two Move 1 cost rows C
rewrote with the same-window pair, C's rewritten E clause, and the day-27 tail already superseded on main, each verified
against both parents (ruling 27). A's merge was clean.

**A day 24 (Move 2 owed item 2, the capture half; memra#536 comment posted).** Pre-registered before code (`dc5eb64bc`).
Tier first: `d2d_capture_draft_publish` (landing is EVERY item's event: a publish with the trunk landed and a draft item
running is refused `NotReady`, both or neither; the receipt term witnessed per class) with its red arm
`d2d_capture_draft_published_with_the_draft_unlanded_fails`; contracts 87 passed (84 before). Worker: slice 1's submit
half factored into one core both publishers call (`host_capture_submit` over `CaptureSubmit`; the seed route's lines
byte-unchanged), and `prefix_spec_capture_off_tick` as the spec-boundary route: refuses by name to the OFF program (TP
shards, latent planes and tails, a snapshot at another position, a short or empty plane, the SWA ring), settles a pending
capture `Block` before a second, skips a prefix the settled capture already published, and hands the capture back
untouched on `OnTick`; the draft scratch rows `[0..pos)` are two more items of the same batch under one fence, ticket and
receipt (`CapturePlaneClass::Draft`, `Done { kv, draft }`), so the `Capturing` entry owns both classes and publishes both
or neither; a refused registration or retention of a draft plane takes every fresh plane back through `refuse`. Census
item 10's open question (the borrow's lifetime) is answered by naming the one path that frees a source inside the tick,
the MTP demotion (`into_demoted` drops the `MtpScratch`), which now settles a pending capture `Block` first (the spec and
the DFlash demotions). No new engine seam, no new flag, no new numeric program. Target card, tree `185c57b4f`, one sitting
10:59Z to 11:13Z with C sharing the card (zero lock retries), verbatim: identity `KV-HOST-SPILL IDENTITY GATE: ALL GREEN
(teeth=0)` x4; failure `KV-HOST-SPILL FAILURE GATE: ALL GREEN` x2; fault `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (93
ok; both D2D cells still refuse); twin `V1=ok ... V6=ok -> PASS` x2; hit OFF `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`
(61 ok); hit ON armed `ALL GREEN (qwen)` (68 ok), `ok: door arm: 30 route submission(s) across the two boots` (day 23:
19); unit `8 passed` (server), `5 passed` (engine). Acceptance met: all 11 of the spec-on boot's `insert (spec-boundary)`
publishes routed with the draft plane (`capture submitted off the tick (spec-boundary): 64 tokens, 34 planes (158.9MB)
...; draft plane 64 rows (118.8KB) in the batch`), 12 receipts `require=ok` (11 `items=34`), 12 published, day-23's
restores still engaging (13 submitted, 12 with the draft plane), zero refused, trunk-only, dropped, skipped or latched
lines, every `spec==plain byte identity` ok. Findings: (1) the draft plane at publication is 118,784 B per 64-token
entry on the 27B, the restore side's figure exactly; (2) on the gate's shape every spec-boundary capture settles at the
SESSION RETIRE (`settled synchronously by a session retire`, `after 1 poll(s)`, 106.3 to 204.6 ms), not at a tick-top
poll, while the seed capture published at a poll: the slice-1 retire seam behaving as this slice requires (the sources
are the live session's planes), but a session retiring in the tick after its prime stop pays a host wait for the 159 MB
copy, so the moved share on that shape is not the copy's full cost; pricing it is owed item 3; (3) no finding against the
engine or the gates. Budget 2.9 against 5.

**C day 29 (Move 1 owed item 4, decision cell (i)).** Arm X = today's tree (the copy stream), arm Y = the day-16 tree
`1646d421b` rebuilt on the box (`/root/wt-c-day16`, the on-tick owner-stream program; `git diff --stat 1646d421b
653c997f4 -- crates/ Cargo.toml Cargo.lock` empty, both binaries hashed), door ON in both, the day-16 script's `on` boot,
twenty boots `XYXYXYXYXY` then `YXYXYXYXYX` (N=5 per arm per order), one collector hold, a dry boot of Y first (the day-16
tree built in 3 m 27 s and booted without a patch), 41 `STALL REPLAY: PASS`, `ADMISSIBLE all_receipts=True`, 33 to 51 C,
32.5 to 360.8 W. Verbatim: `DAY29 CELL(i) class=demote order=o1 y_minus_x=+43.4 unc=0.6 -> isolated (Y owner stream 193.4
N=5; X copy stream 150.0 N=5)`, `order=o2 y_minus_x=+43.0 unc=0.1 -> isolated (Y 193.1; X 150.0)`; `class=promote order=o1
y_minus_x=+13.3 unc=0.4 -> isolated (Y 162.9; X 149.5)`, `order=o2 y_minus_x=+12.8 unc=0.2 -> isolated (Y 162.5; X
149.7)`; `DAY29 CELL(i) CLAUSE class=demote stall_median(second stream)=150.0 idle_p99_sitting=14.9 -> clause_not_met`,
`class=promote ... =149.6 ... -> clause_not_met`; `DAY29 CELL(i) CLAUSE: NOT MET (demote=False promote=False
admissible=True); executed-not-qualified`. **Finding for the lead, not tuned (owed to A):** arm X's promote arm reads
149.6 where the day-18 and C day-23 trees read 81.9 in the same shape; the lines (100 of 100 promote runs) say why: on
today's tree the hit parks TWICE (`promote published off the tick ... 19.6ms`, then `restore submitted off the tick ...
request parked`, `90.3ms to re-admission`, Move 2's restore route, absent on the C day-23 tree), and the inline demote
publishes `after 1 poll(s), 22.4ms` (C day 23: `97.0ms`), so its two on-tick hashes land on one tick and the promote
arm's tenant stall equals the demote arm's (150.0). Which slice between `0713c1a79` and the integ38 tip moved the copy's
landing is not determined. Tenant text sha `264b120d487de2c9` in all 41 receipts of both trees. Lead reading: a promoted
entry's first hit now takes the restore route and parks the request a second time; the re-admission wait (about 90 ms
for a copy that lands in about 2 ms) rides the tick that carries the demote's on-tick hashes, so the promote-then-hit
shape pays Move 1's park plus Move 2's park plus the hash tick. Owed to A (day 25): the second park's cost on this shape,
and whether a hit on an entry the promote just published should restore at all (the promoted planes are already on the
device: the restore copies device to device what the promote just landed). The hit gate under the collector
(`--external-lock 3`, OFF and ON in one hold): `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` both arms (61 and 68 ok), first
line `lock: collector's inherited FD 3 on /tmp/memra-gpu.lock (no flock of this gate's own)`, both `LOCK.json`
`{"owner": "collector", "lock": "/tmp/memra-gpu.lock", "mechanism": "inherited-flock-same-open-description", ...
"qualification": false}`. Budget 2.0 against 4.

**Lead review of A day 24.** Read the submit core (one fence, one ticket, one receipt for both classes; the draft
plane's refusal arms take every fresh plane back), the settle (`Done { kv, draft }` fills the shell's draft slot only from
a landed batch), the route's by-name refusals and the second-capture settle, and every in-tick source free (session
retire from integ37, the spec and DFlash demotions added here) settling a pending capture with Block first. The committed
prefix `[0..pos)` of a live session's trunk and draft planes is append-only past `pos` and never rewritten below it by
verification, so the borrowed source holds for the copy's life. No finding.

**Battery (tree `a6699b4c8`, receipts `integ40-cpu-battery/`, `integ40-serve-smoke-5090/`, `integ40-hit-gate-5090/`).**
All fifteen CPU steps rc=0. Local RTX 5090: `serve-smoke: 0 failed` (gemma4 and Q35 arms SKIP, absent models), engine
`d2d_*` GPU cells `5 passed` under the lock, and the door gates A stated as owed on this tree, the hit gate OFF and ON
armed (9B), verbatim: `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 61 ok OFF, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 68
ok ON, `ok: door arm: 30 route submission(s) across the two boots`; the spec-on boot 11 `capture submitted off the tick
(spec-boundary)`, all 11 with the draft plane in the batch, 12 `D2D capture receipt ... require=ok` (23 `items=18`, 2
`items=16` across both classes), 12 `capture published off the tick`, 13 `restore submitted off the tick`, zero refused,
disabled, trunk-only, dropped or skipped lines. Both draft-bearing routes now carry the hit gate's identity clause on
both cards.

**Revuto round 1 on #643 (real, fixed).** `prefix_insert_from_spec_boundary` is also the DSPARK and GLM5 publishers'
entry, called with `dspark_draft: Some(tail)` when `dspark_prefix_restore_on()` (or a GLM5 `export_draft_tail`); the
route was installed ahead of that handling and never saw the tail, and its by-name refusals (TP shards, latent planes and
tails, a foreign snapshot position, short or empty trunk layers) let a plain DSPARK or GLM5 cache through, so
`host_capture_submit` would have built a shell with `dspark_draft: None` and the settle would have published trunk and
draft and dropped the about 85 MB `DflashKvTail` silently, the regression `export_tail` exists to avoid, against DAY24's
own statement that the tail never reaches the route. The publisher now passes `dspark_draft.is_some()` into the route,
which refuses by name (`|| dspark_tail`) and hands the capture back for the OFF program whole, tail included; the census
test pins both the refusal and the publisher's argument. Server lib suite green after the fix (count in the receipts).

**Revuto round 2 on #643 (real, fixed).** The dspark demotion's comment claimed the DSPARK publisher never routes; it
DOES in the default configuration (`MEMRA_DSPARK_PREFIX_RESTORE` is default OFF, so `dspark_draft` is `None` and a
plain, non-latent, non-TP dspark cache passes the route's by-name refusals), which makes the settle before
`into_demoted` load-bearing, not belt-and-braces: a reader trusting "never routes" could drop it and the demotion would
move planes a copy is still reading. The comment now states the rule that applies; the GLM5 demotion gains the same
settle (`"a glm5 demotion"`) with its own stated reason (a GLM5 cache is refused by name today; the guard states the
rule where the session is consumed). Census test pins the load-bearing statement and the glm5 settle. Server lib 809
passed after the fix.

## integ41 (`lane/spill-integ41-20260922`): C day 30 (the door decision packet; the capture share read) and A day 25 (the double park priced; the retire-settle share priced)
Lane tips merged, in order: C `20a23f4d5` (day 30), A `483425d83` (day 25), on main `88d3dfd49` (#643). Both lanes had
branched from `ebe3fe17d` (integ39) and each carried its own integ40 half, so the union of `HOSTPREFIX-DOOR.md` left the
owed-cell table with three variants of the hit-gate rows and two of the isolating row; the lead rebuilt that table
three-way against the true base (C's rewrite of the isolating row plus A's day-25 sentence appended; A's day-24 capture
row inserted after the day-23 restore row; every row of main present once), `check-conflict-markers: OK`. Ruling 35: a
union across two lanes that both branched from an older main is checked against THAT base, not against current main; a
row "changed" against current main may only be older.

**C day 30.** (1) `research/spill-c-20260919/DOOR-DECISION-PACKET.md`, a draft for the owner's decide-by 2026-10-05 on
`MEMRA_KV_HOST_CONTRACTS` (not a `docs/decisions/` record until the owner decides): the question as section E states it;
what the door is today (Move 1 whole, Move 2 slices 1 to 3, the draft-bearing restore and capture routes, the receipt
term, every fail-closed arm typed, and what still runs on the tick: the recurrent f32 state, the by-reference demote
routes, the H2D settle-time wait, the demote's two on-tick hashes, the spec-boundary capture's retire settle); the
correctness table (every gate, both arms, both cards, tree and receipt path, verdict lines verbatim; the unarmed pre-day-27
hit-gate lines excluded per ruling 33); the cost table per card (the on-tick door, Move 1's halves, the same-window pairs
of C day 23 and cell (i), the isolating cells of days 28 and 30, the receipt price on both cards, the WC and cached pairs,
the hash micro-cell, the arena pair), each with N, order and regime; seven open findings the review must weigh; the three
outcomes the door hygiene rule allows and what each requires, stated without recommending. Every number re-read from its
receipt file (rule lines from `receipt.json`, pairs regenerated by the replay scripts, `ok:` counts counted in the gate
logs, regimes from each cell's `command.gpu.csv`); untraced: none. (2) The capture share, read: no admissible corrected
reading existed in prior receipts (day 28's prime-only arm was a cache-OFF boot whose prime is not split at the 5088 seed
boundary, hence its 301.5 against 283.7). Pre-registered and run: a cache-ON boot with `MEMRA_PREFIX_CACHE_MB=128` keeps
the seed boundary and the 308 MB insert is refused by the existing typed line `[prefix-cache] insert refused: entry
307986432 exceeds budget 134217728 (snapshot preflight, model gate)` before any copy and before the route; eight boots in
one hold, two passes in opposite order, N=5 per arm per order, 33 to 61 C, 32 to 501 W, eight `STALL REPLAY: PASS`,
verbatim: `DAY30 CAPTURE-SHARE VERDICT: pass1 refused on-off -0.1 (unc 0.3) under_resolution; pass1 share off +0.6 (unc
0.3) isolated; pass1 share on +1.2 (unc 0.2) isolated; pass1 capture on-off +0.7 (unc 0.3) isolated; pass2 refused on-off
-0.1 (unc 0.3) under_resolution; pass2 share off +0.5 (unc 0.3) isolated; pass2 share on +1.2 (unc 0.3) isolated; pass2
capture on-off +0.7 (unc 0.2) isolated; admissible=True` (base 283.1 / 283.2): the whole capture class is about a
millisecond of the tenant's tick on this card at this entry size, the door's part about 0.6 ms of it, ON above OFF; day
28's 17 ms was the prime's tail chunk past the seed boundary landing on a second tick, not a capture cost. Also fixed on
C's branch: main carried the day-27 hit-gate row twice (an integ39 union artifact). Budget 3.0 against 4.

**A day 25 (owed to A from cell (i); Move 2 owed item 3).** (1) The double park, target card, one hold, twenty
interleaved boots, N=5 boots per arm per order, both orders, 20 of 20 replays PASS, 33 to 52 C, 32 to 362 W, verbatim:
`DAY25 DOUBLE-PARK stall order=o1 on_minus_off=+64.0 unc=0.1 -> isolated (on 149.4, off 85.4)`; `order=o2
on_minus_off=+64.2 unc=0.1 -> isolated (on 149.4, off 85.3)`; `DAY25 DOUBLE-PARK e2e order=o1 on_minus_off=+105.8 unc=1.2
-> isolated (on 221.4, off 115.5)`; `order=o2 on_minus_off=+105.9 unc=1.1 -> isolated (on 221.4, off 115.4)`; `DAY25
DECOMPOSITION arm=on N_runs=100 parked_per_run=[2] restore_readmission median=90.1 range=89.7..91.5 ...
slack(readmission-completion) median=0.10 ... demote_in-completion median=74.9 idle_p50(tick)=13.46 residual median=+1.8
range=+1.1..+3.1`. The re-admission wait is one tick (the tenant's decode, 13.46) plus the inline demote's two on-tick
hashes (74.9) plus 1.8 residual; the poll cadence is the tick top (the parked-only wait needs `active.is_empty()`), which
polls demote before restore; the copy itself is about 0.5 ms (cell (v)). No existing typed refusal fits, so the pair is
door ON against OFF, stated as such. Both, with the numbers: by construction the restore route parks a whole-entry device
hit whose planes the same request's promote landed 6.5 ms earlier (one tick top for a 0.5 ms copy, unearned); the 90.1
ms magnitude is the hash tick's scheduling artifact; the tenant's +64 is the demote's hashes on one tick (57.9 of 64.1 by
arithmetic), not the second park (inside the +1.8 residual). Which slice moved the demote's landing (C's question) is not
determined. **Proposal 1 (A, not implemented):** in `host_restore_park_probe`, refuse the route by shape when
`hpx.promoted_pin` (the insertion pin of the promote published at this tick top, held exactly for this request's
re-admission) names the hit entry: one typed line, the OFF device-hit copy on the tick, no flag, no new state, no numeric
change; acceptance: this cell again with `request parked` 2 to 1 and `restore submitted` 0 in 100 of 100 ON promote
runs, the request's e2e down by the re-admission median (about 221 to 131), the tenant's stall within IQR of 149.4 (a
move either way is a finding), the day-21 restore arm still parking once and landing 100 of 100, hit and identity gates
ALL GREEN in both arms. (2) The retire-settle share: a typed clause on the capture publish line (`; the settle held the
owner thread H ms, entered A ms after submission`; `PendingCapture.settle_after_ms` / `settle_held_ms`; no new `MEMRA_*`
read); hit gate ON on the target card, same binary, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 68 ok, day 24's census
counts exactly; verbatim `DAY25 RETIRE-SETTLE settled_by='settled synchronously by a session retire' N=11 ... held_ms
N=11 min=0.37 median=0.41 max=0.44 | entered_after_ms N=11 min=106.10 median=150.50 max=204.50 ...
share_held_over_completion ... median=0.00`; the tick-top poll `held_ms ... min=0.37 median=0.37 max=0.38 (N=3)`. Day 24's
"the owner thread still pays a host wait for the 159 MB copy at the retire" is refuted by its own typed figure: the copy
had landed; the 0.4 ms is the settle's fixed cost. **Proposal 2 (A):** no reordering is earned; leave the seam and
record 0.4 ms (the smallest distinguishing form, a source-session identity on `PendingCapture`, named but not
recommended). Budget 3.4 against 4.

**Lead rulings on A's proposals (ruling 36).** Proposal 1 is approved for A day 26 as specified, with its acceptance
gate verbatim and one addition: the refusal's typed line names the pin and the entry so the gate can count it, and the
hit gate's ON arm on both cards must read the same route counts as day 24 minus exactly the promote-then-hit shape
(which the hit gate does not contain, so its counts must not move). Proposal 2 is accepted: the seam stays, 0.4 ms is
the recorded price, and Move 2 owed item 3 closes on this receipt. Lead review of A's day-25 engine change (two `f64`
fields on `PendingCapture`, timed around the settle, printed on the publish line, no parser splits inside the
parenthesis): no finding.

**Battery (tree `e14c270a6`, receipts `integ41-cpu-battery/`, `integ41-serve-smoke-5090/`, `integ41-hit-gate-5090/`).**
Fourteen CPU steps rc=0; `git diff --check origin/main HEAD` rc=2 on the battery's own receipt directory only (its
summary and logs, given the usual `-whitespace` attributes at commit; the tracked tree is clean, re-checked after the
commit). Local RTX 5090: `serve-smoke: 0 failed` (gemma4 and Q35 arms SKIP, absent models), engine `d2d_*` GPU cells `5
passed` under the lock, the hit gate OFF and ON armed (9B), verbatim: `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 61 ok
OFF, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 68 ok ON, `ok: door arm: 30 route submission(s) across the two boots`, 11
spec-boundary captures, 12 capture receipts `require=ok`, 13 restores, zero refused, disabled, trunk-only, dropped or
skipped lines. A's new typed clause on this card: 12 publish lines carry `the settle held the owner thread`, `held_ms N=12
min=0.42 median=0.55 max=2.63` (11 `settled synchronously by a session retire`, 14 `tick-top poll` lines across both
boots); this card's own figure, beside the target card's 0.41 median, never compared.

## integ42 (`lane/spill-integ42-20260922`): A day 26 (the unearned second park refused by shape), C day 31 (the 5090 pair cell and whole-budget arm), C day 32 (the `MEMRA_ADMIT_BY_MEMORY` packet), and #645 (the bare provider-id boundary rule)
Lane tips merged, in order: C `ffe9534a0` (day 31), C `82a26c0cb` (day 32), main `bc6e2f44d` (#645), A `645e649a6` (day
26), on main `f10973ab7` (#644). The door doc's union left variant lines again (both lanes branched from the integ41 tip
`e14c270a6`); the lead redid the file as a true three-way merge against that base (`git merge-file`), one hunk left (C
day 31's 5090 whole-budget row against A's unchanged copy of the base), resolved to C's row; no duplicated table row
(ruling 35 applied as its own procedure: three-way against the common base, union only for append-only ledgers).

**#645 (lead, from C day 32's observation).** `docs/FLAGS.md` named a rented pair by its provider and contract id in the
form `vast NNNNNNNN`; neither `provider_machine_id` (anchored on "machine") nor `provider_contract_id` (anchored on
"contract") saw it. `provider_bare_id = "(?i)\b(vast|runpod)\s+#?[0-9]{6,10}\b"` (severity 3) now does; four live
surfaces scrubbed keeping the card class (`docs/FLAGS.md` four lines, `docs/KERNELS.md`, `docs/RELEASING.md` two lines,
one engine doc comment), nine dated receipts grandfathered for this rule only with the reason (a record, not a
deployment fact; the pin is a content hash, so the next edit un-pins). `public-boundary: 599 matches (599
grandfathered, 0 new)`, `verify-allowlist` clean, self-test 52 passed. Merged as `bc6e2f44d` (revuto passed).

**C day 31 (the local RTX 5090's missing cells; this card's own figures, never compared to the target card).** The pair
cell (9B, cache 64 MB, host 8192 MB, default spec boot, 64-token draft-bearing entries 54.8 MB / `18 items`, four boots in
one collector hold, N=5 per arm per order, both orders, pooled N=10, 54 to 74 C, 9.5 to 169.9 W, `WC PAIR REPLAY: PASS
(12 checks)`; the write-combined premise printed inside the hold by `tier-transfer-gate roundtrip`: `PINNED-DEFAULT
device="NVIDIA GeForce RTX 5090 Laptop GPU" kind=write-combined flags=4`), verbatim: `DAY31 PAIR VERDICT: demote off 20.0
(N=10) on 57.3 (N=10) on_minus_off +37.3 unc 26.4 isolated; promote off 15.6 (N=10) on 21.1 (N=10) on_minus_off +5.4 unc
25.6 under_resolution; promote_minus_inline off 3.5 (N=10) on -19.3 (N=10) on_minus_off -22.9 unc 7.4 isolated; parked per
boot [0, 10, 10, 0]; pinned=write-combined; admissible=True`. Read, not tuned: a first-touch step in both arms straddled by
the pooled medians (IQR 18 to 20); the ON demote's `in` minus `from submission to completion` about 21 to 23 ms per
demote, far under the WC micro-cell's rate (about 490 ms per pass at 54.8 MB), so the hashes are not reading WC memory at
that rate on this tree (section D item 6's question, open); the day-29 double park reproduced on this card (`parked=10,
restore_submitted=5` per ON boot). The whole-budget failure arm (`MEMRA_KV_HOST_TENANT_PCT=100`, cache 64 MB), four
cells, 54 to 61 C: `KV-HOST-SPILL FAILURE GATE: ALL GREEN` x4 (14 ok, 0 FAIL), `[prefix-host] skip demote: entry 54.8MB >
host budget 1MB` (default) and `entry 54.6MB > host budget 1MB` (plain), each refusal in the ON arms after `demote
submitted off the tick: 64 tokens, 54.8MB, ticket seq=3, 18 items ...`, `D2H receipt ... require=ok`, `demote published
off the tick: ticket seq=3 complete after 1 poll(s), 36.2ms from submission to completion` (plain `16 items`, `24.9ms`).
The packet's section 3 and 4 5090 rows filled; items 3 and 4 carry A's day-25 lines and ruling 36; section 2 keeps the
retire settle as a statement, not a cost. Budget 2.6 against 4.

**C day 32 (`MEMRA_ADMIT_BY_MEMORY`, decide-by 2026-09-23).** `research/spill-c-20260919/ADMIT-BY-MEMORY-DECISION-PACKET.md`,
a draft for the owner, no recommendation, every number re-read from its receipt. **The load-bearing finding: every cell B
banked ran with the door OFF** (each `server.log` prints `[admit-mem] door=OFF ...`; the runner never sets the door). The
129x is `arm=i L0 ... median=129.26` (min 126.46, max 141.70, `alloc_B=8271167488`, `used_B_median=63987456`) on one RTX
PRO 6000 Blackwell at 600 W, 27B, `MEMRA_CTX` unset, tree `1c66ff10e`, N=5 per length, four receipts
(`B/pro-single-day26/cells/{ab,ba,warm}/REPORT.txt`, `B/pro-single-day27/cells/after-ab/REPORT.txt`); the local 5090, 9B,
`MEMRA_CTX=65536`: `median=11.30`. Gates covering the ON arm: 13 CPU tests ok on today's tree; the B200 bring-up cell
exists only as a quoted line (no receipt file in either repo); no ON boot anywhere in `research/spill-*` on either card;
the one darklanes ON boot printed the boot line and zero decision lines; the FLAGS row's receipt pointer holds no door
receipt; the requal cell is "not yet run". Nine unmeasured or unbuilt items, including that no `[admit-mem] id=
verdict=` line exists in any tracked receipt, part (b) has never fired and its premise moved on day 27, the two doors
have never booted together, and `docs/SERVING.md:912-914` still states the contract (a) would change. Missing list (none
run): the ON arm's allocated-over-used and `finish_reason` on both cards (B's pre-registered two-card, two-order, N>=5
cell), part (b)'s `reclaim demoted` count and tick cost with the host tier armed, part (c)'s 429 on a card, admitted
concurrency at the served context under ON, completion-digest identity OFF against ON, the FLAGS row's requal cell. **For
the owner:** the door's date is tomorrow and its ON arm has no receipt on any card; the three outcomes and what each
requires are in the packet. C also proposed the `MEMRA_KV_PARK_COMPACT` row text (`DAY32.md` section 2): the conflict is
inside `docs/FLAGS.md:324` itself (B's `72f89e233` put `decide-by: 2026-10-06` in the prose and left `0 = OFF by design`
in the value column); three exact replacements proposed, or the exempt-switch reading (keep `by design`, add the
exemption reason, withdraw the date). **Ruling 38 (lead):** the dated sentence is authoritative; the value column reads
`0 (default OFF), decide-by: 2026-10-06`, the prose sentence takes C's form, the stale PENDING clause takes C's optional
replacement; the lead edits `docs/FLAGS.md` in the next docs lane (a registry edit, its own reviewed change). C also
flagged the FLAGS.md header's provider id, fixed in #645. Budget 2.6 against 3.

**A day 26 (proposal 1, ruling 36).** `host_restore_promoted_this_admission` in `host_restore_park_probe`, between the
lookup and the class check: when `promoted_pin` names the hit entry (same pool key, `px.id_index(pin)` equal to the hit's
index, so an index shift cannot alias), one typed line `[prefix-cache] restore not routed (contracts door): the entry was
promoted for this admission (insertion pin id=P, N tokens, model M); the tick program copies it` and the OFF copy; no
flag, no state, no numeric change (the promote's ticket seq is not on the pin and not printed: carrying it would be new
state, ruling 36's addition applied to the pin id and the entry). Table test over the pin shapes (none, this entry,
another entry, another namespace, an index shift) and a source census. Server lib 811 passed, clippy and the `DOCS_RS=1`
pass clean. Acceptance on the target card, one sitting, four holds 12:55Z to 13:36Z, zero lock retries, 20 of 20 replays
PASS, 32 to 52 C, verbatim: clause 1 `DAY26 CLAUSE 1 arm=on N_runs=100 parked_per_run=[1] restore_submitted_per_run=[0]
not_routed_per_run=[1] runs_with_parked_1_submitted_0_not_routed_1=100 -> PASS`; clause 2 `DAY26 CLAUSE 2 e2e order=o1
on_median=206.8 off_median=115.3 on_minus_off=+91.4 unc=1.2 expected=+15.8 ... -> FAIL` (o2 `+91.4 unc=1.1`); clause 3
`DAY26 CLAUSE 3 stall order=o1 ... on_cell_median=81.8 IQR=0.0 off_cell_median=85.3 on_minus_off=-3.4 unc=0.1 -> isolated
| against day 25's 149.4 (IQR 0.1): -67.6 -> FINDING` (o2 identical); clause 4 `DAY26 CLAUSE 4 restore-arm arm=on
replay=PASS N_runs=100 errors=0 parked_per_run=[1] submitted_per_run=[1] landed_per_run=[1] not_routed_per_run=[0]
runs_parked_1_submitted_1_landed_1=100 cached_tokens=[5088] -> PASS`; clause 5 hit OFF `SPEC-ON-CACHE-HIT GATE: ALL GREEN
(qwen)` (61 ok), ON `ALL GREEN (qwen)` (68 ok), `DAY26 CLAUSE 5 hitgate-on counts against day 24: spec_on_census_equal=True
spec_off_census_equal=True route_submissions=30 ... spec_boundary_captures_with_draft_plane=11 ... not_routed_lines=0 ->
PASS`; identity x4 `ALL GREEN (teeth=0)`, failure x2 `ALL GREEN`, fault `ALL GREEN` (93 ok; the promote cells read
`restore_submitted=0 not_routed=1`), twin x2 `PASS`, unit 8 + 5; the identity ON route lines moved exactly as
pre-registered. Local RTX 5090 (A): OFF 61 ok, ON 68 ok, the ON census identical to day 24. Findings: clause 2 fails on
day 25's premise, not on the mechanism: `advance_sample_emit` samples a session's token from the previous step's logits,
so a one-token request primed on tick A gets its token in tick B's host half, after the demote's two hashes (74.8 ms) ran
at tick B's top; the refusal removed tick A's decode plus the slack (14.6 ms measured, 15.3 by arithmetic) and the hashes
remain in the request's path (a scheduler shape outside Move 2, named, not proposed). Clause 3: the tenant's stall fell
149.4 to 81.8, now 3.4 below OFF (isolated); the two stretched ticks read 92.4 and 95.3 (sum 187.8 against day 25's
185.4): the same work no longer stacked, because the park had decided which tick the prime landed on; day 25's "the
second park cost the tenant nothing" is refuted. **Ruling 37 (lead): the code stays.** Clause 2 is re-derived on the
token-emission reading (the expected e2e move is one tick plus the slack, about 15 ms, and it read 14.6); clauses 1, 3,
4 and 5 hold; no gate moved; the tenant's stall is the door's cost the review weighs and it fell by 67.6 ms on this shape.
The 74.8 ms hash tick in a one-token request's path is Move 1 owed item 2 (the hashes off the tick), not a new item.
Budget 3.6 against 4.

**Lead review of A day 26.** Read the refusal (pin identity by key and `id_index`, not by index alone; placed after the
lookup so a miss never reaches it, before the class check and before the route's own pin), the table test including the
index-shift case, the census ordering, and the typed line's single-line shape. No finding.

**Battery (tree `af92791f6`, receipts `integ42-cpu-battery/`, `integ42-serve-smoke-5090/`, `integ42-hit-gate-5090/`).**
All fifteen CPU steps rc=0. Local RTX 5090: `serve-smoke: 0 failed` (gemma4 and Q35 arms SKIP, absent models; another
session's `decode-batch-gate` held 7778 MiB on the card at the smoke's start, listed and not touched), engine `d2d_*` GPU
cells `5 passed` under the lock, the hit gate OFF and ON armed (9B), verbatim: `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`
61 ok OFF, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 68 ok ON, `ok: door arm: 30 route submission(s) across the two boots`,
11 spec-boundary captures, 13 restores, 0 `restore not routed` lines (the hit gate has no promote-then-hit shape, so
A's refusal must not fire here), 0 refused, disabled, trunk-only, dropped or skipped lines. The day-24 census holds on
this card after A day 26.

## integ43 (`lane/spill-integ43-20260922`): A day 27 (the demote tick cost attributed; the three options for the bundle hash) and C day 33 (the 5090 write-combined hash rate: H1 refuted)
Lane tips merged, in order: C `ab7ef9f18` (day 33), A `8bfd3d103` (day 27), on main `22f7489a3` (#647). One conflict,
`HOSTPREFIX-DOOR.md` section D item 6, both lanes edited (A's code answer, C's measurement); merged three-way against the
integ42 tip both had merged and both paragraphs kept, A's first. No engine path moved: the only code is C's opt-in
`--two-step` arm of the `hash-micro` diagnostic bin (day-18 output unchanged without the flag).

**A day 27 (Move 1 owed item 2, read before designing; no engine code).** Section 1's attribution corrects the owed-item
text: Move 1's two receipt hashes (`progress` at the poll, `tier_transfer.rs:1710`; `bind_tier_image`'s per-plane check,
`worker.rs:9344` over `9440-9451`) run over the KV planes only, about 1.9 MB on the 27B's 64-token entry, 0.9 ms each on
the target host, in the card's `PinnedKind::for_device` (cached on the target, write-combined on the 5090); hash 1 sits
inside `from submission to completion` because `copy_ms` is stamped after the whole settle. The 74.8 ms `in -
completion` is ONE SHA-256 pass of `bind_tier_image`'s `StateBundle` checksum over the whole image: about 157 MB of it
the recurrent f32 state in pageable heap (`HostF32::Heap`, `reserve_image` with no arena) that never crossed the contract
and has no receipt partner, plus the pre-submit f32 D2H and the insert. Under OFF `hpx.tier` is `None`, so OFF never
computes it: **the door's demote tick cost is the bundle checksum, not the copy Move 1 moved.** The 5090's 21 to 23 ms
per 54.8 MB is the same pass at that host's heap rate (37.5 ms per 160 MiB) plus a write-combined read of the one to two
percent KV share, which is why it is not 490 ms per pass; section D item 6 answered (the 9B entry's KV byte split is the
one unmeasured term). Options, pre-registered, then measured where no engine code was needed: (a) a helper thread hashes
the heap payloads (recurrent, logits, hidden) during a `Hashing` phase of `PendingDemote`; the owner thread keeps the two
KV-plane hashes; receipt term unchanged (SHA-256, same bytes, same wire); fault arms `hash-helper-gone` and
`hash-never-lands`; removes about 73 ms from the tick on the target card; **recommended**, acceptance gate in `DAY27.md`
section 2 (`in - completion <= 12.0`, stall ON <= OFF + 2.0, e2e `on_minus_off <= +20.0`, every gate ALL GREEN both arms
both cards, two fault cells, a bitwise digest unit cell, no flag). (b) the slice-3 four-lane program for the D2H receipt
term: moves only the KV-plane hashes unless the bundle program changes too (b'), trading cryptographic naming of the
bytes for transfer integrity; the digest micro-cell (`day27-digest-micro/`, both engine programs, heap, 160 MiB, N=5 per
order, two orders, one hold per host), verbatim: target host `sha_ms=77.589 lanes_ms=70.737 lanes_over_sha=0.912`, 5090
host `sha_ms=37.111 lanes_ms=55.078 lanes_over_sha=1.484`; both digests byte-identical for both programs on both hosts;
no digest swap runs at memory speed, (b') is worth about 7 ms of the 74.8. (c) a GPU digest of the pinned bytes over
PCIe: unmeasured (no engine path launches `d2d_receipt_digest` over a host pointer; the leases carry no `DEVICEMAP`);
arithmetic about 6 ms on the copy stream; its stronger form (digest the source planes before the D2H) needs the recurrent
state on the contract first (Move 2 owed item 1). Section 3, the baseline on the day-27 tree (the day-26 code), one
hold, twenty boots, 20 of 20 replays PASS, 33 to 51 C: stall `on_minus_off=-3.4 / -3.5 unc=0.1 isolated` (81.9/81.8
against 85.2/85.3), e2e `+91.3 / +91.3 isolated` (206.6 against 115.4), `demote_in-completion median=74.8`, gaps 95.3
and 92.4; equal to day 26 within 0.1 ms. **Ruling 39 (lead): option (a) is approved for A day 28 as specified**, with the
receipt term unchanged, `Hashing` as a `PendingDemote` state under the same fail-closed discipline, one long-lived helper
per `HostTierContext` joined at shutdown and `host.disable`, the two fault values under the existing
`MEMRA_KV_HOST_FAULT` row (no new `MEMRA_*` name), and a bitwise digest unit cell. Budget 2.5 against 4.

**C day 33 (the 5090 measurement of section D item 6).** Pre-registered hypotheses (H1: the hash reads the write-combined
destination and the micro-cell rate does not apply to its access pattern; H2: it reads a cached copy, staging buffer or
device digest; H3: the destination is not write-combined despite the printed arm), then one collector hold on the 5090
(14:49Z to 14:50Z, one lock refusal, 56 to 58 C, 9.5 to 31.7 W, premise `PINNED-DEFAULT device="NVIDIA GeForce RTX 5090
Laptop GPU" kind=write-combined flags=4` inside the hold), N=5 per kind per order, both orders, at 54,800,000 B and 160
MiB, 41 ok, 0 FAIL, verbatim: `DAY33 HASH-WC VERDICT: 54.8MB cached 11.9 wc 473.8 heap 11.9 wc_copy 224.2 (N=10 each);
160MiB cached 36.6 wc 1450.8 heap 37.2 wc_copy 686.8 (N=10 each); day31 ON in median 48.3 max 60.4 (N=12)
in_minus_completion median 21.6 (N=12); H1-single fits=False H1-twostep fits=False -> H1 refuted (no WC read route fits
the ON demote's wall time; H2 or H3 stands, separated by the code census, not by this cell); pinned=write-combined;
admissible=True`. The write-combined read rate is size-independent (0.116 GB/s) and the memcpy route reads the same
memory at 0.258 GB/s; the fastest single-pass WC hash at the entry size (468.8 ms) is 7.8x the slowest ON demote (60.4),
the fastest two-step (222.0) 3.7x. A's code census (above) resolves H2 against H3: the pass is over heap memory (H2's
form), with the write-combined share the one to two percent of KV bytes. Two stated deviations (a reading-script field
fix after the run with the first output kept verbatim, 40 ok and 1 FAIL on that check alone; the runner relaunched
detached during a sleep, holding nothing), nothing tuned. Budget 2.4 against 3.

**Lead reading, for the door review.** Sections B and E of the door doc and the packet now carry the attribution: under
ON the demote's tick pays a SHA-256 pass over the whole bundle image that OFF never computes; Move 1's copies are off the
tick and its receipt hashes are 1.8 ms. The 32 ms the cell (i) census could not attribute is this pass minus what OFF
spends on the same tick. Option (a) is the arm that removes it; C day 34 carries the reading into the packet.

**C day 34 (merged after the battery; docs only).** `DOOR-DECISION-PACKET.md` re-read against days 26, 27 and 33: section
2 replaces the two-hashes sentence with A's attribution (file:line on `cb9fa5ef3`; the bundle pass 74.8 ms per 159.8 MB
entry on the target card, 21 to 23 ms per 54.8 MB on the 5090, OFF never computes it; Move 1's two receipt hashes about
1.8 ms over 1.9 MB); item 2 attributes the 32 ms cell (i) could not, and states option (a)'s acceptance gate (ruling 39)
without predicting; item 7 drops the resolved items (5, 6, 9) and names what stays (the arena handoff, the DFlash tail
slice, verify digest v3, the 9B's KV byte split, option (a)'s own receipt, a 5090 tenant-stall cell); section 6's
naked-default bullet carries the bundle checksum on the tick until option (a) lands; section 4 gains five rows (A day 26,
the day-27 baseline, both hosts' digest micro-cells, the day-33 verdict). Every added number regenerated from its receipt
(the day-26 and day-25 readers re-run under the CPU quota, digests from `ev/digest-micro.log`, regimes from
`command.gpu.csv`). `HOSTPREFIX-DOOR.md` item 6 RESOLVED day 27 and 33 with both pointers; section E's day-34 paragraph.
Budget 1.7 against 3. #648 (lead, ruling 38): the `MEMRA_KV_PARK_COMPACT` row's value column reads `0 (default OFF),
decide-by: 2026-10-06.`, the dated sentence names WP-B day 28 and the owner's verdict, the PENDING clause names the two
gates still pending and the received park-time copy cost; merged before this integ.

**Battery (tree `d06a9dd8c`, A day 27 + C day 33; receipts `integ43-cpu-battery/`, `integ43-serve-smoke-5090/`,
`integ43-hit-gate-5090/`).** All fifteen CPU steps rc=0. Local RTX 5090: `serve-smoke: 0 failed` (gemma4 and Q35 arms SKIP,
absent models), engine `d2d_*` GPU cells `5 passed` under the lock, the hit gate OFF and ON armed (9B), verbatim:
`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 61 ok OFF, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 68 ok ON, `ok: door arm: 30
route submission(s) across the two boots`, 11 spec-boundary captures, 13 restores, 0 `restore not routed`, 0 refused,
disabled, trunk-only, dropped or skipped lines. C day 34 and #648 merged after the battery are docs only; the engine tree
under test is `d06a9dd8c`'s, equal to this PR's.

## integ44 (`lane/spill-integ44-20260922`): C day 35 (the 5090 tenant-stall cell); A day 28 held on the lane (option (a) landed, clause 2 red)
Lane tip merged: C `48250588c` (day 35) on main `c5879a59e` (#649), clean. **A day 28 is NOT merged**: its code stays on
`lane/spill-a-20260919` until ruling 40's 2a lands (below). This integ is docs and receipts only; the engine tree equals
main's.

**C day 35 (the 5090 tenant-stall cell; this card's own figures).** A day 16's five arms on the 9B (cache 64 MB, host 8192
MB, `MEMRA_SERVE_SPEC=0`, tenant lengthened to 400 tokens so the 5120-token prime lands inside the window, pre-registered
from the board's 7.3 ms tick; A's intruder shapes byte for byte through a harness copy, diff banked), six boots in one
collector hold on `/tmp/memra-5090.lock` (16:18Z to 16:29Z after a bounded wait behind another session's gates), two
passes in opposite order, N=5 per arm per order, 58 to 89 C, 30.6 to 175.3 W, 10 of 10 `STALL REPLAY: PASS`, verbatim:
`DAY35 STALL VERDICT: prime pass1 10.5 (iqr 174.2) inadmissible; demote pass1 off 63.7 on 67.0 on-off +3.3 (unc 3.3)
under_resolution; promote pass1 off 47.7 on 49.3 on-off +1.6 (unc 12.5) under_resolution; prime pass2 278.4 (iqr 6.6)
admissible; demote pass2 off 63.0 on 63.2 on-off +0.3 (unc 2.7) under_resolution; promote pass2 off 47.5 on 50.1 on-off
+2.6 (unc 2.7) under_resolution; admissible=False`. `admissible=False` is the pass-1 prime arm alone: an unidentified
co-tenant (card-wide `memory.used` 20.3 to 23.2 GB against 7.6 to 9.8 GB in the identical pass-2 boots, gone at
16:22:57Z, not holding the lock) made the memory admission refuse 7 of 10 pass-1 prime intruders (`[admit-oom] capacity
reject: ... does not fit an IDLE box (available 1794MB), HTTP 400`); pass 2 is clean and both passes give the same
classes. All 160 tenant runs of the eight demote and promote receipts: errors 0, tenant text sha `5d59f3ddef257cfb`, every intruder inside the window,
the ON boots' lines complete (21 demote triples, 10 promote triples, 10 `restore not routed (contracts door)`, 0 `restore
submitted`, 43 of 43 `require=ok`, zero refused or latched). Attribution on this card: ON demote `in - completion` median
21.6 (pass 2, N=9) / 23.8 (pass 1), about 1.8 to 2.0 of day 33's 11.9 ms heap pass at the entry size; post hoc and
labelled (`gaps-posthoc.log`): both demote arms stretch two ticks, the worst tick is the same in both arms (about 70 ms,
the intruder's prime plus insert) and the door's share sits on the SECOND tick (ON 69.4 / 70.9 against OFF 41.9 / 41.3),
so the day-16 worst-tick rule under-reads the door's share on this card (a property of the metric; the verdict lines
stand). Records: the per-card cost table B-5090 in `HOSTPREFIX-DOOR.md`, the packet's 5090 table and item 7. Owed and
stated: a prime arm the admission admits in every run, the second-tick split, the 9B KV byte split. Budget 1.3 against 4.

**A day 28 (option (a), ruling 39), held on the lane.** Landed on `lane/spill-a-20260919` (`45f824a75`, `feat:`):
`HostHashWorker`, one long-lived helper per `HostTierContext`; `PendingDemote` gains a `Hashing` phase and an owner-thread
ledger; every settle path meets `Hashing` through `host_demote_settle_with_deadline` before the contract step, a `Block`
settle continuing into the hash wait and naming what it waits on; `bind_tier_image` consumes the handed-in digests after
a byte-count check (same `checksum`, same bytes); `hash-helper-gone` / `hash-never-lands` on the existing
`MEMRA_KV_HOST_FAULT` row (10 s wall deadline); `disable` and shutdown join the helper; two fault-gate cells; eight CPU
cells including the bitwise digest cell and the source census; server lib 830 passed, every CPU check clean. Target card,
one hold, twenty boots, 20 of 20 `STALL REPLAY: PASS`, 33 to 52 C, verbatim: `DAY28 CLAUSE 1a stall order=o1 ...
on_cell_median=81.8 off_cell_median=85.2 rule on<=off+2.0 -> PASS` (o2 `81.9 / 85.2 -> PASS`); `DAY28 CLAUSE 1b e2e
order=o1 ... on=132.3 off=115.4 on_minus_off=+16.9 rule <=+20.0 -> PASS` (o2 `+16.9 -> PASS`; day 27 read +91.3); `DAY28
CLAUSE 1c owner in-completion N=100 median=7.40 min=7.17 max=45.01 runs_with_demote_without_ledger=0 rule <=12.0 -> PASS`
(day 27: 74.8); `DAY28 VERDICT clauses_failed=0 -> ALL PASS`; the helper 73.2 ms per 157.9 MB (98 payloads), 100 of 100
landings at the tick-top poll, the tenant's second gap 19.0 (was 92.4). Clause 3 PASS (`hash-helper-gone` 14 ok,
`hash-never-lands` 14 ok, both cards); clause 4 PASS (`hash_helper_digests_equal_the_owner_thread_digests_bitwise ... ok`
on both hosts; box GPU cells 8 and 5 passed); clause 5 PASS (no flag, no new name). **Clause 2 FAILS** on both cards in
the default (spec) arm: `identity-default-on` `KV-HOST-SPILL IDENTITY GATE: 4 FAILURE(S) (teeth=0)`; `failure-on`
`KV-HOST-SPILL FAILURE GATE: 1 FAILURE(S)` (the `digest` cell); `contract-fault` `KV-HOST-CONTRACT-FAULT GATE: 23
FAILURE(S)` (the four promote cells; the two demote cells' `the next demote publishes`); 5090 `fault-default` `23
FAILURE(S)` (same shape), `fault-plain` `ALL GREEN` (121 ok); green: identity x3, `failure-off`, twins, hit OFF and ON
with the day-24 census on both cards, `d2d-capture` 12 ok, `d2d-restore` 14 ok. One cause for every red cell, not
tuned: a hit that arrives inside the `Hashing` window (73 ms plus a tick) is a MISS, and the default arm's spec-boundary
insert puts the next request there (on the day-26 tree the bind ran inside the completing tick top, before admission).
Also found: the demote's remaining owner-thread cost is the pre-submit segment (32 pinned allocs plus the 100 f32 D2H
copies), about 6 ms steady, 39 to 45 ms at first touch, 149 ms under the verify arm (Move 2 owed item 1 is now priced per
demote by the ledger line); the printed `pre-submit` spans both sides of the `submitted` stamp, so the baseline's
`completion` contained the f32 D2H and the comparison is conservative. Budget 4.9 against 5. **Ruling 40 (lead):** A's
2a is approved for A day 29: a hit whose entry is `Hashing` PARKS the request one tick (the `Promoting` and `Restoring`
pattern: requeue, typed line, tick-top poll lands the hash and publishes, re-admission to the device hit), never a
refusal and never a synchronous settle at the probe; the parked-only bounded wait and the orphan grace gain their arm;
every Block settle settles the hash first; the fault gate's demote cells await the boot's last publication before `stop`.
Acceptance: clause 2 green in every arm on both cards, clauses 1a to 1c re-read and passing, a typed count of hits parked
on a `Hashing` entry. Until then the day-28 code is not the door's serving path and is not integrated (A's own
statement, adopted).

**Battery (tree `f5bf3311d`, the engine tree equal to main's; receipts `integ44-cpu-battery/`, `integ44-serve-smoke-5090/`,
`integ44-hit-gate-5090/`).** All fifteen CPU steps rc=0. Local RTX 5090: `serve-smoke: 0 failed` (gemma4 and Q35 arms SKIP,
absent models), engine `d2d_*` GPU cells `5 passed` under the lock, the hit gate OFF and ON armed (9B), verbatim:
`SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 61 ok OFF, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 68 ok ON, `ok: door arm:
30 route submission(s) across the two boots`, 11 spec-boundary captures, 13 restores, 0 `restore not routed`, 0 refused,
disabled, trunk-only, dropped or skipped lines: main's own results, re-read.

## integ45 (`lane/spill-integ45-20260922`): A days 28 and 29 (option (a): the bundle checksum off the tick; a hit on a `Hashing` entry parks one tick)
Lane tip merged: A `1cce45646` (days 28 and 29) on main `ca5a90e2a` (#651), clean. Engine change under the door:
`crates/memra-server/src/worker.rs` (+1558 lines including tests), `tools/kv-host-contract-fault-gate.sh` (two hash cells,
the demote cells' last-publication wait), `docs/FLAGS.md` (the two fault values on the existing `MEMRA_KV_HOST_FAULT`
row). No new `MEMRA_*` name, no flag, no `unsafe`, no engine or tier crate change.

**A day 28 (ruling 39), as recorded in integ44 and now integrated with day 29.** `HostHashWorker`, one long-lived helper
thread per `HostTierContext` (`memra-host-hash`), fed over a channel, joined at `host.disable` and at shutdown (the run
loop's last statement, census-pinned); `PendingDemote` gains a `Hashing` phase (`PendingHashing`: ticket seq, payload
count and bytes, the hand-off instant, the parked request ids, re-park count) and an owner-thread ledger
(`DemoteOwnerLedger`: pre-submit, copy settle, hash polls); after the copy lands the heap payloads (the recurrent f32
planes, the boundary logits, the hidden) are moved out of the image and handed to the helper; every settle path meets
`Hashing` through `host_demote_settle_with_deadline` before the contract step, a `Block` settle continuing into the hash
wait with a typed line naming what it waits on (no CUDA event and no join: the KV planes settled and the ticket retired
before the hand-off); the reply is checked by ticket seq, payload count and per-payload byte count, each payload restored
into its emptied slot, and `bind_tier_image` consumes the handed-in digests after a byte-count check (the same
`checksum`, the same bytes, the receipt term unchanged); `hash-helper-gone` and `hash-never-lands` (10 s wall deadline)
latch with the parked requests named; the publish line carries the ledger (`the owner thread held X ms across the demote:
pre-submit, copy settle over N polls, hashing polls, take-back bind and publish; owner in-completion Y ms`). Eight CPU
cells including the bitwise digest cell (`hash_helper_digests_equal_the_owner_thread_digests_bitwise`) and the source
census; server lib 830 passed. Cost clauses on the target card (twenty boots, 20 of 20 `STALL REPLAY: PASS`), verbatim:
`DAY28 CLAUSE 1a ... on_cell_median=81.8 off_cell_median=85.2 rule on<=off+2.0 -> PASS`; `DAY28 CLAUSE 1b e2e ...
on=132.3 off=115.4 on_minus_off=+16.9 rule <=+20.0 -> PASS` (day 27: +91.3); `DAY28 CLAUSE 1c owner in-completion N=100
median=7.40 ... rule <=12.0 -> PASS` (day 27: 74.8); the helper 73.2 ms per 157.9 MB, 100 of 100 landings at the tick-top
poll. Clause 2 was red (a hit inside the `Hashing` window was a miss), fixed by day 29.

**A day 29 (ruling 40, option 2a).** `host_hashing_hit` (pure over the host state: the `Demoting` entry with `hashing`
`Some`, same pool key, the host `lookup` rules, deeper than the device hit) and `host_hashing_park` (the request id
recorded once on `PendingHashing.parked`, one typed line `hit parked on a Hashing entry: request R (P tokens) hits the
Demoting entry's N tokens (ticket seq=S, K payloads, M MB on the hash helper for T ms); the request waits one tick for the
digests`, re-parks counted), called in the admission probe BEFORE the promote decision; the parked-only wait's guard gains
the `Hashing` arm (`hpx.demoting ... hashing.is_some()`); the ledger line ends `; H hit(s) parked on the Hashing entry (R
re-park(s))`; the latch tail names the parked requests (they re-admit to a cold prime). The fault gate's demote cells wait
for the boot's last publication before `stop` (`the boot's last publication landed before stop (bounded 15 s wait)`; two
wrong forms of that wait caught on the 5090 before the box ran, recorded). Server lib 826 passed; census and parked-wait
tests extended. Target card, one sitting 17:37Z to 18:10Z, zero lock retries, verbatim: identity default and plain OFF and
ON `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` x4 (12 ok each); failure OFF and ON `KV-HOST-SPILL FAILURE GATE: ALL
GREEN` x2 (15 ok, the `digest` cell in); contract fault `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 ok, 0 FAIL, ten
cells including the two hash cells); twin OFF and ON `-> PASS`; hit OFF and ON `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`
61 / 68 ok with the day-24 census (`12/12/13/13`, `2/2/3/3`, `30 route submission(s)`); unit 8 passed, engine `d2d_*` 5
passed, hash, census and parked-wait cells 10 passed. Local RTX 5090 (A): `identity-default-on` `ALL GREEN (teeth=0)`;
`fault-default` and `fault-plain` `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 ok each); hit OFF and ON `ALL GREEN
(qwen)` 61 / 68 ok, same census. Clauses 1a to 1c re-read on the day-29 tree: `81.7 / 85.4 -> PASS` and `81.8 / 85.1 ->
PASS` (`-3.6 / -3.3 unc=0.1 isolated`), `+16.8 -> PASS` / `+16.9 -> PASS`, `median=7.39 ... -> PASS`, `DAY28 VERDICT
clauses_failed=0 -> ALL PASS`, unchanged from day 28 within 0.1 ms. The parked-on-Hashing count in the identity gate's
default ON arm: ONE per boot on both cards (target: `... (ticket seq=3, 98 payloads, 157.9MB on the hash helper for
0.3ms)`, ledger `1 hit(s) parked on the Hashing entry (35 re-park(s))`; 5090: `53.7MB ... for 0.4ms`, `5 re-park(s)`);
zero hits on a copy-phase `Demoting` entry. **Integrable: yes** (A's statement, adopted after the lead's review). Open
question named by A, not owed work: a hit inside the copy window (before `Hashing`) keeps the day-17 cold-prime rule; zero
observed. A's day-28 door-table rows, lost to two integ-side conflict resolutions, re-added on day 29 (section B). Budget
4.9 + 3.7 agent-hours against 5 + 5.

**Lead review of A days 28 and 29.** Read the helper's lifecycle (spawned with the tier context, one thread, the sender
dropped and the thread joined at the tier's latch and at shutdown, a panic reported on join), the hand-off (payloads moved
out of the pending image, never copied; restored only into their emptied slots; a reply checked by seq, count and byte
count before any digest is consumed; `bind_tier_image` re-checks the byte count against the payload it names), the
deadline arm (a `Poll` that has waited past 10 s or a `Block` that finds no reply latches with the parked requests named;
a late reply for a latched ticket can only reach a disabled tier), the park (a hit names the `Hashing` entry only when the
host lookup rules match deeper than the device hit; re-parks counted; bounded by the deadline, then the cold prime), and
the parked-only wait guard. One numeric program per request holds by construction: bytes are hashed, nothing is
generated, and a parked request is served by the device hit either way. No finding. **Ruling 41:** the day-28 and day-29
code is the door's serving path; Move 1 owed item 2 (the bundle hash off the tick) closes on these receipts; item 2a
closes; the remaining owner-thread cost of a demote is the pre-submit segment (Move 2 owed item 1, A day 30 running).

**Battery (tree `1c540e050`, A's engine tree on main's; receipts `integ45-cpu-battery/`, `integ45-serve-smoke-5090/`,
`integ45-hit-gate-5090/`, `integ45-fault-gate-5090/`).** Fourteen of fifteen CPU steps rc=0; the fifteenth, `diff-check`,
rc=2 on the battery's own receipt bytes (`SUMMARY.txt:13: trailing whitespace`, `test-server.log:1009: new blank line at
EOF`) before the receipt dirs carried `.gitattributes`; with them, `git diff --check origin/main HEAD` is clean on the
committed tree. Local RTX 5090: `serve-smoke: 0 failed` (spec, gemma4 and Q35 arms SKIP, absent models), engine `d2d_*`
GPU cells `5 passed` under the lock, the hit gate OFF and ON armed (9B), verbatim: `SPEC-ON-CACHE-HIT GATE: ALL GREEN
(qwen)` 61 ok OFF, `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 68 ok ON, `ok: door arm: 30 route submission(s) across the
two boots`, 11 spec-boundary captures, 13 restores, 0 `restore not routed`, 0 hits parked on a `Hashing` entry (the park
lives in the identity gate's shape, not this one), and one line matching `refused|DISABLED|trunk-only|dropped|skipped`:
the route-contract registration `refused=[]` (#650's boot line, an empty list). The fault gate on this card, default and
plain arms: `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` 123 ok each, both hash cells (`hash-helper-gone`,
`hash-never-lands`) present.

**Revuto round 1 on #652 (reviewed by claude-cli-opus-5): one finding, real, fixed in `c26255bc7`.** `disable` joined the
hash helper with an unbounded `handle.join()`. Dropping the job sender does not interrupt a job already being hashed, so a
latch that fires because the digests are late (the deadline's own cause: a starved or wedged helper) blocked the owner
thread for the rest of that job, and so did every other `disable` caller (the capture and restore latches, budget
exhaustion) with a hash in flight. Neither hash fault cell reached it: `hash-never-lands` discards its reply and returns to
its job channel, so the latch found an idle helper. Fix: `HostHashWorker::close(why, bound)` drops both channel ends (the
job in hand, if any, ends with a reply that finds no receiver and drops with its payloads, so a late reply can no longer
reach a latched tier) and joins within `bound`, else detaches with a typed line (`hash helper detached (<why>): still
inside a job after N ms; ...`). The latch passes `HOST_HASH_LATCH_JOIN` = 50 ms (an idle helper exits in microseconds);
shutdown passes `HOST_HASH_DEADLINE`. New cell `a_latch_with_the_helper_still_inside_a_job_detaches_it_within_the_bound`:
a stand-in helper holds its job past the latch; the latch returns after the bound and under 2 s, the helper is detached,
and the released job's reply send fails. On the old `join` the cell hangs (the release comes only after `disable`
returns). The census pins both `close` calls. Checks on `c26255bc7`: `cargo fmt --all -- --check` clean, clippy
`-D warnings` on memra-server all targets clean, memra-server suite `835 passed; 0 failed; 14 ignored`. Local RTX 5090
fault gate, default arm, on this tree (receipt `integ45-r1-fault-gate-5090/`): `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`
123 ok, both hash cells present, `hash helper joined (the tier latched off)` once each in the `hash-helper-gone`,
`hash-never-lands`, `d2d-capture` and `d2d-restore` server logs, zero `hash helper detached` lines (the gate's
count-of-one check on the join line stands unchanged). The same-day owner call moved revuto's review model from Opus 5 to
Opus 5.5 (2026-09-22), so round 2 is reviewed on Opus 5.5.

## integ46 (`lane/spill-integ46-20260922`): C days 36 and 37 (the door packet re-read with option (a) landed; the 5090 demote-class tenant-stall cell on the option (a) code)
Lane tip merged: C `cb4bc1a87` (days 36 and 37) on main `0c86309bd` (#652), clean. Docs and receipts only: outside
`research/spill-c-20260919/` the diff against main is the two `research/INDEX.md` rows; the engine tree equals main's.
A day 30 (Move 2 owed item 1, the recurrent f32 state onto the copy stream) is running on BOX3 and is not in this integ.

**C day 36 (no card).** The door packet (`C/DOOR-DECISION-PACKET.md`) re-read against A days 28 and 29 and ruling 41:
item 2 carries the twelve `DAY28` clause and verdict lines verbatim and clause 2 per gate per card; section 2 names the
demote's remaining owner-thread cost, the pre-submit segment (`median=6.07` on the 4th to 11th demote of a boot, N=80;
`42.19` on the first three, N=30; target card), with the helper's `hashed in` `median=73.20` ms per 157.9 MB off the tick;
sections 3 and 4 gain the day-28, day-29 and integ45 RTX 5090 rows; item 7 drops option (a) and names Move 2 owed item 1.
The counting rule (`C/DAY36.md` section 1): every figure added maps to a command in the packet's appendix A or in DAY36,
each output under `day36-cpu/` with its command on the first line. The stopped first run's resync is DAY36 section 0.

**C day 37 (the RTX 5090 class).** Day 35's cell unchanged, run with two binaries in ONE hold on `/tmp/memra-5090.lock`:
`091a931c0` (the day-35 tree) and `8b889dcdf` (the option (a) tree), programs p1-base p2-opta p3-opta p4-base, 24
boots, 40 of 40 `REPLAY: PASS`, 19:47:19Z to 20:31:34Z, 51 to 89 C. The rule and the secondary quantity were
pre-registered in `a50922b27` (committed 19:46:56Z, before the first boot). Verbatim (`C/rtx5090-day37/reading.log`):
`DAY37 VERDICT q=stall: prime o1=+27.9/51.4 o2=-6.4/15.0 under_resolution; demote-off o1=+1.3/4.1 o2=+2.6/5.5
under_resolution; demote-on o1=-1.0/4.2 o2=-3.8/4.7 under_resolution; promote-off o1=+0.5/7.6 o2=-1.9/3.4
under_resolution; promote-on o1=-1.4/4.4 o2=-1.3/7.4 under_resolution; did-demote o1=-2.3/5.8 o2=-6.4/7.3
under_resolution; did-promote o1=-1.8/8.8 o2=+0.7/8.1 under_resolution` and `DAY37 VERDICT q=top1_plus_top2: prime
o1=+45.6/97.7 o2=-11.6/24.4 under_resolution; demote-off o1=+2.5/6.4 o2=+4.5/9.4 under_resolution; demote-on
o1=-21.7/5.2 o2=-24.5/8.5 moved; promote-off o1=+0.9/11.3 o2=-2.4/3.3 under_resolution; promote-on o1=-23.1/8.7
o2=-21.4/9.2 moved; did-demote o1=-24.2/8.2 o2=-29.0/12.7 moved; did-promote o1=-24.0/14.2 o2=-19.1/9.8 moved`. By the
worst-tick rule the two binaries do not separate; by the sum of the tenant's two largest gaps the ON arms and both DiDs
moved by about 20 to 29 ms with the OFF and prime controls under resolution, the direction option (a)'s mechanism
predicts (the bind pass leaves the tick). Which of the two ticks moved is not read by this cell (C day 38, a
pre-registered post-hoc reader over the same receipts, running). The target card's demote class on the option (a) tree
stays unmeasured. The engine's `[gpu-watch] Xid source:` startup line carries an em dash, so each raw `server.log` holds
one; the logs are banked as written (`C/DAY37.md` section 6).

**Checks on this tree** (`integration-day12/integ46-checks/checks.log`, tree `dbb67ac12`): `check-flags` rc=0,
`check-conflict-markers` rc=0, `git diff --check origin/main HEAD` rc=0, `public-boundary: 604 matches (604
grandfathered, 0 new).`, `update-perf-board.py --check` rc=0, zero em dashes in the added Markdown lines. No engine source
in the range, so no GPU battery and no release tag.

**Owner decisions flagged.** `MEMRA_ADMIT_BY_MEMORY` reaches its decide-by tomorrow, 2026-09-23 (packet
`C/ADMIT-BY-MEMORY-DECISION-PACKET.md`, on main since integ42; it recommends nothing). The others are unchanged: 2026-10-04 (MoE slot cache, VMM), 2026-10-05 (the
contracts door), 2026-10-06 (the park door).

## integ47 (`lane/spill-integ47-20260923`): A day 30 (the recurrent f32 state rides the demote's copy stream, the D2H half of Move 2 owed item 1) and C day 38 (the per-tick split of day 37's cell; the 9B entry's byte split)
Lane tips merged: A `0e555c1f7` (day 30) on main `9717e8d57` (#653) as `539936c4c`, then C `18c900fc7` (day 38) as
`160929a92`, both clean. Engine change under the door: `crates/memra-engine/src/pinned_host.rs` (+59),
`crates/memra-engine/src/tier_transfer.rs` (+447), `crates/memra-server/src/worker.rs` (+623),
`crates/memra-tier/src/conformance/d2h_span.rs` (+153, new) and its CPU binding `tests/contracts/d2h_span_bindings.rs`
(+190, new). No new `MEMRA_*` name, no flag, `WIRE_VERSION` stays 1. New `unsafe`: the cached pinned allocation, the
`cuMemcpyDtoHAsync` enqueue and `mark_landed` in `pinned_host.rs`, their two call sites and one byte view in
`tier_transfer.rs`. C day 38 is receipts and readers only.

**A day 30 (Move 2 owed item 1, the D2H half).** The demote's recurrent conv and ssm f32 planes leave the owner thread:
each non-empty plane becomes a typed `D2hSpan` on the KV D2H ticket (`CudaTransfers::submit_d2h_spans`: admission
first, the owner-stream fence and the copy-stream wait, one enqueue and one event per span; an error from the first
enqueue on sets the ticket `unknown`), into a per-context pool of cached pinned staging buffers
(`PinnedHostBuf::new_unwritten`, unreadable until landed). The settle takes the spans after the landing and before the
retire (`retire` is `Busy` while spans are untaken), puts each source plane back into `dead`, and the driver attaches
each staging buffer to its empty heap placeholder; the hash helper copies the landed staging into the payload's heap
`Vec` and hashes that with the same `host_hash_payload_digest` program, and the landed path returns the staging to the
pool. A span refusal hands every span back, returns the sources and the staging and unwinds the ticket through
`host_contract_abort` (`tier D2H spans refused: <error> (<n> f32 spans handed back)`). Tier conformance rules 1 to 5
with three schedules (the red arm `d2h_span_taken_on_the_items_landing_fails`), the CPU census and two native cells, all
committed before the run. Target card (BOX3, one sitting on `a8d6b1df5`), verbatim: `DAY30 A2 pre-submit steady N=80
median=0.62 min=0.58 max=1.10 boots_on=10 demotes_per_boot=[11] rule N>=80 median<=1.5 max<=3.0 -> PASS` (the same
reader on day 29's receipts: `median=6.05 min=5.92 max=6.61 ... -> FAIL`); `DAY30 A3 logs=92 copy_complete_lines=132
receipts_paired=132 receipts_without_copy_line(on-tick)=0 bad=0 rule items==KV+spans, spans=96, KV in [32, 34], receipt
items==KV and suffix==spans -> PASS`; `DAY28 CLAUSE 1c owner in-completion N=100 median=1.96 min=1.83 max=3.03
runs_with_demote_without_ledger=0 rule <=12.0 -> PASS` (day 29: 7.39); clause 1a `76.9` against `85.2` and `85.5`,
clause 1b `+12.0` and `+11.7` (day 29: +16.8 and +16.9); `DAY28 VERDICT clauses_failed=0 -> ALL PASS`; A1 to A5 PASS.
Gates on the same tree: identity x4 `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok each), failure x2 `ALL
GREEN` (15 ok), contract fault `ALL GREEN` (123 ok), twin OFF and ON `-> PASS`, hit OFF and ON `ALL GREEN (qwen)` 61 /
68 ok with the day-24 census, unit cells 10 + 6 + 11 + 3 passed, double-park 20 of 20. The helper reads `79.8` ms steady
(day 29: 73.1): it now pays the heap `Vec` first touch (about +26 ms on demotes 1 to 3 of a boot) that DAY28 had
attributed to the owner thread, off the tick. Local RTX 5090 half on the merged tree `2a83224ab`: ALL GREEN. Findings
(A's section 8): the steady pre-submit segment is the staging allocation now; a post-take abort (the injected
`contract-postpublish` refusal) frees the landed staging instead of pooling it, so the boot's next demote re-allocates
(`pre-submit 28.71`), fail-closed and owed; the verify arm (`MEMRA_KV_HOST_VERIFY`, a diagnostic) keeps its owner-stream
digest. **Integrable: yes, as the complete D2H half** (A's statement, adopted after the lead's review). Owed on Move 2
item 1: the H2D half (the promote's recurrent `engine.htod`), the D2D half (capture and restore), the governor charge of
the staging (157.9 MB per context on the 27B, outside the tier governor's pinned ledger today), the strong-form receipt,
a serving-shape span-refusal cell in the fault gate, and the staging return on the post-take abort. Budget about 2.5 of 5
agent-hours.

**C day 38 (no card; a pre-registered post-hoc reader over day 37's 40 receipts).** The tick rule was committed before
the reader saw a day-37 per-run gap position; day 37's admissibility line is required, nothing re-admitted. Verbatim
(`C/day38-cpu/tick-split.log`): `DAY38 VERDICT q=tick1: ... demote-on o1=-1.0/4.2 o2=-3.9/5.4 under_resolution; ...
did-demote o1=-2.3/5.9 o2=-6.5/7.7 under_resolution; ...` (every arm `under_resolution`); `DAY38 VERDICT q=tick2: prime
o1=+21.4/38.9 o2=-4.4/17.4 under_resolution; demote-off o1=+1.4/2.4 o2=+1.1/2.8 under_resolution; demote-on
o1=-20.4/2.2 o2=-21.0/2.5 moved; promote-off o1=nd o2=nd not_defined; promote-on o1=nd o2=nd not_defined; did-demote
o1=-21.7/3.2 o2=-22.0/3.8 moved; did-promote o1=nd o2=nd not_defined`; `DAY38 HYPOTHESIS P1 ... P5 ... -> consistent
with H`. So the demote-class difference between the two binaries in day 37's summed pair sits on tick 2, the poll tick.
Within each binary, demote ON minus OFF on tick 2: `+29.7 unc=3.8 -> isolated` (base) and `+7.0 unc=3.5 -> isolated`
(option (a)); the remaining `+7.0` matches the size of the ledger's `copy_settle` (8.48 / 8.30), stated, not attributed.
Every demote arm stretches exactly ticks 1 and 2 after the fire (20 of 20 per program), so day 37's `top1_plus_top2` is
exactly tick 1 plus tick 2 there; promote-on stretches one tick on option (a), so its tick 2 is `not_defined`. The 9B
byte split (packet item 7, `C/day38-cpu/byte-split.log`): the 54.6 MB host image is KV planes 950,272 B exactly
(= 64 x 14,848, the admission plain coefficient), token ids 256 B, logits in (950,272, 1,099,744) B, and conv + ssm +
hidden in [52,599,728, 52,699,728) B; no banked line separates conv, ssm and hidden, and DAY38 names the three engine
lines that would. No recommendation, no engine change.

**Lead review of A day 30.** Read the staging lifecycle (taken by exact length or allocated unwritten; unreadable until
`mark_landed`; returned on the landed path and on a refusal before submission; dropped, so freed, on a post-take abort;
cleared at `disable` after the hash helper closes; the set grows only by distinct plane sizes and one demote is in
flight at a time), the ticket rules (a span refusal before the first enqueue is whole; an enqueue error from the first
on quarantines the ticket; the take refuses `NotReady` before the landing; `retire` is `Busy` while spans are untaken;
an unretired entry's drop forgets its spans), the settle order (spans taken before destinations, retire and require;
a `SourceQuarantined` return after the take drops the staging), and the driver (a staging buffer with no heap
placeholder latches with `has no heap placeholder`). The contract route stays guarded off the arena and off Glm. One
numeric program per request holds: the spans are a byte copy of the same planes and the helper hashes the same bytes
with the same program (the bitwise GPU cell `option_b_spans_ride_the_ticket_and_land_bitwise`). No finding beyond A's
own owed list. Noted, not owed today: `PinnedHostBuf`'s drop does not bind a CUDA context (as before this change).
**Ruling 42:** the day-30 D2H spans are the door's serving path; the D2H half of Move 2 owed item 1 closes on these
receipts; the H2D half, the D2D half, the governor charge of the staging, the strong-form receipt, the span-refusal
fault cell and the staging return on a post-take abort remain owed (A day 31 running). C day 38 is read as registered:
the option (a) move is on the poll tick.

**Battery (tree `160929a92`; receipts `integration-day12/integ47-cpu-battery/` and `integration-day12/integ47-5090/`).**
Fifteen of fifteen CPU steps rc=0 (fmt, portable suites `357 passed, 0 skipped (budget 0), 0 failed`, memra-server
suite `836 passed; 0 failed; 16 ignored`, cross-target `DOCS_RS=1` clippy, `check-flags`, publish census, docs registry,
collector pytest `87 passed`, engine CPU lib `532 passed; 0 failed; 31 ignored`, tier suite with the contracts binding
`90 passed`, engine/server/tier clippy `-D warnings`, marker census, workflow keys, perf board, `git diff --check
origin/main HEAD`). Local RTX 5090, one collector hold on `/tmp/memra-5090.lock` (`run-held.sh`, 21:35:47Z to
21:39:56Z, zero lock retries, the card idle before the first cell), verbatim from `battery.log`: `serve-smoke: 0 failed`
(31 ok; spec, gemma4 and Q35 arms SKIP, absent draft and models); engine `d2d_*` and `d2h_span` GPU cells `6 passed`;
the worker's `option_b_span*` cells `2 passed`; identity default ON `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`
(12 ok; two copy-complete lines `items=66 (18 KV, 48 f32 spans)`; one hit parked on a `Hashing` entry); contract fault
default and plain `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (123 ok each, 16 span copy-complete lines each); hit OFF and
ON `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 61 / 68 ok, the day-24 census (`capture_submitted=12
capture_published=12 restore_submitted=13 restore_landed=13`, `30 route submission(s)`). The identity arm's verify
`pre-submit` reads 51.51 and 33.18 (A's 5090 day 30: 50.28, 37.53; finding 5, the diagnostic arm). BOX3 not rerun on
the merged tree: A's target-card receipts are on `a8d6b1df5`, before the lane merged #652's bounded latch close
(`c26255bc7`, one resolved hunk in `disable`, A's section 11); the merged tree's engine equals A's tip `0e555c1f7`, which
the 5090 carried ALL GREEN on `2a83224ab` and again here. The
engine's `[gpu-watch]` startup line and the publish census's `OK` line carry an em dash, so each raw `server.log` and
the battery `SUMMARY.txt` hold one; banked as written.

**Owner decisions flagged.** `MEMRA_ADMIT_BY_MEMORY` reaches its decide-by today, 2026-09-23 (packet
`C/ADMIT-BY-MEMORY-DECISION-PACKET.md`; it recommends nothing). The others are unchanged: 2026-10-04 (MoE slot cache,
VMM), 2026-10-05 (the contracts door), 2026-10-06 (the park door).

## integ48 (`lane/spill-integ48-20260923`): C day 39 (the target card's demote-class tenant-stall cell on the option (a) code, three binaries in one hold)
Lane tip merged: C `a7608c268` (day 39) on main `5f1b0eda4` (#656) as `fa02b6559`, clean (one auto-merged hunk in
`research/INDEX.md`, C's row). Records and receipts only: `git diff origin/main HEAD -- crates tools` is empty, and C's
tip carries no crate diff against B2 (`160929a92`). No flag, no engine line, no `MEMRA_*` name.

**C day 39 (one RTX PRO 6000 Blackwell, collector rig `pro-single`, `/tmp/memra-gpu.lock`).** Day 35's five arms
(prime, demote OFF and ON, promote OFF and ON) on the 27B in A day 16's target-card shape, the harness byte-for-byte
(SHA-256 `9c9b3878...0ccc`), three binaries built on the box from fixed SHAs: B0 `091a931c0` (before option (a)), B1
`9717e8d57` (option (a), 2a and the bounded latch close, no spans), B2 `160929a92` (B1 plus A day 30's D2H spans). One
hold, programs p1-b0 p2-b1 p3-b2 p4-b2 p5-b1 p6-b0, 36 boots in 56.0 minutes, no wait and no refusal, 60 receipts. The
pre-registration `89c2cd521` was committed at 21:57:26Z; the first mark is 21:58:02Z. Verbatim
(`C/pro-single-day39/reading.log`):

```
DAY39 VERDICT contrast=b1-b0 q=stall: prime o1=-0.1/2.3 o2=+0.0/2.3 under_resolution; demote-off o1=-0.6/2.0 o2=-0.5/1.5 under_resolution; demote-on o1=-31.8/0.8 o2=-31.1/1.2 moved; promote-off o1=-0.3/4.0 o2=-0.3/1.4 under_resolution; promote-on o1=-0.1/0.9 o2=-0.0/0.9 under_resolution; did-demote o1=-31.2/2.1 o2=-30.6/2.0 moved; did-promote o1=+0.3/4.0 o2=+0.3/1.7 under_resolution
DAY39 VERDICT contrast=b2-b1 q=stall: prime o1=+0.0/2.4 o2=-0.0/2.3 under_resolution; demote-off o1=+0.3/1.4 o2=+0.3/1.4 under_resolution; demote-on o1=-41.2/0.7 o2=-41.8/1.2 moved; promote-off o1=-0.0/1.9 o2=+0.2/1.7 under_resolution; promote-on o1=-5.0/0.8 o2=-5.1/0.8 moved; did-demote o1=-41.5/1.6 o2=-42.1/1.9 moved; did-promote o1=-4.9/2.1 o2=-5.3/1.9 moved
DAY39 ADMISSIBLE: 60 of 60 receipts; all=True
```

Per tree (pooled N=40, stall medians): demote OFF `117.9`, `117.4`, `117.7` and ON `150.1`, `118.6`, `77.1` on b0, b1,
b2; demote ON minus OFF `+32.1 unc=1.4 -> isolated`, `+1.2 unc=1.7 -> under_resolution`, `-40.6 unc=1.0 -> isolated`;
promote ON minus OFF `-3.3`, `-3.1`, `-8.2`, each `isolated`; prime `301.6` on all three. Per tick
(`C/pro-single-day39/tick-split.log`): `DAY39 TICK HYPOTHESIS contrast=b1-b0 P1 ... P5 ... -> consistent with H`, so
option (a)'s move sits on tick 2 on this card as on the 5090 (day 38); b2-b1 moves demote-on on tick 1 (`-41.2/0.7`,
`-41.8/1.2`) and leaves tick 2 `under_resolution`. Promote tick 2 is `not_defined` on b1 and b2 (the arm stretches one
tick). Described, not ruled on: the ledger (b1 `pre_submit` 43.71 / 42.44 against b2 1.17 / 1.15; helper `hashed_in`
73.3 / 73.2 against 107.3 / 104.8), every one of the 84 b2 copy-complete lines `items=128 (32 KV, 96 f32 spans)`, and
the regime `DAY39 REGIME hold: samples=13410 temp_c=33..59 (median 50) power_w=33.50..499.83 (median 317.76)
clocks_sm_mhz=180..2422 (median 2422) mem_used_mib=0..19573 power_limit=['600.00 W']`, no program differing from its
block partner. One post-run reader fix, a timestamp parse in the regime reader (the box writes 3 fraction digits); no
rule or admissibility clause moved. C also updates the door packet's day-39 rows and status lines ("on `main` since
#656"). No recommendation; `executed-not-qualified`.

**Lead review of C day 39.** Checked on the merged tree: the harness SHA-256 equals the registered one; `box/builds.log`
names trees `091a931c0...`, `9717e8d5...`, `160929a9...`, equal to section 0; `git diff 9717e8d57 160929a92 -- crates`
is exactly the files of `97a9e091f` and `fc637d26a` (7 files, +1448/-33), so b2-b1 isolates A day 30; the
pre-registration commit precedes the first boot mark (C's push time is its statement, not independently re-derived
here). Both readers rerun on the banked receipts under a CPU cap: the stall reader's output is identical to
`reading.log` below its command header, and the tick reader's is identical except the path it names for the reading
log. The regime fix diff (`day39-cpu/diffs/regime-postrun-parse-fix.diff`) changes only the fraction parse. What the
cell says, in the lead's words and inside its scope (one card, the 27B, the plain 64-token class, `MEMRA_SERVE_SPEC=0`):
on the target card the pre-option-(a) door costs the demote-class tenant 32.1 ms over OFF; option (a) brings that to
`under_resolution`; with A day 30's spans the door's demote stall sits 40.6 ms below OFF and its promote stall 8.2 ms
below OFF, both `isolated`. Between b2 and main's engine today sit exactly #655's 12 crate files (DSv4 route policies), checked by file set.
**Ruling 43:** C day 39 is read as registered; the packet's owed target-card demote-class tenant-stall cell on the
option (a) tree now exists, on three trees, and enters the contracts-door packet (2026-10-05) as target-card evidence.
No default moves on it: the door is the owner's decision. Still owed from the cell: why b1 shows no first-touch
pre-submit step here (a per-demote allocation line, lane A's engine code), the b2 helper's `hashed_in` rise of 31.6 to
34.0 ms (off the tick, unattributed), promote tick 2 on b1 and b2, and the unchanged A day 30 section 9 items.

**Checks (tree `fa02b6559`; records only, so no GPU battery and no tag).** `bash tools/check-flags.sh` exit 0,
`bash tools/check-conflict-markers.sh` exit 0, `python3 tools/update-perf-board.py --check` exit 0,
`public-boundary: 604 matches (604 grandfathered, 0 new).`, `git diff --check origin/main HEAD` clean, zero em dashes
in added lines, `.gitattributes` in both new receipt dirs.

**Running.** A day 31 (the H2D half and the day-30 owed items) and B day 31 (`MEMRA_ADMIT_BY_MEMORY`: the OFF/ON cell,
the open-output sweep, the digest identity gate and the survey of how other engines bound an omitted `max_tokens`,
per the owner's 2026-09-23 call to measure; decide-by moving to 2026-10-07).

**Owner decisions flagged.** `MEMRA_ADMIT_BY_MEMORY`: measuring (B day 31); 8192 is a placeholder, not a default.
2026-10-04 (MoE slot cache, VMM), 2026-10-05 (the contracts door; C day 39's rows are new input), 2026-10-06 (the park
door).

## integ49 (`lane/spill-integ49-20260923`): A day 31 (DAY30's owed items 1a to 1d: the staging back on every post-take refusal, the staging charged to the governor, the span-refusal fault cell, the per-slot-class byte tally; the H2D half pre-registered)
Lane tip merged: A `979881aa0` (day 31) on main `9c07b398b` (#658) as `40891cf2d`, clean (`worker.rs` auto-merged beside
#655's 43 lines). Engine change: `crates/memra-server/src/worker.rs` only (the day-31 code commit `33b1285e0`), plus the
`MEMRA_KV_HOST_FAULT` row in `docs/FLAGS.md` (a new value `contract-spans` on the existing door, no new `MEMRA_*` name)
and the fault gate's `span-refusal` cell in `tools/kv-host-contract-fault-gate.sh`. No new `unsafe`. Between A's tip and
the merged tree sit exactly #655's crate, tool and doc files, checked by file set.

**A day 31.** 1a: a `StagedSpans` guard in `host_kv_planes_settle_contract` holds every landed staging buffer from the
take on; its `Drop` puts each back into the context's set, and only `landed()` on the one success exit hands them out;
the driver's two early exits of the `Done` arm put them back too (census: 11 `return Err(` after the take, all inside
the guard's scope). On the served path the `postpublish` cell's next demote reads `pre-submit 0.93` (day 30: 28.71), so
DAY30 finding 4 closes. 1b: a fresh staging buffer is charged to the tier governor's pinned ledger before it is
allocated (`ResidentCharge::reserve`, `Priority::Backup`, the staging tenant `digest("host-tier-staging", ..)`), held
while the set owns it and released at the latch's `clear()`, after which takes refuse and puts free; a refused charge
refuses the attach through the typed `tier D2H spans refused` arm with the tier on. One fill line per boot that
demoted: `tier span staging: 96 fresh pinned buffer(s), 156893184 bytes charged to the governor's pinned ledger; the set
holds 156893184 bytes charged` (22 of 22 target-card logs), 48 buffers and 52690944 bytes on the 9B (19 of 19). 1c:
`MEMRA_KV_HOST_FAULT=contract-spans` refuses the attach once after every span was built; the fault gate's
`span-refusal` cell byte-compares r1 to r4 against a door-OFF boot. 1d: the copy-complete line ends `by slot class:
conv C (B B), ssm S (B B), hidden H (B B), logits L (B B)`; SSM state is 95.6 % of the 27B image's heap bytes and 93.7 to
93.8 % of the 9B's (C DAY38's separating line, now a log term). Target card (BOX3, one sitting on `6d940a97c`),
verbatim: identity x4 `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok each); failure x2 `KV-HOST-SPILL FAILURE
GATE: ALL GREEN` (15 ok each); `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (eleven cells, 142 ok, 0 FAIL) with `refusal
handed back 96 span(s); the next copy-complete carries 96` and `byte-unequal request(s): none`; twin OFF and ON `->
PASS`; hit OFF and ON `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` 61 / 68 ok with the day-24 census `-> PASS`; unit cells
13 + 6 + 13 + 3 passed; double-park `DAY30 A2 pre-submit steady N=80 median=0.63 min=0.60 max=0.66 boots_on=10
demotes_per_boot=[11] rule N>=80 median<=1.5 max<=3.0 -> PASS` and `DAY28 VERDICT clauses_failed=0 -> ALL PASS`. RTX 5090
on the same tree: identity default ON, fault default and plain (142 ok each, `refusal handed back 48 span(s)`), hit OFF
and ON, all ALL GREEN. Day 26's reader keeps its own clause 2 `|d-expected|=3.9 / 4.0 -> FAIL`, as on day 30; not an
acceptance clause. **The H2D half is pre-registered, not landed** (DAY31 section 2: design H, the hash helper fills the
staging from the resident planes while the request stays parked and the owner submits KV plus 96 typed `H2dSpan`s on
one ticket; acceptance B1 to B5, B2 the promote's owner segment median <= 1.5 ms and max <= 3.0 ms over >= 20 steady
promotes; no H2D code in the tree). Budget about 2.5 of 5 agent-hours, 54 minutes of it waiting out lane C's hold.

**Lead review of A day 31.** Read every exit of the settle and the driver after the take: the guard returns the staging
on each refusal; the helper's reply path returns it on the landed path; the reply-mismatch, helper-gone and
latched-while-hashing exits latch the tier (or find it latched), so their buffers free with their holders and the
charges release at `clear()`. `staging_take` refuses on a latched set before any charge, a refused charge allocates
nothing, a failed allocation drops its charge, and a reused buffer is not charged again. The governor guard in the
dimension read is a statement temporary; `ResidentCharge::drop` takes the governor lock, and no production site holds
that lock across anything that reaches `disable`, so the latch's `clear()` cannot deadlock. The fault is one-shot on the demote side
(`is_demote` covers `SpanAttach`). One numeric program per request holds: nothing here changes a byte that is hashed,
published or promoted, and the span-refusal cell's r1 to r4 are byte-equal to a door-OFF boot on both cards. The one
limit I would have raised, the charges releasing at the latch while a quarantined ticket or a detached helper still
holds pinned buffers, is A's own stated limit in DAY31 section 1; it applies only to a latched tier. No finding.
**Ruling 44:** 1a to 1d are the door's serving path and close DAY30's owed items 4 (the staging return), the governor
charge and the span-refusal fault cell; the H2D half, the D2D half and the strong-form receipt of Move 2 owed item 1
remain owed.

**Checks.** The branch was merged with main a second time before the PR (`d1760ac21`, main `f69119ae0`: #556 and
the dsv4 PRs #661 to #663, `worker.rs` auto-merged), so every check ran on that tree. CPU battery on `d1760ac21`,
15 of 15 rc=0: fmt; portable suites (360 passed, 0 skipped of budget 0); server 874 passed; engine lib 542 passed;
tier 4; clippy `-D warnings` on the six crates and on engine, server and tier all targets; check-flags; publish
census; docs registry; pytest battery 87; conflict markers; workflow keys; perf board; `git diff --check`
(`integ49-cpu-battery-r2/`). RTX 5090 on `d1760ac21` (binary `09bbd855`, `memra-0.138.0-7eab7b8c2c2b (id:
source-tree, git: d1760ac21d64)`, one collector hold 04:10Z to 04:18Z), verbatim: serve-smoke `serve-smoke: 0
failed`; the engine `d2d_` and `d2h_span` cells `6 passed`; the worker `option_b_` and `option_c_` cells `13 passed`;
identity default ON `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok); fault default and plain
`KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (142 ok each, `refusal handed back 48 span(s); the next copy-complete carries
48`, `byte-unequal request(s): none`); hit OFF and ON `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 and 68 ok)
(`integ49-5090-r2/`). The first run on the pre-main merge (`integ49-cpu-battery/`, `integ49-5090/`) read the same
lines. One correction to that first run's receipt: its `binary.sha256` names the binary at the hold's start
(`5aa040c9`, built at `40891cf2d`), but serve-smoke rebuilds `memra-server` unconditionally, so every later cell ran
`920eebe9` (built at `aceef3589`: the same crate sources, a new git id). The r2 collector hashes the binary again
after serve-smoke. It also records a foreign co-tenant (another project's python, 1390 MiB) instead of waiting it
out; every cell here is a correctness verdict. Two earlier holds were stopped inside their lock waits, before any
cell ran, so lane B's chain and D4 cell kept the card. BOX3 is not rerun on the merged tree. A's receipts are on
`6d940a97c`, and the merged tree adds #655 and the main commits above, none of which touches the door's code (the
integ47 precedent).

**Running.** B day 31 (`MEMRA_ADMIT_BY_MEMORY`: the local two-order cell finished 03:08Z and its D4 cell 03:43Z; the
target-card run continues; the survey). Lane E (memra#641, a pre-registered deterministic repro, then a bisect and a
fix or a plain non-repro record). The lead's lane `lane/spec-ctx-edge-20260923` for memra#659, which B's cell found:
an open request's last speculative round wrote past the session cache (worker panic, NaN verify, process exit);
its red/green gate runs on the 5090 next.

**Owner decisions flagged.** Unchanged from integ48: `MEMRA_ADMIT_BY_MEMORY` measuring (decide-by moving to
2026-10-07); 2026-10-04 (MoE slot cache, VMM), 2026-10-05 (the contracts door), 2026-10-06 (the park door).

## integ50 (`lane/spill-integ50-20260923`): B day 31 (`MEMRA_ADMIT_BY_MEMORY` OFF against ON at 2048, 8192 and 32768 on both cards, the open-output survey, D4), the door-ON arm reaching memra#659, and #668
Lane tip merged: B `0fb202f2d` (day 31) on main `d544c6b82` (#668) as `a0201500c`. One conflict: the door's
`docs/FLAGS.md` row carried B's decide-by move and #668's budget sentence, and the merge keeps both. Also on this
branch: `584cbe2cf`, comment-only, B's owed item 3. The `DEFAULT_OPEN_OUTPUT_TOKENS` doc now says 8192 is a
placeholder under measurement, and the two request-struct comments in `lib.rs` now state the door's bound and that the
OpenAI contract pins none. No code line, flag or default changes.

**B day 31.** Pre-registration `ee53f7117` was committed at 23:09:14Z; the first boot started at 23:09:33Z. It ran
the full order on both cards, eight boots each: the 9B on the local RTX 5090 (served 65,536, burst of 32) and the 27B
on BOX3 (served 262,144, burst of 64). Verbatim: `DAY31 V-DOOR card=rtx5090 boots=8 excluded=0 v_id_all=True
v_alloc_all=True v_trunc_g_all=True v_retry_all=True -> PASS` and the same line for `card=pro6000`. V-ID is `differ=0`
on all 12 ON boot/order pairs, and `DAY31 D4-ID twins=49 equal=49 differ=0 -> PASS`: where both sides returned, the
door moved no token. `DAY31 D4-C r429=11 refuse_lines=11 retry_after_in_1_60=True ... -> PASS`. The selection rules,
stated and not chosen: `DAY31 SELECT card=rtx5090 R1_smallest_zero_truncation=8192
R2_smallest_v_ge_max_natural_G=8192 (max_natural_G=6405) R3_registry=32768 R4_survey=context` and `DAY31 SELECT
card=pro6000 R1_smallest_zero_truncation=2048 R2_smallest_v_ge_max_natural_G=none of [2048, 8192, 32768]
(max_natural_G=193178) R3_registry=32768 R4_survey=context`. The survey (`OPEN-OUTPUT-SURVEY.md`, each surface
pinned by tag, SHA and file:line) finds vLLM, SGLang, llama.cpp and TensorRT-LLM all bound an omitted `max_tokens`
by the remaining context, and the OpenAI contract pins no bound. **The finding: the open-output bound reaches an
engine defect under default spec.**
- 2048 and 8192 on the 5090, and 8192 on BOX3, went FATAL before their bursts. At 2048 on BOX3 all six long requests
  per order returned 500 a few rows below their allocation. The 32768 arms reached their bursts (5090: 19 x 200,
  11 x 429, 2 x 503 prefill OOM; BOX3: 47 x 200, 17 x 429). The first panic on each rig is
  `spec.rs:4590 mtp_kv_fill: scratch overflow` (5090 `O1-on2048/server.log:116`, BOX3 `O1-on8192/server.log:1423`).
- The door OFF reaches the same edge through runaways (`O1-off/server.log.gz:10286`, and the 500 at pos 65533).
- The two V-DOOR PASS lines are literal readings of the rows that returned. B names three reader gaps, left unchanged
  as ruled (DAY31 2.2): V-DOOR has no crash term, V-RETRY passes vacuously on the six boots whose bursts got no
  response, and V-TRUNC skips ON `length` rows whose OFF twin returned 500.

**#668 (memra#659), merged `d544c6b82` before this integration.**
- The qwen round loop and both gemma burst loops end a burst at a round boundary when the next round cannot land
  inside the cache (glm5's guard form). The verify funnel refuses a window past the cache with a typed error.
- The door-ON budget is `v`, so the 8 slack rows stay free as on a bounded request. Client-visible: an open request
  under the door now emits exactly `MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` tokens (it emitted `v + 8`, then crashed at the
  cap), and a door-OFF speculative runaway ends inside the cap.
- Gate `tools/spec-ctx-edge-gate.sh`, pre-registered and now in `tools/local-ci.sh`: red on the unfixed tree
  (10 FAIL through the NaN trap), green twice on the fix and again on each main merge. Receipts:
  `research/spec-ctx-edge-20260923/`.

**Lead review of B day 31.** The arms, values, orders and reader match the pre-registration. The two V-DOOR PASS
lines do not qualify the door: they carry no crash term, and every ON arm that crashed ran the pre-fix program. The
truncation and concurrency readings for the arms that survived stand as observations of that program. The identity
gate's `differ=0` stands for every row both sides returned. The survey is the missing half of the owner's "checked
over other engines" and needs no rerun. No finding against B's work.

**Ruling 45:**
- Day 31 is read as registered and qualifies nothing.
- Its ON arm is superseded by B day 32: the same cell on the fixed tree, both cards, with G = v exactly and the three
  reader gaps closed in a fresh pre-registration.
- No open-output value is selected. The door stays OFF.
- Decide-by is 2026-10-07 (the owner's 2026-09-23 call, now in the FLAGS row). The owner decides on day 32's
  receipts, with R4 reading `context` on both cards as input.

**Checks.** CPU battery on `584cbe2cf`: 14 of 15 rc=0 (fmt; portable suites 360 passed, 0 skipped; server 875 passed;
engine lib 545 passed; tier 4; clippy `-D warnings` twice; check-flags; publish census; docs registry; pytest 87;
conflict markers; workflow keys; perf board). `git diff --check` was rc=2 on trailing whitespace inside B's
`FAULTS.txt` files, which quote server lines cut at a fixed width. They are now marked `-whitespace` like the
`SUMMARY.txt` beside them, and the check is clean on the branch head (`integ50-cpu-battery/`). No GPU battery. The diff is research receipts, one FLAGS row and
three doc comments, so no binary program changes (the integ48 precedent for research-only integrations).

**Running.** B day 32: the rerun on the fixed tree. Lane E: memra#641, a fix in progress with its serving cell.

**Owner decisions flagged.**
- `MEMRA_ADMIT_BY_MEMORY`: on B day 32's receipts, decide-by 2026-10-07.
- 2026-10-04: MoE slot cache and VMM.
- 2026-10-05: the contracts door (C day 39 and A day 31 are new input).
- 2026-10-06: the park door.

## integ51 (`lane/spill-integ51-20260923`): B day 32 (the `MEMRA_ADMIT_BY_MEMORY` rerun on the fixed tree, both cards; V-DOOR FAIL on V-ALLOC only; memra#680 found)
Lane tip: B `214f2a8a5`, which already carries main `c3eb41d12` (#677 included), so the branch is a fast-forward.
Research only: `git diff origin/main HEAD` touches `research/` and nothing else.

**B day 32.** Pre-registration `51c113659` was committed at 06:32:14Z, and the first boot started at 06:33:04Z. Same
arms, workloads and orders as day 31, run on binaries built from `d544c6b82` (#668). The truncation band is G = v
exactly, and the three day-31 reader gaps are closed (a crash term, a non-vacuous retry, the twin-failed rows counted).
No crash-pattern line on any of the 16 boots. V-ID is `differ=0` on all 12 ON arms, and every ON `length` row, burst
included, ends at exactly G = v. Verbatim: `DAY32 V-DOOR card=rtx5090 boots=8 excluded=0 v_crash_all=True
v_id_all=True v_alloc_all=False v_trunc_band_all=True v_trunc_conserved_all=True v_retry_all=True -> FAIL` and the
same line for `card=pro6000`. The FAIL is V-ALLOC alone, on all 12 ON boots, every mismatch exactly -8 (first at
`rtx5090-day32/boots/O1-on2048/server.log:46`, `ctx=3551` against the registered 3559). The pre-registered expectation
kept day 31's `P + v + 72`. With #668's budget `v`, the engine books `max(ctx_cap, P + budget + 64)` = `P + v + 64`,
while the allocation `P + v + 8` is unchanged. Selection rules as printed, none chosen: `DAY32 SELECT card=rtx5090
R1_smallest_zero_truncation=8192 ... R2_smallest_v_ge_max_natural_G=8192 (max_natural_G=6405)`, `DAY32 SELECT
card=pro6000 R1_smallest_zero_truncation=none ... R2_smallest_v_ge_max_natural_G=none of [2048, 8192, 32768]
(max_natural_G=193178)`, and R3 `32768` and R4 `context` on both cards. Every reading matches what was written
before the first boot. Truncation and concurrency per card are in DAY32 2.5. In short:
- 5090, burst of 32:
  - 2048 cuts all 15 (i) requests and 3 (iv-a) requests, where OFF lets them stop naturally.
  - 8192 and 32768 cut nothing.
  - Bursts peak at 11 in flight OFF and 32 at 2048 and 8192. At 32768: 19 in flight, 11 refused with 429 and 2 lost
    to 503 prefill OOM.
- PRO 6000, burst of 64:
  - 2048 and 8192 each cut 5 requests that stop under OFF at 18,443 to 193,178 tokens; 32768 cuts 4.
  - OFF peaks at 9 in flight, 2048 and 8192 at 64. At 32768 the burst loses 46 (O1) and 47 (O2) of 64 to 503 prefill
    OOM.

Pre-fix against post-fix, as observed (50 main commits apart, so nothing is attributed to one commit):
- Day 31's crash lines in all 16 boots are gone, and the ON FATALs are gone.
- Every ON `length` row ends at v (day 31: `v + 4` to `v + 8`).
- The OFF runaways end 200 `length` instead of 500.

**memra#680, filed from these receipts.** At 32768 on the PRO 6000 the door let the 64-request burst reach 58 in
flight, against its own estimate of `est_bytes=4353602568` per request. The log has 109 `[admission] request cost`
lines but only 12 `[admit-mem] id=` lines (6 `defer`, 6 `refuse`, no `admit`). A `reclaim-on-defer` evicted 52 prefix
entries (13.3 GB) at the first defer. Then 46 and 47 of 64 died in prefill with `CUDA_ERROR_OUT_OF_MEMORY` and
returned 503. The 5090 shows the same shape at a smaller scale (2 prefill-OOM 503s per order on both days). B day 33
owns the repro, the diagnosis and the fix, with an engine change under the door authorized.

**Lead review of B day 32.** The arms, values, orders and reader match `51c113659`. I checked the V-ALLOC delta
against #668's change: `SPEC_SHRINK_SLACK` 64 on top of the new budget `v` gives exactly the observed booking, so the
reader's expectation was stale and the door's arithmetic is consistent. The FAIL stands as recorded. It is not
re-read. The identity, crash, band and retry terms are clean on both cards. The 32768 target-card burst is a real
defect that no pre-registered verdict judges (V-RETRY does not judge 503s), and B quoted it without diagnosing it,
as its scope said. No finding against B's work.

**Ruling 46:**
- Day 32 is read as registered: V-DOOR FAIL on V-ALLOC alone, the cause stated above.
- The door does not qualify. Independently of V-ALLOC, #680 is a disqualifying defect of the door as built.
- No open-output value is selected, and the door stays OFF. The owner decides by 2026-10-07.
- That decision waits for two things: B day 33's fix of #680, and a clean rerun under a fresh pre-registration with
  the booking formula written as `max(ctx_cap, P + budget + 64)`.
- The owner's inputs: R3 reads 32768 and R4 reads `context` on both cards; the 5090 reads R1 and R2 at 8192, while
  the PRO 6000 has no swept value that avoids truncation.

**Checks.** Research-only diff:
- check-flags passed and conflict markers are clean.
- `update-perf-board.py --check`: up to date.
- `git diff --check origin/main HEAD` is clean (B's receipt dirs mark their width-cut quotes `-whitespace`).
- No em dash in authored lines. The public boundary runs in the pre-push hook.
- No battery was run: no code moves.

**Running.** A day 32: the H2D half; its code and 5090 receipts are pushed, and its BOX3 sitting is next. B day 33:
memra#680, on the 5090 first, and on BOX3 after A's `LANE-A-PRO-DONE`.

**Owner decisions flagged.**
- `MEMRA_ADMIT_BY_MEMORY` (decide-by 2026-10-07): after #680's fix and the clean rerun.
- 2026-10-04: MoE slot cache, VMM.
- 2026-10-05: the contracts door (C day 39 and A day 31 are new input; A day 32 is pending).
- 2026-10-06: the park door.

## integ52 (`lane/spill-integ52-20260923`): A day 32 (the H2D half of Move 2 owed item 1, design H: B1 to B5 PASS on the target card; DAY28 clause 1b now over its bound)
Lane tip: A `e009df6d3`, which already carries main `649d96210`, so the branch is a fast-forward. Engine change, all
behind the default-OFF door `MEMRA_KV_HOST_CONTRACTS`:
- `crates/memra-engine/src/tier_transfer.rs` (`submit_h2d_spans` / `take_h2d_spans`) and `pinned_host.rs`
  (`enqueue_to_device_f32`).
- `crates/memra-server/src/worker.rs`: the log-only owner-segment field, the resident `Arc` form, the `Fill` job and
  the parked `Filling` phase, and the settle's take-back.
- `crates/memra-tier` `conformance/h2d_span.rs` with its binding.
- The `contract-promote-spans` value on the existing `MEMRA_KV_HOST_FAULT` row, and the fault gate's
  `promote-span-refusal` cell.

No new `MEMRA_*` name. The new `unsafe` covers three things, each under a stated ownership contract: the documented
`cuMemcpyHtoDAsync_v2` enqueue, its one call site, and `Engine::alloc_f32_uninit`, an uninitialized device
destination beside the engine's existing `uninit` helpers, handed out only after the copy lands.

**A day 32.** The work landed in DAY31 section 2's order, one census each (DAY32 section 1). Two departures are
named: the resident form is `Arc<Vec<f32>>` (converting to `Arc<[f32]>` would copy the bytes on the door-OFF demote's
owner thread), and `take_h2d_spans` also waits for the owner stream's wait on every span event, stricter than
registered. The target-card sitting (BOX3, one hold after lane B's chain) ran the pre-H2D binary `71d21057c` and the
H2D binary together. Verbatim:
- B1:
  - identity x4 `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`, 12 ok each;
  - failure x2 `KV-HOST-SPILL FAILURE GATE: ALL GREEN`;
  - `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 160 ok over twelve cells, with `promote-span-refusal`;
  - twin x2 `-> PASS`;
  - hit OFF/ON `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, 61 / 68 ok, with the day-24 census `-> PASS`;
  - unit cells 16 + 7 + 2 + 17 + 6, all green.
- B2: `DAY32 B2 before (pre-H2D binary) steady owner-segment N=90 median=4.58 min=4.46 max=5.11` and `DAY32 B2 after
  (H2D binary) steady owner-segment N=90 median=0.70 min=0.65 max=1.22 boots_on=10 submissions=100 with_spans=100 rule
  N>=20 median<=1.5 max<=3.0 every-submission-with-spans -> PASS`.
- B3 (a reading): `promote in-ms ... after-minus-before median +10.30`, with the helper fill `median=6.40` and every
  fill landing at its first poll.
- B4: `DAY32 B4 receipts=109 bad=0 by (items, spans)=[((32, 96), 101), ((34, 96), 8)] ... -> PASS`, and on the 5090
  `receipts=13 bad=0 ... -> PASS`.
- B5: the identity digests are unchanged in kind, `verify ok` appears on the ON arms, and
  `option_c_spans_ride_the_promote_ticket_and_land_bitwise` passes on both cards.
- The 5090 half on the H2D tree: identity default ON, fault default and plain (160 ok each), and hit OFF/ON, all
  ALL GREEN.

**Readers of earlier days over the same hold** (their own clauses, not day 32's acceptance; DAY32 section 5):
- DAY28 clause 1b (e2e ON minus OFF, `<=+20.0`) reads `+12.2`, `+12.1 -> PASS` on the pre-H2D binary and `+22.5`,
  `+22.2 -> FAIL` on the H2D binary, so `DAY28 VERDICT clauses_failed=2 -> FAIL`. Clauses 1a and 1c pass on both.
- DAY25's tenant stall ON minus OFF improves from `-8.2`, `-8.4` to `-9.1`, `-9.2`, `isolated`.
- DAY26 clause 2 fails on both binaries, as on days 30 and 31.

**Lead review of A day 32.**
- `submit_h2d_spans` refuses wholly before any enqueue, handing every span back. The copy stream waits on a fresh
  owner-stream event, so each destination's allocation is ordered before its copy. An enqueue or event error after
  the first enqueue keeps every span and quarantines the ticket (rule 5, the D2H twin's shape from day 30).
  `take_h2d_spans` returns spans only after every item's and span's event is observed and the owner-side wait is
  installed.
- `enqueue_to_device_f32` refuses a length mismatch or an unwritten source before the FFI call, and keeps the
  destination's write record alive across the enqueue.
- The `Filling` step goes through the existing contract settle callers and the `Poll` / `Block` wait. A fill past
  `HOST_HASH_DEADLINE`, or a helper that is gone, latches the tier (the day-28 rule), so a parked request cannot wait
  unbounded. B3 shows every fill landing at its first poll.
- One numeric program per request holds: the promoted planes read bitwise equal to the resident bytes (B5's native
  cell on both cards), and the identity gates are unchanged.
- I did not re-derive every line of the 1591-line `worker.rs` diff; its censuses, the twelve-cell fault gate and
  B1 carry it.
- A's finding 5 is owed, not waived: the two native span cells fail once when run in parallel in one process
  (`a batch with a running span has not landed`), and pass serially 3 of 3. Every battery here runs them serially.

**Ruling 47:**
- A day 32 is read as registered, and B1 to B5 PASS on the target card. The H2D half is the door's serving path and
  closes the H2D half of Move 2 owed item 1. The D2D half and the strong-form receipt remain owed, as do Move 1
  items 1, 3 and 4.
- The DAY28 clause 1b FAIL is a reading of an earlier day's clause that day 32 did not re-register. It is recorded
  as it reads, with no threshold moved.
- It is the design's trade, and it goes to the owner as input to the contracts-door decision (2026-10-05): about
  3.9 ms less owner-thread time per promote, which every peer's tick gains, against about 10.3 ms more end-to-end
  on the one parked promoting request.
- The lever that would close the gap is landing the fill inside the same tick instead of at the next poll. It is
  named, not built. The door stays default OFF.

**Checks.**
- CPU battery on `e009df6d3`, 15 of 15 rc=0:
  - portable suites: 363 passed, 0 skipped;
  - tests: server 878, engine lib 546, tier 8, pytest 87 passed;
  - clippy `-D warnings` twice;
  - fmt, check-flags, publish census, docs registry, conflict markers, workflow keys, perf board, `git diff --check`
    (`integ52-cpu-battery/`).
- RTX 5090 on the same tree, binary `de9a0888` hashed after serve-smoke's build, one collector hold from 15:20Z to
  15:25Z (`integ52-5090/`), verbatim:
  - serve-smoke `serve-smoke: 0 failed`;
  - the engine `d2d_`, `d2h_span` and `h2d_span` cells `7 passed`, serial;
  - the worker `option_b_` and `option_c_` cells `16 passed`, serial;
  - identity default ON `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`, 12 ok;
  - fault default and plain `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 160 ok each, with `promote refusal handed back
    48 span(s); the next H2D receipt landed 48` and `byte-unequal request(s): none`;
  - hit OFF and ON `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, 61 and 68 ok.
- BOX3 is not rerun on the merged tree. A's sitting ran the lane's code (the tree adds only main's #677, #681 and
  records around it), and the box was reclaimed after the sitting (below).

**BOX3 lost.** The rented spot instance behind BOX3 was reclaimed at about 14:52Z, after lane A's sitting ended at 14:46:24Z.
The provider's API reads it `discontinued`, and the address now answers with a different host key, so it belongs to
another machine. Nobody accepted the key or reconnected. Lane B day 33's box chain, which started at 14:51:38Z,
died with the instance and is recorded NOT RUN. The OS volume is kept detached, with the staged models, builds and
receipts on it. A restore boots the identical spot type on that volume. The type was unavailable when checked, and
the lead polls for it read-only. The only single-card offer, a confidential-compute variant at about twice the
price, is a different card mode and the owner's call.

**Running.** B day 33: memra#680, the door's prefill OOM at 32768. The fix is at `30a5ab697`: red on main at G2
(35 prefill-OOM 503s), and the green arms clean so far on the 5090. Its PRO 6000 part is owed until the target card
is restored.

**Owner decisions flagged.**
- 2026-10-05, the contracts door: C day 39, A day 31, and A day 32's TTFT trade and clause 1b reading.
- `MEMRA_ADMIT_BY_MEMORY` (decide-by 2026-10-07): after #680's fix and a clean rerun.
- 2026-10-04: MoE slot cache, VMM.
- 2026-10-06: the park door.

## integ53 (`lane/spill-integ53-20260923`): B day 33 (memra#680 fixed under `MEMRA_ADMIT_BY_MEMORY`: pending prefill workspace booked, every admission logged, prefill OOM parked; green on the 5090, PRO 6000 part owed)
Lane tip merged: B `9f335ac48`, which already carries main `0afd88e1d`, so the branch is a fast-forward. The engine
change sits in `crates/memra-server/src/worker.rs` and `admit_memory.rs`, and every new behavior is gated on the
door (`admit_memory_cfg.armed`). The new gate `tools/admit-mem-burst-gate.sh` is wired into `tools/local-ci.sh`
behind `MEMRA_CI_ADMIT_MEM_BURST`, with FLAGS and TESTING rows.

**B day 33.** Pre-registration `9491bc5d9` was committed at 14:08:50Z, and the first boot started at 14:09:07Z.

The diagnosis, from code at `c3eb41d12`, quoted with file:line in DAY33 1.1:
- The 58 in-flight requests were admitted by the generic VRAM gate, which compares each arrival's `required`
  against one live `admission_headroom` reading. The door acted only in the defer branch and logged only defers
  and refusals.
- The resident prefix cache is not counted as free.
- The missing term is the prefill workspace that sessions admitted earlier in the same burst still owe. It grows
  at prime time, and the cost model's own comment says the live reading cannot see it.
- A prefill OOM after admission is sent terminally and promoted to `Overloaded`, which is 503. The step-OOM park
  covers decode steps only.
- The 15.6 s refusal against an 8 s budget comes from one tick running for 15.6 s, with the budget checked once per
  tick.

The fix (`30a5ab697`), door-armed only:
- Every headroom reading in the admission block is reduced by `pending_prime`, the workspace still-priming
  sessions will allocate (`pending_prime_bytes`, `AdmissionHeadroom::less_pending`).
- Every admission prints a `verdict=admit` line against that reduced reading, with a `pending_prime=` field.
- A prefill CUDA OOM on a session that has emitted nothing is requeued like a step-OOM park (`prefill_oom_parkable`,
  bounded by `step_oom_retries`).

Local RTX 5090, verbatim:
- `DAY33 VERDICT card=rtx5090 boots=8 v_boot_all=True green_noom_book_all=True v_id_fix_all=True v_id_all=True
  v_off_all=True -> GREEN`.
- `DAY33 V-OFF card=rtx5090 green=green-off red=red-off rows=16 equal=16 differ=0 admit_mem_lines=0 -> PASS`: the
  default-OFF program is unchanged.
- `DAY33 V-ID-FIX ... eligible=16 equal=16 differ=0 -> PASS` on each green run: the fix moves no token.

The gate shape (open output 8192, a burst of 64):
- On main: 19 x 200, 10 x 429 and 35 x 503 prefill OOM. The gate reads RED with 34 x 503.
- On the fix, twice, `ADMIT-MEM BURST GATE: ALL GREEN`: 36 to 40 x 200 and 24 to 28 typed 429s, no OOM.
- `pending_prime` peaked at 9.24 GB.

The day-33 prefill-OOM park never fired on the card; only its CPU test exercises it. Server unit tests: 886 passed,
including 8 new tests, one per decision arm. **The PRO 6000 half did not run.** B's box chain launched at 14:51:38Z
and died with BOX3's instance before its first boot. The PRO boots remain owed: 32768 with a burst of 64, and OFF,
on both binaries.

**Lead review of B day 33.**
- Door OFF is unchanged by construction: `pending_prime` is 0 unless the door is armed, `less_pending(0)` returns
  the reading unchanged, and `prefill_oom_parkable` checks `armed` first. V-OFF's 16 of 16 equal rows, with no
  `[admit-mem]` line, are the receipt.
- The booked reduction applies only to the primary device's reading, the one the prefill workspace lands on.
- The park requeues only a session that emitted nothing (`generated` empty, `tokens_emitted` 0), under the existing
  retry bound, so a client never receives a duplicated or partial stream.
- One numeric program per request holds: V-ID-FIX 16 of 16 on every green shape.
- The root cause B names is an admission estimate that cannot see memory already promised to sessions admitted in
  the same tick. That class is not specific to the door. The door-OFF path is outside this lane's scope, and no
  door-OFF failure was observed (the OFF bursts return 200).
- B noted two timing-sensitive server tests in untouched files (`health.rs`, `lib.rs`) failing once in a whole-suite
  run and passing alone 3 of 3. That is recorded as flaky, not caused by this diff.

**Ruling 48:**
- B day 33 is read as registered: GREEN on the 5090.
- The fix is the door's admission program and lands under the default-OFF door. #680 stays open until the PRO 6000
  boots run on a restored BOX3.
- The door decision (decide-by 2026-10-07) waits for B day 34, the clean rerun on this tree. B day 34 is
  pre-registered with the corrected booking formula and a V-OOM term, and its 5090 half runs now.

**Checks.**
- CPU battery on `9f335ac48`, 15 of 15 rc=0: portable suites 363 passed and 0 skipped; server 886, engine lib 546,
  tier 8 and pytest 87 passed; clippy `-D warnings` twice; fmt, check-flags, publish census, docs registry,
  conflict markers, workflow keys, perf board and `git diff --check` (`integ53-cpu-battery/`).
- RTX 5090 on the same tree: binary `5c557664`, hashed after serve-smoke's build; one collector hold from 18:27Z to
  18:39Z (`integ53-5090/`). Verbatim:
  - `serve-smoke: 0 failed`.
  - The engine span cells `7 passed` and the worker cells `16 passed`, both serial.
  - Identity default ON `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`.
  - Fault default and plain `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 160 ok each.
  - Hit OFF and ON `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, 61 and 68 ok.
  - The #680 gate, `ADMIT-MEM BURST GATE: ALL GREEN`: `AMB no prefill OOM: oom_lines=0 status={200: 40, 429: 24} ->
    PASS`, `AMB typed refusals: served=40 r429=24 refuse_lines=24 retry_after_in_1_60=True other_non200=0 -> PASS`,
    `AMB booked admits: admit_lines=40 est_over_booked_free=0 served=40 -> PASS`.
  - The #668 gate `SPEC-CTX-EDGE GATE: ALL GREEN`.
- The collector's first launch was stopped inside its first lock wait, before any cell. The relaunch carried a
  75 x 120 s bound and waited out lane B's O1 order and a 6-minute hold by lane A's nsys job (64 attempts,
  `battery.log`).

**Running.** B day 34: the clean door rerun, its 5090 half first. A day 33: the same-tick fill, 5090 first. BOX3
restore: the identical spot type is polled read-only.

**Owner decisions flagged.**
- 2026-10-05, the contracts door: A day 32's TTFT trade. A day 33 aims to remove it.
- `MEMRA_ADMIT_BY_MEMORY` (decide-by 2026-10-07): on B day 34.
- 2026-10-04: MoE slot cache, VMM.
- 2026-10-06: the park door.
- BOX3's replacement: the identical spot type when it frees; the confidential-compute variant at about twice the
  price is the owner's call.

## integ54 (`lane/spill-integ54-20260924`): A days 33 and 34 (design F, the same-tick fill, and design K, the H2D completion checksum on the hash helper; all clauses PASS on BOX4 and the 5090; F's decision A/B owed)
Lane tip merged: A `05fbec3b2`, which already carries main `25bbb91f5`, so the branch is a fast-forward. All of it is
behind the default-OFF door `MEMRA_KV_HOST_CONTRACTS`:
- `tier_transfer.rs`:
  - design F: `SpanFillTask` and the copy stream's `span_fill_on_copy_stream` host function;
  - design K: `H2dSourceView`, `DeferredSums` and `supply_h2d_checksums`.
- `pinned_host.rs`: `enqueue_to_device_f32_after_fill`.
- `worker.rs`: the probe submitting in its own tick, the timeline fields, and the landing poll that waits for the
  helper's supplied digests.
- `memra-tier`: `conformance/h2d_deferred_checksum.rs` and the span rule 6.
- The fault gate's `digest` cell (a corrupted host byte is refused, `VERIFY FAILED`).
No new `MEMRA_*` name.

**A day 33 (design F).** The promote's staging fill runs as one host function on the copy stream, ahead of the span
copies, and the probe submits in its own tick, so the owner waits for nothing.
- On the 5090, F gained nothing: promote_in went 16.90 to 17.10 ms.
- The owner-hold cell refuted the obvious hypothesis: a 200 ms host function does not hold the owner thread's CUDA
  work (3.55 ms for alloc, H2D, D2H, event and sync while the copy stream stayed pending).
- The timeline fields and two Nsight Systems boots placed the cause (DAY33 5a). The copies land 6.3 ms after the
  probe, inside its tick. The completing poll then holds the owner 8.5 ms: `progress` SHA-256s every H2D item's
  host lease, and the 5090's leases are write-combined (`PinnedKind::for_device`), read at about 0.116 GB/s.
- The day-32 binary pays the same 8.5 ms, which is why the 5090 read 1 poll where the target card read 2.
- The D2D half is pre-registered with no code. The capture half is refuted by construction, and the restore half
  is priced at up to 0.2 ms, with a log-only price cell to decide it.

**A day 34 (design K).** The H2D completion checksum moves off the owner thread: the hash helper digests each item's
host source through a read-only `H2dSourceView`, and the item lands only with its supplied digest. Verbatim:
- The 5090:
  - `DAY34 READING arm=d33 ... landing-poll-hold N=90 median=8.50` against `DAY34 C arm=d34 ... landing-poll-hold
    N=90 median=0.12`, with `helper-ms N=100 median=8.80`;
  - `DAY34 D order=o1 ... d34-minus-d33 +0.21 rule <=+1.0 -> PASS` (o2 +0.50).
- BOX4:
  - `DAY34 PRO C run=d34 steady landing-poll-hold N=90 median=0.35 ... helper-ms N=100 median=1.40 ... rule N>=20
    median<=1.5 max<=3.0`;
  - `DAY34 PRO D order=o1 ON e2e d33 N=50 median=133.42 .. ON e2e d34 N=50 median=132.36 .. d34-minus-d33 -1.07 rule
    <=+1.0 -> PASS`;
  - `DAY28 CLAUSE 1b ... on_minus_off=+12.0 rule <=+20.0 -> PASS` (o2 `+12.0`).
- Day 34 (a): the failure gate's `digest` cell prints the injected `host bytes differ` line and `VERIFY FAILED`, and
  the GPU refusal cell is ok.
- Day 34 (b): identity x4 (12 ok), failure x2 (15 ok), contract fault (160 ok), twin x2, hit OFF/ON (61 / 68 ok), and
  unit cells 17 + 10 + 6 + 16 + 13, all green.

On BOX4, the d32, d33 and d34 binaries ran in one hold, with 60 of 60 replays PASS. Day 33's own clauses pass there
too:
- DAY28 1b `on_minus_off=+13.0` on d33, `ALL PASS`;
- B2 `N=90 median=0.76 max=0.82`;
- B4 `receipts=209 bad=0`.

But the day-32 binary already meets 1b on BOX4 (+12.8 / +12.7), so BOX4 cannot credit F with it. F's copy misses
the probe's tick there, because the fill takes 11.4 ms on that CPU, longer than the 13.1 ms to the next tick top once
the span copies are added. That is the pre-registered miss branch. Nothing on BOX4 is compared with BOX3.

**Lead review.**
- F's callback owns only host `Arc<Vec<f32>>` planes and raw staging pointers. It frees only host heap memory on
  the driver's callback thread and makes no CUDA call there. `catch_unwind` keeps a panic from crossing the FFI
  boundary, and a failed launch reclaims the leaked box.
- K's `H2dSourceView` is a read-only raw view that is `Send`. The engine keeps each source until its view returns
  through `supply_h2d_checksums`: the ticket cannot land, retire its sources or retire while one is out. An entry
  dropped unretired leaks rather than frees. So a helper that dies leaves no dangling pointer, only a latched tier.
- One numeric program holds. The digest is the same `checksum` over the same bytes, only on another thread. The
  fault cell proves a corrupted byte is still refused before anything is published.
- F's one-tick landing is proven on the 5090 by trace but not in any e2e reading (the checksum masked it until K),
  and it is flat on BOX4.

**Ruling 49:**
- Days 33 and 34 are read as registered, and every clause passes on both cards.
- K is the door's serving path. It removes the owner thread's H2D checksum hold (8.5 to 0.12 ms on the 5090, 0.35 ms
  on BOX4) without moving a byte.
- F lands as the tree's fill program. Its decision was owed, and lane A day 35 settled it while this branch was in
  its battery (receipts `cbf52cc16`, landing in the next A integration). It was a pre-registered 5090 A/B with K on
  both arms, 30 boots in one hold, 30 of 30 replays PASS, and it reads `DAY35 F DECISION -> KEEP`:
  - e2e without F minus with F: +7.81 ms (o1) and +7.44 ms (o2), against pair noise of 5.57 and 5.03 ms;
  - promote in-ms: +7.60 ms against a noise of 0.90;
  - DAY28 1b: +22.1 without F and +14.3 with F.
  F wins on the 5090 and is flat on BOX4, so it stays.
- The contracts-door trade for the owner (2026-10-05) now reads: DAY28 1b passes on the target card (+12.0 on d34),
  and the owner thread's promote cost is under 1 ms.

**Checks.**
- CPU battery on `05fbec3b2`, 15 of 15 rc=0 (`integ54-cpu-battery/`):
  - portable suites: 368 passed, 0 skipped;
  - tests: server 885, engine lib 547, tier 8 and pytest 87 passed;
  - clippy `-D warnings` twice;
  - fmt, check-flags, publish census, docs registry, conflict markers, workflow keys, perf board and
    `git diff --check` clean.
- RTX 5090 on the same tree, binary `72efd587` hashed after serve-smoke's build, one collector hold from 01:08Z to
  01:20Z (`integ54-5090/`). Verbatim:
  - `serve-smoke: 0 failed`;
  - the engine `d2d_`, `d2h_span` and `h2d_` cells `10 passed`, serial (F's fill and owner-hold cells included);
  - the worker cells `17 passed`, serial;
  - identity default ON `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`;
  - fault default and plain `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 160 ok each;
  - hit OFF and ON `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, 61 and 68 ok;
  - `ADMIT-MEM BURST GATE: ALL GREEN` (40 x 200, 24 typed 429s, no OOM);
  - `SPEC-CTX-EDGE GATE: ALL GREEN`.
- The collector waited 74 bounded lock attempts behind lane B's day-36 O1 order and lane A's day-35 A/B, and took
  the card as A released it.

**Running.** B day 35 (#680's seed-insert term) on BOX4 and the 5090. A day 35 (the F decision A/B, then the
demote's owner-thread KV hashes, pre-registered first).

**Owner decisions flagged.**
- 2026-10-05, the contracts door: DAY28 1b now passes on the target card, and K takes the owner's H2D checksum off.
- `MEMRA_ADMIT_BY_MEMORY` (decide-by 2026-10-07): after B day 35 and the clean rerun.
- 2026-10-04: MoE slot cache, VMM.
- 2026-10-06: the park door.
- BOX3's detached OS volume: keep or delete.

## integ55 (`lane/spill-integ55-20260924`): B day 35 (memra#680's remaining term: the prefix seed a still-priming session will publish is booked under the door; G-NOOM green on the target card)
Lane commit merged: B `1e3e46629` (day 35) on main `55679db26` (#702) as `38b35dc5e`, clean. The engine change sits
in `crates/memra-server/src/worker.rs` (`pending_seed_bytes`, `prefix_seed_bytes_at`, `seed_entry_bytes`,
`session_pending_seed_rows`) and `admit_memory.rs` (the `pending_seed=` field). It is armed only. The FLAGS row gains
the seed term. B's day-36 pre-registration (`88bb3b1e2`) rides the merge, but its partial receipts do not (they come
after `1e3e46629`).

**B day 35.** Pre-registration `164777767` was committed at 20:37:58Z, before any day-35 code or boot. It chose
option (a), booking the seed rather than gating the insert. The diagnosis, from day 34's BOX4 receipts, is DAY33 1.1
item 2 made concrete: every finished prime seeds a prefix-cache entry of about 197 MB during the burst, and the
day-33 booking counted only the prefill workspace. The fix `809c16444` adds each armed session's future seed entry,
sized by `prefix_snapshot_bytes`'s own arithmetic at its seed boundary, to the booked reduction until the seed
publishes or is refused. The cache contents do not change. Verbatim:
- BOX4, the red leg on main `25bbb91f5`: `DAY33 G-NOOM card=pro6000 boot=red-R64 ... oom_lines=3 burst_503=0 ...
  -> FAIL`, with `first_oom=server.log:743 ... [admit-mem] prefill OOM parked`.
- BOX4, the green legs on `809c16444`, twice: `DAY33 G-NOOM card=pro6000 boot=green-R64-r1 ... oom_lines=0
  burst_503=0 crash_lines=0 burst_200=44 ... r429=20 refuse_lines=20 retry_after_in_1_60=True -> PASS` and
  `green-R64-r2` identical, with G-BOOK PASS (`admit_lines=60 ... est_over_booked_free=0`).
  The card reads `DAY33 VERDICT card=pro6000 boots=5 v_boot_all=True green_noom_book_all=True v_id_fix_all=True
  v_id_all=True v_off_all=True -> GREEN`.
- The 5090: `DAY33 VERDICT card=rtx5090 boots=6 ... -> GREEN`, with V-ID-FIX, V-ID and V-OFF 16 of 16. As
  pre-registered, the 5090 has no red for this term: the day-33 tree read NOT-RED at G2 and at L64, so BOX4 carries
  the red leg.
- One post-block resync (a Bedrock 503 stop) and one BOX4 relaunch before any boot are recorded in
  `rtx5090-day35/run.log`.

**Lead review.**
- Door OFF is unchanged: `pending_seed` is 0 unless the door is armed, and the census test
  `pending_seed_is_armed_only_and_joins_the_booked_reduction` pins that it is computed once and joins the one
  `less_pending` reduction.
- No double count. The seed is booked only while `seed_prefix` and `seed_at` are set. The publish path clears both
  (`s.seed_prefix = false; ... s.seed_at.take()`) whether it publishes or refuses, so a published entry's bytes are
  counted once, as resident, never as pending too.
- Vision sessions and sessions without a cache book nothing, matching the publish path's own refusals.
- One numeric program per request: V-ID-FIX 16 of 16. The booking moves no byte of any request; it only defers
  admissions.

**The review fix, before merge.** Revuto on #705 found a real over-booking. `PrefixCache::prepare_snapshot`
evicts or demotes older unleased entries to fit a seed inside the cache's byte budget before it allocates. A warm,
full cache therefore replaces bytes rather than adding them, and booking every still-priming session's full seed
would 429 requests that fit. The cold-burst evidence could not show it. The lead's fix `34a4b7f23` adds
`seed_booking_cap(pending, prefix_cache_budget_bytes(), px.total_bytes)`: the term is capped at the budget the
cache has left. It is armed only, with a unit test over the cold, part-full, full and over-budget cases, and the
census updated.
- Lane B re-ran day 35's green legs on BOX4 with it, under day 35's reader and verdicts, pre-registered at
  `489b267cc` before its boots. Verbatim: `DAY33 G-NOOM card=pro6000 boot=green-R64-a1 ... oom_lines=0 burst_503=0
  crash_lines=0 burst_200=44 ... r429=20 ... -> PASS`, and `green-R64-a2` identical. V-ID-FIX, V-ID and V-OFF are
  16/16, and the card reads GREEN.
- The 5090 r2 battery on `34a4b7f23` is all green (`integ55-5090-r2/`, binary `85708a67`), and the #680 gate is
  unchanged at 37 x 200 and 27 typed 429s.
- As expected, the cap never binds on a cold burst, so both cards read the same as before it. Its effect is in the
  warm steady state, which no gate here measures.
- B's day-36 receipts ride this merge (`0adb1e823`). They are read in integ56. Day 36 measured the uncapped
  booking, and DAY36 2.2 says so.
- BOX4 was destroyed at 04:53Z, after all 559 of B's box receipts were mirrored and sha256-checked file for file.

**Ruling 50:**
- Day 35 is read as registered: G-NOOM is green on the target card twice, with the red reproduced on main. The
  review fix keeps it green (addendum, twice). With day 33's workspace term, #680's two terms are both booked, the
  seed term capped at the cache's remaining budget. #680 closes with this merge.
- The owner's `MEMRA_ADMIT_BY_MEMORY` decision cell is B day 36 on this tree, pre-registered at `88bb3b1e2` and
  running on both cards.

**Checks.**
- CPU battery on `38b35dc5e`, 15 of 15 rc=0: portable suites 368 passed and 0 skipped; server 890, engine lib 546,
  tier 8 and pytest 87 passed; clippy `-D warnings` twice; fmt, check-flags, publish census, docs registry,
  conflict markers, workflow keys, perf board and `git diff --check` (`integ55-cpu-battery/`).
- RTX 5090 on the same tree: binary `cd97b015`, hashed after serve-smoke's build; one collector hold from 03:35Z to
  03:47Z, taken as lane B's day-36 O2-off released the card (`integ55-5090/`). Verbatim:
  - `serve-smoke: 0 failed`.
  - The engine span cells `10 passed` and the worker cells `17 passed`, both serial.
  - Identity default ON `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`.
  - Fault default and plain `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`, 160 ok each.
  - Hit OFF and ON `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)`, 61 and 68 ok.
  - The #680 gate, `ADMIT-MEM BURST GATE: ALL GREEN`: `AMB no prefill OOM: oom_lines=0 status={429: 27, 200: 37} ->
    PASS`, `AMB booked admits: admit_lines=37 est_over_booked_free=0 served=37 -> PASS`. The seed booking refuses a
    few more requests than integ54's run of the same gate (37 against 40 served); its own identity terms say the
    served ones move no token.
  - `SPEC-CTX-EDGE GATE: ALL GREEN`.

**Running.** B day 36: the door decision cell, on BOX4 and the 5090. A day 35: design M, the demote's two KV hashes
off the owner thread.

**Owner decisions flagged.**
- `MEMRA_ADMIT_BY_MEMORY` (decide-by 2026-10-07): on B day 36.
- 2026-10-05, the contracts door: integ54's readings.
- 2026-10-04: MoE slot cache, VMM.
- 2026-10-06: the park door.
- BOX3's detached OS volume: keep or delete.

## integ56 (`lane/spill-integ56-20260924`): B day 36, the owner's `MEMRA_ADMIT_BY_MEMORY` decision cell (V-DOOR PASS on both cards; the decision packet)
Lane tip merged: B `c4cbbe042` on main `54fe991e9` (#705) as `845b2a12a`. Research only: B's STATE.md, the day-36
INDEX row and a one-word DAY35 fix. The day-36 receipts and DAY36.md came in with integ55.

**B day 36.** Pre-registration `88bb3b1e2` was committed before any day-36 boot, and the cell started only once day
35's BOX4 legs read GREEN. It is day 34's cell (OFF against ON at open output 2048, 8192 and 32768, both orders)
with the V-ALLOC booking in the engine's form and a V-OOM term (zero prefill `CUDA_ERROR_OUT_OF_MEMORY` lines and
zero 503s), on the day-35 tree `809c16444`, on the local 5090 (the 9B) and BOX4 (a single RTX PRO 6000 WS, the 27B).
Verbatim, 63 PASS and 0 FAIL per card:
- `DAY34 V-DOOR card=rtx5090 boots=8 excluded=0 v_crash_all=True v_id_all=True v_alloc_all=True v_trunc_band_all=True
  v_trunc_conserved_all=True v_retry_all=True v_oom_all=True g_book_all=True park_all=True -> PASS`, and the same line
  for `card=pro6000`.
- No OOM line, no park and no crash in any of the 16 boots. Identity is 16/16 to 48/48 on every ON arm.
- On the 5090, the 32768 burst is 19 x 200 and 13 typed 429s. On BOX4 it is 44 x 200 and 20 typed 429s. On day 32's
  pre-fix tree the same bursts gave 2 x 503 per order locally and 46 to 47 x 503 per order on the target card.

**The decision packet.** The owner's call, decide-by 2026-10-07.
- **Selection rules as printed, none chosen:** `DAY34 SELECT card=rtx5090 R1_smallest_zero_truncation=8192 ...
  R2_smallest_v_ge_max_natural_G=8192 (max_natural_G=6405) ... R3_registry=32768 R4_survey=context` and `DAY34 SELECT
  card=pro6000 R1_smallest_zero_truncation=none ... R2_smallest_v_ge_max_natural_G=none of [2048, 8192, 32768]
  (max_natural_G=193178) ... R3_registry=32768 R4_survey=context`.
- **Truncation:**
  - 2048 cuts 18 requests per order on the 5090 that OFF lets stop on their own. 8192 and 32768 cut none there.
  - On the PRO 6000 every swept value cuts 4 to 5 requests per order, because the 27B's natural stops reach 193,178
    tokens.
- **Concurrency:**
  - OFF peaks at 11 (5090) and 9 (PRO 6000) sessions in flight.
  - At 2048 and 8192 every burst is served whole: 32 of 32 on the 5090, 64 of 64 on the PRO 6000.
  - At 32768 the door serves 19 and 44 and refuses the rest with typed 429s.
- **What R4 means:**
  - Every surveyed engine (vLLM, SGLang, llama.cpp, TensorRT-LLM) bounds an omitted `max_tokens` by the remaining
    context, and the OpenAI contract pins no bound. Following R4 makes the door's open-output charge the served
    context. That is today's door-OFF charge, so the door's part (a) would not change the program.
  - Parts (b) (demote-before-drop in the reclaim flush) and (c) (the bounded defer and typed 429) are the rest of the
    door.
- **One caveat:** day 36 ran the uncapped seed booking. On a warm prefix cache its ON admissions may be over-booked
  (extra defers or 429s). The cap `34a4b7f23` is on main since #705. Truncation, identity and the no-OOM result are
  unaffected.
- **Options:**
  - (1) Keep the door OFF.
  - (2) Promote it with one open-output value. R1 and R2 give 8192 on the 5090 and nothing on the PRO 6000, where
    every swept value truncates the 27B. R3 gives 32768, the registry value.
  - (3) Promote parts (b) and (c) only, with the R4 charge.
  - (4) Before deciding, rerun the ON concurrency rows on the capped tree: about 0.1 agent-day, 4.5 h on the 5090
    and 6 h on a rented target card.

**Lead review.**
- The pre-registration precedes every boot. The arms, values, orders and reader match day 34's cell with its stated
  changes.
- The V-ALLOC booking now reads the engine's form, and the day-32 FAIL is not repeated.
- The uncapped-booking caveat is recorded in DAY36 2.2 by B, not added after the fact here.
- No finding.

**Ruling 51:**
- Day 36 is read as registered: V-DOOR PASS on both cards.
- The door stays OFF until the owner decides. The lead selects no value.
- The packet above is the owner's input. A capped-tree rerun (option 4) runs only if the owner asks for it.

**Checks.**
- Research only: check-flags, conflict markers, perf board and `git diff --check` are clean.
- No provider name, host, id or price. No em dash in authored lines.

**Running.** A day 35: design M' on the 5090, pre-registered after design M was refuted and reverted. Lane B holds.

**Owner decisions flagged.**
- `MEMRA_ADMIT_BY_MEMORY` (decide-by 2026-10-07): the packet above.
- 2026-10-05, the contracts door: integ54's readings (DAY28 1b passes on the target card, and the owner thread's
  promote cost is under 1 ms).
- 2026-10-04: MoE slot cache, VMM.
- 2026-10-06: the park door.
- BOX3's detached OS volume: keep or delete.

## Lanes
- D day 11 sealed and pushed (`15bd53152`); merged into integ9.
- B day 13 sealed and pushed (`1fef60006`); merged into integ10.
- A day 11 sealed and pushed (`432816926`); merged into integ10. D day 12 sealed and pushed (`b3324c262`); merged into integ10. C day 12 sealed and pushed (`7efedd13d`); C day 13 merged (#591); C day 14 merged (#594); C day 15 merged (#599); C day 16 sealed on the lane (`36fa5262b`, push refused by the perf-ci gate, integrated from the worktree); integ19. A day 13 running (pinned-kind A/B). B day 19 running (spec capture alignment, #379 gate must be green). B day 14 sealed and pushed (`25d252c6f`); merged into integ14. A day 12 running (#384, #385 harness). E, F idle.
