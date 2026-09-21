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
interleaved A/B N >= 5) stays open.

## integ14 (`lane/spill-integ14-20260921`): B day 14
Batteries (`integration-day12/integ14-cpu-battery/`): fmt; `tools/portable-suites.sh`; memra-server 748 tests; clippy `-D warnings`; check-flags; publish census; docs registry census; collector pytest; B's `verify-day14.py` OK; perf board; diff-check: all rc=0. Local 5090 `tools/serve-smoke.sh` (`integ14-serve-smoke-5090/`): `serve-smoke: 0 failed`.

## Lanes
- D day 11 sealed and pushed (`15bd53152`); merged into integ9.
- B day 13 sealed and pushed (`1fef60006`); merged into integ10.
- A day 11 sealed and pushed (`432816926`); merged into integ10. D day 12 sealed and pushed (`b3324c262`); merged into integ10. C day 12 sealed and pushed (`7efedd13d`); C day 13 merged (#591); C day 14 sealed and pushed (`fc7375817`); merged into integ13. B day 14 sealed and pushed (`25d252c6f`); merged into integ14. A day 12 running (#384, #385 harness). E, F idle.
