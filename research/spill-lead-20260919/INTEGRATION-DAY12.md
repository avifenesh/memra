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

## Lanes
- D day 11 sealed and pushed (`15bd53152`); merged into integ9.
- B day 13 sealed and pushed (`1fef60006`); merged into integ10.
- A day 11 sealed and pushed (`432816926`); merged into integ10. D day 12 sealed and pushed (`b3324c262`); merged into integ10. C day 12 sealed and pushed (`7efedd13d`); C day 13 merged (#591); C day 14 merged (#594); C day 15 sealed and pushed (`8ed05bfc6`); merged into integ17. B day 14 sealed and pushed (`25d252c6f`); merged into integ14. A day 12 running (#384, #385 harness). E, F idle.
