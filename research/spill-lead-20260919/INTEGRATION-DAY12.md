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

## Lanes
- D day 11 sealed and pushed (`15bd53152`); merged into integ9.
- B day 13 running (#445/#346/#523.4), twice restarted after transient API errors; not in integ9.
- A day 11 to be dispatched (ruling 9) when an agent slot frees. C idle (next: #552 criterion 1). E, F idle.
