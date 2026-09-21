# local-ci on a one-card rig: pair-only tests and the ignored-suite race

Verdict: the battery's "memra-engine lib suite (GPU-only #[ignore] tests)" stage was red on
the local RTX 5090 Laptop for two independent reasons, neither in the lanes it blocked. Three
tests need two native CUDA devices and say so; the rest of the red is a race between GPU tests
that the default multi-threaded test harness runs concurrently. Skipping the pair-only tests
(derived from their own ignore reasons) and running the stage with `--test-threads=1` is green
6 of 6 on this rig.

## What was red

`raw/battery-red-20260921-fix379-lane.log` (the `tools/local-ci.sh --perf` run on the #578
lane, main `dbf88d467`): 20 passed, 4 failed.

| Test | Failure | Class |
| --- | --- | --- |
| `model_memory::native_tests::glm_indexed_mla_kda_state_materialization` | `native device 1 required; never skip: CUDA_ERROR_INVALID_DEVICE` | pair-only (reason: "requires a native CUDA pair under the coordinator's exact two-card leases") |
| `model_memory::native_tests::glm_peer_admission_materialization_trim_and_refill` | `two native CUDA devices are required; never skip` | pair-only ("native CUDA pair required; run under the provided two-card exclusive locks") |
| `dsv4_graph::tests::replay_rust_partial_submission_and_capture_cleanup` | `CudaContext::new(1)`: `CUDA_ERROR_INVALID_DEVICE` | pair-only ("requires the exclusively locked development pair") |
| `dsv4_grouped::half2_chain_identity_tests::cuda_half2_chain_identity` | `GU half2 launcher did not engage` | race (below) |

The stage's comment still said "the engine's three `#[ignore]` GPU tests"; the tree now has
24. The last banked green of this stage (2026-09-04, 2026-09-07) ran exactly three tests.

## The race

`cuda_half2_chain_identity` passes alone (`raw/half2.log`, 4.0 s). In parallel the GPU tests
share one device and process-global gate doors (`set_moe_f16g_*_for_gate`; `GateRestore`'s
`Drop` clears all five), and stream capture state, so one test's teardown lands inside
another's chain. Measured on this rig (`raw/flake-loop.log`, `raw/serial-loop.log`,
`raw/ignored-ab.log`), all with the three pair-only tests skipped unless noted:

| Harness | Runs | Result |
| --- | --- | --- |
| default (parallel) | 7 | 6 green, 1 red (`dsv4_graph::tests::cuda_capture_runs_once_and_restores_scope_after_failure`) |
| default, battery-exact (no skips) | 4 | 4 red: the pair trio every time, plus `glm_same_ordinal_owners_are_not_lost` and `dsv4_c4::tests::cuda_recent_c4_preserves_hits_misses_wrap_and_rollback` on one run |
| `--test-threads=1` | 7 | 7 green (21 passed each) |

The victim changes with timing, which is why the battery saw the half2 test and the loop saw
two others. Parallel runs finish in about 1.1 s, serial in about 4.2 s: the cost of serial is
three seconds per battery.

## What landed where

While this lane was measuring, a parallel session landed #583 on main with the pair-only
part solved inside the tests: `test_support::skip_unless_native_pair` counts CUDA-runtime
devices (so `CUDA_VISIBLE_DEVICES` is honored) and prints `SKIP-PAIR <test>`, which
`tools/local-ci.sh` counts and reports; the perf runner also skips a spec cell whose drafter or
ranks file is absent. This lane's scanner-from-ignore-reasons, its teeth and its drafter skip
were the same fixes from the outside and are dropped in favor of #583.

What this lane adds on top of #583 is the race: the ignored stage now runs with
`--test-threads=1`. The "never skip" inside the pair tests still holds on the pair box.

## Battery on this lane (2026-09-21, local RTX 5090 Laptop)

`raw/local-ci-perf-stages.log`. First attempt (`raw/local-ci-perf-attempt1-cotenant-oom.log`)
died at kernel-check with `CUDA_ERROR_OUT_OF_MEMORY`: another session's `local-ci --perf`
held a 22 GB card allocation at the time (the battery printed "other GPU compute apps
present"); re-queued behind a quiet-GPU wait. Second attempt: every correctness stage green,
the ignored-suite stage printed `SKIP 3 pair-only #[ignore] test(s)` and `21 passed; 0 failed`
in 4.3 s, and the battery reached its perf stage for the first time on this rig since
2026-09-05 (this run used the lane's own scanner skip; #583's in-test skip gives the same
21-test set). Perf rows measured (both `window_clean`; #583 appended main's own fresh rows the
same night, so these are recorded here, not in the jsonl):

| Cell | tok/s |
| --- | --- |
| 26b-plain-short | 208.15 |
| qwen9b-plain-short | 138.88 |

One perf cell read `26b-spec-d1736: FAIL (no reading)`: the rig has the 26B model but not its
MTP drafter (`gemma4-26b-a4b-qat-gguf/drafter/MTP/...`), and `run_cell` only checked the
model file. A missing artifact is a SKIP, like the model check; `run_cell` now says
`SKIP (no draft: <path>)`, exercised in isolation with a stub runner (model present, drafter
absent prints the SKIP and runs nothing; a missing model still prints `SKIP (no model)`).
The other cells were `SKIP (no model)` as before on this rig.
