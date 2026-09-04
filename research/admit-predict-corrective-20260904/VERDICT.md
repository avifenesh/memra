# Lane Verdict: admit-predict-corrective-20260904

## Title
Corrective Follow-Up: Structural Multi-Device Detection and Subprocess Boot Enforcement

## Context & Rationale
PR #189 (commit `0545dd0ea`) landed the predictive admission enforcement framework closing #187. During audit, two critical P1 corrective requirements were identified prior to unblocking Phase 2 (box13 qualification):

1. **Automatic Expert-Parallel Bypass**: The initial `is_multi_device_deployment` check inspected ambient environment variables (`MEMRA_PP_*`, `MEMRA_STEP_*`, `MEMRA_PARALLEL*`, `MEMRA_GLM5_TP`). However, under `MEMRA_PARALLEL=auto`, expert parallelism (EP) stores topology directly within the loaded model (`HybridModel`, `Glm5TpRt`, `GpuTensor` device ordinals) without requiring ambient environment variables. This created a potential bypass where multi-GPU EP models could pass single-GPU predictive admission enforcement.
2. **Subprocess Boot Safety Gate Coverage**: Grep-based assertions in tests did not verify real runtime boot failure or termination semantics. Real subprocess execution (`Command::new(...)`) was needed across:
   - Single GPU (clean boot, exit 0, no fatal message)
   - Pipeline parallelism (`MEMRA_PP_DEVICES`)
   - Explicit TP/EP (`MEMRA_STEP_TP`, `MEMRA_STEP_EP`)
   - Automatic EP (`MEMRA_PARALLEL=auto`)
   - Environment-unset-after-load (model loaded, structural multi-device state present, environment cleared)
3. **Decoupled Admission Production Helpers**: The predictive evaluation and decoupled admission logic needed extraction into callable production functions (`evaluate_predictive_admission_verdict`, `evaluate_decoupled_admission`, `EvaluatedAdmissionOutcome`) and direct multi-session concurrency testing across shared headroom and exempt tenants.

## Implementation Receipts
- **Engine Capabilities**:
  - Added `ordinal(&self) -> usize` to `GpuTensor` (`crates/memra-engine/src/model.rs`).
  - Added `devices(&self) -> Vec<usize>` to `Glm5TpRt` (`crates/memra-engine/src/glm5_tp.rs`).
  - Added `devices(&self) -> Vec<usize>` and `is_multi_device(&self) -> bool` to `HybridModel` (`crates/memra-engine/src/hybrid.rs`).
- **Server Structural Checks**:
  - `is_multi_device_deployment(&loaded)` inspects loaded model state first:
    - Queries `lm.model.is_multi_device()`.
    - Inspects weight tensor ordinals across primary model and speculative companion model.
    - Inspects PP stage cuts.
    - Falls back to ambient deployment variables.
- **Production Helpers & Concurrency Tests**:
  - Extracted `evaluate_predictive_admission_verdict` and `evaluate_decoupled_admission`.
  - Rewrote `admit_predict_concurrency_decoupled_accounting` using direct evaluation calls across 4 concurrent sessions, verifying tenant exemptions, quota consumption, and rejection when headroom is saturated.
- **Real Subprocess Boot Tests**:
  - Added `admit_predict_enforce_boot_subprocess_runner` driven by `MEMRA_ADMIT_PREDICT_BOOT_SUBPROC_CASE`.
  - 5 subprocess tests verified:
    1. `admit_predict_enforce_boot_single_gpu`: exits 0, no multi-device fatal log.
    2. `admit_predict_enforce_refuses_pp_boot`: fails closed with exact `ADMIT_PREDICT_MULTI_DEVICE_FATAL_MSG`.
    3. `admit_predict_enforce_refuses_explicit_tp_ep_boot`: fails closed with exact fatal msg.
    4. `admit_predict_enforce_refuses_automatic_ep_boot`: fails closed with exact fatal msg.
    5. `admit_predict_enforce_refuses_env_unset_after_load_boot`: fails closed with exact fatal msg even with zero ambient env vars.

## Verification
- `cargo fmt --check`: CLEAN (0 diffs)
- `cargo clippy -p memra-engine -p memra-server --all-targets`: CLEAN (0 warnings, 0 errors)
- `cargo test -p memra-server --lib worker::tests::admit_predict`: 10 passed, 0 failed
- `cargo test -p memra-server`: 599 passed, 0 failed
- `tools/local-ci.sh --perf`: ALL GATES GREEN
  - `serve-stress-gate`: ALL GREEN (c=64 complete, streams well-formed, worker alive, log clean)
  - `spec-on-cache-hit`: ALL GREEN (qwen arm spec-on boot, sampled cells, growth cells, boundary draws, spec-off twin identity reference, rollback posture)
  - `memra-engine lib suite` (GPU-only #[ignore] tests): 3 passed, 0 failed
  - `perf stage`: 0 fail, 0 warn (`qwen9b-plain-short: 133.65 tok/s [OK]`)
  - Appended row to `research/tune-data/perf-ci.jsonl`: `{"ts":"2026-09-04T16:46:41Z","git":"75953e340","cell":"qwen9b-plain-short","toks":133.65,"profile":"performance","load":5.94,"window_clean":true}`
  - Raw run log banked at `research/admit-predict-corrective-20260904/local-ci-perf.log`.
