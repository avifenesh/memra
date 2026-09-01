# Gemma4 PP-2 auxiliary tensor progress

Started 2026-08-11 on `lane/cx-gemmaaux` from `24359bc5`.

## Scope

- Replicate `GemmaAux.ones` per CUDA ordinal and require a fail-fast ordinal accessor.
- Upload decode position data independently on each PP-2 stage and assert the stage-local device invariant.
- Audit every model-level auxiliary tensor read from stage engines, including `suppress_d`, and record a verdict for each candidate.
- Preserve byte-identical single-device behavior.

## Required evidence

- `cargo test -p memra-engine`
- Local RTX 5090 `kernel-check` under `/tmp/memra-gpu.lock`
- Local RTX 5090 `run-gen` argmax checks against pinned Gemma4 12B and E4B goldens under the same lock
- Raw command output retained in `raw/`

## Status

- [x] Baseline established and the required progress-only commit created (`123217da`).
- [x] Mirrored the existing Gemma4 RoPE layout for `ones`: one immutable allocation per
  distinct PP CUDA ordinal plus a fail-fast `GemmaAux::ones(&Engine)` accessor.
- [x] Routed all seven Gemma attention families through the accessor and added debug-only
  tensor/stream ordinal checks before their norm kernels.
- [x] Replaced the shared PP-2 position pointer with independent stage-0 and stage-1 uploads,
  asserted both at stage entry, and removed the obsolete host-bounce refusal.
- [x] Focused `cargo check -p memra-engine` passed on sm_120a; raw output is in
  `raw/cargo-check.log`.
- [x] `cargo test -p memra-engine` passed: 57 library tests, the run-gen unit test, and
  three MLA fixture tests passed; two CUDA-only tests were ignored. Raw output is in
  `raw/cargo-test.log`.
- [x] Completed the model-level auxiliary tensor sweep; the detailed candidate-by-candidate
  verdict will be retained in `RESULTS.md`.
- [x] Ran the required local RTX 5090 gates under one `/tmp/memra-gpu.lock`
  acquisition: full `kernel-check` was ALL GREEN, and the pinned Gemma4 12B and E4B
  generation checks matched both argmax paths and their 20-token goldens.
- [x] Recorded the final bounded verdict in `RESULTS.md`; this lane remains explicitly
  not a Gemma4 PP-2 serving qualification.
