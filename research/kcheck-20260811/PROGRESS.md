# Kernel-check gate integrity progress

Date: 2026-08-11
Branch: `lane/cx-kcheck`
Base: `24359bc5`

## Scope

- Make every optional `kernel-check` cell emit an explicit `SKIP` line when unavailable.
- Add repeatable `--require-cell NAME` and `--require-manifest FILE` enforcement.
- Report final successful runs as `ALL GREEN (<n> cells, <m> skipped)`.
- Wire required cells into local and mechanism-specific validation scripts.
- Preserve raw positive and negative gate logs under `raw/`.

## Status

- [x] Read the lane brief and repository law.
- [x] Verified the clean worktree is on `lane/cx-kcheck`.
- [x] Inventory kernel-check cells and all callers.
- [x] Implement skip accounting and required-cell validation, including resolved-artifact tensor
      fallthroughs and automatic observation of verdict-bearing cell names.
- [x] Add focused tests and wire local, 27B, and Step35 validation scripts.
- [x] Build and run the 5090 positive/negative gate battery.
- [x] Record the final verdict in `RESULTS.md`.

## Checkpoints

- `cargo check --release -p memra-engine --bin kernel-check`: PASS.
- `cargo test --release -p memra-engine --bin kernel-check`: PASS, 4/4 focused tests.
- `bash -n` on every changed validation script: PASS.
- External-review addendum: removed the `gate_exps` / `gate_inp` tensor panics and made
  `nvfp4-gemm` Q5_K, NVFP4, and native-static sub-arm coverage explicit.
- `cargo build --release -p memra-engine --bin kernel-check`: PASS on sm_120a.
- Full 5090 run under `/tmp/memra-gpu.lock`: PASS, `ALL GREEN (98 cells, 0 skipped)`.
- Bogus `MEMRA_KC_MODELS_DIR` with required `DUAL-BATCHED-AUX`: PASS negative proof; explicit
  model `SKIP`, `MISSING REQUIRED CELL`, no green summary, exit 1.
