# cx-tauhist progress

Branch: `lane/cx-tauhist`
Base: `79c3c0b27`
Started: 2026-08-12

## Contract

- Count accepted speculative tokens by draft position in both verifier walks.
- Keep the verifier telemetry-only: atomics, zero hot-path allocation, no launch or numeric changes.
- Export operator-only, per-model `spec_tau` and `spec_accept_by_position`; omit tenant scopes.
- Unit-test synthetic counter math and keep both affected crates green.
- Capture the RTX 5090 9B NVFP4+MTP K=1..8 histogram receipt plus exactness spot-check.
- Do not merge, tag, push, update boards, or run repository-wide formatting.

## Progress

- [x] Confirm clean dedicated worktree and branch.
- [x] Trace verifier accept decisions and existing metrics windowing.
- [x] Add counters and focused unit tests.
- [x] Add operator-only server metrics and focused unit tests.
- [x] Run affected crate tests.
- [x] Run RTX 5090 K=1..8 receipt and `run-gen` argmax spot-check.
- [x] Record raw receipt, hashes, commands, and outcome.
- [x] Review intended diff and commit the completed lane.

## Evidence

- `cargo test -p memra-engine`: 77 passed, 1 ignored; binary/integration tests green.
- `cargo test -p memra-server`: 194 passed.
- Focused synthetic masks: 4 rounds, 12 drafts, 6 accepts -> tau 1.5 and accepted-by-position
  `[3, 2, 1]` over offered `[4, 4, 4]`.
- New first-class metrics use a rolling 30-second per-model window; the cumulative `spec` block
  remains unchanged for compatibility.
- RTX 5090 `run-spec`, Qwen3.5 9B NVFP4 with embedded MTP, K=1..8: eight identical-to-target
  passes and final `SELF-CONSISTENCY PASS`; raw per-position counts are frozen in
  `histogram.json`.
- `run-gen`: prefill/decode argmax `MATCH` and batched-prime/tokenwise argmax `MATCH`.
- Hardware commands ran under `/tmp/memra-gpu.lock`. A scheduled ColBERT refresh held a foreign
  1,392 MiB CUDA context at 0% utilization at command start, so raw timing is explicitly excluded
  from performance evidence.
- `raw/SHA256SUMS` verifies the machine-readable histogram and every retained raw log. Final diff
  review found only the telemetry implementation, serving documentation, and this lane's evidence;
  generated perf boards and kernel code are untouched.
