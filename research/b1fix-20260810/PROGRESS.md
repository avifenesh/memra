# PP-N Step3.5/Step3.7 B=1 correctness default

## Contract

- Branch/worktree: `lane/cx-b1fix` in `wt-cx-b1fix`, base `6bd91634`.
- Fix: on PP-N only, Step3.5/Step3.7 `B=1` must use the same
  `step35_decode_batch_layers` numeric class as `B>1`. Non-Step35 models retain the
  existing `b1_stage_fast` path.
- Gate closure: `step35-b2-geometry-gate.sh` must exercise live defaults and prove an
  explicit `B=1 -> B>1` transition, not only static concurrency widths.
- Target rig: box1, 2x RTX PRO 6000, one `/tmp/memra-gpu.lock` hold per bounded block.
- Stop condition: fixed-binary matrix is one completion hash across c=1 x10, c=8 x10,
  c=2 x10, and first-late x5; any second hash stops the lane.

## Status

- 2026-08-10: read `research/p0iso-20260810/RESULTS.md` in full and confirmed the
  exceptionless 90-boot isolation maps the defect to the PP-N
  `b1_stage_fast`/`step35_batched` boundary.
- 2026-08-10: read `CLAUDE.md`; worktree is clean at `6bd91634` on the dedicated lane.
- 2026-08-10: `~/.lanectl/inbox/cx-b1fix.md` is not present. The exact path will be
  rechecked before every remote block.
- 2026-08-10: box1 preflight is idle. Baseline source/binary remain
  `188154299064a42b67fc8eb1f41757cf6237300d` /
  `e7e6515e9f47030a7137ba9fdf7c40d43f0764d02699b38959f134ee0ace65b3`.
- 2026-08-10: implemented the PP-N Step35 exclusion from `b1_stage_fast`; the Step35
  branch now selects `step35_decode_batch_layers` directly at every width. The PP engine
  battery retains its stage-fast arm for non-Step35 models and marks it inapplicable for
  Step35.
- 2026-08-10: updated `b2geo35` to remove the `MEMRA_SERVE_B1FAST=0` masking pin and add
  a tick-traced cell that starts one streaming row, waits for its first emitted token, then
  admits two late rows and requires `ready=1 -> ready>=2` plus byte identity.
- 2026-08-10: local validation: `git diff --check`, `bash -n`, and ShellCheck pass;
  `cargo check --release -p memra-engine --bin decode-batch-gate -p memra-server` passes
  with CUDA 13.1 / sm_120a.
- 2026-08-10: box1 clean build at code/gate head `6e50efdb` passes with CUDA 13.2 / sm_120a.
  Fixed `memra-server` SHA-256 is
  `6a7c2046eb3197773def91baf012abd629e0b0ced239ec2d38016c93be5ca7e5`.
- 2026-08-10: required fixed-binary matrix **PASS** — 35/35 fresh boots and 150/150
  requests across c=1 x10, c=8 barrier x10, c=2 x10, and first-late x5 returned only
  `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.
  Zero errors and zero divergences. Full receipt mirrored under `raw/matrix/`.
- 2026-08-10: added `perf-box1.sh`: one-lock, N=5-per-arm alternating-order base/fixed
  server boots, c=1 short TTFT and streaming steady-decode receipts with the exact serve
  shape and binary-hash preflight.
- 2026-08-10: interleaved N=5 perf complete. Median sustained c=1 decode is
  85.423 base vs 81.399 fixed tok/s (**-4.710%**); short TTFT is 70.281 vs
  70.125 ms (-0.222%). Raw request rows, server logs, and thermal snapshots are under
  `raw/perf/`.
- 2026-08-10: standard target-rig battery complete: kernel-check ALL GREEN; PP
  decode-batch exactness B=1/2/4/8 has zero failing arms; run-gen MATCH; run-spec K=1..8
  SELF-CONSISTENCY PASS; chunk/tick naked gates and canaries PASS; live-default B2 geometry
  and fail-closed canary PASS; serve-smoke reports zero failures.
- 2026-08-10: reduced receipts, added raw checksums, and wrote `RESULTS.md`. The lane is
  complete and remains unpushed/untagged as requested.

## Terminal state

The fail-closed fix is correctness-green. Stop after `RESULTS.md`; eager-path bit parity is
a separate follow-up, not work for this lane.
