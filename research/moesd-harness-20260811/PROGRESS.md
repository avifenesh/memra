# MoESD target-efficiency harness progress

Date: 2026-08-11
Lane: `lane/cx-moesd`
Starting head: `1592253f`
Rig: box1 (`ubuntu@<rented-box-ip>`), 2x RTX PRO 6000 Server Edition; every remote GPU block holds `flock /tmp/memra-gpu.lock` and starts detached.

## Frozen scope and decision rule

- Build the standalone `target_efficiency` harness following the existing optipipe research-bin pattern; do not change serving defaults.
- Measure the full B in {1,2,4,8,16,24,32} by gamma in {1,2,3,4,6,8} sweep with N=5 forward/reverse interleaving, expert-union telemetry, raw logs, and thermal receipts.
- Pivot decision is owner-approved and frozen before measurement: X=1.5 at B=8, gamma=4, with measured plain B=8 throughput as the second gate.
- A NO-GO/K=0 outcome does not reopen the closed PP-2 speculative-decode verdicts #87/#94.
- This is a read-only measurement lane: no serving-default, perf-board, merge, tag, push, or formatting changes.

## Execution plan

1. Audit the current verify-tier API and the standalone optipipe harness pattern; implement the smallest default-off telemetry and research binary/driver needed for the frozen schema.
2. Rebuild and validate the release harness locally with synthetic inputs only after `cx-ncuspike` releases the local RTX 5090 lock.
3. Wait for `cx-dualpp1` to release box1, then rebuild release and run the complete matrix under one detached exclusive lock hold with idle-before/after and thermal evidence.
4. Verify no exactness regression on final source, summarize N=5 medians and decision-tree inputs, and write `RESULTS.md` with the GO/NO-GO verdict.

## Checkpoints

- Required DESIGN, repo law, queue, and owner approvals read; dedicated worktree is clean.
- Added the standalone `target-efficiency` binary, a diagnostic-only B*gamma row mapping over the existing Step-3.7 batch walk, replay-only expert-union capture, NVML sampling, matrix reduction, and the one-lock box1 driver. The serving wrapper still passes the identity row mapping and its CUDA launch sequence is unchanged.
- The design text names `T(B,1)/T(B,gamma)` as the paper target-efficiency metric but approves X=1.5 for “gamma columns versus gamma serial steps.” Both are retained: `target_eff` is the paper ratio, while `serial_amortization = gamma*T(B,1)/T(B,gamma)` is the dimensionally matching input to the frozen X=1.5 decision tree.
- Summary reducer synthetic self-test, expert-union collector unit test, shell syntax, and diff whitespace checks pass. Local `cargo check -p memra-engine --bin target-efficiency` and the optimized `cargo build --release -p memra-engine --bin target-efficiency` completed under separate atomically acquired 5090 lock holds; the linked CLI smoke also passed. The target Step checkpoint was not loaded locally because it cannot fit this rig.
- A follow-up audit added a pre-measurement causal check: packed B=1, gamma=8 must preserve every sequential target argmax before any timing row is emitted. The detached box driver now also pins the on-box Rust toolchain and refuses a dirty or unexpected source commit. Its shell syntax, reducer self-test, `cargo check`, collector unit test, and optimized release build pass on the final local source under one clean 5090 lock hold.
- The retargeted `cx-ncuspike` lane briefly acquired the local lock between those blocks; an earlier follow-up command refused with rc=75 and did not start. Every successful local block began only after a fresh nonblocking acquisition, so no GPU work overlapped.
- Coordination waited for `cx-dualpp1` to release box1 and for `cx-ncuspike` to release the local
  5090; no MoESD GPU block contended either lane's lock window.
- The live box1 queue added a blocking provenance review before launch. `run-box1.sh` now fails
  closed when `MOESD_EXPECTED_SOURCE` is unset, accepts only an exact source match unless the
  caller deliberately supplies the literal `any` opt-out, and passes its unset-variable smoke,
  shell syntax, reducer self-test, and diff-whitespace checks. The scored launch will use the
  exact full commit, never the opt-out.
- The first detached box1 attempt was stopped during release compilation, before model load or
  any measurement row, when a newly visible queue entry required a conservative handoff. The
  stopped attempt remains preserved. After that lane's own final status confirmed its PRO work
  was deferred behind MoESD, the fresh second attempt completed and reduced all 210 N=5 rows.
  `kernel-check` then passed with `ALL GREEN (83 cells, 21 skipped)` and 380 `OK` lines, but the
  driver exited on its obsolete colon-form `ALL GREEN` grep before the remaining exactness gates.
  The assertion was updated to match the current counted summary; the final run used a fresh output
  directory and reran the complete measurement and exactness block rather than splicing that partial
  battery into scored data.
- The scored detached box1 run rebuilt release from exact source
  `edbf6827d2c6993b15301c966a898a419aebfd40`, emitted all 210 points in the required alternating
  N=5 matrix, retained 42 cell summaries plus the decision row, and ended `MOESD_PASS` with no
  compute apps and 0 MiB on both GPUs. The scored receipt is `raw/box1/`; both excluded attempts
  remain under separately named receipt directories.
- The pivot `(B=8, gamma=4)` measured `T1=42.801 ms`, `T4=169.232 ms`, paper target efficiency
  `0.2529`, serial amortization `1.0116`, and projected realistic throughput `52.38 tok/s` versus
  the frozen plain-B8 threshold `173.62 tok/s`. Both owner-approved gates fail: final verdict is
  **NO-GO / CLOSED**, K=0 remains correct, and #87/#94 remain closed.
- Final-source exactness is green: `kernel-check` reports `ALL GREEN (83 cells, 21 skipped)`, both
  `run-gen` argmax comparisons match, `run-spec` passes self-consistency for K=1..8, and the PP
  decode-batch battery is bit-identical at B=1,2,4,8. `RESULTS.md` records the complete matrix,
  decision-tree inputs, thermal regime, provenance hashes, and scope limits. No serving default,
  performance board, merge, tag, push, or formatting surface was changed.
