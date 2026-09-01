# Vast 27B fused-aux pre-release battery

Lane: `lane/cx-vast27b`

Started: 2026-08-11

## Objective

Run the required pre-release battery for the default-on `MEMRA_NVFP4_AUX_DUAL` path on the Vast
2x RTX PRO 6000 Blackwell verification box. Keep the Step service on `:8002` undisturbed when one
card has enough room; otherwise stop it only for a bounded gate block and restore the production
launcher plus soak before handoff.

The runtime source must contain merge `c58ebd62`. The lane started from local commit
`80575c4b410cc56ad7135ab2e8fb186ac74a3e5c`, which is a descendant of that merge.

## Required evidence

- [x] Capture the remote GPU, process, service, source, toolchain, and model-artifact baseline.
- [x] Verify the full SHA-256 digests of the target and draft GGUF artifacts.
- [x] Build the exact source tree in release mode without running `cargo fmt`.
- [x] Run `kernel-check` and retain an explicit `DUAL-BATCHED-AUX` PASS receipt.
- [x] Run the 27B `run-gen` argmax gate and retain `MATCH`.
- [x] Run `run-spec` K=1..8 and retain eight PASS results.
- [x] Run one-lock interleaved N=5 A/B (`MEMRA_NVFP4_AUX_DUAL=1` versus `0`) and retain raw runs,
  thermal context, token identity, and the median comparison.
- [x] Restore and verify Step `:8002` plus `/root/soak.py` after every disruptive block and at
  final handoff, including `/health` and one streamed completion.
- [x] Commit raw receipts, write the final verdict in `RESULTS.md`, and reconcile
  `research/27btune-20260811/RESULTS.md` with the on-box result.

## Verdict rule

Exactness failure means **revert the fused-aux default**. A flat A/B result does not require a
revert when all exactness gates are green; a measured regression does.

## Log

- 2026-08-11: Read the lane brief and `/home/avifenesh/projects/bw24/CLAUDE.md` before work.
- 2026-08-11: Confirmed the dedicated worktree/branch is clean and the starting commit contains
  `c58ebd62`.
- 2026-08-11: Captured the initial Vast receipt at `2026-08-11T02:26:15Z`. The box reports two
  RTX PRO 6000 Blackwell Workstation Edition GPUs, driver 610.57.04, and CUDA 13.1. GPU 0 had
  51,000 MiB free while production Step PID 13667 remained healthy on `:8002`; soak PID 13803 was
  live. This leaves enough room to attempt the single-GPU 27B battery without stopping serving.
- 2026-08-11: Created the isolated remote checkout `/workspace/cx-vast27b/memra` and detached it
  exactly at `c58ebd6257334c7b2628ec7367efd4713e8126c1`. Production checkout and binary were not
  modified.
- 2026-08-11: The initial artifact transfer found `/usr/bin/rsync` was a zero-byte mode-0444
  placeholder. Reinstalled the pinned Ubuntu `rsync` package (3.2.7) without restarting services;
  the custom 1.24 GB draft completed via SCP and the 15.71 GB target resumed over repaired rsync.
- 2026-08-11: Added `run-vast27b-block.sh` for reproducible remote build, exactness, interleaved
  A/B, and final service receipts. It never stops the production server in the normal fit-beside
  path; the A/B stops only the 21-second-cadence soak and restores/verifies it from an exit trap.
- 2026-08-11: Remote release build at pinned source `c58ebd62` passed in 2m33s. The build selected
  CUDA 13.1 and auto-detected `sm_120a`; post-build Step health remained `ok`, idle, and reported
  zero Xid warnings. Gate binary hashes are retained with the raw build receipt.
- 2026-08-11: Transfer checkpoint at `2026-08-11T02:46:35Z`: the target had reached
  11,428,832,256 / 15,705,920,064 bytes; the complete draft already matched pinned SHA-256
  `b445fbb1...f3581`. Step remained healthy and idle with zero Xid warnings.
- 2026-08-11: Exact on-box manifest completed at `2026-08-11T02:52:46Z`: target
  `d8d71c7e...742d517`, draft `b445fbb1...9f3581`, and prompt `6e00d762...db86b2` all match
  their frozen values. The source remains exact `c58ebd62`; Step remained healthy and idle.
- 2026-08-11: First gate attempt retained as a protocol receipt: `kernel-check` itself exited zero
  with `ALL GREEN`, but did not emit `DUAL-BATCHED-AUX`, so the wrapper correctly stopped before
  `run-gen`/`run-spec`. At `c58ebd62` that cell sits under the historical 9B filename resolver even
  though it consumes any real NVFP4 gate/up pair. The repaired protocol gives that resolver an
  explicit symlink to the already-hashed 27B target; it does not alter or duplicate model bytes or
  change the tested binary. The complete battery will be rerun from the start.
- 2026-08-11: Repaired full battery passed at `2026-08-11T03:00:26Z`. `kernel-check` emitted
  `DUAL-BATCHED-AUX ... bit-bad=0/0 OK` and `ALL GREEN`; `run-gen` emitted both required `MATCH`
  lines; `run-spec` emitted eight K=1..8 self-consistency PASS lines. Production Step and soak
  remained resident throughout and were healthy/idle with zero Xid warnings afterward.
- 2026-08-11: Completed the one-lock interleaved K=3, NGEN=64 A/B in the frozen order
  `A,B,B,A,A,B,B,A,A,B`, with A=`MEMRA_NVFP4_AUX_DUAL=0` and B=the naked default-on path.
  All ten runs produced the same token hash, the same 42/63 (66.7%) acceptance, and
  self-consistency PASS. Median A was 163.88 tok/s; median B was 163.84 tok/s, a -0.0244%
  difference: flat under the lane verdict rule, not a regression and not a revert trigger.
- 2026-08-11: The 500 ms thermal trace contains 130 samples per GPU. Test GPU 0 stayed at
  45--57 C (88.57--438.33 W); resident-production GPU 1 stayed at 39--44 C and 0% utilization.
  The production Step PID remained resident. The A/B trap restarted soak as PID 16815, verified
  `/health`, `/readyz`, `/models`, and a streamed completion, and recorded zero Xid warnings at
  `2026-08-11T03:02:44Z`.
- 2026-08-11: Independent final handoff at `2026-08-11T03:06:38Z` revalidated all three frozen
  artifact hashes, healthy/ready Step with zero Xids, a completed streamed request, live soak PID
  16815, and a fresh successful soak row. Final verdict: PASS, keep the default, no revert.
- 2026-08-11: Wrote the final verdict, reconciled the originating 27B tuning report, and verified
  all 180 retained raw files against `raw/SHA256SUMS`.
