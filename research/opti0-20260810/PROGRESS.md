# optipipe increment 0 progress

Started: 2026-08-10T21:34:34Z  
Lane: `lane/cx-opti0`  
Rig: box1, 2x RTX PRO 6000 Server Edition
Completed: 2026-08-10T22:51:41Z

## Contract

- Split PP speculative verify at the existing stage-0 boundary TX.
- Keep the ordinary serial caller as immediate stage0-then-stage1 issue.
- Release experimental session B stage0 at session A's ticket point.
- Measure actual `A.S1 || B.S0` execution, not enqueue intent.
- Stop before increment 1 if the seam does not produce real overlap.

## Progress

- [x] Read `CLAUDE.md`, optipipe `DESIGN.md`, and specmech `RESULTS.md` in full.
- [x] Confirm current CUDA event re-record/wait semantics against NVIDIA documentation.
- [x] Confirm the dedicated branch/worktree is clean.
- [x] Map the as-built verify body and two-session coordinator.
- [x] Implement the split with a back-to-back serial wrapper.
- [x] Rewire the experimental coordinator and add phase timestamps.
- [x] Prove serial one-hash identity on 10 fresh boots.
- [x] Run `run-spec` K=1..8 and `kernel-check`.
- [x] Prove PIPE=1 spec/plain byte identity.
- [x] Capture N=5 interleaved c=2 seam/plain/old-serial measurements.
- [x] Write the final result and increment-1 GO/NO-GO.

## Notes

- `/home/avifenesh/.lanectl/inbox/cx-opti0.md` did not exist at lane start; recheck at every bounded work block.
- Serial seam checkpoint: `cargo check -p memra-engine` passed, including the sm_120a nvcc build.
- TX-release checkpoint: B stage 0 now waits on A's boundary ticket; B stage 1 still waits on A's full verify issue. Focused engine check passed.
- Trace checkpoint: `MEMRA_SPEC_PIPE_TRACE=1` emits stream-ordered S0+TX and S1+head edges on one monotonic host clock. Trace callbacks are absent by default. Engine check and 54/54 runnable unit tests passed.
- Box1 exactness: serial fresh-boot golden 10/10; plain/serial/PIPE all hash `21b8293f...`; run-spec PASS 8/8; kernel-check ALL GREEN.
- Box1 argmax: prefill/decode and batched-prime/tokenwise both MATCH at 6776.
- Timeline verdict: overlap real YES, 10.429 ms median across 220/220 rounds; 19-round pair interval 48.662 -> 36.746 ms (-24.49%).
- c=2 N=5: seam 63.082 tok/s, serial 55.365 (+13.94%), plain 121.051 (-47.89% seam vs plain).
- Increment 1: GO for fork + reconcile only; promotion/default remains HOLD/OFF.
- No push, tag, `cargo fmt`, `rustup`, or `nsys` in this lane.
