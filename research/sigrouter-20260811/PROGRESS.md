# Device-side sigmoid router progress

Started: 2026-08-11T04:38:38+03:00  
Lane: `lane/cx-sigrouter`  
Base: `e03d758f`

## Objective

Replace Step-3.7's per-MoE-layer host-synchronous sigmoid routing with a device-side
sigmoid/bias/top-k/normalization path. Preserve the host oracle exactly, including active-expert
masking and tie order, and leave `MEMRA_SIG_ROUTER=0` as the rollback seam.

## Gates

- Device selection must be bit-identical and weights exact against `moe_route_sigmoid_host` for
  ties, optional correction bias, `route_norm` on/off, active masks, `t=1..64`, and Step's
  `n_expert=288`, `n_used=8` shape.
- Step-3.7 PP-2: ten fresh-boot one-hash goldens, `run-gen` argmax match, and `run-spec` K=1..8.
- One-lock-hold interleaved x5 rollback/default comparison at c=1 and c=8, plus a receipt
  characterizing the eliminated full-logit transfers and the remaining increment-1 readback sync.
  Zero-D2H/device-resident dispatch is increment 2 and does not block this lane.
- All local and remote GPU work runs under the designated `flock`; raw logs live in `raw/`.
- Do not run `cargo fmt`.

## Status

- Read the lane brief, SOL-gap report, and repository law.
- Confirmed this dedicated worktree is clean and on `lane/cx-sigrouter` at the stated base.
- Traced `moe_route_cfg`, `moe_route_sigmoid_host`, the existing softmax
  `moe_router_topk[_host]` kernel/API, and all production sigmoid call sites.

## Frozen host-oracle contract

For each token row, the current oracle:

1. Computes `score[i] = 1 / (1 + exp(-logit[i]))` in f32.
2. Removes inactive original expert ids before ranking.
3. Ranks retained ids by `(score[i] + optional_bias[i])` descending. Exact equal ranking scores
   break toward the smaller original expert id (`total_cmp`, then ascending id).
4. Takes the first `n_used`; output weights come from the un-biased `score[i]` values in that
   selected order.
5. With `route_norm`, sums those weights in slot order, clamps the denominator to `1e-20`, then
   evaluates `weight / sum * scaling_factor`; otherwise it evaluates
   `weight * scaling_factor`.

The device kernel must preserve original ids across masking and perform the final slot-order sum
and arithmetic on one thread. Bias and active-mask rows will be uploaded once with each MoE layer,
not allocated or copied in the token loop.

## Implementation checkpoint

- Added the device sigmoid/bias/mask/top-k/normalization kernel and the single-sync pinned
  `[sel,w]` readback API.
- Production Step/M3/Hy3 grouped and staged routing now uses it by default;
  `MEMRA_SIG_ROUTER=0` restores the full-logit host oracle. CPU-expert routing keeps its paired
  host readback unchanged.
- Added a `kernel-check` cell covering 68 inputs: every Step batch `t=1..64` at 288 experts/top-8,
  32- and 256-expert controls, and two frozen Step-width adversarial cases for representable
  sigmoid plateaus plus exact/adjacent correction-bias keys. The general guarantee is selection
  and weight-bit exactness on this corpus, not universal host/device `expf` equivalence.
- First local 5090 run: selection, masking, and tie order were exact; CUDA `expf` left 79 weight
  mismatches at at most 2 ULP. Raw receipt: `raw/local-kernel-check-fast.log`. This is RED under
  the lane's weight-exact bar; the gate was not weakened.
- The double-exp-then-f32-round variant passed on the local 5090: all 66 sigmoid-router cases have
  zero id, mask, tie, and weight-bit mismatches (`max_weight_ulp=0`), and the fast synthetic
  battery ends `ALL GREEN`. Raw receipt: `raw/local-kernel-check-fast-double-exp.log`.
- Deployed the checkpointed branch to the fresh box1 clone
  `/home/ubuntu/memra-cx-sigrouter`. Release builds of `kernel-check`, `run-gen`, `run-spec`, and
  `memra-server` are green; raw build receipts are being retained on box1.
- Added bounded one-lock correctness and performance harnesses. The initial 15-minute fast-gate
  lock wait expired while `cx-opti2` remained active; it did not run any GPU work. The full
  correctness battery acquired the idle box1 lock at `2026-08-11T02:43:50Z` and is running from
  source commit `61749925`.
- Box1's full kernel battery is green, including 66/66 exact sigmoid-router cases, and Step-3.7
  PP-2 passed both `run-gen` argmax checks plus `run-spec` K=1..8. The first fresh golden boot was
  RED: default produced `91c89c65...`; a same-binary rollback control reproduced the required
  `21b8293f...` hash.
- Route tracing localized the cause: the first difference is one weight ULP at prefill layer 12;
  selection diverges only after that error propagates. The double-exp cast therefore does not
  reproduce glibc `expf` for all served logits. A CUDA adaptation of Arm Optimized Routines'
  MIT-licensed scalar `expf` evaluation now compiles locally.
- The expanded local 5090 gate is green: 68/68 cases, including the two adversarial boundary
  cases, have zero id, mask, tie, or weight-bit mismatches and the fast synthetic battery ends
  `ALL GREEN`. Receipts: `raw/local-build-kernel-check-arm-expf-adversarial.log` and
  `raw/local-kernel-check-fast-arm-expf.log`.
- Deployed source commit `62b0d629` to box1 and rebuilt `kernel-check`, `run-gen`, `run-spec`, and
  `memra-server` with CUDA 13.2. The first two noninteractive invocations exposed only missing
  PATH and package-target mistakes; both failure receipts are retained beside the green build.
- Box1 acquired `/tmp/memra-gpu.lock` at `2026-08-11T03:16:45Z`. The full kernel battery is green
  (68/68 sigmoid-router cases, zero mismatches, `ALL GREEN`), both `run-gen` argmax gates match,
  and `run-spec` K=1..8 is self-consistent. All 10 independent server boots reproduce the pinned
  `21b8293f...` golden hash; the battery ends `CORRECTNESS_PASS` at `03:25:44Z`. Raw receipt:
  `raw/box1-correctness-arm-expf/`.
- The interleaved x5 c1/c8 performance and Nsight receipt harness is queued behind `cx-opti2`'s
  two-GPU server under the shared box1 lock; no performance point has started yet.
- The performance harness acquired the box1 lock at `2026-08-11T03:33:08Z`. Repetitions 1 and 2
  completed with zero request errors and provisional default wins at both c1 and c8; repetitions
  3..5 completed the same way. The fixed N=5 reducer reports 5/5 paired wins at both loads:
  c1 median 84.4304 vs 83.2441 tok/s (+1.4251%), c8 median 162.9924 vs 156.3440 tok/s
  (+4.2524%). Both Nsight traces completed and the harness ended `PERF_PASS` at
  `2026-08-11T03:48:02Z`.
- Local trace audit confirms the intended increment-1 mechanism: full-router DtoH payload falls
  94.44% (1,152 B -> two 32 B decode copies; 104,832 B -> two 2,912 B prefill copies), while the
  single readback synchronization remains (9,511 calls in both arms). Primary `.nsys-rep` hashes
  match their box1 manifests. Final verdict is ADOPT in `RESULTS.md`; zero-D2H remains increment 2.

## Complete

All scoped increment-1 implementation, exactness, performance, trace, and evidence-retention work
is complete. `RESULTS.md` contains the ADOPT verdict and bounds the deferred zero-D2H increment.
