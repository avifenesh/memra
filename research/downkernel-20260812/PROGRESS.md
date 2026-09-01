# cx-downkernel progress — 2026-08-12

## Scope

Study the live Step B=1 `moe_down8_fma_dev_q8_rows_g` launch on box1, name the
occupancy/tile limiter from NCU receipts, and attempt only a low-risk,
bit-identical launch-geometry or tiling arm. Numeric-order or dtype changes are
out of scope.

## Gates

- [x] Read the frozen ncuspike baseline and current kernel/dispatch.
- [x] Capture the live launch geometry and NCU occupancy-limit receipts.
- [x] Record a mechanism verdict before changing code.
- [x] If a safe arm exists, prove baseline/candidate bit identity before timing.
- [x] If bit-identical, run N=8 ABBA interleaved Step semantic timing.
- [x] Record GO/NO-GO and prepare the complete study lane commit.

## Log

- Started from clean `lane/cx-downkernel` at `79c3c0b27`.
- Created this progress ledger before all research-lane edits, per lane contract.
- No orchestrator steering was present when the lane inbox was checked.
- The frozen ncuspike receipt reports 40 launches/token, 0.9045 ms/token,
  36.83% card BW, 44.91% achieved occupancy, and 0.91 waves/SM.
- Reproduced the exact Step B=1 shape on idle box1 from an isolated
  `f0ab104e7` worktree: IQ4_XS `in_f=1280`, `out_f=4096`, top-8 of 288,
  `grid=(4096,1,1)`, `block=(32,1,1)`. Metric-scoped NCU 2026.1 took three
  replay passes at stock clocks.
- **Mechanism verdict before runtime edits:** the 32-thread CTA is limited by
  the hardware block cap, not registers or shared memory. NCU reports block
  limits SM=24, registers=40, shared-memory=32, and warps=48; therefore only 24
  of 48 warps/SM can be resident (50% theoretical). The finite grid supplies
  21.51 achieved warps/SM (44.82%) and 0.91 waves/SM. Long-scoreboard remains
  18.99 warps/issue and DRAM reaches 688.54 GB/s in the replay.
- Low-risk arm selected: assign one warp to each of the eight expert slots for
  a row, store the eight independently reduced dots in shared memory, and let
  warp 0 lane 0 replay the original slot-ordered `__fmaf_rn` chain. Per-slot
  group assignment/order and the 32-lane reduction tree stay unchanged.
- Implemented `moe_down8_fma_dev_q8_rows_w8` and gated it only on the exact
  Step B=1 shape (`t=1`, 1280x4096, top-8, IQ4_XS). Every other rows shape
  retains the baseline kernel.
- Hard gate passed before timing: all 4,096 output f32 values are byte-identical,
  both dumps hash to `3c509060d071f171e0bd54ac9d0e29f98411fa5d4367b47d1b54480a0e2eccce`,
  and `cmp` returned zero.
- N=8/arm ABBAx4 over 40 physically distinct layer banks (128 token sweeps per
  sample) measured 0.8591155 -> 0.7888570 ms/token-equivalent median: -8.178%
  time / +8.906% throughput. Complete process trace: stock clocks, 26-42 C.
- Candidate NCU confirms the intended movement: theoretical occupancy 50.00 ->
  83.33%, achieved 44.82 -> 73.79%, waves/SM 0.91 -> 4.36, and DRAM 688.54
  GB/s -> 1.02 TB/s. Candidate resource use is 48 registers/thread, 32 B smem,
  one barrier, and zero spills.
- The actual production PP-2 `run-gen` binary was captured launching the new
  `(4096,1,1)x(32,8,1)` symbol; its NCU receipt reports 74.85% achieved
  occupancy and 4.36 waves/SM.
- Production correctness passed on box1: kernel-check `ALL GREEN (88 cells, 21
  skipped)`, both run-gen argmax comparisons MATCH, and run-spec K=1..8
  self-consistency PASS.
- Final verdict: **GO for orchestrator promotion**. No merge, tag, push, format,
  Nsys capture, perf-board update, or release action was run.
