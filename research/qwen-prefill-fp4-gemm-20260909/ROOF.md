# INT8 roof and first exact-kernel checkpoint

The W4A8 kernel is about half the measured K16 instruction roof. Six smaller-tile variants are bitwise exact at both requested chunk-1024 shapes, but every variant is slower. No engine integration or FP8 numerical arm is justified by this cell.

## Roof: published family peak and measured instruction form

NVIDIA's RTX Blackwell architecture whitepaper Table 3 lists the RTX 5090 at **838 dense INT8 TOP/s**, **1676 sparse INT8 TOP/s**, and **2407 MHz boost**. The dense figure is the applicable published INT8 tensor peak. The benchmark explicitly issues the INT32-accumulating `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32`; it also measures K32 as an instruction-rate control. [NVIDIA whitepaper, printed pp. 46-48](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

A register-only loop on the authorized RTX 5090 desktop measures:

| Form | Grid | Independent accumulators | Median TOP/s, N=5 | Min-max TOP/s |
| --- | ---: | ---: | ---: | ---: |
| INT8 K16, INT32 accumulate | 170 | 4 | **515.116** | 515.065-515.170 |
| INT8 K16, INT32 accumulate | 680 | 8 | 513.937 | 513.923-513.978 |
| INT8 K32, INT32 accumulate | 680 | 4 | **1023.860** | 1023.794-1023.945 |

Each sample contains 20 launches, 256 threads/CTA, 65536 iterations, and a runtime-checked integer output. The inner loop has no global or shared memory traffic. Only final output/cycle stores occur after the loop. CUDA-event timing includes that small epilogue and the launch envelope. NACC 1/2/4/8 and grids 170/340/680 are all retained in `int8-roof/summary.json`; 170/4 is the fastest median K16 configuration, not the fastest individual sample. K16 medians with >=2 accumulators span 512.025-515.116 TOP/s. The loop's CTA clock count per warp-MMA is a contention-inclusive interval, not an isolated instruction latency.

The approximately 10-second run has 41 telemetry samples at 250 ms. Temperature spans 33-48 C and median sampled SM clock is 2955 MHz. Clocks are automatic, not locked. The K32 observation above the published 838 figure occurs at clocks above the whitepaper's 2407 MHz boost reference. Do not mistake the sparse 1676 figure for the dense roof or assign the K32 rate to K16. The initial shorter roof run is retained separately because its telemetry was too sparse; its K16 median maximum was 517.287 TOP/s.

For the prior profile's **252 TOP/s**, the fractions are:

- 252 / 838 = **30.07% of the published dense family peak**, a **69.93% gap**.
- 252 / 515.116 = **48.92% of the achieved K16 instruction roof**, a **51.08% gap**.
- The current-window GEMM control below runs 239.713 / 237.777 TOP/s, or **46.54% / 46.16%** of that K16 roof.

These are throughput gaps, not promises that software can recover all of them. The register-only loop does not pay quantization, weight unpack, shared-memory movement or the required FP32 scale folds. The observed current-window fraction is close to the owner's approximate 45% investigation threshold; the kernel is not near its K16 roof. Step 2's FP8 numerical arm is therefore not built.

K32 has twice the arithmetic rate in this control. It is not a valid one-line substitution: NVFP4 has different scales for adjacent 16-element weight groups. The current source applies those scales to separate exact INT32 dots before folding them into the running FP32 output. Read `research/prefill-gemm-20260806/VERDICT.md` and `research/w4a8-prefill-20260806/VERDICT.md` for that arithmetic constraint. A split-K candidate would likewise have to preserve the original per-output FP32 fold order; reducing independent FP32 partial sums is not an exact split-K implementation.

## First exact-kernel cell

One binary, unchanged W4A8 integer-dot and FP32 scale-fold source body, synthetic normal activations, seeded finite NVFP4 weights, production split-plane RP=1, output macro-scale 0.37. All arms include the existing q8 activation quantizer. M=1024. Six interleaved samples per arm, twenty calls/sample, alternating arm order; warmup and raw samples are retained. The first benchmark's 252-254 TOP/s is not reused as this cell's denominator.

| Kernel tile M x N | Pipeline | Max CTAs/SM | Gate/up tok/s, K=5120 N=17408 | Down tok/s, K=17408 N=5120 | Bit mismatches, each shape |
| --- | :---: | ---: | ---: | ---: | --- |
| Current 128 x 128 | yes | 1 | **1,344,754** | **1,333,893** | 0 / 0 |
| 64 x 64 | yes | 2 | 1,174,471 | 1,119,914 | 0 / 0 |
| 32 x 64 | yes | 2 | 921,773 | 885,775 | 0 / 0 |
| 64 x 32 | yes | 2 | 1,002,832 | 990,806 | 0 / 0 |
| 128 x 64 | yes | 1 | 1,290,613 | 1,242,351 | 0 / 0 |
| 64 x 128 | yes | 1 | 1,199,479 | 1,145,559 | 0 / 0 |
| 64 x 64 | no | 2 | 1,048,852 | 999,218 | 0 / 0 |

Each arm compares every output bit: **17,825,792 gate/up outputs and 5,242,880 down outputs**, 23,068,672 total. All six candidates have zero mismatches, zero nonfinite values, max absolute deviation zero and relative L2 zero. This is the first GEMM oracle cell, not the full model or tail-shape gate battery.

The intended two-CTA 64x64 candidate changes shared memory **98,816 -> 49,408 bytes** and resource queries confirm **1 -> 2 maximum CTAs/SM**. Its throughput is **12.66% / 16.04% lower** than current. The best alternative, 128x64, is still **4.03% / 6.86% lower**. No candidate spills; registers/thread are 182-255, with the current kernel at 254 and 64x64 at 248. Higher theoretical residency did not turn into a throughput win. This independently remeasures the occupancy lever on the desktop, rather than importing the laptop verdict.

The tile cell lasts about three seconds, 32-50 C across its telemetry. All timings are direct GPU-event GEMM measurements with automatic clocks, not full-model tok/s or serving TTFT. The current median latency is 0.761478 / 0.767678 ms; the 64x64 candidate is 0.871882 / 0.914356 ms.

## Decision and next investigation

Bank the tile sweep as exact but negative. Preserve current 128x128 dispatch. No new door or naked-default change lands from this cell. The measured half-roof gap keeps same-program instruction scheduling, unpack and pipeline engineering open; the next useful diagnostic is a phase/stall profile of the current desktop kernel to identify which of those costs owns the gap. A tile-loss verdict does not establish a hardware wall.

FP4 activations remain refused as documented in RESULTS.md; the losing fused quantizer is closed. No model gates were spent on those known numerical refusals. No FP8 arm was built, because the INT8 near-roof prerequisite is not met. No integration has occurred at this requested steering checkpoint.

Builds are direct nvcc on sm_120a with `RUSTC_WRAPPER=`; no Cargo/sccache or local-rig gates ran. `build-int8.sh`, source, raw JSONL, failed compile log (research launcher name conflicted with the included tile type), corrected compile log, SASS and symbol inventory are banked. The exact source base remains memra `182819614874be818ff8bef0cfaf13cb3b8051e1`; binary hashes are in `int8-build-evidence.txt`. The job took and released the shared GPU lock, with empty compute-list preflight and postflight. Owned-PID readback lives in the private companion. This was the roof checkpoint; the final bounded iteration and closure are in ATTRIBUTION.md.
