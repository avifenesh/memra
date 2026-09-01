# memra on sm_100a (B200): the block-scale gap and the tcgen05 lane (opened 2026-08-15)

## Finding (B200 block, ptxas CUDA 13.2)

`MEMRA_CUDA_ARCH=100a` builds fail in ptxas: `Instruction 'mma with block scale' not supported on
.target 'sm_100a'`. The whole sm_120a block-scale MMA family
(`mma.sync.aligned...kind::mxf8f6f4.block_scale.scale_vec::1X`, ue8m0 identity trick, NVFP4 W4A8
tiles) is consumer-Blackwell-only. Datacenter Blackwell does block-scaled matmul through
**tcgen05.mma** (5th-gen tensor core: tensor-memory accumulators, smem descriptors, mbarrier
completion) — a different programming model, not a syntax swap.

## Affected kernel census (grep block_scale, 2026-08-15, v0.83.0)

| file | sites |
|---|---|
| cu/mmq_fp4.cu | 14 |
| cu/mmq_fp8_blk.cu | 6 |
| cu/mmq_nvfp4_w4a8.cu | 5 |
| cu/mmq_q8_0_f32acc.cu | 4 |
| cu/qmatvec_gemm.cu | 2 |
| cu/flash_attn.cu, hybrid.cu, mmq_nvfp4_f8f4.cu, moe_f16_grouped.cu, qmatvec.cu | 1 each |

Non-block-scale kernels (int8 m16n8k16 mma, dp4a, f16 paths) are expected to compile; the first
ptxas failure aborts the build so the full pass/fail census needs a per-file sweep (TODO on box).

## Lane plan

1. Research brief (agent in flight): tcgen05.mma block-scale kinds (`mxf8f6f4` / `mxf4` /
   `mxf4nvf4` — NVFP4's e4m3 scales need the nvf4 kind), tmem alloc/ld/st, smem + instruction
   descriptors, mbarrier completion, minimal single-CTA skeleton, CUDA 13.2 ptxas status.
2. Prototype cell on this box (cards 6-7, untimed): smallest compiling tcgen05 block-scale GEMM,
   bit-checked against the existing sm_120a oracle outputs (host-side reference), receipts to
   `raw/b200/memra-sm100/`.
3. Full twin = own lane after the block (week-class): per-kernel tcgen05 twins behind the arch
   split, sm_120a naked build byte-identical, gates per CONTRIBUTING (kernel-check, run-gen,
   run-spec) on a B200 rig.

## Diagnostic (NOT the deliverable — format-shortcut rule honored)

A `MEMRA_CUDA_ARCH=90a` (Hopper) build PTX-JITs forward onto sm_100 via driver JIT. It runs the
HOPPER program (no block-scale instructions), so it proves loader/runtime/driver plumbing on B200
and gives Q-format correctness smoke — it is NOT sm_100 support and sets no default. Building now
for kernel-check on card 6.

## Cross-check note

The build.rs arch whitelist already names `100a (B200)` — the arch flag existed before any kernel
did; whoever added it never got past ptxas. This lane is the first real sm_100a compile attempt
on record (receipts: raw/b200/memra-bins.log).

## Diagnostic closed (2026-08-15 ~09:0xZ)

The 90a-PTX-JIT idea is dead: memra embeds SASS-only fatbins (no PTX section), so the driver
reports CUDA_ERROR_NO_BINARY_FOR_GPU on cc 10.0 even with MEMRA_ARCH_CHECK=0. Receipts:
raw/b200/memra-sm100/kernel-check-90aJIT-q38mint.log. Two findings for the lane:
1. B200 has no diagnostic shortcut at all — first light requires the real sm_100a compile.
2. The arch guard works as designed (clean refusal + named bypass), and the bypass then fails
   safely at module load. Guard behavior worth keeping exactly as is.

## Research brief landed (2026-08-15, agent-verified to ptxas 13.1 assemble level)

Full brief in session log; load-bearing facts:
- sm_120a and sm_100a block-scale ISAs are DISJOINT (tcgen05 rejected on sm_120a, mma.block_scale
  rejected on sm_100a — both verified). Numerics identical; execution model completely different.
- NVFP4 exact program on sm_100a = `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X`
  + idesc bit23=UE4M3; e2m1 packed 2-per-byte in smem (checkpoint bytes usable as-is).
- W4A8 (fp4 weights x fp8 acts) only exists under kind::mxf8f6f4 + ue8m0 + padded 8-bit containers —
  same restriction as sm_120a, format arithmetic ports 1:1.
- MIN TILE M=128 for all block-scaled kinds (no m16 equivalents): batch-1 decode wastes 127/128 of
  the datapath -> tcgen05 block-scale shines at prefill/batch; small-M fallback = non-scaled f8f6f4
  M=64 (DIFFERENT numeric program — gate it as such per one-program doctrine).
- Programming model: tmem alloc (32-col units, mandatory dealloc), 64-bit smem descriptors
  (bits46-48=001 marker; K-major+128B swizzle recommended, canonical-layout table is the oracle),
  32-bit idesc (M>>7 at bits27-28, scale type bit23, SFA/SFB ids), scales MUST live in tmem
  duplicated x4 lane-quarters (`tcgen05.cp.32x128b.warpx4` = one-instruction dup),
  completion = tcgen05.commit -> mbarrier + tcgen05.fence::after_thread_sync before ld,
  fence.proxy.async after generic smem writes (silent-corruption trap #1).
- Verified minimal single-CTA skeleton assembles clean (SASS: UTCQMMA/UTCOMMA/UTCCP/UTCBAR);
  preserved at /tmp/tcgen05_test/bs_min.ptx on the dev rig.
- ptxas CANNOT check idesc shape/type (runtime register) — kernel-check oracle gates are the only
  defense; plan bit-oracle first, perf later.
- References for the lane: PTX ISA §9.7.17 (all subsections), CUTLASS mma_sm100_desc.hpp
  (read-only), gau-nernst tcgen05 tutorial, danielvegamyhre mxfp8 kernel, Colfax part 4.

## sm_100a FP8-leader build (2026-08-15 ~09:4xZ) — GREEN

Native `MEMRA_CUDA_ARCH=100a` build passes with three build.rs deltas (box clone; to be landed on a
proper memra branch): (1) NVFP4 W4A8 stub extended `portable || 100a` (owner call: NVFP4 is not a
B200 quant — no block-scale HW; fail-closed ABI preserved), (2) mmq_fp8_blk.cu compiled with
-DMEMRA_FP8BLK_PLAIN_MMA (plain kind::f8f6f4 arm, bit-identical at ue8m0 identity per TU header
audit), (3) mmq_q8_0_f32acc.cu with -DMEMRA_ACCPROBE_PLAIN_MMA. Perf note: plain form = 32.03
cyc/warp-MMA vs block_scale 16.06 — the tcgen05 lane is the perf recovery, correctness ships first.

## PROTOTYPE FIRST LIGHT (2026-08-15 ~11:4xZ, B200 card 7) — 1024/1024 EXACT

`tools/tcgen05_proto.cu`: D[128x8](f32) = A[128x32](e4m3) x B[8x32](e4m3) via
`tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X` at ue8m0 identity scales,
verified element-exact (1e-4 rel tol; accumulation-order LSBs) against a CPU e4m3 oracle.
Full pipeline: tmem alloc -> core-tiled smem staging -> fence.proxy.async -> tcgen05.cp.warpx4
(SF dup) -> single-thread MMA -> commit/mbarrier -> fence::after_thread_sync -> per-warp
tcgen05.ld -> dealloc.

Hard-won facts beyond the brief (the debugging ladder, ~90 min):
1. nvcc needs explicit `-gencode arch=compute_100a,code=sm_100a` — plain `-arch=sm_100a` lowers
   PTX to compute_100 and ptxas rejects every tcgen05 form.
2. `tcgen05.commit` takes NO state-space qualifier on the mbarrier operand (`.b64 [mbar]`);
   `mbarrier.try_wait` REQUIRES `.shared::cta`. Asymmetric — both learned from ptxas.
3. **Operands must be staged CORE-TILED in smem**: hardware fixes each core matrix as 8 rows x
   16 bytes CONTIGUOUS (128B); LBO/SBO stride only BETWEEN cores. Plain row-major + clever
   strides cannot express within-core row stride — this was the entire numerics bug.
   Layout: smem[(r/8)*GROUP + (kc/16)*128 + (r%8)*16 + (kc%16)]; LBO=128 (between k-cores),
   SBO=256 (between 8-row groups, = 2 k-cores x 128B at K=32).
4. `tcgen05.cp.32x128b.warpx4` writes a 4-COLUMN tmem footprint — SF tensors need 4-col spacing
   (SFA at col d+8, SFB at col d+12; overlap = misaligned-address fault).
5. Faulting tcgen05 ops poison the CUDA context (sticky) — parameter sweeps need
   process-per-combo.
Next rungs: real ue8m0/e4m3 scale vectors (non-identity), K-loop accumulation (enable-input-d
chaining), M=128 x N=256 tile with TMA staging, then the mmq_fp8_blk twin behind the arch seam.

## RUNG 2 (2026-08-15 ~12:3xZ, B200 card 7) — REAL SCALES, 1024/1024 EXACT

`tools/tcgen05_proto2.cu`: same MMA with REAL (non-identity) ue8m0 scale vectors, exponents
2^-3..2^3 on both SFA (128 distinct) and SFB (8 distinct). Verified element-exact against the
CPU oracle including the scale product.

The SF tmem lane mapping, found empirically (two failing layouts pinned it):
- SF staging = a single 32-row x 16B smem tile per SF tensor, `tcgen05.cp.32x128b.warpx4`
  (desc sdesc(addr, 16, 128)). The dup gives every lane-quarter the SAME 32 rows.
- Distinct per-row scales still work because **quarter q of the MMA reads byte-column q*4** of
  its lane's 16B row: scale for row m = 32q+l goes at smem byte `l*16 + q*4` (+ SFA_ID for the
  sub-byte column select). This is CUTLASS's canonical SF atom (Sm1xx blockscaled chunk,
  Stride<_16,_4>) observed from the hardware side.
- Refuted layouts, same failure signature (rows 0-31 exact, rows 32-127 read scale byte 0):
  (a) core-tiled 128-row x 16B tile + `tcgen05.cp.128x128b` with scale at byte 0 of each row;
  (b) warpx4 tile with quarter stride 1 (byte q instead of 4q). Quarter 0 always reads byte 0 —
  identity-scale tests (rung 1) can never distinguish these mappings; real scales were required.
- SFB at N=8 sits entirely in quarter-row bytes `n*16 + 0` (all n < 32, q=0).

Remaining rungs: K-loop accumulation (enable-input-d chaining), M=128 x N=256 with TMA staging +
timing vs plain-MMA (the 4%-of-peak recovery), then the mmq_fp8_blk tcgen05 twin behind the arch seam.

## RUNG 3 (2026-08-15 ~12:5xZ) — K-LOOP CHAINING, 1024/1024 EXACT, first try

`tools/tcgen05_proto3.cu`: K=128 as 4 blocks of 32 with PER-BLOCK scales — the exact mmq_fp8_blk
shape. 4 chained `tcgen05.mma` ops on one accumulator, enable-input-d predicate false only for
kb=0 (`setp.ne.u32 p, kb, 0`), single commit at the end. Facts confirmed: same-CTA tcgen05 async
ops (cp and mma) execute in issue order without intermediate commits; 64-col tmem alloc works;
per-block SF tiles live in separate 4-col tmem slots (SFA at +8+4kb, SFB at +24+4kb).

## RUNG 4 (2026-08-15 ~13:0xZ) — FULL-WIDTH TILE M=128 x N=256, 32768/32768 EXACT, first try

`tools/tcgen05_proto4.cu`: N=256 (idesc N>>3=32), K=128 x4 blocks, per-block scales both sides.
512-col tmem alloc (D cols 0-255, SFA 256+4kb, SFB 272+8kb). SFB at N=256 = TWO warpx4 tiles per
block (rows 0-127 / 128-255), each cp'd to its own 4-col slot. Readback chunked: 32 x
`tcgen05.ld.32x32b.x8` per warp-quarter.

## RUNG 5 (2026-08-15 ~13:1xZ) — THROUGHPUT: THE PERF-RECOVERY NUMBER

`tools/tcgen05_bench5.cu`, B200 card 7 (no compute co-tenant on card per nvidia-smi at run time;
arms training on other cards — thermal/PCIe co-tenancy caveat applies, direction-grade), CUDA
13.2, 148 CTAs (1/SM), N=3 reps each arm, receipts
`/scratch/receipts/memra-sm100/bench5-tcgen05-vs-plain.log`:

| arm | median | % of 4500 TF dense-fp8 peak |
|---|---|---|
| tcgen05 mxf8f6f4 block_scale mainloop (M=128,N=256,K=32/instr, 128/commit) | 4103.9 TFLOP/s | 91.2% |
| plain mma.sync m16n8k32 kind::f8f6f4 (the shipped -DMEMRA_FP8BLK_PLAIN_MMA arm), 16 warps, pure-ALU | 1061.2 TFLOP/s | 23.6% |

**3.87x at instruction level** — and the plain arm's 23.6% is its register-only ceiling (real
kernels sit below it under memory traffic), while tcgen05 at 91% includes smem descriptor reads.
This is the recovery the twin lane ships. Also learned: 128 uncommitted MMAs per commit/wait is a
working cadence; spread stable to 0.1% across reps.

Remaining: TMA gmem->smem staging, then the mmq_fp8_blk tcgen05 twin behind the arch seam
(week-class own lane per plan).

## RUNG 6 (2026-08-15 ~13:3xZ) — TMA STAGING, 32768/32768 EXACT, first try. LADDER COMPLETE.

`tools/tcgen05_proto6.cu`: A and B arrive via `cp.async.bulk.tensor.2d` (TMA) with
`mbarrier.arrive.expect_tx` byte counting, feeding the rung-4 MMA chain unchanged.

The load-bearing trick — **no swizzle needed**: make the k-core the OUTER smem dimension.
A {16B x rows} TMA box (interleave NONE, swizzle NONE) writes its box dense, fastest-dim
contiguous — which IS the 8x16B core tiling within a k-core slab. Layout
`(c/16)*SLAB + (r/8)*128 + (r%8)*16 + (c%16)` (SLAB = rows*16), descs express it as
LBO = slab stride (2048 for A@128 rows, 4096 for B@256), SBO = 128. Per-K-block desc base =
`+ kb*2*SLAB`. 8 A-boxes + 8 B-boxes, one mbarrier, one wait. CUTLASS's 128B-swizzle path is a
perf refinement, not a correctness requirement.

**The mmq_fp8_blk tcgen05 twin now has every ingredient proven on hardware**: tmem lifecycle,
core-tiled staging (manual + TMA), SF tmem atom, block-scale MMA, K-chaining, full 128x256 tile,
91%-of-peak issue cadence, TMA in, tcgen05.ld out. Twin implementation = engineering, not research.

## TWIN v1 GATED + LOOP-SHAPE ECONOMICS (2026-08-15 ~14:xxZ, B200 card 7, co-tenant caveat)

Engine twin (memra lane/sm100a-fp8-bringup, cu/mmq_fp8_blk_tcgen05.cu): fp8-mmq-check ALL GREEN
(EXACT arm bit-identical incl. 5120x1536 m=512 after the token-tile offset fix; RAND < 1e-5 RMS;
254/254 codes). BUT end-to-end fp8_mmq_bench: twin 116-187 TF vs plain arm 130-228 TF — the 3.8x
ISA win is fully eaten. Receipts: /scratch/receipts/memra-sm100/.

Loop-shape probes (tools/tcgen05_bench6*.cu, N=128 tile, per-128k fold contract):
| shape | TF |
|---|---|
| v1: per-block MMA x4 -> commit -> wait -> ld+fold | 1027 |
| v1 with the fold's ld+ALU REMOVED (sync/wait only) | 1041 |
| double-buffered D (fold b-1 during MMA b) | 1077 |
| 2 CTAs/SM | 1067 |
| 3 D-slices per commit (batch-3, fold x3 after one wait) | 1223 |
| commit per 16 blocks, one fold (ceiling) | 3547 |

Findings: (1) the per-block COMMIT/DRAIN round-trip is the wall — an empty fold costs the same;
(2) tcgen05.ld does NOT overlap in-flight MMAs (double-buffer flat) — plan tmem reads as serial
with the tensor pipe; (3) slice-batching amortizes the drain sublinearly (+19% for x3);
(4) the engine kernels run at ~15-20% of even the v1 loop shape — SYNCHRONOUS operand staging
dominates end-to-end, same as the plain arm ("not MMA-bound" redux, now with the right ceiling).

v2 recipe therefore: TMA double-buffered staging (prefetch iter t+1 during MMA/fold of t — bulk
DMA is a separate engine from the tensor pipe) + ITER_K=384 (3 scale blocks, 3 tmem D slices,
one commit) + per-block f32 fold unchanged (contract intact). tmem budget: 3x128 D + SF = 392 of
512 cols. Est. 4-5x the current twin end-to-end.

## TWIN v2 SHIPPED — GATES GREEN, 1.34-1.50x THE FLOOR (2026-08-15 ~15:xxZ, B200 card 7)

memra lane/sm100a-fp8-bringup: v2 rebuilds the GEMM core around the two measured walls —
ITER_K=384 (three scale blocks into three independent 128-col tmem D slices, twelve MMAs one
commit one wait three folds) + TMA double-buffered operand staging (iteration t+1's
cp.async.bulk.tensor issued before iteration t's MMAs; row/k/token tails all ride TMA OOB
zero-fill). Arithmetic contract untouched. TMA preconditions refused -> -1 -> dispatch falls
through to the mma.sync tile. cuTensorMapEncodeTiled fetched via cudaGetDriverEntryPointByVersion
(cudart only, no -lcuda — GPU-less CI unaffected; non-100a arches compile a fail-closed stub).

Gate battery (receipts /scratch/receipts/memra-sm100/ + raw/b200/memra-sm100/):
- fp8-mmq-check ALL GREEN — EXACT arm bit-identical every shape (incl. ragged + m=512 multi-tile),
  RAND < 1e-5 RMS, 254/254 codes. (v2's one bring-up bug — token-tile offset on the act reads —
  was caught by exactly the 5120x1536 m=512 cell fp8-mmq-check exists for.)
- kernel-check 85 cells ALL GREEN (mint gguf, card 7).
- prime-path serving gate (MEMRA_PP_ONLY + MEMRA_PP_LOGITS, 800-token prompt, 2000 twin dispatches
  confirmed in-ledger): argmax MATCH twin-vs-plain-vs-floor; logits drift twin-vs-plain
  rms 5.2e-2 = the SAME class as the accepted plain-vs-floor cross-path drift (4.6e-2), and
  twin-vs-floor (4.1e-2) is CLOSER than plain-vs-floor. NOTE: run-gen's generate path is
  decode-parity MMVQ by design — prefill-arm gates must run on the prime path (MEMRA_PP_ONLY),
  a full run-gen stream compare is vacuous for this arm.
- run-spec K=1,2,4,8 self-consistency PASS x4 (spec verify m<=9 never routes to the twin; gate run
  for battery completeness).

fp8_mmq_bench (interleaved vs Q8_0 floor per rep, median of 9, card 7, arms co-tenant on other
cards — direction-grade until a clean-box rerun):
| shape | v1 twin | plain arm | v2 twin |
|---|---|---|---|
| q_proj 5120->12288 | 0.899x | 0.935x | **1.414x** (283 TF) |
| k/v_proj 5120->1024 | 0.924x | 0.964x | **1.335x** |
| o_proj 6144->5120 | 0.911x | 1.022x | **1.406x** |
| gate/up 5120->17408 | 0.889x | 1.085x | **1.465x** (308 TF) |
| down 17408->5120 | 0.900x | 1.076x | **1.502x** |
| square 5120->5120 | 0.924x | 1.014x | **1.380x** |

Lane state: twin v2 is the sm_100a prefill default (winners are defaults), rollback
MEMRA_FP8BLK_TCGEN05=0. Pre-merge: owner call + clean-box numbers per evidence discipline.

## SERVING-LEVEL IMPACT: TTFT -23.6% (2026-08-15 ~15:3xZ, card 7)

memra-server, ST FP8 27B, 802-token prompt, max_tokens=1, temp 0, warmup + 5 reps per arm
(spread < 0.7%): twin v2 median 0.3275 s vs plain arm 0.4282 s — 1.31x faster end-to-end prefill
at the serving surface. Receipts: /scratch/receipts/memra-b200-tune/cells.jsonl (ttft-pp800-fp8st).

## DECODE WALL MAPPED: qmatvec_e4m3_blk_mmvq is I2F-BOUND (2026-08-15 ~16:xxZ, card 7)

ncu decode composition (run-gen, 400-launch window, FP8-ST 27B): qmatvec_e4m3_blk_mmvq = 72.2%
of decode time (19.3 us/launch avg). Harness ablation (tools/mmvq_harness.cu, q_proj 5120x12288,
200 reps; NOTE single projections fit B200's 126 MB L2, so absolute GB/s is L2-flattered —
RELATIVE deltas are the finding):

| variant | us | verdict |
|---|---|---|
| v0 engine verbatim | 15.5 | baseline (4.1 TB/s apparent) |
| v9 weight loads only | 6.3 | pattern ceiling |
| v6 no act side (loads+I2F removed) | 9.3 | act side = 40% of kernel |
| v7 fp8 cvt swapped for byte I2F | 22.0 | the fp8x2->half2 cvt chain is already the FAST form |
| v1 8 rows/CTA, v2 2-deep pipeline, v3 __ldcs | ~15.5-15.9 | flat (v2 +5% on down_proj only) |
| v5 smem f32 act stage | 71.3 | 128B-stride LDS = 32-way bank conflicts |
| v10 global f32 act mirror | 65.6 | f32 acts break the int8 walk's coalescing |

Rate math pins it: out_f x in_f I2F.S8 per launch (63M for q_proj) at ~32/SM/cyc = 7.9 us — the
measured 6.2 us act cost. The kernel sits ~1.5x above its ALU floor and every bit-identity-
preserving escape (pre-converted mirror, smem stage, warp-shuffle redistribution) prices out at
the same or worse rate. The remaining 1.5x needs a NUMERIC PROGRAM change (half2 dot: e4m3 x int8
products are 12-bit-mantissa, inexact in f16 — a declared contract change with its own oracle,
not a smuggle) or a hardware path with free conversion. Negative-result map banked per research
doctrine; harness committed for the 5090 re-sweep (per-hardware rule: I2F rate may differ).

### 5090 re-sweep (local rig, /tmp/memra-5090.lock held, same harness) — per-hardware split confirmed

q_proj 5120x12288: v0 23.7us, v6 (no act side) 20.6us -> act side = 13% (vs B200's 40%);
v7/v10 lose the same way. down_proj 17408x5120: v0 = v1 = v2 = v3 = v6 = ~105us (849 GB/s) —
on the 5090 the big shapes are purely memory-bound and EVERY variant is flat; the kernel is at
its structural floor there. Verdict: e4m3-blk MMVQ has no shared-fix headroom — B200's only
escape is conversion reduction (a declared numeric-program change, bounded ~1.5x), the 5090 has
none. Two-rig evidence per the per-hardware doctrine; no default changes.

## OWNER NOTE: the "NVFP4 is not a B200 quant" premise is now refuted

That call (2026-08-15, and the build.rs stub comment) was conditioned on "no block-scale HW on
B200". The ladder disproves the premise: tcgen05 block-scale runs at 91% of peak on this card,
and the brief's verified encoding for NVFP4 is `kind::mxf4nvf4.block_scale.scale_vec::4X` +
UE4M3 idesc — checkpoint e2m1 bytes usable as-is. FP8 stays the B200 leader per the standing
call; this note only records that a tcgen05 NVFP4 W4A16/W4A8 twin is now a KNOWN-VIABLE
week-class item (same seam as the FP8 twin), should the mint lane want it on datacenter
Blackwell. Decision is the owner's.

## Post-merge decode regression check (new build, card 7): none

k1-hpost leader cell on the twin-v2 + fused3-QKV build: 85.0 tok/s median x3 (prior build 83.8),
acceptance byte-stable at 0.752. Receipts: memra-b200-tune/cells.jsonl (k1-hpost-tc5v2).

## RUNG 7 (2026-08-15 ~18:5xZ) — NVFP4 EXACT ON B200, 1024/1024. Owner-sanctioned lane
## ("if b200 is bw nvfp is plausible" + "we already support st nvfp", 2026-08-15)

`tools/tcgen05_proto7.cu`: D[128x8](f32) = (A[128x64](e2m1) x SFA) x (B[8x64](e2m1) x SFB),
`tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X`, REAL ue4m3 scales per
16-value block, element-exact vs a CPU oracle. The checkpoint-faithful NVFP4 program runs on
datacenter Blackwell at tcgen05 rate — the sm_100a NVFP4 stub can get a real twin.

Empirical facts (poison-probe ladder, tools/tp7_dbg2.cu — first-light was all-zeros from TWO
compounded encoding errors):
1. idesc bit23 SELECTS UE8M0 WHEN SET, everywhere. The f8f6f4 twin sets it (ue8m0 correct
   there); mxf4nvf4 with ue4m3 scales needs bit23=0. With bit23=1 our ue4m3 bytes (~0x38) read
   as 2^-71 scales — exact-zero outputs, layout-independent (this masked every earlier sweep).
2. idesc atype (bits 7-9) / btype (10-12): code 1 = e2m1 (standard LUT {0,.5,1,1.5,2,3,4,6},
   low nibble = even k). Code 0 decodes as LINEAR sign-magnitude ints 0..7 (measured exactly:
   D linear in code, sign at bit 3) — NOT a documented fp4; do not rely on it.
3. SF-4X tmem atom = the natural extension of the rung-2 1X atom: row 32q+l's FOUR ue4m3 bytes
   occupy the full quad at tile[l*16 + q*4 + j], j=0..3; ONE 32x16B warpx4 tile, ONE cp per SF
   tensor, 4-col footprint unchanged. Refuted: per-block column-quads (4 cps to +4j), j*4+q
   strides.
4. Operand geometry identical to fp8: bytes core-tiled 8x16B (two e2m1 per byte), K=64 values =
   32B/row = 2 k-cores, LBO=128/SBO=256. Checkpoint nibble packing usable as-is.
5. kind::mxf4 + scale_vec::2X faults (illegal instruction) under this geometry — parked, not
   needed for NVFP4.

## RUNG 8 (2026-08-15 ~20:2xZ) — NVFP4 THROUGHPUT: 8.5 PFLOP/s, 94% OF DENSE-FP4 PEAK

`tools/tcgen05_bench8.cu` (mainloop shape of bench5, mxf4nvf4 4X, ue4m3 identity, K=64/instr,
128 MMAs/commit, 148 CTAs, x3 reps, spread < 0.02%): 8181 TF @ N=128, 8498 TF @ N=256 —
2.07x the fp8 tcgen05 rate (4104) and 8.0x the plain mma.sync arm. B200 quant-rate ladder as
measured on this card: NVFP4 8.5 PF > FP8 4.1 PF > plain 1.06 PF. Combined with rung 7's exact
program, NVFP4 on datacenter Blackwell is not merely viable — it is the fastest quant on the
card, mirroring the sm_120a story. The NVFP4 tcgen05 twin (replacing the fail-closed stub) is a
week-class item with every ingredient proven. Receipts: raw/b200/memra-sm100/bench8-nvf4-rate.log.

## RUNG 9 (2026-08-15 ~21:2xZ) — W4A8 CONTAINER SEMANTICS UNDER kind::mxf8f6f4 (tools/tp9_dbg.cu)

atype code 5 = e2m1. Measured semantics: EACH NIBBLE of the byte is read and their products
SUMMED (0x11 -> 0.25+0.25, 0x99 -> -0.25-0.25, sign per nibble); with the byte-PADDED layout
memra's sm_120a W4A8 repack already produces (value in one nibble, other zero) this degenerates
to the single-value read. Decode table = e2m1 LUT / 2 (bias-2 sense) — a constant power of two,
exactly compensable by +1 in the ue8m0 scale exponent (identity byte 0x80 instead of 0x7F).
atype map measured on this kind: 0=e4m3, 1(=0.03125/code?), 3, 4 distinct sub-byte formats,
5=e2m1-nibble-summed, 6/7 illegal. VERDICT: the W4A8-FP8 program (fp4 weights x fp8 acts,
ue8m0, hardware block_scale) ports to tcgen05 with the SAME repacked bytes — the mmq_nvfp4_w4a8
sm_100a stub can get a real twin at the fp8-kind rate (4.1 PF class), and true W4A4 mxf4nvf4
(rung 7/8) offers the 8.5 PF class behind its own act-quant program. Prototype phase COMPLETE
(rungs 1-9); both twins are engineering, not research.

### Small-m dispatch verification (2026-08-15 ~21:5xZ, card 7, reps=5)

Twin vs plain vs floor at m=16/32/64/128 (fp8_mmq_bench): twin 1.25-1.45x the floor at EVERY m,
plain arm 0.59x. Twin latency is flat in m (0.08-0.09 ms q_proj — tile padding is nearly free),
so there is NO plain-favored crossover above the X8 tier; the m>8 twin dispatch stands. Receipts
in cells + this table.
