# B200 sm_100a HBM-speed decode matvecs

Status: kernels, dispatch, doors, docs and bench written; both arches build clean
(`MEMRA_CUDA_ARCH=100a` and `120a`), `cargo fmt --all -- --check`, `tools/check-flags.sh` and
`cargo clippy --release --all-targets -- -D warnings` all green. **NO GPU ON THIS BOX** — every
throughput number below is either the nsys census that motivated the lane or a static build
receipt (ptxas registers, SASS load bursts). Nothing here is a measurement of the new kernels.
Both doors default OFF and stay OFF until the B200 A/B lands.

Branch: `lane/b200-gemv-hbm-20260902`, from `lane/glm5-b200-int2-20260902` (which carries the
per-device cuBLASLt fix and the doors `MEMRA_B200_MATVEC_ARM`, `MEMRA_KDA_FUSED_PROJ`,
`MEMRA_GLM5_W8`; this lane reads their code and does not change their behaviour).

Owner target (2026-09-02): 230 tok/s plain decode on a 2x B200 pair, against 44.9 today with
every occupancy door on. This lane's slice of that is bytes per second INSIDE each launch; the
launch count is a separate (graph) lane.

## 1. The roofline, per family

nsys, 2x B200 SXM (183 GB, 8 TB/s HBM3e, 148 SMs, 228 KB smem/SM), sm_100a build,
GLM-5.3-Flash NVFP4 W4A16 mint, resident PP2, plain decode t=1, both devices summed, **with
every current door ON**:

| kernel | us/launch | bytes/launch | achieved | % of 8 TB/s | 60% target |
|---|---:|---:|---:|---:|---:|
| `qmatvec_kda6_bf16f32` | 93.8 | ~197 MB (3 x [8192,4096] bf16 = 192 MB + ~5 MB f32) | 2.10 TB/s | 26% | 41.1 us |
| `moe_gate_up_preclamp8_q8_w4` | 52.4 | 56.6 MB (8 experts x 1536 rows x 2304 B x gate+up) | 1.08 TB/s | 13% | 11.8 us |
| `moe_down8_fma_q8_w4` | 28.2 | 28.3 MB (8 experts x 4096 rows x 864 B) | 1.00 TB/s | 13% | 5.9 us |
| `matvec_bf16_f32acc_x4_rows_pf` | 23.6 | 64 MB ([8192,4096] bf16) | 2.71 TB/s | 34% | 13.3 us |
| `qmatvec_nvfp4_mmvq_mr2_rp` | 12.2 | (rp singles) | — | — | — |
| `qmatvec_nvfp4_mmvq_fused2_rp` | 9.9 | (rp fused pair) | — | — | — |

On the RTX PRO 6000 (1.8 TB/s GDDR7, 188 SMs) these same kernels sit near their DRAM wall. On
B200 they are 3 to 9x off it. The previous lane's warp-packing and prefetch arms
(`MEMRA_B200_MATVEC_ARM`) bought ~5%, which is the evidence that **the residual is not
block-slot occupancy**.

### The in-flight arithmetic

Little's law at 8 TB/s against a ~700 ns HBM3e round trip:

    B_inflight = 8e12 B/s * 700e-9 s = 5.6 MB across the die = 37.8 KB per SM, continuously.

The shipped kernels can reach that at their PEAK and still miss it on AVERAGE, because they
spend a large fraction of each block's life in a phase that has NO loads outstanding:

* `matvec_bf16_f32acc_x4_rows` walks its four rows SEQUENTIALLY, each ending in a full
  `red[]` tree with a `__syncthreads` per step. At `mmv_block()=128` that is
  **4 x 7 = 28 barriers per block** against 4 x 4 = 16 K-loop iterations, and per block it
  moves 4 x 4096 x 2 = 32 KB of weight: **1.17 KB of DRAM traffic per barrier**.
* it also re-reads the f32 activation ONCE PER ROW: 4 x 16 KB of activation against 32 KB of
  weight, a 2:1 activation:weight ratio on a kernel whose whole job is streaming weights.
* `moe_down8_fma_q8` walks its 8 experts SEQUENTIALLY inside ONE warp, so the launch is
  `out_f` warps wide — 4096 warps for the GLM-5.3 down shape against 148 SMs x 64 warp slots =
  9472, i.e. **0.43 of a single full-occupancy wave**, with each expert's round trip serialized
  behind the last.
* the NVFP4 expert dots hold one 36-byte group's loads per row per iteration, and at in_f=4096
  a lane runs only nsb/32 = 4 iterations of that loop.

So the design target for v2 is: more independent loads in flight per thread, activations loaded
once and reused across the rows that share them, far fewer barriers per byte moved, and grids
that cover the die.

## 2. Step 1 (shipped first, on its own commit): the cuBLASLt REFERENCE door

`MEMRA_B200_BF16_GEMV_LT=1`, commit `ede09242`. Routes `matvec_bf16_rows_into` (the entry the
bf16 KDA/MLA decode projections reach) through `cublasLtMatmul` at m=t, reusing the
already-linked `memra_bf16_pp_gemm` TN plan from `cu/f16_prefill.cu` with its PER-DEVICE handle
(a `cublasLtHandle_t` created with another device current returned
`CUBLAS_STATUS_EXECUTION_FAILED` on this pair). No new CUDA kernel, no new dependency.

It exists to answer one question before any rewrite is judged: **what does a tuned vendor GEMV
reach on these bytes on this part?** That bounds what "well-scheduled" looks like on sm_100a and
says whether the remaining gap is a scheduling gap or a hardware one.

NUMERIC CLASS `bf16_gemv_lt`, and it is not a bit-identical twin: the activation is cast
f32 -> bf16 before the multiply (the shipped kernel keeps the f32 activation and widens only the
bf16 weight), and the K summation order is the library's. It is therefore an INSTRUMENT — default
OFF, sm_100a builds only, never a serving default, and promotion would need its own argmax and
serving acceptance. `MEMRA_KDA_FUSED_PROJ`'s bf16 arm declines while the door is on (its
bit-identity bar is against the unrerouted `matvec_bf16_f32acc_x4_rows`), which is also what puts
the three KDA bf16 projections through the library GEMV. Engagement and decline are each
announced once per shape.

**Bench this first.** The bench prints an `LT-ref` line per bf16 family with us, GB/s and the max
abs diff against the shipped kernel.

## 3. Step 2: `matvec_bf16_v2` and the fused kda6 twin

`MEMRA_B200_GEMV_V2=1`. Design, and why each piece is there:

* **8 rows per block, accumulated CONCURRENTLY** (8 independent `acc` registers) instead of four
  sequential rows. Per block: 8 x 4096 x 2 = 64 KB of weight against ONE 16 KB activation read =
  **1:4 activation:weight**, where the shipped kernel is 2:1. Three times fewer load
  instructions per weight byte.
* **Two-stage software pipeline.** Stage B's 8 weight loads plus its 2 activation loads issue
  BEFORE stage A's 8 x 8 fma chains run. SASS receipt (nvcc 13.1, `-gencode
  arch=compute_100a,code=sm_100a -O3`): the steady-state loop issues **10 consecutive
  `LDG.E.128.CONSTANT`** — 8 weight rows + 2 activation `float4` = **160 B per thread** —
  before the first `FFMA` consumes one. `.CONSTANT` is the `ld.global.nc` path.
* **Grid.** out_f=8192 -> 1024 CTAs, out_f=4096 -> 512 CTAs, both well past 2 x 148 = 296, so
  ksplit stays 1 (see below).
* **Barriers.** The 8 rows' reductions run in LOCKSTEP: the block pays ONE barrier chain
  (7 barriers at blockDim=128) for 64 KB of weight = **9.4 KB per barrier**, an 8x improvement
  on the shipped kernel's 1.17 KB. The tail (s = 16, 8, 4, 2, 1) becomes `__shfl_down_sync`,
  which pairs the SAME lanes in the SAME order as the smem tree — taken only when `mmv_block()`
  is a power of two, because the shipped `s >>= 1` walk lands on 16 only then; other block sizes
  keep the smem loop verbatim.
* `__launch_bounds__(256)`.

**Bit-identity.** For a given row, thread `tid` accumulates exactly the shipped kernel's subset
(`i = tid*8`, `+stride`, ...) in exactly the shipped order, with the same eight `acc +=` fma
expressions on the same operands, and the reduction replays the shipped tree step for step. Only
the ISSUE order of loads belonging to DIFFERENT rows changes, and the 8 rows never share an
accumulator. `mmv_block()` is passed through unchanged because the tree's shape — and therefore
its bits — is a function of blockDim.

`qmatvec_kda6_bf16f32_v2` applies the same walk inside the fused six-projection kernel: the
three BF16 ranges take 8 rows/block, the three f32 ranges keep `f32_mmvq_row1` verbatim, the six
ranges keep their block order. Bit-identical per row to `qmatvec_kda6_bf16f32`, which is itself
bit-identical per row to `matvec_bf16_f32acc_x4_rows`.

### Split-K, and why the shipped shapes never take it

`matvec_bf16_v2_sk` + `matvec_bf16_v2_sk_combine`. NAMED NUMERIC CLASS **`bf16_gemv_v2_splitk`**:
a row's K sum is split into `ksplit` contiguous chunks (multiples of 8 elements so 16 B loads
stay aligned), each reduced independently, and the chunk partials are added in a FIXED ASCENDING
order — never atomics, never a scheduling-dependent reduction — so the class is deterministic but
not bit-identical.

`Engine::gemv_v2_ksplit(in_f, out_f, t)` returns 1, i.e. the BIT-IDENTICAL kernel, whenever the
row grid already covers two waves of CTAs over `sm_count()`; otherwise the smallest split that
does, capped so every chunk still gives each thread one full 8-element step. The GLM-5.3 KDA
decode shapes give 512 and 1024 CTAs against 2 x 148 = 296, so **the dispatch is bit-identical
for every shape this lane exists for**; the class is reachable only for narrower shapes. The
launcher recomputes the EFFECTIVE split from the chunk size, so a plane the kernel never wrote
can never be summed by the combine. The bench prints the chosen `ksplit` and the bit-identity
verdict per shape, so the class cannot engage silently.

## 4. Step 3: the NVFP4 W4A16 expert pair

The 36-byte NVFP4 block layout **forbids wider loads**: a row base plus `sblk*36 + 4 + s*8` is
only 4 B aligned, so `uint2`/`uint4` are illegal here and the four `get_int_b4` reads per group
are not a missed vectorization. Depth and grid are the levers.

* `moe_gate_up_preclamp8_q8_v2`: 8 warps per block on `threadIdx.y` (the shipped kernel is one
  warp per block; the `_w4` arm is four), and the `g` walk unrolled by two so BOTH groups'
  weight, scale and activation loads issue before either dp4a chain runs.
  Bit-identical per (o, j): `accg += dot(g); accu += dot(g); accg += dot(g+32); accu += dot(g+32)`
  is the shipped per-accumulator order, unrolled. Epilogue verbatim.
* `moe_down8_fma_q8_v2`: **ONE BLOCK per output row, warp `j` owning expert slot `j`.** That
  takes the launch from `out_f` warps wide (4096, 0.43 of a full-occupancy wave) to
  `out_f * n_used` (32768, 3.5 waves), with the eight experts' bytes in flight together. Still
  bit-identical: each expert's partial is the shipped per-expert g-strided chain plus the same
  `warp_reduce_sum`, and the final `chain = __fmaf_rn(w.v[k], part[k], chain)` runs on ONE
  thread in the SAME ascending slot order. Parallelising the experts moved no bits because the
  slot chain was never the parallel part.

## 5. Static build receipts (sm_100a, nvcc 13.1, `-O3`)

`ptxas -v` and `cuobjdump -sass`. "LDG in flight" = the longest run of `LDG` instructions with no
float consumer (`FFMA`/`FADD`/`FMUL`/`IDP`/`MUFU`) between them.

| kernel | registers | spill | LDG in flight |
|---|---:|---:|---:|
| `matvec_bf16_f32acc_x4_rows` (shipped) | 38 | 0 | 4 |
| `matvec_bf16_f32acc_x4_rows_pf` (prev lane) | 40 | 0 | 6 |
| **`matvec_bf16_v2`** | 64 | 4 B | **10** (all `LDG.E.128` = 160 B/thread) |
| `matvec_bf16_v2_sk` | 100 | 0 | 10 |
| `qmatvec_kda6_bf16f32` (shipped) | 40 | 0 | 4 |
| **`qmatvec_kda6_bf16f32_v2`** | 96 | 0 | **16** |
| `moe_gate_up_preclamp8_q8_w4` | 40 | 0 | 18 |
| **`moe_gate_up_preclamp8_q8_v2`** | 56 | 0 | **19** (603 LDG total vs 201) |
| `moe_down8_fma_q8_w4` | 40 | 0 | 12 |
| **`moe_down8_fma_q8_v2`** | 40 | 0 | **19** (303 LDG total vs 101) |

Per-SM in-flight arithmetic for `matvec_bf16_v2`: 64 registers at blockDim=128 admits
65536/64 = 1024 threads/SM (dynamic smem is 4 KB/block, 228/4 = 57 blocks, not binding), so
1024 x 160 B = **160 KB per SM outstanding**, 4.2x the 37.8 KB Little's-law floor, 23.7 MB
across the die. That is the ceiling, not the average; the barrier-per-byte improvement above is
what is meant to raise the average toward it.

**The gate/up static burst barely moved (18 -> 19)** — say it plainly. Its shipped `sl` loop was
already issuing both sub-block halves together, so that arm's lever is the 8-warps-per-block
packing and twice the independent dot chains per iteration, not deeper bursts. Whether that is
enough is exactly what the box has to say.

## 6. Bench

```
MEMRA_GPU_LOCK=/tmp/memra-gpu.lock \
  cargo run -p memra-engine --release --bin b200_matvec_bench -- 5 3
```

(`5` = N iterations, median; `3` = distinct device weight copies so no arm is served a warm L2
another arm paid for.) Every arm is called DIRECTLY by name through `_raw`/`_arm_raw` Engine
entries, bypassing the env doors — both doors memoize into a process-wide `OnceLock` and cannot
flip mid-process. Per family it prints per-launch us, **GB/s = bytes moved / kernel time**, and
either `bit-identical` or `MISMATCH n=... max_abs_diff=...` (never a silent pass).

Families, in print order:

1. `moe_gate_up_preclamp8_q8` — shipped vs `_w4` vs **v2** (n_embd 4096, ff 1536, 8 experts, NVFP4)
2. `moe_down8_fma_q8` — shipped vs `_w4` vs **v2** (1536 -> 4096, 8 experts)
3. `matvec_bf16_f32acc_x4_rows` — shipped vs `_pf` vs **v2 (with the chosen ksplit printed)** vs
   **LT-reference**, at [4096 -> 8192] and [8192 -> 4096]
4. `qmatvec_nvfp4_mmvq_mr2_rp` — shipped vs grid-fill (previous lane)
5. `qmatvec_nvfp4_mmvq_fused2_rp` — shipped vs `_g2` (previous lane)
6. **`qmatvec_kda6_bf16f32`** — shipped vs **v2**, at the census's hottest shape: in_f 4096,
   dims [8192, 8192, 8192, 128, 128, 64], 192 MB bf16 + 5.2 MB f32

The bf16 GB/s denominators are weight bytes only (`in_f * out_f * 2`); the kda6 line prints its
bf16 and f32 byte split so the sum is auditable.

## 7. Doors and rollback

| door | default | class | rollback |
|---|---|---|---|
| `MEMRA_B200_BF16_GEMV_LT` | off, sm_100a builds only | `bf16_gemv_lt` (activation cast to bf16 + library summation order) — REFERENCE INSTRUMENT, never a serving default | unset or `=0` |
| `MEMRA_B200_GEMV_V2` | off, sm_100a builds only | bit-identical per output for every dispatched arm; `bf16_gemv_v2_splitk` only for sub-2-wave shapes | unset or `=0` |

Both are strict per-process `OnceLock` reads with no persistent state, so unset leaves every call
site byte-identical to pre-lane behaviour. Both are restricted to `sm_100a` BUILDS via
`MEMRA_BUILT_CUDA_ARCH` (baked in at compile time), so setting either on an sm_120a build is a
documented no-op and the naked sm_120a defaults stay byte-identical — the per-hardware arm
selection law. `MEMRA_B200_GEMV_V2` takes precedence over `MEMRA_B200_MATVEC_ARM` on the MoE pair
(v2 subsumes its packing) and over the LT door on the bf16 rows.

FLAGS.md rows and KERNELS.md rows land in the same commits as their reads and their `.cu`.

## 8. Open items (owner-visible, not silently dropped)

1. **The B200 A/B itself.** No number in this lane is a measurement of the new kernels. The
   session that owns the pair runs the bench above under `MEMRA_GPU_LOCK` between its A/B
   rounds; the doors' defaults move only on that receipt, and a serving decision additionally
   needs the vendor-default sampled twin per the never-serve-greedy law.
2. **Whether 60% of HBM is reached is unknown.** The static receipts say the loads are issued and
   the barriers are gone; they cannot say what the memory system does with them. If v2 lands
   short, the next levers in order are: (a) a persistent-CTA form so a block streams many row
   tiles without relaunch, (b) `cp.async.bulk` / TMA for the weight tiles on sm_100a, (c) 2-CTA
   clusters sharing the activation through distributed shared memory.
3. **The 36 B NVFP4 block layout is the binding constraint on the expert kernels** and this lane
   did not touch it. A 16 B-aligned split-plane repack (the layout the `_rp` family already uses)
   would unlock 128-bit weight loads there. That is an artifact/format change with its own
   qualification, not a kernel lane.
4. **The gate/up arm's static burst barely moved** (item in section 5). If the box shows it flat,
   the honest read is that the shipped `sl` unroll already had the depth and the remaining gap is
   the 4-byte load granularity of item 3, not scheduling.
5. **`bf16_gemv_v2_splitk` is unexercised by any shipped shape.** It is built, deterministic and
   benchable, but the GLM-5.3 decode shapes all pick ksplit=1. It needs its own acceptance
   (kernel-check tolerance, not a bit tape) before anything routes through it deliberately.
6. **The NVFP4 rp singles/fused pair** (12.2 and 9.9 us in the census) keeps the previous lane's
   grid-fill arms; this lane added nothing there.
7. **Three clippy defects on the base branch were fixed in-lane, not disclaimed**:
   `hyper_ffn_branch`'s argument count (`hybrid_forward.rs`), two doc/type-complexity lints in
   `hc_fused_gate.rs`, a `manual_is_multiple_of` in `q8_fuse_gate.rs`, and a collapsible `if` in
   `memra-server/src/worker.rs`. `cargo clippy --release --all-targets -- -D warnings` is green
   on this branch; it was red on the base.
