# B200 sm_100a HBM-speed decode matvecs

Status: **round 1 benched on the box; round 2 (bug fix + verdicts acted on + v3) written and
pending its receipt.** Both arches build clean (`MEMRA_CUDA_ARCH=100a` and `120a`),
`cargo fmt --all -- --check`, `tools/check-flags.sh` and
`cargo clippy --release --all-targets -- -D warnings` all green. No GPU on this box; section 8
carries the box's own numbers and section 9 the v3 arithmetic, which is design, not measurement.
Both doors default OFF.

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
either `bit-identical` or `MISMATCH n=... max_abs_diff=...` (never a silent pass), and it
PANICS on any non-finite output rather than printing `inf` (see section 8.4).

Families, in print order:

1. `moe_gate_up_preclamp8_q8` — shipped vs `_w4` vs **v2** (n_embd 4096, ff 1536, 8 experts, NVFP4)
2. `moe_down8_fma_q8` — shipped vs `_w4` vs **v2** (1536 -> 4096, 8 experts)
3. `matvec_bf16_f32acc_x4_rows` — shipped vs `_pf` vs **v2 (with the chosen ksplit printed)** vs
   **v3 (cp.async staged)** vs **LT-reference**, at [4096 -> 8192] and [8192 -> 4096]
4. `qmatvec_nvfp4_mmvq_mr2_rp` — shipped vs grid-fill (previous lane)
5. `qmatvec_nvfp4_mmvq_fused2_rp` — shipped vs `_g2` (previous lane)
6. **`qmatvec_kda6_bf16f32`** — shipped vs **v2** vs **v3**, at the census's hottest shape:
   in_f 4096, dims [8192, 8192, 8192, 128, 128, 64], 192 MB bf16 + 5.2 MB f32, with
   PER-PROJECTION mismatch counts (`y0=.. .. y5=..`) on both arms

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

## 8. Round 1: the box receipt, and what it changed

`b200_matvec_bench 5 3`, B200 dev 0, int2 `18df26ad6`, 2026-09-02. Medians over N=5 interleaved,
shipped -> `MEMRA_B200_MATVEC_ARM` -> v2:

| family | shipped | arm | v2 | v2 vs shipped | identity |
|---|---:|---:|---:|---|---|
| `matvec_bf16` kda up 4096->8192 | 33.4 | 28.5 | **23.7** (2.83 TB/s) | **1.409x** | bit-identical |
| `matvec_bf16` kda down 8192->4096 | 30.65 | 24.9 | **24.3** | **1.26x** | bit-identical |
| `qmatvec_kda6_bf16f32` | 100.8 | — | **65.0** (3.18 TB/s) | **1.55x** | MISMATCH n=64 |
| `moe_gate_up_preclamp8_q8` | 55.0 | 53.3 | 54.4 | 1.011x | bit-identical |
| `moe_down8_fma_q8` | 37.6 | 36.0 | 43.2 | **0.870x** | bit-identical |
| `qmatvec_nvfp4_mmvq_mr2_rp` (prev lane) | 20.0 | 18.0 | — | 1.11x | — |
| `qmatvec_nvfp4_mmvq_fused2_rp` (prev lane) | 23.2 | 22.0 | — | 1.05x | — |
| `bf16_gemv_lt` reference | — | — | 32.3 (both shapes) | 1.03x / **0.95x** | class, max_abs_diff=inf |

Against section 1's roofline: kda6 went 26% -> **40%** of the 8 TB/s wall, the bf16 rows
34% -> **35%** at 8192 out and 34% -> **~40%** at 8192 in. Real, and short of 60%.

### 8.1 The kda6 mismatch was a real bug, and it was mine

`MISMATCH n=64 max_abs_diff=2.442e2`, and the serving A/B was correctly aborted — a mismatching
kernel never enters a serving arm.

Cause: the first cut of `qmatvec_kda6_bf16f32_v2` **dropped the `b -= nb` before the LAST of the
six ranges.** The block index therefore still carried range 4's offset when it reached range 5,
every `b * R + p` landed past `out5`, `f32_mmvq_row1` returned for all of them, and y5 was NEVER
WRITTEN. The numbers name the bug exactly: n=64 is `out5` on the bench dims, and 2.442e2 is
`max |y5|` against an output buffer left at zeros. The other five ranges were bit-identical.

Why nothing caught it: range 5 is the only one of the six with no `if (b < nb)` guard behind it,
so an off-by-a-range index has nowhere to fail loudly. Fixed, with the reasoning in the kernel
comment. **The bench now prints PER-PROJECTION mismatch counts** (`y0=.. y1=.. ... y5=..`) for
both the v2 and v3 arms, because a single summed count turned a location into one number.

### 8.2 The MoE v2 twins are measured and NOT dispatched

- gate/up: 54.4 vs `_w4`'s 53.3 — inside noise, no gain. Its own static receipt predicted this
  (section 5: the longest LDG burst moved only 18 -> 19, because the shipped `sl` unroll already
  had the depth). Called in section 5 before the box ran, and the box agreed.
- down: 43.2 vs `_w4`'s 36.0 — a **20% REGRESSION**, bit-identical. The one-block-per-row /
  warp-per-expert form multiplied the launch width by 8 and got slower: eight warps in a block
  now contend for the same L2 sectors, and the shipped single warp was already covering the
  latency it stood accused of exposing. The 0.43-of-a-wave framing in section 1 was a correct
  count and the wrong diagnosis.

`moe_fused_epi_launch` now keeps the shipped/`_w4` pair unconditionally and the door no longer
overrides `MEMRA_B200_MATVEC_ARM` there. The two v2 kernels stay in the fatbin and in the bench
as measured arms, because a refuted arm with a number on it is worth more than a deleted one.
`MEMRA_B200_GEMV_V2` is a bf16-row and kda6 door now.

### 8.3 cuBLASLt at m=1 is not a ceiling — the reference door did its job

32.3 us on BOTH bf16 shapes: 1.03x the shipped kernel at 4096->8192 and **0.95x, a loss**, at
8192->4096, against v2's 23.7 / 24.3. The library has no answer for an m=1 GEMV of this shape.

That is exactly the question the door was built to answer, and the answer retires a whole class
of doubt: the decode matvec gap is not "memra is badly scheduled versus a tuned GEMM library".
`MEMRA_B200_BF16_GEMV_LT` therefore stays what it was declared as — an instrument, default OFF,
no promotion path.

### 8.4 `max_abs_diff=inf` was the BENCH, not the compare

The LT line reported `inf`. The compare was fine; the synthetic weights were not. The old
`synth_bf16` randomised the whole 16-bit pattern and cleared only one exponent bit, so bf16
magnitudes reached ~2^127 and a 4096-term dot OVERFLOWED f32. That is invisible while both arms
run the same program (`inf == inf` compares bit-identical, which is why shipped-vs-`_pf` looked
clean) and becomes `inf` the instant an arm sums in a different order.

Fixed two ways: `synth_bf16` now samples the same [-2, 2) range as the activations and truncates
f32 -> bf16, so a full dot is bounded by 16384; and `compare` **panics** on a non-finite value
rather than printing `inf`, because a bench that cannot produce finite numbers must say so
loudly. Round 1's bit-identity verdicts still stand (identical programs), but they were weaker
evidence than they looked.

## 9. Round 2: v3, the cp.async-staged walk (`MEMRA_B200_GEMV_V2=2`)

v2 is at 40% of the wall and its in-flight budget is **register-bound**:

    v2 kda6: 96 registers -> 65536/96 = 682 threads/SM -> 5 CTAs of 128 = 640 threads,
             each holding 10 x 16 B = 160 B          =>  ~102 KB per SM outstanding.

Of the three levers section 8 of the first cut named, only one moves that number. The arithmetic,
done before choosing:

| lever | effect on in-flight bytes/SM | verdict |
|---|---|---|
| persistent CTA | **unchanged** (~102 KB) — same loop, fewer launches | attacks launch count, which is the graph lane's axis, not this one |
| 2-CTA cluster sharing the activation via DSMEM | **unchanged**; saves activation traffic only, and the activation is already just 1/4 of v2's bytes | ceiling ~12% of load instructions |
| **`cp.async` / TMA staging into shared memory** | **~192 KB (1.9x)** — the budget stops being register-bound | taken |

v3 stages the weight tiles global -> shared with `__pipeline_memcpy_async` (16 B per thread per
row) instead of holding them in registers: 2 stages x 8 rows x (blockDim.x * 8) elements x 2 B =
**32 KB in flight per CTA**, plus the 4 KB reduction window = 36 KB of dynamic smem, so
228 / 36 = **6 CTAs/SM => ~192 KB per SM outstanding**, and the register count falls to 58
(`matvec_bf16_v3`) / 64 (`qmatvec_kda6_bf16f32_v3`) so registers stop binding at all. 36 KB also
stays under the 48 KB default dynamic-smem cap, so no `cudaFuncSetAttribute` opt-in is needed —
at `mmv_block()=256` it would be 72 KB, and the launcher declines to v2 there rather than opting
in for a door still pending its receipt.

`cp.async` rather than `cp.async.bulk`/TMA: identical in-flight arithmetic (both land bytes in
smem without occupying registers), a fraction of the machinery, and no mbarrier protocol to get
wrong on a part this lane cannot run. TMA is the follow-up if v3's shape is right and its issue
rate becomes the next wall.

**It adds no barriers**, which is load-bearing and is why v3 keeps v2's 9.4 KB-per-barrier
property while doubling the bytes outstanding: thread `tid` issues the copy for
`dst + p*kch + tid*8` and later reads THAT SAME address, for every one of the 8 rows. Each thread
consumes only bytes it copied itself, `__pipeline_commit`/`__pipeline_wait_prior` are per-thread,
and reissuing a slot overwrites only the issuing thread's own lane after that thread has read it
in program order. No `__syncthreads` between stages.

**Still bit-identical.** The K chunk is PINNED to `blockDim.x * 8` — exactly the shipped
per-thread stride — so chunk `c` hands thread `tid` exactly one index `i = c*kch + tid*8`, and
walking chunks ascending reproduces the shipped `i` sequence element for element. Any other chunk
size would reorder a row's accumulation and cost the identity; it is not a tuning knob.

SASS receipt (sm_100a, nvcc 13.1, -O3): `matvec_bf16_v3` = 24 `LDGSTS` (8 rows x 3 issue sites)
+ 2 `LDG` (the activation `float4` pair) + 57 `LDS`, 58 registers, 0 spill.
`qmatvec_kda6_bf16f32_v3` = 72 `LDGSTS` + 96 `LDG` (its three f32 ranges) + 153 `LDS`, 64
registers, 0 spill.

The door became a LEVEL rather than a boolean: `1` = v2, `2` = v2 plus v3 wherever it fits, any
other value off (a typo must not arm a kernel arm). v3 declines per call — not per process — when
a shape wants split-K or when its smem does not fit.

## 10. Open items (owner-visible, not silently dropped)

1. **v3 has no receipt.** Section 9 is in-flight-bytes arithmetic and static SASS, not
   throughput. The box runs `b200_matvec_bench 5 3` again: the bf16 and kda6 families now print
   a `v3` line with us, GB/s and per-projection identity next to the shipped and v2 lines.
   Nothing about level 2 moves without it.
2. **If v3 also lands short of 60%**, the remaining ladder in order is: `cp.async.bulk`/TMA (same
   arithmetic, higher issue rate and no per-thread address math), then 2-CTA clusters on top of
   staging, then attacking the reduction epilogue itself (v2/v3 still spend 7 barriers per block
   on it).
3. **The 36 B NVFP4 block layout remains the binding constraint on the expert kernels**, and
   round 1 confirmed it from the other side: both MoE v2 arms were bit-identical and neither was
   faster, so the gap there is the 4-byte load granularity, not scheduling. A 16 B-aligned
   split-plane repack (the layout the `_rp` family already uses) would unlock 128-bit weight
   loads. That is an artifact/format change with its own qualification, not a kernel lane.
4. **`bf16_gemv_v2_splitk` is still unexercised by any shipped shape** and still needs its own
   acceptance (kernel-check tolerance, not a bit tape) before anything routes through it. v3 has
   no split-K twin by design.
5. **The serving A/B was aborted in round 1** by the kda6 mismatch and has not run. With the bug
   fixed it needs the full battery, including the vendor-default sampled twin per the
   never-serve-greedy law, before any default moves.
6. **The NVFP4 rp singles/fused pair** keeps the previous lane's grid-fill arms (1.11x / 1.05x on
   this run); this lane added nothing there.
7. **Clippy defects that were red on the base branch were fixed in-lane, not disclaimed**:
   `hyper_ffn_branch`'s argument count (`hybrid_forward.rs`), two lints in `hc_fused_gate.rs`, a
   `manual_is_multiple_of` in `q8_fuse_gate.rs`, and a collapsible `if` in
   `memra-server/src/worker.rs`.
