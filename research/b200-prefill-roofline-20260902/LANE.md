# B200 prefill roofline — GLM-5.3-Flash-NVFP4, 2x B200 PP2 (lane/b200-prefill-roofline-20260902)

Owner target 2026-09-02: TTFT under 1 s. Measured today on the pair: 66 tok 0.2 s, 4k 5.5 s
(1,266 tok/s), 41.9k 17.6 s (2,383 tok/s), 512k 789 s. Arithmetic: `roofline.py` (run it, it
prints every number quoted here). Geometry from
`research/glm53-flash-bringup-20260827/mint-receipts/nvfp4-config.json`: 45 layers = 34 KDA +
11 MLA/DSA, 3 dense MLP + 42 MoE, H 4096, 288 routed experts top-8 + 1 shared, expert FF 2048,
64 MLA heads, qk/v 256, kv_lora 512, NoPE, index_topk 2048, kpool 4, hc streams 4 / sinkhorn 20.
Posture cuts `MEMRA_PP_SPLITS=24`: stage 0 = layers 0..24 (3 dense + 21 MoE), stage 1 = 24..45
(21 MoE) + the prime tail.

## (a) Roofline for one 4k-token prime

Active params 16.06 B (MoE 9.51 + KDA 4.68 + MLA 1.37 + dense 0.45 + router 0.05). The owner's
18 B includes embed + lm_head (0.63 B each); neither is per-token prefill GEMM work.

| term | 4k FLOPs |
|---|---|
| projection + MoE GEMMs (2 x 16.06e9 x 4096) | 131.6 TFLOP |
| MLA absorbed attention, 11 layers, kv capped at topk 2048 | 12.1 TFLOP |
| absorb_q + decompress_v | 1.51 TFLOP |
| DSA indexer over t/4 pools | 0.38 TFLOP |
| KDA delta-rule scan | 0.58 TFLOP |
| **total** | **146.2 TFLOP** |

Bytes. Every one of the 288 experts is hit at any t >= 512 (miss probability e^-14 at 4k pairs),
so a MoE layer touches its WHOLE slab: 288 x 3 x 4,718,592 B repacked NVFP4 stride = 4.08 GB per
layer per prime call, 171.2 GB for all 42 layers ONE pass. Dequant is fused into the GEMM
(`moe_kq_*`, direct-from-quant), so there is no separate dequant read.

Time floors, one B200 (PP2 is strictly serial, see (b), so stage floors add rather than overlap):
bf16 TC at 2.2 PFLOPS -> **66.4 ms**; HBM at 8 TB/s, one pass -> **21.4 ms**. Arithmetic intensity
854 FLOP/B against the B200 ridge of 275: at ONE chunk the 4k prime is compute-bound with 3x
headroom. Measured 5.5 s = 26.6 TFLOP/s = **1.2 % of one card's bf16 peak**. The 41.9k prime runs
at 87.0 TFLOP/s (3.95 %) — a 3.3x efficiency gap between the two sizes that the roofline itself
does not explain, and (b) does.

## (b) The prime call graph as it actually runs at t=4096

Paths are `crates/memra-engine/src/`.

1. `hybrid_forward.rs:3485 prime_cache_overlaid` -> glm5 is an HyperConnections plan, so
   `:3499` -> `:1790 prime_cache_hyper`.
2. **Chunk schedule.** `:1261 hyper_prime_ranges` delegates to `:1180 prime_chunk_ranges` ->
   `:1080 prime_chunk_tokens`. Because `:1057 prime_pp2_auto_geometry` is TRUE under
   `MEMRA_PP_STAGES=2` (`MEMRA_PRIME_PP` defaults on, `pp.rs:774`), the chunk becomes
   `ceil(t/PRIME_PIPE_MICROBATCHES=8).max(128)` = **512 tokens**, then `:1129
   dynamic_prime_chunk_ranges` reshapes it to 8 ranges starting at 256. **A 4k prime is 8
   calls.** That geometry exists to feed `:3771 prime_cache_pp2_pipelined`, the SERIAL trunk's
   microbatched PP-2 prime. The hyper walk never calls it (`prime_cache_hyper` loops
   `prime_cache_hyper_ppn` in a plain `for`, `:1855-1866`), so glm5 pays the split and gets none of
   the pipelining. Cost: expert-slab traffic is per CALL, so 8 chunks read 1,369.8 GB instead of
   171.2 GB and pairs/expert falls 113.8 -> 14.2, below the GEMM's tile crossover (next item).
   CORRECTED BY THE BOX (§4): the roofline crossing this produces (intensity 854 -> 107 FLOP/B,
   under the 275 ridge) is real but NOT the operative cause — the kernel runs ~30x below BOTH
   floors, so the penalty is tile padding and small-m inefficiency, not bandwidth. Measured
   penalty is ~2x per token, not the 3.3x estimated here.
3. `:2715 prime_cache_hyper_ppn`, per chunk: stage 0 scope (`:2761-2790`) embeds, `hyper::expand`,
   `:2085 hyper_range_prime(0,24)`, `rt.tx`; then stage 1 scope (`:2848-2862`) `rt.rx`,
   `hyper_range_prime(24,45)`, `:3007 hyper_prime_tail`. **No overlap of any kind**: device 1 is
   idle for the whole of stage 0 and device 0 idle for the whole of stage 1, every chunk. One
   `htod_i32(pos)` per stage per chunk (`lib.rs:11785`, pageable).
4. `:2085 hyper_range_prime`, per layer:
   - `hyper.rs:350 pre` -> cuBLASLt mixes GEMM + `hyper.rs:430 pre_finish_into`
     (`rowsq_scale`, `hc_sinkhorn_m` with 20 serial iterations per token, `hc_collapse`; or one
     fused launch under `MEMRA_HC_FUSED_PRE`), `rms_norm`. 2 hc sites per layer -> ~8 launches.
   - mixer: KDA `kda.rs:742 kda_prime_cached` -> `kda.rs:269 kda_core`: qkv `matmul_group`,
     conv+silu, two low-rank gate pairs, `memra_kda_scan_s128` (`kda.rs:1156`, grid
     (heads 64, 1, 32) = 2048 blocks, **serial T-loop of t steps inside the kernel** — the
     chunked UT twin is not the shipped path), gated RMSNorm, `wo`. ~12 launches/layer.
     MLA `:8118 mla_attn_cached` -> `:8155 mla_attn_cached_inner` -> `:8203 mla_attn_cached_pre_wo` -> `:7535 mla_attn_core_pre_wo`: wq_a, rms, wq_b, split, wkv_a, split, rms, append,
     `mla_kpool_select`, then the `MEMRA_MLA_TC_PREFILL` door at `:7666` ->
     `:7715 mla_tc_prefill_chain` (strided-batched bf16 cuBLASLt absorb/decompress +
     `fa_mla_gathered_bf16`). ~14 launches/layer.
   - `hyper.rs:658 post`, then `:1680 hyper_ffn_branch` -> `:8537 moe_ffn_il` -> `:9712` ->
     `:14598 moe_ffn_grouped_prefill_sigmoid`.
5. `:14598 moe_ffn_grouped_prefill_sigmoid`, per MoE layer — **the only host synchronization on
   the whole glm5 prime path**:
   - `:9852 moe_router_logits` (GEMM) then `:12551 moe_route_sigmoid_cfg` ->
     `lib.rs:6585 moe_router_sigmoid_topk_host`: device top-k kernel, 2 `memcpy_dtoh` into the
     pinned router stage, then **`lib.rs:6623 stream().synchronize()`**. That is the
     `router=sigmoid-host-oracle` in the log line at `:14884`. 42 hard stream drains per chunk,
     **336 per 4k prime**.
   - host bucket sort into 288 `Vec<i32>` and the CSR build (`:14719-14748`), pure host time
     inside the drained window.
   - 5 pageable `htod`/`htod_i32` uploads per layer (`exi`, `exo`, `exp_d`, `csr_tok_d`, `pw`;
     `:14762-14766`), plus 2 more when the bank carries macros. The three big ones are n_pairs-sized: 3 x 16 KB per
     layer at the current 512-token chunk, 3 x 128 KB at one chunk of 4096; **1,680 uploads
     per 4k prime** (5 x 42 x 8).
   - `moe_f16g_act`, then 3 x `mmq_ffi.rs:2381 moe_f16_grouped` -> `memra_moe_kq_gemm_sk`
     (`cu/moe_f16_grouped.cu:1525`). Grid is capped at `sms * occ` persistent CTAs
     (`:1610`, `:1625`); the form split is `m_e >= MEMRA_F16G_SK_CROSS` (default **64**, swept on
     5090 and H100 only, `lib.rs:184`) -> `moe_kq_sk128v_kernel` (`:1260`, 256 thr, 72 KB
     dynamic smem) else `moe_kq_sktail_kernel` (`:1417`, 128 thr, 43.5 KB static). Every mma is
     `mma.sync.aligned.m16n8k16.f32.f16.f16.f32` (`:425`) — **the sm_80-portable warp MMA, not
     wgmma and not tcgen05**; the file's own header at `:1596` records "the prime measures
     ~13.6 TFLOP/s/rank in the MoE, ~10x under what this pipeline should reach", and the
     `[moe-sk-form]` line it prints carries `occ128` (a `-1` there silently forces every group
     onto the 32-row form).
   - epilogue `swiglu_preclamped_mul_scaled`, down GEMM, `rows_permute`, 2 more pageable
     `htod_i32`, `moe_pairs_scatter`, `moe_ffn_grouped_add_shared`.
   - `MEMRA_PRIME_PROF=1` already prints `[moe-grouped-prefill-prof] il= t= router= gemm_gu=
     down_scatter= shared=` here (`:14704-14879`). **`[prime-prof]` does NOT**: it lives in
     `:5413 step35_prime_batch_layers`, the step37 batched path, which refuses on the first
     non-`Mixer::Full` layer and which glm5 never enters. Please run the box pass with
     `MEMRA_PRIME_PROF=1` for the per-layer MoE split and expect no `[prime-prof]` line; a
     hyper-walk norm/attn/o_proj/moe split does not exist yet and is item L7 below.

Sync/launch census for one 4k prime: 336 full stream drains, ~1,680 pageable H2D, ~2,900 kernel
launches (8 chunks x 45 layers x ~8 hc + ~12 mixer + ~12 MoE). That is why the nsys CUDA trace
inflated the 41.9k prime 10x and returned an empty kernel table: the profiler serializes on the
drains, and the drains are the shape.

## (c) Candidate levers, ranked by expected TTFT gain per unit of work

L1. **Chunk schedule for the hyper/PP2 walk** (`prime_chunk_tokens:1080`). The 8-microbatch
geometry is inherited from a pipeline glm5 does not run. Making it chunk-count-1 up to
`PRIME_CHUNK_MAX_TOKENS` (or 2 chunks if L2 lands) cuts expert traffic 8x at 4k, restores
m_e = 113.8 so every group takes the 128-row form, and should move 4k from 26.6 to the
41.9k-observed 87 TFLOP/s -> predicted TTFT 5.5 s -> ~1.7 s. NUMERIC CLASS: not bit-identical
across chunk sizes and never was — the documented `Engine::linear` m-dependence at
`hybrid_forward.rs:1240-1250`; the gate is the calibrated band against `memra_reference::execute`
that `glm5_chunked_prime_gpu` already holds. Work: one function, one flag. **Highest ratio by
far, and it is measurable today with `MEMRA_PRIME_CHUNK=4096` before any code lands.**

L2. **PP2 microbatch overlap of the prime.** Run chunk i+1's stage 0 on device 0 while chunk i's
stage 1 runs on device 1 (1F1B; the boundary already has double-buffered slots and per-stage
streams, `pp.rs:16-21`). Ideal 2x on the stage-serial half, ~1.7x realistic. Composes with L1
at 2 chunks (342 GB traffic instead of 171 GB) so the two must be tuned together, not stacked
blindly. NUMERIC CLASS: bit-identical — same per-chunk program, only issue order changes.
Work: moderate (the chunk loop at `:1853-1866` becomes a two-deep pipeline).

L3. **Device-side sigmoid router for the grouped prime.** Kills 336 stream drains and the
1,680 pageable H2D per 4k prime by building the CSR on device (count/scan/scatter) instead of
reading sel/w to host. NUMERIC CLASS: routing is bit-identical by construction if the device
CSR reproduces the host bucket order (expert-major, ascending pair index); the GEMM sees the
same rows. Work: 3 small kernels + the ordering gate. Expected gain is the drain cost, which the
box phase timers will size — after L1 there are 42 drains per prime, not 336, so **this ranks
below L1/L2 and its priority depends on the measured router/host share.**

L4. **cuBLASLt route for the expert GEMMs at large t.** `MEMRA_MOE_F16G=1` ALREADY selects
dequant-once-into-f16 + `cublasGemmGroupedBatchedEx` (`mmq_ffi.rs:2465-2520`); default is mode 2,
the memra kq kernel, and mode 1 has never been A/B'd on sm_100a. Workspace is
288 x 2048 x 4096 x 2 = 4.83 GB per projection, reused — affordable against 183 GB with
`MEMRA_MOE_RESIDENT_GB=130`. Extra traffic +29 GB/layer/chunk (+1.2 TB at 4k / 8 chunks, +150 GB
after L1) against a GEMM that on B200 should reach 5-15x the kq kernel's 13.6 TFLOP/s.
NUMERIC CLASS: f16-mirror, same class the kq path already is (both round dequanted weights to
f16 and accumulate f32); the mode-1/mode-2 pair is gated bitwise today by `kernel-check f16g-kq-direct`.
Work: **zero code to measure**, then a door if it wins. Second-cheapest probe after L1.

L5. **sm_120-tuned constants in the kq kernel on sm_100a.** `MEMRA_F16G_SK_CROSS=64` was swept
on 5090 and H100; `SK128_STAGES=3`, `KQ128_B_BUFS=2`, `launch_bounds(256)`, and the grid cap
`sms*occ128` were all sized against sm_120a's ~100 KB/SM smem where "2 CTA/SM is smem-impossible"
(`cu/moe_f16_grouped.cu:545-549`). B200 has 228 KB/SM, so occ128 can be 3 and the whole
persistent-CTA sizing changes. Also `MEMRA_F16G_BDB=1` (double-buffered B, one barrier per
k-block instead of two) is default OFF pending exactly this A/B. NUMERIC CLASS: bit-identical
(the k-chain is unchanged; only which CTA computes a tile and when the smem barrier falls).
Work: a sweep, no new kernel. **Cheap, and the `[moe-sk-form]` line from the box tells us
immediately whether occ128 is -1 (silent 32-row fallback, 4x redundant dequant).**

L6. **Real tensor-core class for the expert GEMM.** The mma is m16n8k16 warp MMA; sm_100a's
rate lives in tcgen05, and our sm_120a FP4 mma path does not exist there at all. A bf16 slab +
cuBLASLt (L4) buys most of this without a port; a tcgen05 port is the only thing that beats it,
and it is the largest piece of work in this list. Park behind L1/L2/L4/L5 measurements.

L7. **Instrument the hyper walk.** `MEMRA_PRIME_PROF` instruments the wrong function for glm5.
Add the same four marks (hc glue / mixer / MoE / tail) to `hyper_range_prime:2085` so the box
can attribute a prime without nsys. Trivial, diagnostic-only, and it is what turns the phase
timers into a decision. **Do this first inside the door, it costs nothing and it sizes L3.**

Not levers on this path, checked and dismissed: KDA and MLA carry **no** host synchronization
(the only `synchronize()` in `kda.rs:527/534` is the trace clock); `MEMRA_MLA_TC_PREFILL` is
already on and already batched; the KDA scan's launch count is 1 per layer per chunk, so its
cost is the in-kernel serial T-loop, not launches.

## Next

Waiting on the box phase timers (`MEMRA_PRIME_PROF=1` at 4k / 24k / 128k, true cold TTFT, and
the `[moe-sk-form]` line) before choosing between L1+L2 and L1+L4/L5 for
`MEMRA_B200_PRIME_V2` (default OFF, FLAGS row, `glm5-prime-v2-gate`).


## §4 BOX RECEIPTS (2x B200, int2 8c31be2f4, all doors + W8, cold primes, prefix cache off)

Boot A, `MEMRA_PRIME_PROF=1`, summing every `[moe-grouped-prefill-prof]` (layer, chunk) line:

| rung | schedule | MoE | TTFT | per token per MoE layer |
|---|---|---|---|---|
| 4k (6,920 tok) | 8 chunks 433/1016/981/950/922/896/872/850 | 5.09 s | 5.5 s | 7.3-8.3 us |
| 42k (41,866) | 4x2048 then ten of ~4393..3634 | 15.8 s | 16.7 s | 4.1 us (5.5 us at 2048) |
| 128k (128.7k) | ~31 chunks | ~44 s | ~113 s | attention/KDA own the rest |

Boot B (baseline) vs boot C (`MEMRA_PRIME_CHUNK=4096`): 4k **4.08 -> 2.68 s**, 1,697 -> 2,577
tok/s (**-34%**); 42k 16.7 -> 16.69 s and 128k 56.3 -> 56.1 s, both unchanged because their
chunks already clamped at 4096. **L1 is confirmed and is the door's arm 1.** Boot D
(`+ MEMRA_MOE_F16G=1`) is a DEFECT, not a number — see below.

Three corrections the box forced on §(a)/§(c):

1. **L5 is refuted.** `[moe-sk-form] dev=0 qt=7 sms=148 occ128=2 occ32=8 occt=7 cross=64
   xcross=64 bdb=0 in_f=4096 out_f=2048 n_active=283 max_m=115` — the 128-row form IS running
   (occ128=2, not the silent -1 fallback the kernel header warns about) and cross=64 is live.
   The sm_120-tuned-constants hypothesis was the wrong suspect.
2. **The expert GEMM is neither compute- nor byte-bound.** A 4096-token MoE layer chunk is
   ~1.24 TFLOP in 16.5 ms = **75 TFLOP/s = 3.4% of the 2.2 PFLOPS bf16 peak**, and the ~2.7 GB
   of expert bytes it touches is **0.34 ms** at HBM speed. Both floors are ~30x away: this is a
   kernel-efficiency problem, ~15x below what a bf16 grouped GEMM should reach on this part.
   Inside the MoE at 4096, gemm_gu ~9.0 ms : down_scatter ~4.5 ms : shared ~1.2 ms — 2:1,
   proportional to FLOPs, so no single projection is anomalous.
3. **The plateau is ~2,500 tok/s at every depth**, which is that kernel wall and nothing else.
   L1 gets the 4k rung up to the plateau; it cannot cross it.

**Boot D defect, and what it cost.** `MEMRA_MOE_F16G=1` (mode 1: dequant-into-a-workspace +
`cublasGemmGroupedBatchedEx`) destroyed the trunk. By layer 43 a 4096-token chunk reported
`n_active=14` where boots A-C showed 209-283 — a router seeing a wrecked residual — then the
corrupt logits produced a token id near `i32::MAX`, `EmbedHost::gather` panicked on
`self.raw[off..off + row_bytes]` (`range start index 17592186036224`, = 2^44), the GPU worker
died, the respawn died on the same prime, and every request after it was connection refused.
Root cause, from the code rather than from the log: `cublasGemmGroupedBatchedEx` issues on
cuBLAS-INTERNAL streams not ordered with ours — the round-46 NaN race that
`cu/moe_f16_grouped.cu`'s own header records — and mode 1's only mitigation is a full stream
sync placed AFTER the `h2f_scaled` pass, i.e. after the unordered read has already happened.
glm5_next reaches this arm with 283 concurrent groups where the families that qualified mode 1
reach it with a fraction of that. Two fixes landed in this lane: mode 1 is REFUSED by name on
the glm5 sigmoid grouped prefill, and `EmbedHost::try_gather` turns an out-of-range id into a
named request error instead of a fleet-fatal worker panic. **L4 as it exists is dead;** the
replacement is the door's dequant-once arm on OUR stream.

## §5 Re-ranked, post-box

L1 (shipped as arm 1, receipt boot C, -34% at 4k) -> **the expert GEMM itself** (dequant once
into a persistent f16 slab, then a cuBLASLt STRIDED-BATCHED matmul on our stream with the CSR
padded to a uniform n per expert: one launch per projection, no internal-stream race, no
per-projection sync, no 4.75 GB alloc/free per projection) -> L2 overlap (arm 2, shipped, pays
only at >= 2 chunks so it is a 42k/128k lever, not a 4k one) -> L3 device-side router.


## §6 Arm 3 shipped, and how to run the gate

**Box invocation, exactly:**

```
CUDA_VISIBLE_DEVICES=0,1 MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 NVIDIA_TF32_OVERRIDE=0 \
  glm5-prime-v2-gate
```

and NOTHING else. The gate is self-contained by construction after the 2026-09-02 box run
(int7 0ea1b07f6) exited 101 on the vacuity assert: it now computes the PP cut from the fixture's
own layer list and pins `MEMRA_PP_SPLITS` itself before any runtime exists, and it clears
`MEMRA_B200_PRIME_V2`, `MEMRA_PRIME_CHUNK`, `MEMRA_PRIME_PIPE` and `MEMRA_MOE_BGEMM_PAD_RATIO`
so no inherited value can decide an arm. Verified by running it with `MEMRA_PP_SPLITS=3` in the
environment: it still selects `cut=2` and passes. The assert stays as the check that the pin
worked. Red arms: `MEMRA_PRIME_V2_GATE_RED=schedule|pipe|bgemm`, all three confirmed to exit 1
on the rig. The shexp `[loader-law]` warnings the box saw are gone (the fixture's shared-expert
trio is Q8_0 now — those weights ARE matmul-class, so F32 was the fixture being wrong); the
remaining `[loader-law]` lines are the F32 trunk weights this fixture shares with
`glm5-hyper-ppn-gate` and `glm5-hyper-batch-gate`, same wall, not a signal from this gate.

**Rig run 2026-09-02 (5090, exactness only, no timing claim), all three arms PASS:**

```
arm 1 SCHEDULE  t=4096: shipped 8 chunks [256,602,580,563,545,531,516,503] -> door 1 chunk
arm 1 SCHEDULE  logits rel 4.153e-5 stack rel 3.016e-2 (f16-mirror grouped class) argmax 3 vs 3
arm 1 CONTROL   shipped reschedule to 2 chunks: logits rel 2.664e-5 stack rel 8.929e-2
arm 2 PIPELINE  chunks=2: serial pipelined-count 0, door 2; bit mismatches 0/32 and 0/1048576
arm 3 BGEMM     dispatches 9/9; logits rel 0.000e0 stack rel 0.000e0; argmax 21 vs 21
arm 3 CONTROL   one-token prompt change moves logits 1.118e0
arm 3 GUARD     at the shipped pad-ratio default: dispatches 0 (want 0 on 2.94 fixture skew)
```

**One finding worth the owner's attention, from arm 1's row profile.** The hidden stack moves
more than a naive band allows, and the row attribution says why: the error is CONFINED to
whichever chunk the schedule ends on (`per-eighth worst [5.9e-3 6.1e-3 6.8e-2 5.6e-3 6.0e-3
4.7e-3 7.9e-2 9.0e-2]`), not spread. The mechanism is this trunk's DSA k-pool selection being
DISCRETE: each query takes `index_topk / index_kpool` pools, and any numeric perturbation can
flip a near-tied pool, which moves that row discontinuously. On the fixture that is 3 pools per
query, so one flip is a third of the attention mass; the real model takes 512, where one flip is
0.2%. It is NOT introduced by the door: an ALREADY-SHIPPED reschedule (an explicit
`MEMRA_PRIME_CHUNK`, door shut) moves the stack 8.9e-2 where the door moves it 3.0e-2 — the door
moves it LESS. Arm 1's bar is therefore relative to that control (4x headroom) with an absolute
band on the logits and absolute argmax equality, which are what a caller actually sees. Anyone
quoting a hidden-stack number for this trunk across schedules should quote the control with it.

**Arm 3 workspace against the 130 GB resident slab.** ~5.8 GB per stage device at GLM-5.3-Flash
geometry: slab 288 x 4096 x 2048 x 2 = 4.83 GB, padded activation 0.30, padded output 0.60,
cuBLASLt scratch 0.03. It is charged to admission once per request while the door is open
(`HyperPrimeWorkspaceShape::bgemm_workspace_bytes`), never per token, because
`AdmissionCostModel::observe` provably cannot learn a prime-time allocation — that is the 262k
2-card hole. On a 183 GB B200 with `MEMRA_MOE_RESIDENT_GB=130` that leaves ~47 GB against a
1M-context session's ~46 GB of KV: **on the 1M tier this arm and a full-depth session do not
both fit**, and admission now refuses instead of finding out mid-prime. Dropping
`MEMRA_MOE_RESIDENT_GB` to 124 is the lever if both are wanted.


## §7 Box A/B for arms 1+2 (boot F), and what boot E killed

| boot | 4k (6,920 tok) | 42k (41,866) | 128k (128,059) |
|---|---|---|---|
| B baseline | 4.08 s / 1,697 tok/s | 16.73 s | 56.30 s |
| C `MEMRA_PRIME_CHUNK=4096` | 2.68 / 2,577 | 16.69 | 56.11 |
| E C + `MEMRA_F16G_BDB=1` | 2.70 | 16.70 | 56.15 |
| **F `MEMRA_B200_PRIME_V2=1`** | **2.09 / 3,307** | **9.59 / 4,364** | **30.63 / 4,181** |

Boot F against boot C: **-22% at 4k, -43% at 42k, -45% at 128k**; against baseline -49 / -43 /
-46%. Decode unchanged (35-37 tok/s in the tail), no errors. Arm 1 is the 4k win and arm 2 is
the depth win, which is exactly the shape predicted in §5: arm 2 needs two or more chunks, and
4k is one chunk under arm 1.

**Boot E killed the cheap kernel lever.** `MEMRA_F16G_BDB=1` engaged (`[moe-sk-form] ... bdb=1
... n_active=288 max_m=816`) and measured within noise of boot C at every depth. So the shipped
schedule's single-B-buffer dequant/mma serialization is NOT where the 16.5 ms goes, and the
sk128v form is bound elsewhere — the `mma.sync.m16n8k16` issue rate on sm_100a, the in-register
dequant, or L2/smem traffic. Every one of those says the same thing: the remaining gap is the
instruction class, not the schedule around it, and the way to a different instruction class
without writing tcgen05 is a library GEMM on a dequantized slab. That is arm 3.


## §8 Arm 3 bucketed (boot H's decline, and the fix)

Boot H ran arms 1+2 at boot F's TTFT and then declined arm 3 on every layer:
`[moe-bgemm] DECLINED: routing skew: 288 experts x n_pad 816 = 235,008 padded rows against
32,768 real (ratio 7.17 > 1.35)`. That is the guard doing its job on a real property of the
model, not a tuning miss: glm5's routing is skewed BY DESIGN (measured 2026-09-02, Gini 0.575
over 288 experts, busiest expert 77% of a layer's tokens, median 1.3%), so ONE uniform-n batch
across the whole bank pads every expert up to the busiest one and can never meet any sane cap.

**Fix: bucket.** Sort the active experts by row count, cut them into contiguous buckets, and
issue one cuBLASLt strided-batched call per bucket per projection — still stream-ordered, still
one slab, still one dequant. Heavy head experts land in singleton buckets, the long tail shares
one. The pad/unpad kernels stay ONE launch each whatever K is because they walk a `pad_map`
(one entry per padded row: the CSR pair it carries, or -1), which is also the only structure
that has to be right for the layout to be correct.

**The per-bucket target is swept, not guessed** — four O(n) passes per call, keeping the
partition with the smallest AGGREGATE padding. A tighter target is NOT monotonically better: it
narrows buckets, which only helps while the bucket budget can absorb the extra ones. On the
modelled 288-expert profile: 1.163 at 16 buckets against 2.207 at 8 for the same target, and
1.325 for the natural 8-bucket partition at the loosest target. That is why `MOE_BGEMM_MAX_BUCKETS`
is 16 (48 launches per MoE layer against a layer measured at 16.5 ms — the launches are noise)
and why the sweep exists at all.

**Held two ways.** `moe_bgemm_plan_tests` is host-only, no GPU, on the measured 288-expert
profile — the width the fixture cannot reach and the width the defect had. It asserts the
profile really is the measured one, that the unbucketed ratio is still past 5x (the control:
if that ever stops being true, the bucketer is being credited for a problem that no longer
exists), that bucketing lands under 1.25 with headroom, that buckets tile the expert list and
the padded plane exactly, that the expert order is a permutation, that every real pair appears
in the pad map exactly once, and that each expert's rows stay contiguous in one bucket.

The gate's fixture now reproduces the skew rather than assuming it: its sigmoid router
correction bias is overwritten with a geometric profile, and arm 3 asserts the partition from
the engine's own published plan stats (`MOE_BGEMM_LAST_{BUCKETS,PAD_MILLI,SPREAD_MILLI}`)
rather than parsing a log line — K >= 2, an n_pad spread >= 4x, and the aggregate ratio inside
the ceiling. Rig run: `buckets=7 pad=[2960,1248,768,608,304,32,16] pad_ratio=1.086`, spread
106.5x, 9/9 dispatches, and a cap of 1.0 makes it decline (the ceiling's own red arm).

**Route taken, since both were open:** bucketed strided-batched, NOT ordering
`cublasGemmGroupedBatchedEx` with events. Grouped-batched needs no padding, but the ordering
question is exactly what killed the worker in boot D, and "we believe the library honours
`cublasSetStream` on 13.1" is a belief about a closed-source scheduler that would be load-bearing
for correctness on every future toolkit. Bucketing costs ~9% padding on the measured profile and
buys an ordering guarantee that is structural rather than empirical. If a future measurement
shows the padding is the binding cost, the grouped route can be revisited with a real ordering
receipt; it should not be taken on trust.
