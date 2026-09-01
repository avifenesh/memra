# KDA + mHC decode launch diet (lever 1 of the step37 transfer map)

Lane: `lane/glm5-launch-diet` (2026-08-30). Parent evidence:
`../step37-transfer-map-20260829/TRANSFER-MAP.md` (lever 1),
`../decode-attribution-receipts/ROADMAP.txt` (step 3: 33.39 ms/token =
"~0 staging / 15.9 roofline / 17.2 launch", "LAUNCH STRUCTURE IS NOW 51% OF THE
TOKEN"), `../engine-survey-20260829/ENGINE-SURVEY.md` (C1: both vLLM and SGLang
independently merged the six KDA input projections into ONE GEMM per layer —
vLLM `in_proj_qkvbfg_a` "6 to 1 launches", SGLang `fused_qkvbfg_a_proj` — plus
one merged qkv conv and in-scan gate/beta; designs copied, no kernel code).

## The decode program, counted (from source, this checkout)

Per decode token, per KDA layer (`kda.rs kda_core`, 34 layers):

| stage | host calls | kernel launches today (serving classes) |
|---|---|---|
| 1. six projections, one shared input | `matmul_group([wq,wk,wv,f_a,g_a,b_proj], x, 1)` = 6x `matmul` | wq/wk/wv are **Q8_0** (the loader's BF16>=1M law; the mint's `exclude_modules` key is not read — `kda_quant_operand_gpu.rs` header): 3x (`quantize_q8_1` + `qmatvec_q8_0_mmvq`) = 6. f_a/g_a/b_proj are **Float f32** (<1M): 3x cuBLASLt f32 GEMV (dot+reduce pair class) = ~6. Total ~12 |
| 2. conv | 3x `kda_conv_silu_decode` | 3 |
| 3. L2 norm | 2x `l2_norm` | 2 |
| 4. gates | `matmul(f_b)` + `kda_gate` + `sigmoid` | f_b is Q8_0 (1.05M): 2 + 1 + 1 = 4 |
| 5. scan | `kda_scan` | 1 |
| 6. out | `matmul(g_b)` + `kda_gated_rmsnorm` + `matmul(wo)` | 2 + 1 + 2 = 5 |

~27 launches/layer x 34 = ~918 in the KDA family (the map's "~600" counted only
the named-kernel share; the quantize/dot/reduce twins ride along). mHC per layer
(`hyper.rs pre/post` + `hybrid_forward.rs hyper_range_decode`): per site
`linear` (cuBLASLt mixes GEMM) + `rowsq_scale` + `sinkhorn` + `collapse` +
`rms_norm` + `post` = 6, x2 sites x 45 layers = 540 — exactly the map's number.

## Phase 1 — the census cell (BUILT, waiting on a box window)

`census-decode-phases.sh` — the decode twin of
`../prefill-gap-20260829/profile-prime-phases.sh`. One nsys pass over a warm
boot + one real ~200-prompt/192-completion request on the residency config with
`MEMRA_MOE_FUSED_EPI=1`, then three bucketed reports: kernel families (counts +
GPU ms + per-token), memcpy/memset, and the cuda-api sum (alloc/sync/launch
host families — the step37 DECODE_V2 term that gpu_ms cannot see). Deliverable:
the measured split of the 17.2 ms launch term by chain, which sizes every
fusion boundary after this one and re-answers the graph question (launch vs
dependency latency) with data. Box requirements are in the script header. DO
NOT take the box: the L2/L3 A/B window owns it; ask the owner for a window.

Named ambiguity the script states instead of hiding: the cuBLASLt f32 GEMV
class serves both the mHC mixes (90/token) and the KDA f32 projections
(102/token before the fused door); the report splits them by count arithmetic.

## Phase 2 — first fusion: the 6-way KDA projection consolidation (BUILT, flag OFF)

`MEMRA_KDA_FUSED_PROJ=1` (default OFF): stage 1's six matvec calls collapse to
ONE `quantize_q8_1` + ONE `qmatvec_kda6_q8f32_mmvq` launch per layer. The
kernel extends the in-tree `qmatvec_q8_0_mmvq_fused2/3` block-offset recipe
(qmatvec.cu, "BIT-IDENTICAL to two separate m=1 launches") to six unequal
ranges across two operand classes:

- **wq/wk/wv (Q8_0)**: per-(token,row) body lifted VERBATIM from
  `qmatvec_q8_0_mmvq` — same lane walk, same dp4a order, same
  `warp_reduce_sum`, same single write. **Bit-identical** to the unfused MMVQ
  arm by construction, asserted bytewise in the gate at every width.
- **f_a/g_a/b_proj (Float f32)**: warp-per-row f32 dot (lane-strided float4,
  shfl tree). This REPLACES cuBLASLt for these three rows: a reduction-order
  numeric-class change, the exact step37 `MEMRA_STEP_TP_QKV_FUSED` precedent
  (its class: "2.4e-3 -> 4.2e-3"). Measured on the gate fixture and pinned
  there; stated against BOTH memra_reference and the unfused arm.

Engagement (all must hold, else silent fall-through to the unfused arm —
behavior unchanged): flag on; t in 1..=15 (the batch cap); wq/wk/wv all
`Quant{Q8_0, rp:false, rp4:None, scale:1.0}` with one `in_f`/`row_bytes`;
f_a/g_a/b_proj all `Float` at the same `in_f`; `in_f % 128 == 0`; and the env
classes under which the unfused arm rides the MMVQ-class per-row program
(`MEMRA_FAST!=0`, `MEMRA_MMVQ!=0`, `MEMRA_NO_BATCHED` unset for t>=2,
`MEMRA_B8!=0` for t>=5) — outside those envs the bit-identity claim would be
against a different unfused kernel class, so the door refuses instead of
weakening the claim. The flag is read PER CALL (the `MEMRA_MOE_FUSED_EPI`
rollback-seam precedent). Engagement is announced once per boot
(`[kda-fused6] engaged ...`) and counted (`kda_fused6_dispatches`) — the
spec-engagement receipt the box A/B arms must show.

Gate: `crates/memra-engine/tests/kda_fused_proj_gpu.rs` — RUN GREEN 5/5 on the
rig 5090 (2026-08-30, `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`,
debug build, at commit `9cadb3261dcedf32b567c74693bb42176d6aa7ba`, invocation
`cargo test -p memra-engine --test kda_fused_proj_gpu -- --ignored --test-threads=1`).
Fixture = the REAL residency classes (Q8_0 wq/wk/wv via the
`kda_quant_operand_gpu` re-encode pattern; Float f_a/g_a/b_proj). Measured:

1. bit-identity, q8 rows: fused vs unfused `to_bits` equality at EVERY t=1..15
   — PASS (engagement counted per call, announce line printed);
2. f32 rows vs the cuBLASLt arm: worst relative maxdiff **4.703e-7** across
   all three rows and all widths (bar 5e-5, ~100x headroom);
3. whole-mixer: fused-vs-unfused worst **1.519e-7** relative at t in {1,7,15};
   vs `memra_reference` the ON and OFF arms print IDENTICAL deviations
   (7.509e-2 / 1.160e-2 / 8.017e-3 at t=1/7/15 — the Q8_0 operand floor of
   this fixture/seed, present in both arms), and the gate asserts ON may not
   exceed OFF beyond the mixer band;
4. RED, transposed slice x6: each weight replaced by its transposed-data twin
   (loads, runs, finite, silently wrong) — comparator FAILS on exactly that
   slice, PASSES on the other five (isolation) — PASS;
5. RED, dropped projection x6: each range removed from the launch in turn
   (out_i=0, output zero-filled) — comparator FAILS on exactly that slice —
   PASS;
6. flag-off: door never engages without the env (counter flat across a mixer
   walk), `kda_proj_fused6` refuses at t=16 (the GEMM tier) — PASS. The
   pre-existing `kda_fixture_gpu` (3/3) and `kda_quant_operand_gpu` (4/4)
   gates re-ran green on the same checkout: the OFF arm is the untouched
   program.

Public boundary from repo root after the change: "677 matches (677
grandfathered, 0 new)".

Flip condition (NOT this lane's call): box A/B interleaved x5 fresh boots on
the serving card class, real prompts, greedy + vendor-default sampled twin,
engagement announce in BOTH arms, per the grouped-prefill acceptance form.

## Expected value — SUPERSEDED BY THE CENSUS (2026-08-30, WINDOW-20260830.md)

The pre-census arithmetic (~1.8 ms/token from ~5.3 us/launch against the
17.2 ms f32-arm launch term) DID NOT SURVIVE measurement on the serving arm.
The census (2-card arm C + fused epi ON) measured: 2125 launches/token,
launch/gap term ~4.4-4.7 ms/token at this box's own 2.216 us/launch, token
GPU-kernel-time dominated (25.1 ms of ~29.5). And on arm C the loader admits
the KDA projections to BF16 residency (`admit=bf16_mmv` receipt), so THIS
door's Q8_0 arm refuses there by design — it binds on non-BF16_MMV shapes
only. Re-ranked increments, sized by the census:

1. BF16 operand arm for the fused-6 door (per-row body =
   `matvec_bf16_f32acc_x4_rows` VERBATIM, same bit-identity recipe): the
   serving-shape twin of this door; ~0.5-1 ms/token class (launch+alloc off
   the sync-serialized path).
2. mHC pre-chain fusion (rowsq+sinkhorn+collapse, one launch/site): 3.3
   ms/token of GPU time in dsv4 site kernels, sinkhorn 18.3 us at t=1 (20
   serial iterations) — time AND launches.
3. Persistent decode workspace (step37 DECODE_V2 class): 2358
   cuMemAllocAsync/token measured.
4. The bandwidth/kernel-time items now outrank launch work entirely:
   kda-proj-bf16 8.2 ms (levers 6/2), mla-decode 4.8 ms, moe-epilogue 4.7 ms.
5. Graphs: capped at ~1.8 ms/token by launch-econ (2.216 -> 1.382 us), and
   43 host syncs/token make whole-token capture structurally impossible.
   Do-not-transfer verdict re-confirmed with this box's numbers.

The census also answered the PREFILL attribution question (the coordinator's
urgent add): the ~5.9 s unattributed residual is the MLA/DSA prefill
attention chain — `memra_mla_attn_gathered` + `absorb_q` + `decompress_v` =
75.8% of a 98%-GPU-busy prime, ~139+44+44 ms per layer-chunk, running 1-2
orders below tensor-core class. Full numbers and the named next lever in
WINDOW-20260830.md section 4.

## Do-not-touch list honored

The mixes GEMM stays `Engine::linear` under the pre_exact law (hyper.rs
untouched); the router (exact f32, D2H drives admission) untouched; L2/L3
lanes' files untouched beyond merging pushed state.
