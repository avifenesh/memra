# glm5_next TP-2 shard map (stage 2 — the review artifact)

Written BEFORE code, per the lane brief. Every row: shard axis, quantization-block
legality, or replicate/owner-stage decision, with the receipt it leans on. Geometry from
`glm-config.json` + CENSUS.md; byte terms from PLACEMENT-RECEIPT.md and the decode-gap
ATTRIBUTION.md; code anchors from the current tree @ 81ad2951b (hybrid_forward.rs = HF,
hybrid.rs = HY, kda.rs = KDA, hyper.rs = HYP, tp.rs = TP, nvfp4_repack.rs = NR).

## 0. Rank topology and design posture

- **2 ranks.** rank0 = root = the PP-owning device of the layer range (step precedent:
  the first device of the rank list MUST be the PP owner, parallel.rs:445). rank1 = peer.
- **Owner-centric trunk.** All per-token sequential glue (mHC sites, norms, router,
  embed, decode tail glue) stays on root; TP fans out and joins at the mixer and FFN
  blocks. This is the MEMRA_STEP_TP house shape ("router, head gate, and replicated
  shared-expert work stay on the owning PP stage", docs/FLAGS.md:269).
- **Column-parallel-first arithmetic.** Wherever possible the cut is on the OUTPUT axis
  and inputs are gathered, so every output element is computed by the SAME kernel over
  the SAME full input vector as the unsharded walk — joins are pure data movement
  (gather/concat), never a partial-sum reduction. This is what makes model-level
  TP-2-vs-plain BYTE identity a realistic bar for the BF16/f32 classes instead of a
  tolerance band. The two places arithmetic does cross ranks (MoE expert combine; any
  future row-parallel arm) carry an explicit canonical-order contract (§3, §6).
- **v1 transport = host-canonical staging** (the step "correctness transport"): fan-out
  and joins bounce through pinned host buffers. Native P2P (with the 16KiB/64KiB/1MiB/
  64MiB byte-integrity ladder, TP:8765) and the join-diet doors are the BOX arm, staged
  exactly like step37's ladder (correctness first at 43.1, then +0.6/+1.85/+2.5/+5.1).
- **TP-2 only.** TP-3 refused by geometry (64 attn heads, 32 indexer heads, 34/11 layer
  classes: nothing divides by 3 — the receipted asym-split dead end applies); TP-4+
  refused until designed and gated (MLA 16 heads/rank and EP-4 72/rank are legal on
  paper but ungated). Refusals by name, before CUDA.

## 1. KDA layers (34; all weights BF16-kept in the artifact) — SHARD BY HEADS, 32/rank

64 heads x head_dim 128; qkv_dim = 8192. Per-head independence is exact: conv, L2 norm,
scan, gated rmsnorm are per-head/per-channel kernels (KDA:297-353).

| tensor | shape [out,in] | serving residency | decision | legality |
|---|---|---|---|---|
| wq/wk/wv | [8192, 4096] | BF16-resident (MEMRA_BF16_MMV) | column-parallel by heads: rank r owns head rows r*4096..(r+1)*4096 | BF16, no blocks; cut on head boundary (128-row multiples) |
| q/k/v_conv1d (fused [3*8192, 4]) | per-channel | f32 | channels follow the head split; conv STATE rank-local | per-channel independent |
| f_a_proj / g_a_proj | [128, 4096] | f32 | REPLICATE compute (both ranks run the identical matvec on the identical x) | deterministic same-kernel-same-bytes; uniform-hardware preflight required |
| f_b_proj / g_b_proj | [8192, 128] | Q8_0 (loader law: 1.05M elems) | column-parallel by heads (per-channel outputs) | Q8_0 blocks are per-32 along IN; an OUT-row cut never touches a block |
| b_proj | [64, 4096] | f32 | column-parallel by heads (one beta row per head) | trivial |
| A_log / dt_bias | [64] | f32 | shard by head | trivial |
| o_norm | [128] | f32 | replicate (per-head norm weight, same 128 vector for all heads) | trivial |
| wo (o_proj, BF16 on KDA layers) | [4096, 8192] | BF16-resident | column-parallel on OUT (2048 rows/rank) over the GATHERED full attn output | requires the 8192-wide attn output on both ranks: exchange the two 4096 halves (16 KB each way), then per-row full-K dots — bitwise-plain |
| recurrent state (ssm_state 64x128x128 f32, ping-pong) | — | state | rank-local halves (32 heads each); rewind/ckpt seams must address both ranks' planes coherently | per-head independent |

Per-token per-layer traffic (v1): x in (16 KB) + attn-half exchange (2 x 16 KB) +
x_out half concat to root (8 KB). Weight bytes/rank: 9.13 GB total -> 4.57 GB.

Alternative rejected: wo as ROW-parallel with a partial-sum join (the step
ResidentStepBf16RowParallel pattern) — saves one 16 KB hop but converts the join into
cross-rank arithmetic that can only be pinned against a TP1-oracle-of-the-same-program,
not against plain bytes. Column-over-gather keeps the plain kernel per output row.
Revisit only in the box join-diet arc with its own gate.

## 2. MLA/DSA layers (11 trunk; q_a/q_b/kv_a/o_proj NVFP4, kv_b BF16->f32 split) — SHARD BY HEADS, REPLICATE THE LATENT

64 heads; qk_nope 256, v_head 256, kv_lora 512, q_lora 1536, NoPE. Weights total
~1.25 GB (bytes-tiny; the 4.76 ms/token family is per-head kernel work, which the head
split parallelizes).

| tensor / plane | decision | legality / rationale |
|---|---|---|
| wq_a (q down, [1536, 4096], NVFP4) + q_a_norm | REPLICATE compute — both ranks compute the identical q_lora residual | NVFP4 matvec on identical bytes is deterministic per arch; uniform-hardware preflight; avoids a 6 KB broadcast in the latency chain |
| wq_b ([64*(256), 1536], NVFP4) | column-parallel by heads (32/rank) | OUT-row cut: scale rows [out, in/16] ride whole; never cuts a block |
| wkv_a ([512, 4096], NVFP4) + kv_a_norm + latent append | REPLICATE compute; the latent plane `LatentKvLayer.rows` (512 f32/token) is REPLICATED on both ranks | per-token shared by all heads; replication = zero cross-rank hop in the per-token chain; capacity cost 2x latent (3.22 GB @128k, 25.8 GB @1M — named in §7) |
| wk_b / wv_b (3D per-head [nope,512,64]/[512,256,64], f32-resident) | shard by head (32 planes each) | per-head 3D slabs; absorb kernels (mla_absorb_q, mla_decompress_v) are per-head grids |
| indexer (wk, wq_b, weights_proj, k_norm, kpool gate/ape; 32 heads x 128) + kpool selection + index ring/pool-key planes | REPLICATE everything — both ranks compute identical scores and the identical top-2048 selection; index planes replicated | selection must be bit-identical on both ranks; deterministic same-bytes compute is the cheapest guarantee (join-diet blessed "replicated deterministic compute"); fallback if a gate ever disproves determinism: root computes, broadcasts the selection list (~8 KB/token/layer) |
| gathered attention (mla_attn_gathered over selected latent rows) | rank-local over the rank's 32 heads, reading the rank's own replicated latent + selection | per-head independent |
| o_proj ([4096, 16384], NVFP4) | column-parallel on OUT (2048 rows/rank) over the GATHERED full 16384-wide attn output (exchange 32-head halves, 32 KB each way) | full-K per-row dots keep plain arithmetic bitwise; an OUT cut never touches NVFP4 blocks. (A row-parallel cut at 8192 = 32 heads x 256 would be 64-aligned and legal per TP:9515-9531, but is rejected for the same reduction-order reason as KDA wo) |

NEW verification surface named honestly: absorbed-decode exactness cross-rank has no
existing gate; the stage-4 fixture gate adds the MLA class arm (2 heads -> 1/rank) and
the kpool-selection replication identity arm.

## 3. MoE (42 sparse layers + MTP twin; NVFP4 experts) — EP-2, 144 EXPERTS/RANK

288 routed experts, top-8, sigmoid noaux_tc (scoring sigmoid + e_score_correction_bias,
routed_scaling 2.5, norm_topk). Expert projection block = 4,718,592 B uniform (gate/up
[2048,4096], down [4096,2048], NVFP4 0.5625 B/elem).

- **EP, not MoE-TP.** Whole experts move; zero cuts, so the NVFP4 repack geometry
  (per-16 e4m3 scales inside 64-elem superblocks, NR:13-28), the per-expert
  `weight_scale_2` macro (`HostExps::macros`), the fused epilogue kernels
  (`moe_gate_up_preclamp8_q8` + `moe_down8_fma_q8`, LIB:5490/5542), and the
  MEMRA_MOE_CACHE single-size-class SLRU all survive untouched per rank. MoE-TP would
  halve every matvec width, split the 4.72 MB block identity the SLRU keys on, and put
  a partial-sum reduction inside every expert. Same verdict as the multicard doc's
  step37 row ("TP4 uses expert ownership because 1280/4 cuts those scale blocks") for
  a different reason: glm5's 2048-wide cuts are legal (2048/2=1024, 1024%64==0) but
  everything downstream of the cut is worse.
- **Ownership: contiguous halves.** rank0 = experts 0..143, rank1 = 144..287
  (owner = expert / (288/2), the `partition_expert_owner_routes` law, TP:383-431).
- **The honest EP-2 arithmetic against top-8** (the multicard doc's demanded row):
  P(a token's 8 experts all land on one rank) = 2 * C(144,8)/C(288,8) ~= 0.7% — the
  peer is touched on ~99.3% of layer-tokens, so EP-2 is NOT sparse communication; it is
  expert locality + parallel HBM, exactly as PRO6000-MULTICARD.md warns. Load split:
  k0 ~ Hypergeometric(288,144,8), E=4, sigma~=1.40, E[max(k0,8-k0)] ~= 5.1 of 8 — the
  slowest rank reads ~64% of the expert bytes, so the expected expert-read speedup is
  **~1.57x, not 2x** (4.76 GB/token -> ~3.05 GB on the critical rank; roofline 2.66 ms
  -> ~1.70 ms). Worst case 8/0 (rare, harmless — root idles).
- **Router: root-computed in v1.** Router logits (`gate_inp` f32-resident) + host
  sigmoid top-8 stay on root exactly as today (HF:8114, HF:9954); the peer receives the
  token's x plus its owned (slot, expert-id, weight) list. Replicated device-router is
  the named join-diet door (step37 +0.6), NOT v1.
- **Combine: slot-ordered canonical accumulation.** The plain kernel FMAs expert
  contributions into one accumulator in router slot order. EP v1 reproduces it: each
  owner computes its experts' UNWEIGHTED down rows with the same dot arithmetic, root
  applies `dst += w_slot * row_slot` for slots 0..7 in order (one tiny canonical-combine
  kernel; the `validate_weighted_route_combine` contract — every canonical pair exactly
  once, TP:475-528). Bit identity of this combine vs the fused `moe_down8_fma_q8` walk
  is a NAMED GATE ARM (stage 4), not an assumption; if the fused kernel's dot/FMA
  rounding cannot be reproduced kernel-for-kernel, the MoE layer class falls back to
  the truth-anchored band bar (memra_reference, the chunked-prime precedent,
  FLAGS.md:290) and the model-level byte-identity claim is scoped to the non-MoE
  classes — stated in the gate table either way.
- **Shared expert (NVFP4, ~14 MB/layer): root-owned**, computed on root where the
  combine lands (and the natural prejoin-overlap filler in the box diet arc — the
  step37 +5.1 lever). Never sharded (step precedent, parallel.rs:392-395).
- **Dense MLPs (layers 0-2, NVFP4 [12288,4096]/[4096,12288], 0.45 GB total):
  root-owned in v1.** Bytes are 0.25 ms/token; a Megatron split would save ~0.13 ms for
  two extra hops and one more gated surface. Named optional later (cuts legal:
  12288/2 = 6144, 6144 % 64 == 0).
- Per-token per-layer traffic (v1): x to peer (16 KB, or the q8_1-quantized 4.3 KB twin
  later) + peer's owned-slot rows back (~4 x 16 KB expected) + slot list (~64 B).

## 4. mHC (45 layers x 2 sites, f32-by-decree, hc_mult 4, 20 Sinkhorn) — ROOT-OWNED

Per-token, no persistent state (HcMix lives between a site's pre/post halves; the
[4, 4096] stream tensor is intra-step, HYP:300, HF:1954-1959). Per the step router
precedent (per-token work = replicate or owner-stage), v1 keeps ALL mHC work on root:
expand, both sites' pre (mixes GEMM + rowsq + sinkhorn + collapse), post, exit-Mean,
decode tail. The mixer/FFN TP blocks receive the collapsed branch input x and return
their outputs to root, which is the same seam the blocks already have.

Rejected for v1: replicated mHC (both ranks maintain identical stream state; removes
the x-broadcast hop; requires symmetric joins everywhere and doubles the Sinkhorn
compute onto the peer — wall-neutral but a much wider exactness surface). Note the mHC
chain (~3.3 ms GPU/token) is the latency class TP-2 does NOT divide; its fix is the
separately-named pre-chain fusion lever (decode-gap lever #2), orthogonal to this lane.

## 5. Embedding, head, MTP, vision

| tensor | decision |
|---|---|
| embed_tokens (BF16, 1.27 GB) | root-owned (one row gather/token) |
| lm_head (BF16-resident, [154880, 4096], 1.27 GB) | column-parallel by vocab: rank r owns 77,440 rows; each rank runs the same matvec over the same hn and DtoH's its half into the host logits buffer; concat is pure movement — bitwise-plain, no reduction. Halves the single largest single-tensor read (0.71 -> 0.35 ms) |
| output_norm | root (replicated input already on root) |
| MTP/NextN layer + glm5 spec machinery | OUT OF SCOPE, root-owned untouched; the TP door and the spec doors co-refuse at boot (two programs on one model never silently coexist — the MEMRA_DSPARK precedent). Spec is currently NO-FLIP for glm5 anyway |
| vision tower (BF16) | out of scope, root-owned |

## 6. Cross-cutting laws

1. **NVFP4 cut legality** (from NR + TP:8978-9040): OUT-row cuts always legal (scale
   rows [out, in/16] ride whole; `weight_scale_2` is per-tensor and applies post-matmul
   exactly once). IN cuts only on 64-element superblock boundaries — and v1 makes NO in
   cuts on NVFP4 tensors at all. Q8_0: per-32 blocks along IN; same rule, v1 makes no
   IN cuts. BF16: head-boundary cuts only (canonical 128-row multiples).
2. **Replicated deterministic compute** requires the uniform-hardware preflight per
   runtime group (`detect_uniform_hardware`, the step pattern) — same arch, same kernel
   selection. Every replicated site (f_a/g_a, q_a, kv_a+latent, indexer+selection,
   router-logits-if-ever) is enumerated in the preflight receipt so the gate can assert
   the list.
3. **Canonical combine order** for the ONE cross-rank arithmetic site (MoE combine):
   slot order 0..7, root-accumulated, `zeros`-seeded — mirrored by the TP1 oracle of
   the same program, and gate-compared against the plain fused walk (§3).
4. **State coherency**: KDA recur planes are rank-split; MLA latent/index planes are
   replicated. Prime, rewind, and cache-checkpoint seams must either address both
   ranks' planes coherently or REFUSE by name (fail-closed; prefix snapshots already
   refuse for this family — TRAP:glm53:prefix-cache-snapshot-refused).
5. **Composition refusals at boot** (all before CUDA): `MEMRA_PP_STAGES>1` + glm5 TP
   (until the TP-2+PP-2 composition gate exists — stage 5 names it); any spec door +
   glm5 TP; `MEMRA_STEP_TP`/`MEMRA_STEP_EP` + glm5 TP (different contracts, never
   co-armed); TP-3+; duplicate devices (except the gate binary's explicit same-device
   emulation constructor, which serving spec parsing never reaches); non-2-rank lists.

## 7. Per-rank byte budget (full residency posture, 2x96 GB pair)

| class | total | rank0 (root) | rank1 |
|---|---|---|---|
| routed experts (EP-2) | 175.31 GB | 87.7 | 87.7 |
| KDA projections (head split) | 9.13 | 4.6 | 4.6 |
| KDA gates/conv/norms (repl a-side + split b-side) | ~0.4 | ~0.25 | ~0.25 |
| MLA weights (head split + replicated a/kv_a/indexer) | 1.25 | ~0.8 | ~0.8 |
| shared experts + dense MLPs + router (root) | ~1.2 | 1.2 | 0 |
| mHC + norms (root) | ~0.3 | 0.3 | 0 |
| embed + lm_head (vocab split; embed root) | 2.54 | 1.9 | 0.64 |
| vision + MTP (root) | ~2.4 | 2.4 | 0 |
| **weights total** | ~192.5 | **~99.2** | **~94.0** |
| KV @128k (latent+index REPLICATED, KDA state split) | 5.19 GB/seq | ~5.1 | ~5.1 |

Full residency is tight-to-over on root (99.2 GB > 95.6 GiB usable) — exactly the
posture QUIRK:glm53:no-resident-pp2-headroom-f32arm warns about. The serving posture
therefore keeps MEMRA_MOE_CACHE SLRU (hot-slot budget per rank) and/or host-resident
vision+MTP, and the box arm MEASURES admission; this map only fixes ownership. The
latent replication cost (2x) is bounded and bought deliberately: it removes every
per-token cross-rank hop from the MLA latent chain.

## 8. What is reused vs built (the port map)

| piece | source | reuse |
|---|---|---|
| spec grammar + parser | `parse_step_layer_specs` TP:735 (flag name already parameterized) | reuse; `all` derives 0..44 from the glm5 contract, not STEP37_TRUNK_LAYERS |
| preflight shape (owner-first, distinct, unique group, per-rank non-empty) | `preflight_step_tp_specs` parallel.rs:403 | reuse the shape; new glm5 arm |
| model contract | `ModelParallelContract::from_plan` (hard is_step37 pin) | NEW glm5 variant keyed on the glm5 plan (HyperConnections + Kda/Mla mixers); the step pin is untouched |
| snapshot-once registry | `StepParallelLoadConfig`/`StepParallelRuntimeRegistry` HY:94-203 | pattern reused for the glm5 door |
| rank runtime (per-rank Engines, root=owner, host-canonical transport) | `TpE4m3HostBounce` TP:1398 | reuse skeleton; glm5 gate constructor additionally allows same-device dual-rank (gate-only path, never env-reachable) |
| P2P ladder | `configure_native_p2p` TP:8765 | inherited unchanged; BOX arm only |
| NVFP4 shard validation/upload | `Nvfp4BlockMatrix::validate` TP:9007, `upload_tensor_parallel_nvfp4` TP:9838, EP twin TP:10194 | reuse for MLA col cuts + EP expert banks |
| EP route partition + combine contract | `partition_expert_owner_routes` TP:383, `validate_weighted_route_combine` TP:475 | reuse with n_experts=288 parameterized |
| BF16 shard helpers | `bf16_column_shard`/`bf16_row_shard` TP:8500/8523 | reuse for KDA/lm_head col cuts |
| join-diet doors (direct join, prestage, prejoin, shexp overlap) | TP:42-130 | NOT v1; box diet arc, receipts pattern reused |
| engagement markers + performance_claim=false discipline | step markers | same discipline, `[glm5-tp-*]` namespace |
| flag | — | NEW `MEMRA_GLM5_TP` twin (choice argued in LANE.md — the step surface's contract, block law, and refusal strings are model-pinned; sharing the flag would run glm5 through step-worded receipts) |

## 9. Expected value, restated honestly (no measurement claim)

Decode-gap arithmetic: TP-2 today (pre-diet) ~42-43 tok/s; after the diet levers ~60;
with weight-batched verify on top ~120-160 class (TP4). TP-2 alone does NOT reach the
100 bar — this lane is the placement leg of the three-lever path (diet + TP + spec),
and the box A/B in stage 5 is priced against that table, not against 100 directly.
