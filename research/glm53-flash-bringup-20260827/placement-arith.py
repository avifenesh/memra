#!/usr/bin/env python3
"""GLM-5.3-Flash NVFP4 placement arithmetic (glm53-flash-bringup lane, 2026-08-28).

Reproduces the byte split of the minted NVFP4 artifact from config + tensor census,
and derives the resident footprint at several SLRU cache sizes.

Inputs (all banked in this directory):
  mint-receipts/nvfp4-config.json  - the minted artifact's config (incl. quantization_config)
  glm-index.json                   - source (FP8) safetensors weight_map, 76,108 entries
  mint-receipts/mint-log-summary.txt - "wrote 20 shards, 190.7 GB, 37338 quantized tensors"

Units: GB = 10^9 bytes (the mint log's unit: index total_size 328,326,771,576 = 328.3 GB).
       GiB = 2^30 bytes (nvidia-smi's unit).
Run:   python3 placement-arith.py
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "mint-receipts", "nvfp4-config.json")))
T, V = cfg["text_config"], cfg["vision_config"]

GB, GiB = 1e9, float(2**30)
H       = T["hidden_size"]          # 4096
MI      = T["moe_intermediate_size"]# 2048
DI      = T["intermediate_size"]    # 12288 (dense MLP)
NE      = T["n_routed_experts"]     # 288
NS      = T["n_shared_experts"]     # 1
VOCAB   = T["vocab_size"]           # 154880
QL      = T["q_lora_rank"]          # 1536
KVL     = T["kv_lora_rank"]         # 512
NH      = T["num_attention_heads"]  # 64
QKN     = T["qk_nope_head_dim"]     # 256
VHD     = T["v_head_dim"]           # 256
IHD     = T["index_head_dim"]       # 128
INH     = T["index_n_heads"]        # 32
POOL    = T["index_kpool"]          # 4
LA      = T["linear_attn_config"]
KH, KHD = LA["num_heads"], LA["head_dim"]   # 64 x 128
CONV    = LA["short_conv_kernel_size"]      # 4
QKV     = KH * KHD                          # 8192

N_KDA, N_MLA, N_MOE, N_DENSE = 34, 12, 43, 3
GS = 16  # NVFP4 group size (quantization_config.config_groups.group_0.weights.group_size)

def nvfp4(elems):
    """ON-DISK compressed-tensors NVFP4: fp4 packed (2/byte) + fp8 group scale (1 per 16)
    + the f32 per-tensor `weight_scale_2` macro."""
    return elems // 2 + elems // GS + 4
def bf16(elems):
    return elems * 2

def slot_bytes(in_f, out_f):
    """The SLRU SLOT size = memra's repacked block stride, NOT the on-disk size.
    `HostExps` repacks modelopt (weight + per-16 weight_scale) into one contiguous block:
    row_bytes = in_f / 64 * 36  (nvfp4_repack::repack_modelopt_to_gguf, model.rs).
    The f32 macro is NOT in the block - it rides HostExps::macros and is folded post-matmul,
    which is why residency can never move it (see glm5_moe_residency_gpu.rs)."""
    assert in_f % 64 == 0, in_f
    return out_f * (in_f // 64 * 36)

# ---------------------------------------------------------------- expert mass
routed_blocks = N_MOE * NE * 3                 # 37,152 (layer, expert, {gate,up,down})
block_elems   = MI * H                         # 8,388,608 per projection
block_bytes   = slot_bytes(H, MI)              # one SLRU slot: 4,718,592 B
# gate/up are [in=4096, out=2048]; down is [in=2048, out=4096] -> identical stride, one size class
assert slot_bytes(MI, H) == block_bytes, "glm5 projections must share one SLRU size class"
routed_b      = routed_blocks * nvfp4(block_elems)   # on-disk artifact bytes
shared_b      = N_MOE * NS * 3 * nvfp4(MI * H)

# ------------------------------------------------------------ non-expert mass
non = {}
non["embed_tokens"] = bf16(VOCAB * H)
non["lm_head"]      = bf16(VOCAB * H)          # tie_word_embeddings: false
non["kda"] = N_KDA * (
      bf16(3 * QKV * H)            # q/k/v_proj
    + bf16(H * QKV)                # o_proj (KDA half of the 46)
    + bf16(2 * (KHD * H + QKV * KHD))  # f_a/f_b + g_a/g_b low-rank gate pairs
    + bf16(KH * H)                 # b_proj
    + bf16(3 * QKV * CONV)         # q/k/v_conv1d
    + bf16(QKV + KH + KHD))        # dt_bias, A_log, o_norm
non["mla"] = N_MLA * (
      nvfp4(QL * H) + nvfp4(NH * QKN * QL)     # q_a_proj, q_b_proj
    + nvfp4(KVL * H) + nvfp4(H * NH * VHD)     # kv_a_proj_with_mqa, o_proj
    + bf16(NH * (QKN + VHD) * KVL)             # kv_b_proj (BF16: absorbed at runtime)
    + bf16(IHD * H + INH * IHD * QL + INH * H) # indexer wk, wq_b, weights_proj
    + bf16(2 * IHD + 2 * IHD))                 # k_norm w/b, kpool ape/gate
non["dense_mlp"] = N_DENSE * 3 * nvfp4(DI * H)
non["router"]    = N_MOE * (bf16(NE * H) + 4 * NE)   # gate.weight + e_score_correction_bias
non["mtp"]       = bf16(H * 2 * H)                   # eh_proj
VH, VI, VD, VM = V["hidden_size"], V["intermediate_size"], V["depth"], V["projection_intermediate_size"]
non["vision"] = (VD * bf16(3*VH*VH + VH*VH + 3*VH*VI + 6*VH)
                 + bf16(2*VM*H + H*VM + H*H)
                 + bf16(VH * 3 * 14 * 14 * 2))
non["norms_hc"] = 46 * 2 * bf16(H) + 45 * 6 * bf16(16) + N_KDA * bf16(KHD)
non_b = sum(non.values())

total_b = routed_b + shared_b + non_b
print("=" * 74)
print("GLM-5.3-Flash NVFP4 ARTIFACT BYTE SPLIT  (modeled from config + census)")
print("=" * 74)
print(f"routed experts   {routed_blocks:>7} blocks                = {routed_b/GB:8.2f} GB  ({routed_b/GiB:7.2f} GiB)")
print(f"                 SLRU slot (repacked stride) = {block_bytes:,} B = {block_bytes/1e6:.3f} MB")
print(f"shared experts   {N_MOE*NS*3:>7} blocks               = {shared_b/GB:8.2f} GB")
print(f"EXPERT TOTAL                                   = {(routed_b+shared_b)/GB:8.2f} GB  ({(routed_b+shared_b)/GiB:7.2f} GiB)")
print("-" * 74)
for k, v in sorted(non.items(), key=lambda x: -x[1]):
    print(f"  {k:<14} {v/GB:8.3f} GB")
print(f"NON-EXPERT TOTAL                               = {non_b/GB:8.2f} GB  ({non_b/GiB:7.2f} GiB)")
print("-" * 74)
print(f"MODELED TOTAL    {total_b/GB:8.2f} GB   vs mint receipt 190.7 GB   (delta {total_b/GB-190.7:+.2f} GB)")
print(f"expert share = {(routed_b+shared_b)/total_b*100:.1f}%   non-expert share = {non_b/total_b*100:.1f}%")

# ------------------------------------------------------------------ KV / state
# Quoted from crates/memra-kv/src/lib.rs LatentKvLayer:
#   latent rows : width = kv_lora_rank + rope_head_dim = 512 + 0 (NoPE) f32 = 2 KiB/tok/layer
#   index_rows  : 2 * index_head_dim = 256 f32 = 1 KiB/tok/layer ("12.9 GB at 1M")
#   pool keys   : [max_ctx/pool * index_head_dim] f32   ("1.6 GB")
lat_tok  = KVL * 4 * N_MLA
idx_tok  = 2 * IHD * 4 * N_MLA
def pool_b(ctx): return (ctx // POOL) * IHD * 4 * N_MLA
kda_state = N_KDA * (KH * KHD * KHD * 4 + 3 * QKV * (CONV - 1) * 4)
print("\n" + "=" * 74)
print("KV / RECURRENT STATE PER SEQUENCE   (memra-kv LatentKvLayer)")
print("=" * 74)
print(f"latent  {lat_tok/1024:.0f} KiB/tok   indexer {idx_tok/1024:.0f} KiB/tok   (12 MLA layers)")
print(f"KDA recurrent state (context-independent): {kda_state/GB:.3f} GB")
for ctx in (32768, 131072, 262144, 1048576):
    kv = ctx * (lat_tok + idx_tok) + pool_b(ctx) + kda_state
    print(f"  ctx {ctx:>8}: latent {ctx*lat_tok/GB:6.2f} + index {ctx*idx_tok/GB:6.2f} "
          f"+ pool {pool_b(ctx)/GB:5.2f} + kda {kda_state/GB:.2f} = {kv/GB:6.2f} GB ({kv/GiB:6.2f} GiB)")

# ------------------------------------------------------- residency table
CARD_GIB = 95.6           # nvidia-smi usable on a "96 GB" Blackwell (97,887 MiB); RE-MEASURE on target
BOX = 2 * CARD_GIB
print("\n" + "=" * 74)
print(f"RESIDENT FOOTPRINT ON A 2x96GB BOX  (box usable ~{BOX:.1f} GiB, {CARD_GIB} GiB/card)")
print("=" * 74)
print(f"{'config':<34}{'weights':>10}{'+KV(1M)':>10}{'+KV(128k)':>11}   verdict@1M")
def row(label, w_gib):
    kv1m  = (1048576*(lat_tok+idx_tok) + pool_b(1048576) + kda_state)/GiB
    kv128 = ( 131072*(lat_tok+idx_tok) + pool_b(131072)  + kda_state)/GiB
    ok = "FITS" if w_gib+kv1m < BOX-6 else "DOES NOT FIT"
    print(f"{label:<34}{w_gib:>9.1f}{w_gib+kv1m:>10.1f}{w_gib+kv128:>11.1f}   {ok}")
row("fully resident (all experts)", total_b/GiB)
for frac in (0.10, 0.15, 0.20, 0.30, 0.50):
    slots = int(routed_blocks * frac)
    w = (non_b + shared_b + slots*block_bytes)/GiB
    row(f"SLRU {frac*100:.0f}% hot ({slots} slots)", w)
print("\nNOTE: shared experts are held resident (every token uses them, 43x3 blocks = 0.61 GB).")
print("      Non-expert mass is resident on-device by construction; routed experts are the")
print("      only tier the SLRU cache places. Host bank stays pinned at 175.31 GB.")
mt = 1 * NE * 3 * block_bytes

# ------------------------------------------- KV-AWARE SLOT SOLVER (the recommendation)
# The resident planner (hybrid.rs should_reside) budgets `free - trunk - headroom(2GB)` and has
# NO KV TERM. The SLRU (moe_cache.rs new) then sizes itself on MEMRA_MOE_VRAM_FRAC (0.85) of
# whatever is free. Both run BEFORE the KV plane is allocated, so on this box both will happily
# consume the memory 1M of context needs. Hence: solve for slots explicitly, and pin them.
CTX_OVERHEAD_GB = 8.0   # CUDA ctx + activations + workspace, BOTH cards (2.0/card is the
                        # FLAGS default reserve; 4.0/card is the honest number at 1M chunked
                        # prefill and must be re-measured on the target box)
print("\n" + "=" * 74)
print("RECOMMENDED SLOT BUDGET  (solve for slots, KV taken out FIRST)")
print("=" * 74)
box_b = BOX * GiB
print(f"{'ctx':>9}{'KV':>9}{'free for experts':>19}{'slots':>9}{'resident':>11}{'host-only':>11}")
for ctx in (131072, 262144, 1048576):
    kv = ctx * (lat_tok + idx_tok) + pool_b(ctx) + kda_state
    avail = box_b - non_b - shared_b - kv - CTX_OVERHEAD_GB * GB
    slots = int(avail // block_bytes)
    slots = min(slots, routed_blocks)
    res_b = slots * block_bytes
    print(f"{ctx:>9}{kv/GB:>8.1f}G{avail/GB:>18.1f}G{slots:>9}"
          f"{slots/routed_blocks*100:>10.0f}%{(routed_b-res_b)/GB:>10.1f}G")
print("\nPIN BOTH, per card (neither default is safe here):")
print("  MEMRA_MOE_RESIDENT=0   - the planner's budget has no KV term; at ~88 GB/card of")
print("                           expert mass against a ~93 GB/card budget it answers RESIDENT")
print("                           and leaves nothing for a 1M KV plane.")
print("  MEMRA_MOE_SLOTS=<N/2>  - else the SLRU takes MEMRA_MOE_VRAM_FRAC (0.85) of free VRAM,")
print("                           which is also measured before KV exists.")

print(f"\nMTP (NextN) layer expert bank alone: {mt/GB:.2f} GB - host-only while MTP speculation is off.")
print(f"Cold per-token miss traffic (arithmetic projection, NOT a measurement):")
print(f"  8 active x 3 proj x {N_MOE} MoE layers x {block_bytes/1e6:.2f} MB = "
      f"{8*3*N_MOE*block_bytes/GB:.2f} GB/token if every routed block missed.")
