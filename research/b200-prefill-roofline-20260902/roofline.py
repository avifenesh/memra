#!/usr/bin/env python3
"""4k-token prime roofline for GLM-5.3-Flash-NVFP4 on the 2x B200 PP2 posture.

Geometry from research/glm53-flash-bringup-20260827/mint-receipts/nvfp4-config.json.
Chunk schedule from crates/memra-engine/src/hybrid_forward.rs prime_chunk_tokens().
"""
H, MI, DI = 4096, 2048, 12288
NE, NU, NS = 288, 8, 1
NH, QKN, VHD, QL, KVL = 64, 256, 256, 1536, 512
IHD, INH, POOL, TOPK = 128, 32, 4, 2048
KH, KHD, CONV = 64, 128, 4
QKV = KH * KHD
L, N_KDA, N_MLA, N_MOE, N_DENSE = 45, 34, 11, 42, 3
SLOT = MI * (H // 64 * 36)            # repacked NVFP4 stride, one expert projection
PEAK_BF16 = 2.2e15                     # per B200, dense bf16 TC
HBM = 8.0e12                           # per B200

def active_params():
    moe   = N_MOE * (NU + NS) * 3 * MI * H
    dense = N_DENSE * 3 * DI * H
    rout  = N_MOE * NE * H
    kda   = N_KDA * (3*QKV*H + H*QKV + 2*(KHD*H + QKV*KHD) + KH*H)
    mla   = N_MLA * (QL*H + NH*QKN*QL + KVL*H + H*NH*VHD + NH*(QKN+VHD)*KVL
                     + IHD*H + INH*IHD*QL + INH*H)
    return dict(moe=moe, dense=dense, router=rout, kda=kda, mla=mla)

def report(t, chunks):
    ap = active_params(); tot = sum(ap.values())
    gemm = 2 * tot * t
    kv = min(t, TOPK)
    attn = N_MLA * 2 * (NH * t * kv * KVL * 2)          # absorbed QK + PV
    absdec = N_MLA * 2 * (t * NH * QKN * KVL * 2)       # absorb_q + decompress_v
    idx = N_MLA * (INH * t * (t//POOL) * IHD * 2)
    kdaf = N_KDA * (t * KH * KHD * KHD * 4)
    flops = gemm + attn + absdec + idx + kdaf
    wbytes = chunks * N_MOE * NE * 3 * SLOT             # expert slab, once per chunk per layer
    print(f"--- t={t}  chunks={chunks} ---")
    print(f"active params      {tot/1e9:6.2f} B  (moe {ap['moe']/1e9:.2f} kda {ap['kda']/1e9:.2f} mla {ap['mla']/1e9:.2f})")
    print(f"FLOPs total        {flops/1e12:8.1f} TFLOP   (proj/moe GEMM {gemm/1e12:.1f}, mla attn {attn/1e12:.1f},")
    print(f"                                            absorb/decomp {absdec/1e12:.2f}, idx {idx/1e12:.2f}, kda {kdaf/1e12:.2f})")
    print(f"expert weight read {wbytes/1e9:8.1f} GB    ({chunks}x {N_MOE*NE*3*SLOT/1e9:.1f} GB one-pass)")
    print(f"arith intensity    {flops/wbytes:8.1f} FLOP/B   (B200 ridge {PEAK_BF16/HBM:.0f})")
    print(f"bf16-TC time floor {flops/PEAK_BF16*1e3:8.1f} ms   HBM floor {wbytes/HBM*1e3:6.1f} ms")
    print(f"  -> stage floor (PP2 serial, layers split 24/21): same wall, work is sequential")
    return flops, wbytes

for t, c in ((4096, 8), (24576, 6), (41900, 11), (131072, 32)):
    f, w = report(t, c)
    print()
print("measured TTFT: 4k 5.5 s, 41.9k 17.6 s, 512k 789 s, 66 tok 0.2 s")
f4, _ = report(4096, 8)
print(f"\n4k effective rate = {f4/5.5/1e12:.1f} TFLOP/s  = {f4/5.5/PEAK_BF16*100:.2f}% of one B200 bf16 peak")
f42, _ = report(41900, 11)
print(f"41.9k effective rate = {f42/17.6/1e12:.1f} TFLOP/s = {f42/17.6/PEAK_BF16*100:.2f}%")
