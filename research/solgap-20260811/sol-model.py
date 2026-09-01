#!/usr/bin/env python3
"""SOL gap audit cost model — Step-3.7-Flash IQ4_XS PP-2 on 2x RTX PRO 6000.

Every input is a committed receipt or the pinned config.json. The optional sigrouter2 summary
overrides the c1/c8 measured anchors while preserving the frozen geometry and ceiling model.
Sources are cited inline. Run: python3 sol-model.py [--sigrouter2-summary summary.json]
"""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument(
    "--sigrouter2-summary",
    type=Path,
    help="interleaved increment-2 performance summary used for c1/c8 anchors",
)
args = parser.parse_args()

GB = 1e9

# ---- model geometry (research/step37-bringup-20260802/raw/config.json) ----
H = 4096            # hidden_size
V = 128896          # vocab_size
HD = 128            # head_dim
NKV = 8             # num_attention_groups
NH_FULL, NH_SWA = 64, 96
L = 45              # main layers
N_FULL, N_SWA = 12, 33
FFN_DENSE = 11264   # intermediate_size (3 dense layers)
E, TOPK = 288, 8    # moe_num_experts, moe_top_k
EFF = 1280          # moe_intermediate_size
SH = 1280           # share_expert_dim
MOE_LAYERS, DENSE_LAYERS = 42, 3
SWA_WIN = 512

# ---- artifact encoding (step37-bringup sizes-math: official IQ4_XS 4.26 bpw) ----
BPW = 4.26
BPP = BPW / 8.0     # bytes per param

# ---- rig (docs/PERFORMANCE.md rigs table; box1 = Server Edition pair) ----
BW_CARD = 1.79e12   # B/s HBM per PRO 6000 (188 SM class, ~1.8 TB/s)
# int8 tensor-core reference: 219 TFLOP/s measured on 82-SM sm_120
# (research/sm120-empirical-capabilities.md via GEMM-PLAN.md) -> SM-scaled
INT8_TFLOPS_CARD = 219e12 * 188 / 82

# ---- measured anchors (receipts) ----
# specpp2-20260810 anatomy: T=1 verify == plain B=1 forward shape
STAGE0_MS, STAGE1_MS = 5.557, 6.179        # raw/anatomy/anatomy-lines.log
B1_TOKS = 85.041                            # eagerpar-20260810 promoted c=1 N=5
C2_TOKS, C4_TOKS = 119.943, 144.665         # eagerpar live A/B
C8_TOKS = 8 / (48.30 / 1e3)                 # throughput-20260810 grouped-on c=8 p50
PP4096_TOKS = 692.7                          # grouped-serve-20260810 solo prefill
PP4096_LOADED = 577.6                        # concprefill-20260808 c=4 mixed
AGG_C64 = 129.70                             # throughput-20260810 full-window ceiling

sigrouter2_summary = None
if args.sigrouter2_summary is not None:
    sigrouter2_summary = json.loads(args.sigrouter2_summary.read_text(encoding="utf-8"))
    if sigrouter2_summary.get("schema") != "memra.sigrouter2.perf.v1":
        raise SystemExit(
            f"unexpected sigrouter2 summary schema: {sigrouter2_summary.get('schema')!r}"
        )
    B1_TOKS = float(sigrouter2_summary["points"]["c1"]["default_median_tok_s"])
    C8_TOKS = float(sigrouter2_summary["points"]["c8"]["default_median_tok_s"])

def attn_params(nh):
    # q,o: H*(nh*HD) each; k,v: H*(NKV*HD) each; head-wise gate: H*nh; norms tiny
    return 2*H*nh*HD + 2*H*NKV*HD + H*nh

ATTN = N_FULL*attn_params(NH_FULL) + N_SWA*attn_params(NH_SWA)
DENSE = DENSE_LAYERS * 3*H*FFN_DENSE
SHEXP = MOE_LAYERS * 3*H*SH
ROUTER = MOE_LAYERS * (H*E)
LMHEAD = V*H
EXP_PROJ = H*EFF                 # params per expert per projection
EXP_ONE = 3*EXP_PROJ             # one expert, gate+up+down

def uniq_experts(draws):
    return E * (1 - (1 - 1/E)**draws)

def decode_bytes_per_chunk(B):
    """Weight bytes read for one B-row decode chunk (trunk once, experts by
    unique-id since grouped dispatch is live at t>1; per-expert at B=1)."""
    trunk = (ATTN + DENSE + SHEXP + ROUTER + LMHEAD) * BPP
    ue = TOPK if B == 1 else uniq_experts(B*TOPK)
    experts = MOE_LAYERS * ue * EXP_ONE * BPP
    return trunk + experts, ue

if sigrouter2_summary is not None:
    print("=== Sigrouter2 measured-anchor override ===")
    print(f"  source: {args.sigrouter2_summary}")
    print(f"  c1 default median: {B1_TOKS:.6f} tok/s")
    print(f"  c8 default median: {C8_TOKS:.6f} tok/s")
    print()

print("=== Step-3.7 active-parameter bill (per token, B=1) ===")
tot = ATTN + DENSE + MOE_LAYERS*TOPK*EXP_ONE + SHEXP + ROUTER + LMHEAD
for name, p in [("attention (all layers)", ATTN), ("dense FFN x3", DENSE),
                ("routed experts 42L x top8", MOE_LAYERS*TOPK*EXP_ONE),
                ("shared experts", SHEXP), ("router", ROUTER), ("lm_head", LMHEAD)]:
    print(f"  {name:28s} {p/1e9:6.2f} B params  {p*BPP/GB:5.2f} GB @4.26bpw")
print(f"  {'TOTAL active':28s} {tot/1e9:6.2f} B params  {tot*BPP/GB:5.2f} GB/token")

print("\n=== Decode weight-streaming SOL vs measured (serial PP-2: one card active at a time) ===")
print(f"{'B':>3} {'GB/chunk':>9} {'uniqExp':>8} {'SOL ms':>7} {'SOL tok/s':>10} "
      f"{'meas tok/s':>10} {'%SOL':>6}")
for B, meas in [(1, B1_TOKS), (2, C2_TOKS), (4, C4_TOKS), (8, C8_TOKS)]:
    byts, ue = decode_bytes_per_chunk(B)
    sol_ms = byts / BW_CARD * 1e3           # serial stages: whole bill at 1-card BW
    sol_tok = B / (sol_ms/1e3)
    print(f"{B:>3} {byts/GB:>9.2f} {ue:>8.1f} {sol_ms:>7.2f} {sol_tok:>10.0f} "
          f"{meas:>10.1f} {meas/sol_tok*100:>5.1f}%")

print("\n=== Achieved per-stage bandwidth at B=1 (specpp2 anatomy stage times) ===")
b1_bytes, _ = decode_bytes_per_chunk(1)
# stage0 = layers 0..22 (22 layers), stage1 = 23 layers + lm_head; split bytes ~ by layers
s0 = b1_bytes * 22/45 * (tot - LMHEAD)/tot  # crude: head lives on stage1
s1 = b1_bytes - s0
print(f"  stage0: ~{s0/GB:.2f} GB / {STAGE0_MS} ms = {s0/(STAGE0_MS/1e3)/GB:.0f} GB/s "
      f"= {s0/(STAGE0_MS/1e3)/BW_CARD*100:.1f}% of card")
print(f"  stage1: ~{s1/GB:.2f} GB / {STAGE1_MS} ms = {s1/(STAGE1_MS/1e3)/GB:.0f} GB/s "
      f"= {s1/(STAGE1_MS/1e3)/BW_CARD*100:.1f}% of card")
print(f"  reference: q27 Q8_0 dense decode on the same 188-SM class achieved 88-96% "
      f"per class, 91% aggregate (research/q27-deepdive-20260805)")

print("\n=== KV bytes per token (q8_0 K 34B/32, q5_1 V 24B/32; memra-kv/src/lib.rs) ===")
kvd = NKV*HD
k_b, v_b = kvd//32*34, kvd//32*24
print(f"  per layer per token: K {k_b} B + V {v_b} B = {k_b+v_b} B")
for depth in [512, 4096]:
    full = N_FULL * depth * (k_b+v_b)
    swa = N_SWA * min(depth, SWA_WIN) * (k_b+v_b)
    print(f"  read at depth {depth:5d}: {(full+swa)/1e6:6.1f} MB/token "
          f"({(full+swa)/b1_bytes*100:.1f}% of the B=1 weight bill)")

print("\n=== Prefill compute SOL (grouped, pp4096 solo) ===")
flop_tok = 2 * tot
ach = PP4096_TOKS * flop_tok
print(f"  ~{flop_tok/1e9:.1f} GFLOP/token active -> measured {PP4096_TOKS} tok/s = "
      f"{ach/1e12:.1f} TFLOP/s aggregate")
print(f"  int8-TC card reference (SM-scaled from measured 82-SM 219 TFLOP/s): "
      f"{INT8_TFLOPS_CARD/1e12:.0f} TFLOP/s")
print(f"  fraction of ONE card's int8 peak: {ach/INT8_TFLOPS_CARD*100:.1f}%")
print(f"  per-expert prefill GEMM shape at 4096 tok: m~{4096*TOPK/E:.0f} rows, "
      f"n={EFF}, k={H} (small-m regime)")

print("\n=== Launch/sync bill per token, B=1 (step35_decode_batch_layers walk) ===")
# fixed per layer: rms, quantize, 4x matmul_pre (q/k/v/gate), 2x qk-rms, rope,
# kv-append, fa_decode, head-gate, wo, add_rms_norm, add  (~15)
per_layer_fixed = 15
# MoE via slab fused pair (hybrid_forward.rs slab_fused_may_fire): router gemv,
# quantize z, gate_up_silu8_q8, quantize act, down8_fma_q8 (~5) + shexp (~4)
moe_fused = 9
# clamped layers 43/44 fall through to the per-expert sequential loop
moe_seq = 2 + TOPK*4
launches = (per_layer_fixed + moe_fused) * (L - 2) + (per_layer_fixed + moe_seq) * 2
print(f"  ~{per_layer_fixed}+{moe_fused} launches/layer fused-pair (43 layers), "
      f"~{per_layer_fixed}+{moe_seq} on the 2 clamped layers")
print(f"  -> ~{launches} kernels/token + lm_head + sampler epilogue")
print(f"  q27 dense reference: 1015 launches/token = 7.5% launch-gap tax at 92.5% busy")
if sigrouter2_summary is None:
    print(f"  step35 adds 42 PER-LAYER HOST ROUTER SYNCS/chunk (sigmoid host oracle: "
          f"e.dtoh(router logits) in moe_route_cfg) — the launch queue drains every MoE layer")
else:
    print("  sigrouter2 default keeps sel/w device-resident through expert dispatch; the 42")
    print("  increment-1 selected-id/weight readbacks and their stream syncs are removed")
print("\n=== Per-chunk host sync count at c=64 (8 serial B=8 chunks/tick) ===")
if sigrouter2_summary is None:
    print(f"  42 router D2H syncs x 8 chunks = 336 full-stream syncs per 381.7 ms tick")
else:
    print("  increment 1: 42 router D2H syncs x 8 chunks = 336 full-stream syncs per tick")
    print("  sigrouter2 default: 0 router D2H syncs in the resident Step dispatch arm")
print(f"  + 1 [B]-u32 token readback per chunk (receipted flat to defer)")

print("\n=== B>1 per-row attention walk (decode_batch.rs:1446-1481) ===")
b = 8
per_row = 4  # prepare/append_kv, dtod q_row, fa_decode, dtod a_row
print(f"  B={b}: {L} layers x {b} rows x {per_row} launches = {L*b*per_row} "
      f"per-row launches per chunk (the eagerpar class, B>1 edition)")

print("\n=== PP boundary at decode (receipted) ===")
print(f"  B=1: 16 KB f32 peer copy, tx 0.013 ms + rx 0.014 ms "
      f"= {(0.013+0.014)/(STAGE0_MS+STAGE1_MS)*100:.2f}% of the token walk (specpp2 anatomy)")

if sigrouter2_summary is not None:
    print("\n=== Sigrouter2 SOL fraction movement (same-lock increment-1 control) ===")
    for B, key in ((1, "c1"), (8, "c8")):
        byts, _ = decode_bytes_per_chunk(B)
        sol_tok = B / (byts / BW_CARD)
        point = sigrouter2_summary["points"][key]
        old = float(point["inc1_median_tok_s"])
        new = float(point["default_median_tok_s"])
        old_fraction = 100.0 * old / sol_tok
        new_fraction = 100.0 * new / sol_tok
        print(
            f"  {key}: {old_fraction:.3f}% -> {new_fraction:.3f}% SOL "
            f"({new_fraction - old_fraction:+.3f} pp; throughput {new / old - 1:+.3%})"
        )
