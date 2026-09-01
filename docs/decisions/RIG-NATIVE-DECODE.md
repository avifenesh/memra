# RIG-NATIVE-DECODE.md — the ST-first decode program for sm_120a (2026-08-15)

Owner direction: build the rig-native layout/program on safetensors — that is the point
of the ST pivot. This doc pins what the measurements say "rig-native" MEANS for decode
on this card, so the kernels get built against the real deficit.

## What the profile says (RTX PRO 6000, Qwen3.8-27B NVFP4, nsys 2026-08-15)

| kernel | transfer/launch | efficiency |
|---|---|---|
| q5_K lm_head matvec | 833 MB | 87% of DRAM peak |
| nvfp4 dual gate+up (fused pair) | ~89 MB | ~80% |
| nvfp4 mr2 singles (attn/ssm projections) | 3–17 MB | 57–60% |

The rp (A6) planes are already coalesced; layout is NOT the decode deficit.
**Efficiency scales with transfer size** — the deficit is launch/ramp overhead on many
small transfers (~272 mmvq launches/token). Rig-native decode therefore = FUSED
multi-tensor walks: fewer launches, larger transfers, same per-(tensor,row) numeric
program (the dual kernel proves +25 efficiency points with bit-identity).

## Increments (each gated: kernel-check cell + argmax MATCH + battery + interleaved A/B)

1. **fused3 QKV (NVFP4, unequal out_f)** — port the Q8_0 unequal-out_f fused recipe
   (block-offset-split mapping) to the nvfp4 mmvq body: wq[6144] + wk[1024] + wv[1024]
   rows in ONE launch (~22 MB) x 16 attn layers. Dispatch: decode_batch Full arm and the
   t-parallel verify's batched projections (m<=16 via the multirow body).
2. **fused4 Linear-mixer projections** (wqkv + gate + beta + alpha; wqkv dominates) —
   same recipe, 48 layers. DONE v0.86.1 (2026-08-15): servegate 4/4 + canary 8/8 on the
   fused binary; interleaved x5 PLAIN arm +6.5% dead-flat (74.8 vs 70.2 decode p50,
   spread +-0.1 — evidence/q38-fused4/). MEMRA_NVFP4_FUSED4=0 rollback seam.
3. **Round-graph (`_dc`) for the spec round** — device-counter twins so the whole
   verify+draft round replays as one graph (~2,200 launches -> ~1). The step35
   ROUND-STREAM machinery is the precedent; qwen35 currently refuses it.
4. **Prefill stays the doctrine plan**: W4A8 MMQ default (1.9x), mxf4nvf4 W4A4 the
   accuracy-gated 762-TFLOPS rung, operands modelopt-shaped from ST.

Ceiling math: trunk mmvq at 75–85% => plain ~85–95 tok/s, masked-spec ~155–175
single-stream on the 6000 — ST-native, ahead of the GGUF arm at median (prod flips).

Head note: lm_head re-encodes Q5_K at ST load (87%-of-peak kernel, best in engine);
"5-bit rig-native" for the head is a <=13% rounding error on one tensor — not the lane.
