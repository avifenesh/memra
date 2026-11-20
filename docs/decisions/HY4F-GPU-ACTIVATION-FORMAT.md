# Hy4-flash GPU route: GEMM inputs as two BF16 terms, not one (2026-11-20)

**Status:** decided with the GPU route (`hy4f-gpu`, numeric class `hy-v4-flash-bf16x2-activations`),
which passes CheckpointParity against the f64 oracle tapes. Scope: Hy4-flash on one GB10.

## Question

The artifact is weight-only NVFP4 (W4A16) with FP8 128x128 attention projections. The natural
GPU program multiplies BF16 activations by the stored weights on the tensor cores. Does a route
whose GEMMs take single BF16 inputs pass the checkpoint gate (argmax equal at every position and a
band set from the pinned program's own backends) on the pinned requests?

## Measurement

Against the f64 oracle tapes of step 7, each program's router and indexer picks forced to the
oracle's (research/modelplan-onboarding-hy4flash-20261119/gpu/):

| program | `long`: max abs logit diff | argmax equal of 2,112 |
|---|---:|---:|
| this route, single BF16 inputs, BF16 cache | 13.3 | 2,096 |
| pinned transformers program, its `bf16` backend | 18.9 | 2,061 |
| pinned program, only its linear inputs rounded to BF16 (`bf16a`) | 21.6 | 2,092 |
| pinned program, linear inputs as two BF16 terms (`bf16x2a`) | 0.0555 | 2,112 |
| this route, two BF16 terms, F32 cache | 0.054 | 2,112 |
| pinned program, f32 (step 7's band backend) | 0.0567 | 2,112 |

The single-BF16 programs all fail at position 756, where the program amplifies a perturbation of
the linear inputs about a thousandfold between layers 17 and 39 (the f32 backend's router scores
move by 2e-6 at layer 1 and 2e-3 at layer 39).

## Decision

Every weight GEMM of the route carries its input as `hi = bf16(x)` and `lo = bf16(x - hi)` and
multiplies both on the tensor cores with F32 accumulation; the weights stay exact (FP8 and NVFP4
codes times their scales are exact in BF16, the block scale and `weight_scale_2` apply in F32);
the cache is F32. The class band is twice the `bf16x2a` backend's forced spread, and the gate
runs the route with the oracle's selections applied, as that backend ran.

## Rejected

- Single BF16 inputs: no band can hold the argmax at every position, and a band wide enough for
  its spread (twice 21.6 logits) refuses no wrong program. That arm was the adopted route with
  the `lo` term dropped and the cache rows stored as BF16; its code was replaced, its tapes are
  in the lane's receipts (`route-*0*`).
- The pinned program's `bf16` backend as the band backend: it rounds the weights to BF16 (the
  checkpoint gate's dequantized-weight red arm) and the streams, and sits 18.9 logits from f64.

## Cost

Two MMAs per K step instead of one: prefill GEMM work doubles; decode reads each weight once, so
it is bandwidth-bound either way. The F32 cache is twice a BF16 one: 33.8 KB a token on the
indexed layers.
