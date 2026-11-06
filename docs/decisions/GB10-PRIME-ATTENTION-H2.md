# GB10 prime attention: the two-head (h2) twin by default (2026-11-06)

**Status:** decided on one device class (unified-memory 120a family, measured on one DGX Spark).
Discrete cards are unmeasured and keep the four-plane db twin.

## Question

The quantized-KV prime dequantizes the KV prefix into a BF16 workspace and runs `fa_prefill_qw_db`:
one query head per 4-warp CTA, four K/V staging planes plus a P plane (69.9 KB), so one CTA (4 warps)
per SM on GB10. At a 131,051-token served prime it took 84.3 s of 240.6 s, 40 TFLOPS against a
103 TFLOPS bf16 mma.sync peak measured on the device. Which exact twin should GB10 run?

## Decision

`MEMRA_PRIME_KV_H2` resolves like `MEMRA_MMQ_F8F4`: `1`/`0` decide on any device, unset is on for a
unified-memory 120a-family device. The h2 twin runs two query heads of one KV head in an 8-warp
CTA over double-buffered K/V planes (64 KB, one CTA per SM at 255 registers), with P operands taken
from the C fragments in registers. Every row keeps its head's tile sequence, QK/softmax/PV order and
scalar RN conversions, so the output is bit-identical to the db and t3 twins.

## Evidence

Served A/B on one GB10, Qwen3.8-27B NVFP4+Q5_K GGUF with the masked MTP draft, 262,144 context,
interleaved B T3 H2 H2 T3 B, one fresh boot per cell ([`research/gb10-prefill-20261106/attention-ab/`](../../research/gb10-prefill-20261106/attention-ab/)):

| arm | 32,784 tokens | 131,051 tokens |
|---|---|---|
| db (main) | 44.198, 44.232 s | 238.836, 239.106 s |
| t3 (`MEMRA_PRIME_KV_T3=1`) | 42.382, 42.343 s (-4.19%) | 210.073, 209.889 s (-12.13%) |
| h2 (`MEMRA_PRIME_KV_H2=1`) | 42.149, 42.077 s (-4.75%) | 205.908, 205.495 s (-13.92%) |

The answer text is identical across all six boots at both lengths. kernel-check ARC B cells at
t >= 128 (192 x 4097, 1024 x 8224) are bitdiff=0 for the h2, t3 and db twins.

## Rejected

- t3 as the GB10 default: exact and -12.1% at 131k, but h2 is 2.0% faster on the same boots. The
  t3 door is past its decide-by date; its RTX 5090 receipt is a separate decision.
- `MEMRA_PRIME_ATTN_FA2`: a different numerical class (FP32 direct PV, BF16-rounded denominator),
  outside an exact-only lane; not measured here.
