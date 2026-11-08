# The f32 output of the fused norm before each mixer (round 4 design, 2026-11-08)

Design note before measurement. Follows `WS-DEQUANT-DESIGN.md` (ledger rows 52-53).

## The part

Before every layer's mixer the prime runs one fused kernel (`rms_norm_e4m3v2_f32` after the
residual epilogue, or `add_rms_norm_e4m3v2_f32`) that writes the f32 norm output `h`, the F8F4 v2
e4m3 activation of `h` seeded for the mixer's projection group, and, before a GDN mixer, the q8_1
pair for its beta and alpha projections. In the 131k profile (`profile-r4-main-131k/parts-req.txt`)
`rms_norm_e4m3v2_f32` is 2.996 s over 4,064 calls (0.737 ms); about half the calls are these
pre-mixer norms (the pre-FFN norm already skips its f32 output: `matmul_prefill_ffn_f8f4v2` reads
only the seed).

When every prime reader of `h` takes a seed (the q/k/v or qkv/z projections on the F8F4 v2 tile,
beta and alpha on the q8_1 pair: every 4,096-row chunk of this model), nothing reads the f32 `h`:
its write is 84 MB per call, about 0.39 ms at the device's copy rate, about 2,048 calls at 131k:
0.6-0.8 s (0.5-0.6% of TTFT), about twice that at 260k.

## The change

- `HybridModel::mixer_group_seeded(il, t)`: true when the next mixer's projections are all on the
  F8F4 v2 route (`f8f4_v2_direct`) and none has an f16 mirror or an A4 stamp (those read the f32
  input through another operand or walk the group without a scope); for a GDN mixer, beta and
  alpha too. The two pre-mixer norm sites pass `dst = None` then.
- Fail closed: the seed and the quantize-once scope carry `x_written`. In a scope over an unwritten
  input, the F8F4 entry and the dp4a launcher refuse to quantize the f32 input (no seeded scratch,
  no q8_1 pair), `matmul_group` refuses a weight that returns without taking a seed, and a pending
  unwritten seed with no scope (one weight, or the f16-mirror walk) is refused. A refusal is an
  error, never a silent read.

Exactness: no value changes; the same seeded bytes reach the same GEMMs. The gate is the
prime-path logits byte compare at 16k and 33k against the base, the served fingerprints, and
kernel-check's norm-only arm of `ADD-RMS-NORM-E4M3V2` (the kernel with `dst = None`).
