# The f32 output of the fused norm before each mixer (round 4 design, 2026-11-08)

Design note, then the measurement (ledger row 54). Follows `WS-DEQUANT-DESIGN.md` (ledger rows 52-53).

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
prime-path last-row logits byte compare at 16k and 33k against the base and the served
fingerprints. kernel-check's `ADD-RMS-NORM-E4M3V2` does not run the two forms this ships
(`rms_norm_e4m3v2_f32` with `dst = nullptr`, either entry with `dst = nullptr` and the q8_1 pair),
and no cell made a refusal fire: kernel-check `NORM-SKIP-UNWRITTEN` (ledger row 54b) now covers
both null-dst forms with the q8_1 pair and three refusals, and the prime-path logits are identical
on Qwen3.8-27B, Ornith-1.5-35B-A3B and Qwen3.5-9B.

After the independent review the predicate is narrower: the full-attention arm excludes the
sliding-gated (step35) program and a separate head gate or TP projections, which read `h` again
after the group, and beta / alpha must be NVFP4 on the dp4a arm (Stage-B fast, under the GEMM
output cutoff), the only arm that reads the group's q8_1 pair. The F8F4 v1 branch and the
A4 and f16 group walks also refuse an unwritten input.

## Result (unit 1, ledger row 54)

Served B N N B B N, 3 boots per arm: 22.691 -> 22.466 s at 32k (-0.99%), 120.730 -> 119.709 s at 131k
(-0.85%), 319.329 -> 317.468 s at 260k (-0.58%), complete separation at every length, identical
answers. The prime-path logits are byte-identical to the base at 16k and 33k, and no run hit a
fail-closed refusal. The gain beat the estimate (1.02 s against 0.6-0.8 s at 131k).

## The stack's 131k profile after it (`profile-r4-stack-131k/`)

One cold 131,051-token served prefill of this head (#1108 + #1110 + this) under Nsight Systems,
unit 1, request window as in round 4's profile: TTFT 119.594 s against main's 122.731 s under nsys
(-2.6%); trunk 117.96 s; kernels 117.84 s: F8F4 v3 tile 55.07, h2p 40.77, GDN chunk scan 6.02,
image pass 5.95, conv + L2 2.29, glue 2.25 (3.23 on main), other 1.57, matvec 1.48, q path and gate
1.39, KV dequant workspace 0.91 (1.37); idle gaps inside the window 0.26 s (the 31 fill gaps are
gone). Raw report: evidence snapshot `92abcad6`.
