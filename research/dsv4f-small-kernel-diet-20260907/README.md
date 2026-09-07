# DSV4F small-kernel diet

Candidate only. Device gates and sampled ABBA are pending.

The existing attention TP2 profile ranks the target families on device 0 as
Sinkhorn 1.63 ms/step, RMSNorm 1.58, routing 0.48, and rowsq 0.32. These are
profiled kernel sums, not promised wall-time savings.

The candidate fuses rowsq, Sinkhorn, and collapse (three launches to one per
HC stage), then Q-LoRA RMSNorm and bf16 pack (two launches to one per layer).
Both retain the f32x 128-thread reduction tree. Register Sinkhorn gathers sums
in ascending order and executes every configured iteration. The norm pack
retains its normalized f32 output before bf16 rounding. Expected identity is
bitwise; no tolerance is admitted. Routing consumes a later projection and
cannot be directly fused with the earlier HC rowsq stage.

`dsv4_tp_ep_sampled_perf_gate MODEL SOURCE --small-kernel-components` captures
the first live HC and Q inputs from the loaded checkpoint on both ranks. Each
component runs 32 interleaved repeats with output guards and CUDA event chain
timing. It checks all HC outputs, scaled mixes, normalized Q and packed Q.

`--small-kernel-abba` runs ten cycles, OFF/ON/ON/OFF, on one loaded model with
fresh state each row. The sampler is radix, temperature 1, top-p 1, top-k 0,
seed 20260907. Each row primes 256 real-source tokens and generates 256 sampled
tokens. The headline is `timing_scope=sample_plus_forward_envelope`, including
sampler, forward and final drain. Tokens, final logits, cache and hidden digests
must agree across every row. The counter asserts actual target-family enqueues:
8 to 3 launches per layer per rank. It does not count other kernel families.

Private controller, binary and hardware receipts are banked in Darklanes under
`research/dsv4f-devpair-20260905/receipts/small-kernel-diet-<sha7>-rN/` and the
private lane report `small-kernel-diet-20260907.md`. No local build, test or
performance gate is allowed by the owner. The dev pair runs CUDA gates; hosted
GitHub CI remains required. This branch is a draft and must not be merged.
