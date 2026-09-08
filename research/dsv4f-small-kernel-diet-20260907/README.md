# DSV4F small-kernel diet

Candidate passes the target component and sampled ABBA gates. Default remains
OFF. This is a plain TP/EP measurement, with no serving-default promotion.

At code source `29a73db7570154cf2d362a33868ce208df458315`, binary SHA256
`363cecb3c501aff04d55ccf75ca6e2e0f0419b642c342b765a372726b17a7cd3`,
both components passed 32 interleaved repeats on each of two RTX PRO 6000
Blackwell ranks, with bit-equal outputs and intact canaries. Ten full-model
OFF/ON/ON/OFF cycles then produced 40 eligible rows, 20 per arm:

```text
ABBA cycles=10 off_tok_s=35.404649 on_tok_s=36.850968 delta_pct=4.085112 digests_identical=true timing_scope=sample_plus_forward_envelope sampler=radix
```

Actual targeted enqueues fell from 344 to 129 per step per rank, five fewer
per layer. All tokens, final logits, cache and hidden digests agree. Private
receipt namespace: `small-kernel-diet-29a73db-r1`; report and raw-log routing
are below. These results do not reach the parent 120 tok/s objective.

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
