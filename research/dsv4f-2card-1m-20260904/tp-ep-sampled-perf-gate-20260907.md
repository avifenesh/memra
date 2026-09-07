# DSV4 plain sampled TP/EP performance smoke

Source gate commit: \`296ae3f380f1de908c995dd694800d03a09c922b\`  
Gate binary: \`dsv4_tp_ep_sampled_perf_gate\`  
Source tape SHA256: \`f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded\`

Hardware runs must compose the down_fusion stale non-owned-slot fix before
building this gate; this source commit alone is not a qualification receipt.

## Protocol

The gate arms the all-layer TP/EP topology before model load and requires:

- native device decode, FP8 dense, matrix MoE, device routing, and device top-k;
- DSpark/MTP disabled;
- 256 real source tokens, primed through one-token steps because batched TP/EP
  cache hydration is not implemented;
- up to 256 vendor-shape sampled tokens: \`temperature=1\`, \`top_p=1\`,
  \`top_k=0\`, fixed seed \`20260907\`; stop at EOS and report the actual
  generated-token and forward-call counts (one forward per generated token);
- two same-topology repeats, with identical generated-token SHA256 required;
- separate prime/prefill wall and sampled decode wall;
- actual per-rank layer, expert, one-shot AR, GU-M1, GU-half2, down-half2,
  grouped-wo_a, and radix-selector counters.

The strongest compatible gates are armed: GU fusion, GU-M1 tensor-core,
packed-half2 GU/down, grouped \`wo_a\`, and the radix selector. The selector is
expected to remain inert at this short context because its eligibility starts
at compressed \`N=2048\`; its actual count is reported and required to remain
zero.

Insufficient, EOS-terminated, or looped output is reported with
\`eligible=false\` and has no headline rate. No speculative, PP, cache-hash,
or hidden-state-hash timing rows are emitted. The final logits/cache/hidden
identities are collected only after timing for the two-repeat consistency
check. Final cache and hidden data must also match between the two ranks.

With the existing `MEMRA_DSV4_NVTX=1` diagnostic enabled, the gate emits a
`TP_EP_DECODE` range over decode steps 32 through 63 in each repeat. The
window is drained at its boundaries only in that instrumented mode. Such
rows are explicitly ineligible for a headline rate. Sync-bracketed
`MEMRA_DSV4_ROUND_PROFILE=1` is refused. An incomplete or early-EOS window
is not a complete profile receipt.

## Limits

This is an internal sampled-path consistency/performance smoke, not a serving
qualification or oracle-equivalence receipt. Prime is serial single-token
continuation, so its wall is not a batched-prefill claim. Each sampled token
is forwarded through one decode call, including the final token when no EOS is
seen; this is not directly matched to an old PP comparison protocol. The two
repeats are only a small same-TP stability check; no ABBA, concurrency,
thermal, or production admission claim is made here.
