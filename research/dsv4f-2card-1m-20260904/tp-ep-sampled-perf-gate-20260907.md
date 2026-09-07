# DSV4 plain sampled TP/EP performance smoke

Source gate commit: \`296ae3f380f1de908c995dd694800d03a09c922b\`  
Gate binary: \`dsv4_tp_ep_sampled_perf_gate\`  
Source tape SHA256: \`f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded\`

## Protocol

The gate arms the all-layer TP/EP topology before model load and requires:

- native device decode, FP8 dense, matrix MoE, device routing, and device top-k;
- DSpark/MTP disabled;
- 256 real source tokens, primed through one-token steps because batched TP/EP
  cache hydration is not implemented;
- 256 vendor-shape sampled tokens: \`temperature=1\`, \`top_p=1\`, \`top_k=0\`,
  fixed seed \`20260907\`;
- two same-topology repeats, with identical generated-token SHA256 required;
- separate prime/prefill wall and sampled decode wall;
- actual per-rank layer, expert, one-shot AR, GU-M1, GU-half2, down-half2,
  grouped-wo_a, and radix-selector counters.

The strongest compatible gates are armed: GU fusion, GU-M1 tensor-core,
packed-half2 GU/down, grouped \`wo_a\`, and the radix selector. The selector is
expected to remain inert at this short context because its eligibility starts
at compressed \`N=2048\`; its actual count is reported and required to remain
zero.

Looped sampled output is reported with \`eligible=false\` and is excluded from
any performance interpretation. No speculative, PP, cache-hash, or
hidden-state-hash timing rows are emitted.

## Limits

This is an internal sampled-path consistency/performance smoke, not a serving
qualification or oracle-equivalence receipt. Prime is serial single-token
continuation, so its wall is not a batched-prefill claim. The two repeats are
only a small same-TP stability check; no ABBA, concurrency, thermal, or
production admission claim is made here.
