# DSpark fused selected-expert dispatch gate

Date: 2026-09-05 UTC

Artifact: exact mixed NVFP4/MXFP4 Safetensors mint of
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

Hardware: two RTX PRO 6000 Blackwell Workstation cards. The provider setting
was 500 W/card; no power or clock bottleneck attribution is made.

The arm is scoped to the three bundled MXFP4 DSpark blocks. It preserves the
host-oracle router ids, weights, and ascending-expert combine order, replacing
only the per-expert projection loop with the already-gated indirect FP4
projection kernels. Trunk prefill, trunk decode, and target verification are
unchanged.

The one-load A/B gate compared the full captured proposal surface:

- main hidden and main projection;
- all three MTP block outputs;
- collapsed state;
- logits before and after the Markov additions;
- Markov embeddings;
- proposal ids and confidence values.

Every f32 component and confidence compared bit for bit; proposal ids were
identical. Ten repeated proposal calls measured 0.069750 s on the per-expert
reference and 0.059587 s fused, a 1.171x drafter-proposal speedup.

Three interleaved whole propose/verify/commit runs of 96 output tokens also had
identical token streams. Median wall was 2.911980 s reference versus 2.882511 s
fused, or 32.9673 versus 33.3043 tok/s: only a 1.010x whole-loop gain.

The existing pinned-host/cache continuation gate passed afterward. The arm
remains explicit and default off because the end-to-end delta is below a
promotion threshold. Its value is reducing DSpark expert-dispatch allocation
and launch topology ahead of graph/workspace work, not a current speed claim.

Durable extracted gate lines are in `remote-gate-lines-20260905.log`.
