# Recompute/load fixture — synthetic, not calibration

`fixtures/recompute-load.csv` contains the full **4 × 4 × 3 = 48** grid:
reusable-prefix tokens `{1024,8192,32768,131072}` × restore decimal GB/s
`{0.1,1,7,14}` × prefill tokens/s `{1000,10000,100000}`.
Every row has explicit 114,688 bytes/token and 0.01 s fixed restore overhead.
These are **synthetic fixture inputs, not Qwen geometry or measured bandwidth**.

The unit test `recompute_load_cross_product_fixture_is_not_a_runtime_default`
parses every row and checks the existing pure `recompute_vs_load` function:

- `load_seconds = prefix_tokens * bytes_per_token / (GB_s * 1e9) + startup_seconds`
- `cold_seconds = prefix_tokens / prefill_tokens_s`
- Load iff strictly cheaper; tie recomputes before admission only.
- Existing tests retain invalid/nonfinite speed refusal, mandatory RequireState,
  no-same-program-proof Load, admitted-state refusal, and nonlinear suffix costs.

## Design reasoning and source

Local source read: `../dsv41f-serving-viz/synth/E-tiered-kv-design-reference.md`
§2.7 lines 138–146, and §3/feasibility lines 289–295. E's [P1] ledger points to
**https://arxiv.org/html/2609.11744v1**, fetched by the reference author on
2026-09-19 (HTTP 200, SHA256
`dc7e650180a447309a24796b5f3794ab6f66ef7c5d023bf21b871d5754ba5bd6`).
No new web retrieval was made by this session.

E records py-kvcache's reasoning: scheduler lookups may decline optional reuse
**before admission**, using per-model/node/tier calibration. Preserve bounded staging
slots and disk/DMA completion, and do not allocate a whole-prefix pool as a prerequisite.
Its break-even law is `max_io(D) = f(D) - [g(D) - t_copy(D)]` (cold TTFT minus
non-copy hit overhead); if nonpositive, no bandwidth rescues that overhead regime.
Partial hits require the `(total_prompt,reused_prefix)` frontier including suffix work.

114,688 B/token is the **reference paper's Llama 3.2 3B example**, used only as a
transparent numeric fixture. It is not substituted into Qwen admission or runtime.
The 48-row linear grid deliberately is not a shipping calibrated policy. The existing
`calibrated_recompute_vs_load` tests include suffix/read/copy/materialization and
overflow, matching the fuller law; actual route costs, artifact/plan/binary/device binding
and measured 64/128/256/512 chunk selection remain pending.
