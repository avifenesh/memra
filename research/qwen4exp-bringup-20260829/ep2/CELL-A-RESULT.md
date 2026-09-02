# EP2 cell A ran (box, 2026-09-02): the routed-expert share of the K=5 round is 32.8%

Two runs, same binary lineage, same serving caches (`kv_quant=q8_0/q5_1 idxq=q8`, seams
`idxsel,kvq,idxq`, selgroup AUTO by default), thinkon shape, K=5, 64 tokens, 18 rounds:

| run | receipt | round attributed | `moe.*` share | note |
|---|---|---|---|---|
| ep2A (pre-fix instrument) | `box/spec-profile-k5-ep2A.tsv` | 2262.9 ms (126 ms/round) | 65.8% incl. `moe.dequant` 30.4% + `moe.expert_gemms` 14.5% | the PROMPT PREFILL (all-rows, per-expert executor) was inside the window |
| ep2A-rounds (fixed) | `box/spec-profile-k5-ep2A-rounds.tsv` | 774.1 ms (43 ms/round, sync-bounded) | **32.8%** (sel_grouped 20.5, shared 5.6, router 3.2, sel_bf16/reduce rest) | prefill split out: `prefill:moe.dequant` 688.5 ms = 46.4% of the 1484.9 ms prefill |

**What it settles.** The 22.9 / 26.4 / 48.5% span EP2-DESIGN.md section 5 wanted closed: the
48.5% class (mtp4/5/6 spec-profiles) was the prefill leak, now reproduced and named on this
box; the serving round's routed-expert work is ~33% — the "crediting MTP MoE + shared expert"
band of section 2, and comfortably below the 63.8% a two-card EP2 would need. The EP2 verdict
stands with a measured number instead of a span.

**Top round sections (fixed instrument):** moe.sel_grouped 20.5%, hyper.read 15.2%, mtp.lm_head
9.8%, gdn.proj 9.6%, gdn.conv_scan 7.4%, moe.shared 5.6%, qsa.proj 3.8%, gdn.norm_gate_out 3.4%,
moe.router 3.2%, qsa.sdpa 3.1% (shallow: selection unsaturated), mtp.qsa 2.6%, lm_head 2.6%.

**Next lever this names:** the DRAFT's full-vocab head (`mtp.lm_head`, 248,320 rows) is 9.8% of
the round every chain step — the FR-Spec rank trim (`--draft-trim`, q38 playbook) is a priced
cost-side lever with an accept-side twin.
