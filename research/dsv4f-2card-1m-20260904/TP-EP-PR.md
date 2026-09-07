## Summary

Add a gate-only all-layer expert-parallel DSV4 decode program as the foundation for the owner's PP-exit work. Attention/cache state is currently replicated; routed experts are partitioned by ID, with a named rank-order FP32 slot sum. This does not claim attention tensor sharding, production qualification, or 120 tokens/s.

- Each rank owns all 43 trunk layers and its local expert bank. No hidden PP fallback when the experimental topology is selected.
- Preserve independent rank cache/checkpoint commit and actual position/counter receipts. Plain-only: DSpark/MTP, generic host-cache serving and unsupported batched paths refuse explicitly.
- Use the already-gated native one-shot out-of-place primitive. Fix stale non-owned contribution slots, including poisoned-reuse regression on both target GPUs.
- Add bounded internal-consistency and sampled performance gates with actual counters, raw tokens, final state identity and loop/EOS exclusion.
- Check both sticky one-shot refusal words before either cache commits; drain both ranks on submission errors and publish position only after logits succeed. A per-rank 40043/40044 red arm checks unchanged persistent cache/position and a refused retry after clearing the diagnostic word.
- The scoped worker candidate preserved exact identity but did not win: 23.7555 versus the prior 24.3093 tok/s same-protocol serial-TP receipt. Its runtime dispatch, endpoint API and CLI seam were removed in this branch. No PP comparison was rerun.

## Evidence and limits

The sampled gate uses a 256-token real-source prime and 256 sampled forward calls, two repeats, T=1/top_p=1/top_k=0, fixed seed. Both corrected serial-TP rows are eligible and agree in tokens/final logits/cache/hidden data. The earlier pooled rate is 24.3093 tok/s, not the 120 objective. That timing predates the token-boundary refusal fix; the new boundary needs fresh target evidence. This is internal consistency and target execution evidence, not a full model-oracle or public serving claim.

Target binary pins and no-go worker receipts are in the lane documents. The delivered one-shot component gate is already on main via #319. The later current-main rebase changes upstream release/server/DSpark code, not the pinned DSV4 arithmetic; hosted checks on this exact head remain required. Native CPU/build tests are being rechecked remotely, never on the user's local machine.

Local CI is explicitly owner-waived for September 7, 2026 while the local machine is in use. Unavailable Revuto may be waived after independent code review; provider failures are not represented as successful review. No production fleet pin, model-support state, pricing, or public performance claim changes.

Related to #4, which remains open. Next performance work is paired MoE/reduction/shared-tail graph replay and actual attention/expert tensor slicing, not PP tuning.
