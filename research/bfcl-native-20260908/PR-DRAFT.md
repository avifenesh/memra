# Spark25 native pack, FP8 activation preservation and prefill admission accounting

Prepared PR body. NOT SUBMITTED: CONTRIBUTING.md:17-19 says a hot-path PR missing any
required proof must not be opened. GATES.md quotes the exact rule and separates it from
the PRO 6000 pair battery required before merge/tag. Remove this paragraph only when the
missing development proof exists and the review findings are resolved.

Spark's fused QKV load was losing the checkpoint's FP8 code/scale layout, and long
prompts reached an hd128-only windowed attention assertion on hd256 heads. This branch
adds the Spark25 native pack, tokenizer and exact-erf GELU program, preserves fused QKV
block slices and the checkpoint's dynamic block-128 FP8 activation requirement, and
instantiates the hd256 windowed single/double-buffer kernels. Unsupported FP8 routes
refuse instead of silently switching to q8_1 activations or per-tensor scale folding.

The server selects the tool parser from the template's GLM tool wire while retaining
the template's own turn rendering. Admission credits already allocated resident prime
slabs, then composes that credit with retained-prefix restore by undoing the cold credit
before replacing full-prompt workspace with suffix workspace. Context, draft, fixed and
reserve charges remain separate. REVIEW.md records the unresolved concat-route credit
risk, GLM malformed-name regression and oracle-input validation defects.

The model remains NativeReference. Strict optimized-logit parity fails the 0.05 band:
approximately 1.97 eager/batch and 2.33 prime versus the FP8 oracle, while the recorded
argmax matches. No tolerance was relaxed. A sampled tool continuation also failed on an
unterminated thinking block. These are open qualification gaps, not NativeQualified or
production admission. Dynamic FP8 component identity does not establish oracle identity
with the BF16 activation container or a different KV format.

## Validation

Historical remote evidence is listed with its receipt path. A rebase does not transfer
an old binary's qualification to the new source head.

| Required or diagnostic surface | Status | Evidence and remaining requirement |
|---|---|---|
| Source-FP32 checkpoint parity | Passed remotely | `receipts/checkpoint-parity.tsv`: max_abs 0.00018501282, max_rel 0.0000083913355, argmax 4=4; native runner SHA 5896d40f2debed6d1bde3c025aeb2e96705a6a80f2c91f7f4701716422d85e7b. Original FP8 block dequant oracle, not optimized FP8 runtime parity. |
| Config, mapping, tensor format unit coverage | Passed remotely | `receipts/gguf-final-test.log`: 241 passed, 1 ignored. Includes row-slice bytes/scales and malformed format refusal. Full rebased-head suite pending. |
| Reference and tokenizer suites | Passed remotely | `receipts/reference-tokenizer-tests.log`: 103 passed across suites, 1 ignored. The log does not independently bind the new rebased head. |
| Windowed hd128/hd256 and dynamic-FP8 component kernel checks | Passed remotely | `receipts/dynamic-kernel-check.log`: ALL GREEN, 93 cells, 22 skipped; dynamic dispatch m=1/2/5/9 bit_bad=0 with differing q8_1 control. `receipts/window256-kernel-check.log` retains the prior window-only run. Skips are not passed cells; this is not the full new-head battery. |
| Resident-slab accounting helper tests | Passed remotely, summary only | `RESULTS.md:44-46` records two passing remote tests; no standalone helper-test log is banked here. Do not treat that summary as an independently bound current-head receipt. |
| Resident-slab pressure, before prefix composition | Passed remotely | `receipts/prime-credit-pressure.json` and `receipts/prime-credit-admission.log`: 3 interleaved windows, baseline 3/12 HTTP200 and 9/12 HTTP429; candidate 12/12 HTTP200, no OOM; 2281 MB credit, unchanged reserve/KV. Candidate 1ea01297a / server 2ca6e5623bade2dfb63d304cd1907d6f854f0cc026b7923906a09bc214f6cc16; baseline 8cfb182ba / server 9c31aee15ad79f1447982cb5178fceab89b4e9d9cac70e71dc8a98ef2ca2bb4b. Serialized pressure is not concatenated-prefill coverage. |
| Retained-prefix and resident credit composition arithmetic | Passed remotely | `receipts/prefix-credit-composition-test.log`: 1 passed, 636 filtered. Includes context/draft/fixed/reserve preservation, full and partial restore, invalid credit and overflow refusal. No new-head GPU proof. |
| Composed credit integration and concatenated-prefill safety | Pending | Fresh source/binary-bound pressure proof after rebase; include the P1 route mismatch in `REVIEW.md`, cold/warm/full/partial restore, batch and serial execution. |
| Strict optimized FP8 oracle parity | Blocked, failed | `RESULTS.md:17-28`: approximately 1.97 eager/batch and 2.33 prime versus the repaired FP8 runtime oracle; strict 0.05 band fails, recorded argmax matches. Per-layer diagnostic needed. No numerical band changed. |
| Full sampled tool continuation | Blocked, failed | `RESULTS.md:30-33`: unterminated thinking block on a sampled tool response. This remains a model/surface failure, not a passing call. Need full sampled continuation and parser regression proof. |
| A real native HTTP path | Historical development evidence only | Pressure receipts contain actual responses with omitted sampling parameters. They do not cover the rebased composition or prove current-head serving qualification. |
| Full current-head kernel, run-gen / prime / argmax, run-spec K=1..8, batching/graph/server regression | Pending | The committed component logs and reference tests do not substitute for these runner paths. `CONTRIBUTING.md:21-43,85-97`. |
| Interleaved main vs branch pp512, pp2048, tg128 at 512 context | Pending | No qualifying N>=3 same-session paired prefill AND decode table. The 49,666-token prefill diagnostic is not this table (`CONTRIBUTING.md:73-83`). |
| Performance regression battery | Pending | No branch-current full battery row. `MEMRA_SKIP_PERF_CI=1` permits a documented push override, not a merge or PR proof waiver. |
| Final non-serving 2x RTX PRO 6000 battery | Blocked in this CPU-only assignment | Required before merge/tag by the rules quoted above. This worker owns no GPU and may not rent or run GPU work. Existing 5090 receipts are not relabeled. |
| Current-head format, pre-push censuses and GitHub CI | Pending | Historical remote checks are recorded in the handoff; no local tests, builds, gates, or CI were run. New documentation and rebased version are not covered by a fresh remote check. |


## Flags and push constraint

No new runtime flag. The standalone oracle inputs MEMRA_REWRITE_ATOL and MEMRA_REWRITE_RTOL
are now catalogued in docs/FLAGS.md with defaults, explicit/unset behavior, reset seams
and failed-qualification receipt pointers. Both remain 0.05 for this lane.

MEMRA_SKIP_PERF_CI=1 is required on push by darklanes CLAUDE.md, "No gates or CI on the
local rig" (lines 192-198). The owner assigned this worker CPU-only documentation and
static review, with no local tests/builds/gates and no GPU access. Commit/push hooks are
disabled per command to comply; stored hook configuration is unchanged. There is no
fresh remote hook result for this rebased head. Required remote gates and GitHub CI are
still pending and still block delivery.

publicity: skipped, maintenance/research work; no release or serving promotion.
