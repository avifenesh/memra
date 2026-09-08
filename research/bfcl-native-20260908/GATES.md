# Native Spark branch: PR and merge evidence

Reviewed 2026-09-08. Scope is `lane/bfcl-native-20260908`, rebased code head
`d49ec5e2bdca3c530b9c637ef11f102cf31c2e6b` on `origin/main`
`001c09e5d451798ef8d570f6087dc11527cbcc19`. The previous published head was
`21f803b6cf29d40eb771533691a5e8599db25252`. All 15 replayed commits are patch-identical
according to `git range-diff`; the rebase had no conflicts. Subsequent lane edits are
documentation only. Existing receipts retain their original source and binary identities.

## Actual rules

`CONTRIBUTING.md:13` says main-lane development can use RTX 50-series hardware, but:

> its final pre-merge/pre-tag battery runs on a designated non-serving 2x RTX PRO 6000 pair

It also says:

> the final verification receipt may be supplied by a maintainer with access to the box.

`AGENTS.md:565-570` is explicit about the stage:

> Before any merge or tag, re-run it on a designated non-serving 2x RTX PRO 6000 pair

That is a merge/tag requirement, not by itself a requirement before opening a draft PR.
The separate PR rule at `CONTRIBUTING.md:17-19` is the actual present blocker:

> Every PR touching a kernel, forward pass, dispatch policy, or anything on the decode/prefill hot path must include, in the PR description, evidence for **all** of the following. A PR missing any one of these is incomplete, not "mostly done": do not open it yet.

No draft exception is stated. This branch touches all those surfaces and lacks several
required development proofs below. No PR, including a draft, is opened. `PR-DRAFT.md`
is a prepared body, not a submitted PR. A live `gh pr list --head
lane/bfcl-native-20260908` returned no PR at review time.

`CONTRIBUTING.md:21-43` requires correctness gates and branch-current output;
`:56-71` requires the performance regression battery; `:73-94` requires both prefill
and decode measurements and real runners/HTTP. `AGENTS.md:569-570` names kernel-check,
run-gen argmax and run-spec K=1..8. `docs/RELEASING.md:13-23` separately requires the
release battery on the tagged commit and the release roster. Its `:33-40` warns that
the release battery uses the calibrated argmax-margin gate rather than raw run-gen's
near-tie assertion. Neither wording permits relaxing Spark's optimized-logit band.

## Gate matrix

Passed means a historical remote receipt passed, not that this rebased head is qualified.
Paths below are relative to `research/bfcl-native-20260908/` unless written absolute.

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

## Push policy and flag review

The owner's CPU-only assignment forbids all local tests/builds/gates and forbids touching
the GPU worker's runtime. Darklanes `CLAUDE.md:192-198`, "No gates or CI on the local rig",
says to push engine changes with `MEMRA_SKIP_PERF_CI=1` and disclose the reason. That
variable alone does not stop the formatting/census hooks. This lane uses a per-command
`-c core.hooksPath=/dev/null` for commit/push to honor the explicit no-local-gates order;
no stored hook configuration is changed. No fresh hook pass is claimed. Remote validation
and GitHub CI remain required before delivery.

Static review found two new diagnostic inputs in the inherited branch:
`MEMRA_REWRITE_ATOL` and `MEMRA_REWRITE_RTOL`. Both now have FLAGS rows naming default
0.05, unset/set behavior, reset seam, and the failed numerical verdict. The existing
`MEMRA_ORACLE_OUT` row now names the Spark runner and its checkpoint receipt.
Resident credit uses existing `MEMRA_PRIME_SLABS` and admission controls; hd256 uses
existing `MEMRA_PRIME_DEQW_DB`; there is no new runtime flag. No numerical setting,
runtime default, or tolerance was changed in this CPU-only pass.
