GLM TP-2 repeats DSA indexer score/select work on both ranks. This PR splits the pool only for grouped-prime chunks (`t>1`) under `MEMRA_GLM5_TP_INDEXER_SPLIT_PRIME`, default OFF, decide-by 2026-09-22. Decode (`t=1`) ignores the door and keeps the replicated kernel sequence. Both symmetric decode split branches and the scalar PRE workspace handling are deleted. The old flag is no longer read; no second decode door remains. The CHECK flag remains a prime diagnostic.

The prior f32 pair receipt gives prime medians of 36.5982 to 32.8214 s at 128k (-10.3%) and 719.6012 to 470.6511 s at 1M (-34.6%). All 12 timed rows have 160 output IDs, byte-identical within every pair. The prime score/select/exchange/merge arithmetic is unchanged by this follow-up. Receipt: https://github.com/avifenesh/darklanes/pull/527 and `research/glm5-tp-indexer-split-20260908/RESULTS.md`.

Decode split is rejected: 128k falls from 69.852 to 66.947 tok/s (-4.2%); 1M is flat at 59.625 to 59.860 (+0.4%). The blocker is merge cost, about 128 ms per GPU over 159 decode steps, plus 36-40 ms exchange. Shared candidate/exchange/merge kernels remain because prime uses them.

Required post-deploy pair cell on this binary: 1M prime OFF/ON, then 1M decode OFF/ON interleaved x3, with byte-identical IDs in every arm. Confirm the prime win and no decode regression. Vendor-default sampled requests and eight-turn cache-on continuation remain serving gates. This follow-up does not claim a new model-scale result or promote the default.

Remote validation passed: release build, fmt, full-workspace clippy `-D warnings`, 463 engine library tests, 4 CPU target tests, the single-device GPU merge gate and RP=1/2 range bit-identity gates. Raw logs and source hashes are recorded in `research/glm5-tp-indexer-split-20260908/PRIME-ONLY.md`. The CPU regression exercises the production flag helper with unset/OFF/ON values and the retired flag ON, before and after the prime latch. Single-device GPU merge checks compare emitted indices to both the replicated selector and CPU oracle; two-device transport remains a separate pair gate.

No cargo, tests or benchmarks ran on the rig. The pre-commit cargo-fmt hook is replaced for this commit by the matching remote fmt receipt. Push uses `MEMRA_SKIP_PERF_CI=1`; hosted CI remains required.

https://claude.ai/code/session_01TFyR32RLUiSejCgrPm5nNj
