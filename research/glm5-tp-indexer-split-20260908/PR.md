GLM TP-2 repeats DSA indexer score/select work on both ranks. This replaces the existing prime-only query split under `MEMRA_GLM5_TP_INDEXER_SPLIT` with disjoint pool scoring, local top-k candidates, device-side exchange and an exact global merge. The eager MLA middle remains between the existing symmetric PRE/FFN graph pieces.

The f32 paired receipt on 2x B200 gives prime medians of 36.5982 to 32.8214 s at 128k and 719.6012 to 470.6511 s at 1M. Decode is NEGATIVE at 128k (69.852 to 66.947 tok/s) and FLAT at 1M (59.625 to 59.860). The named blocker is merge cost, about 128 ms per GPU across 159 decode steps, plus 36 to 40 ms exchange. The owner explicitly retains the code for the prime wins; the door stays OFF with decide-by 2026-09-22.

All 12 timed rows have 160 output IDs, byte-identical within every pair. All 128k rows also match the saved main control. Both 1M decode profiles match the timed IDs. Full rows, hashes, loop screens, graph engagement and per-device kernel CSVs are in `research/glm5-tp-indexer-split-20260908/RESULTS.md` and its receipt directory. Scorer range gates cover f32 and TC1/2; measured full-model performance here is f32 only. No sampled-serving or default-promotion claim.

Validation: remote release build, formatting and three CPU tests passed. Earlier GPU merge/range tests and 4k/128k full-index CHECK receipts are retained. No cargo, tests or benchmarks ran on the rig. The pre-commit cargo-fmt hook is replaced for this commit by the matching remote fmt receipt. Push uses `MEMRA_SKIP_PERF_CI=1`; hosted CI remains required.

https://claude.ai/code/session_01TFyR32RLUiSejCgrPm5nNj
