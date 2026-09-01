# Excluded attempt 8 — repeated known Q27 8 GiB EOS

This continuation incorrectly treated the excluded Q27 repetition 1 / 8,192 MiB boot as needing
a passing replacement. Subsequent orchestrator steering was authoritative: keep that boot
excluded under the `cx-eosclass` reproduction and continue with the rest of the capacity grid.
No row in this directory is scoreable.

The rerun reproduced the same failure immediately at c=4. Working key 87 was a full 4,860-token
hit but returned 11/60 tokens with `finish_reason=stop` and text SHA-256
`ddbbcf35ae93821874be64e15f63feecb2471bbe6ea0c23b63fa00b1e98a9b73` (request
`cmpl-9e8fc49b8c6dd223c864e4efb320636b`). The cell completed 19/20 exact-length responses; its
counters reconcile 20 admitted/completed, three hits, 17 misses, 14,580 cached tokens, 1,151
output tokens, zero admission defers, and zero OOM parks. The following c=5 cell passed. The c=6
cell was not allowed to finish.

After confirming the runner/timeout parentage, the lane sent `TERM` only to its owned sweep
timeout. The fail-closed runner recorded `sweep.exit=143`, drained and stopped its server, and
released both locks. The server failure scan is empty. The 250 ms two-GPU sidecar contains 682
samples from `2026-08-13T02:03:42.302Z` through `2026-08-13T02:07:16.917Z`; GPU1 stayed at
0 MiB and 0% utilization throughout. Compute-app checks before seed and before c=4/c=5/c=6 all
passed with only the owned physical-GPU0 server PID present.

The post-cleanup receipt shows all owned runner/sweep PIDs absent, no memra server, and no target
port listener. A different lane acquired the global lock and started `kernel-check` on GPU0 after
this lane had released it; that foreign PID is preserved in the receipt and was left untouched.
