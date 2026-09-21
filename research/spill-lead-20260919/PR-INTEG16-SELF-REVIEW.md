# integ16 self-review (lead, 2026-09-21)

Read in full: B day 15 `crates/memra-server/src/worker.rs` diff (the SLRU removal: segments, promotion, demotion,
`room_victim_with`, the two doors; what stayed: refusals, throttle, leased preflight, `evict_to_bytes` order),
`worker/host_glm.rs`, `docs/decisions/PREFIX-CACHE-POLICY.md`, the FLAGS "Removed doors" ledger, SERVING and TESTING,
the twin gate's V4 change, `verify-day15.py`; the deleted harness read at `9466b891`; receipts spot-checked.

## Findings
1. **Pre-registration held.** Rule and harness were committed before the runs; the verdict line carries N, both
   orders, the thermal regime and the digest precondition (28/28 identical across arms, runs and the cache-off boot).
2. **Mechanism, not timing.** The 7.3 percent computed-token gap is deterministic across all ten pairs in both orders
   and is explained by the server's own eviction lines; it does not depend on the card. Ruling 19 lands it as the naked
   default with the one-rig caveat stated and the 5090 confirmation owed.
3. **Door hygiene.** Both env reads, the dispatch arms, the tests that existed only for them, the harness and the FLAGS
   rows are gone in the same PR; the rows moved to the Removed doors ledger with the receipt pointer; a decision record
   exists. What the day-14 fix added (refusal lines, throttle, pressure-relief order) is kept and still tested.
4. **Reconciliation with #597.** One conflict hunk resolved by keeping A's whole tenant-share block; 754 server tests
   including A's seven `tenant_share` cells; A's gate parses unchanged lines. Landed binary on the card: twin gate PASS
   with day-14 digests, serve-smoke and cache-meter 0 failed.
5. **Nits (not blocking).** The deleted harness lives only in history; the owed 5090 cell needs it checked out from
   `9466b891`. The Removed doors entry should be cross-linked from `docs/decisions/README.md` (B added the record row).

## Verification this review relied on
integ16 CPU battery (`integration-day12/integ16-cpu-battery/`): fmt, portable suites, memra-server suite, clippy,
censuses, collector pytest, `verify-day15.py` (B), perf board, diff-check; local 5090 serve-smoke
(`integ16-serve-smoke-5090/`). B's target-card A/B and landed-binary gates. This rig cannot run the model gates.
