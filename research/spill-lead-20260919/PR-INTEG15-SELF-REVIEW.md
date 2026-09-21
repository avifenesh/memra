# integ15 self-review (lead, 2026-09-21)

Read in full: A day 12 `crates/memra-server/src/worker.rs` diff (`reclaim_tenant_share`, the hook order at the cap,
the `/metrics` counter, five unit cells), `crates/memra-server/src/lib.rs` (3 lines), the memra-kv `IdentitySlot::leased`
addition, `tools/kv-host-tenant-reclaim-gate.sh` (skimmed: arms, matchers, self-check), `tools/pinned-host-reserve-bench.py`
(skimmed: interleaving, N, order), `HOST-ARENA-STARTUP.md`, TESTING rows; receipts spot-checked.

## Findings
1. **Own row only.** The reclaim walks the demoting tenant's own host entries oldest first, spares the twin and skips
   leased entries by the typed slot state; no other tenant's bytes move (the gate's other tenant keeps equal demote,
   promote and `cached_tokens` counts). The cap and lease protections are unchanged; the refusal stays bounded when
   nothing eligible exists.
2. **Same bytes.** The D2H payload is the same demotion at the same byte count on both arms; eight texts byte-identical
   across arms; the fix arm's only new lines are the eviction and the demote that used to evaporate.
3. **Gate honesty.** Three failed attempts are kept and named (a keyring 401 on `/metrics`, a device-tier skip at a
   256 MiB budget, basic-regex matchers); the final gate carries a matcher self-check. N=1, executed-not-qualified.
4. **#385 is a harness, not a result.** One scaled pair on this box (single versus 8-chunk reserve, N=5 per order,
   both orders, regime stated) proves the harness and reports two medians for this box only; the decision cell is
   the 2x B200 pair and the doc says so.
5. **Push range.** A unset the lane upstream so the perf-ci gate took the merge-base with `origin/main` (the documented
   correction after a merge of main); the hook then ran its full battery and found no engine file. No skip variable.
6. **Nits (not blocking).** The handoff import path (`insert` direct) still evaporates at the cap and is stated as open;
   the gate's fixed byte figures bind to this artifact and budget.

## Verification this review relied on
integ15 CPU battery (`integration-day12/integ15-cpu-battery/`): fmt, portable suites, memra-server suite, clippy,
censuses, collector pytest, A's `verify-day12.py`, perf board, diff-check; local 5090 serve-smoke on this tree
(`integ15-serve-smoke-5090/`). A's target-card gate on both arms. This rig cannot run the model gates.
