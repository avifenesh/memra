# Self-review: integ33 (B day 27: prefix budget from the served context, park cell; B day 28: graph growth reading, park door decide-by; C day 22: door gates with Move 1 whole)

Author's review of the full diff `main..lane/spill-integ33-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-server/src/worker.rs` (B day 27): `init_prefix_cache_budget` derives its context term through
  `prefix_budget_ctx`, which is the cap rule's own `resolve_ctx` (env when set and valid, else the model's declared
  context), per loaded model; `PrefixCacheBudget::Derived` carries `ctx_source`; the boot line prints the served context
  and its source; an unresolvable context contributes no entry with a WARNING rather than a silent 8192. The two-entry
  count and the boot-free clamp are unchanged. I read the fallback constant's removal and the sort key change (a
  three-tuple now) for the max-entry selection: same ordering, same tie-break by name.
- `docs/FLAGS.md`: `MEMRA_CTX` and `MEMRA_PREFIX_CACHE_MB` rows say the budget follows the served context;
  `MEMRA_KV_PARK_COMPACT` carries `decide-by: 2026-10-06` and its deciding cell. No new read (census clean).
- Research: B DAY27 and DAY28 with receipts from both cards, `KV-RESIDENCY-DESIGN.md` addenda, harness scripts; C DAY22
  with receipts from both cards and the `HOSTPREFIX-DOOR.md` rows; STATE files; INDEX rows; the lead record section; this
  file; battery receipts.
- C's lane carried A's day 18 with its own resolution of the same conflict main resolved; main's side was taken on every
  code conflict (no C-specific code in the file), and the engine diff against main is B's day 27 only.

## What I checked
- No numeric change: the budget derivation changes how many entries the prefix cache may keep, not what any request
  computes; B's after cell reads the same outputs as day 26 on 45 of 45 requests with the warm hits restored.
- The test covers `MEMRA_CTX` set and unset against the cap rule's own resolver; the WARNING path is the fail-closed
  reading of an unparsable value (no entry, not 8192).
- No default changed: the park door and the admission-by-memory door stay OFF; the decide-by dates are the owner's
  decision points (2026-10-06 and 2026-09-23).
- G2's FAIL in B's day-28 cell is reported as measured, not relaxed; G1's PASS is why no booking landed.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, clippy, censuses, collector pytest,
  engine CPU lib tests, server clippy `-D warnings`, marker census, workflow keys, perf board, diff-check) and the local
  5090 serve-smoke if the lock frees within the window (stated either way).

## What I did not do
- No GPU cell of my own; the gate verdicts are C's and B's on both cards.
- The pool-full copy-then-refuse observation stays an owner note on the door's review.
