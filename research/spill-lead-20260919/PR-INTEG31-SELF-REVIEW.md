# Self-review: integ31 (C day 21: two stale gates fixed on the gate side; B day 26: memra#539 census, cell and design)

Author's review of the full diff `main..lane/spill-integ31-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `tools/kv-host-spill-failure-gate.sh`: the pool-full assertion follows the server's tenant-share parse
  (`MEMRA_KV_HOST_TENANT_PCT`, default 50, validated 1..100): under the cap it asserts the anchored pre-copy refusal
  text with bytes and budget plus `prefix_host_tenant_rejects >= 1`; at 100 it asserts the insert-path skip line. The
  arm in force is printed. I read both regexes against the server's format strings at the two sites C cites; they
  match the exact text, not a substring.
- `tools/kv-host-contract-fault-gate.sh`: the partial-reject cell derives the entry's plane count from the server's own
  first D2H receipt (`items=N`) and asserts the injected refusal's `1 of M` equals it, then builds the expected literal
  from N; the 12 prior assertions stay, two are added. No per-artifact table.
- Research: C DAY21, receipts from both cards (all ten cells per card), `HOSTPREFIX-DOOR.md` rows; B DAY26,
  `KV-RESIDENCY-DESIGN.md`, harness (`day26-client.py`, `day26-parse.py`, `run-day26-cell.sh`), receipts from both cards,
  STATE files, INDEX rows; the lead record section with ruling 29; this file; battery receipts.

## What I checked
- No engine change beyond A's day-17 commits already integrated in #622 (this branch merges C, which carries them;
  after #622 the engine diff against main is empty).
- Both gate moves are stricter: a message the server never prints under the default can no longer satisfy the pool-full
  check, and the item count is read from the run under test instead of assumed. The rule that a gate match moves only
  to follow a cited deliberate message change is met: C cites `49d1d6f65` (the default 50) and `405466cf7` (the refusal
  suffix, memra#384).
- Verdicts on both cards with A's slice merged: failure, fault and identity gates ALL GREEN in every arm; the
  whole-budget arm is stated as pattern-checked against a banked log, not run.
- B's cell reads ratios and concurrency on each card separately (no cross-box timing); the design note recommends an
  order and leaves the door promotion and the park policy to the owner, with the door's decide-by (2026-09-23) named.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, clippy, censuses, collector pytest,
  engine CPU lib tests, engine and server clippy `-D warnings`, marker census, workflow keys, shellcheck on the two
  gates, perf board, diff-check) and the local 5090 serve-smoke.

## Review round 1 (revuto, addressed in the integ)
- The partial-reject cell's self-consistency check had no floor on the batch size; `items=N >= 2` added so the reject
  stays partial by construction. Banked runs read 34, 32, 18 and 16; the re-run with the floor is lane C's next day.

## What I did not do
- No GPU gate run of my own; the gate verdicts are C's on both cards.
- No default changed; `MEMRA_ADMIT_BY_MEMORY` and `MEMRA_KV_PARK_COMPACT` stay as they are for the owner's decision.
