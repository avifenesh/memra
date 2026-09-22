# Self-review: integ45 (A days 28 and 29: the bundle checksum off the tick on a helper thread; a hit on a `Hashing` entry parks one tick)

Author's review of the full diff `main..lane/spill-integ45-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-server/src/worker.rs`: `HostHashWorker` (one long-lived helper thread per host-tier context, channel-fed,
  joined at the tier's latch and at shutdown); `PendingDemote`'s `Hashing` phase and owner-thread ledger; the heap
  payloads (recurrent f32 planes, boundary logits, hidden) moved to the helper after the copy lands and restored into
  their emptied slots from a reply checked by ticket seq, payload count and byte count; `bind_tier_image` consuming the
  handed-in digests after its own byte-count check (same `checksum`, same bytes); `hash-helper-gone` and
  `hash-never-lands` (10 s deadline) latching with parked requests named; the ledger on the publish line;
  `host_hashing_hit` and `host_hashing_park` in the admission probe (a hit that names the `Hashing` entry parks one
  tick, the `Promoting` and `Restoring` pattern, re-parks counted); the parked-only wait guard's `Hashing` arm; the
  shutdown drain joining the helper. Eight day-28 cells plus the day-29 cell, the census and the parked-wait test.
- `tools/kv-host-contract-fault-gate.sh`: cells `hash-helper-gone` and `hash-never-lands`; the demote cells wait for the
  boot's last publication before `stop`.
- `docs/FLAGS.md`: the two fault values on the existing `MEMRA_KV_HOST_FAULT` row. No new `MEMRA_*` name, no flag.
- Research: A DAY28 and DAY29 with receipts on both cards; the door table's section A, B, B-5090 and C rows; INDEX rows;
  the lead record section (ruling 41); this file; battery receipts.

## What I checked
- Reachability: the helper exists only with a `HostTierContext` (door ON with a host tier armed); OFF never spawns it and
  never enters `Hashing`.
- Receipt term unchanged: the helper computes the same `checksum` over the same bytes; the bitwise digest cell proves
  equality with the owner-thread digests over a fixture image; `bind_tier_image` refuses a digest whose byte count is
  not the payload's.
- Fail-closed: the helper gone, a reply that never lands within the deadline, a reply for another seq or with another
  payload count or byte count, a `Hashing` entry without its shell, the tier latched meanwhile: each latches with a
  typed line, publishes nothing, wastes the pending reclaim, and names the parked requests (they re-admit to a cold
  prime). A `Block` settle at the retire seam, purge, trims, shutdown and `host.disable` continues into the hash wait
  and says what it waits on.
- One numeric program per request: bytes are hashed, nothing generated; a request parked on a `Hashing` entry is served
  by the device hit either way; the hit gate's ON census on both cards is identical to day 24.
- The acceptance: every gate green in both arms on both cards on the day-29 tree (quoted in the record), the cost clauses
  unchanged from day 28 within 0.1 ms (`in - completion` 7.4 ms against 74.8 before; tenant stall 3.3 to 3.6 ms below
  OFF; e2e +16.9), one parked hit per boot in the identity gate's default ON arm.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, cross-target `DOCS_RS=1` clippy,
  censuses, collector pytest, engine CPU lib, tier suite, engine/server/tier clippy `-D warnings`, marker census,
  workflow keys, perf board, diff-check), the local 5090 serve-smoke, the engine `d2d_*` GPU cells, the hit gate OFF and
  ON (armed) and the contract fault gate default and plain ON with the two hash cells (stated either way).

## Push regime
Engine source in the range: pushed with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged). No GPU
qualification claimed; every cell executed-not-qualified. Revuto: if capped or unavailable, this comment is the review.
