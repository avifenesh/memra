# Self-review: integ69 (A's L' and R1 with the staging-fill gate change; C's I18 and two diagnostic pool flags)

Author's review of the full diff `main..lane/spill-integ69-20260926`, posted as a PR comment per the owner rule.

## What the diff is
- `tier_transfer.rs`, `worker.rs`: L', a size-class pool of pinned lease backings under its own governor tenant (cap one
  host budget, closed at the tier latch and engine drop) and the span staging set allocated at boot; each lease charges
  its class. R1: a retire settles a pending capture only when a retiring session is its source.
- `tools/kv-host-contract-fault-gate.sh`: the staging-fill checks count fill events (boot or fresh), still exactly one.
- The worker's GPU span cells updated to L's registered charges (two test-only commits from lane A).
- C's I18 and the `--expert-bank-pool-pageable` and `--expert-bank-pool-registered` diagnostic flags (off unless passed).
- Research: A's and C's days, the integ69 record (ruling 64), three GPU battery runs, four CPU batteries, this file.

## What I checked
- The pool takes only an exact class and kind, charges idle backings to its own tenant and refuses past its cap; a
  reused backing is not re-zeroed, and a census pins that every copy spans the lease length and nothing reads past it.
- R1's source identity (request id or the plain cache's address) matches spec-boundary captures; its red arm fails.
- The gate change keeps the property its checks were written for; its red arm (refusals that drop buffers) is caught.
- The 13 worker-cell failures were each placed before an edit as the cells' own arithmetic under L1.2 and L1.5
  (272 -> 512, 192 -> 256, three budgets); no production code changed for them.
- `moe_cache.rs` is untouched, so the day-4 fixture pin holds.

## Batteries
- CPU battery 15 of 15 on the final head (server 955, engine lib 584, tier 315, portable 388, pytest 87).
- GPU battery run 3 on an RTX PRO 6000: every gate and both native cell groups green (worker span cells 18 of 18); the
  only red is serve-smoke's Q35 arm, main's own stale expectation (#777).

**Hygiene:** no provider name, host, id, price or city in tracked files. No em dash in authored lines.

## Push regime
Engine and server source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development`. No tag.
Revuto: if capped or unavailable, this comment is the review.

## Round 2, after revuto round 1
Revuto found a real defect that this review missed. I noted that a reused backing is not re-zeroed but did not check it
against `purge_tenant`'s promise. Since L', a purge's dropped host leases parked in the pool with the revoked tenant's KV
bytes in them. Lane A's fix `4f297e7bd` (DAY69 design P):
- `LeasePool::drain()` frees every idle backing, releases its charge and advances an epoch. A backing parks only if its
  lease was allocated in the current epoch, so a purged tenant's lease dropped after the purge frees instead of parking.
- `purge_tenant` ends with the drain and zeroes the span staging set's idle buffers in place. That second retention
  predates L' (day 30) and A found it while placing this one.
- The steady path gains one integer compare per drop. A purge costs the pool's warmth: the next demotes run the pre-L'
  program until post-purge leases refill it.

What I checked on the fix:
- The epoch covers the case a plain drain misses: a lease allocated before the purge and dropped after it.
- The red arms fail where they should: the CPU test with `put` ignoring the epoch (`left: 2, right: 3`), the worker
  census with the scrub call removed.
- Other purge paths: the device purge, the restore purge and the promoted-pin release hold no host lease.
- Cross-tenant reuse without a purge is not readable: no read reaches past `len`, and a D2H that does not cover the whole
  lease, or whose lease is shared, is refused.
- Two latent hazards are A's owed items, not defects here: a caller could read a fresh pooled lease before its copy
  lands (no production caller does), and the GLM-5 TP startup arena returns released regions unscrubbed.

Batteries on the new head:
- CPU battery 15 of 15 on `061833794` (server lib 950, engine lib 586).
- GPU run 4 on the same box class, with run 3's cells plus `day69_` and every ignored `tier_transfer::tests::` cell:
  engine cells 18 of 18, worker span cells 19 of 19, every gate ALL GREEN. serve-smoke's Q35 arm is #777 again.
- Main `df006602e` (#800, DSv4 only) merged in clean.

## Round 3, after revuto round 2
Revuto found a second real L' defect, and my round-2 check missed it too: I checked the purge fix's epoch and red arms,
not L1.2's class charge against the LRU's unit. A lease charged its class while the host LRU budgets actual bytes, so
short-prefix residents could hold nearly two budgets of charge. A demote was then refused where main admits it, and the
refusal latched the tier off. Lane A placed it with a CPU test (main admits the same sequence at 81 of 128 MiB). The fix
is `a57f85897` (DAY70 design Q).
- A lease is charged its length, main's charge. The pool carries its idle backings and the live leases' tails, capped
  together at one budget.
- A lease that cannot take a class backing inside the cap gets its exact length, as main does. So a lease is refused
  only where main's ledger refuses it.
- Demote counts and the request's program are unchanged against main.
- The red arm (charge back to the class) fails all three new cells, including the placement refusal.

What I checked:
- The design keeps the ledger and the LRU on one unit instead of patching the refusal. Revuto's two options would still
  refuse the placed shape.
- No earlier reading moves: equal demote counts between arms in every A/B cell, no `TIER DISABLED` in any A/B log.
- The worker cells' class arithmetic returns to lengths (pool cap 0 in those fixtures).

Batteries on the new head:
- CPU battery on `c7a0dcf42`: 14 of 15 (server lib 951, engine lib 589). The 15th, diff-check, flagged run 4's raw logs'
  trailing blank lines. The receipts' `.gitattributes` now exempts them, and diff-check reads rc 0.
- GPU run 5: engine cells 19 of 19 (with `day70_` and both `day63_` pool cells), worker cells 19 of 19, every gate ALL
  GREEN, `tier-transfer-gate` conformance and roundtrip PASS, all seven `kv-tier-gate` fault arms PASS. serve-smoke's
  Q35 arm is #777 again.
- Main `dba926cdc` (#807) merged in clean.
