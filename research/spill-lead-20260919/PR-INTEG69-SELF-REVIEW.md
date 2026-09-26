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
