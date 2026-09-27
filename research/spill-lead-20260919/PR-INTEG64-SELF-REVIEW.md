# Self-review: integ64 (A days 52 to 55: P2 refuted and reverted, item 21 as F1, item 10 priced, item 22 closed)

Author's review of the full diff `main..lane/spill-integ64-20260926`, posted as a PR comment per the owner rule.

## What the diff is
- `health.rs`: T-a, a snapshot and the stall verdict read the clock once; a `#[cfg(test)]` thread-local virtual clock
  for T-c. The one production behavior change in the PR.
- `worker.rs`: the demote publication split and the on-tick publish lines, log only (the on-tick lines print only when
  the contracts door built the host tier). P2 landed and was reverted on its (g) failure.
- `dsv4_serve.rs`: the coalescer's window as a field (`ROW_BATCH_WAIT` in production) and two mechanism cells.
- `lib.rs` tests: F1 (the admission writers take `drain_lock()` first), T-b on tokio's paused clock.
- `Cargo.toml`: tokio `test-util`, dev-dependency only. `docs/TESTING.md`. No new `MEMRA_*` name.
- Research: A's DAY52 to DAY55 with their receipts; the integ64 record (ruling 59), both batteries, this file.

## What I checked
- T-a keeps the verdict's direction and the census pins the single sample in each of the three readers.
- The test clock is unreachable from a non-test build and from any thread but the one that set it.
- The drop helper destructures and drops in declaration order (the compiler's order); a census pins the field list.
- The on-tick refusal conditions are the old ones in the old order; only a reason string is recorded.
- The coalescer's production window is unchanged; T-e's claim moved to the mechanism as pre-registered.
- The verdict lines in the record are copied from A's DAY52 to DAY55.

## Batteries
- CPU battery 15 of 15 after the diff check's rerun (A's verbatim patch receipt marked `-whitespace`): server 937,
  engine lib 573, portable 388 with 0 skipped, tier 301, pytest 87.
- GPU battery on a rented RTX PRO 6000 with the same 9B: every cell green, fault gate 255 ok per arm, the pause gate
  with the 27B `ALL GREEN` (40 ok).

**Hygiene:** no provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Server source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged).
Every other hook ran. No tag: a health fix inside the spill program's integration with no default moved; the release
decision stays with the owner. Revuto: if capped or unavailable, this comment is the review.
