# Self-review: integ62 (A days 42 to 48: S4 the span receipts, V the pause sweep's demotes off the tick, items 3 and 15 closed)

Author's review of the full diff `main..lane/spill-integ62-20260925`, posted as a PR comment per the owner rule.

## What the diff is
- Engine `tier_transfer.rs`: S4 (batched source digests ahead of the copies, landed digests at the seal, required before
  publish) and `take_plane` / `release_device` draining the owner stream only. S2 and S3 are refuted and reverted.
- Server `worker.rs`: V (the pause sweep's demotes off the tick, park release at publish, failed entry reinstated).
- `memra-tier`: the span receipt's conformance rule and binding tests. Gates: the fault gate's new cells and the new
  pause-demote gate. Docs: FLAGS, KERNELS, TESTING. No new `MEMRA_*` name.
- Research: A's DAY42 to DAY48 with the BOX10 receipts; the integ62 record (ruling 57), both batteries, this file.
- The branch is a fast-forward over main `d6515742f`.

## What I checked
- The relaxed drain: a lease cannot be taken back while an unretired ticket names it, and a ticket retires only after its
  landing observed every item and receipt event, so the copy stream cannot touch a lease at take-back; an unknown ticket
  stays unretired and its lease leaks (fail closed). A census pins the release paths.
- V's release paths: park released at publish, failed shape-2 entry reinstated, unpublished demote keeps the park; the
  pause gate's failure and race cells exercise them, each turn byte-compared with a door-OFF, pause-OFF reference.
- The verdict lines in the record are copied from A's DAY42 to DAY48. S2 and S3 failed their price clause and were
  reverted with receipts banked; the pause gate's first-run failures are recorded before its pre-registered fix.
- The deferred admission flush is ruled into lane B's door, not built here.

## Batteries
- CPU battery 15 of 15 rc=0 (server 920, engine lib 570, portable 388 with 0 skipped, tier 301, pytest 87).
- GPU battery on BOX10 (an RTX PRO 6000; the 5090 still needs a reset) with the same 9B model: every cell green, the
  fault gate 255 ok per arm; the new pause gate with the 27B `ALL GREEN` on its rerun (the first attempt refused on a
  lock name the lead's driver did not set).

**Hygiene:** no provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced,
logged). Every other hook ran. No tag: default-OFF doors only. Revuto: if capped or unavailable, this comment is the
review.
