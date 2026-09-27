# Self-review: integ59 (A days 37 to 41: finding 5 fixed, hash 1 off the owner as G4, the threaded fill T, item 5's Sources fault cells; S refuted and reverted)

Author's review of the full diff `main..lane/spill-integ59-20260925`, posted as a PR comment per the owner rule.

## What the diff is
- Engine, behind the default-OFF door `MEMRA_KV_HOST_CONTRACTS`:
  - `cu/tier_receipt.cu`: the framed SHA-256 kernel of a D2H batch's device sources and the `d2h-source-flip` fault's
    one-byte flip (KERNELS.md rows added);
  - `tier_transfer.rs`: G4 (every side kernel on the copy stream, the D2H receipt ahead of the copies), the pooled
    receipt twins (G''), the threaded staging fill T inside the host function.
- Server `worker.rs`: the copy-phase park (design P), one pool context per native span cell (finding 5).
- `memra-tier`: the D2H device receipt's conformance rule and binding tests.
- `tools/kv-host-contract-fault-gate.sh`: the three Sources fault cells.
- `docs/FLAGS.md`, `docs/KERNELS.md`, `docs/TESTING.md`. No new `MEMRA_*` name.
- Research: A's DAY37 to DAY41, OWED.md, STATE.md, the 5090 and BOX7 receipts; the integ59 record (ruling 54), both
  batteries, this file.
- The branch is a fast-forward over main `9eb1326e9`.

## What I checked
**The seven spill review patterns**
- Parked requests and the idle wait: the copy-phase park joins the parked-only bounded wait (`hpx.demoting.is_some()`
  covers both phases), with its census. A copy past the helper's deadline parks nothing more.
- Releases off the happy path: twins return to the pool only on `acknowledge` of a retired batch; every other teardown
  leaks them, so nothing pinned is freed under a live write.
- No move-then-match, no gate literal, no budgeted-cache booking and no reply-identity hand-back in the new code.

**Memory and threads**
- The twin pool grows only by batch width (32 or 64 bytes per item) and concurrency, and is freed at engine drop.
- T's fill threads write disjoint ranges from planes the task's `Arc` keeps alive; scoped threads, joined before the
  host function returns; a refused spawn leaves its share to the calling thread. No CUDA call inside the host function.
- 8 new `unsafe` sites, each with its ownership stated beside it.

**Numerics and verdicts**
- One numeric program per request: the receipts digest the same bytes; the bind's re-hash now also proves landed equals
  source; the corrupted-byte cells are refused.
- The verdict lines in the record are copied from DAY37 to DAY40.
- The 5090 (f) FAIL for G4 stands as it reads. Section 20's base-controlled cell was pre-registered before it ran; it
  read base and G4 flat in a cooler hold, so the FAIL is unplaced between thermal and design, and a replicate in the hot
  regime is owed. No clause moved.
- S failed its timing clauses and was reverted in one commit with its receipts banked.

**Batteries**
- CPU battery 15 of 15 rc=0 (server 911, engine lib 551, portable 371 with 0 skipped, tier 284, pytest 87).
- RTX 5090, binary `0aaccc8b`, hashed after serve-smoke: serve-smoke, the serial engine span (10) and worker (18)
  cells, identity (12 ok), fault default and plain (229 ok each), hit OFF/ON (61 and 68), admit-mem-burst and
  spec-ctx-edge, all green.
- Target card: A's BOX7 sittings, G4 (a) to (f) PASS, every gate green, 2547 receipts checked.

**Hygiene:** no provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced,
logged). Every other hook ran. No tag: the change is behind a default-OFF door. Revuto: if capped or unavailable,
this comment is the review.
