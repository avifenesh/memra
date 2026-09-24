# Self-review: integ57 (A day 35: F settled KEEP, design M refuted and reverted, design M' the bind re-hash on the helper)

Author's review of the full diff `main..lane/spill-integ57-20260924`, posted as a PR comment per the owner rule.

## What the diff is
- Engine and server, behind the default-OFF door `MEMRA_KV_HOST_CONTRACTS`:
  - `tier_transfer.rs`: `read_view` (an `unsafe fn` with a stated ownership contract) and `PinnedLeaseView`, a
    read-only `Send` view whose `digest` runs the bind's `checksum`;
  - `worker.rs`: `host_lease_views`, the `HostLeasesOnHelper` leak-on-drop guard, and the bind re-hash moved into
    the existing `Hashing` job.
- `docs/FLAGS.md` and `docs/TESTING.md` rows. No new `MEMRA_*` name.
- Research: A's DAY35.md with the receipts `rtx5090-day35/ab/`, `m/` (M's red receipts, committed as-is) and `m2/`;
  STATE, OWNER-THREAD-OFFLOAD and the INDEX row; the integ57 record section (ruling 52), both batteries, and this
  file.
- The branch is a fast-forward over main `e20a4c5bb`.

## What I checked
**The view and its guard**
- `PinnedLeaseView` repeats K's reviewed pattern: the owner keeps the lease alive and unwritten until the view
  returns.
- `HostLeasesOnHelper` holds each lease on the owner thread until `land` takes the helper's reply. Every other drop
  leaks the lease rather than freeing it, so nothing is freed under the helper's read.
- A census test pins the single `hashing.leases.take()` after the reply and before the image is published.

**Why M' keeps the copy phase**
- M' moves only work that already sits inside a `Hashing` job whose hits park, so the copy phase keeps its one poll.
- M lost that property, which is what refuted it. Its red receipts are banked, and its code came out of the tip in
  one revert.

**Numerics and verdicts**
- One numeric program holds: the digest is the same `checksum` over the same bytes, on the helper. The
  corrupted-byte cell is refused at the bind and again at the promote.
- The verdict lines in the record are copied from DAY35 sections 4, 6 and 8.
- F's decision A/B was pre-registered with its margin. M' was pre-registered after M's refutation and before its own
  code.

**Process notes, recorded**
- A's first local scratch commit went in with hooks off. It was undone and recommitted with the hook before anything
  was built from it, and nothing was pushed from it.
- M''s first hold read NOT RUN on a foreign idle process. The unchanged script ran once the process left, so the
  drafted amendment was never used.

**Batteries**
- CPU battery 15 of 15 rc=0.
- RTX 5090, binary `33d63f70`, hashed after serve-smoke: serve-smoke, the serial engine span (10) and worker (18)
  cells, identity, fault default and plain (160 ok each), hit OFF/ON, admit-mem-burst and spec-ctx-edge, all green.
- No target-card sitting was pre-registered for M'. It joins the next rented card's sitting.

**The review fix `43d16d73f` (revuto's finding)**
- The guard landed before the reply's `seq` check, so a foreign reply could free leases the helper might still read.
- It now lands only when `reply.seq == seq`, and otherwise leaks on the latch path. The census pins the order.
- Every matching reply takes the same path as before. The 5090 battery on the fixed head (binary `bec0b102`) is
  all green.

**Hygiene:** no provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced,
logged). Every other hook ran. No tag: the change is behind a default-OFF door. Revuto: if capped or unavailable,
this comment is the review.
