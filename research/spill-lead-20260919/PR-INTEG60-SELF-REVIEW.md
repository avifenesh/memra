# Self-review: integ60 (C days 40 to 58: the MoE slot cache door tuned and decided on the target card, verify digest v3, the DFlash tail class, two partial-restore fixes)

Author's review of the full diff `main..lane/spill-integ60-20260925`, posted as a PR comment per the owner rule.

## What the diff is
- Engine and tier: the `--experts-via-tier` MoE slot cache door's ten rungs (`banked_residency/native.rs`,
  `moe_cache.rs`, `memra-tier/src/bank/*`, `governor.rs`), among them the pinned host pool, leases retired on copy
  events, the fill completing before decode, and the prediction prefetch rung. A CLI door, default absent.
- Server `worker.rs`: verify digest v3 (`MEMRA_KV_HOST_VERIFY`, a diagnostic), the DFlash tail as a contracts-door
  entry class, and two fixes in the default-OFF `MEMRA_DSPARK_PARTIAL_RESTORE` path.
- Gates: `tools/kv-host-spill-identity-gate.sh` (drafter arm) and `tools/kv-host-spill-failure-gate.sh` (three digest
  cells). Tests: the bank crate's day-43, 45 and 47 suites and the SLRU oracle.
- Docs: FLAGS.md (the fault row's three new values), TESTING.md, tools/README.md.
- Research: C's DAY40 to DAY58, the target-card receipts (`pro-single-day52/`, 2589 files), OWED and STATE; the integ60
  record (ruling 55), both batteries, this file.
- The merge onto main `5d653e851` had two documentation conflicts, resolved as a union of both sides. A lead commit
  marks C's verbatim logs `-whitespace` so `git diff --check` passes; no receipt byte changed.

## What I checked
**Memory and ownership**
- `PinnedPool`: one cached portable pinned allocation (never write-combined, since the CPU reads it), buffers taken and
  returned under one mutex, zeroed on first take, freed only at pool drop after every buffer's `Arc` is gone. The one
  `cuMemFreeHost` is at teardown, never per lease.
- A lease is finished only after its copy event completes; an enqueue on an unknown stream keeps the lease open (fail
  closed). No buffer returns to the pool under a live DMA.
- 10 new `unsafe` sites, each with its ownership stated beside it.

**Numerics**
- The off-grid strict-prefix restore now serves cold: a second numeric program is forbidden, not tolerated. The
  short-suffix hit serves cold instead of failing with HTTP 500.
- Verify digest v3 covers every round-tripped plane; v2's text is pinned by SHA-256 in its census.

**Verdicts**
- The verdict lines in the record are copied from C's DAY41 to DAY58 and its report.
- DAY51's first run was void on an integrity term that contradicted I10's registered clause; C corrected that term
  only, reran in a new hold with the timing rule unchanged, and DAY51 says C saw the void run's timings. Accepted as an
  integrity definition, stated in the record.
- `MEMRA_MOE_PREFETCH=1` is faster than the door and the legacy on the target card and has no decide-by date; it is now
  C's owed deciding cell.

**Batteries**
- CPU battery 15 of 15 rc=0 (server 917, engine lib 563, portable 383 with 0 skipped, tier 296, pytest 87).
- The local 5090 needs a reset, so the GPU battery ran on a rented RTX PRO 6000 with the same 9B model under the pair
  lock: every cell green (identity 12 ok, fault default and plain 229 ok each, hit OFF/ON 61 and 68, admit-mem-burst,
  spec-ctx-edge, engine span 10 and worker 18). serve-smoke first skipped on the model path; rerun alone with the model
  linked, `serve-smoke: 0 failed`, binary unchanged.
- Target card: C's BOX8 sitting, `door_wins`, G1 and G2 PASS, the C5 and C6 cells green.

**Hygiene:** no provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced,
logged). Every other hook ran. No tag: every change is behind a default-OFF door or a CLI door. Revuto: if capped or
unavailable, this comment is the review.
