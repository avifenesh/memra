# Self-review: integ66 (F's M1 program closed; C days 63 to 74: the door's I13 to I15, the probes, the gap placed)

Author's review of the full diff `main..lane/spill-integ66-20260926`, posted as a PR comment per the owner rule.

## What the diff is
- `memra-tier` bank: C's I13 to I15 for the MoE slot cache door (allocation-free governor accounting, a shared lease
  body, one-call ticket finish, Fx-hashed maps and indexes with their orders pinned, one ticket per prefetched expert).
- `memra-engine`: the door's grouped prefetch in `moe_cache.rs` and `banked_residency/native.rs`; `hybrid_forward.rs`
  calls `prefetch_expert`, whose legacy arm is the old per-block loop; `cpu_probe.rs` and three log-only `run-gen` flags.
- Research: F's M1 receipts from BOX27, C's days 63 to 74, the integ66 record (ruling 61), both batteries, this file.

## What I checked
- The naked path: `prefetch_expert` without the door runs `prefetch_source` per block, gate then up then down, as before.
- The grouped lease: the in-flight bound counts leases, a group finishes once after its last member is consumed and its
  copy lands, teardown drains groups and in-flight leases and keeps a refused finish open (fail closed).
- The Fx indexes: every answer and ordered iteration is pinned against the ordered map by randomized references.
- The probes are flags on `run-gen`, run outside the timed spans.
- The verdict lines in the record are copied from F's and C's receipts; F's post-hoc rescorings are labelled.

## Batteries
- CPU battery 15 of 15: server 943, engine lib 576, tier 309, portable 388 with 0 skipped, pytest 87.
- GPU battery on a rented RTX PRO 6000 with the 9B: every cell green, the pause gate with the 27B `ALL GREEN`.

**Hygiene:** no provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged).
No tag: the door stays default-OFF. Revuto: if capped or unavailable, this comment is the review.
