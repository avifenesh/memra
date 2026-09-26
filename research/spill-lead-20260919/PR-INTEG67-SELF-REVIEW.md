# Self-review: integ67 (A's design B1, the prefix snapshot's and restore's copies as one launch, adopted on both cards)

Author's review of the full diff `main..lane/spill-integ67-20260926`, posted as a PR comment per the owner rule.

## What the diff is
- `cu/kernels.cu`: `copy_batch_items_u8`, the batched item copy and i32 sets in one launch.
- `lib.rs`: its wrapper (zero-byte items dropped, gridDim.y refusal before any device work) and GPU cell (a1).
- `worker.rs`: the snapshot's planes allocated without a memset at the copy's exact size and filled in one launch; the
  restore's copies and length sets in one launch with every range checked first; cell (a2) and the census.
- `docs/KERNELS.md`, `docs/FLAGS.md` (a cell's test input). No door: B1 is the naked program.
- Research: A's DAY59 sections 7 to 10 with the target and 5090 receipts; the integ67 record (ruling 62), both
  batteries, this file. W's code and R2 are not in this merge.

## What I checked
- The kernel's aligned vector path and byte tail cover each item exactly; items are disjoint, so block order is free.
- No uninitialized byte is published: each plane is allocated at the byte count the same launch writes, and a failed
  launch fails the snapshot.
- The cudarc guards are held across the launch, so the cross-stream event discipline matches the per-plane copies.
- The gridDim.y limit: I found it in review; A's guard refuses before any upload, with a CPU test and a red arm.
- Both cards read ADOPT; the verdict lines in the record are copied from A's readers.

## Batteries
- CPU battery 15 of 15 before the guard; on the head, the tests on BOX31 (server 943, engine lib 577) and the three
  environment-refused census steps rerun on the lead's rig, all rc=0.
- GPU battery on a rented RTX PRO 6000 with the 9B: every cell green, the pause gate with the 27B `ALL GREEN`.

**Hygiene:** no provider name, host, id or price in tracked files. No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged).
No tag here: the integration PRs carry no release; the owner decides one. Revuto: if capped or unavailable, this comment
is the review.
