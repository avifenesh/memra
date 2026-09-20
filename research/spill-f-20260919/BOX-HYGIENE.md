# BOX2 disk, build and receipt hygiene

Date: 2026-09-19. Read-only follow-up snapshot at **21:56:03 UTC**:
[`raw/box2-hygiene.json`](raw/box2-hygiene.json). These are point-in-time
resource facts, not storage throughput or a GPU reservation. No cleanup,
mount, lifecycle, power-setting or other-lane operation ran.

## Current disk and GPU state

| Resource | Observation | Interpretation |
|---|---|---|
| Container root | 214,748,364,800 bytes total; 25,365,811,200 used; 189,382,553,600 available (12% used) | 200 GiB quota; about 23.624 GiB used / 176.376 GiB available; overlay, local NVMe ancestry unproven |
| Shared-memory mount | 33,285,996,544 bytes total/available, zero used | tmpfs/RAM, not persistent storage or proof of NVMe |
| Lane F target | `/root/wt-f/target` absent | No lane target allocation measured; does not establish whether the checkout itself exists |
| Cargo registry | 108,900,352 allocated bytes (about 103.9 MiB) | Shared dependency cache inventory only; do not delete while builds may use it |
| Cargo git cache | absent | `du` exits 1 for this missing path; registry measurement still succeeded |
| GPU | RTX 5090; 32,607 MiB total / 2 MiB used; no compute applications returned | Idle at the snapshot only, not clearance for a later GPU launch |
| Enforced / maximum power limit | **400 W / 600 W** | A 400 W capped measurement condition; never describe it as full power |

The aggregate root `df` covers all tenant files without walking their trees.
This follow-up did **not** walk artifact directories, the base checkout, other
lanes' worktrees or their receipt directories. Do not derive per-lane ownership
from the root used total. The earlier inventory remains historical evidence;
its broader directory census is not the command to rerun under this handoff's
narrow ownership rules.

Safe periodic metadata checks (no cleanup implied):

```sh
df -B1 / /dev/shm
# Only the lane's own build directory, if present:
test ! -d /root/wt-f/target || du -sx -B1 /root/wt-f/target
nvidia-smi --query-gpu=name,memory.total,memory.used,power.limit,power.max_limit --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

## CARGO_TARGET_DIR recommendation

**For the current isolated lanes, retain `/root/wt-f/target` as F's target.**
There is ample observed space and F has no target yet. Do not point F at
another lane's target to save an unmeasured amount of disk. Recheck root free
space before a build and reserve at least 20 GiB free; stop rather than clean
someone else's files if the reserve is threatened. Build with at most `-j 16`,
outside the GPU timed window, and record exact toolchain/architecture choices.

If the lead later explicitly authorizes build-cache sharing, use **one new,
lead-owned `CARGO_TARGET_DIR` per compatible toolchain/target/profile/CUDA-arch
configuration**, not the base checkout or a lane's target. Cargo's target lock
serializes writers and is expected contention, not a hang to override. Share
only under a documented owner and do not run `cargo clean` there independently.
No shared target was created by this lane.

Caveats before enabling sharing:

- Caches are build accelerators, not immutable binaries or proof that a feature,
  CUDA architecture, compiler option or linked library matched a previous run.
  Bind every measured executable to its exact source and SHA-256. Freeze a
  lane-owned executable copy before the collector starts so another build
  cannot overwrite a common `release/<bin>` pathname between cells.
- Keep CUDA architecture, toolkit, rustc, target triple, profiles, features,
  build-script inputs and link settings explicit. Different worktree paths
  can change fingerprints and may limit reuse; sharing saves no guaranteed
  amount. An executable hash mismatch or stale artifact is refusal, not reuse.
- One target directory can create head-of-line blocking across lanes. A build
  for a different numeric program should not delay a reserved correctness
  window; schedule compiles and use lane targets when isolation is needed.
- Cargo registry/git source caches already provide some sharing without sharing
  executable paths. Do not delete caches while another build holds or reads
  them. Disk pressure is a lead coordination event, not authority to evict.
- Shared targets contain local source paths and build metadata. Sync receipts
  and the explicitly selected binary, not the entire build tree or machine
  environment. Never copy environment files or credentials.

## Sync cadence and immutability

1. Pre-register the cell and push its source before remote execution. Record
   commit, binary hash, collector hash and resolved arguments before acquiring
   the canonical collector lock. Local commits alone do not bank remote work.
2. Keep F's raw logs in a unique attempt directory under `/root/wt-f/receipts/`.
   Never overwrite an attempt. After **every collector cell**, success or
   failure, sync the completed attempt back before starting another. For a
   longer non-GPU build, checkpoint its log at least every five minutes and
   sync it again at completion; mark partial snapshots as partial.
3. Preserve raw stdout/stderr, exit status, timeout status, before/failure/after
   GPU state, 250 ms telemetry, power-limit snapshots and input/output hashes.
   Close/hash files after collection, then verify byte lengths and SHA-256 at
   the local destination. A disappearing connection is not proof of completion.
4. Export only sanitized, engine-relevant receipts into
   `research/spill-f-20260919/`. Deployment metadata, identities, locations and
   prices stay outside tracked files. The collector may embed instance metadata
   in `CELL.jsonl`; inspect privately and create a separately identified export,
   rather than silently altering a hash-bound raw bundle. Do not publish secret
   values or environment dumps under any circumstances.
5. Commit and push each completed file/milestone with the repository hooks
   enabled. Verify origin's branch SHA; a GitHub timeout is not a successful
   push. Retry at most three times with backoff. Remote results are not banked
   until the local hashed receipt and its commit are on origin.

## Pre-destroy checklist (handoff only; lifecycle operations forbidden here)

The infrastructure owner, not this lane, decides whether and when to destroy.
Before handing back F's completion status:

- [ ] Confirm all F commands have exited; retain final exit/failure state.
- [ ] Close, hash, sync and locally rehash every F receipt, including negative
      and interrupted attempts. If a spot loss prevents recovery, mark missing
      evidence explicitly; never infer a pass from an incomplete log.
- [ ] Confirm the intended F source and receipts exist on origin at the reported
      SHA and that the local worktree contains no unrelated changes.
- [ ] Give the lead a list of F-owned disposable target/temp paths. Remove only
      explicitly owned scratch after receipts are durable and the lead's
      integration/abandon decision is recorded; do not remove an active worktree.
- [ ] Obtain each other lane owner's independent sync acknowledgment. F's root
      `df`, idle GPU, or clean worktree cannot establish their data safety.
- [ ] Leave artifacts, the base checkout, other worktrees and other receipt
      namespaces untouched. Do not remove the canonical shared lock inode.
- [ ] Infrastructure owner verifies the intended resource identity privately,
      performs the separately authorized lifecycle action, and reads back final
      state. This lane executes **no** create/start/stop/destroy/volume action.

Unchecked boxes are a checklist, not a readiness assertion. F's worktree stays
open for lead integration; ordinary finished-lane worktree/branch cleanup
follows the recorded merge or abandon decision, not receipt production alone.

## Power-cap rule

Retain **enforced power limit and advertised maximum separately** for every
measurement arm, with UTC/monotonic time binding. Read both at arm boundaries;
use 250 ms enforced-limit samples if detecting mid-arm changes. Existing
collector `power.draw` samples are not cap samples. Unavailable limit data,
changed cap, missing telemetry, unexpected co-tenancy, or a different PCIe/link
condition makes a same-window comparison unqualified until rerun. Preserve the
invalidated attempt and reason. Never raise the cap, pin clocks, or infer
600 W operation from this card's maximum field. The current 400 W condition
must accompany any future BOX2 performance result, even if measured power draw
stays below it.
