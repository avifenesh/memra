# The MoE slot cache door as the default expert-spill program (2026-09-27)

**Decided by the owner, 2026-09-27** ("accept all clear ones. A. promote."), for three questions lane C put to
the review: promote the MoE slot cache door (C1(c)), make the registered host pool the door's pool (`DAY80.md`
section 4a), and make the slot cache's in-token prefetch the default (C10). This record says what was chosen,
what was rejected, what settled it, and the scope the no-generic-support rule gives it. The registration is
`research/spill-c-20260919/DAY88.md` (phase 1 of the door's promotion work, OWED C2).

Status: decided by the owner; the change lands on `main` only after the promotion's own cells pass on both card
classes (`DAY88.md` section 5), and this line then names their verdicts.

## What was chosen

- **The door.** When a MoE model's experts go to the SLRU slot cache (they do not fit the card, or
  `MEMRA_MOE_RESIDENT=0`), `run-gen` and `run-spec` serve them through the MoE slot cache door: an owner-thread
  host tier that holds the whole expert bank in pinned memory, leases each record through its owner, and stages
  the H2D copies from it, with the same kernels reading the same bytes as the legacy slot cache. The rollback is
  `MEMRA_EXPERTS_VIA_TIER=0` (the legacy slot cache from pinned copies), decide-by 2026-10-11.
- **The pool.** The door's host pool is private anonymous memory registered with `cuMemHostRegister`, which the
  kernel's compaction skips instead of isolating. The rollback is `--expert-bank-pool-allocated` (the
  `cuMemHostAlloc` pool), decide-by 2026-10-11.
- **The prefetch.** The slot cache's in-token prefetch of the next routed expert is on by default, with one
  meaning on both programs (the door's owner-routed grouped prefetch, the legacy's copy-stream prefetch).
  `MEMRA_MOE_PREFETCH=0` turns it off on both, decide-by 2026-10-11.

Each choice is made once per process before the first token, and none changes a kernel or the bytes a kernel
reads, so a request runs one numeric program; every cell below read one token tape across its arms.

## The scope

Every receipt below is on one artifact, Qwen3.6-35B-A3B-UD-IQ4_XS (SHA-256 `df27a780...7adf`). The rule that no
model's support is inferred from another's, whatever they share, gives the default that scope: the door, the
registered pool and the prefetch are the default **on both cards for an artifact the door is qualified on**, a
list the installer keeps by digest (one entry today). Every other artifact runs exactly its previous program.
A new artifact joins only with its own census, gates and receipts.

Outside this decision until their own work lands (OWED C2): `memra-server` (no installer yet, so no
serving-shape bit-identity gate), a PP stage split (the owner registry is per thread), scale-bearing artifacts
and mixed layouts (refused by the installer). Each keeps its previous program and prints why.

## What settled it

| Question | Receipt | Verdict |
|---|---|---|
| The door against the legacy, the deciding cell | `research/spill-c-20260919/DAY51.md` | `DAY51 VERDICT rig=pro-single integrity=ok -> door_wins`; `DAY51 VERDICT rig=rtx5090 integrity=ok -> door_flat` |
| The tuned door against REF (the legacy with its prefetch) | `research/spill-c-20260919/DAY85.md` section 5 | `i22=improves door=i22 vs_ref=matches` on the 9950X class, `i22=flat door=i22 vs_ref=matches` on the 285K class (1 to 2 ms behind REF over 32 tokens by the medians, from 9 to 10 at I15) |
| The registered pool | `research/spill-c-20260919/DAY80.md` | `DAY80 REGPOOL VERDICT rig=box37-285k integrity=ok -> registered_clears`; `DAY80 REGTIME VERDICT ... dr=flat` on the 285K and the 9950X |
| The prefetch | `research/spill-c-20260919/DAY59.md` | G1, G2 and G3 PASS on both cards; `shape=pftime ... -> pf_wins` and `shape=pfnaked ... -> pf_flat` on the target card and on the RTX 5090 |

## What was rejected

- **Deleting the door at its decide-by** (the door document's other outcome): the deciding cell read
  `door_wins` on the target card and the tuning since closed the gap to REF.
- **The `cuMemHostAlloc` pool as the default**: on long-running 9950X and 285K hosts the kernel's compaction
  isolated its pages and failed to migrate them, the door's slow state (`DAY76.md`, `DAY78.md`); the registered
  pool cleared it. Chunked allocations (`DAY76.md`, `chunk_does_not`) and a heap pool (`DAY78.md`) were measured
  and are deleted.
- **A default for every MoE family at once**: refused by the no-generic-support rule; the receipts cover one
  artifact.
- **The door's prefetch without a seam**: before this decision the door always prefetched and only the legacy
  read `MEMRA_MOE_PREFETCH`; one flag now governs both, so the rollback of the prefetch is one switch.

## Open, and where it is tracked

Whether the recent tuning (I21, I22) makes the compaction state's rate higher on a long-running 9950X
(`DAY86.md`: `not_reproduced` on the one host tried, so undecided). The promotion's own cells, on both card
classes (`DAY88.md` section 5): the default against the door as qualified, against its rollback, and in the
resident shape.
