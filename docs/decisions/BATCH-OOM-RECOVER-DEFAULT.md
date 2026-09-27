# Batched decode OOM recovery: ON by default on the RTX PRO 6000 Blackwell class (2026-09-27)

**Status:** decided ON on the RTX PRO 6000 Blackwell class (the owner's ruling, 2026-09-27;
WP-B DAY49 addendum F). `MEMRA_BATCH_OOM_RECOVER` unset runs the recovery on every card whose
name reads as that class; `MEMRA_BATCH_OOM_RECOVER=0` is the rollback seam, `decide-by:
2026-10-11` for deleting the seam. The RTX 5090 keeps the door default off until its own serving
boots read (`decide-by: 2026-10-10`); `=1` runs it there. Receipts:
`research/spill-b-20260919/DAY49.md` sections 2.1 to 2.6, `pro-single-day49d/`,
`rtx5090-day49/`, `rtx5090-day49d/`.

## Question

A quoted CUDA OOM on a batched decode chunk ended every session of the chunk with the typed
overloaded error, peers included (DAY47 2.1, OWED O14): one allocation failure lost up to the
whole wave. The recovery (DAY49 1.2 and addenda A to D) retries such a chunk once, after one
reclaim rung, when the failure came before any session-state write. The retried step is the same
program on unchanged state, so each request still runs one numeric program. The question was
whether it becomes the naked default, and on which cards, under the per-hardware rule.

## Measured

On the target card, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, the 27B:

| cell | door | result |
|---|---|---|
| gate arm j (aimed fault, a 3-session batched chunk), four runs | ON | retried once, `retried (ok)`, all three streams `200` with the no-fault control's digests |
| arm j-red (the retry faulted too), four runs | ON | `retry failed`, all three sessions end with the error event, no 5xx, no panic |
| j-vmm (`MEMRA_KV_ALLOCATOR=vmm`) | ON | one `reap (batch-oom)` line per retry, before it; the red patch prints none |
| serving shape, burst 8 x 6,144, spec-default route, both orders | OFF | 8 of 9 requests end `503` |
| the same | ON | 9 of 9 `200`, one retry each |
| serving shape, plain route, both orders | OFF | 5 of 9 end `503` |
| the same | ON | 9 of 9 `200`, one retry each |

On the local RTX 5090 (the 9B): arm j and j-vmm PASS on both runs (DAY49 2.6). The serving shape
read three of eight boots (the before, 8 of 9 `503`; two ON boots, 9 of 9 `200`); the five unrun
boots run on queue-n.

## Decided

- ON is the default on the RTX PRO 6000 Blackwell class, every variant. The receipts are the
  Workstation Edition's; the class-wide default is the owner's ruling. The recovery is a
  correctness path (a bounded retry of an untouched step), not a power-shaped tuning arm.
- The RTX 5090 stays default off until its serving boots read, then gets its own flip. One-rig
  evidence sets a one-rig default at most.
- `=0` is the rollback seam for two weeks. If nobody uses it by 2026-10-11, the seam is deleted and
  the recovery is the naked default on the class.
- Keyed on the device name through `memra_engine::parallel::hardware_target_of`, the class
  definition the topology code uses, so the default cannot drift from it. The worker prints
  `[batch-oom] recover=<ON|OFF> source=...` once at boot.

## Rejected

- A global flip. It would set the 5090's default on gate evidence alone, before its serving shape
  read.
- Leaving the door off everywhere until both cards read. The target card is read in full, and the
  off arm loses every session of a faulted chunk there.
