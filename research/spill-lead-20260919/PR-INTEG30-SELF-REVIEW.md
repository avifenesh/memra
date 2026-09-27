# Self-review: integ30 (A day 17: memra#536 Move 1 first slice, the D2H demote off the tick under the host-contracts door)

Author's review of the full diff `main..lane/spill-integ30-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-engine/src/tier_transfer.rs`: `CudaTransfers` gains an optional second stream (`copy`), created by
  `new_with_copy_stream` on the same owner thread; in `submit_batch` a D2H item is issued on it behind the producer
  fence's event and records its completion event there, and the owner-stream wait is installed only when the issue
  stream is the owner stream (an H2D or a transfer without the copy stream). `release_device` and `take_plane`
  synchronize the copy stream before they hand a device plane back. Read: the H2D path and the no-copy-stream path are
  the previous code.
- `crates/memra-server/src/worker.rs`, `worker/host_glm.rs`: the door's demote route is split into submit and settle,
  with the previous blocking behaviour kept as a wrapper; a `Demoting` entry is published only at the tick-top poll
  after `require`, planes back, retire, acknowledge and `bind_tier_image`; every path that meets a `Demoting` entry
  settles it first (hit is a cold prime; a second demote, a promote and a tenant purge block on settle; trim proceeds; a
  tier latched off drops it unpublished); failures publish nothing and keep the day-15 typed outcomes. Five CPU tests
  cover the state machine, the exactly-once publication after done, the failing copy, the latched tier and the
  settle-first paths.
- `docs/FLAGS.md`: the door row describes the split. No new flag.
- Research: DAY17 (pre-registration committed before code, what landed, gates, stall cell), receipts from both cards,
  design note update, STATE, INDEX row; the lead record section; this file; battery receipts.

## What I checked
- Reachability: every new path is behind `MEMRA_KV_HOST_CONTRACTS=1` (default OFF); with the door OFF the host tier
  takes its previous path and `CudaTransfers` is not constructed. The identity gate default and plain, OFF and ON, is
  ALL GREEN on both cards; the hit and twin gates OFF and ON on the target card; the contract-fault gate ALL GREEN
  there (62 ok).
- One numeric program: the bytes a reader sees after publication are the bytes a synchronous copy produced (the
  identity gate's digests OFF versus ON); publication cannot precede the copy's event (the poll requires it) and a
  reader cannot see an unpublished entry (a hit on `Demoting` is a cold prime).
- The two reds on the local 5090 (fault gate hardcoding the 27B item count; the failure gate's pre-existing pool-full
  line on both cards) are gate-shape or pre-existing and assigned to lane C day 21 with the rule stated; they are not
  moved here.
- No new `MEMRA_*` read (census clean); `unsafe` unchanged (the stream calls go through the existing `cuda(...)`
  wrapper).
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, clippy, censuses, collector pytest,
  engine CPU lib tests, engine and server clippy `-D warnings`, marker census, workflow keys, perf board, diff-check)
  and the local 5090 serve-smoke (door OFF, the default).

## Review round 1 (revuto, addressed in the integ)
- The settle poll's missing-shell-or-ticket arm dropped a submitted ticket (fail open: the in-flight slot and the
  registered planes leaked, no further demote possible). It now settles and retires the ticket's sources where the
  engine is reachable and latches the tier off; a shell without a ticket drops whole. CPU test added for both arms.

## Review round 2 (revuto, addressed in the integ)
- After the settle-first call the tier may be latched off; `host_demote_prefix_ref` and `host_promote_prefix_hit` now
  re-check `armed()` and return `Off` and a miss respectively.
- The latched-off drop exit, the flip-fault exit and the fail-closed arm now book the pending reclaim wasted like
  every other demote exit; the CPU test asserts `reclaim_pending == 0` and the wasted counter after the arm.

## What I did not do
- No GPU cell of my own beyond the smoke; the door stays OFF; its decide-by review (2026-10-05) reads the stall
  receipts with the rest of the door's table.
