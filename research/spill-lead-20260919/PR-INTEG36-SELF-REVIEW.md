# Self-review: integ36 (A day 20: Move 2 slice 1, the prefix capture off the tick under the door; B day 30: memra#423 is a darklanes fix, memra keeps the seam)

Author's review of the full diff `main..lane/spill-integ36-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-tier`: `CopyDirection::DeviceToDevice` in the contract (the generic transfer path refuses it with
  `Unsupported` at both sites, so nothing outside the capture op can use it), the `d2d_capture_publish` schedule and its
  three bindings including the red arm (publish before the event fails the schedule). The frozen schedules are untouched.
- `crates/memra-engine/src/tier_transfer.rs`: `D2dCapture` (borrowed source slice, owned registered destination lease,
  bytes, producer fence) and `submit_d2d_capture`: requires the copy stream (`Unsupported` otherwise), validates layout,
  epoch and owner (source and destination on this owner's context and stream), overflow-checks the sequences, waits on
  the producer fence's event on the copy stream, `memcpy_dtod`, records the completion event there, fenced at submit like
  a D2H, no owner wait (the destination is read only after publication at the tick top). `progress` leaves a capture's
  checksum `None`, so the host contract gate refuses a capture item as `Corrupt` until a witnessed checksum exists
  (slice-1 clause, tested).
- `crates/memra-server/src/worker.rs`: the `Capturing` state with its pending record; `prefix_capture_off_tick` at the
  one capture site after `prepare_snapshot` and before the tick program (recurrent state cloned on the owner stream at
  the boundary; fresh planes registered with twins; one batch); settle at the tick top after the promote poll, in both
  idle waits, at purge, admission reclaim, the three trims and the run-loop exit; failures typed and fail-closed
  (missing shell or ticket latches the tier off); the reclaim bookkeeping at every exit; five CPU tests. The
  spec-boundary publishes still take the tick program (stated).
- `crates/memra-server/src/metering.rs` (B): the module header says the stock binary wires no metering implementation
  (the reference ledger moved to darklanes on 2026-08-29); no code change.
- `docs/FLAGS.md`: the door row names the capture half. No new flag.
- Research: A DAY20 (pre-registration, the design finding, what landed, gates, the stall cell), target-card receipts;
  B DAY30 (the corrected premise, the sequence diagram, the before and after cells, the #464 gap), receipts; STATE
  files; INDEX rows; the lead record section with ruling 32; this file; battery receipts.

## What I checked
- Reachability: every new path is behind `MEMRA_KV_HOST_CONTRACTS=1`; with the door OFF the capture takes the previous
  synchronous program (the `capture_off_tick_disabled` path). The identity gate OFF versus ON on the target card is ALL
  GREEN in every arm on A's tree, and the hit and twin gates OFF and ON as well.
- One numeric program: a captured entry is published only after every item's event (the schedule's red arm proves the
  rule), and a hit on a `Capturing` entry does not see it; the identity gate compares OFF and ON digests.
- The stall cell is reported as `flat` with the reason (the intruder's prime dominates), not as a door win.
- B's darklanes fix is not in this repo; the memra change is a header sentence. The darklanes branch gets its own PR
  under the owner's merge law (lead action).
- No new `MEMRA_*` read (census clean); `unsafe` unchanged.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, clippy, censuses, collector pytest,
  engine CPU lib tests, tier tests, engine, server and tier clippy `-D warnings`, marker census, workflow keys, perf
  board, diff-check) and the local 5090 serve-smoke if the lock frees within the window (stated either way).

## Review round 1 (revuto, addressed in the integ)
- Source lifetime: a pending capture now settles `Block` before any session leaves `active` (the retiring session's
  cache is dropped on the owner stream or parked for reuse; either could run under the copy stream's read). The engine
  keeps no source, so the worker holds the invariant at the seam that moves the source.
- Refusal path: a refused submission drains the owner stream, releases the producer fence, and latches the route off
  typed if the fence will not release, instead of dropping a `Busy` and leaking the fence for the boot.
- Both pinned by a source census test; the admission-book lock test still holds.

## What I did not do
- No GPU cell of my own beyond the smoke; the door stays OFF; the 5090 door gates on this tree are C day 24's.
