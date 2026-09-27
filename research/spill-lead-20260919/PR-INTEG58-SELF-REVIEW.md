# Self-review: integ58 (A day 36: the D2D half's restore price cell CLOSES on the target card, M' PASS on the target card, the restore's recurrent copy timed, log only)

Author's review of the full diff `main..lane/spill-integ58-20260924`, posted as a PR comment per the owner rule.

## What the diff is
- Server: `worker.rs` +99/-4, a log-only instrument on the contracts-door restore path (`host_restore_submit` and
  the restore's landing line):
  - `RestoreRecurTiming`: the host time around the recurrent copy loop, and two timing events recorded on the owner
    stream around it;
  - the landing line prints the owner-stream time only when the end event is already complete, otherwise `pending`;
  - census `day36_the_restore_recurrent_copy_is_timed_without_a_wait`.
- No new `unsafe`, no new `MEMRA_*` name, no FLAGS or KERNELS row owed.
- Research: A's DAY36.md with the receipts `rtx5090-day36/` and `pro-single-day36/` (506 files checked against the
  box manifest), `day36-reading.py`, STATE, OWNER-THREAD-OFFLOAD and the INDEX row; the integ58 record section
  (ruling 53), both batteries, and this file.
- The branch is a fast-forward over main `1d0cf13bc`.

## What I checked
**The instrument**
- No control flow changes. `host_restore_submit` returns the timing next to its existing tuple, and the two log
  lines format it. The test fixture builds it with `events: None`.
- No host wait: `gpu_term` calls `is_complete()` before `elapsed_ms`, and the census asserts there is no
  `synchronize` in the term or between the first event and `record_producer`.
- The events bracket the copy loop and are both recorded before the producer fence, on the stream the copy already
  runs on. An event record adds no dependency to that stream.
- One numeric program per request holds: nothing the restore copies or computes changes, only a log line.

**Review patterns**
- The seven spill patterns (move-then-match, parked idle wait, gate literals, happy-path release, borrowed sources,
  budgeted booking, guard identity) were read against the diff. None applies: no new state, resource or release.

**Verdicts and pre-registration**
- The verdict lines in the record are copied from DAY36 sections 2 and 4.
- The price cell and its rule were registered in DAY33 section 6 and restated before any day-36 code
  (`0c2c62f77`). The target-card sitting was registered in DAY36 section 3 and amended in 3a before it ran, to put
  #711's guard fix on the M' arm.
- The 5090 reading is labeled a reading, as registered; the rule is read on the target card only.

**Batteries**
- CPU battery 15 of 15 rc=0 (server 894, engine lib 546, portable 368 with 0 skipped, pytest 87).
- RTX 5090, binary `1532f987`, hashed after serve-smoke: serve-smoke, the serial engine span (10) and worker (18)
  cells, identity (12 ok), fault default and plain (160 ok each), hit OFF/ON (61 and 68), admit-mem-burst and
  spec-ctx-edge, all green.
- Target card: A's BOX5 sitting, every gate and unit cell green, receipts checked 506 of 506.

**Hygiene:** no provider name, host, id or price in tracked files (the only address in the added lines is
loopback). No em dash in authored lines.

## Push regime
Engine source changed, so the branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced,
logged). Every other hook ran. No tag: the change is a log line on a default-OFF door. Revuto: if capped or
unavailable, this comment is the review.
