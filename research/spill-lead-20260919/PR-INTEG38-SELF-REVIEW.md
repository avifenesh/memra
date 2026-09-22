# Self-review: integ38 (A day 22: Move 2 slice 3, the D2D receipt term; C days 26 and 27: the hit gate's door arm, unarmed then armed)

Author's review of the full diff `main..lane/spill-integ38-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `crates/memra-tier`: `conformance::d2d_receipt` (the `receipt_digest` program: four wrapping u64 lanes over LE words,
  byte count folded; `ReceiptTerm`; `D2dReceiptFixture`; `d2d_receipt_witnessed` rules 1 and 2, `d2d_receipt_refused`
  rules 3 and 4) and four bindings; the frozen `d2d_capture` and `d2d_restore` schedules gain the receipt clause only.
- `crates/memra-engine`: `cu/tier_receipt.cu` (`d2d_receipt_digest`, `tier_delay_spin`) as its own fatbin, loaded only
  by the copy-stream engine; `ReceiptScratch` per D2D batch (zero-filled lanes, pinned readback, seal event); on the
  copy stream behind the producer fence: source digest, copy, destination digest, D2H of the lanes, receipt event;
  `progress` lands a D2D item only with its lanes; `d2d_receipt`; `inject_d2d_early_reader`; `synchronize` waits on the
  receipt event too (the lead fix, below). Three GPU cells (oracle, early reader, price) and one CPU census test added.
- `crates/memra-server/src/worker.rs`: receipt lines for both classes; `ReceiptMismatch` arms (capture: ticket retired
  and acknowledged, fresh planes taken back and dropped, nothing published, tier and route latch; restore: destination
  cache dropped, source pin released, nothing primed, tier and route latch); `d2d-delay-capture` and `d2d-delay-restore`
  fault values, one-shot, taken by their own class only. Two CPU tests.
- `tools/kv-host-contract-fault-gate.sh`: cells `d2d-capture` and `d2d-restore` (the refused receipt names two different
  digests, exactly one refusal, nothing published or landed off the tick, the tick program served r2, texts equal, no
  ticket leaked). `tools/spec-on-cache-hit-gate.sh`: the door arm arms the host tier (`MEMRA_KV_HOST_MB=8192`), asserts
  the arming line, the door line, no latch line and at least one route submission across the two boots, prints an
  entry-class census; `stop()` kills only its own `flock` child. `docs/FLAGS.md` (fault values, door sentence, fatbin
  plumbing), `docs/KERNELS.md` (two rows), `docs/TESTING.md` (door-arm paragraph). No new `MEMRA_*` runtime read.
- Research: A DAY22 and receipts, C DAY26 and DAY27 with receipts on both cards, `HOSTPREFIX-DOOR.md` corrections
  (every earlier door-ON hit-gate line marked unarmed), INDEX rows, the lead record section (rulings 33 and 34), this
  file, battery receipts.

## What I checked
- Reachability: every new path is behind the door; the naked engine never loads the receipt fatbin
  (`new_with_copy_stream` only); with the door OFF the hit gate boots exactly as before.
- The receipt's ordering on the copy stream: producer wait, zero-fill wait, (fault delay), source digest, copy,
  destination digest, item event, lanes D2H, receipt event; the item cannot land before its lanes are read
  (`progress`), so a landed batch always carries its receipt. The fault arm launches the destination digest on the owner
  stream with no wait and makes the copy stream wait on that read, so the "early" read is ordered before the copy by
  construction and the refusal is deterministic (both fault-gate cells).
- **Finding, fixed here (`269ef2cec`):** `synchronize` (the `ContractWait::Block` host wait at retire seams and
  shutdown) waited on the items' events only; the receipt event is recorded after them, so a Block settle in that window
  read a landed copy as unlanded and latched the tier with `did not land after a host wait on every item's event`. The
  wait now covers the receipt event; census test pins the order and that exactly two recorded-event waits exist.
- **Finding, fixed here (`9d4b17761`):** the GPU-less `DOCS_RS=1` build (docs.rs and the cross-target clippy pass CI's
  runners take) had no placeholder for the new fatbin and failed at `include_bytes!(env!("MEMRA_TIER_RECEIPT_FATBIN"))`;
  the stub branch now emits it. This is the pass that would have turned CI red on the PR.
- **Finding, fixed here (`cc754b476`):** the oracle GPU cell raced its owner-stream upload against the copy-stream
  digest and lost on the local 5090 at 8 MiB (green on the target card twice); the test now orders its producer as the
  engine does. The kernel and the oracle agree on both cards (5090: 4 of 4 under the lock).
- **Revuto round 1, fixed (`d2d-delay-*` spin once per batch):** the fault's spin sat inside the per-item loop and
  multiplied by the item count (6.4 s on a 32-plane entry); it is gated on the first item now, the doc says once, the
  census test pins it; fault gate ALL GREEN on the 5090 with both D2D cells refusing, engine cells 5 passed.
- Fail-closed arms mirror the #638 rules: the capture mismatch retires the ticket before touching planes and counts any
  plane that did not come back; the restore mismatch frees the cache only because the copy landed (never a free under a
  running copy; the `Latched` arm still forgets) and hands the pin back through `RestoreSettled::Dropped(pin)`.
- The digest is a receipt over bytes, not a numeric program over tokens; the CPU oracle and the kernel share the lane
  constants, `j + 1` indexing, LE words and zero padding; `d2d_receipt_digest_matches_the_cpu_oracle` on the target card
  and on the local 5090 (this battery's GPU step, stated either way).
- Cell (v) is reported as A read it and not relaxed: the pair costs 2.12x to 2.15x the copy on the copy stream at
  158 MiB; the day-19 clause is met; the door review weighs it (recorded in the door table and the record section).
- C day 27: the ON arm engagement is asserted by line, not assumed; the seven route submissions and the identity clause
  over the off-tick restore route are quoted; the earlier lines are receipts of the unarmed program and stay as such.
- Battery on this tree in the receipts (fmt, portable suites, memra-server suite, clippy, censuses, collector pytest,
  engine CPU lib tests, tier tests, engine, server and tier clippy `-D warnings`, marker census, workflow keys, perf
  board, diff-check), the local 5090 serve-smoke and the engine `d2d_*` GPU cells under the 5090 lock (stated either way).

## Push regime
Engine source in the range: pushed with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged); no GPU
qualification claimed; every cell executed-not-qualified. Revuto: if capped or unavailable, this comment is the review.
