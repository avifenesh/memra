# WP-A day 64: OWED item 18, the promote's one-tick-late publications under S4 (a placing first)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF). Every cell
`executed-not-qualified`. Pre-registered while DAY59's cells wait for a card; its code follows in order.

## 1. Pre-registration (committed before any code)

**The item** (DAY50 sections 2 and 3): on S4's tree 9 of 90 steady promotes publish one tick after the next tick top (0
of 90 on G4), and 2 of 27 on the tip; S4's H2D destination digests are queued on the copy stream after the span copies
(DAY42 section 1 step 5), and the promote publishes only when both have landed. The delay did not reach S4's (d) (PIN
+0.10 ms per order).

**What is not known.** Whether the late promotes wait on the destination digests (the receipt event lands after the
tick top that saw the copies land), on the copies themselves (a long fill ahead of them), or on the helper's `Sources`
reply (design K's checksums of the source leases, which the publication also requires).

**Step 1, the lines (log only).** The promote's published line gains, per poll, which of its three requirements was
still pending (`copies`, `receipt`, `sources`), and the tick-top poll at which each was first seen complete.

**Step 2, the cell.** The promote cell (`stall_cell.py --mode promote --n 5`, the 27B, door ON), 20 boots on the target
card; `day50-reading.py` plus the new per-poll fields. **The placing rule:** the requirement pending at the extra poll
in at least half of the late promotes is named the place; otherwise not placed.

**The designs, by the place (stated now):**

- `receipt`: the destination digests launched ahead of the fill-and-copy tail they cover is impossible (they read the
  landed destinations), so the design is a second, non-blocking poll of a pending promote after the tick's decode
  launch (an event query on the owner thread, no wait), publishing a promote whose requirements landed during the
  tick instead of at the next tick top; the parked request primes at the same tick either way, and the publication's
  requirements are unchanged.
- `sources`: T-H's split of the `Sources` job (DAY65) is the design, and item 18 closes on DAY65's reading.
- `copies`: the late promotes wait on their own copies; recorded as the copy stream's price, no design.

**Acceptance of the `receipt` design:** 0 of 27 steady promotes published after the next tick top (the tip read 2), the
promote's PIN and e2e at most the tip's plus 1.0 ms, the tenant's stall at most the tip's plus 1.0 ms, the fault and
identity gates door ON.

**Budget.** 0.3 agent-day.
