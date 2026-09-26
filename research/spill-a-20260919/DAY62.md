# WP-A day 62: OWED item 13, the capture retire seam's Block settle (a price first, then a design)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF). Every cell
`executed-not-qualified`. Pre-registered while DAY59's cells wait for a card; its code follows in order.

## 1. Pre-registration (committed before any code)

**The item** (DAY38 section 4): `capture published off the tick (seed): .. settled synchronously by a session retire;
the settle held the owner thread 2607.12ms` under the `d2h-delay` arm; day 25 priced the seam at 0.4 ms on a copy that
had landed. Read from the code (`worker.rs`, the retire pass): when any session finishes while a capture is pending, the
worker settles the capture with `ContractWait::Block` before any session leaves `active`, because the capture reads a
live session's KV planes on the copy stream with no source retained. So the owner waits for the capture's copy and for
everything queued ahead of it on the one side stream (a demote's copies, its 105 ms receipt kernel at long entries, a
promote's fill and spans), and it waits even when the retiring session is not the capture's source.

**Step 1, the lines (log only).** The retire settle's line gains whether any retiring session is the capture's source
(`source retiring: yes|no`) and the copy-stream work queued ahead of the capture at its submission (the tickets in
flight on the copy stream, by kind).

**Step 2, the price** (one RTX PRO 6000, the 27B, door ON; a new `stall_cell.py` mode `retire-seam`): the tenant
streams; at the fire, a 4096-token prompt demotes the resident long entry (its copies and receipt kernel queue on the
copy stream) and, at once, a short fresh prompt (`max_tokens 1`) seeds a capture behind them and retires; and a second
shape where the retiring request is not the capture's source (a third short request retires while the capture is
pending). N=5 per order, both orders against the plain `prime` control; the owner hold of each retire settle read from
its line.

**The designs, selected by the price** (stated now):

- R1 (no-source retires do not settle): the retire pass settles the pending capture only when a retiring session is its
  source. Selected when the no-source shape's median hold is above 1.0 ms.
- R2 (the source's planes held by the capture): a retiring source session's KV planes move into the pending capture,
  which drops them (or parks the session's reuse entry) only after its landing; the retire never blocks. Selected when
  the source shape's median hold is above 1.0 ms.
- Both when both are above; neither when both are at or below 1.0 ms (the seam closes as priced).

**Each design's acceptance, stated now:** the identity gate door ON default and plain, the fault gate default and plain
(its capture cells included), the hit gate door ON; the selected shape's median hold at most 0.5 ms per order; the
tenant's stall and the intruders' e2e at most the tip's plus 1.0 ms; for R2 a census that no path drops, rewrites or
parks a capture source before its landing.

**What each card decides.** The target card (the long demote's receipt kernel is the queue the seam waits behind).

**Budget.** 0.5 agent-day: the lines 0.05, the mode and the sitting 0.1, the card 0.1, the design(s) 0.25.
