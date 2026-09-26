# WP-A day 65: OWED item 20, the hash helper's per-payload work across threads (design T-H)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF). Every cell
`executed-not-qualified`. Pre-registered while DAY59's cells wait for a card; its code follows in order (after items
11 to 14, 17 and 18). DAY64 is reserved for item 18.

## 1. Pre-registration (committed before any code)

**The price on record** (DAY49 section 3, DAY51 section 3, DAY52 section 3): the helper's job is one thread: at
64-token 27B entries copy 23.7 ms plus hash 59.1 ms over 96 independent staged payloads (83.5 ms), and at 5122-token
entries the bind's KV re-hash adds about 56 ms over 32 independent lease views (helper 122 to 140 ms). The demote
publishes only after the job (wall 101 ms at 64 tokens, 362 to 444 ms at long entries), and a request that meets the
entry `Demoting` waits for it (DAY43: at long entries the chained hit waits on the helper, not the copy).

**Design T-H.** The helper's `Hash` job splits its work across `min(8, available_parallelism / 2)` scoped threads (at
least one): the staged payloads (copy and digest, each whole on one thread) and the lease views (digest each), in
contiguous shares of the job's order; the reply keeps the job's order. The digest program per payload and per view is
unchanged (the same `checksum` over the same bytes). The `Sources` job (the promote's checksums) splits its views the
same way. Design T's thread rule for the fill (item 3) is the precedent.

**Acceptance.**

- (a) Bitwise: every digest equal to the one-thread helper's on the same job (a CPU cell over synthetic payloads of the
  27B's shapes, 1 to 8 threads); the identity, fault and hit gates door ON; the pause gate.
- (b) The demote cell (the 27B, 64 tokens): the helper time median at most half of the tip's, and the steady wall t0
  to publication at most the tip's minus 20 ms, per order.
- (c) The chain cell (DAY52's): the chained request's e2e at most the tip's minus 30 ms per order.
- (d) The tenant's stall and the hump at most the tip's plus 1.0 ms and 0.15 ms; the promote cell's PIN and e2e at most
  the tip's plus 1.0 ms.
- Readings: the helper's CPU time and the threads it used; the 5090's half on its own host class.

**What each card decides.** The target card (b) to (d); the 5090 its own (b) and (d) after.

**Budget.** 0.5 agent-day.
