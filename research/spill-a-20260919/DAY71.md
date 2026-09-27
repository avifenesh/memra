# WP-A day 71: design F2 (item 18's next design), pre-registered before any code

Item 18 (DAY64) placed the promote's late publication in its span fill: 8.63 ms of the 12.9 ms span work, the host
function that copies the resident heap payloads into the pinned staging before the H2D. Design F moved the fill to its
own stream and was reverted (DAY64 section 8): the copies waited on it all the same. DAY64 section 5 recorded F2 for
after P2's verdict, and P2 adopted (DAY67 section 5). Every cell is `executed-not-qualified`.

## 1. Pre-registration (committed before any code)

**The price on record.**

- The promote's span work, after its start event, on the target card: fill 8.63 ms, copies 3.30, digests 1.01
  (DAY64B, N=180). 170 of 180 steady promotes land after the next tick top (DAY64 section 3).
- The demote side: the hash helper copies each landed staging span into the entry's heap payload and hashes it (DAY49:
  copy 23.7 ms at 64 tokens before P2). P2's reserve took the copy's page faults off it (DAY67), and T-H' its thread
  time (DAY65 section 10).
- The resident recurrent payloads of a 27B entry are 156.9 MB, pageable today (`HostF32::Heap`, charged to the
  pageable ledger).

**Design F2: under the door, a resident entry's recurrent payloads live in pinned memory, and the staging copies on
both sides go away.**

- **(F2.1) The demote.** Each D2H span lands in a pinned buffer that becomes the entry's own payload
  (`HostF32::Pinned`), instead of a staging buffer that the helper copies to the heap and returns to the set.
  - The helper hashes the payload in place, as it hashes a KV lease view today.
  - The buffers come from a pool of their own on DAY70's design Q terms: charged their length to the entry's residency
    on the pinned dimension, the pool charged its idle buffers within its cap, drained at every purge (DAY69's P), and
    a buffer past the cap allocated at its length.
- **(F2.2) The promote.** Each H2D span's source is the entry's own pinned payload. There is no fill and no promote
  staging: the span copies read the resident bytes directly, as the KV items read their leases.
- **(F2.3) The ledger.**
  - A resident's recurrent bytes move from the pageable dimension to the pinned one. The LRU is unchanged: it budgets
    the entry's actual bytes as today, KV plus recurrent.
  - Residents stay at or under one budget of pinned bytes, where today their KV and recurrent split between the two
    dimensions. The pinned ledger's three budgets (DAY70 section 2) are unchanged.
  - The pageable dimension keeps only what stays on the heap: logits, hidden and the like.
- **(F2.4) What goes.**
  - The span staging set's demote and promote roles.
  - P2's payload reserve, whose only subject is the staging-to-heap copy.
  - The fill task.
  - Each is deleted in F2's commit if F2 adopts (door hygiene), and each is named in the census.
- **(F2.5) Unchanged.** The span receipts (source digests at the D2H, landed digests over the same bytes), the bind's
  checksums over the same bytes, the entry's surface and program identity, and door OFF (byte-identical, nothing
  constructed).

**Acceptance** (DAY64 section 5's for the promote, plus the demote and the memory):

- **(a) Correctness:**
  - the engine's H2D and D2H span cells and the worker's span cells (`option_b_span_*`, `option_c_span_*`), with the
    promote's landed spans bitwise the demoted bytes;
  - the fault gate default and plain, where `span-flip-resident` must still be refused (its flipped byte now sits in
    the entry's pinned payload);
  - the identity gate default and plain, door OFF and ON; the hit gate OFF and ON; the pause gate;
  - a CPU census: under the door no promote reads a heap copy of a recurrent payload, and no demote copies a landed
    span to the heap.
- **(b) The promote cell:** at most 9 of 180 steady promotes late, and the span work's fill term at most 0.5 ms, per
  order.
- **(c) The promote cell:** the intruder e2e at most base's minus 5.0 ms and the PIN at most base's plus 1.0 ms, per
  order.
- **(d) The demote cell:** the wall t0 to publication and the helper at most base's plus 1.0 ms. Reading: the helper's
  copy term.
- **(e)** The tenant's stall at most base's plus 1.0 ms, the hump at most base's plus 0.15 ms, and the chain cell's
  e2e at most base's plus 1.0 ms, per order.
- **(f) Memory:**
  - the demote counts equal between arms in every cell and order, so no pinned refusal where base admits;
  - readings: each boot's peak pinned and pageable ledger use, and the staging set's charge (zero on F2).
- **The rule.** F2 adopts if (a) to (f) hold in both orders; otherwise it is reverted in one commit with its receipts
  kept.
  - A (b) or (c) that passes while (d), (e) or (f) fails is recorded as read: a promote win the demote or the memory
    paid for.

**The sitting.**

- The pair: F2's tip against its parent (main with T-H', P2, L', Q and P). One RTX PRO 6000 Blackwell card, sole
  tenant, a host with at least 16 logical CPUs.
- The cells:
  - the 11 gates on f2;
  - the promote, demote and chain cells, base against f2, 20 boots each, in P2's cell environment;
  - the hump (4 boots);
  - then `f2-reading.py`, whose last line is `F2 VERDICT -> ..`.
- About 2.5 hours of card time. The 5090 half follows under the per-hardware rule.

**Budget:** 1 agent-day: the code and its censuses 0.5, the CPU cells 0.1, the sitting and reader 0.2, the reading
0.2.
