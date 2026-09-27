# WP-A day 67: OWED item 17 re-read on top of L' (design P2 re-applied, DAY52's rule with L' as its base)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF). Every cell
`executed-not-qualified`. The lead's order: item 17 re-read on top of item 14's lease design, now adopted (DAY63 section
6).

## 1. Pre-registration (committed before any code)

**What stands.**
- P2 (DAY52, `f9c849389`) passed (a) to (f) and failed (g) in o2: the chained request read +1.15 / +1.11 ms against
  +1.0. The placing rule put P's millisecond in the replaced twin's 32 pinned lease frees (`DAY52 PLACING -> placed in
  insert, kv`, +0.91 / +0.99 ms). P2 was reverted (`5d7260a0e`).
- L' (DAY63) removes those frees from the serving path: the replaced twin's `kv` drop now reads 0.05 ms. The ruling
  after DAY52 blocked item 17 on item 14 for exactly this reason.

**The design.** P2 re-applied unchanged: the revert of its revert, `5d7260a0e`, on the tip that carries L'. That is
P's payload reserve, copy, refill, retarget, charge and lines, with DAY52's arming rule. Merge conflicts with the code
that landed since (R1, L', T-H, DAY64's lines) are resolved by keeping both programs, never by choosing one. Any
resolution beyond a mechanical merge is named in section 2.

**The arms.**
- `base`: the tip without P2.
- `p2`: the tip with P2.
- The G'' control for the hump, `gpp`: the crates at `358749c9f`, as DAY52 ran it.
- DAY52's diagnostic `p` arm is not rerun; its question, where P's millisecond sat, was answered.

**Acceptance: DAY52 section 1's (a) to (g), verbatim, with L' carried by both arms.**
- (a) Every census and cell of P2 and of L' (the tip's server lib green). On the target card: the unit cells, and
  every gate on p2's binary (identity x4, failure x2, the fault gate default and plain, the hit gate OFF and ON, the
  pause gate).
- (b) demote: p2's copy minflt median at most 0.25 x pages; p2's copy at most base's minus 8.0 ms; at least 90% of
  p2's steady demotes full hits.
- (c) demote: p2's wall at most base's minus 8.0 ms; p2's e2e at most base's plus 1.0 ms.
- (d) promote: p2's PIN and e2e each at most base's plus 1.0 ms.
- (e) hump: `xgpp xp2 xp2 xgpp`, p2's median HUMP at most 0.15 ms with the G'' control above 0.15 (else unread,
  repeated once).
- (f) free: p2's wall at most base's plus 2.0 ms and copy at most base's plus 1.0 ms.
- (g) chain: p2's chained e2e and first e2e each at most base's plus 1.0 ms.
- Per order, both orders.

**The rule.** P2 becomes the door's copy program on the RTX PRO 6000 class if (a) to (g) hold in both orders.
Otherwise it is reverted in one commit, with its receipts banked. No bound moves after a result.

**Prediction.** (g) within +0.5 ms: the twin's frees that carried P's millisecond are gone. (b), (c) and (f) as
DAY52 read them.

**The cells.** `pro-single-p2l/`: DAY52's demote, free, promote and chain cells (base against p2, 20 boots each), the
hump cell, the gates, the hit gate and the unit cells, one collector hold each, and `day52-reading.py` unchanged. The
reader tolerates the missing `p` arm: the placing rule prints `not placed` with its deltas absent. The target card
decides; the 5090 half follows.

**Budget.** 0.4 agent-day: the re-application 0.15, the sitting 0.1, the card 0.15.
