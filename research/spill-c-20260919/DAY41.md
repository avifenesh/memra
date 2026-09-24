# WP-C day 41 (2026-09-24): OWED C9, the 9B entry's conv, ssm and hidden split, reconciled on banked lines

Records only: no card, no artifact read, no engine code. Written while the day-40 `attrib` cell waits for the RTX 5090
(lane B's queue holds it). The tree is the lane tip after `08210a291`.

## 1. Pre-registration (committed before the reader runs)

**What is owed.** `DAY38.md` section 3 bounded the 9B host entry from banked logs and named the line that would split
it: "No banked line separates conv from ssm from hidden." `OWED.md` C9: lane A day 31 item 1d landed that line
(`demote copy complete off the tick: ... by slot class: conv C (B B), ssm S (B B), hidden H (B B), logits L (B B)`,
lead ruling 44), so the split is now a reading of banked lines against day 38's registered bounds.

**The bounds, verbatim from `day38-cpu/byte-split.log`** (registered on day 38, not moved here):

- `DAY38 9B HEAP: ... both hold: [53650000, 53699472)`
- `DAY38 9B LOGITS, all three printed figures: host_bytes in [54600528, 54650000) B; 4 x len(last_logits) in (950272, 1099744) B`
- `DAY38 9B RECURRENT+HIDDEN: conv + ssm + 4 x len(last_h) = dead.bytes - KV in [52599728, 52699728) B`
- `DAY38 9B KV plain: d2d bytes=950272 ...` and 256 B of token ids (`DAY38.md` section 3 item 2).
- `DAY38 9B PAYLOADS: 50 heap payloads = Logits (1) + Hidden (1) + Conv/Ssm slots (48)`.

**Input.** Every `demote copy complete off the tick` line carrying `by slot class:` in lane A's banked RTX 5090 logs
under `research/spill-a-20260919/rtx5090-day3*/` whose heap payload count is 50 (the 9B plain class of day 38; the
27B lines carry 128 or 130 payload items and are not this item). Both 9B classes are read: plain (`hidden 1 (0 B)`)
and draft-bearing (a non-zero hidden row).

**Rule (the reader `day41-9b-split.py`).** For each distinct `by slot class` tuple on those lines: `heap = conv + ssm +
hidden + logits` must equal the line's own heap figure within its one-decimal MB rounding; `conv + ssm + hidden` in
[52,599,728, 52,699,728); `logits` in (950,272, 1,099,744); `heap` in [53,650,000, 53,699,472) for the plain class;
`KV + 256 + heap` in [54,600,528, 54,650,000) for the plain class; slot counts `conv + ssm = 48`, one logits, one
hidden. PASS iff every clause holds on every plain tuple. The draft-bearing tuples are reported with the same
arithmetic and no bound (day 38 registered bounds for the plain class only).

**Decision.** PASS closes C9 in `OWED.md` with the split stated as bytes and shares. FAIL keeps C9 open with the
failing clause quoted, and the reading goes to lane A (the line is A's).
