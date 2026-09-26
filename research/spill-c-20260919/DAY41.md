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

## 2. The reading

Command: `python3 research/spill-c-20260919/day41-9b-split.py research/spill-a-20260919/rtx5090-day3{1..6}` into
`day41-cpu/split.log` (`# exit=0`). Input: 1,120 copy-complete lines with 50 heap payloads in 216 of lane A's banked
RTX 5090 logs (days 31 to 36), two distinct tuples. Verbatim:

- `DAY41 9B TUPLE class=plain lines=983 conv=24x98304=2359296 ssm=24x2097152=50331648 hidden=0 logits=993280 (248320
  f32) heap=53684224 (line 53.7MB) conv+ssm+hidden=52690944 entry=kv+tokens+heap=54634752 | shares of entry: kv 1.739%
  conv 4.318% ssm 92.124% hidden 0.000% logits 1.818% | heap_rounds_to_line=ok slots=ok recurrent_hidden_bound=ok
  logits_bound=ok heap_bound=ok entry_bound=ok`
- `DAY41 9B TUPLE class=draft-bearing lines=137 conv=24x98304=2359296 ssm=24x2097152=50331648 hidden=16384
  logits=993280 (248320 f32) heap=53700608 (line 53.7MB) conv+ssm+hidden=52707328 entry not computed (KV with the draft
  plane is not on this line) | shares of heap: conv 4.393% ssm 93.726% hidden 0.031% logits 1.850% |
  heap_rounds_to_line=ok slots=ok`
- `DAY41 9B SPLIT plain_tuples=1 -> PASS`

One reader fix, stated: the first run printed an `entry` and shares of entry for the draft-bearing tuple using the
plain class's KV figure, which is wrong for that class (its KV carries the draft plane, day 38 section 3). The print was
changed to shares of heap for that class; no clause and no bound involves it, and the plain tuple's line and the
verdict were the same in both runs.

**The split (the 9B plain 64-token entry, 54,634,752 B).** KV planes 950,272 B (1.739%), token ids 256 B, conv 24 slots
of 98,304 B = 2,359,296 B (4.318%), ssm 24 slots of 2,097,152 B = 50,331,648 B (92.124%), the hidden row 0 B (a plain
entry has no `last_h`), logits 993,280 B = 248,320 f32 (1.818%). Every figure is inside day 38's registered bounds.
The draft-bearing class carries the same conv, ssm and logits and a 16,384 B hidden row (4,096 f32). Outside the rule,
one observation: the logits row is 248,320 f32 while day 38's `loaded` lines print 248046; the difference (274) is not
explained by any banked line and is recorded, not read into.

**Decision, as registered.** PASS: C9 closes in `OWED.md`.
