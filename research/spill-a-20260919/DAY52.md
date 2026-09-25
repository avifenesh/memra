# WP-A day 52: OWED item 17 again, the publication split and design P2 (P's revision, DAY51 section 3)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, on `22a96f20e` (P reverted, `a089a5c25`; the crates equal
`e4de9c804`'s). Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF). Every cell `executed-not-qualified`. The lead's order
after DAY51: the publication segment split first, then the reserve that refills only while the copy it replaces would
fault.

## 1. Pre-registration (committed before any code)

**What DAY51 left.** P passed (a) to (f) and failed (g): the chained request read +1.40 / +1.54 ms against +1.0. The
extra millisecond sits in the demote publication's `take-back bind and publish` segment (+0.89 / +1.07 ms: 10.28 /
10.43 on P against 9.39 / 9.36 on base), unplaced inside it. In the chain every publication replaces the host copy of
the same prompt, so that segment binds the image, inserts it, and drops the replaced twin (its 32 pinned leases and its
96 heap payloads) on the owner thread. Where entries free, P's reserve buys nothing (the copy reuses freed memory on
both arms, 7.9 ms, no faults).

**Step 1, the publication split (log only; its own commit; it is the `base` arm).** `host_demote_publish` times its
parts and the insert times its drops, and one line follows each `demote digests landed off the tick` line:
`[prefix-host] demote publication split: ticket seq=S bind X ms, reclaim Y ms, insert Z ms (twin: meta A ms, kv B ms
over K leases, f32 C ms over F payloads, rest D ms; E evicted in G ms), pause release H ms`. The twin and every LRU
victim drop through one helper that destructures the entry and drops its fields in their declaration order (the order
the compiler's own drop runs), timing the groups: `meta` (identity, generation, GLM state, version, key, tokens), `kv`
(the KV planes, pinned leases), `f32` (the conv and ssm payloads), `rest` (every later field: position, logits, draft
plane, DFlash tail, hidden row, sizes, digests, metadata, and the two charges). The same program, the same order: a
census pins the helper's field list against the struct's declaration.

**Step 2, design P2 (P whole, with an arming rule).**

1. P's reserve, copy, refill, retarget, charge and lines, as DAY51 section 1 points 1 to 6 state them (P's code
   re-applied).
2. **The arming rule.** The reserve refills only while the copy it replaces would fault. At each `Hash` job's retarget
   the helper reads the job's fresh pages: the copy's minor faults (a miss faults where memory is new, a hit does not),
   plus, when the job took any reserve buffer, the minor faults of the refill that wrote those buffers. The reserve is
   **armed** (the job's staged lengths become its target) when the fresh pages are at least half the job's staged
   pages (bytes / 4096), and **disarmed** otherwise (its target empty: it holds nothing and releases its charge). A
   disarmed reserve re-arms at the first job whose misses fault at least half its pages. The first job of a context
   arms or not by its own copy. One line when the state changes: `[prefix-host] payload reserve armed|disarmed: the job
   took M fresh pages of P (rule >= P/2)`.
3. What it does in each regime, stated so the cells can check it: while the tier fills (nothing frees), every refill
   takes fresh pages, so P2 stays armed and is P; where entries free, the refill (and a miss) reuse freed memory, so P2
   disarms after the first few demotes and the copy is today's `to_vec`, which reuses that memory without a fault.
4. Stated limits, as P's; the standing reserve exists only while armed.

**The arms.** `base` (step 1's commit: the split lines, no reserve), `p2` (P2's tip, the split lines in it), and, in the
chain cell only, `p` (step 1's tree with P's commit `d82738c14` cherry-picked onto it on the box, the split lines in it:
a diagnostic arm that places P's millisecond; nothing is decided on it).

**Acceptance, stated before any code** (P's (a) to (g), with p2 in P's place and every bound unchanged):

- (a) Semantics: step 1's census (the drop helper's field order equals the declaration order; the twin and the LRU
  victims drop only through it; the split's figures decide nothing) and P's census and cells, re-applied, plus the
  arming rule's cells: a job whose misses take fresh pages arms; a job whose hits came from a refill that reused memory
  disarms and the reserve then holds nothing and no charge; a disarmed reserve re-arms on a faulting miss; the census
  pins that the arming decision reads only the job's copy faults and the refill's faults. On the target card: the unit
  cells and every gate on p2's binary green (identity x4, failure x2, the fault gate default and plain, twin x2, the hit
  gate OFF and ON, the pause gate).
- (b) demote cell: p2's copy minflt median at most 0.25 x pages; p2's copy at most base's minus 8.0 ms; at least 90% of
  p2's steady demotes full hits.
- (c) demote cell: p2's wall at most base's minus 8.0 ms; p2's e2e at most base's plus 1.0 ms.
- (d) promote cell: p2's PIN and e2e each at most base's plus 1.0 ms.
- (e) hump: `xgpp xp2 xp2 xgpp`, p2's median HUMP at most 0.15 ms with the G'' control above 0.15 (else unread,
  repeated once).
- (f) free cell: p2's wall at most base's plus 2.0 ms and copy at most base's plus 1.0 ms.
- (g) chain cell: p2's chained e2e and first e2e each at most base's plus 1.0 ms.
- Per order, both orders, as before.

**The placing rule for P's millisecond** (the chain cell, p against base, per order; a reading that decides no
adoption): the split group (bind, reclaim, insert's twin meta, kv, f32, rest, evictions, pause release) whose median
delta is at least half of the `take-back bind and publish` delta in both orders is named the place; otherwise it is
`not placed`, with every group's delta printed.

**Readings, no clause:** the arming lines per cell (p2 is predicted armed through every demote of the demote cell and
disarmed from about the fourth demote of the free, promote and chain cells); the refill lines; VmRSS and VmHWM per boot;
the split per arm per cell (the base arm's split is also item 14's first price: what the owner holds freeing a
replaced entry's leases).

**The cells** (`pro-single-p2/`, one collector hold each): demote, free and promote A/B `o1 = base p2 x5`, `o2 = p2
base x5` (20 boots each, the DAY51 environments); chain `o1 = base p p2 x5`, `o2 = p2 p base x5` (30 boots); the hump
cell; the gates, the hit gate and the unit cells on p2. One reader, `day52-reading.py`, written before the cells run.

**The rule.** P2 becomes the door's copy program on the RTX PRO 6000 class if (a) to (g) hold in both orders; otherwise
it is reverted in one commit with its receipts banked, and step 1's split lines stay (log only). No bound moves after a
result.

**Predictions.** (b) and (c) as P read them (copy about -15 ms, wall about -12 ms); (f) flat; (g) within +0.5 ms (P2
carries P's reserve only on the chain's first demotes of a boot); the placing rule names `f32` or `kv` (the twin's
frees), not `bind`.

**What each card decides.** Each card its own. The target card first, on the 9950X class (DAY51's); a different class
reads its own verdict. The 5090 half after the card's reset.

**Budget.** 0.5 agent-day: step 1 and its census 0.1, P2 and its cells 0.15, the sitting and reader 0.1, the card 0.15
(about 2.5 hours of card time).
