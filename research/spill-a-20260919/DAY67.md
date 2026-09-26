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

## 2. P2 re-applied on L' (`c58f32f7d`), and the sitting prepared

- The revert of `5d7260a0e` on the tip conflicted in eight places in `worker.rs` and one in `docs/TESTING.md`. Named,
  as section 1 requires:
  - **The payload map (not mechanical).** P2 takes a reserve buffer per staged payload inside the payload map. T-H
    (DAY65) made that map run in shares on scoped threads, and the reserve is the helper thread's own mutable state.
    - The merge hands the reserve's buffers out first, on the helper thread and in the job's order: the same
      assignment P2's sequential map made. `staged`, `reserve_hits` and `staged_lengths` are counted there.
    - Then each share fills its payload's buffer (`copy_from_slice` of the staged slice) or allocates (`to_vec`), and
      hashes, as T-H's shares do.
    - The arming rule reads `split.copy_minflt`, now summed over the shares' threads (each share's `thread_minflt`),
      the same pages counted.
  - **The ledger.** L added a third pinned term and P2 a third pageable term, so both dimensions are now three host
    budgets. The unused `twice` closure is removed. The governor test (`..binds_at_twice_the_budget`) now admits a
    third whole-budget charge and refuses a fourth; its old third-charge refusal came from whichever dimension the
    tree had left at twice.
  - **The helper split line** carries both terms: `(helper Y ms); reserve H of S staged; T threads`. DAY52's reader
    and T-H's reader match their own terms.
  - **The rest are mechanical merges.** The struct fields, the helper loop's head (T-H's thread count, then P2's
    refill loop), the censuses re-pointed at the merged lines (day 51's take, day 65's shares map, the D2H spans'
    order, day 49's split), and TESTING.md's two entries both kept.
  - Server lib `950 passed; 0 failed; 26 ignored` (P2's six cells among them); clippy `-D warnings`; fmt (`day67/`).
- **The base** is the tip with P2 taken back out: branch `lane/spill-a-p2l-base-20260926` at `f4e84f11e`. Server lib
  `944 passed`. It is never merged.
- `day52-reading.py` reads a sitting without the `p` arm (`26b3a3907`).
- **The sitting** `pro-single-p2l/`, receipts `/root/spill-receipts/a-p2l`: `build.sh <tip> f4e84f11e` (p2, base and
  gpp at `358749c9f`), then `driver.sh`. It runs DAY52's cells (demote, free, promote and chain, base against p2, 20
  boots each), the hump (xgpp xp2 xp2 xgpp), the gates with twin and pause, the hit gate and the unit cells, then
  `day52-reading.py`. The last line is `DAY52 P2 -> ..`, read here as item 17's verdict on L'. About 3 hours of card
  time.
- T-H's own verdict is pending. P2 on L' carries T-H in both arms, so its A/B isolates P2. If T-H reverts, P2 is
  re-merged onto the revert under its own named resolution.

## 3. T-H reverted: the running P2L sitting becomes a diagnostic; the corrected pair

- The sitting started at 16:10Z on `c03ca6b06` (base `f4e84f11e`) has T-H's shares in both arms. There, P2's reserve
  buffers are filled inside T-H's parallel shares. Clauses (b), (c), (f) and (g) read the helper's copy time and the
  walls that T-H changes, so its reading does not transfer to the tree that would ship after T-H's revert (P2's
  sequential map).
- By section 2's own sentence ("If T-H reverts, P2 is re-merged onto the revert"), that sitting is a **diagnostic**.
  If it completes, its reading is banked as one; nothing is decided on it.
- **The re-merge onto the revert** (`06b2d31db`): P2's payload map is DAY52's sequential code verbatim (the reserve's
  `take` inside the map, then `copy_from_slice` or `to_vec`, then the hash). The helper split line reads `(helper Y
  ms); reserve H of S staged`. Two censuses are re-pointed back to that code.
  - The ledger keeps both third terms (L's pinned, P2's pageable), and the governor test stays at "a fourth charge
    refuses".
  - Server lib `948 passed; 0 failed; 26 ignored`; clippy `-D warnings`; fmt.
- **The corrected pair.** Tip `06b2d31db` (after push). The base is the integ69 branch
  `lane/spill-a-integ69-20260926` at `dba7c0a0c`, whose crates differ from this tip by P2 alone: `git diff` touches
  only `worker.rs`, and every changed line is P2's.
  - `pro-single-p2l2/`, receipts `/root/spill-receipts/a-p2l2`: `build.sh <tip> dba7c0a0c`, then `driver.sh` (the
    same cells as section 2's sitting).
