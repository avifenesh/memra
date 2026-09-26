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

## 4. P2L2, read as registered: (b) to (g) PASS; (a)'s unit step void (a harness defect), repeated whole

- Run by the lead on one RTX PRO 6000 Blackwell Workstation card, `build.sh 064f9fa0d dba7c0a0c` then `driver.sh`,
  16:25Z to 18:09Z. Mirror `pro-single-p2l2/box/`, sha256-checked against the box manifest (0 mismatches); the
  executables are recorded by hash.
- Verbatim (`box/reading-day52.log`), the clause lines:

      DAY52 P2 (b) cell=demote order=o1 minflt p2-median (0.25 x pages)=+0.00 rule <=+9576.42 | copy p2-minus-base=-15.88 rule <=-8.00 | hits fraction of steady demotes=+1.00 rule >=+0.90 -> PASS
      DAY52 P2 (b) cell=demote order=o2 minflt p2-median (0.25 x pages)=+0.00 rule <=+9576.42 | copy p2-minus-base=-16.29 rule <=-8.00 | hits fraction of steady demotes=+1.00 rule >=+0.90 -> PASS
      DAY52 P2 (c) cell=demote order=o1 wall p2-minus-base=-12.40 rule <=-8.00 | e2e p2-minus-base=+0.34 rule <=+1.00 -> PASS
      DAY52 P2 (c) cell=demote order=o2 wall p2-minus-base=-12.40 rule <=-8.00 | e2e p2-minus-base=+0.07 rule <=+1.00 -> PASS
      DAY52 P2 (d) cell=promote order=o1 pin p2-minus-base=-0.10 rule <=+1.00 | e2e p2-minus-base=-0.03 rule <=+1.00 -> PASS
      DAY52 P2 (d) cell=promote order=o2 pin p2-minus-base=+0.00 rule <=+1.00 | e2e p2-minus-base=-0.02 rule <=+1.00 -> PASS
      DAY52 P2 (f) cell=free order=o1 wall p2-minus-base=+0.20 rule <=+2.00 | copy p2-minus-base=+0.03 rule <=+1.00 -> PASS
      DAY52 P2 (f) cell=free order=o2 wall p2-minus-base=+0.10 rule <=+2.00 | copy p2-minus-base=+0.00 rule <=+1.00 -> PASS
      DAY52 P2 (g) cell=chain order=o1 chain p2-minus-base=-0.06 rule <=+1.00 | first p2-minus-base=-0.00 rule <=+1.00 -> PASS
      DAY52 P2 (g) cell=chain order=o2 chain p2-minus-base=-0.03 rule <=+1.00 | first p2-minus-base=-0.02 rule <=+1.00 -> PASS
      DAY52 P2 (e) hump xp2=+0.028 rule <=0.15 control xgpp=+0.534 (humps) -> PASS
      (a) every gate 0 | unit [unit-cells parallel=3/3 engine-serial-rc=0 door-rc=101 cpu-rc=0 engine-census-rc=0 tier-rc=0] -> FAIL
      DAY52 P2 -> FAIL

- Read:
  - Every timing clause passes. **(g), DAY52's one failure, now reads -0.06 / -0.03 ms against +1.0**, with L' having
    removed the twin's lease frees, as DAY63's ruling predicted.
  - (b): the copy is 15.9 / 16.3 ms faster with every steady demote a full hit and no faults. (c): the wall is 12.4
    ms faster. (e) passes against a humping control. (d) and (f) are flat.
- **(a)'s unit step:**
  - Its `door-rc=101` is the same 13 worker GPU cells integ69 found: 12 of `left: (2304, 0, 0) right: (1392, 0, 0)`
    and the staging refusal cell.
  - They were placed in DAY63 section 7 as test arithmetic that predates L1.2 and L1.5, not as any defect of P2's
    tree or of L'. They fail the same way on every tree carrying L' without `411177fea` and `28c7aa6c1`, including
    P2's base.
  - A cell whose harness is defective reads nothing. The unit step is **void**, as W's a1 filter was (DAY61 section 3),
    and it repeats whole once on P2's own tree with the corrected cell arithmetic.
  - The repeat uses branch `lane/spill-a-p2l2-unit-20260926` at `a2419d3e1`: `064f9fa0d` plus the two test-only
    commits, with the production code byte for byte P2L2's p2 arm. The server binaries are untouched, so (b) to (g)
    stand as read.
  - Script `pro-single-p2l2/unit-rerun.sh`, in its own clone: `bash unit-rerun.sh build`, then under one collector
    hold `bash unit-rerun.sh @COLLECTOR_LOCK_FD@`. It runs the same unit cells as P2L2's.
  - **P2's verdict is ADOPT if that step reads all green (`unit-cells parallel=3/3 ... door-rc=0 ...`), and REVERT
    otherwise.** The same 18 worker cells already read 18 of 18 on the local 5090 from the integ69 branch's frozen
    binary (DAY63 section 7). This is a proposal under the registration's own rule, for the lead's acceptance.
- **PLACING nan** is the absent `p` arm, not missing lines. The placing rule compares DAY52's diagnostic `p` against
  base, and DAY67 section 1 dropped `p`, registering that the rule then prints `not placed` with its deltas absent.
  The publication split lines are present in the build: 16 `DAY52 SPLIT` lines in the reading.
- **Measurement condition, recorded.** The stopped P2L sitting's orphaned GPU sampler (an `nvidia-smi --query-gpu`
  poller) ran from 16:24Z to 18:13Z, through all of P2L2's sitting, both arms interleaved equally
  (`pro-single-p2l/box-diag-stopped/SAMPLER-NOTE.txt`). It is a light host and driver poll, equal across arms. The
  timing clauses all passed with wide margins, the tightest being (c)'s e2e at +0.34 against +1.0, so no verdict
  turns on it. It is stated beside the reading.
- The stopped diagnostic's mirror is banked as `pro-single-p2l/box-diag-stopped/` (T-H in both arms), not read.
- **Accepted by the lead** (2026-09-26): (a)'s unit step is void and repeats whole on `a2419d3e1`. P2 is ADOPT if that
  step is all green and REVERT otherwise, with (b) to (g) read as they read. It is queued on BOX31 after F's sitting,
  as the first half of `/root/units-chain.sh` (receipts `a-p2l2/unit-rerun`).
