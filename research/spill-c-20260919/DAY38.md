# Session C day 38: the per-tick split of day 37's summed gaps, and the 9B entry's byte split, RTX 5090 class receipts

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. No card work today: every figure reads receipts already committed
(`rtx5090-day31/`, `rtx5090-day35/`, `rtx5090-day37/`) or the source at a named commit. Every push today in the
announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). No commit on main, no PR, no engine change, no `docs/` registry edit, no recommendation.
Local CPU work under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=20G`.

## 0. Merge (first action)

- `1215dff35`: `origin/main` at `0c86309bd` (#652) merged, `--no-ff`. Clean; the four conflict markers grep to 0
  lines in the merged tree. Pushed.
- `crates/memra-server/src/worker.rs` is identical on `8b889dcdf` (day 37's option (a) tree) and on the lane tip
  (`git diff --stat 8b889dcdf HEAD -- crates/memra-server/src/worker.rs` prints nothing), so every `worker.rs:<line>`
  below without a commit names both.

## 1. Pre-registration of the per-tick split (committed and pushed before the reader ran on the day-37 receipts)

**What this is.** A POST-HOC READER OVER A COMPLETED CELL. Day 37's 40 receipts were read on day 37 under day 37's rule
(the worst tick, and the two largest gaps summed). This reading splits the summed pair by position. It is evidence for
attribution inside day 37's cell, not a new cell, and it is not a qualification. The form of the tick rule below was
written from day 35's banked `rtx5090-day35/gaps-posthoc.log` and day 37's printed verdict lines only; no day-37
per-run gap position was looked at before this commit.

**Owed by.** `DAY37.md` section 3: "The reader does not split the two gaps it sums, so which of the tenant's two
stretched ticks moved is not read here." The packet's item 7 carries the same clause.

**Admissibility.** Day 37's, not recomputed and not re-admitted: `rtx5090-day37/reading.log:214` reads `DAY37
ADMISSIBLE: 40 of 40 receipts; all=True`. The reader refuses to run unless that line is present and all 40 receipts
exist. No `--replay`.

**The reader.** `day38-tick-split.py`, day 37's reader with one change, the quantity. `pct`, `med_iqr`, `classify`,
`outcome`, the programs, blocks, arms and the pooling are day 37's. The diff against `day37-stall-reading.py` is banked
as `day38-cpu/reader.diff` (631 lines, `diff -u`). Its synthetic self-test (`--selftest`, no receipt read) is banked
as `day38-cpu/selftest.log`: `DAY38 SELFTEST: PASS`. The fixtures cover naming by position when the second gap is the
larger, a stretched gap before the fire excluded, a run with one stretched gap and no tick 2, the strict `>` at 3 x
p50, a late fire time, and the four outcomes.

**The ticks, by position, per arm run.**
- f = the receipt's `fire_at` minus 1 = 23: the index in `itl_ms` of the gap from the tenant's 24th token to its 25th,
  the gap in progress when the harness fired the intruder (`day35_stall_cell.py:114`, `fired.set()` at the 24th
  arrival; `:232`, `itl_ms[i]` spans token i+1 to i+2).
- stretched = `itl_ms[i] > 3 * p50`, p50 the run's own median gap (day 35's descriptive term).
- **tick 1** = the first stretched gap at an index >= f, in time order. **tick 2** = the next stretched gap after tick
  1, in time order. A run with fewer than k stretched gaps at or after f has no tick k.
- q=tick1 and q=tick2 are the per-run values of those gaps in ms, raw gaps as day 37's `top1_plus_top2` is raw gaps.

**The rule, day 37's unchanged, applied to q=tick1 and to q=tick2 separately.** Per arm (prime, demote-off, demote-on,
promote-off, promote-on) and per order block (o1 = p2-opta against p1-base, o2 = p3-opta against p4-base):
d = median(option (a) program) minus median(base program), each side the program's two passes pooled (N=20), unc = the
quadrature of the two IQRs, `isolated` when |d| > unc; `moved` when both blocks are isolated with one sign,
`order_split` when both are isolated with opposite signs, `under_resolution` otherwise. The DiD per class per block,
(on - off)_opta minus (on - off)_base, unc = the quadrature of the four IQRs, the same outcomes.

**One property of the quantity, not a change to the rule.** A tick-k quantity is defined for a program's arm only when
every one of its 20 arm runs has tick k. Otherwise the line prints `not_defined (R of N runs lack tick k)`, and the
block, the arm's outcome and any DiD that uses it print `not_defined`. Day 35 read the OFF promote as stretching ONE
tick per hit (`top2_median=9.9 / 9.8`, `DAY35.md` task 3), so promote-off tick 2 may be `not_defined`; that is stated
now so the state cannot be chosen after the split is seen.

**The hypothesis, from the mechanism, before the numbers.**
- Base tree (`091a931c0`). The insert's tick submits the demote's D2H (`worker.rs:12216` on `091a931c0`, `demote
  submitted off the tick`). A later tick top polls the ticket and, on completion, prints `demote published off the
  tick` (`:12438`) and runs `host_demote_publish` (`:12258`), whose `host.bind_tier_image(&mut e)` (`:12265`, the body
  at `:9296`) is the whole-image pass. Day 35 placed the door's cost on the second stretched tick by rank, "the tick
  whose top polls the ticket, hashes and publishes" (+29.6 / +27.5 over OFF's second tick).
- Option (a) tree (`8b889dcdf`). On completion the poll hands the image's heap payloads to the hash helper and keeps
  the entry `Demoting` (`worker.rs:12915`, `demote copy complete off the tick`). The take-back and publish run on a
  later tick after the digests land (`:13148`, `demote digests landed off the tick`), through `host_demote_publish`
  (`:12663`) with the helper's digests (`host.bind_tier_image(&mut e, hashed)`, `:12671`). Day 37's ledger on this
  tree: `copy_settle=8.48` / `8.30` on the poll tick, `take_back_publish=8.36` / `8.28` on the later tick,
  `pre_submit=23.49` / `25.06` on the insert's tick (`DAY37 LEDGER` lines, `rtx5090-day37/reading.log`).
- **H: option (a) moves the bind pass off the poll tick (tick 2) and leaves the insert's tick (tick 1) alone.**
  Predictions, each read from the `DAY38 VERDICT` lines:
  - P1: demote-on q=tick2 `moved` with d < 0 in both blocks.
  - P2: demote-on q=tick1 `under_resolution`.
  - P3: did-demote q=tick2 `moved` with d < 0 in both blocks.
  - P4: did-demote q=tick1 `under_resolution`.
  - P5 (control): demote-off `under_resolution` on both ticks.
- **What refutes H.** demote-on q=tick2 or did-demote q=tick2 not `moved` negative; or demote-on q=tick1 or did-demote
  q=tick1 `moved` or `order_split` (either sign). If P5 fails, the demote-on lines carry the trees' whole `crates/`
  difference, and H is read on P3 and P4 alone (day 37's own reading through the DiD). A `not_defined` primary
  quantity neither supports nor refutes: `H not readable`. The reader prints the `DAY38 HYPOTHESIS` line mechanically
  from the outcomes.
- **The promote class** is read under the same rule with no prediction registered: day 35 places the inline demote's
  bundle pass on "the same or the following tick top", so no tick is predicted for it.

**Descriptive only (printed, not ruled).** Per run: f, p50, the stretched gaps at and after f as (offset from f, ms),
the stretched count before f, tick 1 and tick 2 with their offsets, the run's two largest gaps. Per program per arm:
the stretched count at and after f (median, min, max), the offsets, the runs where tick 2 is the gap right after tick 1,
the runs where {tick 1, tick 2} equals day 37's {top1, top2}, and the fire check (the harness's `fired_at_ms` between the
reconstructed arrivals of the 24th and 25th tokens, `ttft_ms` plus the cumulative gaps, 0.05 ms slack for the gaps'
3-decimal rounding).

**What this reading cannot say.** The RTX 5090 Laptop GPU, the 9B NVFP4 artifact, the plain 64-token class under
`MEMRA_SERVE_SPEC=0`, day 37's receipts only. A third or later stretched gap is printed and not ruled; the option (a)
take-back tick is ruled only if it is tick 1 or tick 2. The ledger segments are not timed against tenant gaps; a
segment is placed on a tick by the code path, not by a timestamp. No figure is compared with day 35's receipts or with
the target card.
