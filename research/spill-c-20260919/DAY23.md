# Session C day 23: the promote stall cell on main's tree (the parked-only wait), the demote arm beside it, and the HOSTPREFIX door review table in its final form for 2026-10-05

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses the tree `UNQUALIFIED` because the
content-bound census sees the engine files main brought in; the hook prints `UNQUALIFIED DEVELOPMENT:
refs/heads/lane/spill-c-20260919 at <sha>; no GPU qualification claimed` and records the skip in the clone's
`.git/memra-gate-skips.log`). Every cell below is `executed-not-qualified` development evidence; no timing here is a
qualification claim. No commit on main, no PR. No engine change of my own today.

## Merge (first action)

`origin/main` `0713c1a79` (#627, integ32: A's day 18, the promote half of Move 1, plus the lead's three revuto
fixes: the probe's cold memo names the refused entry; a tenant purge clears the tenant's memo and releases the
worker's one-tick pin; the run loop waits boundedly on the command channel when the queue holds only requests
parked on the in-flight promote instead of spinning the owner thread through park-and-requeue ticks) merged as
`d2b8efbf2`. One conflict, `crates/memra-server/src/worker.rs`, at the two sites where my day-22 resolution of
A's day 18 against #622 met the lead's resolution of the same pair on main: main's side taken whole (rerere had
offered the day-22 resolution; discarded). After the merge `git diff origin/main -- crates/` is EMPTY: this lane
carries no engine line that main does not. `research/INDEX.md` merged clean with both rows. `tools/check-conflict-
markers.sh` OK before the commit. Pushed in development mode (`d2b8efbf2` = `origin/lane/spill-c-20260919`).

## Task 1, pre-registration (this section is committed before the run)

**Owed.** integ32 (`research/spill-lead-20260919/INTEGRATION-DAY12.md`, "Owed to the door review"): A's day-18
promote stall receipt (`stall_median=81.9`) was taken with the owner-thread spin that revuto round 2 on #627
removed; the wait changes timing, not bytes, so the promote stall cell is owed on the tree that carries the wait.

**Tree and rig.** Main's tree as merged, `d2b8efbf2` plus the day-23 drivers (the tree SHA is in the cell's
`CELL.txt`); `crates/` equal to `origin/main` `0713c1a79`. One RTX PRO 6000 Blackwell at its 600 W limit through
the canonical collector (`tools/tier-battery.py --rig pro-single --timeout 3600 --external-lock`, lock
`/tmp/memra-gpu.lock`, 250 ms telemetry, `CELL.jsonl` `executed-not-qualified`), the Qwen3.8-27B NVFP4-Q5K MTP
artifact, `MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192`
in both boots, `MEMRA_KV_HOST_CONTRACTS=1` in the ON boot only. Lane A day 19 may hold the box; the runner
retries the lock 75 x 120 s and never inspects or signals the holder.

**Driver and harness.** `day23-stall-cell.sh` is lane A's `pro-single-day18/stall-cell.sh` (itself the day-16
script with the receipts root moved) with three path changes and one shape change, stated: the worktree, the
receipts root and the server binary are arguments instead of `/root/wt-a` and `/root/spill-receipts/a-day18`;
the tree SHA, the harness digest and the compute apps before and after are recorded; and the two boots (door OFF,
then door ON) run inside ONE collector lock hold, where A held the lock once per boot, so OFF against ON is a
same-window pair on one tree today. Each boot runs the demote arm then the promote arm, as A's did. The harness
is `research/spill-a-20260919/stall_cell.py` as it sits on this tree after the merge, byte-identical to lane A's
tip `c96d5186`: the tenant streams a 20-token prompt for 160 tokens, the intruder fires at the tenant's 24th
token, order 1 (idle, arm) x 5 then order 2 (arm, idle) x 5, N=5 per arm per order, N=10 pooled, the rule line
fixed in the harness and recomputed by `--replay`. `day23-box-run.sh` waits for the build receipt and runs the
cell once through the collector. `day23-stall-reading.py` holds the thresholds below as constants and prints one
`DAY23 STALL READING` line per question after replaying every rule line through the harness; it is committed
before the run and not edited after it.

**Claim (integ32's, read as a rule).** The parked-only wait removed the owner-thread spin, so on this tree:
(a) the tenant's stall median for a promote under the door stays at or below day 18's 81.9 ms; (b) the promote's
own `server_promote_ms` (submission to publication) does not grow beyond the day-18 figure by more than the
tick's 2 ms bounded wait. References are same-box, other-sitting figures: A day 16 OFF promote `85.0`, OFF demote
`117.5`, ON promote `162.8`, ON demote `193.5`; A day 18 run 2 ON promote `81.9` with `server_promote_ms` first pair
`60.8, 61.9` and steady `25.9` to `26.4` (median `26.1`), ON demote `149.7` (day 17: `149.6`).

Rules, fixed before the run (stall medians in ms, the harness's `stall_median`):
- P1, promote ON against day 18: `at_or_under_day18` if `stall_median <= 81.9`; `within_wait` if `81.9 <
  stall_median <= 83.9`; `grew` above.
- P2, the promote's own window ON: `within_wait` if the median of `server_promote_ms` over runs 3 to 10 is
  `<= 28.1` (26.1 plus 2.0), else `grew`; the first pair's max against `63.9` recorded the same way
  (`first_pair_within_wait` / `first_pair_grew`).
- P3, promote ON against OFF in the same window: `on_at_off` if `on <= 1.10 x off`, else `on_above_off`.
- P4, promote OFF against day 16's 85.0: `off_stable` inside `76.5..93.5`, else `off_moved` (a regime reading of
  the OFF program, which the door does not touch).
- P5, demote ON against day 18's 149.7: `demote_half_unchanged` inside `134.7..164.7`, else `demote_half_moved`
  (the wait fires only with a `Promoting` entry, so the demote arm should not move).
- P6, demote OFF against day 16's 117.5: `off_stable` inside `105.8..129.3`, else `off_moved`.
- Admissibility (A's day-18 list, applied to the ON promote arm): `STALL REPLAY: PASS` for all four receipts,
  `errors=0`, `tenant_text_identical=True` in every receipt; in the ON boot's server log ten `promote submitted
  off the tick`, ten `promote published off the tick`, ten `contracts door H2D receipt` lines over the promote
  arm's ten intruders, every intruder `cached_tokens=64`, no `demote failed`, `promote failed`, `promote refused`
  or `TIER DISABLED` line, and zero `settled synchronously by a promote` lines. An inadmissible run is reported as
  such and not read against P1 to P6.
- Recorded, no rule: the `poll(s)` count and the submission-to-completion ms in each `promote published off the
  tick` line (day 18 read `1 poll(s), 19.5` to `19.7ms` in all ten).

What this cell is and is not: one card class, one host, one artifact, `MEMRA_SERVE_SPEC=0`, 64-token entries
(160 MB), N=5 per arm per order in both orders, one sitting; P3 is a same-window pair, P1, P2, P4, P5 and P6 are
same-box cross-sitting readings against A's receipts, not same-window A/Bs; no number is divided into a number
from another box. Nothing here answers the door's decision question.
