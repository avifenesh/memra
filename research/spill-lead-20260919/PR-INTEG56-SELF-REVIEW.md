# Self-review: integ56 (B day 36, the owner's `MEMRA_ADMIT_BY_MEMORY` decision cell)

Author's review of the full diff `main..lane/spill-integ56-20260924`, posted as a PR comment per the owner rule.

## What the diff is
- Research only:
  - B's STATE.md, the day-36 INDEX row and a one-word DAY35 fix;
  - the integ56 record section (ruling 51, the decision packet) and this file.
- The day-36 receipts and DAY36.md are already on main (integ55).
- `git diff origin/main HEAD` names only files under `research/`.

## What I checked
- **Pre-registration:** `88bb3b1e2` was committed at 22:19:00Z, before the first day-36 boot at 22:39:47Z
  (`rtx5090-day36/order.log`).
- **Verdict lines:** the V-DOOR and SELECT lines in the packet are copied from `rtx5090-day36/SUMMARY.txt` and
  `pro-single-day36/box/SUMMARY.txt`.
- **Truncation and concurrency:** the numbers match DAY36 2.4.
- **R4 and the door:** R4's meaning follows from the survey (`OPEN-OUTPUT-SURVEY.md`) and the door's FLAGS row. The
  surveyed engines bound an omitted `max_tokens` by the remaining context, which is today's door-OFF charge.
- **Caveat:** the uncapped-booking caveat is B's own (DAY36 2.2) and is stated in the packet.
- **Selection:** the lead selects no value. The options are listed, and the decision is the owner's.
- **Hygiene:**
  - check-flags, conflict markers, perf board and `git diff --check` are clean;
  - no provider name, host, id or price;
  - no em dash in authored lines.

## Push regime
Research only, so no battery. A new branch goes up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` because of
the no-upstream range trap (announced, logged). No tag. Revuto: if capped or unavailable, this comment is the review.
