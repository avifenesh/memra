# Self-review: integ46 (C days 36 and 37: the door packet re-read with option (a) landed; the 5090 demote-class tenant-stall cell on the option (a) code)

Author's review of the full diff `main..lane/spill-integ46-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `research/spill-c-20260919/`: DAY36 and DAY37, STATE, the door packet (`DOOR-DECISION-PACKET.md`: header, section 2,
  sections 3 and 4 rows, item 2, item 7, section 6, appendix A days 36 and 37), `HOSTPREFIX-DOOR.md` section E days 36
  and 37, the day-36 census script and outputs (`day36-cpu/`), the day-37 cell, runner, reader and regime scripts with
  their diffs against day 35, and the day-37 receipts (`rtx5090-day37/`: 24 boots, 40 receipts, server logs, driver log).
- `research/INDEX.md`: the day-36 and day-37 rows.
- The lead record's integ46 section, this file, the checks receipt.
- No engine source, no `docs/` registry, no flag, no tool change.

## What I checked
- Pre-registration before the run: `a50922b27` (the rule, the primary lines and the secondary quantity
  `top1_plus_top2`) is committed at 19:46:56Z and the hold's first boot is 19:47:19Z. The reading applies that rule
  unchanged; `moved` needs both order blocks isolated with one sign.
- Same-window comparison: both binaries ran inside one lock hold, interleaved in both orders; nothing is compared across
  cards or across days as a result.
- Controls: the OFF arms and the prime arm read `under_resolution` on both quantities, so the ON arms' move on the summed
  gaps is read through the DiD, which also moved.
- Admissibility: 40 of 40 `REPLAY: PASS`; the per-run landing rule is day 35's.
- Scope stated: one card class, one model (the 9B, plain class under `MEMRA_SERVE_SPEC=0`); the target card's demote class
  on the option (a) tree is named unmeasured in item 2, item 7 and section 6.
- Counting rule: spot-checked the day-37 verdict lines against `rtx5090-day37/reading.log` (verbatim) and the day-36
  pre-submit medians against `day36-cpu/`.
- No recommendation anywhere in the packet; no host, id, location or cost (`public-boundary: 604 matches (604
  grandfathered, 0 new).`).
- Checks on `dbb67ac12`: check-flags, conflict markers, `git diff --check origin/main HEAD`, perf board `--check`: rc=0;
  zero em dashes in the added Markdown lines (the raw server logs carry the engine's own `[gpu-watch] Xid source:` line
  as written).

## Push regime
Docs and receipts only: no engine source in the range, no GPU battery, no release tag. Revuto: if capped or unavailable,
this comment is the review.
