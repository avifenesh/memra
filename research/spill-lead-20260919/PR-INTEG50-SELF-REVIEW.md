# Self-review: integ50 (B day 31: `MEMRA_ADMIT_BY_MEMORY` OFF against ON with the open-output sweep on both cards, the survey, D4)

Author's review of the full diff `main..lane/spill-integ50-20260923`, posted as a PR comment per the owner rule.

## What the diff is
- Research: B's DAY31.md, DAY31-D4.md, OPEN-OUTPUT-SURVEY.md, STATE.md, the day-31 client, order, parse, compare
  and faults scripts, receipts `rtx5090-day31/`, `rtx5090-day31-d4/` and `pro-single-day31/box/`, and the INDEX
  row. The integ50 record section (ruling 45), the CPU battery and this file.
- `docs/FLAGS.md`: the `MEMRA_ADMIT_BY_MEMORY` row's decide-by moves to 2026-10-07, citing the owner's 2026-09-23
  call. The merge keeps #668's budget sentence on the same row.
- `crates/memra-server/src/admit_memory.rs` and `lib.rs`: three doc comments only (B's owed item 3). The
  `DEFAULT_OPEN_OUTPUT_TOKENS` doc no longer claims 8192 is what the registries pin. The two request structs'
  `max_tokens` comments now state the door's bound and no longer call the context bound the OpenAI default.
  `git diff origin/main HEAD -- crates` is those comment lines and nothing else.

## What I checked
- Pre-registration before any boot: `ee53f7117` committed 23:09:14Z, first boot 23:09:33Z (`order.log`). B changed
  no arm, value, order or reader after a result. The three reader gaps are named in DAY31 2.2 and left as ruled.
- The verdict lines in the record are copied from `rtx5090-day31/SUMMARY.txt`, `pro-single-day31/box/SUMMARY.txt`
  and `rtx5090-day31-d4/SUMMARY.txt`.
- The record does not read the two V-DOOR PASS lines as a clean door. The ON arms that crashed ran the pre-fix
  program, and #668 (merged) is the fix. Ruling 45 hands the decision to B day 32's rerun on the fixed tree.
- The first-panic receipts B names exist at the lines given, and they match memra#659's evidence.
- FLAGS conflict: both sides' insertions are present, and the base text is otherwise unchanged (checked by
  removing #668's sentence and comparing to the base).
- Checks: the CPU battery on `584cbe2cf` read 14 of 15 rc=0. `git diff --check` flagged trailing whitespace in
  B's verbatim `FAULTS.txt` quotes, now marked `-whitespace` like `SUMMARY.txt`, and the check is clean on the
  head. No GPU battery, because no binary program changes.
- No em dash in authored lines. `.gitattributes` in the new receipt dir.

## Push regime
The branch touches `crates/memra-server/src` (comments), so it went up with
`MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged). Every other hook ran. No tag: docs, receipts
and comments. Revuto: if capped or unavailable, this comment is the review.
