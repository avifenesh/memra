# Self-review: integ51 (B day 32: the `MEMRA_ADMIT_BY_MEMORY` rerun on the fixed tree, both cards)

Author's review of the full diff `main..lane/spill-integ51-20260923`, posted as a PR comment per the owner rule.

## What the diff is
- Research only. B's DAY32.md, its day-32 compare script and reader self-check, receipts `rtx5090-day32/` and
  `pro-single-day32/box/`, STATE.md, and the INDEX row. The integ51 record section (ruling 46) and this file.
- The branch is B's tip `214f2a8a5`, which already carries main `c3eb41d12`. `git diff origin/main HEAD` names only
  files under `research/`. No flag, no `MEMRA_*` name, no code.

## What I checked
- Pre-registration before any boot: `51c113659` was committed at 06:32:14Z, and the first boot started at 06:33:04Z
  (`rtx5090-day32/order.log`).
- The V-DOOR and SELECT lines in the record are copied from `rtx5090-day32/SUMMARY.txt` and
  `pro-single-day32/box/SUMMARY.txt`.
- V-ALLOC's -8: `O1-on2048` books `ctx=3551` = 1439 + 2048 + 64 (`max(ctx_cap, P + budget + 64)` with #668's
  budget `v`), against the registered 3559 = 1439 + 2048 + 72 (day 31's budget `v + 8`). The FAIL stands as recorded
  and is not re-read.
- memra#680's evidence lines (`O1-on32768/server.log:6703`, `:6742`, `:6785`) say what the issue quotes: 58 in
  flight, a per-request estimate of 4.35 GB, 12 `[admit-mem]` decision lines against 109 cost lines, a 13.3 GB
  reclaim-on-defer, and 46 prefill OOMs.
- Checks: check-flags passed, conflict markers clean, perf board up to date, `git diff --check` clean. No em dash
  in authored lines.

## Push regime
Research only, so no battery. The first push of the new branch was refused `UNQUALIFIED`: its no-upstream range named main's engine files, not this diff. The branch went up with `MEMRA_RELEASE_QUALIFICATION_MODE=development`, announced and logged, and every other hook ran and passed. Revuto: if capped or unavailable, this comment is the review. No tag.
