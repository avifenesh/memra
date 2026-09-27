# Self-review: integ48 (C day 39: the target card's demote-class tenant-stall cell on the option (a) code)

Author's review of the full diff `main..lane/spill-integ48-20260923`, posted as a PR comment per the owner rule.

## What the diff is
- Research only. `research/spill-c-20260919/`: DAY39.md, the build script and three readers (stall, tick split,
  regime), the target-card receipts under `pro-single-day39/` (builds, collector at 250 ms, per-boot server logs and
  receipts), `day39-cpu/` (ancestry and the regime parse fix diff), and day-39 updates to DOOR-DECISION-PACKET.md,
  HOSTPREFIX-DOOR.md and STATE.md.
- `research/INDEX.md`: C's day-39 row. `research/spill-lead-20260919/`: the integ48 record section (ruling 43) and this
  file.
- No `crates/` or `tools/` change: `git diff origin/main HEAD -- crates tools` is empty. No flag, no `MEMRA_*` name.

## What I checked
- Pre-registration: `89c2cd521` committed 21:57:26Z, first boot mark 21:58:02Z (`marks.tsv`). The harness SHA-256
  equals the registered `9c9b3878...0ccc`.
- Binaries: `box/builds.log` trees equal section 0's SHAs. `git diff 9717e8d57 160929a92 -- crates` is exactly the
  files of `97a9e091f` and `fc637d26a` (7 files, +1448/-33), so the b2-b1 contrast is A day 30 alone.
- Readers: both rerun on the banked receipts; the stall reader matches `reading.log` below its command header and the
  tick reader matches `tick-split.log` except the reading-log path it echoes. The one post-run fix is a timestamp
  parse in the regime reader, which decides nothing.
- Scope in the record: one card, the 27B, the plain 64-token class, `MEMRA_SERVE_SPEC=0`; no cross-box or cross-card
  comparison; `executed-not-qualified`. The lead record states the cell's direction and makes no default change.
- Checks on `fa02b6559`: `check-flags` exit 0, conflict markers exit 0, `update-perf-board.py --check` exit 0,
  `public-boundary: 604 matches (604 grandfathered, 0 new).`, `git diff --check` clean, no em dash in added lines,
  `.gitattributes` in both new receipt dirs.

## Push regime
Records only, so no GPU battery. The release qualification gate refused the first push of the new branch as
`UNQUALIFIED` (its range for a branch with no upstream named main's engine files, not this diff's), so the branch went
up with `MEMRA_RELEASE_QUALIFICATION_MODE=development` (announced, logged); every other hook ran and passed. No tag (docs and receipts). Revuto: if capped
or unavailable, this comment is the review.
