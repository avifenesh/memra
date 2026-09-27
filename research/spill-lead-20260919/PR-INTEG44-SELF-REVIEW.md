# Self-review: integ44 (C day 35: the 5090 tenant-stall cell; A day 28 recorded and held on the lane)

Author's review of the full diff `main..lane/spill-integ44-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- Research only: C DAY35 with the 5090 tenant-stall cell (five arms, two passes, one hold, the harness copy and its diff
  banked, receipts with `.gitattributes`), the per-card cost table B-5090 in `HOSTPREFIX-DOOR.md`, the packet's 5090
  table and item 7, INDEX row, STATE; the lead record section (integ44: C day 35, A day 28's results and ruling 40).
- No engine change. A day 28's option (a) code (`45f824a75` on `lane/spill-a-20260919`) is NOT in this PR: its clause 2
  (every gate green in both arms) is red on both cards for one cause (a hit inside the `Hashing` window is a miss), and
  the code stays on the lane until ruling 40's 2a lands (A day 29, running).

## What I checked
- The engine tree equals main's (`git diff --stat origin/main HEAD -- . ':!research'` is empty); the battery runs on it
  for the receipt's sake and its results are main's.
- C's cell is reported as measured: `admissible=False` is the pass-1 prime arm alone (a co-tenant during pass 1 made the
  admission refuse 7 of 10 prime intruders; pass 2 is clean; both passes give the same classes, under resolution in
  both directions); the attribution reading on this card is labelled post hoc; nothing tuned.
- A day 28's clause results are quoted verbatim, the red arms with their one cause and A's own "not the serving path"
  statement adopted as the integration decision; ruling 40 names the fix and its acceptance.
- The two duplicated table lines in the door doc are header rows (the new B-5090 table reuses section B's header), not
  data rows.

## Push regime
No engine source in the range; pushed with `MEMRA_RELEASE_QUALIFICATION_MODE=development` because the hook judges a new
branch's range from the last qualified pointer. No GPU qualification claimed. Revuto: if capped or unavailable, this
comment is the review.
