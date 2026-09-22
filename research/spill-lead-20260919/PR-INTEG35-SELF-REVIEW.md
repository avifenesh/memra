# Self-review: integ35 (B day 29: the 5090 twin gate's `V3=FAIL` classified as a broken premise; the gate refuses typed; the door gates on the 5090)

Author's review of the full diff `main..lane/spill-integ35-20260922`, posted as a PR comment per the owner rule.

## What the diff is
- `tools/prefix-newest-turn-fits-gate.py`: the gate parses the server's `[admit-oom] reclaim-on-defer` lines per
  window, compares the two boots' parked-session releases window by window, and refuses with `REFUSED: V3 premise: ...`
  (exit 2) when they differ, naming the windows, the releases per boot, the budget and the card at each boot (driver
  free and compute-apps sampled into the receipts). The verdict line it would have printed is kept in `summary.json`
  as `verdict_under_broken_premise`. V3's clause, form and slack are unchanged; two boots that release identically are
  evaluated exactly as before. I read the regex against the server's format string and the premise rows' zip over
  cohort sends and turns; the refusal is stricter (an undecidable V3 no longer prints PASS or FAIL).
- Research: B DAY29 (the three repro runs, the arithmetic naming the parked session, the red arm with a self-owned
  co-tenant, the door gates on the 5090 in every arm, the target-card twin on the patched gate), receipts from both
  cards, `test-day29.py` (13 ok), STATE, INDEX row; the lead record section with ruling 31; this file; battery receipts.

## What I checked
- No engine change; no `MEMRA_*` read added (census clean); the card sampling is read-only `nvidia-smi`.
- The refusal cannot hide a real V3 failure: it fires only when the releases differ between the boots (the premise),
  and the red arm proves it fires under a controlled co-tenant while the clean-card run reads PASS with day 17's values
  to the byte on both cards.
- The named cause is arithmetic from the server's own lines (`3272 x 29696 + 313187668`), not inferred; the co-tenant
  was another session's process and was never touched.
- Battery: the gate compiles and prints its help, B's CPU test passes, censuses and the boundary scan are clean; no
  smoke (no engine change).

## What I did not do
- No GPU cell of my own; B's receipts are the evidence. The 9B twin's `REFUSED: cohort promotion did not happen` stays
  a gate-shape fact for that artifact, unchanged.
