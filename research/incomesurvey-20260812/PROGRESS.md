# cx-incomesurvey progress

- Branch/worktree: `lane/cx-incomesurvey` at
  `/home/avifenesh/projects/wt-cx-incomesurvey`
- Starting revision: `ba3e70c9af455320dc661ab023e5c653539bc447`
- Research/retrieval date: **2026-08-12**
- Scope: docs-only live survey of realistic serving income for one Vast 2x RTX PRO 6000 pair
  serving Step-3.7-Flash.
- Constraints: no code, merge, tag, push, or formatting pass; use current web sources; do not
  extrapolate the measured concurrency curve beyond `c=17`.

## Status

- [x] Verified dedicated worktree/branch, clean starting state, repository instructions, and lane
  inbox.
- [x] Captured live comparable-model prices and the public limits of OpenRouter provider-payout
  disclosure.
- [x] Captured channel economics for Poe, AI Horde, Featherless, Novita, Chutes, and direct API.
- [x] Audited public evidence for new-provider demand and routing/ranking effects.
- [x] Computed gross/net daily income and break-even points at 5/20/50/100% utilization.
- [x] Wrote and source-audited `REPORT.md`; updated this ledger; committed the docs-only result.

## Fixed capacity and cost inputs

- Decode at continuous `c=8`: **158 output tok/s**, or **13.7M output tok/day** ceiling
  (owner-supplied, dual-PP default, box1-validated).
- Decode at `c=10`: **about 210 output tok/s** on the owner-supplied dualpp1 curve; deeper
  concurrency is unmeasured.
- Grouped prefill: **about 639 input tok/s**, or **55M input tok/day** ceiling.
- Vast pair rental: **$2.537/hour = $60.89/day**.

## Evidence notes

- Owner-supplied capacity receipts are treated as fixed inputs, not independently remeasured in
  this docs-only lane.
- Market and channel claims are based on pages and endpoint feeds retrieved on 2026-08-12;
  unavailable or undisclosed economics stay explicitly unpriced rather than being estimated.
- OpenRouter's live endpoint feed superseded one cached promotional rendering during the final
  audit. The indexed Step provider-share observation is explicitly dated/staleness-bounded in the
  report because the live share widget is client-rendered.
- Final arithmetic was independently recomputed from the fixed 55M-input/13.7M-output ceilings.
  All cited URLs resolved during the final audit except Poe's help-center page, which was retrieved
  through the research client but rejects plain command-line requests with HTTP 403.
- Result: the exact Step headline rate grosses $26.755/day at the optimistic full `c=8` ceiling,
  leaving $34.135/day of rent uncovered before any other costs.
