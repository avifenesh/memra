# Hy4-flash: the draft gate, greedy drafts and the NVFP4 drafter (2026-11-22)

**Status:** decided with step 11 of the Hy4-flash onboarding. The serve route prices every DFly2
round from the drafter's own probabilities and the route's measured costs, and verifies the
depth, 0 to 5, with the most expected tokens per millisecond (`MEMRA_HY4F_DRAFT_GATE`, on;
`0` is the rollback seam). Drafts stay greedy for every request. The artifact's NVFP4 drafter
stays. Scope: Hy4-flash on one GB10. Supersedes the fixed five drafts a round of
[HY4F-DFLY2-DRAFTS.md](HY4F-DFLY2-DRAFTS.md).

## Question

At five drafts a round, prose keeps about one draft in four, and a round costs nearly two
one-token calls. Which depth pays, can a gate find it per round, should sampled requests draft by
sampling, and what does the drafter's NVFP4 quantization cost in acceptance?

## Measurement

All on one GB10, sampled at the artifact's `generation_config.json` (temperature 0.9, seed
20261121), the six chat turns and the `long` probe of the lane
(research/modelplan-onboarding-hy4flash-20261119/, step 11, `dfly/receipts/s11/`).

- A teacher-forced replay (`hy4f-spec replay`) drafts a five-draft round at every anchor of the
  served answer. Walking the serve loop over it at five drafts gives main's served rounds, drafts
  and kept drafts exactly on all seven requests, so policies are priced offline against measured
  step costs (`hy4f-spec cost`: 185 ms for a one-token call, 364 ms for a five-draft round).
- Drafter: NVFP4 keeps 1.985 drafts a five-draft round over 991 anchors, the BF16 source drafter
  1.992, BF16 MLPs alone 1.986. Served at five drafts the six turns take 353 and 350 rounds.
  The BF16 drafter costs 3.7 ms more a round and 0.9 GiB more memory.
- Sampled drafts: drawing each draft with the request's own position-keyed draw from the
  drafter's row keeps fewer drafts than the greedy draft on six of seven requests; served at
  five drafts, the six turns take 123.1 s against 112.5 s with greedy drafts.
- Depth: replayed at the measured costs, five drafts read 7.29 tok/s over the decode steps, a
  fixed 3 reads 8.28, the gate 8.58 (17.7% over five drafts), the hindsight bound 10.83.
@@SERVED_DECISION@@

## Decision

The gate is on by default. Each draft step adds a draft whose survival is the product of the
drafter's softmax probabilities of the drafts so far (at the request's temperature, 1 for
greedy), scaled by the request's running ratio of held drafts to those probabilities. The round
stops drafting once one more surely-held draft could not beat the best depth so far, and
verifies that depth. Prices come from the route's own calls (decode calls by rows, draft rounds
by drafts), seeded by a timed warm-up round; each request updates its own copy. A round priced at
no draft ends drafting for a stretch of 8 one-token steps, doubling to 64. The tokens are the
undrafted route's either way: a draft is kept only when it equals the route's own choice.

## Rejected

- Five drafts every round (the step 10 default): the slowest drafting policy replayed.
- A fixed depth of 3: 3.6% under the gate replayed, and the best fixed depth moves with the text
  (code wants 4) and with the device's costs.
- The deepseek-v4.1 route's `DraftGate` on running gains alone, and per-depth running survivals
  from the request's own rounds (step 11's first gate): the gain gate stops prose on its first
  bad rounds (6.76 tok/s replayed), the survivals read 6.89 to 8.31, and served the first gate was
  slower than five drafts on four of six turns.
- The drafter's raw probabilities without the request's ratio: they run high between 0.4 and 0.9
  (drafts the drafter gave 0.6 to 0.8 held 0.46 of the time) and price no better than a fixed 3.
- Drafts drawn with the request's own sampler (the shared uniform): fewer kept drafts than greedy;
  removed from the serve route. At half the temperature it is level with greedy, so not taken.
- The BF16 source drafter: 0.4% more kept drafts, inside the per-request spread, for 3.7 ms more a
  round and 0.9 GiB. The artifact quantizes the drafter with the target's tensor classes.
