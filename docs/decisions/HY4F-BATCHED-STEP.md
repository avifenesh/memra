# Hy4-flash: one call a step across requests (2026-11-22)

**Status:** decided with step 12 of the Hy4-flash onboarding. The serve route runs each step as
one call of every request's rows (`Route::call_batch`) and one batched draft
(`Route::draft_batch`); the draft gate prices a round with the step's other requests in view.
Scope: Hy4-flash on one GB10. Amends the step-by-step program of step 9 and the prices of
[HY4F-DRAFT-GATE.md](HY4F-DRAFT-GATE.md).

## Question

Up to step 11 a step was one call a request, each reading every weight, so `c` requests took
`c` times the one-request step. Can several requests' rows share one call without any request's
tokens moving, and what does it buy?

## Measurement

All on one GB10, sampled at the artifact's `generation_config.json` (temperature 0.9), the
`long` probe (2,104-token prompt, 64 tokens) and the six chat turns, `c` requests at once with
seeds `20261121 + i`, three samples a cell after a warm-up, the binaries interleaved
(research/modelplan-onboarding-hy4flash-20261119/, step 12, `serve/receipts/s12/`).

- Drafts off, decode-phase aggregate tokens/s at 1, 2, 4 and 8 requests: `long` 15.87, 15.88,
  15.89, 15.89 before and 15.87, 24.18, 30.34, 34.61 after; chat 16.89 to 16.93 before and 16.86,
  26.24, 32.37, 36.32 after. The step at four requests on `long`: 251 ms before, 129 after.
- With the draft gate (the default): `long` 18.87, 17.46, 17.67, 18.06 before and 23.06, 24.24,
  26.61, 34.10 after; chat 18.26, 23.46, 21.84, 22.04 before and 21.05, 28.16, 31.10, 42.13 after.
  One request gains because a verify call's rows are cheaper (the experts' small-row tiles and the
  F32 islands past four rows: a six-row verify 172.7 to 178.5 ms before, 163.0 after).
- Identity: `hy4f-spec batch` (4 parts, and 8 parts all drafting) gives every request's ids,
  drafts and logits rows alone and batched bit-equal; `check3` 3 of 3; the served-answer gates
  pass; every served sampled request equals itself alone, and each request alone generates the
  same text on every binary and draft setting of the lane.

## Decision

Each step is one call: every generating request's decode token or verify rows, and the oldest
prefilling request's next chunk, which gives up the rows the decode parts take (decode rows past
the chunk hold the prefill and run as further calls). What belongs to a session (its cache rows,
the indexer's selection over its own keys, its attention, its Engram history, its RoPE rows, its
drafter recall lists) runs part by part; everything else runs once over the call's rows. The
step's draft rounds run as one batched draft. The gate prices a verify of `n` rows in a step of
`b` requests at a `b`-th of the route's measured step of `b * n` rows, and a round drafted beside
others at its share of the batched draft; one request keeps its own prices. The routed experts
run on the small-row tiles up to 16 tokens and the F32 islands up to 16 rows four a launch.

## Rejected

- One call a request (step 9 to 11): `c` requests share the one-request rate.
- Prefill chunks as calls of their own beside the decode call: every step then pays the chunk's
  call and the decode call; inside one call the decode rows add almost nothing to the chunk's.
- The gate at its solo prices in a batched step (the first batched build): it priced each verified
  row at a one-request call's average and drafted rounds one request at a time, and read below
  drafts off at two to eight requests (long, four requests: 25.5 against 30.3 tok/s). The final
  prices still trail drafts off at four requests (26.6 against 30.3 on `long`); the gate stays the
  default because it leads in five cells of eight and leads the step-11 gate in all eight.
- Shares of the step's call by rows alone (the second build): a share prices a row at the call's
  average, not at what it adds, and the step table left out the choices and rollbacks.

## Consequences

- A failed call fails every request it carried (their rows may all be written); a refusal before
  any write leaves the sessions as they were.
- `hy4f-spec batch` is the batched-against-alone gate; any change to the route's call path runs it.
- What is left in a step is mostly the routed experts (four tokens read about 60 of 512 experts a
  layer) and the per-session launches (an attention, merge and cache copy a part a layer).
