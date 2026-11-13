# DeepSeek-V4.1 route: a call flag names its rows (memra#1292, 2026-11-13)

Verdict: PENDING (GB10 serve smoke and kernel gate).

Since #1281 the route stays alive after a request-caused numeric fault, but the call flag only said
that some router row of the step named no expert, so the route failed every request of that step.
Now each call keeps a row map, the plan marks the faulted rows in it, and the route fails only the
requests those rows belong to; the step's other requests, the other rows of the same batched call
included, are served.

## What changed

- **The plan marks rows.** `memra_v41_moe_plan` takes a row map (`rows_bad`), the slice's first row in
  its call (`row0`) and the map's capacity. Each pair whose expert id is outside `[0, ne)` still counts
  in `err`, is skipped and sets bit 0 of the rank's call word, and now also sets bit `row0 + t` of the
  map for its row t (the one-block plan and the three-kernel plan alike; the count kernel's thread is
  the row's).
- **Each call owns a map.** `Runner::forward` gives each call `ceil(s / 32)` zeroed words on every rank,
  `Runner::forward_batch` one map for the batched call; `Group::own_faults` points every rank's plans at
  the call's map around the call's blocks and back after them (`Ctx::fault_rows`). The map comes back
  in `Out::faults`. A plan outside every call (a draft-head stage, a gate's block) marks the rank's own
  word of plans no call owns (word 2 of `Ctx::route_bad`), cleared with the call word when a step opens.
- **The step names its rows.** On a group whose call flag is `attributed`, the reads of a call's logits
  no longer fail on the call word (a transport flag still loses the group). After a step's last read
  the route asks `Runner::step_faults` for each call's faulted rows: no call word, nothing faulted (no
  map is read); the call word with the unowned word, or with no map naming a row, fails the step as
  before; otherwise the rows. The words read with the step's last logits read are reused
  (`Group::seen`, cleared by every forward, batched step and draft), so a clean step adds no read.
- **The route fails those requests alone** (`faulted_requests`): a call of its own is its request's,
  row r of a batched call is the request at that row. Each such request gets the route's error, is
  booked as a request fault and leaves the step; the rest choose, commit and stream as usual.
- **The off-card MoE no longer fails the forward on a skipped pair.** With the experts behind the
  device cache (one or N ranks, the plan read back), a placed-pair count below the rows' top-k returned
  "route: X of Y pairs placed", so a request's non-finite router row failed every request of the step
  before any flag was read. The placed pairs are complete and the ranks' ranges, gathers and combine
  cover exactly them (the combine skips order -1), as the on-card and split paths already assumed; only
  a count above the pairs (a plan bug) still refuses.
- **The decode graph door.** A captured span records the plan's arguments, and a call's map moves from
  call to call, so the map is one more input of the FFN half's span (`ffn_spans`): a replay points the
  plan at the map of the call it runs for. The door stays off on the served layout (`Group::can_capture`).
- **The fault door** `MEMRA_DSV41_CALL_FAULT` takes `step:request` beside `step`: `step:i` starts the
  calls of the running request at index i in that step from NaN streams (`Runner::poison`), so its
  router rows name no expert in every layer, as a request's data can make them, through the real
  kernels; `step` alone keeps the whole-step injection (a call word no map names).

## Which faults fail what

| fault | named by | fails |
|---|---|---|
| a trunk router row naming no expert in a prefill, a verification, a one-token call or a batched row | the call's row map | that request alone |
| a router row naming no expert in a draft-head stage | no call (the unowned word) | the step's requests |
| the door's whole-step injection | no map | the step's requests |
| a draft's non-finite logits, a refused allocation, any other error of a call | the call's error, which carries no request | the step's requests (unchanged) |
| a non-finite logit behind a clean router | the row's own choice (`choose`) | that request alone (unchanged) |
| a lost transport, a failed probe after a step fault, a caught panic | | the route stops (unchanged) |

Draft-head faults stay whole-step: a draft's router fault marks the unowned word, or, when its draft
logits come out non-finite, fails the draft call itself, and neither says which request's data caused
it (the draft's error is not typed). A draft head reads its own request's trunk taps, and a request
whose trunk rows went non-finite has already failed alone in the step that computed them, before its
next draft.

## Why the other rows keep their bytes

A call of one sequence runs alone through every block. A batched call's rows meet only in kernels that
compute each row on its own: the norms and mHC mixes, the GEMMs (activations quantized in 32-wide
blocks of their own row), attention over each row's own sequence keys, the router, and the plan, which
places only in-range pairs (a non-finite router row names no expert at all, so the row enters no expert
tile) and leaves the faulted row's order slots -1 for the combine. That is the batched-equals-solo
property `dsv41-spec batch-rows` gates on finite rows; the smoke below checks it with a NaN row beside a
live one.

## Gates

PENDING.
