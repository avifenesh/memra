# The MTP route verifies a sampled round by position (2026-11-10)

**Status:** adopted for the MTP speculative route, with plain decode's first sampled token moved
to the same rule. A seeded sampled request writes a different text once at the release that
carries this, and from then on the same text at every draft length, alone or beside another
request, cold or resumed, with or without a demotion to plain decode. The DFlash2, GLM5, DSpark
and gemma assistant-drafter routes keep the rejection walk until their own change. Issue
memra#1154. Receipts:
[`research/position-keyed-sampling-20261110/`](../../research/position-keyed-sampling-20261110/).

## Question

A sampled speculative round used Leviathan/Chen modified rejection sampling: accept draft token
x with probability min(1, p(x)/q(x)), else draw from the residual norm(max(0, p - q)). The output
is distributed exactly as the target, and no rule accepts more often (the acceptance is 1 - TV(p,
q), the maximum for any coupling). But which random numbers a token consumes depends on how many
drafts the rounds before it took and kept, so the text a seeded request writes depends on the
draft length K, on a companion that changes the round boundaries, and on the cost model's choice
between depths. memra#1154 measured it on Qwen3.8-27B: one seeded prompt wrote four texts at
K=0, 1, 2, 3, and a pair wrote other texts than its two requests alone. Plain decode had the
property the speculative route lacked: token p is `argmax(filter(logits_p)/T + Gumbel(seed, p))`,
one Philox event per token, keyed by the token index.

Can the speculative route emit exactly that draw, and what does it cost?

## The rule chosen

For generated position p the route emits the plain sampler's own draw from the target's row at
p: the same filter, penalty and Philox kernels, the event keyed by p.

- The draft at position p draws from its own row with the same event (it already did). The two
  draws are Gumbel-coupled: they agree whenever the same token wins both perturbed rows.
- The coupling needs a token to get the same noise in both rows. The Gumbel kernels key the
  noise by row index, which is the token id on a full-vocabulary row. A trimmed draft head (its
  rows are a subset of the vocabulary, row i standing for token `d2t[i]`) therefore passes its
  map to the draft-side kernels, which key row i by `d2t[i]`. Without the map the two draws are
  close to independent and acceptance falls while the text stays right: on Qwen3.5-9B with its
  own-trim drafter, acceptance per drafted token at K=1, 2, 3 is 0.655, 0.521, 0.383 on main,
  0.510, 0.362, 0.260 without the map and 0.647, 0.472, 0.356 with it; on Qwen3.8-27B with its
  trimmed FR-Spec drafter 0.687, 0.510, 0.417, then 0.549, 0.414, 0.313, then 0.667, 0.520,
  0.406 (RTX 5090, eight seeded prompts per K, `pr/rtx5090/trim-*`).
- A draft is accepted when it equals the target's draw at its position. The first disagreement
  emits the target's draw and ends the round; a fully accepted round emits the draw of the row
  after the last draft.
- No accept uniform and no residual draw. The emitted stream is a function of the target rows,
  the seed and the position, so it cannot depend on K or on where rounds fall.
- A penalised request penalises each verify row over the window plain decode holds at that
  position (the round-entry window plus the drafts before the row).
- All K+1 draws of a round are issued before anything is read back: one K+1-token readback.
- Plain decode's first token was the one draw off this rule: the host sampler drew it from its
  own stateful RNG and event 0 went unused, while a speculative burst's boundary token is the
  event-0 draw. It is now the event-0 device draw for every request whose later tokens are
  device draws: penalised ones, and grammar-constrained ones (from the masked row). A failed
  device draw fails the request; it does not fall back to the host RNG.
- A new request's first generated token is event 0 whatever was restored. A sampled request
  whose prompt is exactly a parked session's history used to take that session's stashed
  token, drawn by another request; it now primes cold. A recurrent trunk already rewound to a
  checkpoint and primed (memra#1093).
- The demotion from speculation to plain decode needs no counter handoff any more: the session
  counter equals the number of emitted tokens, so plain decode continues at the next position.

The output distribution is the target's by construction: each emitted token is a plain sample of
the target row at its position.

## What was rejected

- **Keeping the rejection walk.** It accepts at the maximum rate and it is why one seeded
  request wrote several texts. A text that depends on scheduling cannot be reproduced by a
  client, compared across a default change, or used as a regression reference.
- **Keying the rejection walk's random numbers by position.** The accept uniform can be keyed by
  position, but the residual draw is not the plain draw at that position, so plain decode and
  speculation would still write different texts, and a demotion would still switch streams.
- **Leaving plain decode's first token on the host RNG.** Then K=0 differs from K>=1 from the
  first token (three of six prompts in the cell).

## Measurement

Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head, context 8,192, two sessions, thinking off; an
RTX 5090 Laptop GPU and GB10. Receipts and method:
[`research/position-keyed-sampling-20261110/`](../../research/position-keyed-sampling-20261110/).

- **The text.** `tools/seeded-invariance-gate.py` on the RTX 5090: on main, 28 of 44 rows leave
  the plain reference (every K writes another text; a demoted request another one again); with
  this change every prompt writes one text over plain decode, K=1, 2, 3, a pair, a resumed turn,
  a request sent after the one that parked exactly its prompt, and a forced demotion (54 rows).
  GB10: the same, for every request plain decode draws on the device.
- **Acceptance per drafted token**, main then this change: 0.455 and 0.446 on the RTX 5090 and
  0.475 and 0.446 on GB10 over six sampled prompts at K=3, and 0.512 and 0.525 on the RTX 5090
  at the landed form, where main serves long prompts at K=2; 0.438 and 0.442 at temperature 1 and
  0.467 and 0.450 at temperature 0.7 with top_p 0.95 over 16 prompts (paired differences +0.003
  and -0.018, each +- 0.016); 0.729 and 0.708 at K=1, 0.587 and 0.563 at K=2, 0.458 and 0.443 at
  K=3 over the invariance cell's nine runs per K.
- **Decode rate**, twelve boots per device, candidate over main on per-boot medians: on GB10
  0.972 over the six sampled prompts (per prompt 0.911 to 1.010) with the greedy control at
  1.000; on the RTX 5090 0.999 (0.897 to 1.084) with the control at 0.987 in the prototype's cell,
  whose first boot, a main boot, ran unthrottled, and 1.039 (0.960 to 1.101) with the control
  at 1.009 at the landed form. A sampled round is 0.2 to 1.6 ms shorter on GB10
  (109.1 to 107.5 ms at 1.2k tokens): no residual draw. The whole decode difference is
  acceptance.
- **Greedy** writes the same texts with the same acceptance on both devices.

## Cost and what follows

The rejection walk accepts with probability 1 - TV(p, q), the maximum; the Gumbel coupling is
bounded below by (1 - TV) / (1 + TV). Measured, the coupling costs about 2 points of
acceptance per drafted token on average over eleven estimates (from 1 point better to 6 worse by
cell; the gate's eight 240-token prompts are the high end, 3.5 to 5.8). In the 12-boot decode
cells that is 0.972 of main's sampled decode on GB10 and 0.999 and 1.039 on the RTX 5090, and
nothing for greedy. A trimmed draft head pays no more than an untrimmed one. That buys a
text a client can reproduce and a regression reference that survives a change of K, of the cost
model's choice, of the batch and of the resume point.

Not covered, each named where a reader will look (`docs/SERVING.md`, the `MEMRA_SPEC_TEMP` row of
`docs/FLAGS.md`):

- The DFlash2, GLM5, DSpark and gemma assistant-drafter routes: still the rejection walk.
- A penalised sampled request where the device penalty arm is off (`MEMRA_SERVE_DEVPENALTY`:
  every model and card pair except the dense Qwen35 class on the RTX 5090 and RTX PRO 6000
  classes): plain decode draws it on the host, so its speculative text is one text across K but
  not plain decode's.
- `MEMRA_SPEC_SAMPLED_BOUNDARY=0` (a rollback seam) puts an argmax at each burst boundary and
  breaks the position key.

Follow-up, done: the draft side kept producing the old accept test's inputs (its retained head
logits and the filter statistics it read back every slot, up to nine scalar readbacks and three
launches a round at K=3). memra#1169 removed them, with the two pooled-draft probe fields that
compared them: `research/spec-draft-accept-leftovers-20261110/`.
