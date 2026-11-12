# A plain session behind speculative bursts advances once per round (2026-11-12)

**Status:** adopted, memra#1244. Receipts:
[`research/spec-plain-round-yield-20261112/`](../../research/spec-plain-round-yield-20261112/).
Builds on memra#1227 ([`research/spec-cohort-newcomer-20261112/`](../../research/spec-cohort-newcomer-20261112/)),
which found it. Left open: memra#1246 (the DFlash2, glm5 and gemma bursts) and memra#1253 (a
long prime's chunks between full bursts).

## Question

The worker tick runs every speculative burst to its target in phase (a), then primes plain
prompts in phase (b), then steps every plain decoding session once in phase (c). A plain
session that could not ride a pooled cohort therefore got one token per burst. On GB10's
Qwen3.8-27B a burst is 13 rounds (32 or 33 tokens, 2.5 tokens a round) and one round takes 99,
111 and 126 ms beside one, two and three streams, so the session's tokens came 1.4, 1.6 and
1.8 s apart, against 0.074 s alone. A 24-token answer took 35 to 48 s, against 1.9 s alone.

Who that was: a request admitted to plain decode whose sampler runs on the host (greedy with a
repetition, frequency or presence penalty; a sampled penalty where the card keeps it on the
host), a constrained request that is sampled or post-think, a vision request, a fork. The first
class could not ride only because the worker asked a rider for a device-sample meta. The engine
did not need one: a rider row without it comes back as the host logits row the plain batched
step returns (`rider_sample_rows`, the plain step's own epilogue).

Neither existing mechanism saw it. The admission yield ends a burst for a request that waits to
be admitted, and this one is admitted. The cost model prices the speculative sessions' own
rate, which was fine, and takes no sample from a mixed tick.

## The change

1. **A host-sampled plain session rides.** `plain_rider_candidate` no longer asks for a
   device-sample meta. The session still has to be unconstrained, with no vision input and no
   fork, on a model whose plain rows the pooled walk reproduces bitwise. It then advances once
   per lockstep round inside the cohort's forwards, like every other rider, and it no longer
   keeps the model's other riders in phase (c).
2. **Round yield.** Before phase (a) the tick names its plain sessions (`plain_tick`) and
   decides whether one of them gets its step only after the bursts (`spec_round_yield_due`):
   - a session that decodes and is no rider candidate;
   - a rider candidate nothing carries this tick: its model has no pooled-cohort candidate at
     the tick's start, or a decoding non-rider of its model keeps the riders in phase (c);
   - a session still to prime, once one prefill chunk of its lane completes its prompt.

   While that holds, every MTP burst of the tick ends at its first round: the burst's entry
   hook call says continue and every round's call says stop (`spec_admit_keep`), through the
   exit a reached burst count and the admission yield take. A tick is then one round of every
   speculative session plus one plain step, the share a rider gets inside a cohort.
3. `MEMRA_ADMIT_YIELD=0` (lockstep fairness) turns the round yield off with the admission
   yield. No new switch.

Where a burst ends is content-neutral, so this moves when tokens come, never which: the
burst-boundary and seeded-invariance gates pass on the change, and every text in the cells
below is equal across arms and equal beside the streams and alone.

## Measurement

Qwen3.8-27B (NVFP4+Q5_K, MTP head) on one GB10 (DGX Spark, `121a` build) and on one RTX 5090
Laptop GPU (150-160 W enforced limit, `120a` build), 2026-11-12. Each cell runs L long
speculating streams (greedy or t1), then sends plain requests of 24 tokens one after another
beside them, then the same prompts alone (`tools/plain_cell.py`). The parent is #1227's head, the
change is this branch on it. Arms are interleaved on one unit in ABBA or BAAB order, two boots
per arm unless a row says one. "Gap" is the time between two tokens of the plain request;
"streams live" is the streams' summed rate while a plain request is live. Raw rows, texts and
server logs: `gb10/` and `rtx5090/` in the receipts.

**GB10, plain request beside the streams** (`gb10/plain`, ABBA):

| L | streams | plain | gap p50 parent -> change s | E2E p50 parent -> change s | streams live parent -> change tok/s |
|---:|---|---|---|---|---|
| 1 | greedy | greedy + penalty | 1.361 -> 0.101 | 34.56 -> 2.68 | 23.8 -> 23.1 |
| 2 | greedy | greedy + penalty | 1.605 -> 0.114 | 40.55 -> 2.99 | 40.8 -> 40.5 |
| 3 | greedy | greedy + penalty | 1.844 -> 0.132 | 47.57 -> 3.47 | 52.8 -> 52.0 |
| 2 | t1 | greedy + penalty | 1.738 -> 0.115 | 44.47 -> 3.03 | 37.8 -> 39.8 |
| 1 | greedy | t1 json-constrained | 1.361 -> 0.172 | 35.66 -> 5.70 | 24.1 -> 17.3 |
| 2 | greedy | t1 json-constrained | 1.613 -> 0.187 | 42.10 -> 6.39 | 40.9 -> 31.0 |
| 1 | t1 | t1 json-constrained | 1.447 -> 0.173 | 39.00 -> 5.82 | 22.4 -> 15.8 |
| 2 | t1 | t1 json-constrained | 1.728 -> 0.188 | 45.32 -> 6.46 | 38.4 -> 28.9 |

The same requests alone: gap 0.072 s, E2E 1.85 to 1.90 s. The t1-streams rows are `gb10/t1yield`.
A greedy-with-penalty request now rides the cohort, and the streams lose 1 to 5% of their
rate while it is live (its row joins their forward; the t1 row is 5% up, within spread). A constrained
request cannot ride; the round yield gives it one step per round, and the streams pay for its
forward while it is live: -28% and -24% at L=1 and 2 beside greedy streams, -29% and -25% beside
t1 streams. With the yield on, a burst is one round of 2.48 tokens (13 rounds of 2.54 before).
The constrained request's time to the first token still includes a full burst between admission
and its first step (memra#1249).

**RTX 5090, the same cell** (`rtx5090/plain`, ABBA):

| L | streams | plain | gap p50 parent -> change s | E2E p50 parent -> change s | streams live parent -> change tok/s |
|---:|---|---|---|---|---|
| 1 | greedy | greedy + penalty | 0.457 -> 0.059 | 11.74 -> 1.56 | 70.7 -> 41.7 |
| 2 | greedy | greedy + penalty | 0.720 -> 0.073 | 18.40 -> 1.94 | 91.1 -> 62.2 |
| 1 | greedy | t1 json-constrained | 0.470 -> 0.060 | 12.53 -> 2.06 | 69.2 -> 49.0 |

Alone: gap 0.026 s, E2E 0.50 to 0.74 s. This card has no cost table for this model, so the
rider's forward caps leave no room for a rider row: the greedy-with-penalty request rides but
takes its plain step inside each round, and the streams pay for it while it is live (-41% and
-32%), as they do for the constrained one (-29%). The trade is per card: on GB10 the rider row
is inside the cohort's forward and costs the streams 1 to 5%.

**Newcomers behind a cohort** (`gb10/new`, `rtx5090/new`, BAAB; L streams, then newcomers of
the same class one after another): on GB10 the time to the first token is unchanged (L=1, 2, 3
greedy 0.282/0.266/0.280 -> 0.269/0.271/0.283 s p50, L=2 t1 0.257 -> 0.271 s, p95 within 0.005 s)
and the streams' rates are equal. On the 5090, at L=2 the band admits the third request plain,
and it now rides: TTFT 0.842 -> 0.202 s (greedy) and 0.896 -> 0.214 s (t1) p50; L=0, 1, 3
unchanged. All newcomer texts are prefixes of their solo runs (240 of 240 per card).

**Closed loop on the 5090** (`rtx5090/norm`, one boot per arm, c greedy requests at once, no
plain request): aggregate 73.2/94.7/116.4/146.5 -> 73.5/95.4/118.3/154.0 tok/s at c=1, 2, 4, 8.
Nothing there asks for the round yield; the rows show no loss where it is idle.

**A long plain prompt** (`gb10/longprime`, ABBA; a 23,444-token greedy-with-penalty prompt
beside 2 greedy streams): TTFT 57.7 to 59.8 s parent, 55.2 to 55.9 s change, 19.4 to 19.6 s
alone. The prime is 24 chunks: 21.7 s of prefill and 36.0 s (33.3 s on the change) of full
bursts between them. The streams run at 26 to 27 tok/s until its first token on both arms,
and its decode gap after that falls from 1.59 to 0.119 s. The chunks keep their full bursts (memra#1253).

**Final crates.** #1227 changed under this lane, so the cells ran again with `parent2`
(#1227's final head 51b933a584) and `change2` (this branch on it, 7107ce3eba), whose crates
are the ones that land apart from main's own later changes:

- GB10 plain cell (`gb10/plain2`, BAAB): greedy with a penalty 1.337/1.586/1.834 ->
  0.100/0.114/0.132 s gap p50 at L = 1, 2, 3, streams live 24.2/41.3/53.1 -> 23.1/39.9/52.3 tok/s;
  json-constrained 1.441 -> 0.172 s (L = 1, t1 streams, streams -29%), 1.590 -> 0.187 s (L = 2,
  greedy, -25%), 1.710 -> 0.187 s (L = 2, t1, -24%). 144 plain texts identical.
- RTX 5090 plain cell (`rtx5090/plain2`, one boot per arm): greedy with a penalty 0.463/0.717 ->
  0.059/0.076 s at L = 1, 2, streams live 70.0/91.7 -> 41.6/61.1 tok/s; json-constrained 0.476 ->
  0.060 s, streams 68.6 -> 48.4 tok/s.
- GB10 newcomer cell (`gb10/new2`, one boot per arm): TTFT p50 equal within 0.006 s at L = 0 to 3
  and L = 2 t1, long rates equal, 120 newcomer texts prefix-equal.
- GB10 long prompt (`gb10/longprime`, ABBA): TTFT 57.6 to 59.8 s -> 55.0 to 55.8 s, decode gap
  1.590 -> 0.118 s p50, texts identical. One parent2 boot overlapped another lane's page-cache
  fill on the unit; its row is kept apart (`void.jsonl`) and boot `parent2-5` replaces it.

**Gates on the change.** Seeded invariance (`seeded-invariance-gate.py`): PASS on GB10 for
`change` and `change2` and on the 5090 for `change2`, 54 rows each over 10 prompts (ref, K=0
to 3, demotion). Burst boundary (`spec-burst-boundary-gate.py`, GB10): PASS for `change` and
`change2`, 229 rows each over the grid default, 1:2, 3:2, 3:3. Texts: every plain text is
identical across arms and identical beside the streams and alone; every stream text is a
prefix of the longest run of its stream. CPU checks on the rebased head bbcec42e31 (`cpu/`):
fmt, clippy, the engine (741) and server (1,216) suites and the flags and numeric censuses
pass; the red arms fail the tests they were built to fail.

## Rejected

- **Cap the burst at N tokens while such a session is live.** `MEMRA_SPEC_BURST=1` on main is
  not a round yield: a one-token target runs no speculative round, each stream advances one
  token per serial pass (76 ms), and the streams fell to 6.6, 8.8 and 9.9 tok/s for a plain gap
  of 0.15, 0.23 and 0.30 s. Any larger N is a knob between the two ends with no principled
  value.
- **Step plain sessions several times per tick.** The same shares as the round yield, but the
  tokens arrive a burst's worth at a time after a 1.3 to 1.8 s gap, and the time to the first
  token does not move.
- **Demote the speculative sessions while such a session is live.** Demotion is one way for a
  request, so a 3,000-token stream would stay plain after a 24-token neighbour left.
- **Make greedy with a penalty speculative on the MTP route**, as the DFlash2 route did
  (memra#310). It removes one class and is its own lane; constrained, vision and fork sessions
  would still wait a burst.

## Limits

- MTP route only, as the admission yield is. The DFlash2, glm5 and gemma bursts have no round
  hook that ends them, for a waiting admission or for a plain session (memra#1246).
- A plain prompt longer than one prefill chunk keeps its old cadence until its last chunk:
  each earlier chunk is followed by full bursts, as a speculative newcomer's long prime is
  under memra#1227 (memra#1253).
- While the round yield is on, the speculating streams pay for the plain session's forward:
  see the table above. The cost model takes no rate sample from those ticks (a mixed tick is
  not a steady regime), so it does not price that cost, and a stream is not demoted for it.
- GB10 and the RTX 5090 Laptop GPU on Qwen3.8-27B only. Two boots per arm in one order pair,
  one where a row says so; the gaps move by a factor of 8 to 14, far outside boot-to-boot
  spread, and the stream costs are stated per card, not as a timing claim.
