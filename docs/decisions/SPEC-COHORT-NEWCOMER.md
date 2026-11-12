# A newcomer primes before the pooled cohorts and joins the tick's lockstep (2026-11-12)

**Status:** adopted, memra#1227. Receipts:
[`research/spec-cohort-newcomer-20261112/`](../../research/spec-cohort-newcomer-20261112/).
Builds on [`SPEC-STALL-MARK.md`](SPEC-STALL-MARK.md) (memra#1211) and
[`SPEC-ENTRY-YIELD-BOUND.md`](SPEC-ENTRY-YIELD-BOUND.md) (memra#1201), whose bound noted that a
pooled cohort steps before the cold-first loop. Follow-up: memra#1244 (a plain request that
cannot ride a cohort).

## Question

The admission lane of 2026-08-06 made a newcomer's first token independent of a peer's burst:
a request arriving mid-burst ends the burst at its round (the admission yield), and a session
that has emitted nothing steps before a mid-generation peer (cold-first). The pooled cohorts
(lane/batch-spec: same-model continuation bursts in lockstep) were added in front of that loop.
A newcomer admitted at the tick top is no longer a waiting admission, so the yield does not see
it, and it is still priming, so it is no cohort member. It waited out the cohort's whole burst:
32 tokens per member in lockstep.

memra#1227 recorded this from the c = 4 phase starts of an ordinary cell (3 of 10 starts,
first tokens at 1.6 to 2.0 s instead of 0.68 s). Measured directly it is every newcomer: on
GB10's Qwen3.8-27B, 12 of 12 newcomers behind two long streams waited 1.6 to 2.2 s for their
first token, and 12 of 12 behind three waited 1.8 to 2.4 s, against 0.27 s behind one stream
and 0.21 s on an idle server.

A second wait sat beside it. Where the model pools, a prime burst ends at its first token so
the session can join "the next tick's cohort". Behind one long stream (no cohort yet) the
newcomer primed first, and its second token then waited that stream's serial burst: 1.27 s.

## The change

Phase (a) of the worker tick runs in two passes (`crates/memra-server/src/worker.rs`):

1. **The newcomers' pass.** An MTP speculative session that has emitted nothing, is no cohort
   candidate (it still primes), has a same-model speculative peer on a model that pools and
   holds no suspended prime steps first, in the tick's existing order
   (`spec_cold_serial_first`). With such a peer its prime burst ends at one token (the trim
   lane/batch-spec added), so this step is one prime chunk, or the whole prime of a prompt
   that fits one chunk, plus one token.
2. **The cohort pass.** The cohort pass first reads the prime quantum (a peer's prime that
   suspended in the first pass is pending now), then the cohorts form, so a session whose
   prime completed in this tick joins this tick's lockstep. Its pooled step is accounted as
   any member's (`spec_pass_steps`: a session that primed first is handled a second time only
   as a member of a cohort it joined). Every other serial session steps after the cohorts, as
   before.

- **A long prime.** A prime that spans ticks (the prime walker) runs its first chunk in the
  newcomers' pass and then takes the place the prime policy gives a pending prime, after the
  others. The cohort reads the quantum before it sizes its burst, so between the newcomer's
  chunks it runs one round, not a whole burst. A cohort therefore waits one prime chunk per
  newcomer per tick at most.
- **Every untrimmed first step keeps its place after the cohorts.** A lone speculative session
  (no same-model speculative peer, also when the model's other sessions are plain riders), and
  the DFlash2, glm5 and gemma routes: their first step is a prime plus a whole first burst, and
  it runs where it ran before. (The first form of this change named every session that had
  emitted nothing. The review found that a lone speculative newcomer beside plain riders then
  ran its whole first burst and a second one as a one-member cohort in the same tick, and that
  those untrimmed steps made a cohort wait a whole burst, not a chunk.)
- **`MEMRA_ADMIT_YIELD=0` keeps the cohorts first.** That value already selects lockstep
  fairness (index order, full-burst holds); the newcomers' pass is part of the newcomer-first
  order it turns off.
- **No new flag, no engine change.**

**Why no text can change.** The order of independent sessions inside a tick and the membership
of a cohort are already content-neutral: each session owns its cache and scratch, a pooled
member runs the serial step's preamble, round-cadence emission and post-burst accounting, and
where a burst ends does not move a greedy token or, since memra#1154 and memra#1186, a seeded
sampled one. The change moves when a session steps and which tick it joins a cohort in. It adds
no transition between numeric programs: prime, then pooled continuation, is the sequence a
newcomer already ran one tick later.

## Measured

GB10 (DGX Spark), Qwen3.8-27B NVFP4 + Q5_K with its MTP head, context 8,192, eight sessions,
every switch at its default, so the calibrated cost model owns the gate; tick trace on. The base
is memra#1241's head a20fd8c59f (its crates are main's); both arms of a comparison ran on one
unit, one boot per lease, arms interleaved. Rows, server logs and the table tools are in the
receipts. The newcomer cell, the closed loop and most stall boots ran the first form of the
change (every session that had emitted nothing stepped first); the long-prompt cell, run 2 of
the 0 ms stall cells and the exactness gates ran the final form. For a newcomer with a
same-model speculative peer and a one-chunk prime, which is every newcomer of the newcomer and closed-loop cells, the two forms run the same code (the stall cells' few newcomers admitted speculative beside demoted streams are one-token requests that end at their prime), and the final form's short-prompt cells repeat them.

**The newcomer cell.** L long greedy streams run alone for 8 s, then 12 streamed 16-token
newcomers are sent one at a time. Two boots per arm (base, change, change, base), 24 newcomers
per figure:

| long streams | first token p50 s, base / change | first token max s | first to second token p50 s | newcomer end to end p50 s | all tok/s (long + newcomers), mean of two boots |
|---:|---|---|---|---|---|
| 0 | 0.211 / 0.210 | 0.254 / 0.252 | 0.099 / 0.098 | 0.709 / 0.703 | 7.75 / 7.77 |
| 1 | 0.267 / 0.273 | 0.302 / 0.300 | 1.262 / 0.005 | 2.184 / 0.912 | 27.81 / 28.87 |
| 2 | 1.863 / 0.268 | 2.173 / 0.319 | 0.007 / 0.007 | 2.563 / 1.015 | 43.62 / 43.73 |
| 2, sampled (T = 1, seeded) | 1.881 / 0.267 | 2.214 / 0.317 | 0.007 / 0.007 | 2.704 / 1.093 | 40.96 / 42.99 |
| 3 | 2.029 / 0.289 | 2.419 / 0.336 | 0.009 / 0.009 | 2.934 / 1.144 | 55.16 / 54.51 |

- The long streams' own rate is about 6% lower on the change (23.2 to 21.7, 39.6 to 36.9 and
  51.4 to 48.1 tok/s behind one, two and three streams). It does not compare across arms: the
  newcomer window is 36 to 41% shorter when newcomers wait less, and the same twelve newcomers
  (a prime and fifteen rounds in a wider cohort each) fall inside it. The throughput figure is
  all tokens of the window: +3.8% behind one stream, +0.3% behind two, -1.2% behind three.
- **What a member pays per newcomer**, from the long streams' gaps above 150 ms: behind two
  and three streams the median of a member's twelve longest gaps is 0.41 to 0.64 s on the base
  and 0.41 to 0.44 s on the change (sampled: 0.41 to 0.66 s and 0.41 s), and its longest gap falls or holds (0.91 to 0.61 s, 0.96 to
  0.93 s). Behind one stream see Limits.
- Texts: 44 long texts prefix-equal and 336 newcomer texts equal across the four boots.

**A newcomer whose prime spans chunks** (the same cell with 5,117-token newcomer prompts, on
the final form; two boots per arm, 24 newcomers per figure):

| long streams | first token p50 s, base / change | first token max s | long streams' longest gap s |
|---:|---|---|---|
| 1 | 3.917 / 3.928 | 3.951 / 4.047 | 2.52 / 2.60 |
| 2 | 5.354 / 3.987 | 5.776 / 4.066 | 3.11 / 2.61 |

- Behind a cohort the first token comes 1.37 s sooner: the first chunk runs ahead of the
  cohort and the cohort runs one round between chunks. Behind one stream nothing moves; there
  was no cohort burst ahead of the prime.
- A member's gap per long newcomer is one prime chunk on both arms (2.5 s). The long streams
  emit about 51 fewer tokens per newcomer inside the newcomer window (1,105 against 1,722
  tokens in 69.8 against 85.7 s): the burst the cohort used to run ahead of the first chunk.
- The short-prompt cells on the final form repeat the table above (first token behind two and
  three streams 0.266 and 0.303 s against 1.881 and 2.027 s).

**Closed loop** (c clients, three greedy 160-token requests each; c = 2, 3, 4 twice per boot,
two boots per arm on one unit, four phases per figure):

| c | aggregate tok/s, base / change | first token p50 s | first token p95 s |
|---:|---|---|---|
| 2 | 36.35 / 40.49 (+11.4%) | 0.301 / 0.401 | 0.483 / 0.486 |
| 3 | 45.15 / 49.11 (+8.8%) | 0.825 / 0.452 | 2.445 / 0.835 |
| 4 | 53.44 / 58.12 (+8.8%) | 1.924 / 0.604 | 2.706 / 0.913 |

- The aggregate gain is the join: a request that starts beside running ones shares their
  verify forwards in the tick it primes in.
- c = 2 trades 0.10 s of first-token p50 for it; see Limits.
- 0 stall marks, 0 demotions, K = 3 at every admission on both arms; 216 texts equal.

**memra#1241's stall cells with two long streams** (a cohort; closed-loop clients of one-token
requests for 30 s, written `clients:think ms`; one unit; two boots per arm for every cell, arms
interleaved; a short request is one token, so all tokens = long tokens + short requests):

| cell | arm, run | long tokens in 30 s (both streams) | short requests served | all tokens | short p50 s | short p95 s | stall marks | newcomers admitted plain |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2:0 | base, run 1 | 361 | 145 | 506 | 0.386 | 0.414 | 2 | 139 of 145 |
| 2:0 | base, run 2 | 364 | 144 | 508 | 0.384 | 0.418 | 2 | 138 of 144 |
| 2:0 | change, run 1 | 381 | 192 | 573 | 0.304 | 0.313 | 1 | 184 of 192 |
| 2:0 | change, run 2 (final form) | 382 | 194 | 576 | 0.301 | 0.313 | 1 | 186 of 194 |
| 4:0 | base, run 1 | 295 | 233 | 528 | 0.456 | 0.935 | 2 | 216 of 233 |
| 4:0 | base, run 2 | 296 | 235 | 531 | 0.453 | 0.933 | 2 | 218 of 235 |
| 4:0 | change, run 1 | 305 | 308 | 613 | 0.362 | 0.755 | 1 | 292 of 308 |
| 4:0 | change, run 2 (final form) | 306 | 312 | 618 | 0.358 | 0.748 | 1 | 296 of 312 |
| 2:200 | base, run 1 | 757 | 67 | 824 | 0.664 | 0.672 | 0 | 0 of 67 |
| 2:200 | base, run 2 | 770 | 68 | 838 | 0.655 | 0.665 | 0 | 0 of 68 |
| 2:200 | change, run 1 | 435 | 111 | 546 | 0.336 | 0.352 | 2 | 106 of 111 |
| 2:200 | change, run 2 | 437 | 111 | 548 | 0.335 | 0.349 | 2 | 106 of 111 |
| 4:200 | base, run 1 | 462 | 104 | 566 | 0.855 | 1.094 | 0 | 0 of 104 |
| 4:200 | base, run 2 | 330 | 181 | 511 | 0.409 | 0.735 | 2 | 168 of 181 |
| 4:200 | change, run 1 | 371 | 193 | 564 | 0.401 | 0.633 | 2 | 180 of 193 |
| 4:200 | change, run 2 | 370 | 193 | 563 | 0.397 | 0.640 | 2 | 180 of 193 |

- **Runs.** Two boots per arm for every cell, arms interleaved on one unit. The 200 ms cells
  and run 1 of the 0 ms cells ran the first form of the change (301df182c6); run 2 of the 0 ms
  cells ran the final form (bab448b40d crates). The two forms differ only for a newcomer
  without a same-model speculative peer and for a prime that spans chunks.
- **0 ms think: both classes gain, in both runs.** The long cohort gets 381 and 382 tokens
  where it got 361 and 364 at two clients, and 305 and 306 where it got 295 and 296 at four;
  the short p50 falls from 0.39 to 0.30 s and from 0.45 to 0.36 s, and a third more short
  requests are served. Both arms set a stall mark and demote the streams.
- **2:200 is a stable trade.** The long cohort gets 43% fewer tokens (435 and 437 against 757
  and 770) and the cell 34% fewer tokens in all; the short requests get half the latency (p50
  0.34 s against 0.66 s) and 65% more of them are served (111 against 67 and 68). The base set
  no stall mark in either run; the change set it in both.
- **4:200, n = 2 per arm.** The change set the mark in both runs: 371 and 370 long tokens, 193
  short requests at a p50 of 0.40 s, 564 and 563 tokens in all. The base set the mark in one
  of its two runs: without it 462 long tokens, 104 short requests at 0.86 s, 566 tokens in
  all; with it 330 long tokens, 181 short requests at 0.41 s, 511 tokens in all. Two boots do
  not say how often the base marks.
- **Why.** On the base a one-token request admitted beside the cohort waited out the cohort's
  burst and primed after it, so the next arrival was rarely waiting when a burst began: no
  pressed entries, no stall mark (memra#1211), and the streams kept speculating while each
  short answer took 0.66 to 0.86 s. With the change the short requests prime first, the next
  arrival is waiting when the cohort's burst begins, the mark fires after four such entries,
  the streams demote and newcomers are admitted plain. That is memra#1241's outcome for one
  long stream at 2:200 (226 tokens for the stream, short p50 0.32 s): here 218 and 217 tokens
  per stream at 0.34 s. The share policy memra#1241 landed now reaches a cohort; a cohort no
  longer keeps its rate by making arrivals wait a burst.
- Long texts are prefix-equal across arms and runs in every cell (32 texts).

**Exactness on the final form:** the seeded-invariance gate passes on GB10 and on the RTX 5090 (54 rows each); the burst-boundary
gate on GB10 passes (229 rows). Both also passed on the first form (54 and 229 rows).

**RTX 5090 Laptop GPU** (the same 27B, where the band owns the gate; newcomer cell two boots
per arm on the first form and two more on the final form, closed loop c = 1, 2, 4, 8 one boot
per arm): behind one stream the second token comes at 0.002 s where it came at 0.47 s and
nothing else moves. Behind two streams the band admits the third request to plain decode, so
that row is memra#1244's on every arm (first token 0.82 to 0.86 s). All tok/s per boot overlap
across arms in every row (the card drifts 4 to 8% between boots of one arm), and every text is
equal.

**CPU:** the three tests (`spec_cold_serial_first_names_the_newcomers`,
`spec_pass_steps_each_step_once`, `spec_newcomers_prime_before_cohorts_is_wired`) pass on the
change. On main's phase (a) with only the two pure functions and the tests added, the wiring
test fails. Each of five red arms fails its test: nobody named for the first pass, no second
handling of a session that joined a cohort, the cohort pass first, every newcomer named
whether its prime is trimmed or not, no prime-quantum read before the cohorts form.

## Rejected

- **Count a newcomer admitted at the tick top as waiting**, so the cohort's burst entry yields
  to it. Not built. The newcomer gets the same first token, but the cohort emits nothing in
  that tick, the yield feeds the entry-yield bound and the stall mark (memra#1211) with
  entries that no arrival caused, and the newcomer's second token still waits a burst.
- **Reorder only: the newcomers' pass without the same-tick join.** Not built. The base arm's
  one-stream row is this case measured: the newcomer primes first (first token at 0.27 s) and
  its second token waits the stream's whole burst (1.27 s; 0.005 s with the join). Behind a
  cohort the reorder alone would turn a 1.9 s first token into a 0.27 s first token followed
  by a 1.5 to 1.7 s gap.
- **A shorter cohort burst while a session primes.** Not built. It gives up the burst's
  amortization on every tick with a newcomer for a wait the reorder already removes.

## Limits

- **A plain newcomer is not covered** (memra#1244). A request admitted to plain decode primes
  in phase (b), after every speculative burst of the tick, and one that cannot ride a cohort
  then gets one token per tick: on GB10 a greedy request
  with a frequency penalty waited 1.4 to 1.85 s for its first token and 1.3 to 1.7 s for each
  later one beside one or two speculating streams (16 tokens in 15 to 28 s, against 1.29 s
  alone), the same on both arms. On the RTX 5090 the band admits a third request plain beside
  two speculating streams, and its first token waits their burst (0.8 s).
- **One long stream pays one longer gap per newcomer.** On the base it had two gaps of 0.31 s
  per newcomer (the prime, then a tick shared with the newcomer); on the change it has one of
  0.40 s (the prime and a two-member round together). Its longest gap grows from 0.32 to 0.40 s.
- **Two closed-loop clients trade first-token p50.** At c = 2 the two streams now run in
  lockstep from their first tick and end in the same burst, so the clients' next requests
  arrive together and one waits the other's prime and tick: p50 0.30 to 0.40 s, p95 equal
  (0.48 s), each request ending about 1 s sooner, aggregate +11.4%. On the base the two ran
  one tick apart in four of four phases (the slow start mode of memra#1240).
- **A cohort is now inside memra#1211's share policy.** See the stall cells above: under
  short arrivals with a think time, two long streams used to keep speculating because the
  arrivals waited out their burst; they now meet waiting requests at their burst entries, the
  stall mark fires and they demote, as one stream already did.
- **A long prime costs the cohort the burst it used to run first.** With a multi-chunk prime
  the cohort gets one round between chunks from the first chunk on, where it ran a whole burst
  before the first chunk: about 51 fewer long-stream tokens per newcomer in the measured cell,
  for a first token 1.37 s sooner.
- **Memory in the join tick is not measured.** A newcomer's prime captures are released at the
  next tick top, so the first pooled burst it joins now runs while they are still held. No
  cell here is memory-tight (context 8,192, eight sessions).
- **Scope of the evidence:** Qwen3.8-27B on GB10 (the calibrated cost model owns the gate) and
  on an RTX 5090 Laptop GPU (the band owns it). Other models that pool take the same code path
  and were not measured. Whether arrivals spread over minutes (agentic traffic) meet a cohort
  often is not measured; when they do, this is the wait they no longer pay.
