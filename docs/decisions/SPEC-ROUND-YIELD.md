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

1. **An interactive host-sampled plain session rides.** `plain_rider_candidate` asks for a
   device-sample meta only of a judge or harvest session. An interactive session still has to
   be unconstrained, with no vision input and no fork, on a model whose plain rows the pooled
   walk reproduces bitwise. It then advances once per lockstep round inside the cohort's
   forwards, like every other rider, and it no longer keeps the model's other riders in
   phase (c). A dark host-sampled session stays in phase (c) at one token per burst: on a card
   where a rider takes its plain step inside every round it would cost the interactive streams
   that step in every round (the RTX 5090 rows below). A device-sampled dark session rides as
   it did before this change.
2. **One riding rule.** Phase (a) forms its riders, and the round yield's outlook names the
   riders nothing carries, through one helper (`carried_rider_models`, over `ride_facts`): a
   model's riders are carried when it has a pooled MTP cohort candidate and no interactive
   plain session of it decodes without riding. A dark one no longer keeps interactive riders
   in phase (c).
3. **Round yield.** Before phase (a) the tick names its interactive plain sessions
   (`round_yield_weighs`, `plain_tick`) and decides whether one of them gets its step only
   after the bursts (`spec_round_yield_due`):
   - a session that decodes and is no rider candidate (constrained, vision, fork, an
     eager-only model); a prompt fed tokenwise (an eager monolithic carried suffix, a
     remainder under `PRIME_MIN_T`) is the same trade, one prompt token per tick;
   - a rider candidate nothing carries this tick (item 2);
   - a session still to prime whose prime phase (b) runs this tick with the rest of its prompt
     within one chunk (`plain_prime_runs`). Phase (b)'s own predicates decide "runs": the
     refill grace, a cached hit waiting for its first token, the hit-suffix fence, the fanout
     formation window, the batch-formation hold and the deadline order are read through the
     helpers phase (b) calls (`cached_hit_waiting`, `hit_suffix_fence_rows`,
     `prefix_fanout_waiting`, `prefill_deadline_block`), as the tick's start knows them; the
     4 ms time windows among them count as closed when shorter than the shortest of the last
     64 phase (a)s that ran a round (`prefill_windows_close`, see Limits). A follower of an n>1
     choice group still being admitted and a prompt on the hyper-trunk plain prime walker,
     which may suspend inside the call, run no prime call that tick. A capture boundary
     inside that last chunk splits it into two calls on two ticks; both count, since each
     runs a real prime call that would otherwise wait a full burst (the long-prompt cell below
     shows the trace).

   It decides only in a tick with an MTP burst and with no DFlash2, glm5 or gemma burst
   (`round_hooked`): with no burst there is nothing to end, and beside a burst the round hook
   cannot end the plain session waits that burst out anyway while the MTP streams would lose
   their rounds.

   Judge and harvest sessions are never weighed, and on the ticks the yield shortens dark
   work waits (`DarkCadence`): phase (c) steps dark rows, and phase (d) primes a dark chunk,
   once every N yielded ticks, N being the rounds of the longest of the last 64 bursts the
   yield did not end, and on every other tick, which also restarts the count. A tick counts
   as yielded when the yield was on and a burst ran a round; the decision is taken after
   phase (a), so a tick whose MTP sessions only primed or ended at entry counts as any other.
   The yield therefore does not run dark work a burst's rounds times as often at the
   interactive streams' cost. A yielded tick is a round plus the plain session's step (on
   GB10 about 114 + 74 ms), so N of them take longer than one burst, and dark work gets about
   half its burst cadence while the yield holds (measured below), which is the lanes' order:
   "interactive is never preempted; harvest yields first".

   While the yield holds, every MTP burst of the tick ends at its first round: the burst's
   entry hook call says continue and every round's call says stop (`spec_admit_keep`), through
   the exit a reached burst count and the admission yield take. A tick is then one round of
   every speculative session plus one plain step, the share a rider gets inside a cohort. The
   first tick of each class prints `[spec-round-yield] engaged: decoding|priming|rider-uncarried`.
4. `MEMRA_ADMIT_YIELD=0` (lockstep fairness) turns the round yield off with the admission
   yield. No new switch. Host-sampled riding (item 1) is not under it, as device-sampled
   riding never was: where a rider takes its plain step inside each round (a card with no cost
   table for the model, see the RTX 5090 rows below), a host rider costs the streams that step
   in every round of a full burst with `0` too.

Where a burst ends is content-neutral, so this moves when tokens come, never which: the
burst-boundary and seeded-invariance gates pass on the change, and every text in the cells
below is equal across arms and equal beside the streams and alone, apart from one
pre-existing case on the speculative path (the vendor-shape rows of the riders cells).

## Measurement

Qwen3.8-27B (NVFP4+Q5_K, MTP head) on one GB10 (DGX Spark, `121a` build) and on one RTX 5090
Laptop GPU (`120a` build; enforced power limit 150 to 160 W for the first cells; for the
review-head cells, after a reboot, 150 to 160 W at a boot's start and 175 W under load, a
150 W TGP plus 25 W of dynamic boost, recorded per boot and left as it was), 2026-11-12. Each
cell runs L long speculating streams (greedy or t1), then sends plain requests of 24 tokens one
after another beside them, then the same prompts alone (`tools/plain_cell.py`). The parent is
#1227's head, the change is this branch on it, until the review-head cells below. Arms are
interleaved on one unit in ABBA or BAAB order, two boots per arm unless a row says one. "Gap" is the time between two tokens of the plain request;
"streams live" is the streams' summed rate while a plain request is live. Raw rows, texts and
server logs: `gb10/` and `rtx5090/` in the receipts.

**GB10, plain request beside the streams** (`gb10/plain`, ABBA). "All live" is the streams'
and the plain request's tokens together while the plain request is live:

| L | streams | plain | gap p50 s | E2E p50 s | streams live tok/s | all live tok/s |
|---:|---|---|---|---|---|---|
| 1 | greedy | greedy + penalty | 1.361 -> 0.101 | 34.56 -> 2.68 | 23.8 -> 23.1 | 24.5 -> 32.0 |
| 2 | greedy | greedy + penalty | 1.605 -> 0.114 | 40.55 -> 2.99 | 40.8 -> 40.5 | 41.4 -> 48.5 |
| 3 | greedy | greedy + penalty | 1.844 -> 0.132 | 47.57 -> 3.47 | 52.8 -> 52.0 | 53.3 -> 59.0 |
| 2 | t1 | greedy + penalty | 1.738 -> 0.115 | 44.47 -> 3.03 | 37.8 -> 39.8 | 38.3 -> 47.8 |
| 1 | greedy | t1 json-constrained | 1.361 -> 0.172 | 35.66 -> 5.70 | 24.1 -> 17.3 | 24.8 -> 21.3 |
| 2 | greedy | t1 json-constrained | 1.613 -> 0.187 | 42.10 -> 6.39 | 40.9 -> 31.0 | 41.4 -> 34.8 |
| 1 | t1 | t1 json-constrained | 1.447 -> 0.173 | 39.00 -> 5.82 | 22.4 -> 15.8 | 23.0 -> 19.8 |
| 2 | t1 | t1 json-constrained | 1.728 -> 0.188 | 45.32 -> 6.46 | 38.4 -> 28.9 | 38.9 -> 32.5 |

The same requests alone: gap 0.072 s, E2E 1.85 to 1.90 s. The t1-streams json rows are
`gb10/t1yield`. A greedy-with-penalty request now rides the cohort: its row joins the
streams' forward, the streams lose 1 to 5% of their rate while it is live (the t1 row is 5%
up, within spread), and the total rises 11 to 31%. A constrained request cannot ride; the
round yield gives it one step per round, and the streams pay for its forward while it is
live: -24 to -29%, and the total falls 14 to 16%. On those rows the change is a trade of
throughput for latency: the plain request's gap falls from 1.4 to 1.7 s to 0.17 to 0.19 s,
and the streams give up a quarter of their rate for the seconds it is live. With the yield on,
a burst is one round of 2.48 tokens (13 rounds of 2.54 before). The constrained request's time
to the first token still includes a full burst between admission and its first step
(memra#1249).

**RTX 5090, the same cell** (`rtx5090/plain`, ABBA):

| L | streams | plain | gap p50 s | E2E p50 s | streams live tok/s | all live tok/s |
|---:|---|---|---|---|---|---|
| 1 | greedy | greedy + penalty | 0.457 -> 0.059 | 11.74 -> 1.56 | 70.7 -> 41.7 | 72.8 -> 57.1 |
| 2 | greedy | greedy + penalty | 0.720 -> 0.073 | 18.40 -> 1.94 | 91.1 -> 62.2 | 92.5 -> 74.6 |
| 1 | greedy | t1 json-constrained | 0.470 -> 0.060 | 12.53 -> 2.06 | 69.2 -> 49.0 | 71.0 -> 59.9 |

Alone: gap 0.026 s, E2E 0.50 to 0.74 s. This card has no cost table for this model, so the
rider's forward caps leave no room for a rider row: the greedy-with-penalty request rides but
takes its plain step inside each round, as a device-sampled rider on this card always has.
The streams pay for that step while it is live (-41% and -32%), as they do for the
constrained request (-29%), and the total falls 16 to 22%. Every 5090 row is the same
throughput-for-latency trade: the plain request's gap falls from 0.46 to 0.72 s to 0.06 to
0.07 s. On GB10 the rider row is inside the cohort's forward, so a riding request raises the
total there.

**Several host-sampled riders** (`gb10/riders10`, `gb10/riders9`, `rtx5090/riders7`,
`rtx5090/riders6`, `rtx5090/riders5`; L greedy streams, then R greedy-with-penalty requests of 64 tokens at once,
then the same R alone; one boot per arm, the review head against main). Riders exist only
while the streams stay speculative:

| card | L + R | rider gap p50 s | rider E2E max s | streams live tok/s | riders live tok/s | all live tok/s |
|---|---|---|---|---|---|---|
| GB10 | 2 + 4 | 1.607 -> 0.132 | 106.3 -> 8.8 | 40.3 -> 33.4 | 2.4 -> 29.2 | 42.7 -> 62.5 |
| GB10 | 2 + 5 | 0.093 -> 0.093 | 6.5 -> 6.5 | 20.6 -> 20.5 | 49.5 -> 49.4 | 70.0 -> 69.9 |
| GB10 | 2 + 6 | 0.095 -> 0.096 | 6.7 -> 6.7 | 19.9 -> 20.0 | 57.6 -> 57.6 | 77.5 -> 77.5 |
| GB10 | 2 + 8 | 0.106 -> 0.106 | 7.5 -> 7.5 | 18.2 -> 18.2 | 68.6 -> 68.5 | 86.8 -> 86.7 |
| RTX 5090 | 2 + 1 | 0.751 -> 0.069 | 48.7 -> 4.5 | 88.3 -> 64.3 | 1.3 -> 14.3 | 89.6 -> 78.6 |
| RTX 5090 | 1 + 2 | 0.486 -> 0.057 | 31.8 -> 3.8 | 67.8 -> 42.7 | 4.0 -> 33.8 | 71.8 -> 76.5 |
| RTX 5090 | 2 + 2 | 0.030 -> 0.030 | 2.1 -> 2.1 | 64.7 -> 62.0 | 61.8 -> 59.7 | 126.5 -> 121.7 |
| RTX 5090 | 1 + 3 | 0.030 -> 0.033 | 2.1 -> 2.3 | 31.2 -> 30.3 | 90.6 -> 84.2 | 121.8 -> 114.4 |
| RTX 5090 | 2 + 4 | 0.037 -> 0.037 | 2.6 -> 2.7 | 50.9 -> 50.4 | 97.3 -> 95.6 | 148.2 -> 145.9 |
| RTX 5090 | 2 + 8 | 0.055 -> 0.055 | 3.9 -> 3.9 | 34.6 -> 34.0 | 132.2 -> 129.9 | 166.8 -> 163.9 |

With 4 riders beside 2 streams on GB10 (6 sessions in flight) the streams stay speculative
and carry the riders: a burst keeps its 13 rounds, each about 25 ms longer (a 13-round burst
1,523 -> 1,848 ms, `gb10/riders`, which ran the same cell on the first review head: 1.615 ->
0.133 s). The streams lose 17% while the riders are live, and the total rises 46%. With 5 or
more riders (7 sessions in flight) GB10's cost model demotes the streams to batched decode in
both arms. The RTX 5090 has no cost table for this model, and its band (HIGH = 4) demotes them
from 4 sessions in flight. Those rows are plain batches on both arms, nothing rides, and the
arms are equal within the spread of single boots (the riders' alone rate, with no stream in
flight, differs as much between the same boots). On the 5090,
several riders ride only at 3 in flight: 1 stream and 2 riders. Each rider there takes its
plain step inside every round, so the stream gives up 37% while the riders are live, and the
total still rises 6%; with 2 streams and 1 rider the streams give up 27% and the total falls
12%, the latency-for-throughput trade of the plain-cell rows. Every rider text is identical
across arms, beside the streams and alone.

**Which requests ride, per card.** A plain session rides when it is unconstrained, has no
vision input and no fork, and the cost model or the band admitted it to plain decode; on a
host sampler, only in the interactive lane. On
GB10 a greedy request with a penalty is admitted plain (eligibility) and samples on the host,
so it is the host rider these cells measure. A sampled request with a penalty samples on the
device on GB10, the RTX 5090 and the RTX PRO 6000 (`MEMRA_SERVE_DEVPENALTY` on); with the
device penalty off it samples on the host, and on GB10 the cost model admitted that shape
(the vendor's non-thinking sampling: T 0.7, top_p 0.8, top_k 20, presence penalty 1.5)
speculative at K = 1 beside two streams with 4 and 8 of them, so it did not ride there
(`gb10/riders`, vendor rows: rates equal across arms); on the RTX 5090 the band demoted the
streams with 4 and 8 of them beside, in both arms (`rtx5090/riders5`). With the device penalty
off, that shape's seeded texts are not invariant to a demotion in either arm: a request the
spec gate demotes after its first token or two writes another text (memra#1154).

**Newcomers behind a cohort** (`gb10/new`, `rtx5090/new`, BAAB; L streams, then newcomers of
the same class one after another): on GB10 the time to the first token is unchanged (L=1, 2, 3
greedy 0.282/0.266/0.280 -> 0.269/0.271/0.283 s p50, L=2 t1 0.257 -> 0.271 s, p95 within 0.005 s)
and the streams' rates are equal. On the 5090 at L=2 the band admits the third request plain.
It rode in both arms (first to second token 0.035 and 0.034 s); what moves is its prime: the
`Priming` class ends the tick's bursts at their first round, so its first token comes 0.842 ->
0.202 s (greedy) and 0.896 -> 0.214 s (t1) p50 after it arrives. The streams pay for those
shorter ticks: 162.9 -> 158.3 tok/s (greedy, -3%) and 142.7 -> 131.2 tok/s (t1, -8%; all
tokens 152.2 -> 142.9) summed over the two boots. L=0, 1, 3 unchanged. All newcomer texts are
prefixes of their solo runs (240 of 240 per card). The review head against main
(`rtx5090/new5`, one boot per arm): at L = 2 0.860 -> 0.212 s (greedy) and 0.999 -> 0.207 s
(t1), first to second token 0.032 s in both arms, the streams 77.6 -> 74.7 tok/s (greedy, -4%)
and 65.9 -> 66.5 tok/s (t1); at L = 1 0.159 -> 0.170 s (p95 0.176 -> 0.197 s), the stream
65.2 -> 63.2 tok/s. 72 of 72 newcomer texts are prefixes of their solo runs.

**Closed loop on the 5090** (`rtx5090/norm`, one boot per arm, c greedy requests at once, no
plain request): aggregate 73.2/94.7/116.4/146.5 -> 73.5/95.4/118.3/154.0 tok/s at c=1, 2, 4, 8.
Nothing there asks for the round yield; the rows show no loss where it is idle.

**A long plain prompt** (`gb10/longprime`, ABBA; a 23,444-token greedy-with-penalty prompt
beside 2 greedy streams): TTFT 57.7 to 59.8 s parent, 55.2 to 55.9 s change, 19.4 to 19.6 s
alone. The prime is 24 chunks: 21.7 s of prefill and 36.0 s (33.3 s on the change) of full
bursts between them. The streams run at 26 to 27 tok/s until its first token on both arms,
and its decode gap after that falls from 1.59 to 0.119 s. The chunks keep their full bursts
(memra#1253). The first token comes 2.5 to 3.6 s sooner, about two bursts, because the last
916 tokens are two prime calls on two ticks: the grid-aligned seed stops the first call at
896 tokens and the next tick primes the last 20 and decodes. The tick trace
(`[tick] ... spec_ms`) shows the parent running a full burst before each (1,517 and 1,405 ms
on boot parent2-1), the change one round (111 and 135 ms on boot change2-2).

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

**The review head.** The review changes (the lane filter, the prime outlook, the time-window
rule) ran as `change5` (3eeb4710f0 on cdfdfd6006) against main `parent3` (cdfdfd6006, with
#1233's prefix-hit fence). Main then moved to 9d7c086c67 (memra#1173's drafter admission
refusal, in the admission path) and 11b368f7d7 (memra#1221's pre-capture guard, in `spec.rs`);
the branch rebased onto 11b368f7d7 with the same code delta. One boot
per arm, prompts of 1,234 tokens (two prefill chunks, so a cold prime's first chunk keeps its
full burst, memra#1253) and a `+hit` cell that sends each prompt again, so the prefix cache
restores 1,234 of 1,234 tokens and the request is a prefix-hit newcomer:

| card | L | plain | gap p50 s | TTFT p50 s | E2E p50 s | streams live tok/s | all live tok/s |
|---|---:|---|---|---|---|---|---|
| GB10 | 2 | greedy + penalty | 1.608 -> 0.115 | 5.85 -> 3.11 | 44.96 -> 5.84 | 40.2 -> 33.4 | 40.7 -> 37.5 |
| GB10 | 2 | greedy + penalty, prefix-hit re-send | 1.42, 1.62 -> 0.114 to 0.116 | 1.57, 1.61 -> 0.07 to 0.13 | 35.8, 40.4 -> 2.84 | 42.0 -> 35.9 | 42.6 -> 41.4 |
| GB10 | 2 | t1 json-constrained | 1.617 -> 0.188 | 7.65 -> 4.72 | 46.83 -> 9.24 | 39.9 -> 29.4 | 40.4 -> 32.0 |
| RTX 5090 | 1 | greedy + penalty | 0.500 -> 0.054 | 2.81 -> 1.54 | 15.31 -> 2.82 | 61.2 -> 36.7 | 62.9 -> 45.3 |
| RTX 5090 | 2 | greedy + penalty | 0.746 -> 0.071 | 3.58 -> 1.95 | 21.52 -> 3.61 | 83.9 -> 53.3 | 85.0 -> 60.1 |
| RTX 5090 | 2 | greedy + penalty, prefix-hit re-send | 0.73, 0.74 -> 0.073 to 0.077 | 0.86, 0.82 -> 0.19 to 0.23 | 18.7, 18.4 -> 1.85 to 2.01 | 87.8 -> 54.9 | 89.0 -> 63.4 |
| RTX 5090 | 1 | t1 json-constrained | 0.523 -> 0.057 | 3.66 -> 2.14 | 16.29 -> 3.49 | 58.7 -> 40.0 | 60.2 -> 46.8 |

(`gb10/plain5`, `rtx5090/plain5`, L greedy streams. The re-send rows list each re-sent
request; their stream columns pool the cell's first sends and re-sends.) On main a re-sent
prompt waited a full burst for its first token (1.6 s on GB10, 0.8 s on the 5090); on the
change it gets it within about one round. A cold prompt of two chunks still waits a full burst
after its first chunk (memra#1253). With prompts this long, the streams lose more while the
request is live than in the 40-token cells: the prime's forward is the same work in both arms,
and it now falls inside a window 5 to 8 times shorter. The total while the request is live
falls on every row of this table, so here the change trades throughput for latency on both
cards. Texts are identical across arms and alone on both cards.

**Round 2 of the review.** The rules above that the second review added (interactive host
riders, one riding rule, the yield beside hooked MTP bursts only, the dark cadence) ran as
`change6` (11ae9661d1) against main `parent4` (11b368f7d7) on one GB10 unit, one boot per arm.
The plain cell (`gb10/plain6`) repeats `gb10/plain5` within 0.003 s on every gap: greedy with
a penalty 1.609 -> 0.115 s, the re-send's first token 1.58/1.61 -> 0.07 to 0.13 s, t1 json
1.617 -> 0.188 s, texts identical. The mixed cell (`gb10/mixed`, `tools/mixed_cell.py`; L = 2
greedy streams, then the requests named, 24 tokens each unless stated):

| requests beside the streams | request | gap p50 s | streams live tok/s | all live tok/s |
|---|---|---|---|---|
| interactive greedy + penalty, then interactive t1 json 1 s later | penalty rider | 1.607 -> 0.115 | 40.8 -> 29.7 | 41.9 -> 36.0 |
| | t1 json | 1.597 -> 0.189 | | |
| interactive t1 json | t1 json | 1.597 -> 0.189 | 41.1 -> 30.1 | 41.7 -> 33.9 |
| harvest greedy + penalty (48 tokens), then interactive t1 json 3 s later | t1 json | 1.610 -> 0.189 | 40.7 -> 29.5 | 41.9 -> 33.6 |
| | harvest (p50 over its life) | 1.610 -> 1.619 | | |

The first row is the `rider-uncarried` receipt: the penalty request rides until the json
request decodes, the json request then keeps it in phase (c), and the yield names it
(`[spec-round-yield] engaged: rider-uncarried` on a tick with two bursts, followed by
one-round bursts). The dark harvest request's tokens across the yield came 3.15 and 3.01 s
apart on the change: it stepped on one of the 25 yielded ticks (the 16th; N was 16), against
1.2 to 2.0 s between its tokens beside the full bursts of the same boot (a full burst there
takes 1.2 to 1.75 s, `spec_ms`), so about half its burst cadence. The table's harvest gap is its p50 over its whole life (84 s), of which the
yield held about 6 s. The streams' 30.1 -> 29.5 tok/s beside the json request with and
without it is inside the spread of the json-only rows on the change boots (29.7, 30.1,
30.1): the harvest row added about 3 ms to the one tick it stepped on, so this cell does not
resolve its cost. It is admitted only
with `MEMRA_SHED_HARVEST=100`: GB10's interactive step p99 (the 74 ms plain step) is above
0.9 times the 50 ms SLO, so by default the lane gate sheds it with a 429 (`gb10/mixed-shed`,
the first change boot). The raise moves admission only; phase (c), phase (d) and the cadence
read the SLO. Texts are identical across arms, beside the streams and alone.

**Gates on the change.** Seeded invariance (`seeded-invariance-gate.py`): PASS on GB10 for
`change`, `change2`, `change3`, `change5` and `change6` and on the 5090 for `change2`,
`change5` and `change6`, 54 rows each over 10 prompts (ref, K=0 to 3, demotion). Burst
boundary (`spec-burst-boundary-gate.py`, GB10): PASS for `change`, `change2`, `change3`,
`change5` and `change6`, 229 rows each over the grid default, 1:2, 3:2, 3:3. Texts: every
plain text is identical across arms and identical beside the streams and alone, and every
stream text is a prefix of the longest run of its stream, except the vendor-shape riders noted
above. CPU checks on the round-2 head (`cpu/checks.log`, 11ae9661d1 on main 11b368f7d7): fmt,
clippy, the engine (743) and server (1,222) suites and the flags and numeric censuses pass.
Each of the 18 red arms fails exactly the tests it was built to fail, and the head passes all
seven (`cpu/redgreen.log`).

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
  hook that ends them, for a waiting admission or for a plain session (memra#1246). A tick that
  holds one of them does not yield at all, so a server holding both routes keeps full bursts
  while one of those sessions bursts.
- A plain prompt longer than one prefill chunk keeps its old cadence until its last chunk:
  each earlier chunk is followed by full bursts, as a speculative newcomer's long prime is
  under memra#1227 (memra#1253).
- While the round yield is on, the speculating streams pay for the plain session's forward:
  see the table above. The cost model takes no rate sample from those ticks (a mixed tick is
  not a steady regime), so it does not price that cost, and a stream is not demoted for it.
- Interactive lane only. A judge or harvest plain session never engages the yield, never
  rides on a host sampler, never keeps interactive riders in phase (c), and steps once every N
  yielded ticks (`DarkCadence`): about half its burst cadence on GB10 while the yield holds.
  A device-sampled dark rider rode before this change and still does; on a card where a rider
  takes its plain step inside every round it costs the interactive streams that step, as it
  did.
- The cadence's N is the longest of the last 64 bursts the yield did not end. Until one such
  burst has run (a server whose first bursts are all yielded), dark work runs on every tick.
  N does not move while the yield holds: if the bursts before a long yield were all cut short
  (for example to one round each by waiting admissions), N stays small and dark work runs on
  most yielded ticks until a full burst runs again.
- The prime outlook is read at the tick's start. Phase (b)'s time windows (refill grace,
  fanout formation window, batch-formation hold, each `MEMRA_PRIME_BATCH_HOLD_MS`, 4 ms by
  default) count as closed when they are shorter than the shortest of the last 64 phase (a)s
  that ran a speculative round, since phase (b) comes at least that long after the start of a
  tick whose bursts the yield would end (`prefill_windows_close`): the fanout window and the
  formation hold run from the session's arrival and the refill grace from the previous tick's
  end, all at or before the tick's start. A longer configured window, a server that has not
  yet run a round, or a recent short phase (a) (a small co-loaded model) keeps them open and
  the outlook holds the prime back, as phase (b) may, until that phase (a) leaves the window.
  Read at the start instead, a newcomer admitted at the tick's top is always inside its 4 ms
  window, and the first form of this review held its prime behind a full burst (GB10,
  `gb10/plain3`: first token 1.79 s where the earlier change had 0.31 s).
- The outlook reads phase (b)'s inputs before phase (b) does: the prefill rate before phase
  (b) marks this tick's, no fanout session advanced yet, and the pooled cohorts as the tick's
  start knows them (phase (a) forms them after the newcomers' pass, memra#1227). On the tick
  after a prefill its deadline order, and on a tick where a newcomer completes a cohort its
  riding verdict, can differ from phase (a)'s and (b)'s for that one tick.
- GB10 and the RTX 5090 Laptop GPU on Qwen3.8-27B only. Two boots per arm in one order pair,
  one where a row says so; the gaps move by a factor of 8 to 14, far outside boot-to-boot
  spread, and the stream costs are stated per card, not as a timing claim.
