# The cost model marks a concurrency stalled when speculation keeps meeting waiting requests (2026-11-11)

**Status:** adopted, memra#1211 (part: the price; the share-policy questions 1 and 2 stay open). Receipts:
[`research/spec-stall-mark-20261111/`](../../research/spec-stall-mark-20261111/). Follows
[`SPEC-ENTRY-YIELD-BOUND.md`](SPEC-ENTRY-YIELD-BOUND.md) (memra#1201), whose deferred slot-arm
item this settles. Builds on memra#1228 (memra#1223: the batch-formation hold no longer skips a
whole tick).

## Question

The entry-yield bound keeps a speculating stream alive under sustained short arrivals, but it is
not a share policy. On GB10 the long stream still ran at 2 to 5 tok/s while the cost model kept
pricing speculation above plain at n = 2..5 (42.8 against 25.6 tok/s at n=2). The realized-rate
sampler takes a sample only between two steady ticks that emitted tokens. Under these arrivals no
tick is steady (every tick admits a newcomer), and a session that yields at its burst entry emits
nothing, so the boot table's price stands.

The cost it cannot see is structural: a speculative burst runs serially and gives its tick back
when a request waits (the admission yield), while a plain session steps in the batched phase and
never yields. The #1201 probe's slot arm (a zero-rate sample in the realized slot) reached the
right verdict but stored it wrongly. A zero slot is never re-measured, and its neighbours scale
from it through `priced()`.

## The change

- **A pressed entry** is a speculative burst whose entry hook call sees a request waiting for
  admission. It counts whatever the entry-yield bound then does, and whether or not the call
  carries a token. Each session keeps its last four burst entries. "Waiting" is the gauge the
  admission yield already reads (`PENDING_ADMITS`): any reserved request, of any lane or model.
- **Four pressed entries in a row** (`SPEC_STALL_ENTRIES`) make the tick:
  - mark the tick's live count n_tick (the count its demotion check reads, so the stalled session
    demotes; never n = 1);
  - open one admission mark per request that was waiting at the run's last entry. Each of the
    next admissions of the model uses one and marks the wave its own verdict reads, before the
    verdict (`spec_cost_admit_wave`); unused ones lapse at the next tick.
  - Why the admission site: the wave is known exactly only there. A first form marked n_tick
    alone, and a second form n_tick plus the waiting count; both missed the newcomers whenever the
    tick count and the admission wave differ (sessions finishing inside the tick, requests queued
    behind others).
- **The demoted session holds its waves.** A session demoted by a live mark keeps that mark's
  count. Before every admission, the model's hold is the latest expiry among those sessions' live
  marks, and the admission marks its own wave until that expiry, never on a fresh time-to-live.
  Why: at 2 clients and 200 ms think the two short clients overlap, so admissions alternate between
  wave n and n+1. Nothing marks n+1 otherwise: the demoted stream is plain and never bursts again,
  one-token requests never burst, and at 200 ms think no request is waiting when the stall is
  marked. The exact form left 80 of 109 newcomers speculative at 2:200 (89 of 151 at 4:200).
  A held wave never lives past the mark that holds it. When that mark expires, the session
  drops its count (before the next admission, or at its next decode step under a waiting
  request) and a later mark at the same n does not revive it. A speculating session that a held
  wave's mark demotes holds and refreshes that wave in its own right.
- **Pressure on the demoted session refreshes its mark.** When a session a live mark demoted steps
  its plain decode (in the batch, or as a rider beside a pooled cohort) while a request waits for
  admission, the mark lives on a full time-to-live from now. Only a live mark is refreshed, never
  created. Why: the demoted session never enters a burst again, so without this the mark expired
  64 s after the stall even while arrivals continued. On GB10 the 120 s cells kept the gain for
  60 s, then fell back to main's level (2:0: 196, 201, 91, 63 long tokens per 30 s). With the
  refresh the gain holds in every segment.
  - The two rules are not symmetric, on purpose. Setting a mark takes four pressed entries in a
    row; keeping one takes a single waiting request at one decode step of the demoted session.
    So a mark ends 64 s after the last such step. See Limits.
- **A marked n is plain for every draft length** in `spec_cost_verdict`. Admission, tick demotion
  and sampled restore all read that verdict.
- **A mark expires** `SPEC_COST_REALIZED_TTL` (64 s, the existing evidence lifetime) after its
  last refresh (a new stall at n, or the pressure above). It lives beside the realized slots, never in them: no price changes, nothing
  spreads to n-1 or n+1, the memra#1216 tie-break never sees it, and on expiry the verdict
  returns to the price it had.
- **No new flag.** `MEMRA_SPEC_GATE_MODEL=0` (the band owns the gate) bypasses the cost model,
  marks included. The bound stays at N=1.

## Measured

GB10 (DGX Spark), Qwen3.8-27B NVFP4 + Q5_K with its MTP head, context 8,192, eight sessions,
every switch at its default, so the calibrated cost model owns the gate. A long greedy streamed
request runs alone for 8 s, then closed-loop clients send one-token requests for 30 s (or 120 s),
then it runs alone for 12 s. A cell is written `clients:think ms`. The main arm is main
98337756cf; the form is the shipped tree on it. Both arms of a pair ran on one unit, one boot
per arm, single runs, pass/fail style. Rows, server logs with the tick trace and the table
tools are in the receipts; so are the earlier forms' own sweeps (two boots per arm for the hold
form), which show the same pattern.

**30 s cells:**

| cell | long tokens, main / form | short p50 s, main / form | short p95 s, main / form | newcomers admitted plain |
|---|---|---|---|---|
| 2:0 | 77 / 195 | 0.419 / 0.296 | 0.512 / 0.307 | 190 of 198 |
| 4:0 | 59 / 155 | 0.823 / 0.353 | 0.925 / 0.674 | 300 of 316 |
| 2:200 | 141 / 226 | 0.312 / 0.323 | 0.320 / 0.327 | 110 of 116 |
| 4:200 | 52 / 189 | 0.717 / 0.388 | 0.731 / 0.615 | 184 of 197 |
| 2:0 sampled (T=1) | 76 / 196 | 0.422 / 0.296 | 0.511 / 0.306 | 190 of 198 |
| 4:0 sampled (T=1) | 49 / 156 | 0.823 / 0.354 | 0.921 / 0.691 | 300 of 316 |

- **2:200 is a measured trade, not a pass on latency.** The long stream gets 60% more tokens
  (226 against 141) and the short requests pay 11 ms at p50 and 7 ms at p95 (13 and 13 to 16 ms
  in the hold form's two-boot sweep). The tick trace says where it goes: on main the shorts are
  speculative and prime inside bursts that give their tick back to an arrival; with the mark a
  short arriving mid-tick waits for the long stream's plain decode step (76 ms), which never
  yields, and each lone plain prime pays memra#1228's 4 ms formation wait. About 4 ms is the
  formation wait; the rest is structural to a plain decode step.
- **One client, and every 1000 ms cell: 0 marks, the same draft length at every admission** on
  both arms (1:0, 1:200, 1:1000, 2:1000, 4:1000), rows within a few tokens and ms (1:200 long
  tokens 450 against 453).
- **The longest gap of the long stream is the bound's** (0.92 and 1.76 s at 2:0 and 4:0 on
  both arms): the four pressed entries before the mark are two silent yields and two forced
  rounds.
- **Long texts** are prefix-equal to main's in every cell, greedy and sampled.

**120 s of arrivals, per 30 s segment** (long tokens; short p50 s):

| arm | 2:0 | 4:0 |
|---|---|---|
| main | 77 / 81 / 76 / 80; 0.42 to 0.51 | 56 / 169 / 169 / 169; 0.82 then 0.36 |
| the hold without the refresh (an earlier boot, main then at 77 / 81 / 72 / 84 and 56 / 168 / 168 / 168) | 196 / 201 / 91 / 63; 0.30 then 0.48 | 153 / 170 / 68 / 34; 0.35 then 0.89 |
| shipped | 195 / 202 / 200 / 202; 0.30 | 153 / 169 / 169 / 168; 0.36 |

- At 2:0 the gain holds in every segment (396 refreshes for one mark). Without the refresh the
  mark expired 64 s after the stall and the cell fell back to main's level or below it.
- At 4:0 main recovers by itself after the first segment: its acceptance window ages out, the
  n=5 verdict falls back to the band and the stream demotes. The mark's gain there is the first
  segment (153 against 56) and the whole-cell short p95 (0.359 s against 0.864); segments 2 to
  4 are a tie.

**Recovery, the long stream still open** (a c=2 follow of six 160-token requests; the main rows
are main with memra#1228 on the other unit):

| pressure cell | follow starts | follow admissions, shipped | follow aggregate tok/s, main / shipped |
|---|---|---|---|
| 2:0 | 5 s after | plain x6 (3 refreshes) | 29.98 / 25.21 |
| 2:0 | 70 s after | K=3 x6, priced as at boot | 31.03 / 36.44 |
| 2:200 | 70 s after | K=3 x6, priced as at boot | 31.73 / 36.45 |

- Inside the time-to-live the marked waves serve plain: the follow costs 16% of its aggregate
  (25.2 against 30.0 tok/s) while the demoted stream is live. That is the price of a verdict
  that outlives the pressure.
- After 70 s of quiet the mark and the demoted session's count are gone and admissions return to
  K=3 with the boot price, although the demoted stream is still open. The follow's aggregate is
  then higher than main's because the long stream beside it is plain there and speculating on
  main.
- Follow texts are equal across arms in every cell.

**Ordinary concurrency** (c = 1, 2, 4, 8, three 160-token requests per client, three boots of
the form before the review's fixes): 0 marks, 0 stalled verdicts, the same draft length at every
admission as main (K=3 x3, x6, x12; c=8 plain by the far rule with its one or two ramp
admissions, which vary between main's own boots the same way), every text equal to main's,
aggregates equal at c = 1, 4 and 8.

**The c = 2 start mode.** At c = 2 the census has start modes on every arm, set by a race
between the second client's first request and the worker's admission pass: both requests
admitted in one tick and ending in one burst (40.4 to 40.7 tok/s), admitted in one tick and
ending apart (37.7 to 38.3), or admitted one tick apart (36.3 to 37.4). Interleaved on one unit,
16 phases per arm: main started in the same tick in 11, the shipped tree in 7 (Fisher exact,
two-sided, p = 0.29). By phase position (first to fourth c = 2 phase of a boot) the same-tick
counts go both ways: 1 of 4 against 3 of 4, 4 against 2, 4 against 0, 2 against 2. The aggregate
per mode is equal across arms (40.58 against 40.39, 37.97 against 37.93, 36.86 against 37.13
tok/s), so the arms differ only in how often each mode occurs; mean aggregate 38.60 against
37.79 tok/s (-2.1%), and the bound if the slowest mode replaced the fastest on every start is
-8.5%. No phase sets a mark, and no verdict or admission differs. No code path was found that
would move the race (the change adds microseconds to an admission, which would favour the same
tick). Pooled with the earlier traced census boots (not interleaved: 5 of 8 against 0 of 6) the
count is 16 of 24 against 7 of 22, p = 0.04, so a shift is not excluded; it is tracked in
memra#1240.

**Exactness on the shipped binary (GB10):** the seeded-invariance gate passes (54 rows) and the
burst-boundary gate passes (229 rows).

**RTX 5090 Laptop GPU** (measured on the hold form; the later commits act only on a live mark
or on a session a mark demoted, and this card sets no mark): Qwen3.8-27B, where the band owns
the gate, and Ornith-1.5-35B-A3B, where the cost model owns it and already picks plain from
n=2: 0 marks, rows and texts equal to main with memra#1228 in the stall cells and the c = 1..8
census; with `MEMRA_SPEC_GATE=0` only the bound acts (0 marks); seeded-invariance passes (54
rows).

## Rejected

- **The slot arm as built** (zero-rate samples in the realized slot, memra#1201's probe). Never
  re-measured, and `priced()` scales n-1 and n+1 from it. The red arm of the unit tests is this
  storage: the n=2 prices fall from 33 / 42 / 47 to 12 / 15 / 17 tok/s when n=3 stalls.
- **The same zero with a time-to-live on the slot.** Expiring the zero expires the slot's
  positive history with it, and while it lives it still scales the neighbours.
- **A mark at the tick's count only** (the first form built). At 2:200 the mark landed at n=2
  while newcomers were priced at the projected wave n=3: 25 of 103 newcomers plain, short p95
  0.55 s against 0.32.
- **The tick's count plus the number waiting** (second form). Requests that finish inside the
  tick leave room: at 4:200 the mark named n=4 and n=7 while admissions ran at wave 5 (54 of 152
  plain).
- **The exact admission mark without the hold.** 26 of 109 newcomers plain at 2:200 and 51 of
  151 at 4:200: at 200 ms think no request waits when the stall is marked, and nothing marks the
  second wave afterwards.
- **The hold without the refresh.** The 120 s cells above: the gain ends 64 s after the stall.
- **A refresh that can create a mark**, or one keyed on anything but a session a live mark
  demoted. It would mark waves in traffic that never stalled. The unit test's red arm is this.
- **A half-life discount on the price** from the fraction of silent entries. Under the bound
  that fraction sits near 0.5 and at EMA weight 0.3 swings between 0.41 and 0.59, across GB10's
  flip thresholds at n=2 and n=4 (0.42 and 0.42): it flaps. Not built.
- **Sampling non-steady ticks.** Every turn arrival in agentic traffic becomes a sample, so the
  calibrated choices at c = 1..8 move. Not built.
- **Per-session demotion only** (memra#1201's demote arm). Newcomers stay speculative; measured
  worse than the bound on both classes.
- **A probe tick that re-measures a marked n.** Under pressure it measures a stall; without
  pressure no speculating session exists. Expiry gives the same information for free.
- **An exception to the 4 ms formation wait for a marked wave** (to recover the 2:200 short
  latency). Not taken: the wait is memra#1228's rule and the trade is stated instead.
- **A rate term in the trigger** for 4 clients at 1000 ms. Not needed: that cell presses about
  half its entries, never four in a row, and sets no mark.

## Limits

- The refresh cannot tell sustained pressure from ordinary arrivals. While a session a mark
  demoted is live, any request that waits during its plain decode step extends the mark, and
  every admission of the model marks its own wave meanwhile. So after a stall of a second or
  two, the model's admissions can stay plain for as long as that session lives and requests
  arrive less than 64 s apart: minutes, for a long stream. The 5 s recovery row prices it (the
  follow's aggregate 16% lower). It ends with that session or with 64 s without a waiting
  request.
- The waiting request is any reservation on the admission gauge: another model's request or an
  embedding request presses and refreshes too, as it already makes a burst yield. An admission
  mark opened by such a request is used by the model's next admission, which may not be the
  request that waited.
- Only the MTP route's bursts press. Where the band owns the gate, or with
  `MEMRA_SPEC_GATE=0`, no mark acts and the entry-yield bound alone holds the stream.
- One model on one calibrated card. The RTX 5090 rows show the change is inert where the cost
  model does not own the gate or already picks plain.
- memra#1211's questions about the share itself stay open: how many rounds a speculating stream
  should keep under sustained arrivals (question 1), and whether a one-token request should take
  the speculative path at all (question 2: a request-size rule, not a price). Its question 3
  (should the cost model see ticks in which a speculating session ran nothing) is what the mark
  answers, beside the sampler instead of inside it.
