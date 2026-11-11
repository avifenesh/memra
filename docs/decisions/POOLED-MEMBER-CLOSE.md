# A pooled cohort member closes at the round its stream ended (2026-11-11)

**Status:** adopted for the pooled MTP and DSpark cohorts (the round-cadence arm). Issue
memra#1209. Receipts:
[`research/spec-pooled-close-20261111/`](../../research/spec-pooled-close-20261111/).

## Question

Several sessions that speculate on one model run their bursts in lockstep, one shared verify
forward per round (`step_spec_pooled`, `step_dspark_pooled`). Since memra#1207 a member whose
stream ends (EOS, a stop id, a stop string, its budget) stops its own rounds at the round hook's
verdict, and its counts end at the token that ended it. But the worker ran each member's post
step (`spec_burst_post`, `dspark_burst_post`) and with it `finish`, which sends the token snapshot
and `Event::Done`, only when the whole cohort call returned. The client got its usage and
`[DONE]` when the slowest member's burst ended.

How late was it, and what is the smallest change that closes the member at its round without
moving anything else?

## Measured on main

`tools/spec-stop-string-gate.py --cohort 2,4`: member 0 ends early while one or three companions
run 192 tokens. Close is the time from the member's last visible chunk to the end of its
response; the solo level is the same request alone. Qwen3.8-27B NVFP4 + Q5_K.

| device | route | solo non-stop, p50 | cohort of 2: p50 / max | cohort of 4: p50 / max |
|---|---|---|---|---|
| RTX 5090 Laptop | MTP | 35 ms | 293 / 899 ms | 863 / 2,011 ms |
| RTX 5090 Laptop | DSpark | 0 ms | 126 / 597 ms | 271 / 1,276 ms |
| GB10 | MTP | 38 ms | 619 / 1,715 ms | 1,382 / 2,498 ms |
| GB10 | DSpark | 0 ms | 90 / 853 ms | 88 / 1,683 ms |

Members that ended on a stop string (16 rows per MTP cell: five one-word and three three-word
stop strings, greedy and temperature 1; 9 to 11 on DSpark, whose sampled stream is its own and
does not always reach the plain stop words). Members that ended on EOS or `max_tokens` waited
too: on GB10 MTP, 598 and 699 ms on EOS, 232 and 400 ms at `max_tokens`. On DSpark the solo close
is 0 ms because its round hook sends the last token and the stream ends in the same round.

With `MEMRA_SPEC_GATE=0` DSpark admits one session at a time (`dspark_load_admits`), so no DSpark
cohort forms; the DSpark cohort boots run with the gate on, its LOW band at the cohort size.

## The change chosen

The pooled drivers take a per-member end hook (`PooledBurstReq::on_end`,
`DsparkPooledReq::on_end`). They call it at the gate pass where the member's outcome forms, the
round after the member's burst ended, while the other members run on. It receives the burst,
its drafted and accepted counts, and the session counters the post step reads: the MTP
telemetry snapshot, the DSpark round count. The engine runs no further round for that member,
so these are final.

The worker's hook decides through `spec_burst_verdict`, the post step's verdict (the id rule,
the stop-string clamp, the first EOS, the budget) moved into one function both call. If the
stream ended, it sends what `finish` sends (the token snapshot and Done) from the values the post
step writes back into the session later: the generated tokens plus the kept ones, the spec
counters plus this burst's rounds, drafted and accepted. It records the receipt in
`Session::early_done` and logs `[spec-pool] member finished inside the cohort`.

Nothing else moves. The post step, park, prefix publication and retire run where they ran, after
the cohort returns. `finish` sees `early_done`, sends nothing, and compares what it computed with
what was sent; a difference prints `[spec-pool] EARLY DONE MISMATCH` and fails a debug assert. A
step error after an early Done (a fault later in the cohort) is not sent: the client has its
complete answer, and the session is still retired as errored, never parked.

What stays with the post step: a continuing member, a gone client (the abort), a context-full end
(it reads the session's committed length, which the hook does not have) and the DSpark
burst-cadence arm (`MEMRA_SPEC_FIRST_TOKEN_EAGER=0`, which emits in the post step).

A request the client sends right after the early `[DONE]` is admitted at the next tick: the
worker reads commands only at the tick top, after this tick's post step, park and publication.
That is the order the serial route already has. Its arrival raises the admission count, so the
cohort's remaining members yield at their next round (the existing admission yield).

## Rejected

- **Run the whole post step at the member's end.** Move each member's SpecSession or
  DsparkSpecSession out of its Session for the cohort call, share the Session through a RefCell
  between the round and end hooks, and restore the sessions under `catch_unwind` if the CUDA
  cohort call panics. It also closes the member at its round, but it takes a lot of new failure
  surface (a session without its spec state during the call, unwinding across a device call) for
  host bookkeeping the client never sees.
- **Return finished members from the driver early.** The bursts' state lives on the driver's
  stack for the whole lockstep, so returning early would mean making the bursts resumable.

## Exactness

The other members' rounds, numerics and emission are untouched, so their texts are unchanged by
construction. The gate holds every companion to one text across its cohorts and across main and
the fix, and member 0 to plain decode's text, finish reason and usage (on DSpark at temperature 1,
to its own solo row, since that route's sampled stream is its own).

Measured on the final head (`tools/spec-stop-string-gate.py --cohort 2,4`, judged per boot):

| device | route | solo non-stop, p50 | cohort of 2: p50 / max | cohort of 4: p50 / max |
|---|---|---|---|---|
| RTX 5090 Laptop | MTP | 35 ms | 36 / 179 ms | 38 / 304 ms |
| RTX 5090 Laptop | DSpark | 0 ms | 0 / 169 ms | 2 / 103 ms |
| GB10 | MTP | 37 ms | 77 / 334 ms | 81 / 373 ms |
| GB10 | DSpark | 0 ms | 0 / 217 ms | 18 / 322 ms |

The maxima are stop strings that plain decode also holds back, whose held tokens arrive a cohort
round apart. Every member 0 equals plain decode's text, finish reason and usage, and every
companion wrote one text across its cohorts, main and the fix. Throughput of a four-member cohort,
fix/main over interleaved pairs: 0.992 mean in both samplers on the RTX 5090 (six pairs, 0.978 to
1.017), 1.009 and 1.002 on GB10 (three pairs, 0.999 to 1.023), inside the spread of one boot.
