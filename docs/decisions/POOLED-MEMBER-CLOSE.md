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

MEASURED-MAIN-TABLE

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

MEASURED-FIX-TABLE
