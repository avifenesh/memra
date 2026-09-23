# Cooperative prefill qualification

Issue #521 owns scheduling between existing numerical chunks. Main's
`MEMRA_PRIME_YIELD` default is ON for eligible routes; `0` disables it. An unset
setting retains the existing program on routes without a cooperative boundary.
An explicitly enabled request still refuses those unsupported feature routes.
Historical September 8–11 receipts describe their own binaries and policies; they
do not qualify this controller or establish current PRO 6000 latency.

## Scheduling contract

The ON policy preserves the saved walker's frozen chunk tape, capture boundaries,
draft ingestion carry and committed speculative-round surplus. It does not change
chunk sizes, kernels, request admission, reservations or batch construction.

After a saved prime returns a chunk of measured wall time C above S, the
configured `MEMRA_SLO_P99_MS` interval (default 50 ms), ready decode peers
receive ordinary worker ticks for `C - S` before any saved prime advances again.
The interval is worker-wide: it is the latest `completed + C - S` over every live
pending prime, so with m saved primes one chunk runs per interval, not m chunks
per tick. Each tick retains one public decode quantum per speculative peer and
the normal plain decode phase. When the interval closes, the least recently
served pending prime has the first claim for one further S; if its route does
not run in that window, any other pending prime may advance after it. Only a
prime's own advance opens an interval. New arrivals and peer steps cannot extend
it, and a retired or completed prime releases it. With no ready peer, the primes
proceed immediately: closed streams, exhausted output budgets, unfinished priming
and absent output state do not buy a delay. Chunks at or under S keep the
per-advance program: every pending prime advances every tick.

With ready peers, the resulting bounds are structural: peers get at least `C - S`
of service after every saved-prime chunk, and the primes together keep one chunk
per at most `2C` plus one worker tick, taking turns in least-recently-served
order. A lone long prime therefore pays at most `C - S` plus one tick per chunk
over its per-advance TTFT. The one exception is admission: a new prime is not
pending before its first chunk, because its walker does not exist yet, so each
admitted prime may run that first chunk inside an open interval, and the chunk
then opens its own. Admission (`MEMRA_MAX_SESSIONS`) bounds that to one chunk
per admitted prime. Tick-bounded plain prefill never sets a saved-prime state
and is never deferred.

The cached-first-token/refill preference reserves at most S of elapsed time while
interactive prefill is waiting. It then permits one normal prefill phase even if
cached arrivals keep coming. This prevents the preference itself from starving an
admitted long prompt. Other work and already running chunks can still delay that
phase; this is not a wall-clock deadline for admission or first token.

S is a local service-sharing interval, not a promised ITL or TTFT percentile. A
frozen chunk is unpreemptible and can exceed S. The native gate must report its
actual cost, peer gaps, completion counts and long-prime progress. A zero,
negative, nonfinite or unrepresentable S refuses cooperative worker startup.
With `MEMRA_PRIME_YIELD=0`, these controls are disabled. An implicit setting on
the legacy scheduler also skips cooperative SLO validation and keeps its shared
`step_session` path, including the separate lane admission policy's existing
interpretation of S. The legacy path cannot select a cooperative MTP walker;
that selection requires both the batching scheduler and the existing model
walker capability. Eligible routes remain enabled when the setting is unset.

## Structural route audit at 435a57a7

| Selected request route | Existing worker boundary | Current scope |
| --- | --- | --- |
| GDN MTP compatible plan, MTP attached, no hyper state, no PP cuts | `MtpPrimeWalker`, frozen trunk and draft-fill tape | Native qualification pending; unsupported MTP plans/topologies refuse an explicitly enabled cooperative request |
| DFlash cold or resumed | `DsparkPrimeWalker`, saved taps and 256-row ingestion carry | Native qualification pending; admission/session-cap wait must be measured separately |
| GLM plain text | `Glm5PlainPrimeWalker`, saved trunk and owned cache | Native qualification pending per actual mixer and topology |
| GLM native MTP or DFlash spec | `Glm5PrimeWalker`, trunk/draft preparation and final anchor | Native qualification pending per actual route and topology |
| Generic or qualified Gemma plain chunked prime | Existing `prefill_tick` token budget and carried state | No new numerical partition; whole-prompt nonwalker branch refuses explicit ON and keeps its existing program when implicit |
| Gemma assistant speculative prime | Synchronous constructor | Explicit ON refuses; implicit keeps the constructor, without claiming internal chunks yield to the worker |
| Vision/capture request shapes | Setup/capture outside the text walker contract | Explicit ON refuses; implicit and OFF retain the existing program |
| DSv4 serial serving worker | Internal chunk loops, no peer request scheduling boundary | Explicit ON refuses at load; implicit and OFF retain serial serving |
| Legacy `MEMRA_SERVE_BATCH=0` scheduler | No saved plain-prime service loop | Explicit ON refuses at startup; implicit and OFF retain the legacy path |

This is a capability audit, not a native support promotion. No route or hardware
default is selected from CPU fixtures. PP/TP and all supported mixer shapes must
be recorded explicitly; the existence of a GLM plain or DFlash adapter is not in
question. Refusal is an explicit policy limitation, not a model-format rejection.

## Required controls and native record

Run `python3 tools/test_prime_fairness.py` for the actual std-only engine walker
and scheduler, quantum observer and Trace modules. It tests frozen tapes, failed ownership/finalization,
elapsed recovery, one interval shared by concurrent primes, least-recently-served
turns, an absent first claimant, continuous arrivals, initial-prefill starvation,
non-runnable peers and invalid service intervals. It does not type-check the complete server.
Linux server compilation and affected CPU tests remain separate prerequisites.

The diagnostic wiring follows [REQUEST-LIFECYCLE.md](REQUEST-LIFECYCLE.md): actual
frozen input rows, host return through the existing last-chunk finalization,
unknown continuation counts and no invented cache-hit/queued-cancellation proof.
Event-capacity and explicit retirement-site coverage remain separate prerequisites
for a complete long trace and queued-cancellation qualification. Correctness and
mechanism cells may collect timing as unscored pilot data; performance/default
verdicts require later registered targets and holdout evaluation.

For each proposed route/default, freeze the model/artifact, source, binary,
topology, numeric settings, chunk tape and service interval before measurement.
On a parent-allocated PRO 6000 physical-card lease, the serving gate must include:

- A cold prompt of at least 131072 actual tokens, admitted beside short and cached
  peers; serial and mixed runs must retain identical output bytes/token IDs and
  all expected completed-request counts.
- Continuous short/cached arrivals while an admitted long prime makes nonzero
  progress and finishes. Report admission delays and refusals separately; do not
  score only surviving requests.
- Cancellation during actual prefill with measured time to observation and
  resource release, then a successful peer request. #526 owns the lifecycle
  trace; synthetic cached TTFT prime timestamps cannot establish this boundary.
- Queue/prime time, peer TTFT, emitted-token gaps and cancellation delay, with raw
  per-request rows and p50/p95/p99. Record the measured maximum quantum and worker
  tick. Declare target budgets before the scored cell; do not infer a pass from
  the configured service interval.
- Source/binary/model hashes before and after, actual route engagement, all child
  and lease exits, wrapper identity and 250 ms hardware telemetry. Default or
  performance decisions require balanced interleaved A/B in both orders, N>=5.

The service-interval A/B is `tools/prime-fairness-gate.py --shape service`: the
previous binary against the candidate, `--reps 6` (six boots per arm, the order
alternating each rep), decoders already streaming when a 131k prime and a second
32k prime arrive. It records the decoders' rate and ITL inside each prime's
window and each prime's TTFT, so the interval's price to the long prime is
measured beside what it buys the peers.

The 5090 compatibility follow-up cannot alter unrelated serving or replace the
blocking PRO 6000 evidence. Fresh qualification of the changed source awaits
source review and the coordinator's physical-card allocation; carrying main's
existing default is not a new performance or native qualification result.
