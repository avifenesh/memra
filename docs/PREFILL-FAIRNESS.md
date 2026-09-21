# Cooperative prefill qualification

Issue #521 owns scheduling between existing numerical chunks. `MEMRA_PRIME_YIELD`
remains default OFF while the current source, model and topology are qualified.
Historical September 8–11 receipts describe their own binaries and policies; they
do not qualify this controller or establish current PRO 6000 latency.

## Scheduling contract

The ON policy preserves the saved walker's frozen chunk tape, capture boundaries,
draft ingestion carry and committed speculative-round surplus. It does not change
chunk sizes, kernels, request admission, reservations or batch construction.

After a saved prime returns a chunk of measured wall time C, a ready decode peer
can receive ordinary worker ticks for `max(C - S, 0)`, where S is the configured
`MEMRA_SLO_P99_MS` interval (default 50 ms). Each tick retains one public decode
quantum per speculative peer and the normal plain decode phase. The deadline is
fixed to this prime's last successful advance. New arrivals cannot extend it.
With no ready peer, the prime proceeds immediately: closed streams, exhausted
output budgets, unfinished priming and absent output state do not buy a delay.

The cached-first-token/refill preference reserves at most S of elapsed time while
interactive prefill is waiting. It then permits one normal prefill phase even if
cached arrivals keep coming. This prevents the preference itself from starving an
admitted long prompt. Other work and already running chunks can still delay that
phase; this is not a wall-clock deadline for admission or first token.

S is a local service-sharing interval, not a promised ITL or TTFT percentile. A
frozen chunk is unpreemptible and can exceed S. The native gate must report its
actual cost, peer gaps, completion counts and long-prime progress. A zero,
negative, nonfinite or unrepresentable S refuses cooperative worker startup.
With `MEMRA_PRIME_YIELD` unset or `0`, these controls are disabled and the prior
scheduler behavior is retained, including the existing interpretation of S by
the separate lane admission policy.

## Structural route audit at 435a57a7

| Selected request route | Existing worker boundary | Current scope |
| --- | --- | --- |
| GDN MTP compatible plan, MTP attached, no hyper state, no PP cuts | `MtpPrimeWalker`, frozen trunk and draft-fill tape | Native qualification pending; unsupported MTP plans/topologies refuse cooperative prime |
| DFlash cold or resumed | `DsparkPrimeWalker`, saved taps and 256-row ingestion carry | Native qualification pending; admission/session-cap wait must be measured separately |
| GLM plain text | `Glm5PlainPrimeWalker`, saved trunk and owned cache | Native qualification pending per actual mixer and topology |
| GLM native MTP or DFlash spec | `Glm5PrimeWalker`, trunk/draft preparation and final anchor | Native qualification pending per actual route and topology |
| Generic or qualified Gemma plain chunked prime | Existing `prefill_tick` token budget and carried state | No new numerical partition; whole-prompt nonwalker branch refuses cooperative prime |
| Gemma assistant speculative prime | Synchronous constructor | Cooperative prime refuses; no claim that internal chunks yield to the worker |
| Vision/capture request shapes | Setup/capture outside the text walker contract | Cooperative request refuses; serial behavior remains available with the policy off |
| DSv4 serial serving worker | Internal chunk loops, no peer request scheduling boundary | Cooperative policy refuses at load |
| Legacy `MEMRA_SERVE_BATCH=0` scheduler | No saved plain-prime service loop | Cooperative worker startup refuses |

This is a capability audit, not a native support promotion. No route or hardware
default is selected from CPU fixtures. PP/TP and all supported mixer shapes must
be recorded explicitly; the existence of a GLM plain or DFlash adapter is not in
question. Refusal is an explicit policy limitation, not a model-format rejection.

## Required controls and native record

Run `python3 tools/test_prime_fairness.py` for the actual std-only engine walker
and scheduler, quantum observer and Trace modules. It tests frozen tapes, failed ownership/finalization,
elapsed recovery, continuous arrivals, initial-prefill starvation, non-runnable
peers and invalid service intervals. It does not type-check the complete server.
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

The 5090 compatibility follow-up cannot alter unrelated serving or replace the
blocking PRO 6000 evidence. Native execution and defaults await source review and
the coordinator's physical-card allocation.
