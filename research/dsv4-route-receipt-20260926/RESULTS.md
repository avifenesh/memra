# The DSv4 route's two-card receipt: health, admission, memory (memra #500, #501, #503, 2026-09-26)

Model: `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`. Hardware:
2x RTX PRO 6000 Blackwell Server Edition. Program: the served TP/EP default (four lanes, B-row
graph steps, PDL, vocab-parallel head, push joins), naked defaults, `MEMRA_ENV_AUDIT=on`.

The code halves of the three issues landed in #655 with CPU and fake-route tests. This directory
is the pending two-card half: each claim below is read off a real boot, with the client's view in
`cells.jsonl` and the server's verdicts in `serve.log`.

| run | binary | what it drives | raw |
|---|---|---|---|
| v5t | main (push lane top) | #500 across a 12000-word prime, #501 c16, #503 first pass | `raw/v5t/` |
| v5u | same | #500 across a 24000-word prime, #503 under contention | `raw/v5u/` |
| v5w | same | #500 past the 120 s stall bound (`MEMRA_TIMEOUT_MS_MAX=600000`) | `raw/v5w/` |
| v5x | this lane's fix | #501 c16 after a long prime | `raw/v5x/` |

The listener opens after the weights load, so the load-phase polls got no answer (the files were
empty and are not kept). Readiness from then on is `/readyz`, which waited for the route.

## #500: route-owned progress in /health

**v5t, 12000-word prime (24007 rows, 68 s).** Every poll answered 200. Through the prime:
- the process phase read `busy`;
- `scheduler_phase` read `idle`, and the central worker's beat aged to 68 s;
- the route record read `busy`, with `prime_rows` climbing 0, 5632, 11776, 17920, 24007 and
  `progress_age_ms` at most 1.2 s.

After the request it read `idle`.

**v5u, 24000-word prime.** The request hit its own 90 s default deadline and answered 408 at
31232 rows, so the prime never reached the stall bound. In all 90 polls:
- the answer was 200;
- the central beat reached 89 s;
- the route's progress age stayed at most 1.45 s.

The shipped `timeout_ms` ceiling is 90 s and the stall bound is 120 s, so on the shipped surface
a served prime ends before the bound can judge it.

**v5w, past the stall bound.** This boot sets the measurement override
`MEMRA_TIMEOUT_MS_MAX=600000`, and the request carries a 600 s deadline. The prime covered 48007
rows in about 139 s, then decoded 16 tokens (200 at 141.0 s).
- All 141 polls answered 200.
- 21 of them came after the central worker's beat had passed the 120 s bound, reaching 140.0 s.
- In every one of those, the process read `busy`, the scheduler read `idle`, and the route read
  `busy`, with `prime_rows` rising (40960 at 120 s, 47616 at 140 s) and a progress age of at most
  1.57 s.

Before #655 the route had no record, so `/health` read the central worker's `idle` whether the
serving thread was priming or stuck. Now the route's own record is judged against the stall bound.
Here it stayed live past the bound because its progress was fresh. The stuck case (a busy route
with stale progress goes red by name) is the #655 fake-route test
`a_stalled_route_fails_liveness_beside_a_healthy_central_worker`. A live pair has no safe way to
wedge the serving thread on demand.

The top-level `forward_progress_age_ms` is the central worker's own age (89 s above) even while
the aggregate phase is `busy`. Liveness does not read it for a route: `live()` judges the
scheduler only when the scheduler itself is busy, and judges each route by its own record. A
monitor should read the status code or the route record, not that field.

## #501: route-derived admission

**v5t, c16 after the 12000-word prime.**
- Served: 8 requests with HTTP 200, finishing in 2.6 to 8.6 s. `X-RateLimit-Limit: 4` is the
  lane count.
- Shed: the other 8, each 429 `shed_deadline` with `Retry-After: 60`, stating an "estimated queue
  wait ~138s on route "dsv4f" exceeds this request's remaining timeout_ms deadline (89999 ms)".

That estimate was the defect. The route priced a queued request at the p50 of observed service
wall time, and its only sample was the 68 s prime. Two waves ahead then read as 138 s, and short
requests that would have been served in seconds were refused.

**Fix (this lane).** `RouteLoad::service_estimate_s` now prices a request by its decode: the mean
rounds per completed request times the round p50. This is the way the hybrid lane prices one:
mean tokens per request times the step p50.
- Before any round is recorded, it is the service p50.
- Before any request completes, it is the static fallback.

Prime time is left out on purpose. Missing a queued prompt's prefill can answer late. The other
error refuses work that would have been served, and the non-stream deadline gate already ruled
that a false refusal is the worse of the two. A request's rounds ride its `ServeStats`.
`route_telemetry::tests::a_long_prime_does_not_price_the_requests_behind_it` pins the receipt's
shape: 68 s before any round, 1 s after the prime's 16 rounds.

**v5x, the fix on the same shape.** Same boot shape, the same 12000-word prime (200 in 68.4 s), then
the same 16 concurrent short requests:
- all 16 answered 200, finishing in 2.6 to 20.0 s, four at a time on the four lanes;
- none was shed;
- `X-RateLimit-Reset` read 1 s where v5t read 69 s.

The estimate now under-reads a deep queue: the last four waited about 16 s against a 1 s price
per wave. That is the error this fix chose.

## #503: the memory door

The boot calibrates a fixed term of 4.71 GB per card and a ceiling of 15.66 / 15.69 GB. From that
point the door charges each session's planned cache and C4 gathers before anything allocates.

**Never fits.** A 2,000,000 `max_tokens` streaming request on the 1M route was admitted, with
capacity 1,048,480. The request clamps to the route's context, and a full-context session needs
13.57 GB per card, under the ceiling. On this pair every in-context session fits an empty route,
so the 400 `context_length_exceeded` path is unreachable here. It stays covered by the #655 unit
tests.

**Under contention (v5u).** Each part runs session A, which holds about 12.3 GB per card. B asks
for a 900k session 5 s after A starts, and C asks 24 s later.

| part | A holds the card by | B | C |
|---|---|---|---|
| 1 | a long generation, client closes at 31 s | defer, then 429 at 8011 ms (short 4.34 GB on dev0), `Retry-After: 5` | defer, admitted after 2391 ms, when A's client left |
| 2 | a 12000-word prime (headers only at 67 s) | 429 after 8391 ms (short 9.11 GB) | 429 after 8634 ms |

- **Part 2's shortfall.** It is larger than part 1's because the prime's own workspace is live
  while it runs, and the door reads live free memory.
- **Release.** A small request after each part was admitted at once, so every session released
  its reservation.
- **v5t's four staggered 900k sessions.** They were admitted one after another. One waited 50 ms,
  short 5.78 GB, because each finished its short answer before the next needed the memory.

The server's `[admit-mem]` lines carry each verdict with its per-card need, ceiling, shortfall
and wait: 16 in v5t, 11 in v5u.

## Documentation

`docs/SERVING.md` said `X-RateLimit-Limit` reads 1 for DSv4. It reads the route's lane count,
which is 4 on the TP/EP default. The route admission paragraph now describes the decode estimate.
