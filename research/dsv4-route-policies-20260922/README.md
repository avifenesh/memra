# DSv4 route policies: health, admission, memory cost (memra#500, #501, #503)

Verdict: the DSv4 serving thread now owns its liveness record, its admission book and its memory
door, and the route contract declares occupancy, progress, service-metrics and memory-cost as
implemented. Code half with CPU teeth and two fake routes through the completions handler. The
two-card receipt is pending. The DSv4 serving bring-up stays paused
(`docs/models/deepseek-v4-flash.md`); nothing here is a serving claim.

## The class

The DSv4 thread serves requests the central worker never sees. Before this lane, `/health` read
the central worker (idle on `recv`) over a busy or wedged route, a DSv4 request was priced as one
of the hybrid lane's 64 sessions, `/metrics` never counted it, and nothing sized its session
against the cards before allocating, so a session that did not fit died as a driver OOM mid-prime.

## What changed

- **#500, health (`health.rs::RouteHealth`, `progress.rs::ProgressSinkScope`).** The route
  registers its own record: `loading` until the thread's first idle, `idle`, `busy`, `dead` on
  exit. Forward progress is the thread's own prime odometer (a thread-local sink, so the global
  odometer and the central worker's verdict are untouched) plus a stamp per decode step or spec
  round. A busy route past `MEMRA_HEALTH_STALL_S` fails liveness by name; readiness waits for
  every route to publish; the process phase and `idle_for_ms` aggregate every serving thread.
- **#501, admission and telemetry (`route_telemetry.rs`).** Per-route book: waiting tickets
  (released at the route's dequeue, or wherever the request drops first), HTTP in-flight, running,
  served counters, a service-time window and a round-time window. The HTTP layer sheds, estimates
  and stamps `X-RateLimit-*` from the route's book, not the lane's. `/metrics` folds the route's
  served counters into the process totals and lists a `routes` array.
- **#503, memory cost (`dsv4_admit.rs`, `dsv4_serve.rs`, `dsv4_gpu.rs`).** Before a parked prefix
  is consumed or any state allocates, each owning card is charged the planned cache
  (`plan_session_cache_bytes`, the allocator's own arithmetic), a fixed per-session term measured
  at boot as occupied-memory deltas at a 1024-token calibration session (batched decode
  transaction, chunked-prefill transients at the default chunk `min(512, ctx)`, spec verify state,
  DSpark taps), and the lazily grown C4 gathers. The host tier is charged the active host-C4
  history against `MemAvailable` plus the parked bytes an eviction returns. The decision is
  `admit_memory::decide`: device on every card, then host with LRU eviction that spares the entry
  the request would restore from and never runs for a device shortfall, then a defer that
  re-reads every 50 ms within `MEMRA_ADMIT_DEFER_BUDGET_MS` clamped to half the stall bound.
  Refuse: 429 `rate_limit_exceeded`, `Retry-After: 5`. A session above a card's idle ceiling:
  400 `context_length_exceeded` naming the card, the bytes and the largest fitting session. A
  client that leaves mid-defer: dropped, counted `cancelled`. One `[admit-mem] ... route=dsv4-thread`
  line per decision; the boot line carries the calibrated terms.

## Receipts (`raw/`, tree in `raw/tree.sha`, rebased on main 0c86309bd)

- `memra-server-lib.log`: the full `memra-server` lib suite after the review fixes below,
  870 passed, 0 failed, 14 ignored.
  Includes `dsv4_admit::tests` (12), `route_telemetry::tests`, `health::tests` route cases,
  `route_contract::tests` (the DSv4 contract refuses only #449 and #535; the wiring gate finds
  `load.begin()`, the progress sink, `run.finish(stats)` and `dsv4_admit::admit_session(` in
  `dsv4_serve.rs`), `dsv4_serve::host_reclaim_tests`, and the two fake routes:
  `a_fake_route_serves_through_its_own_admission_and_books_its_metrics` (#501) and
  `a_fake_route_memory_door_refuses_defers_and_recovers_through_the_handler` (#503: a short peer
  card defers then 429s with `Retry-After: 5`; a charge above card 1's ceiling 400s with the
  largest fitting session; memory freed on the third reading admits and serves 200; a client
  abort mid-defer books `cancelled` with waiting, running and in-flight back to 0; the `/metrics`
  row reads refused 2, cancelled 1, completed 1). The fake route runs the real `admit_session`,
  `admission_answer` and `settle` the DSv4 thread runs; only the probe is scripted.
- `memra-engine-plan-progress.log`: `dsv4_gpu::session_plan_tests` (5) and `progress::tests` (6).
- `clippy-workspace.log`: `cargo clippy --workspace --all-targets -- -D warnings`, clean.
- `fmt-check.log`: `cargo fmt --all -- --check`, exit 0.
- `local-ci/local-ci-run1.log`: `tools/local-ci.sh` on the 5090, lane tree at `da6deb682`. No
  DSv4 checkpoint on this rig, so it proves the shared paths the lane touched: the prime
  odometer, the HTTP admission arm, the `/metrics` fold, and the health verdict with no route
  registered. GPU chain green: kernel-check ALL GREEN (109 cells, 10 skipped for absent
  models), sample-check, prime continuation gate PASS, decode-batch, graph-decode and
  graph-session gates, serve-smoke, health-fault-gate arms a/a2/b/d/e/f PASS, serve-stress c=64,
  request-fault (#525), prime fairness (#521), GPU probe recovery (#516) and the Qwen
  spec-on-cache-hit gate twice, all green. CPU chain: clippy clean, `memra-server` 868 passed.
  It then went red at the `memra-engine` lib suite: `dsv4_doors` found
  `MEMRA_DSV4_PREFILL_CHUNK` in an engine source with no door row. The name sat in a comment
  this lane added to `dsv4_gpu.rs`. The variable is a server knob the engine never reads, so the
  comment now says "a zero prefill chunk" and the census is unchanged.
- `local-ci/local-ci-rerun-cpu-and-gpu-lib.log`: the stages run 1 never reached, rerun on the
  fixed tree with the same commands: `memra-engine` lib 538 passed, 0 failed; the gguf
  artifact-present census 297 passed, 10 skipped (budget 10); the tier, KV and CLI portable
  suites 354 passed, 0 skipped; the GPU-only `#[ignore]` engine tests under
  `/tmp/memra-5090.lock`, 30 passed, 3 pair-only tests skipped with `SKIP-PAIR` (one card).
  The fix touches a comment only, so run 1's GPU and server results carry to the fixed tree.

## Review fixes (`raw/self-review/`)

- **Route shed text.** The three route shed messages in `reserve_route_admit` had lost their
  `\` line continuations, so each client-visible message carried a 22-space run mid-sentence.
  The continuations are back, and the shed tests now read the message body
  (`assert_route_shed_text`: names the route, no whitespace run). `shed-text-red.log` is the
  check on the old literals (3 failed), `shed-text-green.log` the fix (3 passed).
- **Route books scoped to this state.** The route registry is process-global. `/metrics` already
  read only routes this state serves; the hybrid in-flight netting and the request's route
  selection did not, so a book registered by another state in the process (a test's fake route)
  could price this state's requests. `served_routes(st)` and a served-model filter on selection
  close both. `a_route_book_this_state_does_not_serve_is_not_its_traffic` is red with both
  scopings reverted (`foreign-book-red-both-unscoped.log`: a foreign model selected the route),
  red with only the netting reverted (`foreign-book-red-netting-unscoped.log`: remaining 64,
  expected 63), and green on the fix.
- **One calibration delta per card** (revuto round 2). `stage_memory()` reads each stage's card
  whole, so two stages on one card each carried the card's full delta and the card was charged
  once per stage. `fixed_delta` now gives the card's delta, net of every co-located stage's
  cache, to the card's first stage and 0 to the rest; the DSpark tap goes on that same stage,
  so the spec term stays `max(prefill + tap, verify)` per card.
  `co_located_stages_share_one_card_delta` (two stages on card 0, one on card 1): the old fold
  gives `[200, 250, 20]`, card 0 charged 600 for a 300 delta
  (`calibration-shared-card-red.log`); the fix gives `[150, 0, 20]`, card 0 charged 300.
- `clippy-memra-server.log`: `-D warnings`, clean after all three.

## Limits (stated, not hidden)

- Forward-time scratch and the monolithic-prime scratch (chunk 0, or a prompt within one chunk)
  are not charged. A driver OOM there still answers 503 `overloaded`, the existing backstop.
- The per-card ceiling is effective free at boot. A co-tenant that arrives later reads as a defer
  and then a refusal, not as a never-fits.
- The C4 gather terms for the three widths are summed, an upper bound; pool rounding is not
  modelled.
- The fixed term is measured at one calibration capacity (1024). It is capacity-independent by
  construction (the batched transaction is sized by `max_seq`); the two-card receipt checks it.

## Pending: the two-card receipt

On a PRO 6000 pair with the DeepSeek V4 Flash checkpoint: the boot `[admit-mem] calibrated` line,
one admitted session whose measured occupied delta is at or under its charge on both cards, a
never-fits 400 at a capacity above the ceiling, a peer-card defer and refusal with a co-tenant
holding memory, and `/health` busy then idle across a long prime. Tracked on #500, #501 and #503.
