# Agent-pause KV demotion (lane/kv-pause-demote-20260831)

Executes Arc E of the tiering spec (darklanes
`research/engines-kv-oversubscription-20260830/SPEC-SESSION-TIERING.md`), consuming the
merged host tier (Arc B, lane/kv-host-spill-20260830) and the tenancy/compaction work
(lane/kv-tenancy-compaction-20260831). Base: b78b439bc. CPU-side unit coverage only in this
lane; the GPU cells below are named and PENDING. Both flags ship OFF.

## Mechanism

A response whose generation ends in a completed tool-call block is a session about to pause
for a client-side tool round trip. Today its KV either parks on device (continuation pool)
or holds device residence in the prefix cache until SLRU pressure. With
`MEMRA_KV_PAUSE_DEMOTE=1` (and the host tier armed):

1. **Arm (retire path)**: a retiring session that declared tools, is not vision, is not
   SWA-ring, and whose decoded generation tail passes
   `toolcall::tail_ends_with_tool_call` arms a `PauseCandidate` carrying the committed
   token tape (spec `committed` or `fed`) and a deadline `now + MEMRA_KV_PAUSE_DEMOTE_MS`.
   Exact (pool_key, tape) twins refresh in place; the list is bounded
   (`PAUSE_PENDING_CAP = 32`, above the A3 warm-session ceiling of 9).
2. **Wait (existing tick/timer machinery, never a thread)**: the worker's idle block stops
   sleeping indefinitely while candidates are pending; `idle_recv_wait` bounds the
   `recv_timeout` at the nearest pause deadline (constraint compiles keep their 5 ms poll
   exactly as before). A box with zero active sessions, which is exactly the tool-pause
   shape at low concurrency, wakes once at the deadline.
3. **Fire (per-tick sweep, after admission and retire)**: two demotable shapes per
   candidate, every copy on the CUDA owner thread:
   - the exact-fed PLAIN continuation-pool park: its boundary is published through the
     existing capture machinery (`prefix_snapshot` at the committed boundary), demoted
     via `host_demote_prefix_ref`, and ONLY a published host copy (or the tenant-cap
     evaporation that is the tier's law) releases the parked device cache; any failure
     keeps the park;
   - the deepest resident device prefix entry prefixing the tape: demoted by reference,
     removed from the device cache only after the host copy publishes.
4. **Resume**: the next request takes the unchanged admit-time path; the host probe finds
   the entry and promotes (Arc B machinery, untouched by this lane).

Cancellation is detected at fire time: a request that returned before the deadline consumed
the park (resume) or touched/pinned the entry (admit probe), and the candidate counts as a
cancel. Counters: `prefix_host_pause_demotes` (subset of `prefix_host_demotions`) and
`prefix_host_pause_cancels`, published in `/metrics`.

## Design decisions (with reasons)

- **Delay default 5000 ms, from measurement**: the A3 census (darklanes
  `research/kv-fastband-20260830/a3-census/RESULTS.md`) puts in-session gaps at p50 1.9 s
  with 56% under 5 s; an unconditional eager demote pays PCIe on the majority of pauses.
  At 5 s the demote fires only on the tail (p90 1.9 min, p99 6.7 min) where device bytes
  would otherwise idle for minutes against a ~21-42 ms promote. `0` keeps the unconditional
  shape as a gate/diagnostic arm.
- **Worker-side tool-calls predictor**: `finish_reason: "tool_calls"` is decided at the
  HTTP layer (`ToolStreamParser`), which the worker never sees. v1 predicts it with a tail
  check over the decoded last `PAUSE_TAIL_TOKENS` (24) generated tokens against the three
  served close markers, owned by `toolcall.rs` so predicate and parser cannot drift. Both
  divergence directions are safe (documented at the predicate): a conservative miss just
  waits for SLRU; a false positive costs one demote round trip, never bytes. Exact
  finish-reason plumb-through (an HTTP-to-worker per-session backchannel) is the named v2
  refinement; it was not worth an `Event::Done` shape change plus identity plumbing for a
  timing heuristic.

  AMENDED 2026-08-31 (lane/kv-battery-fixups-20260831), from the pod battery: the
  "conservative miss" direction was not an edge case: it was 100% of qwen3.8's natural
  tool pauses. The natural finish lands the request's stop id AFTER the close marker
  (`...</tool_call><eos>`; the stop id is pushed into `Session::generated` before the stop
  check fires), so the decoded tail never ENDED with the marker: the v1 predicate armed 0
  of 6 real `finish_reason: "tool_calls"` turns, while the same requests with
  `stop: ["</tool_call>"]` armed 2/2 (darklanes
  `research/kv-fastband-20260830/battery-20260831/pause-gates/RESULTS.md`, FINDING 1). Fix:
  `pause_tail_window` in worker.rs trims trailing stop ids at the TOKEN-ID level (driven by
  `Session::params.eos`, the exact per-request stop set: caller eos union `eog_ids()`)
  before the decode, reconstructing precisely the byte tail the HTTP parser judged. The
  predicate stays text-only and marker-only; the plumb-through stays a v2 refinement: it
  would buy nothing extra for this miss class, at the cost of a per-session HTTP-to-worker
  backchannel plus holding candidate state past retire.
- **Race law**: a demote racing the next request loses cleanly. The decision half
  (`pause_px_decision`) refuses pinned and post-arm-touched entries; the cache half
  (`PrefixCache::remove_at`) refuses pinned entries as a second guard; the park half only
  releases the park after the host copy published (`host_demote_prefix_ref` returns
  `Demoted`), and `pause_park_index` requires the exact fed tape so a consumed park reads
  as a cancel. All of this is single-threaded on the worker, so the "race" is tick
  ordering, and the sweep runs after admission and retire so a same-tick arrival wins.
- **Evaporation at the caps**: demotion respects the per-tenant cap and the pool budget
  exactly as the tier's law prescribes; a pause demotion that would push a tenant past its
  share EVAPORATES (park or entry drops, bytes cease) rather than staying resident, one
  semantics for the cap across both demote sources (`MEMRA_KV_HOST_TENANT_PCT=100`
  disarms, as before).
- **`host_demote_prefix_ref` refactor**: the demote body now works by reference and returns
  `HostDemoteOutcome` (Off / Evaporated / Demoted / Failed); the SLRU sink keeps its
  by-value wrapper with byte-identical behavior. This is what lets the pause path hold the
  source state until the copy publishes and fail closed on any failure.
- **SPEC/DSPARK parks are out of scope by design** (spec Arc C scoping): they hold live
  engine sessions (draft scratch, captured CUDA graphs, sampler/Philox state) and are not
  hostifiable in a small diff. Their device boundary prefix entries (published by the spec
  and dspark capture sweeps) still take the entry-shape demote.
- **Plane-bearing exclusions inherited**: SWA-ring sessions skip arming (and
  `prefix_snapshot` refuses them as the backstop); glm5 never reaches `prefix_snapshot`
  at all in this path. Vision sessions never arm (they bypass every reuse tier).
- **Flags OFF by default, written**: `MEMRA_KV_PAUSE_DEMOTE` (0) and
  `MEMRA_KV_PAUSE_DEMOTE_MS` (5000) have FLAGS.md rows in the same commit with both arms,
  the rollback seam (unset restores today's behavior exactly; OFF adds zero retire-path
  work), and receipts pointers.

## Unit coverage (this lane, CPU-only, anchored on invocations)

- `parse_kv_pause_demote_ms` fallback matrix (junk, negative, empty, zero).
- `idle_recv_wait`: pause-only wait sleeps to the nearest deadline (never the 5 ms poll
  spin), expired deadlines floor at 1 ms, constraint cadence unchanged, min of both timer
  classes with a real `PendingConstraintCompile`.
- `arm_pause_candidate`: twin refresh in place, distinct tapes/namespaces, cap enforcement.
- `pause_park_index`: exact-fed match only (prefix/extension/gone all decline).
- `pause_px_decision`: deepest untouched unpinned committed-prefix entry demotes; pinned
  and post-arm-touched entries lose cleanly; longer-than-tape entries never match.
- `pause_demote_second_guard_pinned_entry_stays_resident`: the cache-level race guard
  (`remove_at` refuses a leased entry; accounting asserted).
- `pause_demote_wiring_arms_fires_waits_and_publishes`: comment-stripped source anchors on
  the invocations (`tail_ends_with_tool_call(`, `arm_pause_candidate(`,
  `pause_park_index(`, `prefix_snapshot(`, `pause_px_decision(`, two
  `host_demote_prefix_ref(` calls, `px.remove_at(`), the idle-block wait wiring, and both
  counters reaching the metrics publish and the `/metrics` render.
- `toolcall::pause_tail_predicate_matches_all_three_close_dialects_and_nothing_else`,
  including the conservative-divergence direction (a call followed by trailing prose does
  not arm).

## PENDING GPU gates (battery box; defaults stay OFF before these)

> RAN 2026-08-31 on the battery box (memra `09bcd66c9`, receipts: darklanes
> `research/kv-fastband-20260830/battery-20260831/pause-gates/RESULTS.md`). P1/P2/P2b/P3
> GREEN, on a documented deviation (`stop:["</tool_call>"]`) because the v1 predictor
> armed 0/6 natural qwen3.8 tool pauses (FINDING 1; fixed, see the amended predictor
> bullet above). Gate 1's byte-identity oracle as written below is NOT an oracle (FINDING
> 2: the two arms take different serving programs and restore at different depths; the
> corrected same-depth oracle is `MEMRA_KV_HOST_VERIFY` digests + repeat determinism).
> Still pending before ON: the natural-shape arm receipt on the fixed predictor.

1. **Tool-round-trip byte identity via host vs never-demoted**: an agent-shaped multi-turn
   tool conversation (real prompts from the owner-blessed sxc/opencode corpora) run twice:
   flag ON with `MEMRA_KV_PAUSE_DEMOTE_MS=0` (forced demote during every tool pause,
   promote on the tool-result turn) vs flag OFF (never demoted). Greedy oracle, byte
   identity of every post-pause turn, 16/16 partial and 16/16 full shapes, spec re-arm
   receipts on the hybrid; teeth: the ON arm must show `pause demote` plus `promote` lines
   and a nonzero `prefix_host_pause_demotes`, or the gate is measuring the wrong server
   (arm identity per the boot-nonce law).
2. **Decode-tax A/B with the flag armed at the A3 pause distribution**: replay the A3
   in-session gap distribution (p50 1.9 s, 56% under 5 s, tail to minutes) over an
   agent-replay pool with vendor-default sampled decoding, interleaved x3 (x5 on anomaly),
   fresh boot plus boot-nonce arm identity, `git log -1` in every receipt. Measures:
   co-run decode tok/s tax while pause demotes fire (the D2H owner-thread stall, tick
   p95), resume TTFT vs never-demoted at each gap bucket, cancel-to-demote ratio at the
   5000 ms default, wasted-demote PCIe bytes. The multiturn cache twin (8-turn
   larger-prompt, cache on/off, per-turn TTFT and accept) rides the same cell per the
   measurement law.
3. **Failure paths executed** (loud-failures law): pause demote against a host pool at the
   tenant cap (evaporation observed, park released, counters exact), against a latched-off
   tier (candidate cancels, park kept), snapshot refusal on a `pos != fed` park (park
   kept), and a promote racing the sweep in the same tick (request wins, cancel counted).

## Named debt observed in this lane (not introduced by it)

- `cargo build` warns `field affinity is never read` on `DsparkReuseEntry` (worker.rs).
  SpecReuseEntry's twin field IS read by the H5 double-park guard; the dspark pool sets it
  at park and never consults it. That may be a real resume-path gap, so this lane
  deliberately did NOT silence the warning with an allow; it needs its own look (either
  wire dspark affinity nomination like spec's, or remove the field with a written reason).
- `field vision_memory is never read` on `Session` (worker.rs): held for its Drop
  semantics per its own doc comment; wants an underscore rename or a documented allow.
- `unused variable` warnings in memra-engine (`eps` in hybrid_forward.rs, `mtp` in
  spec.rs): engine-side files under active churn by other lanes; left untouched here.
- The leftover `dspark_draft_ready` unused binding in worker.rs admission WAS removed in
  this lane (pure dead code, no semantic change).

## v2 follow-ups (named, out of scope for v1)

- **Predictive re-upload (TokenCake pattern)**: promote the host entry back to device
  shortly BEFORE the predicted tool-return time so the resume never pays the promote
  latency. Needs the A3 per-tool-latency distribution and a bound on wasted re-uploads for
  sessions that never return; re-run gate 2 with it armed.
- **Exact finish-reason plumb-through**: HTTP-to-worker per-session backchannel replacing
  the tail predictor.
- **Continuation-pool hostification for spec/dspark parks**: spec Arc C2, blocked on the
  live-session decomposition question, not on this lane.
- **TTL from measured reload cost (Continuum pattern)**: replace the fixed
  `MEMRA_KV_PAUSE_DEMOTE_MS` with a per-model knee derived from gate 2's promote-latency
  rows.
