# specmech-20260810 — c=2 stage-resident multi-session speculative pipeline

## Mission

Implement the c=2 recovery mechanism from `research/specpp2-20260810/RESULTS.md`:
two spec sessions A/B interleaved across PP-2 stages so stage0 verifies B's round
while stage1+head verifies A's round, swapping each interval. The specpp2 receipt
established: verify = 95.13% of a K=1 round, rounds are host-issued serial per
session, and a second session queues instead of filling the idle PP stage
(worker.rs burst loop finishes session A before touching session B).

Gate: behind `MEMRA_SPEC_PIPE=1`, default OFF. WIN = c=2 pipelined-spec beats
plain (115.2 tok/s class on box1 step-3.7-flash IQ4_XS+MTP shape).

## Plan

1. DESIGN COMMIT — read `generate_spec_pp2` round path (spec.rs ~1700-1800,
   MEMRA_SPEC_PP_ANATOMY markers), `PpNRt` (pp.rs: stage streams,
   fence_stages_behind, publish_to), worker.rs spec burst loop (1255-1257,
   1377-1397). Write as-built design here: round owner, where A-stage1 / B-stage0
   overlap can be enqueued without a host sync, what state goes per-session,
   where the single-stream scratch invariant (lib.rs ~616-618) bites. COMMIT.
2. INCREMENT 1 — round-granularity software pipeline across existing stage
   streams: enqueue A stage0 verify; after A's boundary copy departs stage0,
   enqueue B stage0 on the stage0 stream. No overlapping accept/draft yet.
   Per-session scratch duplication where the invariant demands. cargo test
   green locally. Commit per compile-able step.
3. Box1 exactness gates (flock /tmp/memra-gpu.lock, step-3.7-flash IQ4_XS+MTP,
   PP_STAGES=2 DEVICES=0,1 CTX=262144 MOE_GROUPED=1 PREFILL_TICK=2048):
   run-spec K=1..8 PASS, spec/plain byte identity, b1fix one-hash golden with
   PIPE=1.
4. A/B on box1 (N=5 interleaved, one lock hold): c=2 pipelined-spec vs plain vs
   serial-spec. Then c=1 sanity (no regression vs serial spec) + c=4 (plain
   policy untouched).
5. RESULTS.md: as-built schedule, A/B, exactness table, promote/hold.

## Stop conditions

- If increment 1 beats serial-spec but not plain: land the receipt anyway,
  project whether overlapping accept/draft closes the gap, stop cleanly.
- If the design step shows per-session scratch duplication is a bigger refactor
  than the round interleave: stop after design commit, deliver design + effort
  re-estimate as RESULTS.md.
- NO origin pushes.

## Block list

- [x] Design commit (as-built anatomy + overlap points + per-session state bill)
- [x] Increment 1: two-session round interleave behind MEMRA_SPEC_PIPE=1
- [x] Local cargo test green
- [x] Box1 exactness gates (run-spec K=1..8, byte identity, b1fix golden)
- [x] Box1 A/B c=2 (pipelined vs plain vs serial-spec), c=1 sanity, c=4
- [x] RESULTS.md + promote/hold call

## Design notes (as-built)

### Round ownership

- `HybridModel::generate_spec_inner2` (spec.rs:3551) owns the WHOLE burst: prime,
  init feed, draft-graph capture, then the round loop (spec.rs:4549 `while
  keep_going && out.len() < max_new`). One call = one session's burst; the worker
  calls it via `generate_spec_session_{sampled,constrained}` (spec.rs:3430/3451).
- Worker side: spec sessions burst solo in phase (a) of the tick
  (worker.rs:3369-3455, `for i in spec_order { step_session(...) }`) —
  `step_session` (worker.rs:6006) runs one whole burst per session, serially.
  This loop is where a two-session pipelined entry would be dispatched.
- Verify: `decode_step_t_core` (spec.rs:1539) dispatches to
  `decode_step_t_core_ppn` (spec.rs:1687) when the ppN door is open. Issue order
  inside: `fence_stages_behind(caller)` -> stage0 scope (embed+layers+TX on
  stage0 stream) -> last-stage scope (RX+layers+head on stage1 stream) ->
  `publish_to(last, caller)`. All enqueues are async; the host sync lands later
  at the accept phase's `preds = e.dtoh_u32(&preds_d)` (spec.rs:5017) or the
  devacc `dtoh_u32(&acc_out)` (spec.rs:5084).

### The two ordering laws that shape the pipeline

1. `publish_to` (pp.rs:907): after session A's verify enqueue, the CALLER
   (primary worker) stream carries a WAIT on A's stage1 completion. Anything the
   host then enqueues on the primary stream — session B's draft, B's accept
   kernels, commits — is queued BEHIND A's stage1. So a naive "enqueue A's
   verify, then draft B, then enqueue B's verify" serializes: B's draft won't
   run until A's verify completes.
2. `fence_stages_behind` (pp.rs:942, the #87 law): every verify entry orders ALL
   stage streams behind the caller stream's current point. If B's verify fences
   after A's publish wait is on the caller stream, B's stage0 is transitively
   ordered behind A's stage1 — overlap dead. The pipelined entry must fence ONCE
   per interval (before either verify is enqueued), then run both verifies
   fence-free. Safety holds because the interval fence already ordered stage
   streams behind every queued primary consumer, and A's verify body frees only
   stage-owned buffers with no primary readers between the two enqueues.

### What overlaps in increment 1 (round-granularity, host-serial accept)

Issue order per interval, single caller stream S (the worker primary):
draft A (host-synced per token) -> draft B -> interval fence -> enqueue verify A
-> enqueue verify B -> A accept+commit -> B accept+commit. GPU timeline:
A-stage0 | A-stage1 overlapped with B-stage0 | B-stage1. Pipeline length
s0 + 2*s1 ~= 25.9ms per TWO rounds vs 2*17.2 = 34.4ms serial (~25-30% round-rate
gain), NOT the 1.89x steady-state bound — that needs accept/commit/draft-next
of one session riding concurrently with the other session's verify (increment 2,
explicitly out of scope here).

### Shared-scratch invariant (lib.rs:591-648 Engine pools)

`fa_part_pool`, `fa_vf16_scratch`, `argmax_partials` are per-Engine,
single-stream-safe BY DESIGN (lib.rs comment + pp.rs:564-575 build note). Stage
engines are already per-stage, so A-stage1 || B-stage0 is safe (different
Engines). The hazard is the PRIMARY engine: draft argmax
(`argmax_token_device`), verify-col argmax (`argmax_token_device_col`), and
accept kernels all run through `e` (primary) — two sessions' primary-engine work
must never be in flight concurrently. Increment 1 satisfies this by
host-serializing all primary-engine phases (draft/accept/commit end in sync
readbacks before the host switches sessions); per-session scratch duplication is
NOT needed at round granularity. It becomes the refactor bill only for the full
steady-state schedule.

### Per-session state that must split (increment 1)

Already per-session (SpecSession, spec.rs:322): cache, MtpScratch (draft KV),
committed/last_h/next_pred, sctr/uctr, DraftGraphCtx (g_* buffers), pending_tok,
turn_ckpt, telem. Round-LOCAL state currently living in generate_spec_inner2
locals (h_seed_buf, fill_prev, preds_d, snap, col bufs, phase counters) is
per-call, so a two-session entry needs one copy per session — allocation, not
refactor. The monolith itself (~1500 lines, all feature arms inline) is the real
cost: increment 1 takes a REDUCED-MATRIX pipelined entry (greedy, unconstrained,
non-sampled, session-mode continuation bursts only, devacc path) behind
MEMRA_SPEC_PIPE=1 rather than restructuring the monolith; anything outside the
matrix falls back to the serial loop.

### Design pass 2: accept/commit tail

- The verify return is still asynchronous. In the devacc arm the PRIMARY stream
  first enqueues all verify-column argmaxes, `spec_accept_greedy`,
  `spec_seed_gather`, and `spec_rollback_kv`; the 8-byte
  `dtoh_u32(acc_out)` is the phase's host wait. `n_acc` and `bonus` are therefore
  authoritative before any host cache mirror is changed.
- `commit_verified_prefix` is part of the primary-engine critical section, not
  part of either PP stage. It updates host KV lengths and `cache.pos`, and its
  recurrent-state rebuild launches (`ssm_conv_ring_rebuild_dc` and
  `gdn_scan_s128_dc`) read the session's verify checkpoint plus the session's
  device accept result. A and B cannot share the primary engine here.
- The true-hidden refresh is also inside that same critical section. It reads
  the OLD per-session `fill_prev` plus the current verify's `vx`, rewrites only
  that session's `MtpScratch`, then the devacc epilogue publishes the gathered
  `h_seed_buf` into that session's `fill_prev`. The next draft must not start
  before this publication, but the other session's already-enqueued PP verify
  is independent of it.
- Full, partial, and zero-accept non-replay arms all leave a pending bonus and a
  cache-authoritative session boundary. The burst tail parks `pending`,
  `fill_prev`/`last_h`, graph context, counters, and telemetry back into the same
  `SpecSession`; no cross-session state merge is needed.

### Design pass 2: worker admission and pairing

- Keep `spec_gate_defaults(true) == (0, 1)` and `choose_spec_k` unchanged. The
  pipeline is an explicit measurement door, not a placement-policy promotion;
  the box1 arms force speculative K exactly as the prior receipt did. c=4 plain
  therefore remains the live default.
- Pair after the existing cold-first `spec_order` sort, at the point that today
  calls `step_session` once per speculative session. A pair is admitted only
  when both entries are warm empty-suffix continuations for the same model and
  K, greedy, unconstrained, PP-2 cross-device, devacc-enabled, fixed-depth,
  non-replay/non-stream rounds. Cold suffix primes, sampled/grammar paths,
  mismatches, diagnostics, and an odd leftover call the serial path verbatim.
- Increment 1 uses two scoped host call stacks around the existing monolith.
  That gives every round-local allocation one natural copy per session without
  moving it into `SpecSession`. A small coordinator admits primary work in the
  fixed order `draft A -> draft B -> verify enqueue A -> verify enqueue B ->
  accept/commit A -> accept/commit B`; a finishing lane releases its peer to a
  serial tail rather than deadlocking on unequal accepted-token counts.
- The verify funnel gains an internal pipelined mode: A prewarms both boundary
  slots and performs the interval reverse fence; both calls use the forced
  alternating PP slots; B skips the second reverse fence. Ordinary callers keep
  the existing fence plus ordinary slot-selection path byte-for-byte.
- The paired worker flushes each returned burst after the joint engine call.
  This changes only event timing behind `MEMRA_SPEC_PIPE=1`; token ids and all
  session accounting use the existing worker commit logic. Round-cadence SSE and
  admission-yield callbacks remain on the serial fallback in increment 1.

## Log

- 2026-08-10: lane started at 3f8ca2ef on lane/specmech (worktree wt-specmech).
- 2026-08-10: design read pass 1 — round owner, publish/fence ordering laws, the
  increment-1 overlap shape and its ~25-30% bound, scratch-invariant analysis.
- 2026-08-10: design read pass 2 — closed the devacc/commit/refresh ordering,
  worker admission and pairing point, reduced-matrix fallback, and scoped-call
  implementation shape. Design complete; run-spec harness remains a gate-read,
  not an implementation dependency.
- 2026-08-10: increment 1 implemented in three compile-able commits: private PP
  verify pipeline seam, scoped two-call phase coordinator, and warm-session
  worker pairing. The PP-2 K=0 policy is unchanged; default-off and every
  unsupported shape use the original serial path.
- 2026-08-10: local gates green at `74eae8da`: `cargo check` PASS and
  `cargo test --workspace` PASS (all runnable tests; the two CUDA-only tests
  remained explicitly ignored by their harnesses).
- 2026-08-10: Step35 serving-target alignment and the cold affinity-prime
  boundary were made authoritative for spec. Final box1 source `4e0a8ce25`
  passed run-spec K=1..8 plus plain/serial/pipe b1fix byte identity at golden
  SHA-256 `21b8293f...`.
- 2026-08-10: the first perf attempt captured a post-drain
  `CUDA_ERROR_DEINITIALIZED` pending-flush failure. Main now joins the GPU worker
  before CUDA teardown (`6b7acac8` local); the direct regression and all 37
  final battery shutdowns are clean.
- 2026-08-10: final local `cargo check` and `cargo test --workspace` PASS after
  the shutdown fix. Box1 one-lock N=5 medians: c=2 plain 114.878, serial spec
  53.544, pipeline 53.691 tok/s; c=1 fallback neutral; c=4 remains K=0.
- 2026-08-10: RESULTS complete. HOLD: pipeline is +0.27% versus serial but
  -53.26% versus plain; perfect overlap of all measured primary work projects
  only 55.195 tok/s, so `MEMRA_SPEC_PIPE` stays default OFF.
