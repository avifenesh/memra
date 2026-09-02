# b200-spec-ttft-20260902: where the glm5 spec route's first token waits

Lane: `lane/b200-spec-ttft-20260902`, branched from `lane/glm5-b200-int2-20260902` at
`8c31be2f4`. Investigation plus instrumentation, no GPU in this lane; the box runs the
instrumented binary (invocation in section 4).

## 1. The measurement this lane explains

2x B200 pair, GLM-5.3-Flash NVFP4, PP2, resident posture, prefix cache OFF
(`MEMRA_PREFIX_CACHE_MB=0`), the same 66-token prompt, vendor-default sampling, 5 warm reps
per boot, six fresh boots per arm:

| arm | TTFT |
|---|---|
| plain route | 0.188-0.210 s |
| DFlash2 spec route (`MEMRA_GLM5_SPEC=1 MEMRA_GLM5_DFLASH=<dflash2> MEMRA_SPEC_PMIN=0.7 MEMRA_GLM5_VERIFY_BATCH=1 MEMRA_SPEC_GATE_LOW=2 MEMRA_SPEC_GATE_HIGH=4`; log `[glm5-spec] route=spec K=3 ... prompt=66 wave=1 sampled=1 penalized=0 cold=1 restored=0`; draft source dflash2 @ b33c0347, draft head = FULL target vocab, native MTP head NOT loaded) | 0.633-0.807 s |

The spec route adds 0.45-0.60 s before the first streamed token on a prompt whose target
prime costs about 0.19 s. A decode token costs about 13 ms on this route.

## 2. The path, request admission to the first streamed token

Worker (`crates/memra-server/src/worker.rs`):

1. Admission decides the route (`[glm5-spec] route=` line, worker.rs:19549) and, for a
   cold glm5 session, allocates NO plain cache (worker.rs:19587, `let cache = if dspark_on
   || (glm5_on && ...`): the session owns its cache, born inside the first spec tick.
2. The tick dispatch (`else if active[i].glm5_on`, worker.rs:14662) calls
   `step_glm5_spec` (worker.rs:22512). Turn 1 drains the whole prompt from the prefill
   queue and calls `glm5_spec_session_new` (worker.rs:22567 region) between the TTFT
   trace's `mark_prime_start` / `mark_prime_end`.
3. The SAME tick then runs ONE `glm5_spec_session_burst` with
   `burst_target = min(request_room, MEMRA_SPEC_BURST = 32)` (worker.rs:22604 read,
   worker.rs:22672 call) and only AFTER the burst returns computes `public_len`
   (worker.rs:22701) and emits every `Event::Token` through `emit_spec_token_events`.

Engine (`crates/memra-engine/src/glm_spec.rs`), `glm5_spec_session_new` (glm_spec.rs:1598):

4. `pp::new_cache_planned` (glm_spec.rs:1769): the trunk cache at `ctx_cap`
   (= prompt + max_new + 8 for a finite request; the whole server window when
   `max_tokens` is omitted).
5. `HcTapSink::new` (glm_spec.rs:1785) arms a HOST-staged tap sink, then
   `prime_cache` (glm_spec.rs:1790, `hybrid_forward.rs:3439`) walks the prompt. Inside the
   walk `glm5_hc_tap` (glm_spec.rs:895) does one SYNCHRONOUS `dtoh` per tapped layer per
   prime chunk (glm_spec.rs:940): five stalls per chunk, 5 x t x 4096 f32 bytes.
6. Prefix boundary capture (glm_spec.rs:1914): skipped, `MEMRA_GLM5_SPEC_PREFIX` is unset on
   the box (glm_spec.rs:234).
7. The boundary token: `sample_boundary_token` (glm_spec.rs:1811, `spec.rs:1886`) uploads
   the prime's host logits (154880 x f32 = 620 KB), filtered Gumbel, argmax, one readback.
   THIS IS THE FIRST TOKEN. It is known here, about 0.19 s in.
8. `DflashKv::new` (glm_spec.rs:1833, `dflash.rs:2532`): the drafter's ctx KV,
   2 x 5 layers x (ctx_cap + 8) rows x n_kv x head_dim x f32, uninit (pool alloc). The
   prompt's tap rows ride `pending` (host Vec) into round 1; nothing else is warmed here.
   No per-request graph capture exists on this route (the verify walk is eager).

`glm5_spec_session_burst_inner` (glm_spec.rs:2301):

9. The anchor is pushed to `out` first (glm_spec.rs:2358) and the loop
   `while out.len() < target && !sess.done` (glm_spec.rs:2368) runs rounds until 32 tokens
   are out. The anchor is NOT handed to the worker until the loop ends.
10. Round 1 (`glm5_spec_round`, glm_spec.rs:2415 -> `glm5_dflash_round_drafts`,
    glm_spec.rs:2868): the drafter prime, i.e. the whole prompt's feature rows go back
    HOST -> DEVICE in 256-row chunks (`eh.htod`, glm_spec.rs:2910, synchronous pageable
    copy), `ctx_features` (fc over 5 x 4096 -> 4096) and `ingest_ctx` (`dflash.rs:2558`:
    per layer k/v projections, k-norm, rope, two copies) (glm_spec.rs:2913). Then the
    block forward (`forward_round`, glm_spec.rs:2936), the lm_head over the 7 mask-fill
    rows through the FULL 154880-row target head (glm_spec.rs:2964), the sampled selector
    walk (host readbacks), then the t=K+1 verify walk (glm_spec.rs:2616), the rejection
    accept (per-slot readbacks), rollback, tap drain (`glm5_tap_drain`, glm_spec.rs:954,
    five DtoHs of the kept rows) and the round's committed tokens are appended to `out`.
11. Rounds 2..N repeat step 10 minus the ingest (only the kept rows are re-ingested), each
    round committing j+1 tokens, until `out.len() >= 32`.
12. Back in the worker: `spec_visible_len`, then `emit_spec_token_events` sends the anchor
    and the other 31+ tokens in one go; the HTTP side serializes the first SSE body byte
    (`mark_first_sse_byte`). TTFT is stamped here.

So the answer to the question in the task: yes, the first token IS withheld. Not until the
first verify completes, but until the ENTIRE first burst completes.

## 3. Candidate costs, ranked

Estimates are derived from the box's own numbers (0.19 s prime, about 13 ms per spec-route
token) and the artifact geometry (hidden 4096, 5-layer drafter, block 8, taps [5,14,24,33,42],
n_vocab 154880). The timer in section 4 replaces every estimate with a measured bucket.

| rank | cost | where | estimate on the 66-token probe | scales with |
|---|---|---|---|---|
| 1 | **First burst withheld**: the anchor is known at prime end but the worker emits nothing until the 32-token burst returns | worker.rs:22672 (burst call before any emit), glm_spec.rs:2358 + 2368 (anchor pushed, loop to `target`) | 31 extra tokens x ~13 ms = **~0.40 s** (0.35-0.50 s across accept-rate spread; 8-11 rounds at K=3) | `MEMRA_SPEC_BURST` and the per-token decode cost; NOT prompt length |
| 2 | Round-1 drafter prime: prompt tap rows HtoD (synchronous pageable) + fc + 5-layer k/v ingest | glm_spec.rs:2910-2913, dflash.rs:2558 | 5.4 MB HtoD + ~2 ms compute = **3-6 ms** | linear in prompt tokens (one 256-row chunk per 256 tokens: about 21 MB HtoD + ingest per chunk); the only prompt-linear cost after the target prime |
| 3 | Host-staged tap DtoH inside the prime: five synchronous stream drains per prime chunk, 5 x t x 16 KB each | glm_spec.rs:940 (via `glm5_hc_tap`, called from the prime walk) | 5.4 MB DtoH + 5 stalls = **1-3 ms** | linear in prompt tokens (bytes) and prime chunks (stalls); on PP2 a stall on a stage-0 tap layer sits inside the walk's critical path |
| 4 | Round 1 fixed cost (block forward, full-vocab head over 7 rows, selector readbacks, verify t=4, accept readbacks, rollback, tap drain) | glm_spec.rs:2936, 2964, 2616, 954 | one round: **~25-40 ms** (inside the 13 ms/token figure; listed because it is on the first-token path even with the door on) | K, `MEMRA_SPEC_PMIN` truncation |
| 5 | Boundary token draw: 620 KB logits HtoD + filter/Gumbel/argmax + readback | glm_spec.rs:1811, spec.rs:1886 | **0.5-1.5 ms** | n_vocab |
| 6 | Drafter KV allocation, uninit | glm_spec.rs:1833, dflash.rs:2532 | pool-served: **<1 ms**; fresh device alloc: ms-class | `ctx_cap` (2 x 5 x rows x n_kv x head_dim x 4 B: at the server window a `max_tokens`-omitted request allocates GB-class drafter KV per request; the timer's `draft_alloc` bucket says whether the pool absorbs it) |
| 7 | Trunk cache allocation | glm_spec.rs:1769 | same allocation the plain route makes at admission; no delta | `ctx_cap` |
| 8 | Prefix boundary capture | glm_spec.rs:1914 | **0** on the box (door shut) | only with `MEMRA_GLM5_SPEC_PREFIX=1` |
| 9 | Per-request warmup, restore, reseed, graph capture | (none found) | 0 | n/a |

Rank 1 accounts for the delta on its own: 0.19 s prime + ~0.40 s burst + ~10 ms of ranks
2-6 lands at ~0.60 s, inside the measured 0.63-0.81 s once the sampled accept-rate spread
across reps (8 vs 11 rounds) is included. The rest is single-digit milliseconds today and
grows only with prompt length (ranks 2-3) or with `ctx_cap` (rank 6).

Why this route has it and the qwen MTP route does not: the qwen spec burst got the
sse-cadence `on_commit` hook on 2026-08-05 (`spec.rs:11490`, worker.rs `flush_cb` in the
`s.spec` arm) and the admission-yield verdict on 2026-08-06 (`MEMRA_ADMIT_YIELD`); the glm5
session burst (and the dspark one, `dflash.rs:4728`, same shape) were written burst-cadence.
The FLAGS row for `MEMRA_SPEC_BURST` records exactly this failure class on the qwen route:
"at 128: 2 chunks/1.15s first text vs 8 chunks/0.41s at 32", fixed by round-cadence flush.

## 4. The timer: `MEMRA_SPEC_PROF=1`

One `[spec-prof]` line per served glm5 spec request, printed by `step_glm5_spec` after the
first burst's emission. Engine buckets ride the session (`Glm5SpecSession.prof`,
`spec_phase::SpecFirstTokenProf`), the worker adds its marks. Phase boundaries drain the
stream (the `SpecPhaseNs::clock` contract), so the traced burst is slightly slower than the
untraced one: attribute with this line, claim TTFT from an untraced boot.

Fields (ms): `since_admit_ms` (Session creation to the first spec tick), `session_ms`
(`glm5_spec_session_new` wall) split into `cache_alloc`, `prime` (target prime incl. the
tap DtoHs and the logits readback), `capture`, `anchor`, `draft_alloc`; `round1_ms:`
`draft_prime` (the drafter ingest of the whole prompt), `draft`, `verify`, `accept`,
`roll`, `maint`, `tokens`; `burst1: wall_ms hook_ms rounds tokens` (`hook_ms` = time inside
the round-cadence commit hook, detext + sends, 0 with the eager door off; under the
default-ON door the hook runs inside the burst window, so `wall_ms - hook_ms` is the
engine-only burst wall on either arm); `first_emit_ms` (first `Event::Token` send, from
step entry); `step_ms`.

Box invocation (the spec-route boot from section 1, plus the timer; one boot, the same
66-token prompt, vendor-default sampling, `stream: true`):

```
MEMRA_SPEC_PROF=1 MEMRA_GLM5_SPEC=1 MEMRA_GLM5_DFLASH=<dflash2 dir> MEMRA_SPEC_PMIN=0.7 \
  MEMRA_GLM5_VERIFY_BATCH=1 MEMRA_SPEC_GATE_LOW=2 MEMRA_SPEC_GATE_HIGH=4 \
  MEMRA_PREFIX_CACHE_MB=0 <the pair's serve command> 2>&1 | tee spec-prof.log
grep -E '^\[spec-prof\]|^\[glm5-spec\] route=' spec-prof.log
```

Expected shape on the current binary (door OFF): `first_emit_ms` ~= `session_ms` +
`burst1.wall_ms`, with `burst1.wall_ms` ~= 0.40 s and `prime` ~= 0.19 s. Add
`MEMRA_TTFT_TRACE=1` to see the HTTP-side `sse_handoff_ms` beside it.

## 5. The door: `MEMRA_SPEC_FIRST_TOKEN_EAGER` (default ON since the section 8 receipts; `=0` = rollback)

What it does: `step_glm5_spec` drives `glm5_spec_session_burst_streamed`, which calls a
hook with every committed slice as it lands (the prime's anchor first, ALONE; then each
round's j accepted drafts + bonus). The worker's hook emits the slice through the same
`emit_spec_token_events` machinery immediately, so the first `Event::Token` goes out right
after `glm5_spec_session_new` returns instead of after the 32-token burst. The post-burst
bookkeeping (sampler, generated, fed, stop reasons, budget clamp) is shared by both arms.

Numeric class: byte-identical token stream ON/OFF, greedy and sampled. The burst loop's
control flow never reads the hook (it returns nothing), and the hook's slices are disjoint,
in-order views of the same `out` vector the un-hooked burst returns, so the worker emits the
same ids in the same order; only event timing moves. Pinned by the rig gate
`gpu_dflash_streamed_burst_slices_concat_to_the_unhooked_burst`
(`crates/memra-engine/tests/glm5_dflash_session_gpu.rs`, greedy + pinned-seed sampled: slices
concatenate to the burst, first slice == anchor, tokens AND counters equal the un-hooked
twin on a fresh session, greedy == plain decode).

Expected effect: spec TTFT moves from prime + burst (~0.63-0.81 s) to prime + anchor draw
(~0.20 s), i.e. within a few ms of the plain route; per-token cadence becomes one round
(~30-40 ms at K=3) instead of one burst.

Step-OOM park interplay (PR #93 review finding, fixed in the same PR): the step-OOM
park path (`park_requeue`) replays the prompt on the SAME stream and is legal only while
nothing reached the client. Its guard read `generated.is_empty()`, which the eager arm
fills only after the burst, so a CUDA OOM in a later round of the first burst would have
parked and re-sent the streamed prefix. Now both flush hooks (this route's and the qwen
sse-cadence hook, which had the same latent gap) advance `tokens_emitted` at the send,
and the guard is the pure `step_oom_parkable(generated_len, tokens_emitted, retries, max)`:
post-flush OOMs take the honest error (terminal error event, no replay); pre-flush OOMs
still park. Unit truth table + comment-stripped wiring test pin it. The byte-identity
claim is untouched: the marker is bookkeeping, the emitted ids are the same.

Rollback: `MEMRA_SPEC_FIRST_TOKEN_EAGER=0` (the pre-lane literal is the door-off arm, kept
verbatim, including the wiring test's `glm5_spec_session_burst(engine, sess, burst_target, k,`
literal). The default flipped ON on the section 8 receipts (a cadence change, no numeric
change).

## 6. Follow-ups this lane names, not does

- Admission-yield twin for glm5 (`MEMRA_ADMIT_YIELD` semantics): end the burst at a round
  boundary when `PENDING_ADMITS > 0`. Contended first-text on this route still scales with
  `MEMRA_SPEC_BURST`. The hook would grow a continue-verdict, the qwen shape.
- The dspark route (`step_dspark_spec`, `dspark_spec_session_burst`) has the same
  withheld-burst shape; the same door name should cover it once its gate runs.
- Device-resident tap sink for the prime (ranks 2-3): the prompt's tap rows go
  device -> host -> device today (host staging chosen for ppN placement invariance,
  `research/glm53-flash-bringup-20260827/dflash-draft-src-20260830/LANE.md` section 2).
  A device-staged prime sink plus an on-device ingest removes both copies and the five
  per-chunk stalls; matters at long prompts, not at 66 tokens.
- `DflashKv::new` at the server window for `max_tokens`-omitted requests: GB-class uninit
  allocation per request; check the `draft_alloc` bucket and the admission charge.

## 7. Gates run in this lane (no GPU here)

- `cargo fmt --all -- --check`, `cargo clippy --release --all-targets -- -D warnings` at
  `MEMRA_CUDA_ARCH=120a` and `100a`, `tools/check-flags.sh`,
  `cargo test -p memra-server --lib glm5_route_wiring_is_live_in_comment_stripped_source`
  (the door's two arms are pinned as invocations), `cargo test -p memra-engine --lib
  spec_phase` (results in section 8 once run).
- Rig gate to run on the box or the 5090 before the default flips:
  `NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock cargo test -p memra-engine --test
  glm5_dflash_session_gpu gpu_dflash_streamed_burst -- --ignored --test-threads=1`.

## 8. Box receipts (2026-09-02): the door measured, the default flipped

2x B200 pair, GLM-5.3-Flash NVFP4, DFlash2 K=3, vendor-default sampling, the 66-token
prompt, 5 warm reps per boot, three interleaved fresh-boot pairs (door OFF -> door ON):

| prompt | TTFT, door OFF (s) | TTFT, door ON (s) | plain route, same prompt |
|---|---|---|---|
| code | 0.674 / 0.647 / 0.653 | 0.187 / 0.226 / 0.225 | 0.19-0.21 s |
| prose | 0.785 / 0.740 / 0.804 | 0.185 / 0.237 / 0.238 | 0.19-0.21 s |

Wall-inclusive tok/s unchanged: code 72.1 / 73.2 / 73.6 -> 73.2 / 73.1 / 73.3, prose
51.5 / 52.2 / 51.6 -> 51.8 / 51.8 / 50.5. The 128-token greedy digits tape sha
`9437b599f6b9d2a9` is identical on all six boots.

`MEMRA_SPEC_PROF=1` attribution on the same pair with the door OFF: prime 185 ms,
burst1 432-475 ms (9-10 rounds, 32-35 tokens), first_emit 619-662 ms. Rank 1 of section 3
measured as predicted; the remaining buckets are the single-digit-ms class.

Rig gate `gpu_dflash_streamed_burst_slices_concat_to_the_unhooked_burst`: PASS on the 5090.

Raw: darklanes `research/glm5-b200-20260902/floor/raw/ab-spec-{w8s,eager}-{1,2,3}/` (the
A/B) and `research/glm5-b200-20260902/box/` (the spec-prof lines).

Decision: default ON for the glm5 spec route, `=0` the rollback seam (FLAGS.md row). The
dspark route's identical shape stays untouched, a named follow-up.
