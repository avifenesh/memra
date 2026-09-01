# glm5_next worker spec routing (lane/glm5-spec-routing, 2026-08-30)

The last engine seam on the glm5 high-performance critical path: a SERVED request can now
reach the already-built MTP-draft + T-parallel-verify loop. Everything below the worker was
gated green before this lane (mtp-draft-20260830, tparallel-verify-20260830); nothing routed
into it — `mtp_spec_capable` failed closed because the MTP_SPEC manifest carries neither the
HyperConnections topology nor the KDA/MLA/kpool operation classes. This lane removes that
refusal DELIBERATELY, for exactly the glm5_next class, and leaves it standing for everything
else.

Base: origin/lane/glm53-flash-bringup @ cc718b988.

## What was built

### 1. Capability — a SECOND manifest, not a widened first one

`crates/memra-gguf/src/execution_manifest.rs`: `GLM5_SPEC` KernelManifest ("glm5-spec") +
`RewriteSurface::Glm5Spec` + the `glm5-spec.v1` row in `execution_rewrites`. The support
table names exactly the operation classes the gated walk/draft run: hc residuals +
KDA/MLA(+kpool) mixers + sigmoid-MoE/dense trunk FFN (verify side), serial MLA-mixer MoE MTP
block (draft side). Deliberately absent: Full/SWA/GDN/KvState/Serial-trunk (the qwen/step
classes — they keep MTP_SPEC), CompressedMlaAttention (dsv4 — dspark), Gemma residuals,
draft-side DenseMlp (`mtp_head_forward_mla_cached` refuses Dense FFN by name).

Two-program disjointness is BY MANIFEST, not by luck: MTP_SPEC still reports glm5 plans
unsupported (blocker names HyperConnections), so `mtp_spec_capable` stays false for glm5 and
no plan can satisfy both predicates. Sealed bundles fail closed: `glm5-spec.v1` is now an
ELIGIBLE rewrite for glm5 plans, so `rewrite_allowed(Glm5Spec)` refuses any sealed bundle
that has not banked its receipt (the real-artifact qualification lane's job).

### 2. Engine serve-session — `Glm5SpecSession` (crates/memra-engine/src/glm_spec.rs)

The dspark/gemma session twins' shape: the session owns its trunk cache + MTP plane state +
pending (token, h_seed) pairs + the live anchor + session-continuity Philox counters
(`sctr` device events, `uctr` host accept uniforms via `spec::host_u01`, tag 0xFFFF_FFFE).

- `glm5_spec_session_new(e, prompt, ctx_cap, sampling)`: prime, MTP-plane warm (gap-free
  (token[i+1], h_i) pairs), boundary token (greedy argmax, or `sample_boundary_token`
  through the session stream). Refusals: no hc trunk, no MTP head, prompt < 2, HPOST,
  ppN split, penalized sampling (no penalty arm yet — worker admission keeps those plain).
- `glm5_spec_session_burst(e, sess, target, k, eos)`: rounds of feed-pending -> chain K
  drafts -> `glm5_verify_rows` (one t=K+1 walk) -> accept -> `glm5_verify_rollback` +
  plane reset -> re-seed, until `target` tokens out / EOS / ctx guard. Round-boundary
  invariant: `cache.pos == committed.len()`, token-for-token; the bonus is the live anchor
  (the dspark `last` convention). Engine overshoot (a round commits j+1 atomically) stays
  in the session; the WORKER clamps public emission.
- SAMPLED accept seam (the tparallel LANE's stated follow-up, landed here): rejection
  sampling `u_j * q_j(x_j) < p_j(x_j)` over the verify rows — filtered p/q via the existing
  `filter_stats` / `softmax_gather_filtered` kernels (one batched stats+gather pass over
  the k verify rows), draft chain drawn by `gumbel_perturb_filtered` from the SAME filtered
  draft distribution the q gather reads (exactness of the pairing), full-accept bonus =
  filtered Gumbel from the last verify row, rejection bonus = `residual_sample_filtered`
  (d2t `scatter_trim_logits` arm for trimmed heads). Walk and rollback unchanged — one seam.
- `generate_spec_glm5[_gated]` REIMPLEMENTED on the session (one round machinery, no
  drifting twin); the whole pre-existing tparallel battery re-run green (below), so the
  one-shot contract is byte-preserved.
- FR-Spec trim: unchanged contract (`MEMRA_FRSPEC_TRIM`, d2t remap of every draft pick
  before chain/verify; verify full-vocab).

### 3. Worker route (crates/memra-server/src/worker.rs)

- `glm5_spec_capable(lm)`: hc trunk + loaded MTP head + `MEMRA_GLM5_SPEC=1` + no ppN +
  GLM5_SPEC manifest + `rewrite_allowed(Glm5Spec)`.
- `glm5_route_admits(...)` (pure, unit-gated): capable + serve_spec + K>0 + greedy-or-T>0 +
  NOT penalized (greedy AND sampled: no penalty arm) + NOT constrained + NOT vision + COLD
  only (no restore/resume/park arm in this lane).
- K policy: the SHARED `choose_spec_k` surface (operator pin `MEMRA_SPEC_K`, concurrency
  shed `MEMRA_SPEC_GATE*`, prompt/cache table cold 3 / cached-long 2 / trim 5) +
  `glm5_clamp_spec_k`: **K+1 <= 15 hard bound** (hyper_batch_cap; the shexp decode-exact
  knee at t=16). Default depth = the table's cold-short **K=3** — the kpolicy-20260808
  precedent, adopted as the conservative default, unmeasured on glm5 (the box A/B re-prices).
- Session fields `glm5`/`glm5_on`/`glm5_k`; tick dispatch arm (before gemma/plain);
  phase-(a) filter includes `glm5_on`; cache literal never allocates a plain cache under
  the route; `step_session` refuses a session holding a glm5 session (the dspark
  dispatch-disagreement guard, extended).
- `step_glm5_spec`: the step_dspark_spec twin — lazy prime, one burst per tick
  (MEMRA_SPEC_BURST cap 32), `spec_visible_len` budget clamp + `emit_spec_token_events`
  (one Event::Token per public id, EOS text never streamed), EOS/stop-string/MaxNew/
  ContextFull, usage.spec rounds/drafted/accepted through `finish`.
- RECEIPTS (never-serve-greedy law — engagement grep-able from the server log):
  - boot (hybrid.rs, printed at load, ONLY when MEMRA_GLM5_SPEC=1):
    `[glm5-spec] serve route ARMED: MTP head loaded; draft head TRIMMED to N rows
    (FR-Spec d2t engaged)` | `... draft head FULL target vocab (no FR-Spec trim)` |
    `[glm5-spec] MEMRA_GLM5_SPEC=1 but no MTP head loaded (set MEMRA_GLM5_MTP=1) — route
    stays fail-closed, plain serving`. Flag off = ZERO `[glm5-spec]` lines.
  - per-request: `[glm5-spec] route=spec|plain K=.. model=.. tenant=.. prompt=.. wave=..
    sampled=.. penalized=.. cold=..` (admission) and
    `[glm5-acc] ctx={} burst={a}/{d} cum={A}/{D}={acc}` per burst (the q38 `[dspark-acc]`
    shape).

### 4. Flags

NO new flag. `MEMRA_GLM5_SPEC` stays the ONE master (default OFF; off = byte-identical
pre-lane serving; one flag flips the whole route). FLAGS.md rows updated in this lane:
`MEMRA_GLM5_SPEC` (rewritten for the serving route: capability, K policy + clamp,
exclusions, receipts, flip condition) and `MEMRA_GLM5_MTP` (stale "worker never routes to
it" clause corrected). `tools/check-flags.sh`: green, no uncovered names.

## Gate table (all rig 5090, TF32 off, flock-serialized, 2026-08-30)

| gate | result |
|---|---|
| memra-gguf `glm5_spec_class_matrix` (4 tests): glm5+MTP supported + `glm5-spec.v1` eligible; glm5 w/o MTP blocked (blocker = DraftPlan); MTP_SPEC still blocks glm5 (blocker names HyperConnections) AND GLM5_SPEC blocks qwen35 (blocker names SerialResidual) — no plan claims both programs; dsv4 blocked (CompressedMlaAttention) + dense qwen3 blocked | PASS 4/4 |
| worker `glm5_route_class_matrix_is_exact`: green rows K=1/3/14 + EVERY exclusion flipped alone refuses (off-flag/capable, serve_spec, K=0, temp, penalized, constrained, vision, warm) | PASS |
| worker `glm5_spec_k_clamp_holds_the_verify_knee`: 14->14, 15->14, 64->14; all policy depths pass through | PASS |
| worker `glm5_route_wiring_is_live_in_comment_stripped_source` (dispatch arm + order, phase-a filter, glm5_route_admits + clamp invocations, cache no-alloc arm, `[glm5-acc]` emission inside step_glm5_spec, burst invocation, emit_spec_token_events, capability anchors) | PASS; RED-PROVEN: mutating the step's `[glm5-acc]` literal (probe preserved) fails the gate by name |
| worker `a_plain_step_refuses_a_session_holding_a_glm5_session` (guard before model bind) | PASS |
| pre-existing `glm5_tparallel_verify_gpu` (7 tests: walk bit-identity, accept-j-then-continue, stale-KDA red, pool-key red, e2e K=1..7 + forced arms, rollback-disabled red, FR-Spec battery) RE-RUN over the session-based reimplementation | PASS 7/7, 2.45s |
| pre-existing `glm5_mtp_head_gpu` | PASS 5/5 |
| NEW `glm5_spec_session_gpu` gate 1+2: served bursts (target 3 — worker cadence in miniature), greedy tape byte-identical to plain decode K=1..7, burst-boundary invariants (`pos()==committed.len()`, committed == prompt + tape minus live anchor) at every boundary | PASS |
| gate 3: forced-rejection j-sweep (j = round % K) through the SERVED burst API, 2-token bursts so every partial accept continues across a boundary — tape byte-identical | PASS |
| gate 4 sampled twin: pinned seed reproducible; **burst-split invariant** (3-token bursts == one whole-budget burst — Philox counters live on the session); different seed diverges (the seam is alive, not secretly greedy) | PASS |
| gate 5 EOS: session finishes mid-burst, public prefix through EOS matches plain, overshoot bounded by the final round, post-EOS burst empty | PASS |
| gate 6 RED: rollback disabled + forced rejections through the served burst — tape diverges (or loud failure), never byte-identical-and-green | bites |
| gate 7 receipt log (subprocess-captured stderr, per-arm fresh env): ARMED boot line (FULL + TRIMMED-to-32-rows variants), fail-closed warn (flag on / head off), **red arm: flag off = zero `[glm5-spec]` lines** with the head loaded or not | PASS |
| `cargo test -p memra-server` (472) / `-p memra-gguf --lib` (183) / `-p memra-engine --lib` (252) | PASS, 0 failed |
| `cargo fmt` + workspace `cargo clippy --all-targets` | zero warnings (three pre-existing reds fixed in-lane: parallel.rs lib-test `vision_glm5` initializer, tparallel-test range-loop/collapsible-if, vision-upstream doc/cast) |
| `tools/local-ci.sh --perf` | run 1 (~03:21Z): correctness ALL GREEN; qwen9b-plain-short recorded 127.34 tok/s window_clean=false (co-resident: another memra lane's gate binary + owner services hermes/colbert), tripping the -8.37% drift tripwire. Run 2 (~04:1xZ): serve-smoke cache-metering fanout flaked 0-hit/6-miss while that other lane's gate ran co-resident; standalone serve-smoke re-run immediately after: 0 failed. Run 3 (~04:2xZ, other lane gone): **ALL GREEN, perf stage 0 fail 0 warn, qwen9b 139.32 tok/s [OK]** (above the 138.97 median even in a still-dirty window — hermes 350 MiB + colbert 1390 MiB idle contexts persist on this rig). Settled per the tripwire's own protocol: the run-1 drop was the contended window, not the diff (which adds nothing to the qwen plain path — glm5 arms are flag-gated and plan-keyed; this cell runs neither). |

Repro:
```
cargo test -p memra-gguf --lib glm5_spec_class_matrix
cargo test -p memra-server glm5
NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
  cargo test -p memra-engine --test glm5_tparallel_verify_gpu -- --ignored --test-threads=1
NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
  cargo test -p memra-engine --test glm5_spec_session_gpu -- --ignored --test-threads=1
```

## Deliberately NOT in this lane (each a named follow-up)

- **ppN twin of the verify walk** — `glm5_verify_rows` still refuses `pp_cuts` by name and
  `glm5_spec_capable` requires a single-engine placement; the 2-card serving shape cannot
  run spec until the `[t, streams, n_embd]` stage-split twin lands gated (the tparallel
  LANE's box-A/B prerequisite 1).
- Penalty arm for the sampled accept walk (penalized requests serve plain, loudly named).
- Session parking/resume/affinity/prefix-restore for glm5 sessions (cold-only route);
  spec-gate DEMOTION (under load, NEW glm5 requests take K=0 plain via the shared
  concurrency shed; live sessions run to completion).
- Admission memory pricing of the K-column KDA rollback transient (~0.95 GiB at K=7 on the
  real artifact): admission still prices the session as plain. Must be priced before any
  high-concurrency flip on the real card.
- Real-artifact receipts of any kind (fixture-scale only). The flip condition is unchanged
  from tparallel-verify-20260830 §Box A/B: ppN twin -> real-artifact accept/rollback
  battery -> acceptance measurement (+ trim A/B with owner-minted ranks) -> interleaved x5
  A/B on the serving shape with the vendor-default sampled twin + spec-engagement receipts
  + the 8-turn cache-on twin.

## Remaining preconditions for the serving-shape A/B (stated for the coordinator)

1. ppN verify-walk twin + MTP head under the split (engine lane, gate: same accept-j/e2e
   batteries at stages=2, both split arms) — the only remaining ENGINE gap.
2. glm5 ranks mint (owner CPU lane; SXC pools through GLM's tokenizer, q38 format).
3. Real-artifact battery + acceptance measurement + A/B per the tparallel LANE §Box A/B —
   all on box time through the coordinator; this lane's route is the machinery they drive
   (`MEMRA_GLM5_MTP=1 MEMRA_GLM5_SPEC=1`, research door, never a sealed bundle).
