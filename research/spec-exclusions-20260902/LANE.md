# lane/spec-exclusions-20260902: every request class the glm5 DFlash2 spec route sends plain

Owner direction (2026-09-02): "Penalty / vision / warm / TP-restore currently hard-plain, all
glm5": every request class the glm5 spec route excludes silently loses the 1.48x. This lane
(1) enumerates every exclusion predicate with its anchor, what it protects, and which real
traffic hits it; (2) lands spec support for the two that were pure "unimplemented" (penalties,
the tail-less warm carrier) behind two default-OFF doors; (3) leaves vision and TP as named
follow-ups with their exact blocker. No GPU on this lane beyond the rig 5090 exactness gates.

Branch `lane/spec-exclusions-20260902` off `origin/main` 50f409fa8 (PR #100 merged). Written
against PR #93 (merged: streamed first token) and beside PR #101 (open: `MEMRA_SPEC_MAX_PROMPT`;
its `prompt-above-spec-max` cap composes AFTER every predicate below, so it is listed as pending,
not anchored).

Line numbers are this branch's tree (`crates/memra-server/src/worker.rs` = W,
`crates/memra-engine/src/glm_spec.rs` = G, `crates/memra-engine/src/dflash.rs` = D).

## 1. The exclusion table

The route decision is `glm5_route_admits` (W:16857), a PURE conjunction; the reason an
operator reads is `glm5_route_decline_reason` (W:16795), the same matrix read out in order
(capability -> operator door -> request shape -> session shape), pinned equal by
`the_route_reason_agrees_with_the_route`. Both are called from `admit` (W:19737 / W:19759) and
the verdict lands on `[glm5-spec] route=spec|plain K=.. ... penalized=.. cold=.. restored=..
reason=..`.

| # | `reason=` token | predicate (anchor) | protects | who hits it in real traffic | this lane |
|---|---|---|---|---|---|
| 1 | `model-not-spec-capable` | `glm5_spec_capable` (W:16682): hc trunk + a loaded draft source (`MEMRA_GLM5_MTP=1` or `MEMRA_GLM5_DFLASH`) + `MEMRA_GLM5_SPEC=1` + placement (single engine, or a ppN split at stages 2..3 with a qualified pipeline rewrite; `glm5_sharded_placement_admits` W:16718 refuses any `MEMRA_STEP_TP`/`MEMRA_STEP_EP` composition) + `RewriteSurface::Glm5Spec` receipt + the `GLM5_SPEC` manifest class | correctness: unqualified placements and unreceipted rewrites fail closed | every request on a box whose recipe lacks the flags or whose bundle lacks the receipt; every request on a step-TP/EP composition | unchanged; TP is follow-up B below |
| 2 | `spec-serving-off` | `serve_spec` (W:17796): `!confidence_trace_enabled() && !pp_host_bounce_active() && !vision_req && !capture_req && MEMRA_SERVE_SPEC != 0 && peer_probe_allows_spec` | correctness (vision overlay, capture prime, pp bounce) and the operator door | vision requests land here FIRST (before #7); embeddings/rerank captures; the peer-probe shed | unchanged |
| 3 | `k-shed-to-zero` | `glm5_k == 0` (W:19680): `choose_spec_k` (W:4803) returns K=0 on `MEMRA_SPEC_K=0` (OperatorPin) or `MEMRA_SPEC_GATE` concurrency/placement shed (`projected_wave > thresholds.low`); then `glm5_clamp_spec_k` (W:16892) and the DFlash2 block clamp | throughput policy (the c>2 shed is measured protective: 3way cell 5) | every request admitted while the projected wave exceeds the low threshold: the load shape | unchanged (policy, not correctness); NOTE it is the main producer of tail-less prefix entries that #10 then hits (a K-shed turn publishes no drafter tail) |
| 4 | `sampler-not-spec-eligible` | `temp_ok` = `sampler.is_greedy() \|\| sampler.temperature() > 0.0` | shape sanity (unreachable: a Sampler is one or the other) | nobody | unchanged |
| 5 | `penalized` | `glm5_penalty_plain` (W:18682) = (`greedy_penalized` W:17811 \|\| `spec_sampling_for(..).pen_on()`) `&& !MEMRA_SPEC_PENALTY`; engine twin `glm5_penalty_admit` (G, session_new + from_restored) refuses a penalized session while the door is dark | correctness: the accept walk argmaxed / gathered RAW target rows, so admitting would silently drop the request's penalties (the failure class dspark named on 2026-08-21) | THE COMMON ONE: every agent harness whose vendor defaults ship `presence_penalty`/`frequency_penalty`/`repetition_penalty`, and every greedy request with any penalty (excluded on every spec route pre-lane) | ADMITTED under `MEMRA_SPEC_PENALTY=1` (section 2); dark = pre-lane verbatim |
| 6 | `constrained` | `constraint.is_some()` (W:17818): `response_format` json_schema / grammar, prepared off-tick | unimplemented: the grammar mask hooks (`mask_logits`, `consume`) exist only on the qwen spec program (`spec.rs`), the glm5 round has no masking seam on the draft or verify rows | structured-output callers (json_schema), the post-think grammar shape | follow-up C: port the qwen route's draft-side `MEMRA_DRAFT_MASK` + verify-row masking into `glm5_spec_round` (mask the K+1 target rows before accept; mask the DFlash2 selector's candidate set) |
| 7 | `vision` | `vision_state.is_some()` (W:17325, `req.glm5_images` non-empty); also shadowed by #2 (`!vision_req` inside `serve_spec`), so a vision request reads `spec-serving-off` today | unimplemented: `glm5_spec_session_new` primes through `prime_cache` (G:1960 area) with no embedding-overlay seam; the plain prefill tick carries the `VisionState` overlay (`hybrid_forward` overlay residency) | image requests only (glm5 vision live since 2026-08-30); text traffic never | follow-up A below (exact blocker) |
| 8 | `prompt-below-prime-floor` | `prompt.len() >= 2` (memra#9) | correctness: the MTP plane warms on (token[i+1], hidden[i]) pairs, `generate_spec_glm5` refuses a 1-token prompt past admission (a 500 otherwise) | a one-word prompt with no template (rare on chat: the template alone is > 2 tokens) | unchanged |
| 9 | `warm-session-no-carrier` | `!(glm5_cold \|\| glm5_restored_carrier)` (W:19717, W:19729) with `carrier_detail == None`: the session is warm (a live spec session of any family, a continuation resume, a legacy `kv-reuse` pool hit that is NOT a prefix-cache hit, `dspark_on`, `gspec_k > 0`, a spec-K replay) but never entered the glm5 carrier probe (W:18699 requires `prefix_hit`) | correctness: the session owns its cache and has no resume arm; a legacy-pool carrier was never proven at `cache.pos == fed.len()` under `from_restored` | conversation continuations served through the legacy continuation pool (`[worker] kv-reuse: N of M prompt tokens resumed`) rather than the prefix cache | follow-up D: the legacy carrier is the same shape as a tail-less prefix hit (restored trunk at `fed.len()`, suffix to prime, cold drafter); it needs the probe's `prefix_hit` term widened and the park invariant `cache.pos == fed.len()` proven for glm5 plain parks (`compact_parked_plain_cache` interplay) |
| 10a | `prefix-restore-off` | `glm5_carrier_admits` (W:16758) arm 1: `MEMRA_GLM5_SPEC_PREFIX` x `MEMRA_PREFIX_LATENT` unset | the restore door (`from_restored` is unmeasured on the box: its FLAGS row pre-states the flip condition) | EVERY multi-turn hit in production today: both flags are default OFF, so warm = plain by posture, not by mechanism | unchanged: the box battery flips it (restored-vs-cold byte identity on the continuation, the 8-turn twin); `MEMRA_SPEC_WARM` lives UNDER it |
| 10b | `full-cover-hit` | arm 2: `suffix_len == 0 && !MEMRA_GLM5_SPEC_FULLCOVER` | its own door (memra#74; gate 13) | the repeated-prompt shape (retries, eval harnesses) | unchanged |
| 10c | `full-cover-hit-no-boundary-logits` | arm 3: full cover armed, the entry's boundary row absent or not `n_vocab` wide | correctness: no anchor row to start a round from | a capture-side defect only | unchanged |
| 10d | `suffix-below-prime-floor` | arm 4: `0 < suffix_len < PRIME_MIN_T` (16) | the two-programs door: a sub-floor suffix under a spec session would ride tokenwise `decode_step` inside a prime program (the `eager_mono && carried` class) | a very short new user turn on a hit (the template usually pushes a turn past 16 tokens) | unchanged |
| 10e | `no-drafter-tail` | W:18783: the entry has no `dspark_draft` plane (published by a PLAIN glm5 session: a K-shed turn under load, a constrained/vision turn, a penalized turn with #5 dark, or a session whose capture refused) or `DflashKv::from_tail` refused (tail short of the window) | unimplemented: the restored session had no drafter context to start from; the trunk planes hold K/V latents, not the tapped residual rows the drafter consumes, so the drafter cannot be re-primed from the restored prefix without a second trunk prime | the turn AFTER any plain-served turn in a conversation, i.e. after every load shed | ADMITTED under `MEMRA_SPEC_WARM=1` as a COLD DRAFTER at the boundary (section 3); a refusal there names itself `warm-cold-drafter-refused` |
| 11 | (engine refusal at the tick, loud 500 by the dspark law) | `glm5_spec_session_new` (G): no hc trunk; TP-sharded model without `MEMRA_GLM5_SPEC_TP=1`, or with it but `MEMRA_GLM5_VERIFY_BATCH=0`; no draft source; prompt < 2; `MEMRA_SPEC_HPOST` on the MTP carrier; unqualified pipeline rewrite; penalties with #5 dark; ctx room `prompt + 4 > ctx_cap`. `from_restored` additionally: native MTP source; empty prefix; full cover without the arm or its logits; `cache.pos != fed.len()`; `dkv.len != fed.len()`; `dkv.cap != ctx_cap`; ctx room | invariants admission is supposed to have pre-validated | none in steady state (each has an admission twin) | unchanged |
| 12 | (400 at the HTTP layer, never reaches the route) | `reject_unsupported` (`lib.rs:8489`): `logit_bias`, `logprobs`, `n != 1` | honesty gate: semantic params the engine cannot honor 400 loudly | callers passing those fields | unchanged; not a spec exclusion (plain refuses them too) |
| 13 | (not an exclusion) | stop strings: `contains_stop_string` runs on the glm5 tick after every burst (W: `step_glm5_spec`, `StopReason::Callback`), EOS inside a burst is cut by `spec_visible_len` (W:20228); tools/reasoning are template text; `max_tokens` is the burst budget; `top_k`/`top_p`/`min_p` ride the filtered rejection walk; `seed` pins the Philox stream | served on the spec route already | the whole text-only, unconstrained, unpenalized cold class | unchanged |
| 14 | `prompt-above-spec-max` (PR #101, pending) | `MEMRA_SPEC_MAX_PROMPT` cap composed after the predicate | route policy (bimodal decode at depth on the B200 pair) | prompts longer than the cap once a fleet sets one | not on this branch; composes unchanged |

What the table says in one line: text-only, unconstrained, unpenalized, COLD requests route
spec; everything warm routes plain BY POSTURE (10a) today; penalties (5) and the tail-less
carrier (10e) were the two pure "unimplemented" exclusions and are the two doors this lane
lands; vision (7), constrained (6) and TP (1) need engine seams named in section 4.

## 2. `MEMRA_SPEC_PENALTY=1`: penalties inside the verify

Mechanism (engine, `Glm5SpecSession::pen` + `pen_hist`, G):

* Admission twin `glm5_penalty_admit`: a penalized request (`SpecSampling::pen_on`, the one
  predicate) opens a session only under the door; dark = the pre-lane refusal verbatim, and
  worker admission keeps such requests plain (`glm5_penalty_plain`, W:18682) so the refusal
  never reaches a customer as a 500. The worker seam `glm5_spec_sampling_for` (W:918)
  carries a GREEDY request's penalties in as `SpecSampling { temp: 0, pen_on }`;
  `spec_sampling_for` keeps its meaning (None = greedy) for every other route.
* Window: `pen_window_seed` over the prompt (the same function every spec route seeds
  with), then the ANCHOR at every round start and the ACCEPTED DRAFTS after every accept (the
  dspark `pen_hist` convention), trimmed to `PEN_WINDOW_MAX` (8192) like the dspark walk.
* Round: after `glm5_verify_rows`, the K+1 target rows are penalized IN A COPY through
  `penalize_logits_rows_inc` (the dspark accept walk's kernel): row r sees
  `pen_win ++ drafts[..r]`, i.e. the plain sampler's history at that position on every path
  where row r is consulted, same-round accepts included. Every p read points at the copy:
  the greedy per-row argmaxes, the native sampled walk's stats/gather, the DFlash2 arm's
  `dspark_accept_sampled` (handed the copy and a penalty-neutral config so the round has
  exactly one penalty pass), the full-accept bonus, the reject-slot residual, the PMIN0
  zero-draft bonus. `pen: None` = no copy, no launch, `plogits` IS `vlogits`.
* Anchor: `glm5_anchor` draws the session's first token from the boundary row by the plain
  route's own rule: sampled = `sample_boundary_token` with the window (it was always handed
  `&[]` before); greedy+penalties = one `penalize_logits` pass on the uploaded row + the
  device argmax (host tie-break contract); greedy = the host argmax, byte for byte.
* The DRAFT stays unpenalized. Rejection sampling is unbiased for any proposal with
  `u*q(x) < p(x)` + residual `norm(max(0, p-q))`; penalizing q would only buy acceptance
  overlap and would cost an evolving-history pass inside the drafter (the dspark lane's
  argument, verbatim).

NUMERIC CLASS, stated: the accepted stream's distribution equals the plain penalized
sampler's. For GREEDY that is tape identity: the plain route is penalize-then-argmax on the
host (`Sampler::sample`), the spec route is penalize-then-argmax on device on the same rows,
and the two penalty passes are now the SAME BITS: `keskar_penalize_rn` in
`cu/spec_sample.cu` is `apply_penalties_dense`'s three statements with
`__fdiv_rn`/`__fmul_rn`/`__fsub_rn` so nvcc's default `-fmad=true` cannot contract the freq
step into an FMA. The pre-lane kernels did `v -= freq*cnt + present` (one rounding of a sum,
and contractible), which is a 1-ulp difference and a 1-ulp difference is an argmax flip on a
near-tie. All FOUR penalty kernels (`penalize_logits_f32`, `_rows_f32`,
`_sparse_rows_f32`, `_rows_inc_f32`) now route through the one rule; the qwen spec route,
the dspark route and the `MEMRA_SERVE_DEVPENALTY` serving form therefore moved by at most an
ulp on penalized logits, inside their distribution-exact class (no byte-identity receipt
existed for any of them against the host). For SAMPLED, token-for-token identity against
the plain route is impossible by construction (host SplitMix64 + host CDF draw vs the
session's device Philox Gumbel); the claim is distributional and rests on gate 16 (same p
bits) + the walk's exactness.

Gates (rig 5090, exactness only; `crates/memra-engine/tests/glm5_dflash_session_gpu.rs`):

* 15 `gpu_penalized_greedy_spec_tape_matches_the_plain_penalized_sampler`: strong mixed
  penalty on the 32-id fixture vocabulary; RED first (door dark: the session refuses and
  names `MEMRA_SPEC_PENALTY`); penalized plain tape != unpenalized plain tape (the penalties
  provably move the tape, so identity is not vacuous); K=1..7 served bursts byte-identical
  to the host `Sampler`'s tape.
* 16 `gpu_device_penalties_are_bit_identical_to_the_host_sampler`: `penalize_logits_rows_inc`
  (5 rows, evolving window) and `penalize_logits` (the anchor) vs `Sampler::penalized_logits`
  (a new public oracle seam on memra-sampling) over random signed rows and a 200-token
  history from a 40-id alphabet with out-of-row ids, for rep-only / freq-only /
  present-only / all / negative coefficients / a 64-window that slides. Zero differing bits.
* 17 `gpu_penalized_sampled_twin_is_deterministic_split_invariant_and_engaged`: pinned
  seed reproduces, burst split invariant, seed sensitive, and penalized != unpenalized at
  the same seed.
* worker: `glm5_spec_sampling_seam_carries_greedy_penalties_and_nothing_else`,
  `spec_penalty_and_warm_doors_are_wired_in_comment_stripped_source`.

Receipts: `[glm5-spec] penalty arm ENGAGED (MEMRA_SPEC_PENALTY=1): ...` once per process;
per request `[glm5-spec] route=spec ... penalized=1 ... reason=-` (dark: `route=plain ...
penalized=1 ... reason=penalized`, byte-identical to pre-lane).

## 3. `MEMRA_SPEC_WARM=1`: the tail-less carrier re-arms a cold drafter

Blocker analysis first. The DFlash2 drafter's context is the STREAM-MEAN of the completed
trunk layer output at its tap layers (`hc_taps`), ingested into its own KV
(`DflashDraft::ingest_ctx`). A restored trunk cache holds K/V latents, KDA state and pool
keys for the prefix, NOT those tapped residual rows, so "re-prime the drafter from the
restored prefix" means re-running the trunk over (at least) the drafter's 2048-row window of
the prefix: a second prime that costs what the restore saves, and a numeric-program question
(a truncated-context re-tap would produce features from a different recurrent state). The
alternative the owner named, "a cold drafter prime while the target restores", is what
lands: the drafter starts EMPTY at the restored boundary and fills from the suffix prime's
taps and every committed round, i.e. it runs exactly the program a shorter prompt runs.

Mechanism (D + G + W):

* `DflashKv::floor` (D): the first ctx row the KV owns; `0` on every cold-primed KV and every
  full-tail import (the pre-lane shape, untouched). `DflashKv::new_cold_at(e, cfg, cap, pos)`
  = `len == floor == pos`, the `window_rows` below zero-filled (belt), refuses under
  `MEMRA_DFLASH2_SDPA_CLIP=0`.
* `sdpa_naive_w_lo(.., kv_floor)` (lib.rs): `kv_lo = max(window lo, floor)`; the clipped
  kernel never scores rows below the floor (the same "masked keys contribute exact zeros"
  identity the clip already rests on). `d2_windowed_attn` threads `kv.floor`; the legacy
  full-scan arm has no floor and refuses a binding one by name.
* `DflashKvTail::floor` + `tail_geometry_ok(.., tail_floor, ..)`: an export from a floored
  KV starts at the floor; a tail is legitimately short exactly when nothing below the floor
  ever existed; the import inherits `floor = tail.base`. A short tail from a floor-0 exporter
  is still the refusal it always was (pure test extended).
* Worker (W:18740 area): in the carrier probe's no-tail arm, under the door and with the
  DFlash2 source loaded, `new_cold_at(engine, cfg, ctx_cap, carrier.fed.len())` becomes the
  restored dkv; `from_restored`'s `dkv.len == fed.len()` law holds unchanged; the suffix
  prime's tap rows ride `pending` exactly as a tail import's do. Print-once
  `[glm5-spec] MEMRA_SPEC_WARM=1: ...` plus a per-request `[prefix-cache] glm5 cold-drafter
  restore: N prefix tokens ...` line; `reason=warm-cold-drafter-refused` when it cannot.

NUMERIC CLASS: output is target-exact by construction (verify arbitrates); the arm moves
ACCEPTANCE only. Expected shape on the real artifact: the first rounds draft from a
context of `suffix` rows only (a new user turn: ~20-500 tokens with the template), so
acceptance sits below a full-context session's until the window fills with committed rows;
a full-cover hit under the arm starts from ZERO context rows (anchor + masks only) and is
the worst case. That cost is unmeasured and is why the door ships dark.

Gate 18 `gpu_cold_drafter_restore_bytes_match_plain_decode_and_republishes_a_floor_tail`:
RED (clip off: `new_cold_at` refuses); leg 1: fresh boundary cache + cold drafter at the
prefix boundary, suffix = one drafter block, 12 served tokens byte-identical to plain
decode, drafted > 0; republish: the exported tail carries `floor == prefix.len()`, starts at
or above it, is SHORT (the floor admits it); leg 2: import it (floor inherited at the tail's
base), restore at the deeper boundary with the next 8 plain tokens as the suffix, 12 more
tokens byte-identical to plain decode's continuation.

## 4. Follow-ups with the exact blocker

* A. VISION (`reason=vision`, and `spec-serving-off` via `serve_spec`'s `!vision_req`):
  `glm5_spec_session_new` primes through `self.prime_cache(e, prompt, &mut cache, 0)` with
  token ids only; the plain prefill tick hands `hybrid_forward` a `VisionState` overlay
  (image units substituted at the `<|image|>` placeholder runs, residency hardened on
  `port/vision-overlay-residency-hardening`). The seam to add: an `Option<&VisionState>` on
  the spec prime (and on `from_restored`'s suffix prime) that reaches the same overlay hook
  the plain prime uses, plus the DFlash2 tap sink capturing over the overlaid rows (the
  drafter's features for image positions are then the trunk's real post-overlay residuals).
  Gate: spec-vs-plain byte identity on a fixture image request; the can't-hallucinate probe
  on the real artifact. Also the prefix-cache side: vision requests bypass every token-keyed
  reuse tier (pad runs are byte-identical across different images), so warm vision stays
  cold-only regardless.
* B. TP (`model-not-spec-capable` via `glm5_sharded_placement_admits`, and the engine's
  `MEMRA_GLM5_SPEC_TP` co-refusal): the spec x TP composition (verify rows over TP shards,
  per-rank KDA snapshot/replay, per-replica MLA latent truncation) is rig-gated
  (`glm5-tp-gate`) but has ZERO real-artifact receipts and no serving wiring on the sharded
  placement (the `lane/glm5-tp-serve-wiring-20260902` worktree is that increment). The
  exact blocker is a box cell: `MEMRA_GLM5_TP=2 MEMRA_GLM5_SPEC=1 MEMRA_GLM5_SPEC_TP=1` on the
  2x B200 pair with spec-vs-plain byte identity + the vendor-default sampled probe.
  Separately, `glm5_sharded_placement_admits` refuses ANY `MEMRA_STEP_TP`/`MEMRA_STEP_EP`
  composition by name: never co-gated with the glm5 walk.
* C. CONSTRAINED (`reason=constrained`): the grammar hooks (`mask_logits` on the verify
  rows and the draft head, `consume` per emitted token, `MEMRA_DRAFT_MASK`) live only on the
  qwen spec program; `glm5_spec_round` needs the same two masks (K+1 target rows before
  accept; the DFlash2 selector's candidate set before the walk) and the burst loop needs
  `consume` per committed token.
* D. LEGACY CONTINUATION POOL (`reason=warm-session-no-carrier`): same carrier shape as a
  tail-less prefix hit; needs the probe's `prefix_hit` term widened to `reused` entries,
  and the plain park invariant `cache.pos == fed.len()` proven for glm5 under
  `MEMRA_KV_PARK_COMPACT` (`compact_parked_plain_cache`), then it rides `MEMRA_SPEC_WARM`.
* E. THE POSTURE DOORS: `MEMRA_PREFIX_LATENT`, `MEMRA_GLM5_SPEC_PREFIX`,
  `MEMRA_GLM5_SPEC_FULLCOVER` are what make warm = plain in production today; their flip
  conditions are pre-stated on their rows and need the box battery, not code.

## 5. Box env to test (a non-serving glm5 DFlash2 box, never the prod origin)

    MEMRA_GLM5_SPEC=1 MEMRA_GLM5_DFLASH=<pinned drafter> \
    MEMRA_PREFIX_LATENT=1 MEMRA_GLM5_SPEC_PREFIX=1 MEMRA_GLM5_SPEC_FULLCOVER=1 \
    MEMRA_SPEC_PENALTY=1 MEMRA_SPEC_WARM=1

Cells, in the never-serve-greedy shape (vendor-default sampled, spec-engagement receipt
from the log, greedy only as the byte instrument):

1. Penalty, greedy instrument: `presence_penalty=0.6` (or the vendor default), the
   128-token digits/prose tapes, spec vs `MEMRA_GLM5_SPEC` unset: tape sha identical.
2. Penalty, sampled: vendor-default sampling + penalties, interleaved x5 vs plain:
   `[glm5-acc]` acceptance under penalties beside the unpenalized number, tok/s, TTFT;
   `[glm5-spec] route=spec ... penalized=1 reason=-` on every request; loop-law flags.
3. Warm: an 8-turn larger-prompt conversation where turn 1 is forced plain (`MEMRA_SPEC_K=0`
   for that boot, or c>low), then the door: turn 2 must read `[prefix-cache] glm5
   cold-drafter restore` + `route=spec ... restored=1`, restored-vs-cold byte identity on
   the continuation, per-turn `[glm5-acc]` acceptance vs a tail-carrying restore.
4. Both doors OFF in the same window: the route lines must be byte-identical to pre-lane
   (`reason=penalized`, `reason=no-drafter-tail`).

## 6. Receipts on this branch

Filled at close: rig gate output (`raw/`), fmt/clippy/check-flags lines, PR link.
