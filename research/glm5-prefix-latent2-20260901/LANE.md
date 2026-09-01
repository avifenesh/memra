# glm5 prefix-latent 2: the cache FIX lane (launch-gating, owner order 2026-09-01)

Lane: `lane/glm5-prefix-latent2-20260901`. Parent: `lane/glm5-prefix-latent`
(research/glm5-prefix-latent-20260830/ — DESIGN.md + box-window/WINDOW-STATUS.md, merged).
Owner order (verbatim, launch pricing thread): "fix the cache", plus the correction that
`supports_implicit_caching false` does not mean no cache — competitors board a real
$0.03 cache-read on GLM-5.3-Flash via explicit/prefix caching; we are the outlier with
NO working cache on this model. The GLM-5.3-Flash launch holds on this lane: the pricing
row needs a real cached-prompt price, and launch gate 7 flips from asserting
`cached_tokens==0` to asserting engagement.

## What the parent lane already banked (do not re-derive)

- Entry layout v3 CARRIES the glm5 planes (f32 latent rows, kpool keys travel,
  `index_pools_ready`, live tail-ring rows; KDA conv/ssm whole at the endpoint).
  `MEMRA_PREFIX_LATENT=1` arms capture+restore; OFF (default) refuses loudly.
- C1 whole-entry restore is BYTE-IDENTICAL with engagement everywhere, including
  cross-box sha identity on the same binary/recipe (box A + box B, both silicon editions).
- Box B narrowed the defect set to ONE: **a restored suffix primes at decode speed**
  (~33 ms/token, near-exact arithmetic fit; suffixes 469/1407/1899 tok → measured
  15.10/46.13/62.49 s vs predicted 15.5/46.4/62.7 s). The pp host-bounce defect was
  fixed by 93927b1fac (1m-demo chunk schedule), already in main.
- C3 zero-suffix restores are green and near-instant (0.011-0.017 s at 4.4k/8k).

## The defect, located (this lane's recon)

`worker.rs prefill_tick`: the prime branch carries the term `!(eager_mono && carried)`.
`eager_only_model()` includes every hyper-connections trunk
(`plan_backend::decode_batch_unconverted` — glm5_next), so ANY carried suffix (prefix
restore, reuse hit, continuation) rides tokenwise `decode_step` for the whole suffix.
The veto exists for gemma4, whose engine REFUSES pos>0 prime. glm5's engine does not:
`prime_cache_hyper` / `prime_chunk_hyper` are positional by construction (pos from
`cache.pos`, KDA state carried via `cache.recur`, MLA/DSA via the latent planes — every
chunk after the first inside one cold prime already runs exactly this program).

Second defect class, spec interplay (the launch shape is `MEMRA_GLM5_SPEC=1` +
DFlash2): under spec serving the prefix cache is structurally dead —
(a) `glm5_cold` requires `cache.is_none()`, so a prefix HIT demotes the request to the
plain route; (b) spec sessions own their cache (`s.cache=None`), so `maybe_prefix_seed`
never captures and the retire sweep has no glm5 arm — nothing ever populates the cache;
(c) `prefix_insert_from_spec_boundary` refuses latent-bearing caches at the source (the
anticipated fail-closed seam, its own comment names this lane's work).

## Stages (merge-early: each gate-green stage PRs immediately)

### PR 1 — carried suffixes ride the PRIME program (plain route)
- `MEMRA_HYPER_SUFFIX_PRIME` (default OFF, FLAGS.md row same PR): carves hyper trunks
  out of the `eager_mono && carried` veto in `prefill_tick`. gemma stays vetoed by
  predicate (engine refusal is real there). Engagement receipt both directions
  (`[suffix-prime] ENGAGED/DECLINED`, hyper trunks only).
- Retain `snapshot_at` (LCP split) + H11 hit re-seed + hit-LCP deepening for hyper
  trunks under the flag: the suffix is prime-provenance, so the R16 chained-provenance
  refusal no longer applies. Required for the 8-turn twin to deepen per turn.
- Worker unit tests: predicate truth table, gemma negative control, wiring assertions
  anchored on invocations (comment-stripped), reseed predicate.
- Routing note (PR #93 review finding 4a): `step_session`'s per-session prefill phase has
  no bound_rem stop and no lcp-split capture, so the deepening lever exists only on the
  `prefill_tick` path. Pre-existing for every family; a snapshot_at-armed session that
  prefills through `step_session` simply never captures (the arm is cleared at retire) —
  no wrong entry can be published from there, the lever is just absent. The provenance
  bit (finding 3a) additionally covers any tokenwise leak on either path.

### PR 2 — spec x prefix (`MEMRA_GLM5_SPEC_PREFIX`, default OFF, requires PREFIX_LATENT)
- Capture: `Glm5SpecSession` takes a boundary capture at `session_new` prime-done
  (conv/ssm snapshot + DSA tail-ring rows + boundary logits + last_h; latent ROWS and
  kpool keys are append-only below the prime boundary, sliced live at publish — the
  qwen/dspark `SpecBoundaryCapture` + drain pattern). Worker publishes via
  `prefix_insert_from_spec_boundary` gaining a latent arm; glm5 DFlash2 drafter tail
  exported alongside (`DflashKv::export_tail`, the q38 `MEMRA_DSPARK_PREFIX_RESTORE`
  pattern — drafter is sliding-window 2048, tail suffices).
- Restore: hit + spec-capable + tail present + non-empty suffix →
  `glm5_spec_session_from_restored` (restored trunk cache, hc_taps armed at
  `base=cache.pos`, suffix primes through `prime_cache` continuation, drafter KV
  `from_tail` + suffix pending rows). Native-MTP source refuses restore (no plane fill
  for the restored range); DFlash2 is the drafter of record anyway. Full-cover hits
  (empty suffix) keep the plain boundary-logits resume. Billing `cached = fed` (the
  dspark resume seam).
- Republish after restore so multi-turn deepens (the `MEMRA_SPEC_RESTORE_REPUBLISH`
  posture).

### Gates (pre-registered)
- Rig (5090, exactness only, TF32 off, flock): restored-then-suffix-prime vs
  donor-continues-priming byte compare at fixture geometry (planes + logits bits +
  24-step greedy tape); red arms (truncated rows, wrong-entry ssm, stale keys) stay
  biting; existing 34 prefix worker tests + latent gpu gates green (negative control:
  non-glm5 byte-identical, flag-off arm byte-identical).
- Entry-level provenance compare (PR #93 review finding 3c — R16's second prerequisite
  is about the PUBLISHED ARTIFACT, not just the continuation): a DEEPENED entry
  (hit -> suffix prime -> republish) must digest-match a COLD-primed entry of the same
  depth (`prefix_entry_state_digest` both sides); plus the checked-invariant red arm —
  force a tokenwise prompt token under the flag (starved budget) and assert BOTH capture
  sites refuse with the `[suffix-prime] ... REFUSED` receipts and no insert line.
- Box (the shared 2-card dev box — identity in the private ops repo, never here;
  cleanup-batch protocol; artifact staged
  2026-09-01: glm53-nvfp4 178G + drafter 2.2G rev dc77ff1c):
  - restored-vs-cold byte identity ON THE CONTINUATION, real conversations, plain +
    spec routes (mind dspark-session-reuse truths: eos-terminated parking, spec twin
    runs spec-ON with DFlash2);
  - the owner-law 8-turn larger-prompt cache twin, vendor-default sampling,
    reasoning_effort pinned, per-turn TTFT + cached_tokens>0 receipts, cache-bust
    control; acceptance bar: restored TTFT beats cold by ~ the cached fraction;
  - snapshot/restore round-trip + negative control.
- Serving-shape (3-card PP3 SPLITS=15,30, slot-B on the serving box — identity in
  the private ops repo — via the launch
  coordinator, END of lane): the ship-recipe round-trip + the twin re-run; numbers
  feed the pricing row directly.

### Close
- FLAGS rows final states; glm53 card trap line scope-noted fixed-at-version;
  darklanes research/INDEX.md row; corpus updates quoting receipts.
- Corpus note owed at close (PR #96 review round 2, finding 5 design note):
  `snapshot_plane_at` has NO content epoch — its safety is entirely the rollback
  arithmetic (round base >= prime boundary, keep >= 1, truncate_index_pool_keys can
  never clamp below the capture, publish clamps pools_ready to the capture's value).
  Any FUTURE lane that rewinds a trunk latent plane below a live session's prompt
  boundary and regrows it would publish foreign rows silently and owes this seam a
  re-gate.

## Receipts land in this directory. Commit pinned in every receipt.
