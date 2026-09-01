# Spec disengages on every prefix-cache hit — mechanism (from source, v0.91.0 @ 022d848148)

Date: 2026-08-18. Lane: lane/spec-on-cache-hit. Evidence driving this lane:
darklanes ops/bench/endpoint-bench-20260818 — on cache-hit rows `usage.spec`
vanishes on BOTH served models (qwen3.8-27b DE, gemma-4-31b-it NJ); uncached
identical prompts engage every rep (qwen acc 0.55–0.76, gemma 0.45–0.47); qwen
repeats drop ~135 → ~75 tok/s once cached; `cached_tokens` itemizes full prefix
(239/241).

## Where `usage.spec` comes from

`finish()` (crates/memra-server/src/worker.rs:11263) sets
`spec: (s.spec_rounds > 0).then(|| SpecUsage {..})` (worker.rs:11301).
`spec_rounds` only advances inside the two spec routes (qwen MTP burst
`finish_pipelined_spec_burst` / `step_session` spec arm; gemma `step_gemma_spec`
worker.rs:11172). A session that admits onto the plain path can never produce
`usage.spec`. So the symptom "usage.spec absent on cache hits" == "cache-hit
requests are ADMITTED PLAIN", not a telemetry loss.

## The two disengage sites (both deliberate v1 guards, not bugs)

### 1. Qwen/MTP arm — explicit downgrade-on-hit

worker.rs:8766-8776:

```rust
// Downgrade-on-hit (lane/spec-prefix-cache): a restored prefix carrier is plain-session
// state; serving it through spec would need a draft plane the entry doesn't carry (v1).
// The plain path serves the hit exactly as spec-off did at 8.5 req/s on the sold shape.
if prefix_hit && spec_eligible {
    spec_eligible = false;
    ...
}
```

Belt-and-suspenders second guard: the spec-session block only opens on
`spec_eligible && seed_fed.is_empty()` (worker.rs:8824), and a prefix hit fills
`seed_fed` with `entry.fed` (worker.rs:8777-8796) — so even without the explicit
downgrade the restored carrier can never enter a `SpecSession`.

History: this shipped with the COMMIT-GATED PUBLICATION port
(lane/spec-prefix-cache, 2026-08-14, merge e7b56d1b1d; design
research/cache-spec-design-20260814/{REPORT,PORT-PLAN}.md). Before that lane,
spec-eligible requests BYPASSED the prefix cache entirely (policy comment
worker.rs:1408-1411), which produced the measured 4x sold-shape loss (canonflip:
spec-on c=16 2.14 req/s @ 18.4% hit vs spec-off 8.50 @ 99.5%). The lane made
spec requests probe and publish (`prefix_insert_from_spec_boundary`
worker.rs:3988, `SpecBoundaryCapture` spec.rs:653) but implemented only items
1–2 of the port plan; item 3 (`spec_session_from_restored` + a draft plane in
`PrefixEntry`) was deferred and the downgrade shipped as the v1 stopgap. The
2026-08-18 endpoint bench is the cost of that stopgap: the sold agent-loop shape
(repeated prompt) now always hits, therefore always decodes plain.

### 2. Gemma assistant-drafter arm — cold-only admission

worker.rs:9276-9292 (`gspec_k` decision): the gemma spec route requires, among
its verified arms,

```
&& spec_resumed == 0
&& seed_fed.is_empty()
&& cache.is_none()
```

with the stated reason "COLD sessions only: no prefix/reuse seed (seed_fed /
spec_resumed assume plain or qwen-spec cache shapes; the gemma session primes
its own cache)" (worker.rs:9271-9273). A prefix hit sets `cache = Some(restored)`
and `seed_fed = entry.fed`, so every cache-hit gemma request fails the guard and
admits plain. (The solo-admission arm `n_active == 0` is the banked coexistence
policy — new arrivals under load take plain — and is NOT part of this defect.)

## Why the guards exist — the real structural gap

A `PrefixEntry` (worker.rs:2769-class) stores trunk state only: token key,
full-attn K/V rows `[0..pos)`, GDN conv/ssm snapshot, boundary logits. What each
spec route additionally needs:

- **Qwen MTP** (`SpecSession`, spec.rs:469): `scratch: MtpScratch` — the MTP
  head's OWN K/V rows, one per committed token (spec.rs:1117), filled during
  prime from trunk hiddens under the predecessor-pairing convention (row i =
  f(token_i, trunk_hidden_{i-1}); fill at spec.rs:7187-7256) — plus `last_h`
  (pre-output_norm hidden of the last committed row, the row-0 fill anchor and
  chain seed). Neither is in the entry. Restoring trunk-only state into a
  SpecSession would leave the drafter attending over uninitialized scratch rows:
  exactness would hold (verify is trunk-side argmax) but acceptance would
  collapse — hence the v1 refusal.
- **Gemma** (`GemmaSpecSession`, gemma_spec.rs:1717): NO draft-side KV at all —
  the 4-layer Q-only assistant drafter attends the TRUNK's KV cache
  (gemma_spec.rs:468-471, 1744-1747). Session state beyond the trunk cache is
  just `h` (post-norm hidden of the last committed row) and `pending`
  (argmax of the last logits). Both are products of the prime, and
  `gemma_spec_session_new` (gemma_spec.rs:1757) only knows how to build them by
  priming the WHOLE prompt into a FRESH `Cache::new` — there is no constructor
  that accepts a restored trunk carrier. That, plus gemma4's eager-only prime
  (the engine refuses pos>0 `prime_cache`; carried suffixes ride tokenwise
  `decode_step` — worker.rs:9904-9911), is why the guard says "the gemma session
  primes its own cache".

## Why this is NOT the rolled-back partial-restore hazard

`MEMRA_PREFIX_PARTIAL_RESTORE` (worker.rs:2003-2008) is default OFF because
lane/cx-lcprestore (merged 6249b0096, defaulted off in c6cac1e1c) byte-diverged
at LCP splits 512/2048. splitiso (0b0ffa13c6) later BOUNDARY-IDENTIFIED it: the
divergence is a two-programs defect specific to `eager_mono && carried` — cold
gemma primes monolithically while a restored suffix continues tokenwise; split
POSITION was never the variable, the priming PROGRAM was. Whole-entry restores
at exactly `e.pos` (the shipping default path, `prefix_restore` worker.rs:3791)
are captured at genuine prime boundaries and are the banked production path.
The disengage under diagnosis lives entirely in ADMISSION ROUTING
(spec-vs-plain), not in restore mechanics; the fix must arm spec on the proven
whole-entry restore and must NOT reintroduce mid-entry (`at < e.pos`) restores.

## Cache tiers not at fault (checked)

- Spec continuation pool (`spec_reuse`, exact/text/affinity probes,
  worker.rs:8825-9070) restores WHOLE parked sessions (trunk + scratch + last_h)
  and stays spec — but it is move-out, capped 2/ns, and the prefix probe at
  worker.rs:8591 runs at a tier that fills `reused`/`seed_fed` for the
  identical-prompt shape, so agent-loop repeats land on the prefix tier and get
  downgraded before the pool is consulted for spec.
- `choose_spec_k` (worker.rs:2257) already has a CachedLong arm
  (prompt >= SPEC_K_LONG_PROMPT_MIN && cached >= SPEC_K_LONG_CACHE_MIN →
  K = SPEC_K_CACHED_LONG): the K policy anticipated cached spec traffic; it is
  the eligibility guards above that never let it fire.

## Summary

Two admission guards — worker.rs:8769 (`prefix_hit` clears `spec_eligible`) and
worker.rs:9287-9289 (`gspec_k` requires a cold, cache-less session) — route
every prefix-restored request onto the plain path on both served models. They
exist because a v1 `PrefixEntry` carries no draft plane (qwen) and no
restored-carrier constructor exists (gemma). The fix is PORT-PLAN item 3,
scoped to whole-entry restores only: publish the draft plane + boundary hidden
from spec boundary captures, add `spec_session_from_restored` (qwen) and
`gemma_spec_session_from_restored` (gemma), and drop the two guards for
qualifying hits.
