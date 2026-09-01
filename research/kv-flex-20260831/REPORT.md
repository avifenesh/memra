# Intra-process KV budget flex on the co-resident box (lane/kv-flex-20260831)

Executes Arc G of the tiering spec (darklanes
`research/engines-kv-oversubscription-20260830/SPEC-SESSION-TIERING.md`): on orn, ONE memra
process serves ornith-1.5 (chat, the paying interactive lanes) plus qwen3-embedding-8b and
qwen3-reranker-8b (the capture surfaces, batch-class subordinates). Arc G inverts the
harvest/shed borrow direction: the chat model's device KV may borrow the capture surfaces'
idle headroom and MUST shed it the instant their traffic arrives.

Base: c4145956b (origin/main). CPU-side unit coverage only in this lane; the GPU cells
below are named and PENDING. Flag: `MEMRA_KV_FLEX`, default 0 = OFF (FLAGS.md rows for the
flag and its two companions land in this same change).

## 0. Premise verdict, investigated first (receipts at c4145956b)

The spec's premise is that a partition exists to flex. Investigated before any code:

1. **There is NO reserved embed/rerank VRAM partition.** The capture surfaces are
   prefill-only worker requests (`max_new: 0`) forced onto the HARVEST lane
   (`embed_api.rs:115-124`, module law at `embed_api.rs:1-19`); their transient session KV
   is admitted through the same dynamic VRAM gate as chat and freed at `Done`. Their
   "batch headroom" is shared free VRAM bounded only by lane concurrency
   (`memra-lanes/src/lib.rs:166-170`, `MEMRA_LANE_MAX_HARVEST` default 8) and the per-tick
   harvest prefill budget (`memra-lanes/src/lib.rs:156-160`). "Borrowing embed/rerank
   headroom" therefore means borrowing from the shared free pool under a shed guarantee,
   not moving a fence between two named partitions.
2. **Chat SESSION KV already flexes by construction; that half of the premise needs no
   code.** Session admission is `free >= cost + reserve` against live
   `effective_free_bytes` (driver free + allocator pool cached, `worker.rs:12761`; the
   gate and its cost model at `worker.rs:9493-9536`). When capture traffic arrives and
   does not fit, the admission pass ALREADY synchronously reclaims: `px.evict_all()` plus
   oldest-parked eviction across the plain/spec/dspark pools until the arrival fits
   (`worker.rs:9730-9781`). Active session KV can never shed (started requests are never
   killed), so the only borrowable-and-instantly-sheddable device KV classes are parked:
   prefix-cache entries and parked pool entries.
3. **The one static partition that does NOT flex is the device prefix cache budget, and
   real headroom sits above it.** An explicit `MEMRA_PREFIX_CACHE_MB` is a fixed,
   deliberately un-clamped ceiling (`init_prefix_cache_budget`, `worker.rs:3291-3342`;
   enforcement is the insert-time eviction loop `while total_bytes > budget`,
   `worker.rs:5449-5470`). On orn the floor is 32 GiB
   (darklanes `deploy/ornith15/ornith15_serve_launch.sh`, `MEMRA_PREFIX_CACHE_MB=32768`,
   `MEMRA_MAX_SESSIONS=24`) against a fresh-boot 36.6 GB used / 60.6 GB free on the 96 GB
   card (A3 orn facts, quoted in the same launcher). Cache full + light sessions leaves
   roughly 20 GB idle that today can hold nothing parked. That slack is the borrow target.
4. **Single allocator owner exists and is kept.** `PrefixCache::total_bytes`
   (`worker.rs:4714` at base) is the sole byte accountant, owned exclusively by the
   scheduler thread; the budget policy was a process-lifetime `OnceLock`
   (`PREFIX_CACHE_BUDGET`, `worker.rs:3237`). The flex layer adds ONE policy authority
   (the scheduler-owned `KvFlex`) and derives borrowed occupancy as `total_bytes - floor`,
   never a second counter.

Verdict: the premise HOLDS in refined form. The feature is the prefix-cache borrow with
instant shed; the session half is already-borrowing by construction and is documented, not
re-implemented. Parked pools (count-capped, `worker.rs:1745-1773`) stay out of scope: they
park LIVE engine sessions (Arc C scoping) and already yield via the reclaim ladder.

## 1. Mechanism (all in `crates/memra-server`)

- `MEMRA_KV_FLEX` (default 0): armed only with a live device cache (floor > 0, batched
  serving). Effective insert budget = floor + grant; the grant is re-derived once per
  scheduler tick as effective-free VRAM minus `MEMRA_KV_FLEX_GUARD_MB` (default 4096, the
  fleet VRAM margin the orn launcher budgets) and published through `KV_FLEX_GRANT`,
  written ONLY by `KvFlex` methods. With the flag off the grant is 0 forever and
  `kv_flex_effective_budget() == prefix_cache_budget_bytes()`: byte-identical by
  construction, which is also the rollback seam.
- **Shed on capture arrival**: in the admission pass, a request with `capture.is_some()`
  (embeddings/rerank; they are also forced-harvest) triggers `KvFlex::shed` BEFORE any of
  its admission math, lane p99 gate included. The shed zeroes the grant, arms the
  `MEMRA_KV_FLEX_HOLD_MS` window (default 2000; an embeddings array admits one request per
  input, so a burst is many arrivals and every arrival re-arms), and evicts residency back
  to the floor via `PrefixCache::evict_to_bytes` using the SAME `capacity_victim` order as
  the insert-time budget loop (probation LRU first). Borrowed bytes EVAPORATE, never
  demote: the `evict_all` law, an admission-tick shed must not stall behind D2H copies.
- **Borrowed-first at session pressure**: at admission insufficiency the borrowed slice
  sheds first, upstream of the nuclear `evict_all` ladder, so a chat arrival that merely
  needs the borrowed slice back does not evaporate the floor residency the operator
  budgeted. The pre-existing ladder is unchanged downstream and still covers everything.
- **Pinned entries never shed** (in-flight fanout leases are authoritative); a shed that
  finds only pinned bytes above the floor logs loudly instead of no-opping silently.
- Metrics: `kv_flex_borrowed_bytes` (gauge, derived from the one accountant),
  `kv_flex_sheds`, `kv_flex_shed_ms` (cumulative; ms per shed = shed_ms / sheds) on
  `/metrics`; `[kv-flex] on:` boot line; per-transition `[kv-flex] shed (capture arrival)`
  and `[kv-flex] shed (admission headroom)` receipts.

## 2. Unit cells (green; invocation-anchored per the wiring-assertions law)

- `kv_flex_evict_to_bytes_sheds_to_the_target_and_spares_pins`
- `kv_flex_grant_policy_is_guarded_held_and_disarmed_to_zero`
- `kv_flex_shed_returns_borrowed_bytes_to_the_floor_and_counts_the_transition`
- `kv_flex_shed_on_capture_arrival_is_wired_before_admission_math` (comment-stripped
  source, invocation anchored, position-checked against the admission-requirement site)
- `kv_flex_borrowed_first_shed_precedes_the_nuclear_reclaim_ladder`
- `kv_flex_single_owner_wiring` (exactly two production grant writers, both inside
  `impl KvFlex`; both production insert wrappers consult the one effective-budget
  function; the metrics gauge derives from `KvFlex::borrowed_bytes(&px)`)

## 3. GPU gates (PENDING; the arc's real cost, pod of the serving card class)

Per Arc G and the measurement laws (interleaved x3, x5 on anomaly; fresh boot + boot-nonce
arm identity; `git log -1` in every receipt; vendor-default sampled rows for anything that
feeds a serving decision; reasoning_effort pinned; 128-token output floor):

1. **Shed-transition p99 under a synchronized embed burst. THE number.** Flex armed, cache
   deliberately grown above the floor (borrowed bytes verified via
   `kv_flex_borrowed_bytes`), then a synchronized burst of embed/rerank requests. Measure
   the p99 of exactly the capture-arrival-to-shed-complete transition (per-event
   `[kv-flex] shed (capture arrival)` ms lines; cumulative cross-check shed_ms/sheds), and
   the burst's end-to-end embed latency distribution.
2. **Embed/rerank latency unchanged vs the no-borrow arm.** Interleaved A/B: flag ON with
   borrowed residency vs flag OFF (byte-identical seam), same burst shape, p50/p99 per
   surface. The zero-tax law is the pass bar: no measurable capture-side regression.
3. **Chat-session correctness and no chat regression with flex armed.** Cached-vs-fresh
   greedy identity gate with a flex arm (borrowed entries served before a shed must be
   byte-identical, and post-shed misses must serve cold correctly); `serve-stress-gate.sh`
   with flex ON (64/64, zero new OOM parks); the 8-turn larger-prompt cache twin on the orn
   serving shape (per-turn TTFT and accept, engagement proven from `cached_tokens`) since a
   grown cache is a cache-behavior change; hit-rate delta ON vs OFF is the value receipt
   (the measured slack Arc G promises).
4. Teeth for gates 1-2: a forced-tiny arm (huge `MEMRA_KV_FLEX_GUARD_MB`, so grant 0) must
   collapse ON to OFF and invert any borrowed-bytes assertion (wiring, not prose).

## 4. Deploy clause (B5-shaped, per Arc G)

Flag OFF at merge. orn is the only eligible box (the co-resident shape). Flip lands in the
tracked `launcher_src` (darklanes `deploy/ornith15/ornith15_serve_launch.sh`) and deploys
via install-box/serve-deploy blue/green, exact launcher bytes banked; post-restart check
verifies process env plus the `[kv-flex] on:` boot line; post-deploy vendor-default sampled
probe on ornith with a spec-engagement receipt (K>0), plus one embed and one rerank probe
and a `[kv-flex]` receipt line in the same battery. Membership in the default-ON register
row (spec §3, accidental-default-ON risk) rides the FLAGS.md rows in this change.
