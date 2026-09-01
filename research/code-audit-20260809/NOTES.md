# code-audit-20260809 — working notes

Audited tree: `/home/avifenesh/projects/wt-public-split`, branch `restructure/public-split`,
commit `74afcaf6c7e2610b5f7e79f778040d0168c78028` (train tip at audit start). All file:line
references in PAPER.md are against this commit. READ-ONLY lane: no code changes, no commits.

## Inputs (the lens)

- `research/cachespec-20260809/RESULTS.md` (read from the wt-public-split copy — the lane
  merged, receipt on the tip): frozen 6,148-token prefix snapshot + linear TTFT growth
  (1.758 ms/uncached-token, R2=0.99971); two parked full-ctx (262144) plain continuation
  entries pin ~22 GB → admission serializes c=4 at ~25 s intervals, 4,659 VRAM defers;
  spec exonerated for the deployed policy (K=0 pp2-placement); forced-spec found
  max_tokens overshoot (5/17 at 769–770 of 768) and `/metrics.tokens_out == 0` under spec.
- `~/projects/wt-cx-longdepth/research/longdepth-20260809/PROGRESS.md`: cross-lingual token
  soup on Step-3.7-Flash at temperature 0.7 (as early as completion token 281), greedy clean;
  context axis eliminated (both 131072 and 262144 corrupt); spec receipt bug already found
  (round-coalesced text reported `n_tokens: 2048` with only 803 token ids). Bisect order:
  SWA position math → MTP drafter geometry → RoPE precision → SWA KV wrap.

## Method

Four parallel deep-read lanes (session lifecycle + admission; prefix cache + affinity/tenancy;
depth-dependent arithmetic in the engine; streaming/event contract + error honesty), each
returning file:line-cited findings, cross-checked against my own direct read of
worker.rs (full pass: lines 1–5700) before anything entered PAPER.md.

## Own-read observations (verified directly, worker.rs @ 74afcaf6)

1. **Spec turn-1 prime is monolithic.** `step_session` spec arm drains the WHOLE
   `prefill_queue` as one suffix (worker.rs:4909 `let suffix: Vec<u32> = s.prefill_queue.drain(..).collect();`)
   and hands it to one `generate_spec_session_*` call. The plain path's PREFILL_TICK_T=1024
   fairness chunking (worker.rs:46) does not apply — a 256k spec prompt holds the worker for
   the entire prime; disconnect sweep and admission both wait. on_commit yields only at
   round boundaries, which begin after the prime.
2. **Aborted sessions still park.** The disconnect sweep comment says so explicitly
   (worker.rs:2418 "Retire still parks reusable KV") and the retire sweep (worker.rs:3352+)
   makes no abort distinction — a client that hangs up after a 256k prompt still parks a
   full-cap cache (`cap = cache.max_ctx`, worker.rs:3430) into the reuse pool.
3. **Parked ReuseEntry is full-cap, not right-sized** (worker.rs:3430–3434): the cachespec
   receipt's ~11.2 GB dead entries come from here; the spec pool has a right-size ladder
   (worker.rs:4119–4165) but the plain park path has none.
4. **session_vram_cost is per-model, ctx_cap is per-request** (worker.rs:3689–3702 vs
   2309–2343): the admission `cost` is the first admit's free-VRAM delta; a small first
   request calibrates the gate for all later 256k requests (and vice versa — the receipt's
   11,174 MB cost came from a full-ctx first admit and then over-gated everything).
5. **cost==0 never memoizes** (worker.rs:2384–2393 `if cost > 0`): a first admit that is a
   reuse-pool hit consumes a parked cache (no new alloc, delta ≈ 0) → gate can stay unarmed
   for a model until a cold admit happens to run first.
6. **Admission gate never reclaims parked/prefix state before deferring**
   (worker.rs:2308–2368): the defer decision reads `free + pool_cached` and queues; the
   eviction hooks that exist elsewhere (`px.evict_all()` at 4204, spec evict-first at 4090)
   are not consulted here. Matches the cachespec P0 recommendation.
7. **spec-burst disconnect is detected only between bursts**: `flush_cb` sets
   `send_ok = false` on a failed send (worker.rs:4989–4992) but returns `keep` regardless;
   abort happens post-burst (worker.rs:5045–5050). Bounded by MEMRA_SPEC_BURST (32
   default) — minor, but with turn-1's monolithic prime the first "burst" includes the prime.
8. **Step-OOM park is pre-emission only** (worker.rs:2836 `active[i].generated.is_empty()`)
   and `park_requeue` clones tx and replays render inputs (worker.rs:3564–3594) — correct
   shape; the park frees device state by dropping the Session at retire.
9. **Graph-session promotion excludes step35** (worker.rs:2509) and eager-only models
   (2508); the dc SWA refusal is a named exclusion.
10. Retire sweep unpins prefix entries on EVERY exit path (worker.rs:3354–3355) — one
    central release, good.

11. **max_tokens overshoot — exact sites confirmed.** Engine: spec.rs:5221–5238 — in
    session mode COMMIT pushes ALL accepted draft columns plus the bonus into `out` even
    past `max_new` ("overshoot past max_new included — or `committed` under-counts the
    cache rows", spec.rs:5222) and `bonus_emitted = session_mode || out.len() < max_new`
    (spec.rs:5235). Worker: worker.rs:5024–5028 pushes every burst token into
    `s.generated` and streams them; the budget check is post-burst
    (worker.rs:5066 `if s.generated.len() >= s.budget`). Net: up to K+1 tokens over
    `max_tokens` reach the client — the cachespec 769–770/768 receipt, CONFIRMED.
12. **tokens_out increment sites: 3, all plain.** worker.rs:2604 (graph tick), 3087
    (eager-only decode), 3204 (batched decode). The spec-burst arm (step_session
    worker.rs:4894–5072) and the legacy round-robin arm (`!batching`, worker.rs:2426–2436)
    increment NOTHING — `/metrics.tokens_out` and `lane_tokens` both under-count on those
    paths. The cachespec `tokens_out=0` receipt, CONFIRMED, plus the round-robin miss is new.
13. **`PrefixCache::touch` is dead code in production** (worker.rs:1409
    `#[cfg_attr(not(test), allow(dead_code))]`): lookup hits refresh recency only through
    `pin_n`'s `last_use = Instant::now()` (worker.rs:1437–1438). Recency semantics OK today
    because every hit pins, but any future non-pinning read path silently loses LRU recency.
14. **Effective-free accounting** (worker.rs:2342 `free.saturating_add(engine.pool_cached_bytes())`):
    engine pins the async pool RELEASE_THRESHOLD to u64::MAX (lib.rs:943–1017), so
    `pool_cached_bytes` is genuinely allocatable headroom. Direction of error is
    conservative (under-count) per the lib.rs:1017 comment.

15. **insert_with_budget_pins self-eviction (benign-ish, verified).** worker.rs:1522–1579.
    The entry is pushed first, then the budget loop evicts LRU-first. An UNPINNED insert
    beside large pinned bytes skips the pinned-headroom check (worker.rs:1542 guards only
    `initial_pins > 0`), so after evicting every other unpinned entry the loop can evict
    the just-inserted entry itself (it is in the lru map). Callers of `insert` ignore the
    return, so no dangling handle — cost is a wasted multi-hundred-MB device snapshot copy
    that is immediately discarded, repeated per insert while pinned bytes crowd the budget.
    Perf bug, not correctness. remove_at's swap_remove fixup re-points the moved entry via
    BTreeMap key replacement (same (Instant,id) key) — no index drift found on this path.
16. **Namespace derivation verified end-to-end.** Both public surfaces overwrite
    `cache_ns` post-build with `tenant_namespace` (main.rs:2615, 2698); keyring mode wraps
    as `t:<tenant>\x1f<salt>` (auth.rs:416–418) and NS_SEP is excluded from tenant ids
    (forged-salt test auth.rs:786). No-keyring mode passes the raw salt through — single
    trust domain by design. One wrinkle: in NO-keyring mode a client-chosen salt
    `t:acme\x1f...` flows raw into `meter_key` (auth.rs:426–431) which strips it to
    `t:acme` — metering-row spoofing in open deployments (accounting only, no cache reach).
17. **Affinity in the shared "" namespace**: `affinity_key` (main.rs:1018+) accepts
    body/`user`/header names; in an open deployment client A can nominate client B's parked
    spec session by guessing its session_id — but `affinity_match` requires exact token
    reproduction through the checkpoint, so this is a same-namespace prefix-confirmation
    oracle (the documented PROMPTPEEK class), not content theft. Same boundary as the
    prefix cache; keyring closes it.
18. **PENDING_ADMITS yield blind spot**: the gauge decrements at channel pop
    (worker.rs:3536–3540), so VRAM-deferred requests sitting in the worker-local `queue`
    no longer register — an in-flight spec burst's round-boundary yield
    (worker.rs:4962–4964) only sees channel arrivals, and deferred requests wait out whole
    bursts. LOW (spec is off at the PP-2 default).

19. **Spec-path duplicate plain cache: NOT present (stale doc only).** The Session doc
    comment (worker.rs:1758–1760, "cache above stays as the (unused) admit allocation…")
    is stale: admit defers the plain cache ("legacy cache deferred: allocated below ONLY
    if the spec path doesn't take the session", worker.rs:3886) and the spec branch keeps
    `cache=None` (worker.rs:4197–4198; spec requires `seed_fed.is_empty()` at 4917, so no
    reuse-carried cache either). No multi-GB duplicate alloc — the comment should be
    updated, nothing more.
20. **Worker panic drops in-flight sessions without an Error event.** On a scheduler
    panic, unwinding drops `active` (every Session's `tx` closes with no
    `Event::Error`) and drops `queue` (queued requests likewise). The supervisor
    (worker.rs:5386–5483) keeps the cmd Receiver alive and respawns; clients of in-flight
    requests observe bare channel closure — what the HTTP layer surfaces for that is the
    streaming-lane question (Area 4/5).

21. **The longdepth n_tokens/803-ids receipt bug — exact site.** Native (non-OpenAI)
    blocking response collects `tokens.push(id)` once per Event::Token (main.rs:3016) and
    reports `n_tokens` from Done (worker truth). Under spec round-cadence one Event::Token
    covers a whole round (worker.rs:4989 sends one event with `last_id`), so
    `tokens.len()` ≈ rounds, not tokens — 803 ids for 2048 committed. CONFIRMED, same root
    as the coalescing finding (4.3).
22. **Mid-stream Error handling is honest** (main.rs:2896–2921): error object as final
    data chunk + [DONE] + close on the OpenAI surface; named `error` event native. Channel
    death without Done/Error → 503 Retry-After "worker restart in progress"
    (main.rs:3073–3080). Non-streaming Error → engine_error_response with class-mapped
    status (main.rs:3069). cmd_tx send failure → worker_unavailable_response + gauge
    rollback (main.rs:2627–2630). No 200-after-partial-failure found on the non-streaming
    path (Error preempts assembly).
23. **Streaming stop-string tail**: non-streaming truncates via truncate_at_stop
    (main.rs:3026) but the worker emits the delta BEFORE the stop check
    (worker.rs:4710–4716: send, then stop_strings scan) — a stream client receives text
    past the stop string within the final token's delta, and the stream-side scrubber
    (`scrub`) holds that back on BOTH branches (piece_chunks routes Content through scrub,
    main.rs:2781–2784 — initially suspected a gap there; verified covered). Remaining gap
    is only the NATIVE stream (scrub armed only for `chat || openai_compat()`,
    main.rs:2752), which is documented as byte-identical by design. Not a finding.

24. **Step-OOM park can RETAIN the VRAM it claims to free (pool-resumed spec sessions).**
    The park comment says "its caches drop (freeing exactly the VRAM the retry needs)"
    (worker.rs:2818–2820). The park is reachable only from the spec-phase loop
    (spec_order = sessions with spec.is_some(), worker.rs:2764–2766; Err match at 2834)
    and requires generated.is_empty(). But the park routes through `finished` → the
    retire sweep, which parks a spec session whole whenever
    `committed.len() >= 16 && next_pred.is_some()` (worker.rs:3380). A POOL-RESUMED
    session that step-OOMs on its first burst satisfies both (committed = resumed prefix,
    next_pred carried) → the session is re-parked into spec_reuse, retaining the very
    cache the retry needs, and the re-queued request must clear the VRAM admission gate
    (which cannot see that the parked entry would be consumed by the pool probe inside
    admit()). Self-heals when active empties (gate skipped `if !active.is_empty()`,
    worker.rs:2308) — serialization, not deadlock. A genuinely FRESH session that OOMs
    mid-prime has committed empty → drops → comment holds for that case only. Secondary:
    the retire sweep runs `spec_flush_pending` (a trunk pass) on an OOM-parked session's
    pending (worker.rs:3374–3378) — a device call fired right after a card-full OOM.
25. **Plain sessions never park on step-OOM** — the batched-decode error path kills up to
    a whole chunk (8 rows) with one quoted error (worker.rs:3208–3213); only spec-phase
    sessions get the park-retry. Deliberate per the admit-oom receipts (plain c=64 passed
    unaided) but worth stating: a transient OOM landing in a plain batched step is 8
    client-visible 503s, no retry.

28. **Depth-arithmetic own-read (corruption lens).** The SWA math is present in three
    places and all use the same token-aligned view-offset form (keys carry absolute rope,
    positional mask): eager decode hybrid_forward.rs:8900 `if swa && kvl.len > win {
    (kvl.len - win, win)`; batched decode decode_batch.rs:1415-1419 (verbatim, per-session
    offset — no cross-session term); prime hybrid_forward.rs:8684-8690 (offset ALIGNED
    DOWN to BK=32 for FA-tile chunk-invariance, seq_end predicate at 8706). The prime path
    is extensively chunk-invariance-hardened (chunkinv35/tickinv35 gates). KV is allocated
    at full max_ctx even for SWA layers (memra-kv/src/lib.rs:319-320 FullAttention arm uses
    `max_ctx * k_tok_bytes`), so there is NO ring-buffer wrap — the window is a VIEW, not a
    modular slot index. That eliminates the "rejected-draft clobbers slot pos%512" class:
    slots are absolute, append-only, never overwritten. STRONG SIGNAL for the longdepth
    hunt: greedy is clean and the corruption is temperature-only, while SWA/RoPE/KV-view
    math is SHARED by greedy and sampled — so the corruption is unlikely to be in the
    attention position math and more likely in the SAMPLING path (gumbel/philox device
    sampler, spec.rs rejection-sampling q/p buffers, penalty buffers). The device gumbel
    counter (spec_sample.cu:371-382, memra_sctr_inc + gumbel_perturb_ctr) and the
    per-column serving gumbel (lib.rs:2001, stream_pos = generated.len() as u32) are the
    depth-dependent sampling sites — deferring full ranking to the depth-lane subagent.

29. **F10 (cost==0 disables gate) likelihood correction.** The session-lane subagent rated
    it MED via "worker restarts mid-conversation with MEMRA_REUSE_POOL populated" — but
    `reuse`/`spec_reuse` are run() locals (worker.rs:2112-2113), wiped on any restart, so
    the FIRST admit of a process always faces empty pools and allocates a real cache
    (nonzero delta). Prefix-cache hit also allocates fresh (worker.rs:3825). The only
    cost==0-on-first-admit path is mem_get_info noise flooring to 0. Real bug (the `if cost
    > 0` guard + Some()-gated read means one missed sample = gate off for the model's
    process life) but LOW likelihood, not MED. Fix still stands (memoize analytic cost at
    load, per F9).

30. **CORRUPTION ROOT CAUSE FOUND + verified.** `cu/spec_sample.cu:36-39` `u01()` can
    return exactly 1.0f (128 of 2^32 u32 values round to 2^32 as f32; +1.0 absorbed;
    x2^-32 exact). Then gumbel `g = -logf(-logf(1.0)) = +inf` → that vocab id always wins
    the argmax. Verified the float arithmetic myself (u01(0xFFFFFFFF)==1.0f exactly,
    spacing-256 near 2^32). Temperature-only (greedy returns at spec_sample.cu:50 before
    Philox), P=3.83e-3/token at step35 vocab 128896, E[first]≈261 vs receipt token-281.
    Promoted to exec-summary #0 and Area 3.1. This closes the longdepth lane. Also from the
    depth lane: device top_k/top_p is a UNION not intersection (spec_sample.cu:250-253,
    diverges from host memra-sampling/src/lib.rs:154-175) — 3.2; filter_stats 2^-24 floor
    3.3; RoPE powf depth-drift 3.4; f16 O-acc demoted for step35 (hd128 arm is f32) 3.5.

31. **CORRECTION (my finding 13/5.7 was wrong on the mechanism).** I claimed
    `debug_assert!(px.unpin(&pin))` runs the unpin as the condition and only discards the
    return. FALSE — empirically verified: `debug_assert!` compiles out its ENTIRE argument
    when debug-assertions is off, and `rustc -O` on a `debug_assert!(side_effect())` ran the
    side effect 0 times. Workspace has no `[profile.release]` override, so `cargo build
    --release` (all gates + serve scripts) = debug-assertions OFF = px.unpin NEVER runs in
    any shipped build. This is the prefix-cache lane's F1 and it is CRITICAL, not MED:
    prefix pins leak forever → served/fanout entries permanently un-evictable → budget stops
    bounding → pinned inserts refused (second cause of the cachespec freeze) → evict_all
    frees 0 bytes → hard `cache alloc failed`. Paper 5.7 rewritten + promoted to exec #0b.
    Note 13's touch-is-dead-code observation still stands; its recency-OK conclusion was
    predicated on "every hit pins" — which is now the leak, not the safety.
32. **Reconciliation with prefix-cache lane (agent ae4e1397).** Its F1=my corrected 5.7
    (CRITICAL). F2=my 1.7 (unbounded PoolKey, agree CRITICAL). F3=2.1 + it adds that the
    HIT path has NO reachable insert site at all (snapshot_at/seed_prefix set only under
    `reused.is_none()`, worker.rs:3862) — total freeze not partial; folded into 2.1's
    understanding. F4/F5 = release `expect()`/self-evict — added as hardening note to 2.5.
    F6=my 4.5/2.4 (tier-dependent prompt_tokens). F7=my 2.3 wrinkle, escalated (unvalidated
    salt = client-writable /metrics rows, unauthenticated endpoint). F8=NEW (2.3b, default
    tenant not reserved) — my tenancy pass missed it. Its H1-H7 plain-affinity hazards
    overlap my 2.6 + add H6 (prefix_restore slice_mut panics not Errs) and H5 (spec/plain
    double-park via demotion) — both worth carrying. Its "LRU no index drift" independently
    confirms my own trace. NO conflicts after the 5.7 correction.

## Session state x memory table (worker truth)

| state | device memory held | host memory held |
|---|---|---|
| queued (worker VecDeque) | none | Request (prompt ids/text, turns) |
| admitted, plain, prefill | Cache (full ctx_cap alloc via pp::new_cache) + prefix_pin lease | prompt, ReplayPlan, sampler |
| admitted, plain, decode | Cache + mask_dev (constrained) + last_logits_dev park (lean) | last_logits [n_vocab] f32, generated, fed |
| admitted, spec | SpecSession (trunk cache + draft scratch + draft-graph ctx + turn_ckpt snapshot), NO plain cache (verified: deferred alloc, worker.rs:3886/4197) | same + committed |
| graph solo | GraphSession (owns cache + capture arena) | graph_pending |
| retired → plain park | ReuseEntry: full-cap Cache + last_logits (host) until 2 later parks LRU it | fed clone |
| retired → spec park | SpecReuseEntry: whole SpecSession incl. checkpoint snapshot | committed_text (detok), fingerprint |
| prefix cache entry | per-layer KV byte copies + conv/ssm copies (bytes-counted, budget-bound) | toks, last_logits |
| step-OOM parked (requeued) | whatever the retire sweep parked (see 24) | rebuilt Request via ReplayPlan |
| aborted (client gone) | parks same as normal retire (see 1.2) | — |

26. **Streaming interactive requests deliver ALL admission-time 400-class errors as
    HTTP 200.** `peek_shed` peeks the first event only for non-interactive lanes
    (main.rs:2256–2258 `if lane == lanes::Lane::Interactive { return Ok(rx); }`); the SSE
    response commits immediately (main.rs:2636–2638). Model validation happens on the
    worker (handle_cmd worker.rs:3543–3546), context-length/empty-prompt/template errors
    in admit() — all AFTER the 200+headers are gone. An unknown-model or
    context_length_exceeded streaming request returns 200 with an in-band error chunk;
    status-code-based clients/alerting/uptime math never see the 4xx. OpenAI validates
    pre-stream. MED severity, HIGH likelihood (every streaming client mistake takes this
    path). Fix shape: peek the first event for interactive streams too (bounded — worker
    admission is fast) or validate model/ctx on the HTTP thread against cached caps
    before committing the stream.

27. **Graph-promoted sessions never park reusable KV.** Promotion takes the cache
    (worker.rs:2535 `let cache = s.cache.take().unwrap();`) into the GraphSession; the
    cache returns to `s.cache` only on DEMOTION when a second session arrives
    (worker.rs:2450–2451). A solo graph session that runs to completion retires with
    `s.cache == None`, so the retire sweep's plain-park branch
    (worker.rs:3409 `if let Some(cache) = s.cache`) never fires — the conversation's KV
    is dropped and the next turn cold-primes. The graph path targets exactly the
    long-solo-greedy case (budget >= gs_min=384) where a parked continuation would pay
    most. MED/perf-only. Fix shape: at retire, recover `g.cache` from a live GraphSession
    (same handoff the demotion path already does) before the park branch.

## Subagent verification status

## Subagent verification status

Findings from the four lanes were spot-checked as follows before inclusion:
- every CRITICAL/HIGH finding's primary line was re-read directly (Read tool, exact offset);
- findings whose quoted text did not match the file at 74afcaf6 were dropped or re-verified;
- severity/likelihood assignments are mine, not the subagents'.

(See PAPER.md for the final, deduplicated findings list.)
