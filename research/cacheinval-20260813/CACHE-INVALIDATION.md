# Prefix-cache invalidation and coherence — what memra actually drops, what it trusts, and how that compares to the field

Date: 2026-08-13
Tree read: `9a7091ea338345cf634a7ed10476310fd9a0c7c2` (main). Every `worker.rs`/`main.rs`/`ledger.rs`/
`auth.rs`/`decode.rs` line number below was re-verified against that revision; where an earlier
audit or survey cited a drifted line, the number here is the corrected one.
Scope: correctness and coherence of the cross-request prefix cache and its billed
`cached_tokens`. No code changed, no measurement run, no GPU taken.

---

## 1. What memra invalidates today

**A prefix-cache hit is trusted for content and validated only for structure, and the key does not
record which numeric program produced the bytes.** `prefix_restore_at` checks — before the first
device copy — SWA-ring exclusion, layout version, pool-key identity, `pos == toks.len()`, restore
bounds, destination freshness, layer counts, recurrent presence at a mid-entry split, per-plane row
strides and bounded source sizes (`crates/memra-server/src/worker.rs:3297-3423`). It never
checksums the K/V bytes, and the module's own EXACTNESS CONTRACT states that an entry holds bytes
"from WHATEVER prime config ran (single, chunked, or concat batch-prime)" (`worker.rs:1320-1332`).
So the bytes a paying request restores, and the boundary logits it samples its first token from
(`last_logits: e.last_logits.clone()`, `worker.rs:7141`), depend on the concurrency shape that
happened to be live when the entry was written. That is load-history dependence in exactly the
place the early-EOS class lives (`research/eosclass-20260813/`, and the Q27 restored-hit 11/60-token
EOS at `research/cachesize-20260813/raw/attempt7-cache-hit-eos/`).

**The complete invalidation inventory is three mechanisms, and nothing else.**

1. **Byte-pressure eviction** inside `insert_with_budget_pins_and_pct`: `while self.total_bytes >
   budget { capacity_victim() … }` (`worker.rs:3162-3169`). `capacity_victim()` returns
   `self.probation_lru.values().next()` — the probation LRU only (`worker.rs:3029-3033`).
2. **`evict_all()`** — drop every UNPINNED entry (`worker.rs:3179-3188`), called from seven sites,
   all of them allocation pressure or a security degrade: `worker.rs:4065` (PP host-bounce degrade),
   `4857`, `6974`, `7169` (whole-entry hit: session-cache alloc failed), `7268` (partial restore:
   alloc failed), `7484`, `7695`.
3. **Worker restart** — `px` is a local of `run()`, so a respawn drops the whole cache.

There is **no TTL, no age expiry, no generation/epoch counter, no flush API, no invalidation hook on
model state, and no invalidation on keyring reload.** `last_use` is only ever an ordering key
(`lru_key`, `worker.rs:2830`); it is never compared against a duration.

**Key construction.** Visibility is partitioned by `type PoolKey = (String, String)` =
`(model_alias, cache_namespace)` (`worker.rs:980`), built identically at admission and from the
session (`worker.rs:3832-3834` / `6837`-region). `cache_ns` is `t:<tenant>\x1f<salt>` under a keyring
(`auth.rs:471-473`) and the raw salt otherwise; the separator and `t:` prefix are unforgeable by a
client (`main.rs:1102-1125`). Entry identity is the **exact token vector**, compared elementwise:
`n >= PREFIX_CACHE_MIN_TOKENS && n <= prompt.len() && prompt[..n] == e.toks[..]`
(`worker.rs:2782-2794`). No hash is load-bearing anywhere in prefix-cache identity — the only hash
in the path (`fnv1a`, `worker.rs:8154`-region) is a log field. So memra **cannot** serve one
prompt's bytes for a different prompt; that whole collision class is absent by construction, and it
is one of the two places memra is genuinely stronger than the field (§3).

Three further properties follow from the code and matter as much as the event list:

- **Entries are immutable once inserted.** Exact-key dedupe keeps the OLD entry and drops the fresh
  snapshot with no byte comparison (`worker.rs:3098-3104` — `key_index` hit returns `None` for an
  unpinned insert, and the incoming `e` is dropped). Whatever program wrote the first entry for a
  key keeps serving that key until byte pressure evicts it.
- **A restore failure does not invalidate the entry.** The non-alloc `Err` arm logs
  `"{msg}; cold path serves"` and leaves the entry resident (`worker.rs:7148`, `7172`). Then
  `!px.has_key(&pool_key, &prompt[..l])` blocks a replacement LCP snapshot (`worker.rs:7288-7290`)
  and `has_covering` blocks a replacement seed (`worker.rs:3682`), so the prefix class becomes
  permanently un-cacheable while the bad entry keeps occupying budget.
- **The one live mid-process config change reclaims nothing.** `prefix_cache_budget_bytes()`
  deliberately re-reads `pp_host_bounce_active()` outside the `OnceLock` and returns 0
  (`worker.rs:1805-1820`, with the comment explaining why). The degrade calls `evict_all()`
  (`worker.rs:4065`), which skips pinned entries — and with the budget now 0 the eviction loop never
  runs again, so those bytes stay resident and keep being published as `prefix_cache_bytes`
  (`worker.rs:6363`, `main.rs:2302`). Everything else is process-lifetime: `Cmd` has only
  `Generate` (no model reload), and every prefix flag is a `OnceLock` (`worker.rs:1825-1893`).

---

## 2. Coherence holes, ranked by blast radius

**H1 — The Gemma-4 whole-entry hit lands on the boundary splitiso proved divergent, default ON, with
no guard.** The hit branch is gated only on `if prefix_on && reused.is_none() && !spec_eligible`
(`worker.rs:7126`) — no eager-only exclusion. `lookup` admits an entry whose tokens are a *strict*
prefix of the prompt (`n <= prompt.len()`, `worker.rs:2787`), so a whole-entry hit routinely carries
a suffix. The restored cache has `pos > 0`, hence `carried = true` (`worker.rs:8238`), and
`prefill_tick` vetoes the prime branch on `!(eager_mono && carried)` (`worker.rs:8269`) — so every
remaining PROMPT token goes through `decode_step` one at a time. That is verbatim the condition
`research/splitiso-20260813/RESULTS.md` names: "a genuinely cold Gemma request executes one
monolithic `gemma4_prime`, while a partial-prefix hit must execute every suffix token through T=1
`decode_step`" — verdict BOUNDARY-IDENTIFIED, 51 PASS / 18 FAIL on the frozen dense map, 80/116 on
the targeted map. The remedy actually shipped was only keeping `MEMRA_PREFIX_PARTIAL_RESTORE` off
(`docs/FLAGS.md:527`); the whole-entry path has the same shape and was never gated. The loop closes
on itself: `maybe_prefix_seed` has no eager filter (`worker.rs:3674-3686`), so Gemma-4 sessions seed
the entries they later hit, and the continuation pool is a second unguarded entrance to the same
`carried` state (`worker.rs:6892-6898`). Guard today: **none**. Gemma-4 26B-A4B is task #127 (4.5x
Q27 volume on OpenRouter), so this is about to become the highest-traffic model.
Caveat, stated plainly: splitiso measured the *partial-restore* arm. That the whole-entry-plus-suffix
arm diverges is a **structural inference from the same `eager_mono && carried` boundary, not a
measured receipt** — see Q3 in §6.

**H2 — Cross-request boundary logits.** On an exact full-prompt hit the suffix is empty and request
B samples its first token from request A's prime logits (`worker.rs:7141`;
`cached_hit_needs_first_token`, `worker.rs:143-149`, exists precisely to allow that shortcut). Guard
today: a prose argument appealing to the "documented batched-prime near-tie first-token law"
(`worker.rs:1320-1332`). Under the one-numeric-program rule this is a cross-request transition that
is neither forbidden nor proven bit-identical. `docs/FLAGS.md:527` ties this same restored-hit class
to a live symptom.

**H3 — The absent-layer rule is length-based and fails OPEN.** Capture records ANY layer matching
`Some(l) if l.len == 0 && cache.pos > 0` as absent (`worker.rs:3232-3239`), restore deliberately
tolerates `(Some(_), None) => {}` (`worker.rs:3385-3388`) and then sets `cache.pos = restore_len`
unconditionally (`worker.rs:3420`). So a TRUNK layer at len 0 restores nothing, is billed as a full
cached hit, and every other preflight in that function fails closed. `prefix_snapshot_trunk_layers`
already computes the exact trunk/NextN boundary (`worker.rs:1712-1718`) but is consumed only for
byte geometry (`worker.rs:1742-1748`). This is the arm the zero-insert fix (`96361c531`) widened —
correctly, since an allocated-but-unexecuted MTP head legitimately carries len 0, but the predicate
cannot tell that head from a corrupt trunk layer. Guard today: **none on identity**.

**H4 — The step35 prefill attention arm is not part of the key.** The arm is chosen on the request's
total end position (`swa_naive = seq_end > win`, `crates/memra-engine/src/hybrid_forward.rs:9159`),
`PREFIX_CACHE_MIN_TOKENS = 64` (`worker.rs:1345`), and `PrefixEntry` records only model, namespace,
tokens, layout version, per-plane strides, pos, logits and bytes (`worker.rs:3274-3288`). An entry
captured under one arm can be restored beside a suffix running the other. Worse for measurement,
`MEMRA_STEP35_SWA_FA`, `MEMRA_NOFA` and `MEMRA_PRIME_CALLLOCAL` are read PER CALL, not `OnceLock`'d
(`hybrid_forward.rs:9173`, `:9183`, `:727-733`), so flipping an arm in-process serves pre-flip
entries to post-flip requests — the A/B seam is not cache-safe. Whether the two FA prefill kernels
differ bitwise at `seq_end <= win` is **UNVERIFIED**; no receipt exists either way.

**H5 — A failed restore poisons its prefix class permanently** (mechanism in §1). Blast radius per
event: one prefix class re-pays the full prime forever (4,860 tokens on the sold Q27 shape) while a
301 MB entry sits resident. Reachability today is low on a single-model worker — `has_swa_ring`,
non-fresh destination and layer/stride mismatch are structurally excluded — so this is a latent
design property, not a confirmed live failure. It becomes load-bearing the moment any recommendation
below starts returning `Err` deliberately.

**H6 — Tenant isolation is a post-hoc field overwrite, and revocation does not touch the cache.**
Both request builders set `cache_ns: cache_namespace(&req.cache_salt)` — the RAW client salt, no
tenant prefix (`main.rs:2952`, `:3122`) — and only the two handlers repair it afterwards
(`main.rs:3686`, `:3861`). `PoolKey` equality is the cache's only isolation mechanism
(`worker.rs:980`), so a third admission route (task #117, direct self-serve) that omits that one
assignment shares prefixes across tenants with no error and no log. Separately, the keyring
hot-reloads on mtime so `--revoke-key` kills authentication immediately (`auth.rs:372-402`), but
`PrefixCache` has no flush-by-namespace method and `auth.rs`/`admin.rs` contain no `cache_ns`
reference at all — the revoked tenant's resident prefixes survive and are reachable by the next key
issued to the same tenant id, along with their `cached_tokens` credit.

**H7 — Billing: a cancelled fanout sibling is charged its whole reused prefix at the full input
rate.** `Event::PromptUsage` is the only pre-terminal usage event and its sole emitter fires once at
admission with `s.n_cached` as of admit (`worker.rs:5004-5007`). The fanout credit is applied AFTER
admission (`s.n_cached += group.prefix_len`, `worker.rs:8162`, plus `worker.rs:8166-8171`) and never
re-emits, so `PendingReceipt::drop` prices the abandoned row from the stale value
(`ledger.rs:1867-1889`). `record_prompt_usage` SETS rather than adds (`ledger.rs:1757-1769`), so a
re-send is safe. Cached input is contractually cheaper (`ledger.rs:177`, `:186-197`) and fanout is ON
by default (`worker.rs:1869`; `docs/FLAGS.md:523`) — on the sold Q27 shape that is ~4,860 tokens
overcharged per cancelled sibling, pointing the wrong way on first paying traffic.

**H8 — A step-OOM park/re-admit double-counts every published cache metric.** `park_requeue` rebuilds
the Request with the SAME `tx` (`worker.rs:6801`-region) and re-admission re-runs the whole
accounting block (`worker.rs:5000-5012`) and the whole prefix probe (`worker.rs:7156-7160` hit /
`7282-7286` miss). Accumulators are worker-lifetime with no offset (published at
`worker.rs:6361-6392`, surfaced at `main.rs:2250-2302`). The ledger is immune because it sets rather
than adds — which is exactly why this drifts undetected: the bill is right and the router-visible
scoreboard is wrong.

**H9 — The cache freezes at two entries at the naked budget (derived, not measured).** Three verified
facts compose: `capacity_victim()` sees probation only (`worker.rs:3029-3033`); the only unpinned
promote/`last_use` writer `touch()` is `#[cfg_attr(not(test), allow(dead_code))]`
(`worker.rs:2888-2906`), so live promotion is `pin_n` on every hit (`worker.rs:2910-2928`); and
demotion fires only when `protected_bytes > protected_target_bytes` (`rebalance_protected`,
`worker.rs:2864-2884`). Arithmetic on verified constants and the geometry receipt
(`research/cachesize-20260813/RESULTS.md:57-58` — Q27 = 156,893,184 B fixed + 29,696 B/token): at
`MEMRA_CTX=8192` an entry is 400,162,816 B, so the derived budget is 800,325,632 B (763.25 MiB)
(`worker.rs:1742-1757`, `PREFIX_CACHE_DEFAULT_ENTRIES = 2` at `:1683`) and the 80% protected target
(`worker.rs:1350`) is ~640,260,505 B. Two sold-shape entries (2 x 301,215,744 = 602,431,488 B) sit
UNDER the target, so no demotion ever fires; the third insert lands in probation as the only
probation member and `capacity_victim()` picks it as its own victim. Steady state: the first two
entries to earn a hit are immortal, every later insert pays a full 301 MB device snapshot and
self-evicts, and `prefix_inserts`/`prefix_evictions` both inflate with zero residency gained. The
byte ceiling itself holds only by that self-eviction, and the assertion behind it is a
`debug_assert!` compiled out of the serving build (`worker.rs:3170-3171`). Against the 96-key
tenant-isolated working set the capacity campaign actually runs
(`research/cachesize-20260813/RESULTS.md:17-24`) the request hit-rate ceiling is 2/96 = 2.1%.
**This is arithmetic from verified constants, not an on-box receipt** — see Q1 in §6.

**H10 — `MEMRA_PREFIX_CACHE_POLICY=lru` does not restore LRU; it makes protected unbounded.**
`prefix_cache_slru_enabled()` has exactly two consumers: `worker.rs:1828-1834`, where false forces
`protected_pct = 100`, and a startup log line. There is **no policy branch in `capacity_victim()`**.
Entries still always enter probation and still promote on first reuse via `pin_n`, so a 100%
protected share does not "put every entry in the protected segment" as the in-code comment claims —
it sets `protected_target_bytes = budget`, so `rebalance_protected()` can never demote and a promoted
cohort becomes permanently unevictable. The seam exists because
`research/slrutarget-20260813/RESULTS.md` measured LRU 75% vs SLRU 0% after a hot-cohort turnover
(quoted in `docs/FLAGS.md:525`); the rollback as shipped **degrades that same failure**. The true
global-oldest victim function is already written and unit-tested — `oldest_evictable()`
(`worker.rs:3035-3044`) — and is reachable only from `evict_all`.

**H11 — Cache DEPTH is frozen by the first entry that covers a class.** Both deepening paths sit
inside `if reused.is_none()` (`worker.rs:7281`): `snapshot_at = Some(l)` (`:7290`) and
`seed_prefix = true` (`:7293`). A hit sets `reused = Some(entry)` (`worker.rs:7163`), and
`maybe_prefix_seed` additionally returns early on `s.n_cached > 0` (`worker.rs:3679`). No insert
happens at retire. So a 100-token system-prompt-only request seeds a 100-token entry, and every later
4,860-token request with that system prompt hits it, is credited 100 cached tokens, primes 4,760, and
can never create the deep entry. `cached_tokens` is the billed quantity and the OpenRouter-published
routing metric.

**H12 — `prefix_cache_bytes` is a logical sum, spent against a driver-measured budget.**
`bytes += kb + vb` while the allocations are `kb.max(1)`/`vb.max(1)` (`worker.rs:3245-3256`), and the
derived budget is clamped against driver free VRAM (`worker.rs:1751-1757`). Small in absolute terms,
but it means the published figure cannot be reconciled against `nvidia-smi` and must never be quoted
as VRAM held.

**H13 — `MEMRA_KV_PREFETCH` unwraps a layer that some models leave `None`.**
`crates/memra-engine/src/decode.rs:2666-2671` does `cache.kv[il].as_ref().unwrap()` behind the flag,
while Gemma-4 E4B's trailing `shared_kv_layers` push `None` by construction
(`crates/memra-kv/src/lib.rs:510-517`). Correction to an earlier survey claim: the flag **is** now
documented (`docs/FLAGS.md:418`, off by default, no positive verdict on any rig, queued in
`research/tuningarms-20260813/ARM-QUEUE.md`). Whether a shipped model actually reaches that
`unwrap` with a `None` layer is **UNVERIFIED** — the shared-KV layers route through
`hybrid.rs:1423-1424`'s `kv_share` map, so the panic may be unreachable in practice.

**Guarded surfaces, verified, so nobody re-hunts them:** SWA ring refuses both snapshot and restore
(`worker.rs:3208`, `:3304`) and step35+ring is excluded at admission (`worker.rs:7082-7088`);
recurrent mid-entry split fails closed (`worker.rs:3352-3358`, `partial_prefix_decision`
`:2518-2536`); a whole-entry restore with empty logits fails closed (`worker.rs:3359-3361`), as does a
snapshot with no boundary logits (`worker.rs:3217-3219`); KV dtype/row-stride/truncation changes fail
closed pre-copy (`validate_prefix_plane_shape`, `worker.rs:2539`); layout version and pool identity
are checked on both insert and restore (`worker.rs:3085-3097`, `:3306-3320`); destination freshness
and ctx capacity are checked (`worker.rs:3333-3338`, `:3328-3331`); an oversized entry is refused
before push (`worker.rs:3105-3110`); sampler penalty history is replayed over the restored prefix
(`worker.rs:7325`); grammar state is compiled per request off-tick and never cached
(`worker.rs:7070-7080`); there is no multimodal surface, so no placeholder-token collision class
(`main.rs:2780-2790`-region).

---

## 3. How the field does it

Read from source at the revisions noted; doc-only claims are marked. vLLM
`b2506d62aec7e6bccc5959b829221a7ae217abf3` (2026-08-10), SGLang
`03c942dfab6ef5c6c0f07eebaa425405a2960dbb` (2026-08-11), TensorRT-LLM main depth-1 clone taken
2026-08-13, LMCache `0373a57393a3d9d267fdc7b96be1cfe3f6f13ffa` (2026-08-13), llama.cpp
`c73069749e3c56ca3169b2416ba1730b3d41e1a7` (2026-06-30, the local memra-era fork — **not upstream
tip**, so llama.cpp claims are scoped to that tree).

| | cache key | hit validation | invalidation triggers | eviction | sharing |
|---|---|---|---|---|---|
| **memra** | `(model, cache_ns)` for visibility + exact token vector for identity (`worker.rs:980`, `:2782-2794`) | **structural only** — version/identity/pos/bounds/freshness/strides (`worker.rs:3297-3423`); bytes never checksummed; writer program not recorded (`worker.rs:1320-1332`) | byte pressure; `evict_all` on alloc failure or host-bounce degrade; worker restart. No TTL, no epoch, no flush API (§1) | byte-budgeted SLRU, probation-only victim selection (`worker.rs:3029-3033`, `:3162-3169`) | cross-request in-process; **no** cross-worker path (task #129) |
| **vLLM** | chained content hash: `hash_fn((parent_hash, block_token_ids, extra_keys))`, sha256 default; extra keys carry LoRA id, multimodal hashes, prompt-embeds sha256, per-request `cache_salt` (`vllm/v1/core/kv_cache_utils.py:538-604`, `vllm/config/cache.py:40,98`) | **none** — `get_cached_block` is a dict lookup on the digest, no token compare (`vllm/v1/core/block_pool.py:198-223`); own docs say a weak hash "can cause undefined behavior or even leak private information" (`docs/design/prefix_caching.md:25-30`) | `reset_prefix_cache` (refuses if any block in use, `block_pool.py:763-780`); `_reset_caches` on weight update (`engine/core.py:810-823`); stale shorter partial-block hash removed on longer registration (`block_pool.py:284-299`) | block-count LRU over a free queue; ties evict the longer-chain block first (`kv_cache_utils.py:178-195`, `block_pool.py:647-731`) | in-process global block pool; cross-worker only via a KV connector + `kv_events`, and NOT with the default random `NONE_HASH = os.urandom(32)` (`kv_cache_utils.py:95-114`) |
| **SGLang** | radix tree on **literal token ids**; `extra_key` namespaces `child_key`; SHA-256 chained `hash_value` only for the L2/L3 storage tier (`radix_cache.py:161-213`, `mem_cache/utils.py:106-131`) | **token-exact by construction** locally (`RadixKey.match`, galloping+binary exact compare, `radix_cache.py:161-196`); a cross-namespace compare RAISES (`:153-159`). Remote L3 bytes are **not** re-verified | node removal on eviction; ref-counted `lock_ref` protects in-use nodes; L3 metadata deliberately not cached — queried live (`hicache_design` doc) | priority heap over evictable leaves, LRU by `last_access_time` default; `inc/dec_lock_ref` moves bytes between evictable/protected (`radix_cache.py:562-627`) | in-process tree; L2 host / L3 cluster storage keyed by SHA-256 hex + tp/pp/cp rank, MLA and layout flags (`hicache_storage.py:27-41`); gateway routing is approximate-only (`policies/cache_aware.rs`) |
| **TensorRT-LLM** | `unordered_map<BlockKey, BlockPtr, BlockKeyHasher>` — 64-bit non-crypto hash selects a bucket, `BlockKey::operator==` compares `uniqueTokens` elementwise plus `usesExtraIds`, `loraTaskId`, `extraKeys`, `cacheSalt` (`kvCacheManager.h:302`, `blockKey.h:42-53`) | **hash-then-verify-tokens** — the cleanest in the survey; partial reuse via `numMatchingTokens` (`std::mismatch`, `blockKey.h:88-102`); partial matching disabled under multimodal extra keys (`:78-86`) | structural refusals rather than invalidation: partial reuse only for an unreferenced leaf or with copy-on-partial-reuse (`kvCacheManager.cpp:399-440`); prefix-only match = miss for transfer/pinning with pin rollback (`:2210-2229`) | LRU stratified by retention priority: `mFreeQueues[cacheLevel][priorityIdx]`, plus `mSecondaryOffloadMinPriority` (`evictionPolicy.h:37-126`) | in-process reuse trie; cross-instance via connector hashes chained through `mPrevBlockInSeq`, fenced to beam-width 1 (`kvCacheManager.h:2295-2317`) — 64-bit non-crypto identity, unverified |
| **LMCache** | `model@world_size@worker_id@chunk_hash@dtype` where `chunk_hash` is a **64-bit truncation** of a chained SHA-256 (`utils.py:397-456`, `token_database.py:34-57`); note the in-tree comment "Ignore extra keys for now" (`:282-284`) | **none on the prefix path.** The one hash-then-verify design is `BlendIndex` (fragment lookup): rolling 64-bit fingerprint probe then `np.array_equal` on token ids, fail-closed (`mp_coordinator/blend_index.py:64-72`, `:180-222`). `audit_connector` sha256s retrieved BYTES (transport integrity), default off (`:216-224`, `:328-340`) | none content-driven; logs ERROR "This will cause incorrect KV cache transfer" if `PYTHONHASHSEED` is unset under P/D disaggregation (`token_database.py:309-320`) | tiered backends; chunked at 256 with `save_unfull_chunk` controlling tail admission (doc-level) | this is the dedicated cross-worker product; identity crosses the process boundary as a 64-bit digest with no verification |
| **llama.cpp** (local fork) | the **token vector itself**; matching is `get_common_prefix`, a token-by-token LCP scan (`tools/server/server-common.cpp:471-519`) | **the only engine that validates the restored state** — and only structurally: restore byte-count round-trip (`server-task.cpp:1704-1746`), layer count, cell count, V transposition, per-layer ggml type, key row size (`src/llama-kv-cache.cpp:2198-2400`), GGSQ magic+version for the file path (`src/llama-context.cpp:3045-3085`). Failure clears the slot (`server-context.cpp:1668-1672`) | **subsumption on tokens**: on save, skip if already contained and ERASE every cached entry contained in the new prompt (`server-task.cpp:1623-1652`); load requires `f_keep = lcp/entry_len >= 0.25` and moves the entry out of the cache (`:1678-1700`) | front-of-deque under **dual** byte and token budgets, always keeping one entry, with a `bad_alloc` self-shrink to `0.4*size()` (`server-task.cpp:1646-1660`, `:1753-1790`) | one prompt cache shared by all slots; slot choice and cache save/restore are ONE decision (LCP-similarity selection, `server-context.cpp:1590-1676`); no cross-worker path |

**Where memra is stronger.** Token-exact identity with no load-bearing hash (`worker.rs:2782-2794`)
puts it alongside SGLang and llama.cpp and ahead of vLLM, LMCache and TRT-LLM's connector path, all
of which trust a digest — down to 64 bits in two cases. Its pre-copy structural preflight
(`worker.rs:3297-3423`) is more thorough than anything except llama.cpp's.

**Where memra is weaker, plainly.** Four things:

1. **Tenant identity is a lookup FILTER, not part of the key.** vLLM injects `cache_salt` into the
   first block's hash so a cross-tenant match cannot be *produced*
   (`docs/design/prefix_caching.md` §Cache Isolation for Security, RFC vllm#16016), explicitly
   against TTFT-timing inference of other users' cached content (NDSS'25 "I Know What You Asked",
   arXiv 2409.20002, 2508.08438). A filter can be bypassed by any future code path that forgets it —
   and H6 shows the un-scoped default is what the builders construct.
2. **No engine validates the restored state numerically, and no engine keys on the CONSUMING
   program** — but the other four manage the class by **structural avoidance**, and memra does not:
   llama.cpp restores a whole sequence state and runs the suffix through the ordinary prefill path;
   TRT-LLM refuses partial reuse unless the block is an unreferenced leaf or is copied, and refuses
   partial matching under content-dependent keys; vLLM `drop_eagle_block` **shortens the accepted hit
   so the divergent boundary is always recomputed**
   (`vllm/v1/core/single_type_kv_cache_manager.py:764-778`). memra's response to the same hazard was
   to default a flag off (`docs/FLAGS.md:527`) and leave the whole-entry arm open (H1).
3. **No graceful degradation or dual budget.** llama.cpp holds byte AND token limits and self-shrinks
   its limit on `bad_alloc`; memra's byte ceiling in the serving build rests on a self-evicting
   insert and a compiled-out `debug_assert!` (H9).
4. **No subsumption maintenance.** llama.cpp erases contained entries on save so depth ratchets
   upward (`server-task.cpp:1623-1652`); memra's `has_covering` does the opposite and locks a class
   at its first depth (H11).

**Accounting comparison that must be stated wherever memra's cached-input billing meets a
competitor's:** both hashing engines report the TRUNCATED hit. vLLM rounds `hit_length` down to
`alignment_tokens` and applies the eagle reduction before the number becomes `num_computed_tokens`
(`single_type_kv_cache_manager.py:764-778`); SGLang page-aligns every key and floors the match to
`page_size` (`radix_cache.py:135-138`, `:190-196`). Their published example is explicit: 10 shared
tokens hit as 8. memra's geometry is per-token linear
(`research/cachesize-20260813/RESULTS.md:57-58`), so memra's `cached_tokens` is a **different
quantity in kind** — finer-grained and not block-quantized.

**Two field mechanisms worth importing on their own merits:** a per-entry checksum verified at
scheduling time (arXiv 2604.17249 — 13 of 16 BF16 bit positions produce coherent-but-altered output,
cumulative in requests served, bounded to one batch by a checksum at "negligible overhead"), and
FLOP-per-byte utility eviction (Marconi, MLSys'25 — and Marconi is also the published statement of
why memra's hybrid layers cannot roll back a recurrent state partially, which memra already enforces
at `worker.rs:3352-3358`).

---

## 4. Ranked recommendations

Correctness first. "One-program" column = interaction with CLAUDE.md §One numeric program per
request. Efforts are agent-hours.

### Correctness

| # | change | seam | gate | effort | one-program |
|---|---|---|---|---|---|
| **R1** | Exclude eager-only models from the whole-entry hit, the continuation resume, and the seed — `&& !eager_only_model(lm)` at the hit guard, matching filters on the resume and seed | `worker.rs:7126`, `:6892-6898`, `:3674-3686`; precedent already in-tree at `worker.rs:8028` (`!eager_only.contains(&s.model)`), set built at `:4328-4330` | R2's gate with a Gemma-4 arm; refusal counters (R21) so the lost hits are visible | 4h | **Closes** H1 by making the crossing impossible. Costs Gemma-4 its prefix cache until R4 restores mono-to-mono hits legally |
| **R2** | Promote `restore_oracle.py` into a wired serving-shape bit-identity gate under `tools/`, add arms for the four uncovered credit sites (sequential whole-entry hit `worker.rs:7156`, continuation/plain-affinity `:6892`/`:7002`-region, spec resume `:7542`) plus an eager-only arm | `research/cachesize-20260813/restore_oracle.py:135-172` (already asserts `cached_tokens == len(prompt)` AND `text_sha256 == cold baseline`, plus exact counter deltas); wire into `tools/serve-smoke.sh:132` → `tools/local-ci.sh` | itself; `--canary` must inject one wrong suffix token and require RED | 10h | The proof vehicle for everything else. Today the only wired accounting gate fires one simultaneous burst (`tools/cache-meter-gate.py:11-21`), so it exercises `promote_miss_to_hit` alone — the sequential restore a paying hit takes has no gate. Violates H100 LAW 3 today |
| **R3** | Retract the seed capture by `PRIME_MIN_T`: capture at `prompt.len() - 16` so every hit has a legal full-prime suffix and `last_logits` is always recomputed by the consumer | `worker.rs:3674-3686` (writer to move), `:7141` + `:143-149` (the door this deletes), `prefill_tick_take` `:93-117` (`bound_rem` residual = FLOOR, so no sub-floor tail), `:8269-8270` (the veto it must clear, not trip) | R2 on an exact repeat + `run-spec` K=1..8 | 8-12h | **Closes** H2 for the dominant insert path. Same shape as vLLM `drop_eagle_block`. Costs 16 of 4,860 billed cached tokens (0.33%). **Must land with the W1 boundary fix (#125)** or it makes that door more reachable. Flips `cached_hit_needs_first_token` always-false, so re-measure hit TTFT (current claim 1.3 ms, `research/prefixmoney-20260812/`) |
| **R4** | Store a `ProgramClass` in the entry (prime shape mono/chunked/concat-fanout, `eager_only_model`, pp stage+device topology, step35 arm verdict, graph-vs-eager, the per-call env arms) and refuse a restore whose writer class differs | `worker.rs:3274-3288` (entry fields), `:3306-3320` (the fail-closed identity block to extend), snapshot call sites; `hybrid_forward.rs:9159-9195`, `:9173`, `:9183`, `:727-733` | R2 with a cross-class arm proving refusal; R21 counters to see the hit-rate cost | 16h | **Closes** H4 and generalises H1. No surveyed engine keys on the consuming program (§3) — this would be ahead of the field, not behind it. Design call to record: refuse-and-recompute (needs R6) vs partition-the-key (costly at a 2-entry budget) |
| **R5** | Make the absent-layer set identity-based: assert it is a subset of the NextN indices, store the bitmap, check it on restore | `worker.rs:3232-3239`, `:3385-3388`, `:3420`; `prefix_snapshot_trunk_layers` `:1712-1718` already computes the partition | unit tests BOTH ways — fabricated trunk-layer-at-len-0 returns `Err`, AND every currently-inserting model still inserts (`prefix_inserts > 0` per model) | 6h | **Closes** H3. Regression risk against `96361c531` itself if the NextN derivation is wrong for any model — hence the two-sided test |
| **R6** | `px.remove_at(&pool_key, i)` in the non-alloc restore `Err` arm, with a counted reason | `worker.rs:7169-7174` (and the partial arm `:7262-7275`); `remove_at` refuses only pinned entries (`:2992-2996`) and the entry is unpinned there | R21 counter + a unit test that a poisoned class re-seeds | 2-3h | **Closes** H5. Becomes load-bearing the moment R4 or R7 return `Err`. Do NOT widen to `evict_all` |
| **R7** | Device-side per-plane checksum folded during the insert copy, verified inside the restore preflight on the CUDA owner thread | `worker.rs:3245-3256` (bytes already being touched), `:3306-3338` (preflight); `Engine::reduce_slots` is the kernel-shape precedent. Do NOT reuse `prefix_entry_state_digest` (`worker.rs:3446+`) on the hot path — it D2Hs every plane, ~301 MB against a 51 ms TTFT p50, and `docs/FLAGS.md:528` calls it intentionally expensive | R2 plus a real canary: poke one byte in a resident plane through a test hook, require refusal | 14h | Orthogonal: catches CORRUPTION of stored bytes, **not** a wrong-program restore. Mis-selling it as the latter would be worse than not having it. Would have caught the zero-insert class as a validation failure. New kernel ⇒ both-rigs measurement before default |
| **R8** | Tenant namespace by construction (a `TenantNs` newtype so an un-scoped `cache_ns` is unconstructible) + `flush_namespace` called on revoke | `main.rs:2952`, `:3122` (un-scoped default), `:3686`, `:3861` (post-hoc repair), `:3663`/`:3821` (`tenant_namespace`); `auth.rs:372-402`, `:471-473`; `worker.rs:980` | a gate arm asserting a second tenant with the same salt gets `cached_tokens == 0`, and that a revoked tenant's entries are gone | 8h | No numeric interaction. Task #117 is adding exactly the third admission route this construction cannot survive. Pinned entries survive a flush (`worker.rs:2994-2996`) — document that residual, don't paper over it |
| **R9** | Re-emit `PromptUsage` after the fanout credit; add a streaming-disconnect ledger arm to the gate | `worker.rs:5004-5007` (sole emitter), `:8162-8171` (post-admission credit); `ledger.rs:1757-1769` (set-not-add makes re-send safe), `:1867-1889`; `tools/serve-smoke.sh` never sets `MEMRA_REQUEST_LEDGER` | new gate arm asserting the durable row's `cached_prompt_tokens == prefix_len` | 7h | None. Confirm both consumers are idempotent (`main.rs:4115`, `:4420`) before re-sending. Overcharging on first Onlist traffic is the worst direction for this error |
| **R10** | Carry a `readmitted` flag so the accounting block and prefix counters run exactly once per request | `worker.rs:6801`-region (`park_requeue` reuses `tx`), `:5000-5012`, `:7156-7160`/`:7282-7286`, publish `:6361-6392` | `/metrics` invariant `prefix_cache_hits + misses <= n_admitted` in the gate | 4h | None. Inventory any other shared-`tx` re-admission before asserting the invariant |
| **R11** | Replace the `MEMRA_KV_PREFETCH` `unwrap` with a let-else skip; keep or kill the flag on its queued cell | `decode.rs:2666-2671`; `memra-kv/src/lib.rs:510-517`; `docs/FLAGS.md:418`; `research/tuningarms-20260813/ARM-QUEUE.md` | existing decode gates; a decode-throughput cell on both rigs if it is ever promoted | 2h | None. Reachability UNVERIFIED (Q5); fix regardless, since the flag holds no positive verdict on any rig |

### Hit rate

| # | change | seam | gate | effort | one-program |
|---|---|---|---|---|---|
| **R12** | `capacity_victim()` falls back to `oldest_evictable()` when probation is empty and the budget is still exceeded | `worker.rs:3029-3033`, `:3035-3044`, `:3162-3169` | offline sim arms (R22) on the slrutarget hot+scan shapes; extend the eviction unit tests near `worker.rs:11397+` | 4h | None — victim choice only. Weakens scan resistance slightly; keep strict ordering (protected only when probation is empty) to preserve the SLRU argument |
| **R13** | Branch `capacity_victim()` on `prefix_cache_slru_enabled()` and return `oldest_evictable()` in the LRU arm; correct `docs/FLAGS.md:525` in the same commit | `worker.rs:1825-1858`, `:3029-3044`; reference impl `research/slrutarget-20260813/simulate.py:42-80` | R22 turnover shape must reproduce LRU 75% | 3h | None. Fixes H10 — today the advertised rollback degrades the failure it exists for |
| **R14** | Derive the budget from the tenant working set, not `MEMRA_CTX`: `max(2, working_set) * entry_bytes(typical_prompt_len)` under the existing clamp | `worker.rs:1679-1690`, `:1742-1757`, `:980`; `auth.rs:471-473` | #124's budget sweep on the repaired tip | 6h | None. Largest single hit-rate lever and needs no new mechanism. Intercept arithmetic bounds it: covering K Q27 classes costs at least K x 156,893,184 B at any depth, so 59 resident classes (Morph's published 61.2%) is ~9.26 GB of intercept alone against a 763.25 MiB naked default. **Fix R10 first** or the sizing receipts inflate |
| **R15** | Token-only GHOST entries (no device bytes) keyed by the same `PoolKey`: second-sighting admission + a live infinite-capacity hit-rate oracle | `worker.rs:7281-7295`, `:2798-2810`, `:2594`-region (mirror the map shape without `kv/conv/ssm`); `lcp_hist` already has a publication path (`worker.rs:6392`, `main.rs:2293`) | counter arms in R2's gate; sim arm in R22 | 10h | **Zero** — ghosts carry no KV and are never restored. Needs its own LRU cap and byte accounting, and inherits the tenant-isolation obligation (never a global list). A 4,860-token ghost is 19,440 B: 15,494 ghosts for the price of one Q27 entry |
| **R16** | Unfreeze depth: on a hit, arm a deepening seed at prefill-done and drop the subsumed shallower ancestor (llama.cpp's rule, `server-task.cpp:1623-1652`) | `worker.rs:7281` gate, `:7290`/`:7293`, `:3674-3686`, `:2811-2820` (`has_covering`), `:3211-3216` (boundary precondition already holds) | **serving-shape** proof that a deepened entry is bit-identical to a cold-primed entry of the same depth (R2's `text_sha256` arm), not a unit test | 10h | **Hard interaction.** The deeper entry's provenance is chained (restore + continuation prime), which is exactly the property `worker.rs:1320-1332` admits. Two requirements: refuse deepening for `eager_only_model` (else it multiplies traffic onto H1), and gate on bit-identity. Do **not** ship before R1+R3 |
| **R17** | Replace the flat 64-token floor with a per-model bytes-efficiency floor `max(absolute_floor, alpha * F/m)` from the already-computed geometry | `worker.rs:1345`, `:7288-7290`, `:3679`, `:2787`/`:2814`, `:1720-1748` | R22 arms; per-model insert assertions | 6h | None. Mechanism argument from measured geometry: Q27 `F/m` = 5,283 tokens, Q35 ≈ 7,097, so a 64-token Q27 entry is 98.80% fixed overhead (2.48 MB per cached token vs a 29.7 KB marginal rate). Under budget B held as k entries, cached tokens = (B - k·F)/m — 16,383 at k=2, 5,816 at k=4, 534 at k=5, i.e. **depth strictly dominates breadth** for the billed quantity. Do NOT move the same constant in `prefix_fanout_groups`' first-64 equality prefilter (`worker.rs:2676-2711`) |
| **R18** | Add `hits: u64` to `PrefixEntry` and choose victims by GreedyDual-Size-Frequency (`L + hits * toks/bytes`, optionally latency-denominated) | `worker.rs:3274-3288`, `:3029-3033`, `:2830` (recency is the entire current signal), `:2992` (`remove_at` must keep any new index exact) | R22 first (no GPU lock needed), then both rigs per §Per-hardware arm selection | 12h | None. The axis memra can compute EXACTLY where the field estimates: at the sold shape Q35 delivers 4.379e-5 cached tokens/byte vs Q27's 1.613e-5, a 2.71x spread, so a size-blind LRU over one global budget systematically over-retains the expensive model. The GDSF aging term is not optional, and the `(Instant, id)` tie-break determinism must survive |
| **R19** | Per-pool byte fair share: pick the victim from an over-quota pool first, global order otherwise | `worker.rs:2594`-region, `:2596-2612` (the doc stating targets are deliberately global), `:3029`, `:980`; per-tenant publication already exists at `main.rs:2320-2331` | R22 multi-tenant arm | 10h | None. Whether aggregate hit rate should be traded for a per-tenant floor is an **owner call**, not an engineering one — `cache_hit_token_ratio` is already published per tenant, so unfairness is customer-visible |
| **R20** | Pre-snapshot admission preflight (`would_retain(predicted_bytes)`) so a doomed insert never pays the copy | `worker.rs:3610`-region (snapshot-then-insert), `:3201-3290`, `:3122-3150` (the pins-only preflight to mirror), `:1742` (`model_prefix_entry_bytes` predicts exactly) | counter arm: inserts/evictions stop moving in lockstep | 4h | None. Removes a per-miss 301 MB allocate-copy-free round trip in the saturated regime, and makes `prefix_inserts`/`prefix_evictions` mean what an operator reads. Keep the post-copy `e.bytes > budget` refusal (`worker.rs:3105-3110`) as the authoritative backstop |
| **R21** | Refusal-reason counters: no-entry vs program-class mismatch vs eager exclusion vs restore `Err` vs each `PartialPrefixDecision` refusal; plus eviction bytes/reasons, per-entry hits, resident-depth, and the four-tier split inside `cached_tokens_in` | `worker.rs:7282-7286` (undifferentiated miss), `:2518-2536` (reasons named but uncounted), `:7169`, `:6361-6392`; `main.rs:2284-2302` | it IS the instrument for R1/R4/R12-R19 | 3-5h | None. Two honesty constraints: `prefix_cache_bytes` is logical, not VRAM (H12); per-tenant fields stay operator-only (timing-inference surface). Also fixes a wrong gate criterion — `research/sellgate-20260812/reduce.py:367-370` enforces `prefix_cache_hit_tokens_drift == 0`, which is a property of the research workload (every prefix in its own namespace, `research/cachesize-20260813/RESULTS.md:17-24`), not an invariant, so it will red on legitimate session-affinity traffic |
| **R22** | Score every eviction arm offline first — extend the existing simulator with `capacity_victim` fallback, GDSF, ghost-admission, per-pool-quota and infinite-capacity-oracle arms | `research/slrutarget-20260813/simulate.py:42-165`, `:285-292` (arm registry), `traffic_model.lock.json` | itself; arms must stay bit-faithful to the shipped code (the current LRU arm models a policy the product cannot produce — H10) | 8h | None, and it takes **no GPU lock**. Scored GPU work serializes behind `flock /tmp/memra-gpu.lock` and one attempt of this exact lane was already discarded to a co-tenant (`research/cachesize-20260813/raw/attempt1-gpu1-overlap/`). A simulator result is a POLICY result, never a perf result, and it cannot see any correctness item |

---

## 5. Cheapest high-value item

**One function, ~6 lines: give `capacity_victim()` a protected-segment fallback and a policy branch.**
This is R12 and R13 in a single edit, and it fixes two independent freezes.

```rust
// crates/memra-server/src/worker.rs:3029
fn capacity_victim(&self) -> Option<(PoolKey, usize)> {
    if !prefix_cache_slru_enabled() {
        return self.oldest_evictable();          // R13: the advertised plain-LRU rollback
    }
    self.probation_lru.values().next().cloned()
        .or_else(|| self.oldest_evictable())     // R12: protected only when probation is empty
}
```

`oldest_evictable()` (`worker.rs:3035-3044`) is already written, already handles both indexes with a
deterministic `(Instant, id)` tie-break, and is already unit-tested through `evict_all`
(`worker.rs:11516-11520`). The fallback fires only when the budget is still exceeded and probation is
empty, so the scan-resistance argument SLRU was added for is preserved. Ordering inside the eviction
loop is unchanged (`worker.rs:3162-3169`).

What it buys. Without it, at the naked derived budget the cache freezes at two immortal entries and
every later insert self-evicts after paying a full 301 MB device snapshot (H9) — a 2/96 = 2.1%
request hit-rate ceiling on the working set the capacity campaign actually runs, against the field's
published 61.2%. And `MEMRA_PREFIX_CACHE_POLICY=lru`, which `docs/FLAGS.md:525` advertises as the
remedy for the one measured losing shape (LRU 75% vs SLRU 0%,
`research/slrutarget-20260813/RESULTS.md`), currently makes protected unbounded and degrades that
same failure (H10). Cached input is 50.8-62.2% of the bill.

Ship it with: the `docs/FLAGS.md:525` correction in the same commit, extended eviction unit tests,
and an R22 simulator run on the slrutarget hot / scan / turnover shapes — none of which needs a GPU.
It carries **no** interaction with the one-numeric-program rule: victim selection cannot change any
token's arithmetic, only whether a later request hits.

---

## 6. Open questions needing a measurement

- **Q1 — Does the two-entry freeze actually occur on box?** H9 is arithmetic from verified constants
  (`worker.rs:1683`, `:1742-1757`, `:1350`, `:3029-3033`, `:2864-2884`) plus the geometry receipt; no
  on-box receipt shows `prefix_inserts` climbing with residency pinned at 2. One instrumented cell on
  the #124 campaign settles it, and R21's counters make it self-evident thereafter.
- **Q2 — Do the two step35 FA prefill kernels differ bitwise at `seq_end <= win`?** UNVERIFIED, no
  receipt either way (`hybrid_forward.rs:9159-9195`). Determines whether the attention arm must be
  part of R4's program class or is precautionary.
- **Q3 — Is the Gemma-4 whole-entry-hit-plus-suffix arm output-divergent?** splitiso measured the
  partial-restore arm (`research/splitiso-20260813/RESULTS.md`). The `eager_mono && carried` boundary
  is identical by inspection (`worker.rs:8238`, `:8269`), but the receipt does not cover this arm. R2
  with a Gemma-4 arm answers it directly, and R1 should ship regardless.
- **Q4 — What is the real working-set size and the infinite-capacity hit ceiling on live traffic?**
  R15's ghost oracle answers this continuously without GPU time, and it is the exact justification
  R14 needs before a budget increase.
- **Q5 — Is the `MEMRA_KV_PREFETCH` `unwrap` reachable for any shipped model?** The E4B shared-KV
  layers route through `hybrid.rs:1423-1424`'s `kv_share` map, so the `None` slot at
  `decode.rs:2667` may be unreachable. One boot with the flag on per model settles it.
- **Q6 — Cost of a device-side per-plane checksum on both rigs.** R7 is only a mechanism win if the
  fold is bandwidth-bound and sub-millisecond over 301 MB; the existing D2H+sha256 path is ~301 MB of
  transfer against a 51 ms TTFT p50 and must not be used as the estimate.
- **Q7 — Do the inherited boundary logits actually change token 1?** Capture `prefix_logits_digest`
  for the same prompt primed under c=1, c=4 and concat-batch shapes
  (`MEMRA_PREFIX_SPLIT_TRACE=1`, `docs/FLAGS.md:528`) and compare. If they differ, H2 is a measured
  defect rather than an unproven crossing, and R3's priority rises above R4's.
- **Q8 — Is `prefix_cache_hit_tokens == cached_tokens_in` nonzero-drift on conversational traffic?**
  Session affinity is ON by default and credits the same `s.n_cached` (`worker.rs:5000-5012`), so the
  sellgate criterion (`research/sellgate-20260812/reduce.py:367-370`) is expected to red on real
  traffic. Measure the drift before anyone "fixes" it by suppressing a real signal.
