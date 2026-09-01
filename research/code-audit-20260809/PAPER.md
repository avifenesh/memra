# Internal code audit — memra serving path (2026-08-09)

**Audited tree:** `/home/avifenesh/projects/wt-public-split`, branch `restructure/public-split`,
commit `74afcaf6c7e2610b5f7e79f778040d0168c78028`. Read-only audit; no code changed, nothing committed.
**Scope:** `crates/memra-server/src/{worker.rs,main.rs}`, `crates/memra-engine/src/{lib.rs,spec.rs,pp.rs}`
plus the engine files the depth math actually lives in (`decode.rs`, `decode_batch.rs`,
`hybrid_forward.rs`, `cu/spec_sample.cu`, `memra-kv/src/lib.rs`).
**Lens:** two live bugs — the cachespec frozen-prefix / parked-pin receipt
(`research/cachespec-20260809/RESULTS.md`) and the longdepth token-soup hunt
(`research/longdepth-20260809/PROGRESS.md`).

Notation: each finding is CONFIRMED (path traced end to end at these line numbers) or
SUSPECTED (needs a repro). `file:line` is against the audited commit. NOTES.md carries the
full working log (findings 1–30) including confirmed non-findings and the state/memory table.

---

## Exec summary — top findings by (severity x likelihood)

Ranked; the number in brackets is the section below.

0. **[3.1] `u01()` can return exactly `1.0f` → Gumbel noise `+inf` → a uniformly random
   token always wins** — CRITICAL x certain-at-depth. CONFIRMED ROOT CAUSE of the longdepth
   token-soup bug. `cu/spec_sample.cu:36–39`. Temperature-only (greedy returns before
   Philox), P=3.8e-3/token at step35's vocab → E[first hit] ≈ 261 tokens vs the receipt's
   observed token 281. This is the single most important finding in the audit — it closes a
   live corruption lane. Detail in Area 3.
0b. **[5.7] The only production `unpin` is compiled out of every release build** — CRITICAL x
   certain. `debug_assert!(px.unpin(&pin))` (`worker.rs:3354`): with no `[profile.release]`
   override, release builds compile out the whole macro ARGUMENT, so prefix pins NEVER
   release in any shipped/gated binary. Empirically verified (`rustc -O` runs the side effect
   0 times). Served/fanout entries become permanently un-evictable → budget stops bounding →
   pinned inserts refused → a second independent cause of the cachespec freeze, plus a hard
   `cache alloc failed` where a retry was promised. Corrects my own initial MED rating.
1. **[6.1] Admission defers without reclaiming dormant parked state** — CRITICAL x high.
   The gate reads `free + pool_cached`, counts the parked entries in its own log line, and
   queues the request anyway. This is the direct cause of the receipted 25 s c=4
   serialization. Fixing this one restores throughput.
2. **[6.2] `session_vram_cost` is one first-admit scalar; ctx_cap is per-request** —
   CRITICAL x high. A short first request calibrates a ~119 MB bar that then waves through
   ~11.2 GB/session 256k admits (guaranteed step-OOM); a 256k first request over-gates
   everything. The receipted 1.49x understatement is the small end of this.
3. **[1.3 / 6.3] Parked plain caches are full-cap and never reclaimed** — CRITICAL x high.
   The `reuse` pool has no eviction hook anywhere except a same-key LRU; entries are
   `cap = cache.max_ctx` (~11.2 GB at 262144). This is the ~22 GB pin term of the receipt.
4. **[1.7] Unbounded, client-controlled parked-pool key space** — CRITICAL x medium.
   `reuse`/`spec_reuse` key on `(model, cache_ns)` with `cache_ns` = raw client
   `cache_salt`; the "cap 2" is per key, and nothing bounds the number of keys. A
   salt-spraying unauthenticated client parks unbounded full-cap caches = remote VRAM DoS.
5. **[5.5] A step error at budget is laundered into a clean `MaxNew`/200** — HIGH x
   medium. The single highest-value honesty bug: a CUDA fault on the last step becomes
   `finish_reason: "length"`, invisible to every error-rate view.
6. **[4.1/4.2] max_tokens overshoot + spec Token coalescing** — HIGH x high (both are
   the cachespec/longdepth receipts). Emission runs past budget (769–770 of 768); one
   Event::Token carries a whole spec round under the last token's id (803 ids for 2048).
7. **[4.3] tokens_out / lane_tokens / step_stats miss the spec + round-robin paths** —
   HIGH x certain. `/metrics.tokens_out = 0` under spec (receipt), and two scheduler gates
   read stale/empty percentiles on a spec-only box.
8. **[5.6 / 1.6] Requests during worker respawn silently queue; streaming worker-death is
   a truncated 200** — HIGH x medium. `/readyz` knows the worker is down; the inference
   handlers never consult it, so the retry contract is absent exactly when needed.
9. **[1.1] Spec turn-1 prime is monolithic** — HIGH x medium (gated off today by PP-2
   K=0). One 256k spec prompt freezes the whole scheduler for the entire prime.
10. **[4.6/4.7/4.8] No terminal finish_reason on abort / mid-stream error / worker death**
    — HIGH x high. Three mid-stream terminations all reach the client as a truncated 200.
11. **[3] Corruption-hunt signal: the bug is in the SAMPLING path, not the position math**
    — the depth-arithmetic candidate list, ranked against the greedy-clean/temp-dirty
    constraint. Deliverable for the longdepth lane.

Structural theme behind most of the metering/streaming findings: accounting was written
per dispatch-path and the newest paths (spec burst, round-robin, graph, park re-admit)
were not wired in. One test that drives a request through each arm and asserts the same
counter set moved would have caught findings 4.3, 4.4, 5.2 at authoring time.

---

## Area 1 — Session lifecycle

### 1.1 CONFIRMED — spec turn-1 prime is monolithic: one 256k prompt freezes the whole scheduler
`worker.rs:4909` — `let suffix: Vec<u32> = s.prefill_queue.drain(..).collect();` (spec arm of
`step_session`), handed whole to `generate_spec_session_*` (`worker.rs:5000–5004`). The plain
path's per-tick chunking (`PREFILL_TICK_T = 1024`, `worker.rs:46`) never applies to spec.
Scenario: a spec-eligible 100k+ prompt admits; the worker spends the entire prime inside one
engine call — no disconnect sweep, no admission, no peer decode tick runs until it returns; the
`on_commit` yield fires only at round boundaries, which begin after the prime. Gated off today
(PP-2 placement chooses K=0) but inherited by any spec-on single-card long-context serving.
Fix shape: give the spec turn-1 prime the plain path's chunked treatment — prime the suffix
through the tick loop into the spec trunk cache before entering the burst, re-entering the
scheduler between chunks; at minimum bound the first burst's prime chunk.

### 1.2 CONFIRMED — aborted (client-disconnected) sessions still park full-cap caches
`worker.rs:2418–2424` — the disconnect sweep pushes the index onto `finished` and nothing else;
`worker.rs:2418` comment even states "Retire still parks reusable KV". The retire sweep
(`worker.rs:3408–3437`) then parks any plain session with `fed.len() >= 16 && prefill_done` at
`cap = cache.max_ctx`. An abandoned conversation is the entry least likely to be resumed yet
parks at full priority and (given 1.3/6.1) is never reclaimed. A client opening and cancelling 4
long-context streams pins ~22 GB nobody will match. Fix shape: mark aborted retires; skip
parking them, or park only when bound to an explicit affinity id (`session_id`), which the spec
tier already computes at `worker.rs:3387–3392`.

### 1.3 CONFIRMED — the plain `reuse` pool has no eviction path and parks at full ctx_cap
`worker.rs:3430` — `let cap = cache.max_ctx;` then `pool.push(ReuseEntry { fed, cache,
last_logits, cap })` (`worker.rs:3432`). Every mutation of `reuse` in the file is: the park
(`worker.rs:3423`), the same-key LRU at park (`worker.rs:3426–3429`
`while pool.len() >= reuse_pool_per_model().max(1) { pool.remove(0); ... }`), and the consuming
probe (`worker.rs:3727`). There is no `retain`/`clear`/headroom-driven eviction — every
headroom eviction in the file targets a different tier (`px.evict_all()` at 3854/4204, spec
`p.clear()` at 4092/4103). So a parked full-cap entry is freed only by a new same-`PoolKey`
park or an exact-prefix match (`prompt.starts_with(&e.fed)`, `worker.rs:3730`); rewritten
history never matches, so the entry is immortal. This is the ~22 GB receipt term. Fix shape:
right-size at park (a parked entry needs `fed.len()` of KV, not `ctx_cap` — the
`prefix_snapshot` byte-copy machinery at `worker.rs:1600+` already fits KV planes), and give
the pool a global byte budget mirroring `prefix_cache_budget_bytes()`, registered in the same
headroom-discipline path as the other tiers.

### 1.4 CONFIRMED — step-OOM park can RETAIN the VRAM it claims to free
`worker.rs:2818–2820` comment: "its caches drop (freeing exactly the VRAM the retry needs)".
The park (`worker.rs:2845–2849`) pushes the index onto `finished` → the retire sweep. For a
POOL-RESUMED spec session that OOMs on its first burst, `committed.len() >= 16 &&
next_pred.is_some()` (`worker.rs:3380`) both hold (`next_pred` is set on every normal verify,
`spec.rs:3136`), so the session is re-parked into `spec_reuse` — retaining the very cache the
retry needs — and the re-queued request must clear the VRAM gate (which cannot see the parked
entry will be consumed by the pool probe inside `admit`). A genuinely fresh session that OOMs
mid-prime has `committed` empty → drops → comment holds for that case only. Self-heals when
`active` empties (`worker.rs:2308` gate skipped) — serialization, not deadlock. Secondary: the
retire sweep runs `spec_flush_pending` (a trunk pass) on the OOM-parked session
(`worker.rs:3374–3378`), a device call fired right after a card-full OOM. Fix shape: make the
OOM-park path explicitly destructive — take `s.spec`/`s.cache` out at the park site and drop
them before `finished.push(i)`, so the comment's "re-primes from scratch" contract holds.

### 1.5 CONFIRMED — a parked (step-OOM re-admitted) request double-counts input metering
`park_requeue` (`worker.rs:3564–3594`) rebuilds the Request and re-injects it through the
normal admission path, so `n_prompt_in`/`n_cached_in` (`worker.rs:2377–2378`) and
`meter_account` (`worker.rs:2381`) all run twice for one client request. Client `usage` is
correct (from the final Done); the server aggregate silently diverges from the sum of what
clients were told — worst under memory pressure, where parks cluster. Fix shape: flag the
rebuilt Request as a re-admission (the park path already constructs it) and skip the
admission-time input counters when set; `n_step_oom_parks` stays the record that a park
happened.

### 1.6 CONFIRMED — worker panic drops in-flight sessions and streams with no terminal event
`worker.rs:2108–2109` — `active` and `queue` are `run()` locals; a scheduler panic unwinds and
drops every Session (its `tx` closes with no `Event::Error`) and every queued Request. The
supervisor (`worker.rs:5386–5483`) keeps the cmd Receiver alive and respawns. Non-streaming
clients get an honest 503 (channel closed without Done → `worker.rs:3073–3080`); STREAMING
clients (SSE loop `main.rs:2810`) just see the stream end — 200 already sent, no error chunk,
no `[DONE]`, no `finish_reason` — so a well-behaved client reads a truncated success. Buffered
`Cmd`s survive the respawn (`rx` borrowed, `worker.rs:5392`), so a poison request re-panics to
`exit(70)` (`worker.rs:5469`) — documented. Fix shape: track whether the SSE loop saw a
terminal event; on `None` without one, emit the terminal error frame (see 4.8) with the
overloaded/restart-in-progress class the non-streaming path already produces. Same terminal-frame
machinery as 4.7.

### 1.7 CONFIRMED — parked-pool key space is unbounded and client-controlled (VRAM DoS)
`worker.rs:518` — `type PoolKey = (String, String);`; `worker.rs:1845–1847` keys on
`(model, cache_ns)`; `main.rs:999–1001` `cache_namespace` returns the raw client `cache_salt`
verbatim when no keyring is configured. `reuse_pool_per_model()` (`worker.rs:442`, default 2)
is a cap PER `PoolKey`, not per model, and nothing bounds the number of keys (contrast the
metering map's `METER_TENANT_CAP = 256`, `worker.rs:761`). Scenario: an unauthenticated client
loops `/v1/completions` with `cache_salt: "a1".."a200"`, each ≥16 tokens fed to prefill_done;
each salt gets its own pool with up to 2 full-cap caches, none ever cross-key evicted → VRAM
exhausted by parked state alone. Same shape on `spec_reuse` (`worker.rs:3394`). Fix shape: one
global byte-accounted LRU across all `PoolKey`s for both pools (mirror `PrefixCache.total_bytes`
+ the `lru` BTreeMap), so a new namespace's park evicts the globally oldest entry; additionally
cap distinct namespaces per tenant the way `meter_account` caps tenants.

### 1.8 CONFIRMED — graph-promoted solo sessions never park reusable KV
Promotion takes the cache (`worker.rs:2535` `let cache = s.cache.take().unwrap();`) into the
GraphSession; it returns to `s.cache` only on DEMOTION when a second session arrives
(`worker.rs:2450–2451`). A solo graph session that runs to completion retires with
`s.cache == None`, so the plain-park branch (`worker.rs:3409` `if let Some(cache) = s.cache`)
never fires — the conversation's KV is dropped and the next turn cold-primes. This hits exactly
the long-solo-greedy case (budget >= gs_min=384) where a parked continuation pays most.
MED/perf-only. Fix shape: at retire, recover `g.cache` from a live GraphSession (the demotion
handoff already does this) before the park branch.

### 1.9 CONFIRMED (doc drift, non-finding for memory) — no duplicate plain cache on the spec path
The Session doc (`worker.rs:1758–1760`, "cache above stays as the (unused) admit allocation…")
is stale. admit defers the plain cache ("legacy cache deferred: allocated below ONLY if the spec
path doesn't take the session", `worker.rs:3886`) and the spec branch keeps `cache = None`
(`worker.rs:4197–4198`; spec requires `seed_fed.is_empty()`, `worker.rs:4917`). No multi-GB
duplicate allocation exists at ctx 262144 — this was worth confirming because the comment sends
the next reader hunting a phantom 11.2 GB. The unreachable `(Some(_), c)` arm at `worker.rs:4197`
is a latent trap: if a future change lets a reuse hit coexist with spec eligibility it silently
retains a full-ctx cache. Fix shape: correct the comment; turn the unreachable arm into an
explicit reject/`debug_assert`.

(Full state x memory table in NOTES.md finding 24.)

---

## Area 2 — Prefix cache + affinity correctness surface

### 2.1 CONFIRMED (receipt + line) — frozen checkpoint: a prefix hit can never publish a newer one
`worker.rs:1707` — `if s.n_cached > 0 || s.cache.is_none() || s.fed.len() < PREFIX_CACHE_MIN_TOKENS
{ return; }` in `maybe_prefix_seed`, plus the `has_covering` dedupe at `worker.rs:1710`. This is
the root of the cachespec TTFT slope: a resumed session (`n_cached > 0`) can never seed a newer
checkpoint, and an entry that already covers the prompt blocks the insert, so the reusable
frontier freezes at the first snapshot while the conversation grows. Owned by `lane/plain-affinity`
(2.6); recorded here for completeness.

### 2.2 CONFIRMED — unpinned insert beside pinned bytes can evict itself (wasted device snapshots)
`worker.rs:1522–1579` — the entry is pushed, then `while total_bytes > budget` evicts LRU-first;
the pinned-headroom refusal (`worker.rs:1542`) guards only `initial_pins > 0`. An unpinned insert
(seed / lcp-split) while pinned entries crowd the budget evicts every other unpinned entry, then
the just-inserted entry itself. No dangling handle (plain `insert` ignores the id), but each such
insert costs a full `prefix_snapshot` device deep-copy (hundreds of MB at depth) discarded
immediately, repeated per qualifying request. Perf drain, not corruption. The LRU index stays
consistent: `remove_at` (`worker.rs:1471`) does swap_remove + moved-entry key re-point, and
`unpin` re-derives the index by id (`id_index`, `worker.rs:1399`) so a prior swap_remove cannot
make it release the wrong entry — traced, no drift. Fix shape: check fit against
`budget - pinned_bytes` BEFORE the snapshot copy on the unpinned path too; skip the copy when it
cannot survive insertion.

### 2.3 Tenancy — verified sound, one metering wrinkle
Both HTTP surfaces overwrite `cache_ns` post-build with `tenant_namespace` (`main.rs:2615`,
`main.rs:2698`); keyring mode is `t:<tenant>\x1f<salt>` (`auth.rs:416–418`) with `\x1f` excluded
from tenant ids (forged-salt test `auth.rs:786`). All three reuse structures
(`reuse`/`spec_reuse`/`px`) key on `PoolKey=(model, cache_ns)`; fanout groups compare keys before
token bytes (`worker.rs:1289` `candidates[j].key != candidates[i].key`); the isolation unit tests
(`worker.rs:5721`, `5859`) pin both directions. No cross-namespace reach found. Wrinkle (LOW): in
NO-keyring deployments a client-supplied `cache_salt` of the form `t:acme\x1f...` passes raw into
`meter_key` (`auth.rs:426–431`), which strips it to `t:acme` — metering-row spoofing in open
deployments (accounting only, no cache access). Also (LOW): in the shared `""` namespace of an
open deployment, `affinity_key` (`main.rs:1018`) lets client A nominate client B's parked spec
session by guessing its `session_id` — but `affinity_match` (`worker.rs:699`) requires exact token
reproduction through the checkpoint, so this is the documented same-namespace prefix-confirmation
oracle (PROMPTPEEK class), not content theft; a keyring closes it. Fix shape: reject/escape
`t:`-prefixed raw salts when no keyring is configured; validate + length-cap `cache_salt` at the
HTTP edge (kills the meter-row forgery since `:` is then rejected) and apply `scope_namespace`
UNCONDITIONALLY (open/single-key uses the `"default"` tenant), so `cache_ns` is always
`t:<tenant>\x1f<salt>` and `meter_key` always returns a server-derived tenant. `/metrics` is
unauthenticated (`main.rs:1522`), so the forged rows are also client-writable billing receipts —
gate `/metrics` behind the keyring when one exists.

### 2.3b CONFIRMED (new, from the prefix-cache lane) — `"default"` is not a reserved tenant id
`auth.rs:149–153` `valid_tenant` accepts `"default"` (alphanumeric), and `default_tenant()`
hardcodes `tenant: "default"` (`auth.rs:125–133`) for BOTH the open server and a `MEMRA_API_KEY`
match (`auth.rs:393–395`, `406–408`), which the doc says compose. A keyring entry named `default`
therefore shares `t:default\x1f…` — hence prefix cache, both session pools, and its meter row —
with every single-key caller. All byte tests still pass, so it is a silent scoping/attribution
MERGE of two identities the operator believes distinct, not a disclosure. MED x medium. Fix shape:
reject `"default"` (case-insensitively) in `valid_tenant` with a startup FATAL (matching the
existing bad-config path), and rename the built-in identity to a syntactically unreachable
sentinel so no keyring entry can collide.

### 2.4 CONFIRMED — cached_tokens divergence on text-prefix spec resume
Same defect as 4.5 from the cache side: `worker.rs:4222–4226` sets `n_cached = spec_resumed`
(committed-token count) and `n_prompt = spec_resumed + suffix_len` where `suffix_len` is an
independently re-encoded tail (`worker.rs:3941`). When the BPE seam re-merges, the credited
`cached_tokens` and reported `prompt_tokens` are on different bases and neither equals a
client-side count. Fix folded into 4.5 (derive both from the single `prompt` vector; `n_cached` =
LCP with the resumed committed prefix).

### 2.5 CONFIRMED (sound) — fanout credit ordering
`dedup_interactive_prefixes` (`worker.rs:4471–4491`): a sibling is credited (`s.n_cached +=
group.prefix_len`, `n_cached_in`, `meter_cached_credit`, `px.promote_miss_to_hit`) only AFTER its
`prefix_restore` succeeds (`worker.rs:4458–4470`); the leader's snapshot is taken first. The
`checked_sub().expect()` in `promote_miss_to_hit` (`worker.rs:1344–1353`) cannot fire: the
provisional miss is recorded at admission (`worker.rs:3863–3866`) before the fanout tick, and
`prefix_miss_lcp.take().expect(...)` (`worker.rs:4471`) guarantees the sibling carries it, under
single-threaded worker ownership. No divergence found. HARDENING NOTE (from the prefix-cache
lane): both guards are release `checked_sub().expect()` / `.expect()`, NOT `debug_assert` — so if
a future path (plain-affinity sets `n_cached`/`prefix_miss_lcp` on new admission routes) ever
breaks the invariant, a telemetry bookkeeping slip PANICS the worker thread and takes every
concurrent stream with it. Metering must never be able to down serving. Fix shape: `saturating_sub`
+ an `eprintln!` receipt on underflow at `worker.rs:1345–1349`; `let Some(miss_lcp) = … else {
continue }` (decline the fanout, warn) at `worker.rs:4471`.

### 2.6 Plain-affinity (`lane/plain-affinity`) integration risks — for the in-flight lane
The lane (branch `lane/plain-affinity`, PROGRESS committed off the cachespec receipt) parks plain
sessions under `(model, cache_ns, affinity)` with a pre-generation checkpoint and right-sized
caches. Hazards in the CURRENT code it must fix rather than inherit:
- **1.3 / 6.1 / 6.3 are its work**: right-sizing plain sessions is the fix for the full-cap park
  (1.3) and gives admission reclaim (6.1) something bounded to evict; the lane should build the
  global byte-budgeted LRU (1.7) for the plain pool at the same time.
- **`has_covering` will freeze the new checkpoint** (2.1): the plain-affinity publish must NOT
  route through `maybe_prefix_seed`'s cold-only guard (`worker.rs:1707`), or the checkpoint
  freezes exactly as the prefix cache does today.
- **`session_vram_cost` becomes wronger under right-sizing** (6.2): the analytic-cost fix is a
  prerequisite once plain sessions are no longer full-cap.
- **prefix_pin is one lease per session** (`worker.rs:3354`): if an affinity session both restores
  a prefix AND parks a checkpoint, confirm it does not need two leases through one `prefix_pin`.
- **double-parking (H5)**: the spec→plain demotion path (`worker.rs:2698–2740`) takes `s.spec`
  and sets `spec_k=0`, so a conversation served by spec on some turns and demoted on others parks
  into BOTH `spec_reuse` (`worker.rs:3369` branch) and `reuse` (`worker.rs:3406` branch) — once
  plain parks under an affinity key, one conversation holds entries in two pools with two caps,
  doubling its footprint and making tier selection order-dependent. Enforce one park slot per
  (conversation, model, ns) with the tier recorded as a property of the entry.
- **prefix_restore panics instead of erroring (H6)**: `prefix_restore` copies via
  `engine.copy_u8_into` (`worker.rs:1665`), whose `dst.slice_mut(off..off+len)` (`lib.rs:2238`)
  PANICS on an out-of-range range — so the `?` there is misleading; an oversized source is a
  worker panic, not a clean `Err`. Unreachable today (lookup bounds `e.toks.len() <=
  prompt.len()`, ctx guard, F5 ladder never shrinks below `need`), but plain-affinity IS a
  right-sizing change and one wrong sizing decision here crashes the worker (same class as the
  `remove(0)`-at-cap-0 crash already recorded at `worker.rs:3395–3397`). Add an explicit
  capacity precondition in `prefix_restore` returning `Err` (the shape exists for layer/kind
  mismatch), and make `copy_u8_into` bounds-check rather than relying on `slice_mut`'s panic.

(Insertion/eviction/credit surface fully traced across both my read and the prefix-cache lane;
findings reconciled with no conflicts — see NOTES.md finding 32.)

---

## Area 4 — Streaming / event contract ("where else can receipts lie?")

### 4.1 CONFIRMED — spec burst sends one Event::Token per ROUND under the last token's id
`worker.rs:4989` (round flush) and `worker.rs:5055` (post-burst tail) each send one
`Event::Token { id: last_id, text: <whole-round delta> }`. The plain paths
(`advance_sample_emit` 4710, `advance_token_emit` 4760) are genuinely 1:1. Failure = the
longdepth receipt: native blocking collects `tokens.push(id)` once per event (`main.rs:3016`)
while `n_tokens` comes from `s.generated.len()` (`worker.rs:5296`), so the surface publishes
`n_tokens: 2048` beside an 803-element `tokens` array — intermediate ids were never sent, not
merely unlabeled. Breaks logprob alignment, replay, per-token billing. Fix shape: emit one
Event::Token per committed id in the burst commit loop (`worker.rs:5024–5029`, which already
walks each id and advances a byte cursor) rather than accumulating; if round cadence regresses,
widen the event to carry `ids: Vec<u32>` alongside coalesced text. Assert
`tokens_emitted == generated.len()` in `finish()`.

### 4.2 CONFIRMED — max_tokens overshoot: engine session contract leaks through HTTP
Engine: `spec.rs:5225–5237` — session mode pushes all accepted columns + bonus into `out`
past `max_new` (guard is `!session_mode && out.len() >= max_new`, and
`bonus_emitted = session_mode || out.len() < max_new`); session mode returns before the tail
`out.truncate(max_new)` (`spec.rs:5634–5637`). Worker: the burst commit loop pushes every token
into `s.generated` and streams it (`worker.rs:5024–5029`) before the budget check
(`worker.rs:5066`). Client receives up to K+1 tokens over `max_tokens` (the cachespec
769–770/768 receipt), `usage.completion_tokens` bills them, and `finish_reason` is still
`length`. The engine overshoot is load-bearing (committed must equal cache rows,
`spec.rs:5601–5612`). Fix shape: separate committed (KV truth) from emitted/billed — clamp
emission to `s.budget` BEFORE the flush, so surplus tokens stay in `committed`/`pending_tok` for
a legitimate continuation resume and are never detokenized into a delta; the crossing round
emits only its in-budget prefix then stops MaxNew. Assert `n_tokens <= requested_max`.

### 4.3 CONFIRMED — tokens_out / lane_tokens / step_stats miss the spec and round-robin paths
Increment sites are exactly three, all non-spec: `worker.rs:2604` (graph), `worker.rs:3087`
(eager-only), `worker.rs:3204` (batched). The spec arm's success case is empty
(`worker.rs:2844` `Ok(true) => {}`) and the legacy round-robin arm (`worker.rs:2426–2436`)
increments nothing. Consequences beyond the receipt's `tokens_out=0`: the rate-limit reset
estimate (`main.rs:508–511`) collapses; per-lane throughput mis-attributes; `step_stats` is
EMPTY on a spec-only box, so the dark-lane QoS gate (`worker.rs:2253`) runs on a default not a
measurement; `last_interactive_decode` never refreshes so the starvation sentinel fires
permanently. Fix shape: count from `active[i].generated.len()` delta across `step_session` in
the `Ok(true)` arm (correct whether the round committed 1 or K+1), stamping `n_tokens_out`,
`lane_tokens`, `step_stats`, `last_interactive_decode`; count EMITTED not committed so it agrees
with 4.2's clamp. Add the one-request-per-arm counter test.

### 4.4 CONFIRMED — the graph path counts a token (and a latency sample) for a FAILED step
`worker.rs:2604–2606` — `n_tokens_out += 1; lane_tokens[0] += 1; step_stats.record(...)` sit
outside the `match g.step(...)` arms, so both Ok and Err fall through them. A graph step that
errors (CUDA fault, capture mismatch, the degrade path `worker.rs:2455–2462`) records a token
that never reached the client and poisons the p99 the dark-lane gate reads. MED x low-medium.
Fix shape: move the three statements into the `Ok` arm; record failures on a distinct counter.

### 4.5 CONFIRMED — n_prompt on a text-prefix spec resume is built from a re-tokenized suffix
`worker.rs:4222–4226` — `n_prompt = spec_resumed + suffix_len`, where `spec_resumed` counts
committed tokens (the rewind position) and `suffix_len` is the INDEPENDENTLY re-encoded tail
(`worker.rs:3941` `text_suffix = Some(lm.tok.encode(rem, false))`). Tokenization is not
compositional (`encode(a)+encode(b) != encode(a++b)` at the BPE seam), so
`spec_resumed + suffix_len != prompt.len()` in general — contradicting the Done doc's "ONE
source of truth" (`worker.rs:104`). `usage.prompt_tokens` drifts ±1–3 from a client-side count,
only on resumed conversations, so it reads as nondeterministic. MED x high (every text-prefix
resume = the common multi-turn case). Fix shape: `n_prompt = prompt.len()` unconditionally;
`n_cached` = LCP of `prompt` with the resumed committed prefix (also makes the cache credit
conservative-correct when the seam token re-merged).

### 4.6 CONFIRMED — an aborted stream has no terminal reason, no usage, but still counts as completed
Disconnect sweep (`worker.rs:2418–2423`) → `abort_log` (stderr only, `worker.rs:5257`) → the
retire sweep runs ordinary completion accounting: `n_completed += 1` (`worker.rs:3358`),
`lane_completed[..] += 1` (`worker.rs:3368`). `StopReason` (`decode.rs:143–148`) has no
abort/disconnect variant. So delivered work (the tokens streamed before cancel) is the only
durable record a stderr line, `completed` is inflated, and every rate derived against it
(`tokens_out/completed`, mean output length, per-lane success) is computed against a
numerator-less denominator. Cancellation is the normal interactive case. Detection is
tick-top-only, so a cancelled long prefill burns to the next sweep. HIGH x certain (for
metering honesty). Fix shape: add an abort terminal state; increment `n_aborted` not
`n_completed`, add the emitted tokens to `tokens_out`, record partial usage where the metering
sink sees it; give `stop_reason_to_finish` an explicit abort mapping.

### 4.7 CONFIRMED — mid-stream Event::Error emits a bare error object, no terminal finish_reason
`main.rs:2896–2918` — the OpenAI branch writes `data: {"error":{...}}` then `data: [DONE]`; the
tokens already sent were well-formed chunks, but the error frame is not a chunk (no
`choices`/`finish_reason`), so a strict chunk parser either throws or (commonly) skips it and
treats `[DONE]` as normal termination — a server fault reads as a short natural answer, and
`usage` is only emitted on the success Done path (`main.rs:2839–2895`). A tool-calling client
gets half a JSON args block and no signal the server faulted. HIGH x medium. Fix shape: precede
the error frame with a real terminal chunk carrying a non-`stop` `finish_reason` + usage-so-far,
then the error object, then `[DONE]`; align `type`/`code` with `engine_error_body`
(`main.rs:1129`).

### 4.8 CONFIRMED — streaming worker death = silent truncated 200 (non-streaming is honest)
Same mechanism as 1.6 from the HTTP side. Non-streaming falls through to
`overloaded("worker closed the stream without completing")` → 503 + Retry-After
(`main.rs:3073–3080`). Streaming just ends the stream at 200. The two surfaces disagree about
whether the request failed, so a client that switched to streaming for UX loses the retry it
had. HIGH x low-medium. Fix shape: the same terminal-frame emitter as 4.7, keyed on "rx closed
without a terminal event", using the overloaded class — one fix covers 4.7 and 4.8.

### 4.9 SUSPECTED — byte-indexed stop/tool holdback + GemmaLabel tail-drop
`main.rs:2960–3000` `StopScrubber::push` slices at a BYTE index (`buf.len() - keep`) that may
land mid-codepoint; the ASCII-only invariant that keeps it safe is enforced by comment only, and
a user-supplied non-ASCII `stop` string (e.g. `"→"`) reaches `StopScrubber::new` from request
JSON — a boundary landing mid-sequence panics on slice. Separately, `ToolStreamParser::finish()`
has `State::GemmaLabel => {}` which DISCARDS the buffered tail (generated content silently lost
if a stream ends inside a partial `<|channel` label). Panic is SUSPECTED (not executed);
tail-drop is CONFIRMED as code shape. Fix shape: validate stop strings are ASCII at parse (400
otherwise) or advance the emit index to the next `is_char_boundary`; flush the GemmaLabel buffer
as content at finish.

### 4.10 CONFIRMED — native completion surface is not stop-scrubbed
`main.rs:2752–2753` — the scrubber is built only for `chat || openai_compat()`. A native
request with `stop` gets the engine to halt but no HTTP-side trim, so the stop string itself is
emitted; the same request on the chat surface has it trimmed. The native surface is where both
seed receipts were taken, so a stop-inclusive tail silently inflates the `n_tokens` those
receipts compare. LOW-MED x medium. Fix shape: apply the scrubber whenever `stop_strings` is
non-empty; if native intentionally wants raw text, document it at the field.

### 4.11 SUSPECTED — spec burst adopts its detok cursor only on the success path
`worker.rs:5044` — `s.emitted_bytes = cursor;` runs after the commit loop; every `?` on the
engine call earlier returns before it, while `flush_cb` (`worker.rs:4989`) may already have SENT
deltas from the advanced local cursor. If a recoverable path resumes a session after a mid-burst
engine error, the next emission recomputes its delta from the stale `emitted_bytes` and re-sends
bytes the client already has (duplicated content — worse than truncation). Reachability depends
on whether any path continues a session after a mid-burst engine error; the step-OOM park is
pre-first-token (5.1), which is what keeps this SUSPECTED. Fix shape: advance `s.emitted_bytes`
inside `flush_cb` as each delta is sent, so the session's notion of "bytes the client has" can
never lag what went out.

### 4.12 CONFIRMED (dead-on-serve trap) — round-stream ring drain drops committed tokens
`spec.rs:4364–4369` — the ring-drain loop discards ring tokens once `out` is full (mirror of
4.2's overshoot); gated behind `MEMRA_SPEC_STREAM=1 && !session_mode` (`spec.rs:4105–4112`),
mutually exclusive with the session mode serve uses. No serving impact today, but a latent trap:
if promoted, it desyncs `out` from committed KV. Fix shape (flags doctrine): kill the flag +
dispatch arm if the experiment concluded, else apply 4.2's commit-vs-emit split.

### 4.13 CONFIRMED — stop_reason_to_finish collapses unknown reasons to "stop"
`main.rs:1180–1186` maps via STRING match with `_ => "stop"`, so an engine-side rename or a new
variant silently reads as natural completion — the wrong fail-safe direction, and it compounds
4.6 (no abort reason exists). Also conflates `MaxNew` and `ContextFull` into `length` (OpenAI-
compliant, but a native client cannot tell "ask for more" from "window exhausted"). MED x low.
Fix shape: match the typed `StopReason` enum (compiler flags a new variant); if a catch-all
remains, map it to a non-natural terminal value.

---

## Area 5 — Error-handling honesty

### 5.1 CONFIRMED (good, non-finding) — step-OOM park is correctly gated pre-first-token
`worker.rs:2826–2837` — the guard includes `active[i].generated.is_empty()`, so a session that
has streamed anything cannot park; `park_requeue` reuses `tx: s.tx.clone()` and re-primes with
no client-visible event. The honest shape. Recorded so the guard is not "simplified" away — it
is exactly what makes the duplicate-token restart impossible (and keeps 4.11 SUSPECTED).

### 5.2 CONFIRMED — parked re-admit double-counts input metering
Same as 1.5 (listed under lifecycle; the fix flag serves both).

### 5.3 CONFIRMED — spec-session alloc failure silently downgrades to tokenwise; only stderr says so
`worker.rs:4168–4170` "spec session alloc failed (...); tokenwise path" (ladder swallows its own
errors at `worker.rs:4150` `Err(_) => None`). Under VRAM pressure every spec alloc can fail and
every request quietly serves the slower path — correct output, halved throughput, no metric.
Compounded by 4.3 (tokenwise counts tokens, spec does not), a box that falls out of spec mode
looks like it STARTED working on the dashboard. MED x medium. Fix shape: a `spec_alloc_failures`
counter + a gauge of tokenwise-after-spec-denied, published beside the per-model `spec`
telemetry; same for the graph-degrade path (`worker.rs:2455–2462`).

### 5.4 CONFIRMED — sibling prefix_restore failure kills an otherwise-servable request
`worker.rs:4458–4470` — in the fanout, the LEADER's snapshot failure is log-only (everyone
serves cold = correct), but a SIBLING's restore failure sends `Event::Error` and kills that
request, which would have served fine on the cold path. The asymmetry is the tell — the cold
fallback is already implemented (`worker.rs:3826–3856` does exactly it for the admit-time
restore). MED-HIGH x low. Fix shape: on sibling restore failure, log, drop it from the fanout
group, let it serve cold.

### 5.5 CONFIRMED — a step error at budget is laundered into a clean MaxNew (highest-value honesty bug)
`worker.rs:2585–2595` — when a graph step errors and the session is at budget (`at_budget`,
`g.cache.pos + 1 >= g.bucket_max`), the error is discarded and the session finishes
`StopReason::MaxNew`. A CUDA fault on the last step becomes `finish_reason: "length"` + 200, seen
by no error-rate view; at-budget is the most common terminal state, so any last-step fault
distribution hits it. HIGH x low-medium. Fix shape: if the final step errored, the error is the
truth even at budget — terminate with the error class (4.7 frame) or deliver the already-emitted
output with a non-`stop`/non-`length` terminal signal and record the fault.

### 5.6 CONFIRMED — requests during the respawn window silently queue instead of getting 503
`main.rs:2627` / `main.rs:2717` — the only pre-flight gate is `draining()`; availability is
inferred from `cmd_tx.send(...).is_err()`. But the supervisor owns `cmd_rx` across restarts
(`worker.rs:5392`), so `send` SUCCEEDS while no worker is alive — the channel is unattended, not
closed. `/readyz` correctly 503s while faulted/loading (`health.rs`) but the inference handlers
never consult it. Every request in the ~2 s backoff + multi-second reload window is accepted onto
a 200-track connection and waits — no 503, no Retry-After — so a load balancer keeps routing to
the instance and the retry-storm protection is bypassed on the one outage it exists for. HIGH x
medium. Fix shape: gate both handlers on the readiness signal `/readyz` already computes,
immediately after `draining()`, returning the overloaded/503 + Retry-After contract
(`drain_response` at `main.rs:550` is the template); pick Retry-After per health state.

### 5.7 CONFIRMED — debug_assert on prefix unpin is a release no-op → silent pin leak
### 5.7 CORRECTED to CRITICAL — the ONLY production `unpin` is compiled out in every release build
`worker.rs:3354–3356` — `debug_assert!(px.unpin(&pin), "retired session held a missing prefix
pin");`. **I initially rated this MED believing the `unpin` call runs as the condition and only the
return is discarded — that is WRONG, and I verified it empirically.** `debug_assert!` compiles out
its ENTIRE argument (side effect included) when `debug-assertions` is off; the workspace has no
`[profile.release]` override (`grep` over all Cargo.toml → none), so `cargo build --release` — what
every gate and serve script uses — has debug-assertions OFF. Probe: `rustc -O` on a
`debug_assert!(side_effect())` ran the side effect 0 times (debug build: 1 time). So in every
shipped build `px.unpin` NEVER RUNS. It is the sole release release-path unpin (grep: 5 sites, 4
under `#[cfg(test)]`).

Consequence: every prefix-cache HIT (`worker.rs:3840`) and every fanout participant
(`worker.rs:4501`) leaves `pins >= 1` forever. `pin_n` already removed the entry from the `lru`
index (`worker.rs:1433–1435`), and `remove_at` refuses a pinned entry (`worker.rs:1473–1474`),
which both eviction loops treat as STOP not skip (`worker.rs:1568–1586`
`else { break }`). So a served entry becomes permanently resident AND permanently un-evictable:
(a) `total_bytes` ratchets past `MEMRA_PREFIX_CACHE_MB` and never comes back (the budget stops
bounding); (b) once leaked pins reach the budget, every pinned insert is refused by the
pinned-headroom guard (`worker.rs:1542`) — a SECOND independent cause of the RESULTS.md
frozen-checkpoint symptom; (c) the "sessions win over the cache" headroom escape
(`worker.rs:3854`, `4204` `px.evict_all()`) frees ZERO bytes and the client gets a hard
`cache alloc failed` where a retry was promised. CONFIRMED from code + compiler semantics
(empirically checked). The cachespec receipt does not exhibit it only because that run set
`MEMRA_PREFIX_CACHE_MB=2048` and never crossed budget. Fix shape: call `px.unpin(&pin)` as a plain
statement and log a warning on `false` (never a panic — a missing pin is a real bug worth a
receipt); harden both eviction loops to SKIP a pinned/stale victim and continue walking `lru`
rather than breaking; add a release-mode test asserting `pins == 0` after a retire; and a
`tools/local-ci.sh` grep guard forbidding `debug_assert!` around any required side effect (this
finding is the standing example). Credit: independently found by the prefix-cache lane; this
corrects my own 5.7.

### 5.8 CONFIRMED — non-streaming Error discards collected text unaccounted
`main.rs:3069` — `blocking_response` returns `engine_error_response(&err)` on `Event::Error`,
dropping accumulated text. Dropping is the correct CLIENT contract (a partial 200 body would be
4.7's failure), but the generated tokens are also unaccounted — no path adds them to
`tokens_out` or a partial-work counter. LOW-MED x medium. Fix shape: before returning, add the
generated count to the same "produced but not delivered" counter 4.6 introduces — one counter
covers aborts, mid-stream errors, worker death, and this path; its ratio to `tokens_out` is a
useful operational signal.

### 5.9 CONFIRMED (clean inventory) — 503 producers all carry Retry-After
`drain_response` (`main.rs:550`, coded/clamped/ms-twin), `class_http(Overloaded)` →
`engine_error_response` (`main.rs:1096–1178`, RETRY_AFTER_S_OVERLOADED=5, with `is_cuda_oom`
promoting driver OOM to Overloaded — correct), `worker_unavailable_response` (Retry-After 2),
the non-streaming channel-closed fallthrough (`main.rs:3085`, inherits the contract). No bare
503 found. Two observations: Engine 500s carry no `x-should-retry: false` (client can't
distinguish "retry pointless" from "forgot the header"); `worker_unavailable_response`'s 2 s is
shorter than the reload it covers, so a client retrying at 2 s hits 5.6's silent queue. Fix
shape: mark non-retryable 5xx with `x-should-retry: false`; align the unavailable window with the
real respawn+reload cost once 5.6's readiness gate makes it observable.

### 5.10 CONFIRMED (swallow) / SUSPECTED (direction) — mem_get_info failure yields no VRAM signal
`worker.rs:2371` / `worker.rs:2309` — `.ok()` converts a driver query failure into `None`;
admission then proceeds without a free reading. A failed query is itself a strong sick-context
signal and is discarded at the call site. Fail direction (open=admit / closed=defer) is
SUSPECTED pending the consumer read. Fix shape: log-and-count the failure; make the None policy
explicit and conservative (defer + overloaded class), since a failed query correlates with a
sick context.

---

## Area 6 — Admission gate at 256k

### 6.1 CONFIRMED — the gate defers without reclaiming any parked entry (the receipt's serialization term)
`worker.rs:2343` (`if free < cost.saturating_add(reserve)`) → `worker.rs:2365`
(`requeue.push_back(req)`). Between them the gate has `&mut reuse/spec_reuse/px` and its own
diagnostic even COUNTS the parked entries (`worker.rs:2353` parked spec, `worker.rs:2360` plain
reuse) — then queues anyway. Reclamation exists everywhere else (`px.evict_all()` 3854/4204,
spec `p.clear()` 4092/4103) but only on the alloc-failure path inside `admit`, which the
deferred request never reaches. This is the 4,659-defer / 25 s c=4 serialization. Fix shape:
before deferring, on the first shortfall of a tick evict LRU across the tiers (cheapest first,
byte-accounted, stopping at the shortfall, preferring to keep entries whose affinity matches a
queued request), re-read `mem_get_info + pool_cached_bytes`, and only defer if still short;
bound to one reclaim pass per tick; log evicted bytes on the existing defer line.

### 6.2 CONFIRMED — session_vram_cost is one first-admit scalar; ctx_cap is per-request
`worker.rs:2384–2390` captures cost once per model from the first admit's free delta;
`worker.rs:3689–3702` shows ctx_cap is request-dependent (explicit `max_ctx`, or the bounded
arm's `prompt.len()+ctx_floor`, or `prompt.len()+max_new+8`), varying ~100x. Scenario: a first
request with `max_tokens: 64` calibrates ~119 MB (the in-tree 9B@8192 figure); later 262144
sessions cost ~11.2 GB and the gate waves through ~94 per imagined GB. On the plain path
`reserve = cost` too (`worker.rs:2318`), so the whole bar is ~238 MB against 22.4 GB of demand —
guaranteed step-OOM (→ 1.4). The receipted 1.49x understatement is the small end. Fix shape:
compute cost analytically from `cfg` (layers/heads/head_dim/kv dtype) x ctx_cap — the same
`Cache::new` allocation (`memra-kv/src/lib.rs:175–217`) — keep the measured delta as a per-model
calibration factor (the transient/arena term), and scale by the request's own ctx_cap on every
admit (high-water or EWMA), not once.

### 6.3 CONFIRMED — cost==0 on a reuse-pool first admit disables the gate for the process life
`worker.rs:2387` `if cost > 0` skips insertion; `worker.rs:2309` requires
`session_vram_cost.get(&req.model)` to be `Some`, so if the first admit consumed a parked cache
(delta ≈ 0) or `mem_get_info` noise floored to 0, the gate never engages for that model again.
Likelihood LOW not MED (per NOTES.md finding 29: `reuse`/`spec_reuse` are run() locals wiped on
restart, so the first admit of a process always faces empty pools and allocates; only
mem_get_info noise reaches 0 on a fresh cold admit). Fix shape: record cost at model load
(analytic, per 6.2) so the gate is armed before the first request; treat a zero delta as "no new
information", not "no cost".

### 6.4 CONFIRMED — the `if !active.is_empty()` skip + unvalidated client max_ctx guarantees a step-OOM path
`worker.rs:2308` (`if !active.is_empty()`) skips the gate for the first admission by
construction; `main.rs:697/805/2296/2471` plumb `max_ctx` from the request body with NO
validation anywhere (grep-confirmed), and the only guard (`worker.rs:3703`) rejects prompts too
LARGE for the cap, never a cap too large for the card (the model-trained-context cap at
`worker.rs:3698` applies only to the bounded arm, not the explicit-`max_ctx` arm). A single
request with `max_ctx: 2_000_000` onto a nearly-full card goes straight to `new_cache`, fails,
does one `px.evict_all()` retry, and errors — or succeeds marginally and step-OOMs on first
decode, never consulting the parked `reuse` entries. HIGH x medium. Fix shape: validate `max_ctx`
at the HTTP boundary against `cfg.context_length` (the same value the bounded arm uses) with the
existing `context_length_exceeded` 400; and make the gate unconditional — for the empty-`active`
case compare against a floor derived from the request's own ctx_cap.

### 6.5 CONFIRMED (mechanism) / low today — pool_cached_bytes can overstate headroom
`worker.rs:2342` adds `engine.pool_cached_bytes()` (`lib.rs:1027–1030` = reserved − used) to
`free`. Two overstatement edges: (a) fragmentation — `reserved − used` is a sum, not the largest
contiguous block, so an 11.2 GB plane can fail after the gate passes (→ 1.4); (b) wrong device
under PP — the query hits the PRIMARY ordinal's default pool (`lib.rs:1042`) while `pp::new_cache`
allocates per-stage on OTHER devices (`pp.rs:1004+`). Fails safe when the pool cannot be queried
(returns (0,0), term vanishes). The term is genuinely load-bearing (the admit-oom receipt is
solid); this is about its edges, not removal. Fix shape: bound the credit to
`min(reserved−used, largest_known_free_block)` if exposed, else a conservative fraction; under an
open ppN door query the device the KV will land on, or drop the term on that path; belt-and-braces
mark the model's pool credit untrusted for a cooldown after a gate-passed step-OOM.

---

## Area 3 — Depth-dependent arithmetic (corruption-hunt)

### 3.1 CONFIRMED ROOT CAUSE — `u01()` returns exactly `1.0f`, making Gumbel noise `+inf`
`cu/spec_sample.cu:36–39`:
```
// u32 -> (0,1] uniform (never 0 -> log is finite).
static __device__ __forceinline__ float u01(uint32_t v) {
    return ((float) v + 1.0f) * (1.0f / 4294967296.0f);
}
```
The comment guards the wrong end. `u == 0` was prevented (would give `g = -inf`, harmless — the
token always loses). `u` rounding to `1.0f` is reachable and gives
`g = -logf(-logf(1.0)) = -logf(-0.0) = -logf(0) = +inf` — that token ALWAYS wins. Verified the
float arithmetic directly: float spacing near 2^32 is 256, so 128 of 2^32 u32 values round to
`4294967296.0f`; `+1.0f` is absorbed and `× 2^-32` is exact → `1.0f`. Per-lane P = 128/2^32 =
2.98e-8. At step35's real vocab 128896 that is P = 3.83e-3 per sampled token → E[first hit] ≈
261 tokens, P(corrupt by token 281) ≈ 0.66, P by 1000 ≈ 0.98. The winner is whichever Philox
lane hit the bucket — a uniformly random vocab id = cross-lingual soup, which then poisons the
context and compounds. `+inf` also beats the `-3.4e38f` ban sentinel, so grammar-banned ids can
win too. Unguarded consumers: `spec_sample.cu:54` (`gumbel_perturb_f32`), `:382` (`_ctr`), `:400`
(`_filtered`).

Consistency with all four longdepth constraints:
- **greedy-clean**: `spec_sample.cu:50` `if (temp <= 0.0f) { y[i] = x[i]; return; }` returns
  before Philox; on serve greedy takes pure `argmax_token_device_col` (`decode_batch.rs:334`) and
  the perturb kernel never launches. The corrupting path is temperature-only by construction.
- **temp-0.7-dirty**: any `temperature > 0` rolls the dice per token; `main.rs:856`
  `default_temperature() -> 1.0`, so temperature-omitting clients are on the corrupting path by
  default.
- **onset ~281**: E[first] = 261. Direct match.
- **6–14k prompts**: prompt length is not causal, generated-token count is (long prompts
  correlate with long generations) — the one constraint it explains only indirectly.

Served emission paths reachable at temp 0.7: `decode_batch.rs:1543–1545` (default batched serve
tick — step35 runs plain decode here, the live path), `decode_batch.rs:337` (B=1 twin),
`spec.rs:5060–5062` (spec bonus). Draft-side (`spec.rs:4656`) is shielded (garbage draft gets
tiny p_j, rejected). The HOST sampler is NOT affected: `memra-sampling/src/lib.rs:287–289` uses
`>> 40 / 2^24` → strictly `[0,1)`. Why gates missed it: `bin/sample_check.rs` tests determinism
and temp-0 continuity, never samples enough (token,event) pairs to hit a 3e-8 tail, and never
asserts finiteness.

Fix shape: make `u01` half-open on the TOP end (e.g. `((float)(v >> 8) + 0.5f) *
(1.0f/16777216.0f)`, matching the host sampler's 2^24 form) so `g` is always finite; add a
finiteness assertion after every `gumbel_perturb*` launch as a permanent canary. The fix changes
the token stream, so seeded-reproducibility gates (`run_spec.rs:311–336`) need rebaselining.
Fastest discriminating experiment: apply the half-open fix and re-run temp 0.7 for ≥1000 tokens;
pre-fix, a finiteness assertion should trip inside ~261 tokens.

### 3.2 CONFIRMED (host/device divergence) — device top_k/top_p combined as UNION not intersection
`cu/spec_sample.cu:250–253` — `keep_more` from EITHER predicate lowers the threshold, whereas the
host sampler (`memra-sampling/src/lib.rs:154–175`) applies them sequentially (intersection). Host
and device disagree on identical config; explains why a `top_k=40` guard would fail to contain
3.1. MED / CONFIRMED divergence (not depth-dependent alone). Fix shape: apply the two filters as
an intersection on device, matching the host order.

### 3.3 SUSPECTED — filter_stats 24-iteration bisection floor no-ops top_p on peaked rows
`cu/spec_sample.cu:237` — the bisection resolution is 2^-24 in `e = p/p_max` units; on very
peaked rows the top_p boundary token sits below the floor, `lo` never leaves 0, and the
truncation silently does nothing. Peakedness varies with depth. Fix shape: raise the iteration
count or switch to an exact partial sort for the tail.

### 3.4 SUSPECTED — RoPE `theta = (float)pos * powf(theta_scale, j)` degrades with position
`cu/kernels.cu:1431–1432` (`rope_neox2_f32`, the step35 arm; twins `:807`, `:1459`,
`flash_attn.cu:566`) — no precomputed table, per-thread `powf`, f32 phase resolution degrades
∝ pos at 262144. Same formulation as llama.cpp (not a divergence) but genuinely depth-dependent;
predicts gradual drift, not abrupt soup — so it is NOT the longdepth culprit but is a real
long-context precision candidate. MED / SUSPECTED.

### 3.5 LOW (demoted for step35) — f16 O-accumulator in the flash arm
`flash_attn.cu:984/988/1802/1942–1957` (doors `lib.rs:362/379`) — the repo already measured this
class degrading depth acceptance 0.883→0.405 invisibly to argmax gates, BUT step35's hd128 SWA
prefill reaches `fa_prefill_view_ws_w_hd128` → `fa_prefill_qw_db_w_hd128`, whose O_acc is f32
`CTile` (`flash_attn.cu:4479`, `:88`) — the f16 arm is not on step35's path. Live for hd512
globals on other SKUs. LOW for this arch.

### Hypotheses affirmatively KILLED (structural, so no lane should chase them)
- **No ring buffer anywhere.** SWA is a read-side mask + host base-pointer view over a full
  `max_ctx` allocation (`memra-kv/src/lib.rs:316–317`), keys carry absolute rope. The
  "rejected-draft clobbers slot pos%512" class is structurally impossible; rollback is pure `len`
  truncation (`spec.rs:658` `MtpScratch::set_len`, `memra-kv:407–433` `Cache::rollback`).
- **No 32-bit overflow at 262144.** All token×stride byte products are `(size_t)`-promoted (417
  sites); residual `int` products are bounded by head/split count, not context.
- **Window mask off-by-none.** All 9 sites are
  `if (window > 0 && (k0+col) < q_pos - (window-1)) s = NEG_INF;` = exactly `window` keys
  inclusive, matching the CPU oracle (`kernel_check.rs:3547`) and llama's standard SWA. The
  prefill/decode `win-1` vs `win` asymmetry is the 32-align trim (`raw & !31`,
  `hybrid_forward.rs:8686`) keeping ≤31 already-masked keys — correct.
- **Graph/dc decode refuses step35** (`decode.rs:1918/2191/2475/2844`); `graph_update.rs` /
  `prime_graph.rs` have zero step35/window refs. **pp.rs has no u16/u32 position packing.** RNG
  counters (sctr/uctr u32 monotonic per-session; `ctr = generated.len()`) do not wrap at
  realistic depth.

### Lower-priority latent defects worth logging (not the longdepth culprit)
- `penalize_logits_f32` (`spec_sample.cu:351–366`) bound-checks target-space `hist` ids against
  `n = d_vocab` for the trimmed draft head, penalizing wrong q rows and breaking p/q symmetry
  (needs penalties + FR-Spec trim).
- `softmax_gather_f32` (`:67`, `:78`) aliases float max and int argmax in one 32-slot `__shared__`
  — safe only while `blockDim <= 512` (launched 256).
- windowed-decode `start = T_kv - window` unclamped in-kernel (`flash_attn.cu:5823` + 4 twins),
  guarded only by host convention.

---

## What this paper does NOT change

Read-only lane. No defaults changed, no merge, no tag, no push. The findings seed the next
hardening lanes; the fix shapes are directions, not patches. Precision note: every CONFIRMED
finding's primary line was re-read against `74afcaf6` directly; SUSPECTED findings are labeled
because they need a repro to promote (4.9 panic, 4.11 mid-burst-resume duplication, 5.10 fail
direction). The plain-affinity lane (`lane/plain-affinity`, in flight from the same cachespec
receipt) owns 2.1 and should absorb the 1.3/6.1/6.3 reclaim + right-size work.
