# WP-B day 26: memra#539 census from the code, the allocated-over-used cell on both cards, the KV residency design note

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, merged with `origin/main` `8e94a480b` (#619) at
`1c66ff10e` (INDEX.md conflict resolved by hand, day-25 row kept, `tools/check-conflict-markers.sh` OK). Every push today
went out in the announced development mode (`MEMRA_RELEASE_QUALIFICATION_MODE=development`, the hook printed
`UNQUALIFIED DEVELOPMENT ... no GPU qualification claimed` and logged it); no qualification is claimed anywhere in this
record. **No engine change today**: census, cell and design only. Every cell is `executed-not-qualified`; every verdict
line is verbatim; no timing is compared across the two cards.

Rigs. Local RTX 5090 Laptop GPU (24,463 MiB, `power.limit [N/A]`, lock `/tmp/memra-5090.lock` by `flock`, CPU work under
`systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`), Qwen3.5-9B-NVFP4-MTP (`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`,
5,657,607,424 B). Target card: one RTX PRO 6000 Blackwell (97,887 MiB, 600 W cap, collector
`tools/tier-battery.py --rig pro-single`, lock `/tmp/memra-gpu.lock`), Qwen3.8-27B-NVFP4-Q5K-mtp (15,705,922,304 B). The
Qwen3.8 the task names is this 27B mint; it is the only Qwen3.8 artifact on the target card
(`/root/artifacts` also holds a Qwen3.6-35B-A3B, outside today's scope). Both binaries were built from `1c66ff10e`
(`rtx5090-day26/build-local.log` `exit=0`; `pro-single-day26/build.log` `exit=0`, `source.txt`).

## 1. memra#539 census, from the code (tree `1c66ff10e`)

### 1.1 The cap: how many rows a session gets

`request_ctx_cap(server_ctx, model_ctx, prompt_len, max_ctx, max_new, open_output_tokens)`
(`crates/memra-server/src/worker.rs:21705-21745`), called once per request from `prepare_request`
(`worker.rs:21863-21869`) after tokenization:

| Branch (`worker.rs`) | Condition | `ctx_cap` (rows) | Verified by |
| --- | --- | --- | --- |
| `(Some(c), _)` :21715 | the request carries `max_ctx` | `c`, then `min(model_ctx)` | test `request_ctx_cap(262_144, 262_144, 128, Some(131_072), 64, None)` :31652 |
| `(None, MAX_NEW_CTX_BOUNDED)`, door OFF :21730-21736 | `max_tokens` omitted (`lib.rs:7752`, `:8246`: `max_new: req.max_tokens.unwrap_or(MAX_NEW_CTX_BOUNDED)`, `MAX_NEW_CTX_BOUNDED = usize::MAX` :50) | `server_ctx`; `prompt + server_ctx` when `prompt + 16 > server_ctx`; then `min(model_ctx)` | this cell: every open request booked `ctx=262208` on the target card and `65600` locally (section 2) |
| `(None, MAX_NEW_CTX_BOUNDED)`, door ON :21717-21723 | as above with `MEMRA_ADMIT_BY_MEMORY=1` (`admit_memory.rs:84`, default OFF, decide-by 2026-09-23) | `charged_ctx_tokens(prompt, None, 8192, model_ctx)` = `prompt + 8192 + 8` (`admit_memory.rs:52`, `:139-147`) | tests `:34602-34636` (memra#365 gap 1) |
| `(None, max_new)` :21739 | the client bounds `max_tokens` | `prompt + max_new + 8` | test `assert_eq!(request_ctx_cap(8_192, 262_144, 128, None, 64, None), 200)` :31684 |

`server_ctx` is `resolve_env_ctx(model_ctx)` (`:21862`): `MEMRA_CTX` when set, else the checkpoint's own
`context_length` (262,144 for both Qwen artifacts here; `docs/FLAGS.md:313`). Before the worker, `lib.rs:8646-8656`
resolves an omitted `max_tokens` to the registry's `default_output_length` (or `max_output_length`) when the model
registry supplies one, so a registry deployment never reaches the open arm; a naked `MEMRA_MODELS=` boot (this cell)
does. The API contract text is `docs/SERVING.md:912-914` (omitted means context-bounded, "the OpenAI
default-when-omitted semantics") and `docs/SERVING.md:373-386` (send `max_tokens`; the fallback strands memory).

What admission books is not `ctx_cap` but `admission_cap = ctx_cap.max(need)` (`worker.rs:2773-2779`) with
`budget = max_new.min(ctx_cap - prompt)` and `need = prompt + budget + SPEC_SHRINK_SLACK (64)` (`:21876-21879`). On the
bounded arm `need = P + max_tokens + 64 > ctx_cap = P + max_tokens + 8`; on the open arm `budget = ctx_cap - P` so
`need = ctx_cap + 64`. Both arms therefore book 56 rows more than they allocate (the cell's lines read `ctx=262208`
for a 262,144-row cache and `ctx=1875` for a `1715 + 96 + 8 = 1819`-row cache).

### 1.2 The allocation: what `ctx_cap` buys, per layer class

`memra_kv::Cache::new_inner(pick, cfg, plan, max_ctx, allocator)` (`crates/memra-kv/src/lib.rs:2817`) walks the plan's
layers once and allocates eagerly, at construction, for every layer:

| Layer class (`StatePlan`) | Allocation (`lib.rs`) | Rows | Bytes |
| --- | --- | --- | --- |
| `KvCache` / `SlidingKvCache` (full attention) :2857-2921 | K plane `alloc_kv_plane(alloc_rows * k_tok_bytes + 8)`, V plane `alloc_kv_plane(alloc_rows * v_tok_bytes + 8)` (:2892-2905, `kv_plane_allocation_bytes` :2291) | `alloc_rows = max_ctx`, or the SWA ring rows `min(max_ctx, window + 4096 + 512 + 31)` when the ring is on (`swa_ring_rows` :147; `KvRing` only for a planned sliding window) | `k_tok_bytes = (kv_dim_k / 32) x 34`, `v_tok_bytes = (kv_dim_v / 32) x 24` (`kv_blk_bytes` :21: q8_0 K blocks, q5_1 V blocks; `full_attention_kv_layout` :2237-2290) |
| `Recurrent` (GDN/KDA) :2922-2934 | `conv_state`, `ssm_state`, `ssm_state_alt` as `zeros(...)` | fixed per session | not context-scaled |
| `LatentKvCache` (MLA) :2935-2985 | `rows: e.zeros(max_ctx * width)` f32 (:2949) plus the indexer plane (flat `max_ctx * index_width` or the 5,120-row tail ring, `index_ring_rows` :280) | `max_ctx` | `width x 4` B/token plus the pool keys (`latent_kv_bytes_per_token_for_plan` :2364) |

The SWA ring default is armed only by the step37 SlidingGatedMoe program (`memra-engine/src/lib.rs:1442-1444`
`set_swa_ring_default(true)`); the Qwen3.5/3.8 plans compile no sliding layer, so `ring = None` and
`alloc_rows = max_ctx` for every full-attention layer of both models here. `len` starts at 0 and `len_d` at 0: the
planes are allocated, not written.

When it happens and with what `max_ctx`. Every session cache is born at `ctx_cap`: the cold plain session
`pp::new_cache_planned(engine, cfg, plan, ctx_cap)` (`worker.rs:25336`, with `alloc_with_single_reclaim_retry` :6811,
which evicts prefix entries once before failing), the cold MTP-spec session `lm.model.new_session(engine, ctx_cap)`
(`worker.rs:24718`, `spec.rs:9064-9069`; its `cache_max_ctx()` :1564 is that cache's `max_ctx`), the prefix-hit
carrier (`worker.rs:23410`, `:23565`: a fresh `new_cache_planned(.., ctx_cap)` before the restore), and the affinity
grow (`:22862`, `:23003`). Nothing grows a cache in place: a grow is a new cache at the larger cap plus a D2D
checkpoint copy.

### 1.3 Bytes per token per class

Per full-attention layer per token: `kv_dim = n_head_kv x head_dim = 4 x 256 = 1024`, so
`(1024 / 32) x 34 + (1024 / 32) x 24 = 1,088 + 768 = 1,856 B`. Spec sessions add one MTP block's scratch K+V of the same
shape (`spec_session_kv_bytes_per_token`, `spec.rs:9008-9020`, `mtp_scratch_layout`). GGUF headers read on this rig:

| Class | Artifact | `block_count` / trunk layers | `full_attention_interval` -> full-attention layers | plain B/token | spec B/token | server line today |
| --- | --- | --- | --- | ---: | ---: | --- |
| Qwen3.5-9B | `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` | 33 / 32 | 4 -> 8 | 8 x 1,856 = **14,848** | 9 x 1,856 = **16,704** | `[admission] request cost: model="q9" ctx=65600 path=spec = 16704 B/token x ctx ...` (this cell); `path=plain = 14848` (day 24) |
| Qwen3.8-27B (the mint; the class Qwen3.8 belongs to) | `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` | 65 / 64 | 4 -> 16 | 16 x 1,856 = **29,696** | 17 x 1,856 = **31,552** | `[admission] request cost: model="q38" ctx=262208 path=spec = 31552 B/token x ctx ...` (this cell); `path=plain = 29696` (day 14) |

The 24 (9B) and 48 (27B) recurrent layers cost a fixed per-session state (`conv_width x (kernel - 1)` plus two
`state_width` f32 planes), inside the measured `fixed` term of the request-cost line (0 to 794 MB high-water below),
not a per-token row. `cache_bytes_per_token_for_plan` (`lib.rs:2313-2335`) is the one function admission and the
allocator share for the coefficient, so the line's B/token is the allocator's B/token.

### 1.4 Free or parked

At retirement (`worker.rs:20717` `active.remove(i)`) a cache that may park (`retire_may_park(aborted, oom_teardown)`
:29010: not aborted, not an OOM teardown; `prime_service.pending` refuses too :20797) goes to the whole-session reuse
pool at its **unchanged** capacity: `let cap = cache.max_ctx` (`:21023`) unless `MEMRA_KV_PARK_COMPACT=1`
(`kv_park_compact_on` :2391, default OFF, `docs/FLAGS.md:324`). Pools: `MEMRA_REUSE_POOL` = 2 entries per (model,
cache namespace) for each of the continuation, MTP-spec and dspark pools (`:2356-2364`), `MEMRA_REUSE_POOL_GLOBAL_CAP`
= 16 across all (`:2370-2378`). A parked entry has no TTL (no `parked_at` expiry exists in the worker); it leaves by
LRU eviction on a later park (`prepare_park` :2926) or by pressure reclaim. Otherwise the cache drops, and its bytes
return to the CUDA pool, not to the driver (`KV-PHYSICAL-RECLAIM.md`: `RECLAIM-DIAG: freed but not observable`).
Step-OOM parking is only for a session with nothing generated and nothing emitted (`step_oom_parkable` :22500-22507);
an OOM teardown never parks (`oom_teardown_sessions_never_publish_reusable_kv` :29443). An emitting session is never
preempted.

The day-14 observation (request path retaining 67,200 B per prompt token with the prefix cache idle, 27B, plain path,
bounded output) is consistent in order with this pool: two continuation-pool entries per namespace at `ctx_cap =
P + max_tokens + 8` retain `2 x 29,696 = 59,392` B per prompt token as the two newest turns grow; the remaining
7,808 B/token is not attributed by this census (the prime workspace and the pool's block rounding are the candidates).
Today's cell records `cuda_pool_used_bytes` before and after every request so the parked bytes are read, not inferred
(section 2.4).

### 1.5 What a prefix hit copies and what it shares

Publication: `prefix_snapshot` (`worker.rs:13150`) allocates a fresh `alloc_u8(pos x k_tok_bytes)` and
`alloc_u8(pos x v_tok_bytes)` per full-attention layer and D2D-copies exactly `pos` rows (`:13236-13243`), plus the
recurrent state and any latent plane; the entry holds `pos` rows, not `max_ctx` (day 14: 29,750 B/token on the 27B;
today: 202.3 MB for a 1,440-token spec-boundary entry on the 27B, 140,486 B/token, because a spec-boundary capture
also carries the draft plane and the boundary hidden). Consumption: `prefix_restore` (`:13578`) ->
`prefix_restore_at` (`:13368`): the destination must be a fresh cache at the session's `ctx_cap` (`:13443-13448`,
`restore_len <= cache.max_ctx` :13436), then per layer `copy_u8_into(dst.k, 0, src.k, restore_len x k_tok_bytes)` and
the same for V (`:13547-13553`), `copy_into` for the recurrent state (`:13557-13558`), `restore_plane` for latent
(`:13561`). Nothing is shared: the entry keeps its own bytes and the session holds a private copy inside a
`ctx_cap`-row allocation. A hybrid model refuses a mid-entry restore (`:13470-13476`, recurrent state exists only at
the captured endpoint), so a hit on these classes is whole-entry or nothing.

### 1.6 How the admission cost model prices the same request

`AdmissionCostModel::estimate(admission_cap, prompt_rows, spec)` (`worker.rs:2719-2723`) =
`context_cache_bytes(bpt, ring_bpt, ring_rows, admission_cap)` (`admit_predict.rs:494-511`: `bpt x cap` with no ring)
`+ prefill_workspace_bytes(prompt_rows)` (`:2705-2717`, keyed on the prompt, not the cap, by design)
`+ activation_bytes` (the learned fixed high-water, `observe` :2740). The receipt is the `[admission] request cost`
line (`:17911-17920`); the per-session draft-state line adds `D`; the predictive book (`RequestCharge::from_physical_cost`,
`admit_predict.rs:564-581`, memra#476, day 24) re-keys only the context term to `prompt + predicted + 8`. So the
physical book charges the allocation (`bpt x (ctx_cap + 56)` plus workspace and fixed) and the shadow book charges a
prediction; neither charges what a request ends up using, and the allocation itself follows the physical book.

### 1.7 Allocated over used, from the census alone (before the cell)

`allocated = bpt x ctx_cap`, `used = bpt x (P + G)`. Open arm at the checkpoint context: 27B `262,144 x 31,552 =
8,271,167,488 B` per request whatever the prompt; 9B `262,144 x 16,704 = 4,378,853,376 B` (the local cell pins
`MEMRA_CTX=65536`: `1,094,713,344 B`). Bounded arm: `(P + max_tokens + 8) x bpt`, so the ratio is `(P + max_tokens
+ 8) / (P + G)`, 1.00 whenever the client's `max_tokens` is spent. Warm continuation: the same `ctx_cap` allocation as
the open arm plus the D2D copy of the hit, the entry staying resident. Section 2 measures these.

## 2. The cell: allocated versus used KV under a fixed mix, both cards

### 2.1 Pre-registration (before any run)

Harness: `run-day26-cell.sh <cell> <AB|BA>` (boots `memra-server` with `MEMRA_COMPAT=openai`,
`MEMRA_MODELS=<key>=<artifact>`, `MEMRA_ADMIT_PREDICT_SHADOW=1` for receipts, stderr stamped per line),
`day26-client.py` (the mix and a 250 ms sampler of `nvidia-smi memory.used` and `/metrics`
`cuda_driver_free_bytes`, `cuda_pool_used_bytes`, `cuda_pool_reserved_bytes`, `admission_booked_bytes`,
`active_sessions`, `prefix_cache_bytes`), `day26-parse.py` (the rows and the per-arm summary). Mix, per prompt length
L0/L1/L2 (5,000 / 10,000 / 20,000 characters of `docs/SERVING.md`, about 1.5k / 3.1k / 5.8k tokens), N=5 reps each
with a distinct salt so no rep shares a prefix with another: (i) `max_tokens` omitted, "summarize in one sentence",
greedy; (ii) `max_tokens=96` at the same lengths; (iii) the (i) conversation continued (`user P_k`, `assistant
reply_k`, `user "Add one more sentence"`), `max_tokens` omitted. Order AB runs (i) then (ii) inside each length, BA the
reverse; (iii) runs after every first turn (the deferred shape) and, in the `warm` cell added after the first two
orders, right after each (i) (the hit shape). Per request the parser records P, G, `cached_tokens`, the
`[admission] request cost` line stamped inside [submit, done] (path, B/token, booked ctx, booked MB), allocated KV
`= B/token x ctx_cap` (open: booked ctx minus 64; bounded: `P + 96 + 8`), used KV `= B/token x (P + G)`, their ratio,
driver free and pool used before and after, and the elapsed ms. The concurrency line divides the idle driver free
before the first request by the median booked cost of each arm: arithmetic on the cell's own numbers, not an
admission run. One lock hold per order; no engine change; the pre-registered comparison is the ratio per arm, both
orders, both cards. `MEMRA_CTX` unset on the target card (the checkpoint's 262,144); `MEMRA_CTX=65536` on the 24 GB
local card (an unset value would allocate 4.38 GB per open request and the two reuse pools would retain 17.5 GB at
their default caps beside the 5.66 GB weights, so the naked 262,144 does not fit the mix on this card; the census
arithmetic for that value is in 1.7).

### 2.2 Target card, one RTX PRO 6000 Blackwell, 27B, `MEMRA_CTX` unset (`pro-single-day26/`)

Collector cells `cell-ab` and `cell-ba` (`command.capture.json`: `status executed-not-qualified`, `qualification
false`, `exit_code 0`; 250 ms `command.gpu.csv`, 779 and 778 rows), receipts `cells/ab`, `cells/ba`. Regime: rig line
before `32 C, 32.65 W` / after `48 C, 90.71 W` (AB); `44 C, 53.15 W` / `49 C, 89.28 W` (BA); `power.limit 600.00 W`;
`compute-apps` empty before and after. 46 requests per order (1 warmup + 45), `non-200=0`. Boot lines, verbatim:

```text
[prefix-cache] on: budget 800MB (800325632 B, derived: 2 x 400162816 B max entry for model "q38" at MEMRA_CTX=8192, requested 800325632 B; boot driver free 86519709696 B, post-reserve clamp 84909096960 B), policy plain-LRU (global oldest unleased entry first; leases untouchable), min prefix 64 tokens, immediate partial restore=off (rollback) (transformer-only; hybrid mid-entry + routed-MoE N/A)
[admit-predict] shadow armed: budget_bytes=80963874512 budget_src=derived(effective_free_bytes=84065415872 - prefix_cache_budget_bytes=800325632 - admission_reserve_bytes=2301215728) exempt_tenants=0 enforce=false (logging only, nothing is rejected; enforcement is a separate flip)
[admit-mem] door=OFF open_output_tokens=8192 defer_budget_ms=8000 (...)
```

Per-arm allocated over used, verbatim from `cells/ab/REPORT.txt` (order AB) and `cells/ba/REPORT.txt` (order BA);
P and G are identical across the two orders on every request (greedy, same prompts), so the ratio rows are identical
and only the booked MB differ where the learned `fixed` high-water had not yet moved:

```text
AB arm=i L0 N=5 P=[1481, 1481, 1484, 1484, 1483] G=[591, 369, 589, 457, 545] cached=[0, 0, 0, 0, 0] ratios=[126.52, 141.7, 126.46, 135.06, 129.26] min=126.46 median=129.26 max=141.70 alloc_B=8271167488 used_B_median=63987456 booked_MB=[9291, 9777, 9779, 9779, 9778] inherited=1
AB arm=i L1 N=5 P=[3138, 3138, 3139, 3138, 3136] G=[478, 954, 673, 508, 494] cached=[0, 0, 0, 0, 0] ratios=[72.5, 64.06, 68.77, 71.9, 72.22] min=64.06 median=71.90 max=72.50 alloc_B=8271167488 used_B_median=115038592 booked_MB=[10571, 10571, 10572, 10571, 10570] inherited=1
AB arm=i L2 N=5 P=[5797, 5803, 5802, 5805, 5803] G=[629, 497, 505, 443, 354] cached=[0, 0, 0, 0, 0] ratios=[40.79, 41.61, 41.56, 41.96, 42.58] min=40.79 median=41.61 max=42.58 alloc_B=8271167488 used_B_median=198777600 booked_MB=[11065, 11066, 11065, 11066, 11066] inherited=0
AB arm=ii L0 N=5 P=[1715, 1714, 1713, 1713, 1710] G=[96, 96, 96, 96, 96] cached=[0, 0, 0, 0, 0] ratios=[1.0, 1.0, 1.0, 1.0, 1.0] min=1.00 median=1.00 max=1.00 alloc_B=57393088 used_B_median=57077568 booked_MB=[1675, 1675, 1674, 1674, 1673] inherited=1
AB arm=ii L1 N=5 P=[3000, 3003, 2999, 2992, 2991] G=[96, 96, 96, 96, 96] cached=[0, 0, 0, 0, 0] ratios=[1.0, 1.0, 1.0, 1.0, 1.0] min=1.00 median=1.00 max=1.00 alloc_B=97937408 used_B_median=97653440 booked_MB=[2332, 2333, 2331, 2328, 2327] inherited=0
AB arm=ii L2 N=5 P=[5706, 5702, 5695, 5692, 5689] G=[96, 96, 96, 96, 96] cached=[0, 0, 0, 0, 0] ratios=[1.0, 1.0, 1.0, 1.0, 1.0] min=1.00 median=1.00 max=1.00 alloc_B=183317120 used_B_median=182717632 booked_MB=[2975, 2975, 2975, 2975, 2975] inherited=0
AB arm=iii L0 N=5 P=[1572, 1565, 1595, 1563, 1580] G=[384, 239, 227, 484, 561] cached=[0, 0, 0, 0, 0] ratios=[134.02, 145.31, 143.88, 128.06, 122.44] min=122.44 median=134.02 max=145.31 alloc_B=8271167488 used_B_median=61715712 booked_MB=[9821, 9818, 9832, 9817, 9825] inherited=0
AB arm=iii L1 N=5 P=[3236, 3220, 3228, 3234, 3223] G=[564, 621, 407, 319, 211] cached=[0, 0, 0, 0, 0] ratios=[68.99, 68.25, 72.12, 73.78, 76.34] min=68.25 median=72.12 max=76.34 alloc_B=8271167488 used_B_median=114691520 booked_MB=[10618, 10611, 10615, 10617, 10612] inherited=0
AB arm=iii L2 N=5 P=[5887, 5884, 5884, 5868, 5885] G=[345, 487, 576, 354, 539] cached=[0, 0, 0, 0, 0] ratios=[42.06, 41.15, 40.58, 42.13, 40.81] min=40.58 median=41.15 max=42.13 alloc_B=8271167488 used_B_median=201017792 booked_MB=[11067, 11067, 11067, 11067, 11067] inherited=1
BA arm=i L0 ... ratios=[126.52, 141.7, 126.46, 135.06, 129.26] min=126.46 median=129.26 max=141.70 alloc_B=8271167488 used_B_median=63987456 booked_MB=[9294, 9777, 9779, 9779, 9778] inherited=1
BA arm=i L1 ... ratios=[72.5, 64.06, 68.77, 71.9, 72.22] min=64.06 median=71.90 max=72.50 alloc_B=8271167488 used_B_median=115038592 booked_MB=[10571, 10571, 10572, 10571, 10570] inherited=1
BA arm=i L2 ... ratios=[40.79, 41.61, 41.56, 41.96, 42.58] min=40.79 median=41.61 max=42.58 alloc_B=8271167488 used_B_median=198777600 booked_MB=[11065, 11066, 11065, 11066, 11066] inherited=0
BA arm=ii L0 ... ratios=[1.0, 1.0, 1.0, 1.0, 1.0] min=1.00 median=1.00 max=1.00 alloc_B=57393088 used_B_median=57077568 booked_MB=[1189, 1192, 1191, 1191, 1190] inherited=1
BA arm=ii L1 ... ratios=[1.0, 1.0, 1.0, 1.0, 1.0] min=1.00 median=1.00 max=1.00 alloc_B=97937408 used_B_median=97653440 booked_MB=[2332, 2333, 2331, 2328, 2327] inherited=0
BA arm=ii L2 ... ratios=[1.0, 1.0, 1.0, 1.0, 1.0] min=1.00 median=1.00 max=1.00 alloc_B=183317120 used_B_median=182717632 booked_MB=[2975, 2975, 2975, 2975, 2975] inherited=0
BA arm=iii L0 ... ratios=[134.02, 145.31, 143.88, 128.06, 122.44] min=122.44 median=134.02 max=145.31 alloc_B=8271167488 used_B_median=61715712 booked_MB=[9821, 9818, 9832, 9817, 9825] inherited=0
BA arm=iii L1 ... ratios=[68.99, 68.25, 72.12, 73.78, 76.34] min=68.25 median=72.12 max=76.34 alloc_B=8271167488 used_B_median=114691520 booked_MB=[10618, 10611, 10615, 10617, 10612] inherited=0
BA arm=iii L2 ... ratios=[42.06, 41.15, 40.58, 42.13, 40.81] min=40.58 median=41.15 max=42.13 alloc_B=8271167488 used_B_median=201017792 booked_MB=[11067, 11067, 11067, 11067, 11067] inherited=1
```

Concurrency arithmetic at the served context (262,144), verbatim (idle driver free before the first request,
82,759,516,160 B, the same in both orders; the ready-time `/metrics` gauges read 0 until the first tick, so the parser
takes the first idle samples):

```text
AB L0: P~1483 open booked_MB=9778 -> 8 sessions; bounded booked_MB=1674 -> 49 sessions (free_ready=82759516160)
AB L1: P~3138 open booked_MB=10571 -> 7 sessions; bounded booked_MB=2331 -> 35 sessions (free_ready=82759516160)
AB L2: P~5803 open booked_MB=11066 -> 7 sessions; bounded booked_MB=2975 -> 27 sessions (free_ready=82759516160)
BA L0: P~1483 open booked_MB=9778 -> 8 sessions; bounded booked_MB=1191 -> 69 sessions (free_ready=82759516160)
BA L1: P~3138 open booked_MB=10571 -> 7 sessions; bounded booked_MB=2331 -> 35 sessions (free_ready=82759516160)
BA L2: P~5803 open booked_MB=11066 -> 7 sessions; bounded booked_MB=2975 -> 27 sessions (free_ready=82759516160)
```

(BA L0's bounded cost is 1,191 MB against AB's 1,674 MB because in BA the bounded arm ran first, before the first open
request raised the learned `fixed` term from 308 to 794 MB; the `[admission]` lines show it.) `MEMRA_MAX_SESSIONS` is
64, so under (ii) the session cap, not memory, is the binding limit at L0.

The warm arm on both orders: `cached=0` on all 30 continuations and `prefix-cache hit lines: 0`. The server's own
lines say why: `45 [prefix-cache] insert (spec-boundary)` (one per first turn, 202.3 MB for 1,440 tokens up to
about 800 MB for the L2 entries) against `43 [prefix-cache] evict (LRU)` inside the derived 800 MB budget, whose
derivation line reads `at MEMRA_CTX=8192` while the sessions were capped at 262,144. In the deferred shape every
entry had been evicted before its continuation arrived, so (iii) measured the cold allocation of a second turn: the
same 8,271,167,488 B as (i). The hit shape is the `warm` cell (2.4).

Pool retention, read from the per-request `cuda_pool_used_bytes` (bytes): the first open request took pool used from
17,914,203,332 (after warmup) and the pool sat at 36.4 to 36.6 GB through the bounded arm and the continuations;
`iii-L2-r4` moved it from 36,476,670,024 to 45,858,568,652 within one request; run maximum 45,858,568,652
(AB) / 45,871,448,012 (BA); driver free ended at 52,694,745,088 (AB) / 52,896,071,680 (BA) from an idle
82,759,516,160: about 30 GB of the card stays with the process after 45 sequential requests of which none is active.
Four parked whole-session entries at 8,271,167,488 B each (two per pool, MTP-spec and continuation) account for
33.1 GB of pool used; this cell reads the pool, it does not attribute entries by name.

### 2.3 Local RTX 5090 Laptop GPU, 9B, `MEMRA_CTX=65536` (`rtx5090-day26/`)

Order AB (`ab/`, `ab.exit` 0, 46 requests, `non-200=0`, `compute-apps` empty before and after, rig line before
`54 C, 9.32 W` / after `76 C, 30.62 W`, `power.limit [N/A]`; 2,789 samples). Boot: `[prefix-cache] on: budget 2052MB
(2051538944 B, derived: 2 x 1025769472 B max entry for model "q9" at MEMRA_CTX=65536 ...)`, `[admit-predict] shadow
armed: budget_bytes=12831671424 ...`, `[admit-mem] door=OFF ...`. This model reasons before it answers: the open arm
generated 2,413 to 5,048 tokens per one-sentence summary (one continuation 17,815), which is what the used KV
includes. Verbatim:

```text
AB arm=i L0 N=5 P=[1439, 1439, 1442, 1442, 1441] G=[3921, 4363, 4123, 4761, 5048] cached=[0, 0, 0, 0, 0] ratios=[12.23, 11.3, 11.78, 10.57, 10.1] min=10.10 median=11.30 max=12.23 alloc_B=1094713344 used_B_median=96916608 booked_MB=[1718, 1839, 1840, 1840, 1840] inherited=1
AB arm=i L1 N=5 P=[3096, 3096, 3097, 3096, 3094] G=[3009, 3799, 2413, 2833, 3476] cached=[0, 0, 0, 0, 0] ratios=[10.73, 9.5, 11.89, 11.05, 9.98] min=9.50 median=10.73 max=11.89 alloc_B=1094713344 used_B_median=101977920 booked_MB=[2436, 2436, 2437, 2436, 2436] inherited=1
AB arm=i L2 N=5 P=[5755, 5761, 5760, 5763, 5761] G=[4817, 3715, 4137, 4584, 3248] cached=[0, 0, 0, 0, 0] ratios=[6.2, 6.92, 6.62, 6.33, 7.27] min=6.20 median=6.62 max=7.27 alloc_B=1094713344 used_B_median=165319488 booked_MB=[2824, 2824, 2824, 2824, 2824] inherited=0
AB arm=ii L0 N=5 P=[1673, 1672, 1671, 1671, 1668] G=[96, 96, 96, 96, 96] cached=[0, 0, 0, 0, 0] ratios=[1.0, 1.0, 1.0, 1.0, 1.0] min=1.00 median=1.00 max=1.00 alloc_B=29683008 used_B_median=29515968 booked_MB=[858, 858, 858, 858, 856] inherited=1
AB arm=ii L1 N=5 P=[2958, 2961, 2957, 2950, 2949] G=[96, 96, 96, 96, 96] cached=[0, 0, 0, 0, 0] ratios=[1.0, 1.0, 1.0, 1.0, 1.0] min=1.00 median=1.00 max=1.00 alloc_B=51147648 used_B_median=50997312 booked_MB=[1343, 1344, 1343, 1340, 1340] inherited=0
AB arm=ii L2 N=5 P=[5664, 5660, 5653, 5650, 5647] G=[96, 96, 96, 96, 96] cached=[0, 0, 0, 0, 0] ratios=[1.0, 1.0, 1.0, 1.0, 1.0] min=1.00 median=1.00 max=1.00 alloc_B=96348672 used_B_median=96031296 booked_MB=[1824, 1824, 1824, 1824, 1823] inherited=0
AB arm=iii L0 N=5 P=[1486, 1500, 1519, 1498, 1492] G=[2173, 4715, 4038, 2606, 4473] cached=[0, 0, 0, 0, 0] ratios=[17.91, 10.54, 11.79, 15.97, 10.99] min=10.54 median=11.79 max=17.91 alloc_B=1094713344 used_B_median=92824128 booked_MB=[1856, 1861, 1868, 1860, 1858] inherited=0
AB arm=iii L1 N=5 P=[3152, 3152, 3173, 3151, 3151] G=[2490, 4315, 2997, 2567, 3359] cached=[0, 0, 0, 0, 0] ratios=[11.62, 8.78, 10.62, 11.46, 10.07] min=8.78 median=10.62 max=11.62 alloc_B=1094713344 used_B_median=103063680 booked_MB=[2457, 2457, 2464, 2456, 2456] inherited=2
AB arm=iii L2 N=5 P=[5803, 5815, 5822, 5821, 5812] G=[4511, 4564, 3246, 17815, 3724] cached=[0, 0, 0, 0, 0] ratios=[6.35, 6.31, 7.23, 2.77, 6.87] min=2.77 median=6.35 max=7.23 alloc_B=1094713344 used_B_median=172285056 booked_MB=[2825, 2825, 2825, 2825, 2825] inherited=0
AB L0: P~1441 open booked_MB=1840 -> 8 sessions; bounded booked_MB=858 -> 18 sessions (free_ready=16029908992)
AB L1: P~3096 open booked_MB=2436 -> 6 sessions; bounded booked_MB=1343 -> 11 sessions (free_ready=16029908992)
AB L2: P~5761 open booked_MB=2824 -> 5 sessions; bounded booked_MB=1824 -> 8 sessions (free_ready=16029908992)
```

Warm arm: `cached=0` on all 15, `prefix-cache hit lines: 0`, `60 [prefix-cache] insert (spec-boundary)` against
`45 [prefix-cache] evict (LRU)` inside 2,052 MB: the same deferred-shape eviction as on the target card. Pool:
`pool_used max=14527966764 last=13121243464`, driver free `last=8580825088` from an idle 16,029,908,992. On this card
the bounded arm's booked cost is the prefill workspace (1.5 GB at 5.7k prompt rows) and the fixed term, not KV
(96 MB at L2), so bounding `max_tokens` buys 8 -> 18 sessions at L0 and 5 -> 8 at L2, where on the target card the
KV term dominates and the same change buys 8 -> 49 (or 69) at L0.

Order BA (`ba/`, `ba.exit` 0, 46 requests, `non-200=0`, rig before `62 C, 17.85 W` / after `75 C, 29.25 W`; 2,813
samples). P and G identical to AB on 44 of 45 requests; the one difference is `iii-L2-r3`, G 17,815 (AB) against
18,314 (BA): both runs of that request took exactly 90,003 ms submit-to-done and the server lines inside the window
are identical (`[worker] spec-affinity: declined (history diverged at 6 of checkpoint 5792; 2 parked, 5821 prompt
tokens; model q9)`, `[spec-k] ... K=3 source=cold-long prompt=5821 cached=0 lcp=6`), so the runaway generation was
cut by the 90 s request deadline at whatever token the clock reached; tokens per second differed, the program did
not. Verbatim:

```text
BA arm=i L0 ... ratios=[12.23, 11.3, 11.78, 10.57, 10.1] min=10.10 median=11.30 max=12.23 alloc_B=1094713344 used_B_median=96916608 booked_MB=[1721, 1839, 1840, 1840, 1840] inherited=1
BA arm=i L1 ... ratios=[10.73, 9.5, 11.89, 11.05, 9.98] min=9.50 median=10.73 max=11.89 alloc_B=1094713344 used_B_median=101977920 booked_MB=[2436, 2436, 2437, 2436, 2436] inherited=1
BA arm=i L2 ... ratios=[6.2, 6.92, 6.62, 6.33, 7.27] min=6.20 median=6.62 max=7.27 alloc_B=1094713344 used_B_median=165319488 booked_MB=[2824, 2824, 2824, 2824, 2824] inherited=0
BA arm=ii L0 ... ratios=[1.0, 1.0, 1.0, 1.0, 1.0] min=1.00 median=1.00 max=1.00 alloc_B=29683008 used_B_median=29515968 booked_MB=[737, 740, 739, 739, 738] inherited=1
BA arm=ii L1 ... ratios=[1.0, 1.0, 1.0, 1.0, 1.0] min=1.00 median=1.00 max=1.00 alloc_B=51147648 used_B_median=50997312 booked_MB=[1343, 1344, 1343, 1340, 1340] inherited=0
BA arm=ii L2 ... ratios=[1.0, 1.0, 1.0, 1.0, 1.0] min=1.00 median=1.00 max=1.00 alloc_B=96348672 used_B_median=96031296 booked_MB=[1824, 1824, 1824, 1824, 1823] inherited=0
BA arm=iii L0 ... cached=[0, 0, 0, 0, 0] ratios=[17.91, 10.54, 11.79, 15.97, 10.99] min=10.54 median=11.79 max=17.91 alloc_B=1094713344 used_B_median=92824128 booked_MB=[1856, 1861, 1868, 1860, 1858] inherited=0
BA arm=iii L1 ... cached=[0, 0, 0, 0, 0] ratios=[11.62, 8.78, 10.62, 11.46, 10.07] min=8.78 median=10.62 max=11.62 alloc_B=1094713344 used_B_median=103063680 booked_MB=[2457, 2457, 2464, 2456, 2456] inherited=2
BA arm=iii L2 N=5 P=[5803, 5815, 5822, 5821, 5812] G=[4511, 4564, 3246, 18314, 3724] cached=[0, 0, 0, 0, 0] ratios=[6.35, 6.31, 7.23, 2.72, 6.87] min=2.72 median=6.35 max=7.23 alloc_B=1094713344 used_B_median=172285056 booked_MB=[2825, 2825, 2825, 2825, 2825] inherited=0
BA L0: P~1441 open booked_MB=1840 -> 8 sessions; bounded booked_MB=739 -> 21 sessions (free_ready=16029908992)
BA L1: P~3096 open booked_MB=2436 -> 6 sessions; bounded booked_MB=1343 -> 11 sessions (free_ready=16029908992)
BA L2: P~5761 open booked_MB=2824 -> 5 sessions; bounded booked_MB=1824 -> 8 sessions (free_ready=16029908992)
```

`prefix-cache hit lines: 0` (60 inserts, 45 LRU evictions); `pool_used max=14534983212 last=13121243464`; driver free
`last=8513716224`. The `[worker] spec-affinity: declined (... 2 parked ...)` line on every continuation is the
whole-session reuse pool at its default cap of 2 per namespace: with fifteen conversations interleaved, the parked
sessions a continuation finds are never its own.

### 2.4 The hit shape (`warm` cells, (iii) right after its (i))

Target card (`pro-single-day26/cell-warm`, `cells/warm`; `status executed-not-qualified`, `qualification false`,
`exit_code 0`, `elapsed_seconds 183.4`; rig `37 C, 34.06 W` before / `47 C, 90.92 W` after; 46 requests, `non-200=0`).
Same binary, same prompts, order AB with each continuation sent right after its first turn. Now every continuation
hit: `15 [prefix-cache] hit` and `15 [prefix-cache] spec` lines, the first pair verbatim
`[prefix-cache] hit: 1440 of 1572 prompt tokens from cache (model q38)` and
`[prefix-cache] spec restore: 1440 of 1572 prompt tokens + draft plane from cache [suffix queued] (model q38)`, and
`cached_tokens` equal to the spec-boundary entry's rows on all 15 (`1440`, `3104`, `5760`; the entry ends where the
spec prime left the prompt tail, so a hit on this class is the whole entry, never a partial). The allocation did not
move: the carrier for the restore is a fresh `ctx_cap`-row cache, so (iii) allocates the same 8,271,167,488 B as (i)
and copies the entry's rows into it (`pool_used` before/after `iii-L0-r0`: 17,914,203,332 -> 27,443,996,232 on a
request whose used KV is 61,715,712 B). Verbatim:

```text
warm arm=i L0 N=5 P=[1481, 1481, 1484, 1484, 1483] G=[591, 369, 589, 457, 545] cached=[0, 0, 0, 0, 0] ratios=[126.52, 141.7, 126.46, 135.06, 129.26] min=126.46 median=129.26 max=141.70 alloc_B=8271167488 used_B_median=63987456 booked_MB=[9291, 9777, 9779, 9779, 9778] inherited=0
warm arm=i L1 N=5 P=[3138, 3138, 3139, 3138, 3136] G=[478, 954, 673, 508, 494] cached=[0, 0, 0, 0, 0] ratios=[72.5, 64.06, 68.77, 71.9, 72.22] min=64.06 median=71.90 max=72.50 alloc_B=8271167488 used_B_median=115038592 booked_MB=[10571, 10571, 10572, 10571, 10571] inherited=0
warm arm=i L2 N=5 P=[5797, 5803, 5802, 5805, 5803] G=[629, 497, 505, 443, 354] cached=[0, 0, 0, 0, 0] ratios=[40.79, 41.61, 41.56, 41.96, 42.58] min=40.79 median=41.61 max=42.58 alloc_B=8271167488 used_B_median=198777600 booked_MB=[11065, 11066, 11066, 11066, 11066] inherited=0
warm arm=iii L0 N=5 P=[1572, 1565, 1595, 1563, 1580] G=[384, 239, 227, 484, 561] cached=[1440, 1440, 1440, 1440, 1440] ratios=[134.02, 145.31, 143.88, 128.06, 122.44] min=122.44 median=134.02 max=145.31 alloc_B=8271167488 used_B_median=61715712 booked_MB=[9821, 9818, 9832, 9817, 9825] inherited=0
warm arm=iii L1 N=5 P=[3236, 3220, 3228, 3234, 3223] G=[564, 621, 407, 319, 211] cached=[3104, 3104, 3104, 3104, 3104] ratios=[68.99, 68.25, 72.12, 73.78, 76.34] min=68.25 median=72.12 max=76.34 alloc_B=8271167488 used_B_median=114691520 booked_MB=[10618, 10611, 10615, 10617, 10612] inherited=0
warm arm=iii L2 N=5 P=[5887, 5884, 5884, 5868, 5885] G=[345, 487, 576, 354, 539] cached=[5760, 5760, 5760, 5760, 5760] ratios=[42.06, 41.15, 40.58, 42.13, 40.81] min=40.58 median=41.15 max=42.13 alloc_B=8271167488 used_B_median=201017792 booked_MB=[11067, 11067, 11067, 11067, 11067] inherited=0
```

The hit changes the prefill (a 1,440-row restore instead of a 1,481-row prime; `iii-L0-r0` 2,999 ms against
`i-L0-r0`'s cold turn) and nothing about residency: the entry stays resident (202.3 MB), the session holds a private
copy of the same rows inside 8.27 GB, and the ratio is the open arm's. `pool_used max=45779546572`, `last=19336305864`;
driver free `last=53063843840` from the same idle 82,759,516,160.

Local RTX 5090 Laptop GPU (`rtx5090-day26/warm`, `warm.exit` 0, 46 requests, `non-200=0`, rig `68 C, 34.04 W` before /
`76 C, 29.72 W` after, 2,811 samples): the same shape, `15 [prefix-cache] hit` and `15 [prefix-cache] spec` lines
(first: `[prefix-cache] hit: 1408 of 1486 prompt tokens from cache (model q9)`), `cached_tokens` equal to the entry
rows on all 15 (`1408`, `3072`, `5728`), `45 [prefix-cache] insert (spec-boundary)`, `32 [prefix-cache] evict (LRU)`.
Same allocation as (i) (1,094,713,344 B); `iii-L0-r0` moved `pool_used` from 10,009,861,704 to 11,461,794,604 for
61,119,936 B of used KV. Verbatim:

```text
warm arm=iii L0 N=5 P=[1486, 1500, 1519, 1498, 1492] G=[2173, 4715, 4038, 2606, 4473] cached=[1408, 1408, 1408, 1408, 1408] ratios=[17.91, 10.54, 11.79, 15.97, 10.99] min=10.54 median=11.79 max=17.91 alloc_B=1094713344 used_B_median=92824128 booked_MB=[1856, 1861, 1868, 1860, 1858] inherited=0
warm arm=iii L1 N=5 P=[3152, 3152, 3173, 3151, 3151] G=[2490, 4315, 2997, 2567, 3359] cached=[3072, 3072, 3072, 3072, 3072] ratios=[11.62, 8.78, 10.62, 11.46, 10.07] min=8.78 median=10.62 max=11.62 alloc_B=1094713344 used_B_median=103063680 booked_MB=[2457, 2457, 2464, 2456, 2456] inherited=0
warm arm=iii L2 N=5 P=[5803, 5815, 5822, 5821, 5812] G=[4511, 4564, 3246, 16378, 3724] cached=[5728, 5728, 5728, 5728, 5728] ratios=[6.35, 6.31, 7.23, 2.95, 6.87] min=2.95 median=6.35 max=7.23 alloc_B=1094713344 used_B_median=172285056 booked_MB=[2825, 2825, 2825, 2825, 2825] inherited=0
```

(The (i) rows of this cell equal the AB and BA rows to the token; `iii-L2-r3` is again the 90 s deadline cut, here at
16,378 tokens.) `pool_used max=14527258156 last=10815941064`; driver free `last=8478064640`.

Reading the two shapes together: a warm continuation on these hybrid classes hits only when its whole spec-boundary
entry is still resident, and a hit changes the prefill cost and nothing about residency. The restore's carrier is
allocated at `ctx_cap` exactly like a cold session, the entry stays resident beside it, and the ratio is the open
arm's on both cards.

## 3. Design note

`research/spill-b-20260919/KV-RESIDENCY-DESIGN.md`: options (a) admission by memory with a bounded open-output
default (the existing `MEMRA_ADMIT_BY_MEMORY` door, decide-by 2026-09-23; policy only, 0.5 agent-day for its decision
cell), (b) grow-on-demand contiguous planes on the VMM door (receipts `ACTIVE-8K G1 PASS` and the classified 32k series
on both cards; 3 to 4 agent-days; the address never moves so the one-program law holds by construction, a grow series
cell and a boundary stall cell are the new gates), (c) paged KV (every FA decode and prefill kernel, the append writers,
the batched pointer table, the spec repair, graph capture; a per-kernel byte-tape identity gate on two arch guards;
10 to 15 agent-days), (d) copy-on-write prefix sharing on top of (c) (3 to 5 agent-days). Recommended order: (a), then
(b), then (c) only if the residual after both justifies the kernel rewrite on the shared-prefix workload, (d) riding on
(c). Reason in one line: today's cell shows the open arm's ratio is the cap rule (126 to 142 at L0 on the target card
with 8.27 GB allocated for 64 MB used), which (a) removes with no numeric change, and the bounded arm's ratio is
already 1.00, so the layout options buy the prefix copy and preemption, not the ratio. The owner decides.

## 4. Checks actually run

| Check | Result |
| --- | --- |
| `git merge --no-ff origin/main` (`8e94a480b`) | one INDEX.md conflict, resolved by hand (both rows kept), `check-conflict-markers: OK` |
| `cargo build --release -p memra-server --offline` on `1c66ff10e`, local (CPU quota) and target card | `exit=0` both (`rtx5090-day26/build-local.log`, `pro-single-day26/build.log`) |
| `python3 -m py_compile` on the two clients and the parser; `bash -n` on the driver | OK |
| Target-card cells through the collector (`cell-ab`, `cell-ba`, `cell-warm`) | `status executed-not-qualified`, `qualification false`, `exit_code 0` in all three; 45 of 45 requests 200 in each |
| Local cells under `flock /tmp/memra-5090.lock` (`ab`, `ba`, `warm`) | `ab.exit` 0, `ba.exit` 0, `warm.exit` 0; 45 of 45 requests 200 in each |
| Full GPU exactness battery (`kernel-check`, `run-gen`, `run-spec`) | NOT RUN (no kernel or numeric change; nothing to qualify) |
| `cargo fmt --all -- --check`, `git diff --check`, `tools/check-flags.sh`, `tools/check-conflict-markers.sh`, `python3 tools/check-public-boundary.py check` | `fmt: PASS`; `diff --check: PASS`; `check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)`; `check-conflict-markers: OK`; `public-boundary: 582 matches (582 grandfathered, 0 new)` |

## 5. Boundaries and record

- No engine change; no new `MEMRA_*` read; no format substitution; no `unsafe`; no third lock name; no bare GPU run
  (local cells under `flock` on the canonical 5090 lock, target-card cells under the collector's `/tmp/memra-gpu.lock`);
  no `--no-verify`; no touch of `/root/artifacts`, `/root/memra-spill` or other lanes' worktrees; no host, id,
  location or cost in a tracked file; no cross-box timing; every median with N=5 and its regime line.
- Two harness failures kept as labelled failures: the first local BA launch hit a bash syntax error at line 36 after
  taking the lock because the driver file was rewritten (the `WARM_IMMEDIATE` seam) while bash was still reading it
  (`rtx5090-day26/ba.launch.log` of that attempt is overwritten by the rerun; the rerun's `ba/` is the receipt), and
  the first `warm` launch on the target card was refused by the collector with `REFUSED: [Errno 11] Resource
  temporarily unavailable` because another lane held the canonical lock (`pro-single-day26/collector-warm-refused-lockheld.log`);
  the retry was a bounded 900 s poll (`warm-wait.sh`), per the brief.
- Receipts: `rtx5090-day26/{ab,ba}` (server.log stamped, client.jsonl, samples.csv, metrics snapshots, REPORT.txt,
  gpu and compute-apps before and after), `pro-single-day26/` (mirror of the target card's `b-day26`: collector cells
  with `CELL.jsonl`, `command.gpu.csv`, `command.capture.json`; `cells/{ab,ba,warm}` with the same per-cell files;
  `build.log`, `source.txt`, `chain.log`).
- RULE (lead, 2026-09-21) kept: no process that this lane did not start was signalled; the driver stops only a server
  whose cwd is this worktree.
