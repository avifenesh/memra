# WP-B day 27: the prefix-cache budget's context term (fixed), the park policy's receipt (owner's decision), both cards

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, merged with `origin/main` `dc192cd95` (#621, day 25 on
main) at `9d6a0e282`, clean (`tools/check-conflict-markers.sh` OK). Every push today went out in the announced
development mode: the merge range carries main's engine files and the #589 hook refused it `UNQUALIFIED`, so each push
was `MEMRA_RELEASE_QUALIFICATION_MODE=development git push`, the hook printed `UNQUALIFIED DEVELOPMENT ... no GPU
qualification claimed` and logged it to `.git/memra-gate-skips.log`. No qualification is claimed anywhere in this
record. **One engine change today** (section 1.2, `crates/memra-server/src/worker.rs`), no new `MEMRA_*` read, no
default changed. Every cell is `executed-not-qualified`; every verdict line is verbatim; no timing is compared across
the two cards.

Rigs. Local RTX 5090 Laptop GPU (24,463 MiB, `power.limit [N/A]`, lock `/tmp/memra-5090.lock` by `flock`, CPU work
under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`), Qwen3.5-9B-NVFP4-MTP
(`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`). Target card: one RTX PRO 6000 Blackwell (97,887 MiB, 600 W cap, collector
`tools/tier-battery.py --rig pro-single`, lock `/tmp/memra-gpu.lock`), Qwen3.8-27B-NVFP4-Q5K-mtp. Day 26's harness
(`run-day26-cell.sh`, `day26-client.py`, `day26-parse.py`) carries the day; today it also records a sha256 per
completion, the park door in `shape.txt`, and a retention, pool and park summary (`day27-compare.py` diffs two cells).

## 1. The prefix-cache budget derived at `MEMRA_CTX=8192` when `MEMRA_CTX` is unset

### 1.1 The defect, as arithmetic (tree `9d6a0e282`, before the fix)

`init_prefix_cache_budget` (`crates/memra-server/src/worker.rs:4158`, before the fix) sized the derived budget as
`min(2 x entry(ctx), boot_free - 1.5 GiB)` with `entry(ctx) = bpt x ctx + recurrent` (`prefix_entry_geometry_bytes`,
`derived_prefix_cache_budget`), and read `ctx` as `MEMRA_CTX` when set, else a literal
`PREFIX_CACHE_CTX_FALLBACK = 8192`. The cap rule two screens down (`request_ctx_cap` via `resolve_env_ctx`,
`worker.rs:21862`) reads `MEMRA_CTX` when set, else the checkpoint's own `context_length`. So with `MEMRA_CTX` set the
two agreed, and with it unset (the target card's boot, and every naked boot since the 2026-09-10 rule made unset mean
the checkpoint's context) the budget was sized for sessions of 8,192 rows while every open session was capped at
262,144. `docs/FLAGS.md`'s `MEMRA_CTX` row recorded this on purpose ("keeps its own named 8192 sizing floor ... which
publishes and caps nothing"); the code comment feared a 1M declared context. The fleet launchers pin `MEMRA_CTX`
explicitly (262144 on q38, 1048576 on the glm5 pair), so the explicit path already derived from 262k and 1M; only
the unset path substituted a number.

Target card, Qwen3.8-27B, `MEMRA_CTX` unset, from the day-26 boot line
(`[prefix-cache] on: budget 800MB (800325632 B, derived: 2 x 400162816 B max entry for model "q38" at MEMRA_CTX=8192,
requested 800325632 B; boot driver free 86519709696 B, post-reserve clamp 84909096960 B)`):

| Term | Before (literal 8192) | From the served context (262,144) |
| --- | ---: | ---: |
| trunk bytes per token (16 full-attention layers x 1,856) | 29,696 | 29,696 |
| recurrent state (48 GDN layers, fixed) | 156,893,184 | 156,893,184 |
| max entry `bpt x ctx + recurrent` | 8,192 x 29,696 + 156,893,184 = **400,162,816** | 262,144 x 29,696 + 156,893,184 = **7,941,521,408** |
| requested `2 x entry` | 800,325,632 | 15,883,042,816 |
| clamp `boot_free - 1.5 GiB` | 86,519,709,696 - 1,610,612,736 = 84,909,096,960 | 84,909,096,960 |
| budget `min(requested, clamp)` | **800,325,632 B (800 MB)** | **15,883,042,816 B (15.9 GB)** |
| sessions the budget was sized beside | 8.27 GB each (`262,144 x 31,552`) | the same 8.27 GB |

What 800 MB evicted on the day-26 deferred shape: a spec-boundary entry costs 140,486 B/token on this class (the
draft plane and the boundary hidden ride along), so the mix's entries were 202.3 MB (L0, 1,440 tokens), 436.1 MB (L1,
3,104) and 809.2 MB (L2, 5,760). Two L0 entries fit; one L1 plus one L0; one L2 alone (809 MB) fits and evicts
everything else. The tape read `45 [prefix-cache] insert (spec-boundary)` against `43 [prefix-cache] evict (LRU)`,
`prefix_cache_entries=2` at the end, and every one of the 15 continuations `cached=0` on both orders because its entry
had left before it arrived (deferred shape: all 30 first turns, then the 15 continuations). At 15.9 GB the 30 first-turn
entries total `5 x (202.3 + 436.1 + 809.2) + 5 x (236 + 418 + 796)` MB, about 14.5 GB, so all 30 are resident when the
first continuation arrives; the continuations then insert their own entries (about 215, 450 and 830 MB) and LRU takes
the oldest. Arithmetic expectation, registered before the run: L0 and L1 continuations hit (their entries survive the
1.1 GB and 2.3 GB the earlier continuations add), L2 is marginal (the iii-L2 inserts need about 4.2 GB and the older
i-L1 plus ii-L1 entries supply about 4.3 GB). The pre-registered comparison is `cached_tokens` per continuation and the
digest per request against the day-26 cold run.

Local card, Qwen3.5-9B: day 26 pinned `MEMRA_CTX=65536`, so the derivation already read 65,536 there (entry
`65,536 x 14,848 + 52,690,944 = 1,025,769,472`, budget 2,051,538,944 B) and the fix changes nothing on that boot by
construction. Unset on this card the served context is 262,144 and the entry would be
`262,144 x 14,848 + 52,690,944 = 3,945,005,056` B, requested 7,890,010,112 B against a clamp of about 14.4 GB from the
16.0 GB idle free: the budget would be 7.89 GB on a 24 GB card whose open sessions cost 4.38 GB each. That shape did
not fit the mix on day 26 (the two spec pools alone would retain 8.76 GB) and is not run today; the local after cell
is the day-26 shape (`MEMRA_CTX=65536`) as the control that the set arm is byte-identical.

### 1.2 The fix

`crates/memra-server/src/worker.rs`: `prefix_budget_ctx(env_ctx, model_ctx) -> Result<usize, String>` (new, a named
wrapper over `resolve_ctx`, the one resolver behind `request_ctx_cap`), and `init_prefix_cache_budget` now resolves
the served context per loaded model (`MEMRA_CTX` when set, else that model's `cfg.context_length`), sizes the entry
at it, and records `ctx_source` (`"MEMRA_CTX"` or `"checkpoint"`) on `PrefixCacheBudget::Derived` for the boot line,
which now reads `... max entry for model "q38" at served ctx 262144 (from checkpoint; MEMRA_CTX unset), requested ...`.
A model whose context does not resolve (undeclared, unusable `MEMRA_CTX`) contributes no entry and prints
`[prefix-cache] WARNING: served context unresolved for model ...`; its requests refuse for the same reason through the
cap rule, so nothing substitutes a number. **The formula's other two terms did not move**: the two-entry count
(`PREFIX_CACHE_DEFAULT_ENTRIES = 2`) and the boot-free clamp (`boot_free - SPEC_SHRINK_RESERVE`) are the product
decision on how much of the card the prefix cache may hold, and only the context term was wrong. No new flag; the
`MEMRA_PREFIX_CACHE_MB` override is unchanged and still un-clamped. `docs/FLAGS.md` rows `MEMRA_CTX` and
`MEMRA_PREFIX_CACHE_MB` updated.

CPU test `derived_prefix_budget_context_term_is_the_served_context_set_or_unset` (worker.rs tests): `Some("8192")`
reproduces the day-26 line (400,162,816 B entry, 800,325,632 B budget, clamp 84,909,096,960); `None` with
`model_ctx = 262_144` gives 7,941,521,408 B and 15,883,042,816 B under the same clamp; the clamp still binds on a
16.0 GB card; `None` with an undeclared checkpoint, `Some("abc")` and `Some("0")` refuse; and the source text of
`init_prefix_cache_budget` reads `prefix_budget_ctx(env.as_deref(), model_ctx)` with no `PREFIX_CACHE_CTX_FALLBACK`
left in production code. Result: section 4.

### 1.3 The after cell (pre-registered)

Same harness, same mix, same prompts, deferred shape (`WARM_IMMEDIATE` unset), order AB, N=5 per arm per length, the
day-27 binary. Target card `MEMRA_CTX` unset (`pro-single-day27/cell-after-ab`); local `MEMRA_CTX=65536`
(`rtx5090-day27/after-ab`). Compared against the day-26 `ab` cells with `day27-compare.py`: `cached_tokens` per
continuation (day 26: 0 on all 15), the sha256 per completion (must be equal on every request whose `cached_tokens`
is 0 in both, and on the hits the byte-identity of restore is the day-14 and day-19 gate's claim, re-read here as
digest equality), the boot line, `prefix_cache_evictions`, and the retained bytes at idle. Results: section 1.4.

### 1.4 After: target card, one RTX PRO 6000 Blackwell, 27B, `MEMRA_CTX` unset (`pro-single-day27/cell-after-ab`)

Collector cell (`command.capture.json`: `status executed-not-qualified`, `qualification false`, `exit_code 0`), tree
`de2c781e6`, `build.log` `exit=0`, `binary.sha256` recorded; 46 requests (1 warmup + 45), `non-200=0`; rig and
`compute-apps` lines in `cells/after-ab/gpu-*.csv` and `compute-apps-*.csv`. Boot line, verbatim:

```text
[prefix-cache] on: budget 15883MB (15883042816 B, derived: 2 x 7941521408 B max entry for model "q38" at served ctx 262144 (from checkpoint; MEMRA_CTX unset), requested 15883042816 B; boot driver free 86519709696 B, post-reserve clamp 84909096960 B), policy plain-LRU (global oldest unleased entry first; leases untouchable), min prefix 64 tokens, immediate partial restore=off (rollback) (transformer-only; hybrid mid-entry + routed-MoE N/A)
```

The deferred shape now hits on all 15 continuations where day 26 read `cached=0` on all 15, verbatim from
`cells/after-ab/REPORT.txt` (the (i) rows are the day-26 rows to the token):

```text
arm=i L0 N=5 P=[1481, 1481, 1484, 1484, 1483] G=[591, 369, 589, 457, 545] cached=[0, 0, 0, 0, 0] ratios=[126.52, 141.7, 126.46, 135.06, 129.26] min=126.46 median=129.26 max=141.70 alloc_B=8271167488 used_B_median=63987456 booked_MB=[9291, 9777, 9779, 9779, 9778] inherited=1
arm=i L1 N=5 P=[3138, 3138, 3139, 3138, 3136] G=[478, 954, 673, 508, 494] cached=[0, 0, 0, 0, 0] ratios=[72.5, 64.06, 68.77, 71.9, 72.22] min=64.06 median=71.90 max=72.50 alloc_B=8271167488 used_B_median=115038592 booked_MB=[10571, 10571, 10572, 10571, 10570] inherited=1
arm=i L2 N=5 P=[5797, 5803, 5802, 5805, 5803] G=[629, 497, 505, 443, 354] cached=[0, 0, 0, 0, 0] ratios=[40.79, 41.61, 41.56, 41.96, 42.58] min=40.79 median=41.61 max=42.58 alloc_B=8271167488 used_B_median=198777600 booked_MB=[11065, 11066, 11065, 11066, 11066] inherited=0
arm=iii L0 N=5 P=[1572, 1565, 1595, 1563, 1580] G=[384, 239, 227, 484, 561] cached=[1440, 1440, 1440, 1440, 1440] ratios=[134.02, 145.31, 143.88, 128.06, 122.44] min=122.44 median=134.02 max=145.31 alloc_B=8271167488 used_B_median=61715712 booked_MB=[9821, 9818, 9832, 9817, 9825] inherited=0
arm=iii L1 N=5 P=[3236, 3220, 3228, 3234, 3223] G=[564, 621, 407, 319, 211] cached=[3104, 3104, 3104, 3104, 3104] ratios=[68.99, 68.25, 72.12, 73.78, 76.34] min=68.25 median=72.12 max=76.34 alloc_B=8271167488 used_B_median=114691520 booked_MB=[10618, 10611, 10615, 10617, 10612] inherited=0
arm=iii L2 N=5 P=[5887, 5884, 5884, 5868, 5885] G=[345, 487, 576, 354, 539] cached=[5760, 5760, 5760, 5760, 5760] ratios=[42.06, 41.15, 40.58, 42.13, 40.81] min=40.58 median=41.15 max=42.13 alloc_B=8271167488 used_B_median=201017792 booked_MB=[11067, 11067, 11067, 11067, 11067] inherited=1
   45  [prefix-cache] insert (spec-boundary)  first: [prefix-cache] insert (spec-boundary): 1440 tokens, 202.3MB (resident 202.3MB / 15883MB, model q38)
   15  [prefix-cache] hit  first: [prefix-cache] hit: 1440 of 1572 prompt tokens from cache (model q38)
   15  [prefix-cache] spec  first: [prefix-cache] spec restore: 1440 of 1572 prompt tokens + draft plane from cache [suffix queued] (model q38)
prefix-cache hit lines: 15
prefix_cache_entries=45 prefix_cache_bytes=11986255872 prefix_cache_hits=15 prefix_cache_hit_tokens=51520 prefix_cache_inserts=45 prefix_cache_evictions=0 prefix_cache_misses=31
```

Against day 26 (`day27-compare.py`, day-26 `ab` as A, today as B): `tags=45 digest_equal=0 digest_differs=0
missing=0 rows_with_both_digests=0 (a day-26 tape records content_chars only, no sha256) P_G_chars_equal=45`, and
`cached_tokens A: [0 x 15]` against `cached_tokens B: [1440, 1440, 1440, 1440, 1440, 3104, 3104, 3104, 3104, 3104,
5760, 5760, 5760, 5760, 5760]`; against the day-26 `warm` cell (the hit shape) `P_G_chars_equal=45` and the same
`cached_tokens` on all 15. The day-26 client recorded completion lengths, not digests, so the byte comparison with
day 26 is P, G and `content_chars` on 45 of 45 (every (i) and (iii) completion ends in `stop`, every (ii) in
`length`); the sha256 per completion is recorded from today on (`REPORT.txt` "completion digests") and the digest
comparison proper is between today's cells on the same binary (section 2.5). The (ii) arm's digest is the empty
string's (`e3b0c442...`): the 96 bounded tokens are reasoning and `content` is empty, on day 26 as today.

Eviction arithmetic against the pre-registration: 45 entries resident at 11,986,255,872 B (my per-entry estimates for
the (ii) entries were high; the actual total sits under the 15.9 GB budget with no eviction at all), so L2 was not
marginal in practice. Retention at idle moved with it, verbatim: `idle driver free before the first
request=82759516160 last sample=43668602880 retained_by_process=39090913280`; `pool used after
warmup=17914203332 last=47778680904 delta=29864477572; pool reserved last=57680068608 cached last=9901387704`;
`spec_pool_entries=2 continuation_pool_entries=0`. The 39.1 GB is the 12.0 GB of resident prefix entries (the budget's
purpose, yielding under pressure through `alloc_with_single_reclaim_retry` and the step-OOM reclaim), the two parked
spec sessions of the last open requests (2 x 8.27 GB), and the pool's cached free blocks (9.9 GB); day 26 held 30.1 GB
with 0.68 GB of entries and 12.2 GB cached. Executed, not qualified.

## 2. About 30 GB retained on the target card after 45 sequential requests with none active

### 2.1 What the tape says is retained (day 26, re-read with the pool gauges, before any new run)

Day 26 attributed the retention to "four parked whole-session entries at 8.27 GB each (two per pool, MTP-spec and
continuation)". The gauges in `metrics-end.json` of the three target-card cells say otherwise, verbatim:

| Cell | `cuda_driver_free_bytes` idle before first request -> end | `cuda_pool_reserved_bytes` | `cuda_pool_used_bytes` (after warmup 17,914,203,332) | `cuda_pool_cached_bytes` | `spec_pool_entries` | `continuation_pool_entries` | `prefix_cache_bytes` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ab` | 82,759,516,160 -> 52,694,745,088 (30.06 GB retained) | 48,653,926,400 | 36,475,660,360 | 12,178,266,040 | 2 | 0 | 682,354,688 |
| `ba` | 82,759,516,160 -> 52,896,071,680 (29.86 GB) | 48,452,599,808 | 36,488,539,720 | 11,964,060,088 | 2 | 0 | 682,354,688 |
| `warm` | 82,759,516,160 -> 53,063,843,840 (29.70 GB) | 48,284,827,648 | 19,336,305,864 | 28,948,521,784 | 2 | 0 | 671,248,384 |

So: `continuation_pool_entries = 0` in every cell. The mix runs on the spec path (`path=spec` on every request-cost
line), and the retirement code parks a spec session in the spec pool only (`worker.rs:20880-20925`, `if let Some(mut
sess) = s.spec { ... ParkedPool::Spec ... } else if ... { ... ParkedPool::Plain ... }`); the plain pool never sees a
spec session. The 30 GB decomposes as pool reserved minus the post-warmup used (48.5 - 17.9 = 30.6 GB), of which in
`ab`/`ba` about 18.6 GB is still used (two parked spec sessions of the last open requests, 2 x 8.27 GB; the two prefix
entries, 0.68 GB; draft state) and 12 GB is pool-cached free memory the CUDA pool holds back from the driver; in
`warm` the last requests were bounded (`ctx_cap = P + 104`), so the two parked spec sessions are about 183 MB each and
28.9 GB of the 30 GB is pool-cached free memory. Either way the driver sees the same 30 GB gone: the parked sessions
and the pool's cached blocks trade places and the sum does not move. This is the `RECLAIM-DIAG: freed but not
observable` class (`docs/decisions/KV-PHYSICAL-RECLAIM.md`), which the `--kv-allocator vmm` door (decide-by
2026-10-04) and `/admin/trim` address; the park door does not.

### 2.2 The park path, its door and its decide-by

`retire_may_park(aborted, oom_teardown)` gates every park (`worker.rs:29010`); a spec session parks whole at
`sess.cache_max_ctx()` (its `ctx_cap`) into `spec_reuse` when `committed >= 16` and `next_pred.is_some()`; a plain
session parks its `Cache` at `cap = cache.max_ctx` into `reuse` unless `MEMRA_KV_PARK_COMPACT=1`
(`kv_park_compact_on`, `worker.rs:2391`), in which case `compact_parked_plain_cache` (`:2976`) copies rows `[0, fed)`
into a fed-length cache and drops the big one; `park_compact_rows` (`:2400`) declines SWA rings, TP mirrors, host
bounce, `pos != fed`, and already-tight caches. Pools: 2 per (model, namespace) per pool (`MEMRA_REUSE_POOL`), 16
global; no TTL (`parked_at` is recorded, nothing expires it); LRU on a later park or pressure reclaim.
`docs/FLAGS.md:324` (`MEMRA_KV_PARK_COMPACT`): `**0 = OFF by design**`, "SPEC/DSPARK pools are OUT OF SCOPE by
design", the resume byte-identity gate, the step-OOM adjacency replay and the park-time copy-cost receipt "PENDING"
(`research/kv-tenancy-20260831/REPORT.md`). **The row carries no `decide-by:` date**; the door-hygiene rule
(`CLAUDE.md`, 2026-09-05) wants one on every default-OFF door. Recorded here and in the #539 comment for the owner.

### 2.3 What the parked entries buy against what they cost

Buy: continuation hits. Day 26 (both cards, three cells each): `continuation_pool_hits=0`, `spec_pool_hits=0`,
`spec_pool_misses=46` (`ab`, `ba`) / `31` (`warm`), `spec-affinity: declined (history diverged at 48 of checkpoint
...; 2 parked ...)` on all 45 (or 30) continuations and cold turns, 0 hits. Day 20 (`rtx5090-day20`, the
`prompt_ids` replay tapes, plain path): `plain-affinity: declined (no checkpoint retained; 1 parked, 1248 prompt
tokens; model gate)` on 19 of 19 replays in each of `01-AB-0-slru`, `02-AB-0-lru`, `03-BA-0-lru`, `04-BA-0-slru` and 20
of 20 in `cold`; 0 hits (DAY20.md: a `prompt_ids` turn extends the previous PROMPT, not the parked session's committed
sequence). Every `prompt_ids` replay and every chat continuation on these tapes declined the pool.

Cost: bytes the prefix cache and the admission gate cannot use. At the served context a parked spec session is
`ctx_cap x 31,552` = 8,271,167,488 B on the 27B, two per namespace per pool, so up to 16.5 GB of the target card sits
in the spec pool after two open requests, until a later park evicts it; on the 9B at 65,536 the same two entries are
2.19 GB. The pool bytes count as used by the CUDA pool, so the admission gate's live-free reading is lower by that
amount and the prefix cache's `alloc_with_single_reclaim_retry` competes with them. The day-14 observation (67,200
B per prompt token retained with the prefix cache idle) is the plain-path twin of the same pool.

### 2.4 The park cell (pre-registered, before any run)

The day-26 mix (order AB inside the mix, deferred shape), two arms: `default` (`MEMRA_KV_PARK_COMPACT` unset) and
`compact` (`MEMRA_KV_PARK_COMPACT=1`), N=5 per arm per length per run, both arm orders (`default` then `compact`,
`compact` then `default`), one server boot per arm, both cards, the day-27 binary; `MEMRA_CTX` unset on the target
card, `65536` locally. Per arm and order the parser prints: retained bytes at idle (`idle driver free before the first
request` minus the last sample, and pool used/reserved/cached at the end), `continuation_pool_*`, `spec_pool_*`,
`prefix_cache_*`, `park-compact lines`, `affinity lines`, and the sha256 per completion; `day27-compare.py` diffs the
two arms of each order.

Prediction from the code, registered now: the mix is spec-path, spec sessions park in the spec pool, the door acts on
the plain pool only, so **`park-compact lines: 0` in the `compact` arm, retained bytes and pool gauges equal to the
`default` arm within the run-to-run noise of the last parked sessions, `continuation_pool_hits=0` and
`spec_pool_hits=0` in both arms, and digests equal across arms on every request** (the door changes no numeric
program). If that holds, the receipt for the owner is: the door cannot touch what this mix retains; the retention is
the spec pool plus the pool's cached blocks. If time permits, an extra pair on the plain path (`MEMRA_SERVE_SPEC=0`)
gives the door a plain pool to act on; it is labelled as such and is outside the pre-registered comparison.

No default change today; the receipt goes to the door's `docs/FLAGS.md` row and to #539 for the owner's decision on
the park policy.

### 2.5 The park cell: target card, one RTX PRO 6000 Blackwell, 27B, `MEMRA_CTX` unset (`pro-single-day27/cell-park-*`)

Four collector cells, each `status executed-not-qualified`, `qualification false`, `exit_code 0`, tree `de2c781e6`, the
same binary (`binary.sha256`), 46 requests each, `non-200=0`, `compute-apps` empty before and after; regime lines
(before / after): `park-o1-default` `42 C, 36.31 W` / `47 C, 91.07 W`; `park-o1-compact` `42 C, 34.54 W` / `47 C,
57.36 W`; `park-o2-compact` `43 C, 35.32 W` / `47 C, 89.13 W`; `park-o2-default` `42 C, 36.36 W` / `47 C, 89.15 W`;
`power.limit 600.00 W`; elapsed 183.4 to 183.6 s each; `shape.txt` records `park_compact=unset` or `park_compact=1`.
The per-arm ratio rows equal section 1.4's to the token in all four cells (`REPORT.txt`; regenerated locally from the
same inputs with the parser's pool-baseline fix, `python3 day26-parse.py <cell>`). The day-27 summary block reads the
same in all four, verbatim:

```text
idle driver free before the first request=82759516160 last sample=43668602880 retained_by_process=39090913280
pool used after warmup=17914203332 last=47778680904 delta=29864477572; pool reserved last=57680068608 cached last=9901387704
continuation_pool_entries=0
continuation_pool_hits=0
continuation_pool_evictions=0
spec_pool_entries=2
spec_pool_hits=0
spec_pool_evictions=44
spec_pool_misses=31
prefix_cache_entries=45
prefix_cache_bytes=11986255872
prefix_cache_hits=15
prefix_cache_hit_tokens=51520
prefix_cache_inserts=45
prefix_cache_evictions=0
prefix_cache_misses=31
step_oom_parks=0
admission_vram_defers=0
park-compact lines: 0
affinity lines: spec-affinity: declined=30
```

`cached_tokens` on the 15 continuations: `[1440, 1440, 1440, 1440, 1440, 3104, 3104, 3104, 3104, 3104, 5760, 5760, 5760,
5760, 5760]` in every cell. Digests (`day27-compare.py`): order 1 `default` against `compact` `tags=45 digest_equal=45
digest_differs=0 missing=0 rows_with_both_digests=45 ... P_G_chars_equal=45`; order 2 `compact` against `default` the
same line; across boots `after-ab` against `park-o1-default`, `park-o1-default` against `park-o2-default`, and
`park-o1-compact` against `park-o2-compact` each `digest_equal=45 digest_differs=0`. The (ii) digests are the empty
content's on both arms (section 1.4).

Reading, against the pre-registration in 2.4: exactly as predicted from the code. `MEMRA_KV_PARK_COMPACT=1` wrote no
`[kv-reuse] park-compact` line in either order (the door acts at a plain-pool park; `continuation_pool_entries=0`
because every session here is a spec session and parks in the spec pool), retained bytes at idle are the same
39,090,913,280 B in both arms, both pools hit nothing (`continuation_pool_hits=0`, `spec_pool_hits=0`,
`spec-affinity: declined` on all 30 continuations and cold second turns), and every completion is byte-identical
across arms and boots. On this mix the door cannot touch what is retained: the two parked spec sessions of the last
open requests (2 x 8,271,167,488 B at the served context) and the pool's cached blocks (9,901,387,704 B) sit outside
its scope, and the 11,986,255,872 B of prefix entries are the budget's purpose. Executed, not qualified. The receipt for
the owner: the park policy on the spec pool (TTL, a cap at the served context, or compaction in scope) is where the
bytes are; the plain-pool door, as written, is not. A labelled extra pair on the plain path (`MEMRA_SERVE_SPEC=0`,
`pro-single-day27/cell-plain-*`) follows in 2.6 to give the door a plain pool to act on; it is outside the
pre-registered comparison.

### 2.6 Labelled extra pair, plain path (`MEMRA_SERVE_SPEC=0`), target card (`pro-single-day27/cell-plain-*`)

Outside the pre-registered comparison; the same mix, prompts and binary with `MEMRA_SERVE_SPEC=0` so every session
is a plain session and parks in the continuation pool the door acts on. Four collector cells, each `status
executed-not-qualified`, `qualification false`, `exit_code 0`, 46 requests, `non-200=0`, elapsed 275.5 to 276.0 s;
regime (before / after): `plain-o1-default` `33 C, 32.94 W` / `47 C, 89.99 W`; `plain-o1-compact` `42 C, 36.06 W` /
`46 C, 90.33 W`; `plain-o2-compact` `42 C, 35.55 W` / `46 C, 89.05 W`; `plain-o2-default` `42 C, 35.07 W` / `47 C,
90.44 W`. Request-cost lines read `path=plain = 29696 B/token x ctx`; the open arm allocates `262,144 x 29,696 =
7,784,628,224 B`. P and G equal the spec path's on every request (greedy; the (i) rows are the section 1.4 rows with
`alloc_B=7784628224`), and `cached_tokens` on the 15 continuations reads `[1440 x 5, 3104, 3104, 3104, 3104, 0, 5760
x 5]` in all four plain cells: `iii-L1-r4` misses on the plain path in both arms and both orders (`prefix_cache_hits=14`,
47 inserts, 0 evictions), so it is a plain-path capture-boundary fact, not the door's. Verbatim summary blocks, the two
`default` cells identical to the byte and the two `compact` cells identical to the byte:

```text
plain-o1-default / plain-o2-default:
idle driver free before the first request=84942651392 last sample=45751074816 retained_by_process=39191576576
pool used after warmup=16396420244 last=46866169896 delta=30469749652; pool reserved last=55633248256 cached last=8767078360
continuation_pool_entries=2 continuation_pool_hits=0 spec_pool_entries=0 spec_pool_hits=0
prefix_cache_entries=47 prefix_cache_bytes=12105383936 prefix_cache_hits=14 prefix_cache_hit_tokens=48416 prefix_cache_inserts=47
park-compact lines: 0
affinity lines: plain-affinity: declined=45

plain-o1-compact / plain-o2-compact:
idle driver free before the first request=84506443776 last sample=61857202176 retained_by_process=22649241600
pool used after warmup=16395174548 last=30720791848 delta=14325617300; pool reserved last=39527120896 cached last=8806329048
continuation_pool_entries=2 continuation_pool_hits=0 spec_pool_entries=0 spec_pool_hits=0
prefix_cache_entries=47 prefix_cache_bytes=12105383936 prefix_cache_hits=14 prefix_cache_hit_tokens=48416 prefix_cache_inserts=47
park-compact lines: 46
   46  [kv-reuse] park-compact  first: [kv-reuse] park-compact: 164 of 172 rows retained in 2.1ms (model q38)
affinity lines: plain-affinity: declined=45
```

Digests: order 1 `default` against `compact` `tags=45 digest_equal=45 digest_differs=0 missing=0
rows_with_both_digests=45 ... P_G_chars_equal=45`; order 2 `compact` against `default` the same line.

Reading. With a plain pool to act on, the door does what its row says: every retiring plain session was compacted
(46 `[kv-reuse] park-compact:` lines, one per request including the warmup; the open ones read `2071 of 262144 rows
retained` in about 2 ms), and idle retention fell from 39,191,576,576 B to 22,649,241,600 B, a difference of
16,542,334,976 B, which is the two parked 262,144-row plain caches (2 x 7,784,628,224 = 15,569,256,448 B) plus the
pool rounding around them; the pool's cached blocks (8.8 GB) and the 12.1 GB of prefix entries did not move, and the
pool bought nothing in either arm (`continuation_pool_hits=0`, `plain-affinity: declined` on all 45 replays and
continuations). Digests are equal across arms on every request. This is the door's first serving receipt on the target
card class; it is labelled extra, `executed-not-qualified`, N=5 per arm per length, both orders, and it does not decide
the door: the row's own pending gates (the compacted-park resume byte-identity gate on both resume shapes, the
step-OOM adjacency replay) are still pending, and on the served spec path the pool that holds the bytes is out of the
door's scope (2.5).
