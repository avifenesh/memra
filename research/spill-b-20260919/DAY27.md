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
