# WP-B day 15: the prefix-cache default policy, re-decided on the incident's shape (#523 item 2)

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, merged with `origin/main` `70038ed01` at
`a8c79d896` (day 14's fix, #596, is in `main`). This record is written in two parts: the **pre-registration**
below was committed before any GPU run of the day; the **result** sections follow it and quote the runs
verbatim. One card class today (one RTX PRO 6000 Blackwell, 600 W): a one-rig result sets at most a one-rig
default, so this day produces the target-card verdict and the decision record, not a global default flip.

## Pre-registration (committed before the runs)

### The question

Issue #523 item 2, verbatim: "Re-decide the default policy on the incident's shape (one agent loop at
105k to 168k growing about 300 tokens/turn beside promoted 30k conversations), interleaved A/B both orders
N>=5, and make the winner the naked default per door hygiene."

Arms: A = `MEMRA_PREFIX_CACHE_POLICY=slru` (today's default: byte-segmented LRU, promotion on first reuse,
protected share 80 %, the newest-turn-fits rule from day 14: probation LRU first, then protected LRU oldest
first, leases untouchable) and B = `MEMRA_PREFIX_CACHE_POLICY=lru` (plain global LRU, the rollback arm).
Both arms are set explicitly per boot; the same binary, the same artifact, the same ids and seeds, one lock
hold, one thermal window, one card.

### The replay (`tools/prefix-policy-ab.py`), scaled from the incident's ratios

The incident (2026-09-05): budget 8192 MiB, protected 80 %, other tenants' promoted 30k conversations at
about 6.1 GB (74 % of the budget), the loop's entries 2.7 GB to 4.3 GB (33 % to 52 %), 300 tokens per
turn. On this artifact (`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`) an entry costs 156,716,667 B fixed plus 29,750 B
per token (the day-14 fit from the server's own `insert` lines), so the shape is rebuilt at a 2048 MiB
budget:

- **Cohort**: four tenants (`cache_salt=cohort-1..4`, distinct id ranges, no shared prefix with each other
  or with the loop) with conversations of 7,800, 8,000, 8,200 and 8,400 ids. Each is sent twice; the second
  send is a whole-entry hit whose lease promotes the entry. Cohort bytes: 4 x 156.7 MB + 32,400 x 29,750 B
  = 1,590,766,664 B, 74.1 % of the 2,147,483,648 B budget, inside the 1,717,986,918 B protected share.
- **Loop**: the agent tenant (`cache_salt=grow`) replays 12 turns from 27,300 ids, +300 ids per turn, to
  30,600 ids. Entry 1 = 968,891,666 B (45.1 % of the budget); entry 12 = 1,067,066,666 B (49.7 %). 27,300
  is the largest start for which two consecutive turns still fit the budget together (entry 11 + entry 12
  = 2,125,208,332 B <= 2,147,483,648 B); above it the loop hits the leased-refusal boundary, which is a
  different shape (day 14's fourth CPU arm), not the incident's.
- **Returns**: after loop turns 3, 6, 9 and 12 one cohort tenant continues its conversation (its prompt
  plus 300 new ids), round robin; after the loop every cohort tenant continues once more. 28 requests per
  run: 8 seed, 12 loop, 4 mid-loop returns, 4 final returns.
- **Shape gates** (refuse, exit 2, unless): cohort <= protected share; cohort + entry 1 > budget; every
  loop entry <= budget; entry 11 + entry 12 <= budget. The bytes are read from the first run's own
  `insert` lines, not assumed.
- **Server**: `MEMRA_CTX=32768`, `MEMRA_PREFIX_CACHE_MB=2048`, `MEMRA_SERVE_SPEC=0`, `MEMRA_COMPAT=openai`,
  `MEMRA_TTFT_TRACE=1` (the `[ttft]` receipt gives `first_decode_ms` per request), greedy, `max_tokens=8`,
  `prompt_ids` requests. One boot per run (the policy is a process-wide read); the boot line must report
  `byte-SLRU` for A and `plain-LRU` for B or the cell refuses.
- **Schedule**: the collector's pair vocabulary, `AB-0, BA-0, AB-1, BA-1, ..., AB-4, BA-4`: five pairs per
  order, ten pairs, twenty runs, every pair's two runs adjacent. A cache-off boot (`MEMRA_PREFIX_CACHE_MB=0`)
  replays the same 28 requests first and gives every request its cold digest.

### Recorded per run

`cached_tokens` per loop turn and per cohort return, computed tokens per request (`prompt_tokens -
cached_tokens`) and their sum, loop cold turns after turn 1, the `[prefix-cache]` window (insert, hit,
evict with segment, demote, typed refusal), the `[spec-k]` receipt, `first_decode_ms` p50/p95 over the loop
and over all requests, client E2E p50/p95, settled `/metrics` after every request, the completion digest
per request, GPU temperature/power/clocks at run start and end (the collector's `command.gpu.csv` is the
250 ms record), and the server log.

### The verdict rule

- **Precondition**: every completion digest identical across the twenty runs and the cache-off boot on all
  28 requests. The policy moves bytes' residency, never bytes' values; a mismatch is a FAIL of the day (exit
  1), not a data point.
- **Primary**: total computed tokens across the whole replay (the billable cost), lower is better.
- **Secondary**: the cohort tenants' `cached_tokens` on their returns (scan resistance), stated in the
  record whichever way it points.
- **Win**: a policy wins only if it is better on the primary at every one of the five pairs in both orders
  (ten of ten). A tie at any pair is not "better". Otherwise INCONCLUSIVE.
- **Landing**: a winner becomes the naked default on the target card and the losing arm's door is deleted in
  the same PR (`MEMRA_PREFIX_CACHE_POLICY` env read, dispatch, tests, FLAGS row to the Removed doors
  ledger, TESTING rows), with the statement that this is the target-card verdict and the local RTX 5090 run
  is a follow-up. INCONCLUSIVE leaves the door default-OFF (SLRU stays the default) with a `decide-by:` date
  14 days out and the concrete missing gate named in the FLAGS row.

### Predicted outcome (offline model, `day15-predict.py`, written before the runs)

`day15-predict.py` replays the plan through a model of `worker.rs` (pin promotes at lookup, rebalance
demotes protected LRU over the target, the snapshot preflight and the insert loop select with
`room_victim_with`, unpin at retire refreshes recency; the `lru` arm has a 100 % protected target and the
global oldest victim). On the day-14 shape it reproduces the recorded fix run event for event (9 evictions,
2 of them protected, the same demote order, `cached_tokens` equal to the previous turn on every turn).

On the day-15 shape it predicts: both arms serve the loop identically (0 cold turns after turn 1; every
loop turn hits its predecessor except the turn after a return, which hits the turn before it, because the
returning tenant's publication evicts the loop's newest entry under both policies: the hit entry's recency
is refreshed at retire, after the new entry was published, so the newest published entry is the OLDER of
the two); every mid-loop return is cold under both (the cohort was evicted at loop turns 1 and 2 by both);
the arms differ on the last event only: cohort-4's final return hits its return@12 entry under `lru`
(8,700 tokens cached) and is cold under `slru`, because `slru` protects the loop's last promoted entry
(30,300 tokens, never reused again) over the fresh one-hit cohort entry, while `lru` evicts the global
oldest. Predicted primary: slru 132,300 vs lru 123,600 computed tokens, lru better by 8,700 (6.6 %) at
every pair, deterministically. Predicted secondary: return cached slru 0, lru 8,700. If the live runs match,
the rule names `lru` the winner on this shape; the record will say plainly that the whole difference is one
end-of-replay event of the `slrutarget-20260813` class (a stale protected entry outliving a fresh one), that
the loop itself is served identically by both arms after the day-14 fix, and what the `slrucache-20260813`
hot-set shape (where SLRU won 115/120 vs 107/120) loses if SLRU goes. The live runs are authoritative; the
prediction is recorded so a deviation is visible.

## Result: the target card, verbatim

Binary `5e544691205585bbeaf6c9d1e3596f915f6f328efb803da327357057e761f2da` built natively from the lane tip
`9466b891` (`pro-single-day15/build-tip/`, `dirty.txt` empty, 15.5 s incremental), artifact
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, one RTX PRO 6000 Blackwell Server Edition (97,887 MiB, 600.00 W of
600.00 W, driver 580.178.04), every GPU command through `tools/tier-battery.py --rig pro-single` with the
canonical `/tmp/memra-gpu.lock` (the harness inherited the collector's FD; proof in `cell/LOCK.json`).

### Smoke cell (`ab-smoke`, one pair per order, no verdict by design)

```text
PREFIX-POLICY-AB: budget_bytes=2147483648 cohort_tenants=4 cohort_bytes=1589723136 turns=12 start_tokens=27300 grow=300 return_every=3 pairs_per_order=1 runs=4 requests_per_run=28 digests_identical=28/28 computed_tokens slru_median=132300 lru_median=122700 (N=2 each) pairs_slru_better=0/2 pairs_lru_better=2/2 ties=0/2 return_cached slru_median=0 lru_median=8700 loop_cold_after_1 slru_max=0 lru_max=0 refusals slru=0 lru=0 temp_c=46.0..49.0 power_limit_w=600.0 -> SMOKE
```

Run to catch driver faults before the hour-long cell (day 14's first sitting refused twice on the driver).
It found none; it did find one deviation from the offline model, recorded below.

### The campaign (`ab-full`, five pairs per order, 20 runs, one lock hold, 1,534.5 s)

```text
PREFIX-POLICY-AB: budget_bytes=2147483648 cohort_tenants=4 cohort_bytes=1589723136 turns=12 start_tokens=27300 grow=300 return_every=3 pairs_per_order=5 runs=20 requests_per_run=28 digests_identical=28/28 computed_tokens slru_median=132300 lru_median=122700 (N=10 each) pairs_slru_better=0/10 pairs_lru_better=10/10 ties=0/10 return_cached slru_median=0 lru_median=8700 loop_cold_after_1 slru_max=0 lru_max=0 refusals slru=0 lru=0 temp_c=46.0..50.0 power_limit_w=600.0 -> WINNER=lru
```

Collector: `executed-not-qualified`, exit 0, `"qualification": false`, 250 ms telemetry captured
(`command.gpu.csv`, 6,124 samples, GPU temperature 32 to 62 C over the cell, peak power 509 W under the
600 W cap; at the run boundaries the harness sampled 46 to 50 C). Shape gates from the first run's own
`insert` lines: fit 29,650 B per token plus 157,260,000 B fixed, cohort 1,589,723,136 B (74.0 % of the
budget, inside 80 %), turn-1 entry 966,705,000 B (45.0 %), turn-12 entry 1,064,550,000 B (49.6 %),
cohort + turn 1 > budget, every turn fits, turns 11 and 12 fit together.

Per pair, both orders (computed tokens, lower is better; medians are of N=10 runs per arm; the regime is
the same lock hold, 46 to 50 C at run boundaries, 600 W cap):

| pair | order | slru computed | lru computed | slru - lru | better | slru return cached | lru return cached | slru loop cold | lru loop cold | slru ttft loop p50/p95 ms | lru ttft loop p50/p95 ms | slru temp C | lru temp C |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| AB-0 | AB | 132300 | 122700 | 9600 | lru | 0 | 8700 | 0 | 0 | 158.708/266.009 | 157.908/159.509 | 50.0->47.0 | 47.0->46.0 |
| BA-0 | BA | 132300 | 122700 | 9600 | lru | 0 | 8700 | 0 | 0 | 158.837/266.359 | 157.814/159.509 | 46.0->47.0 | 46.0->46.0 |
| AB-1 | AB | 132300 | 122700 | 9600 | lru | 0 | 8700 | 0 | 0 | 158.754/265.961 | 157.882/159.319 | 47.0->47.0 | 47.0->46.0 |
| BA-1 | BA | 132300 | 122700 | 9600 | lru | 0 | 8700 | 0 | 0 | 158.931/266.187 | 157.932/159.555 | 46.0->47.0 | 46.0->46.0 |
| AB-2 | AB | 132300 | 122700 | 9600 | lru | 0 | 8700 | 0 | 0 | 158.6/265.986 | 157.845/159.261 | 47.0->47.0 | 47.0->46.0 |
| BA-2 | BA | 132300 | 122700 | 9600 | lru | 0 | 8700 | 0 | 0 | 158.791/266.218 | 158.127/160.452 | 46.0->47.0 | 46.0->46.0 |
| AB-3 | AB | 132300 | 122700 | 9600 | lru | 0 | 8700 | 0 | 0 | 158.644/266.041 | 157.918/159.202 | 47.0->47.0 | 47.0->46.0 |
| BA-3 | BA | 132300 | 122700 | 9600 | lru | 0 | 8700 | 0 | 0 | 158.692/266.031 | 157.806/159.276 | 46.0->47.0 | 46.0->46.0 |
| AB-4 | AB | 132300 | 122700 | 9600 | lru | 0 | 8700 | 0 | 0 | 158.653/266.059 | 157.906/159.531 | 47.0->47.0 | 47.0->46.0 |
| BA-4 | BA | 132300 | 122700 | 9600 | lru | 0 | 8700 | 0 | 0 | 158.763/266.115 | 157.871/159.424 | 46.0->47.0 | 46.0->46.0 |

Per arm (N=10 runs each): computed tokens slru median 132,300 (min 132,300, max 132,300), lru median
122,700 (min 122,700, max 122,700); cohort return `cached_tokens` sum slru 0 in every run, lru 8,700 in
every run; loop cold turns after turn 1: 0 in every run of both arms; refusal lines: 0 in both arms;
evictions per run slru 21 (3 from protected, 8 demotions), lru 19 (15 of entries that had been
promoted). Server-side `first_decode_ms` over the 120 loop turns per arm: slru p50 158.6, p95 8,377.9
(turn 1's cold 27,300-token prefill in both arms); lru p50 157.8, p95 8,378.0. Over all 280 requests per
arm: slru p50 266.0 ms, p95 2,489.1 ms; lru p50 158.8 ms, p95 2,442.4 ms (the slru median sits on the
600-token post-return turns, the lru median on the 300-token ones). Client E2E of the 8-token
completions: slru p50 0.392 s, lru p50 0.285 s, p95 2.555 s in both arms (the cold cohort sends). One
card class, one artifact, one budget; no timing is compared with any other box.

**Verdict, by the pre-registered rule:** digest identity held (28/28 requests identical across the 20
runs and the cache-off boot: the policy moved residency, never bytes), `lru` was better on the primary at
every one of the ten pairs in both orders, so **`lru` wins on this shape on the target card**:
`-> WINNER=lru`. Secondary, stated: the cohort tenants' returns were cold under both arms except the last
tenant's final turn, which `lru` served from cache (8,700 tokens) and `slru` did not. The offline replay
of the mirrored receipts (`verify-day15.py`) re-derives the same verdict: `DAY15 REPLAY OK`.

### Where the 9,600 tokens per pair come from (per-request rows, identical in all ten pairs)

| idx | role | tenant | turn | prompt | cold digest[:16] | slru cached | slru computed | slru events | lru cached | lru computed | lru events |
| ---: | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| 8 | loop | grow | 1 | 27300 | 6eb8e535cc2c3bbe | 0 | 27300 | insert 27300, evict 7800 (Protected), evict 8000 (Protected) | 0 | 27300 | insert 27300, evict 7800 (Protected), evict 8000 (Protected) |
| 9 | loop | grow | 2 | 27600 | 24a97a2867768b2d | 27300 | 300 | hit 27300, insert 27600, evict 8200 (Probation), evict 8400 (Protected), demote x1 | 27300 | 300 | hit 27300, insert 27600, evict 8200 (Protected), evict 8400 (Protected) |
| 10 | loop | grow | 3 | 27900 | 65eeb1ef8f716af1 | 27600 | 300 | hit 27600, insert 27900, evict 27300 (Probation), demote x1 | 27600 | 300 | hit 27600, insert 27900, evict 27300 (Protected) |
| 11 | return | cohort-1 | 3 | 8100 | 7632408d9a818cfc | 0 | 8100 | insert 8100, evict 27900 (Probation) | 0 | 8100 | insert 8100, evict 27600 (Protected) |
| 12 | loop | grow | 4 | 28200 | 22f023976ebc22fd | 27600 | 600 | hit 27600, insert 28200, evict 8100 (Probation) | 27900 | 300 | hit 27900, insert 28200, evict 8100 (Probation) |
| 15 | return | cohort-2 | 6 | 8300 | bb95b75c9882e0cf | 0 | 8300 | insert 8300, evict 28800 (Probation) | 0 | 8300 | insert 8300, evict 28500 (Protected) |
| 16 | loop | grow | 7 | 29100 | 0d6d7198ce6e8bc4 | 28500 | 600 | hit 28500, insert 29100, evict 8300 (Probation) | 28800 | 300 | hit 28800, insert 29100, evict 8300 (Probation) |
| 19 | return | cohort-3 | 9 | 8500 | c309d824900a712c | 0 | 8500 | insert 8500, evict 29700 (Probation) | 0 | 8500 | insert 8500, evict 29400 (Protected) |
| 20 | loop | grow | 10 | 30000 | d3cbc5d1ae768103 | 29400 | 600 | hit 29400, insert 30000, evict 8500 (Probation) | 29700 | 300 | hit 29700, insert 30000, evict 8500 (Probation) |
| 23 | return | cohort-4 | 12 | 8700 | 58fcbed59f3d168f | 0 | 8700 | insert 8700, evict 30600 (Probation) | 0 | 8700 | insert 8700, evict 30300 (Protected) |
| 25 | final | cohort-2 | 12 | 8600 | e31d97b3c8ee4728 | 0 | 8600 | insert 8600, evict 8700 (Probation) | 0 | 8600 | insert 8600, evict 30600 (Probation) |
| 27 | final | cohort-4 | 12 | 9000 | 16eff1da2acb0d16 | 0 | 9000 | insert 9000, evict 8600 (Probation) | 8700 | 300 | hit 8700, insert 9000 |

(The full 28-row table, with the seeds and the remaining loop turns where the arms are identical, is
`pro-single-day15/ab-full/cell/REQUESTS.md`.) Three mechanisms, all visible in the server's own lines:

1. **The loop itself is served identically by both arms** after the day-14 fix: turn 1 cold, every later
   turn hits its predecessor, 0 cold turns after turn 1, the cohort evicted oldest first at turns 1 and
   2, no refusal. Item 1 of #523 is closed under either policy.
2. **After a cohort tenant's return, SLRU evicts the loop's newest entry; LRU evicts the older one.**
   The return's publication needs room. Under SLRU the probation LRU goes first and the only probation
   member is the loop's newest entry (the hit entry was promoted at the hit), so the next loop turn hits
   the turn before it and computes 600 tokens instead of 300 (rows 11 to 12, 15 to 16, 19 to 20: 900
   tokens). Under LRU the global oldest goes: the hit entry, whose lease was released after the restore
   fence, BEFORE the new entry was published, so it is the older of the two. This is the one point where
   the offline model was wrong: it had the unpin at retire (after publication), predicted LRU would evict
   the newest entry too, and put the difference at 8,700; the live server put it at 9,600.
3. **After the loop stops, SLRU protects a dead entry over a fresh one.** The loop's last hit entry
   (30,300 tokens) is promoted and never reused again; the last cohort tenant's return (8,700) sits in
   probation. When the finals need room, SLRU evicts the probation entry (row 25: `evict 8700
   (Probation)`) and the tenant's final turn is cold (row 27); LRU evicts the loop's old 30,600 entry
   and the final turn hits (8,700 cached, 300 computed). This is `research/slrutarget-20260813/` in
   miniature: a stale promoted entry outliving a fresh one.

Under the newest-turn-fits rule every unleased byte was already reclaimable, so on this shape the
protected share bought no scan resistance (the loop's entries evict the cohort under both arms); it only
decided which of two entries went first, and both times it chose the one that would be needed next.

## What landed (commit `87d9e5963`)

Per the pre-registered landing rule and door hygiene, the winner is the naked default and the losing arm is
deleted in the same PR:

- `crates/memra-server/src/worker.rs`: plain global LRU is the only policy. Deleted: the
  `MEMRA_PREFIX_CACHE_POLICY` and `MEMRA_PREFIX_CACHE_PROTECTED_PCT` reads (`prefix_cache_slru_enabled`,
  `prefix_cache_protected_pct`, `prefix_cache_protected_bytes`, `DEFAULT_PREFIX_CACHE_PROTECTED_PCT`), the
  `PrefixSegment` enum and `PrefixEntry::segment`, the two segment indexes and three byte counters (one
  `lru` index and `total_bytes` remain), `promote_segment`, `rebalance_protected`, `capacity_victim_with`,
  `room_victim_with` (the victim is `oldest_evictable`: the global oldest unleased entry, never the
  newcomer because it is the newest), `pinned_admission_reclaimable_bytes` (the leased preflight is the
  whole rule), `evict_to_bytes_with`, the `insert probation` and `demote (protected bytes)` lines
  (`[prefix-cache] insert (<why>): ...`, `evict (LRU)`, `evict (snapshot preflight, LRU)` now), and the
  byte-SLRU boot line (`policy plain-LRU (global oldest unleased entry first; leases untouchable)`).
  Unchanged: the newest-turn-fits invariant, the two typed refusal lines and their throttle, the leased
  preflight, the pressure-relief order (`evict_to_bytes` selects with the same `oldest_evictable`),
  `evict_all`, leases, namespaces, the budget. 509 lines deleted, 101 added before tests; five SLRU-only
  tests deleted (`prefix_cache_slru_protects_reused_bytes_from_cross_tenant_scan`,
  `prefix_cache_slru_demotes_by_protected_bytes_not_entry_count`,
  `prefix_cache_slru_fitting_inserts_keep_the_same_victims_in_the_same_order`,
  `prefix_cache_pinned_probation_refuses_before_displacing_protected_share`,
  `evict_to_bytes_takes_protected_oldest_first_once_probation_is_empty`), the rest adapted to the
  single-policy contract (`prefix_cache_newest_turn_fits_beside_a_reused_cohort` keeps the incident's
  211-turn shape: cohort oldest first, never its own victim, no refusal;
  `prefix_cache_evicts_the_global_oldest_including_reused_entries`;
  `prefix_cache_host_promote_pinned_admission_follows_leases`;
  `kv_flex_shed_reaches_the_floor_through_reused_entries_and_warns_only_when_all_is_leased`;
  `prefix_cache_pin_refcount_blocks_eviction_until_last_release` now expects the released entry to be
  the next victim after the entry inserted before its release). `worker/host_glm.rs`: its fixture and
  two `prepare_snapshot` calls follow the new signatures. No new `MEMRA_*` read; captured, restored and
  served bytes untouched (the A/B's 28/28 digest identity is the receipt for the two arms; the landed
  binary's twin gate below records restored == cold on every turn).
- `tools/prefix-newest-turn-fits-gate.py`: the boot line must report `plain-LRU`; V4 is "at least one
  cohort eviction and no turn evicting the entry it just published" (`cohort_evictions=`,
  `self_evictions=` in the verdict line); both insert-line forms parse. `tools/prefix-evict-reclaim-gate.py`:
  both insert-line forms parse. `tools/prefix-policy-ab.py`: deleted with the door (source at `9466b891`;
  the receipts and `verify-day15.py` carry the measurement).
- Docs: `docs/SERVING.md` (the eviction paragraph), `docs/FLAGS.md` (the `MEMRA_PREFIX_CACHE_MB` row, the
  two door rows moved to "Removed doors, 2026-09-21", three rows that described the prefix cache as SLRU
  reworded), `docs/TESTING.md` (the twin-gate section and the day-15 paragraph),
  `docs/decisions/PREFIX-CACHE-POLICY.md` (new; indexed in `docs/decisions/README.md`).
- One card class: this is the target-card verdict and the decision record. The primary metric is a
  function of bytes and policy, not of the card, so the deletion is not a timing default; the owner's
  one-rig rule is stated and the local RTX 5090 replay of the same shape is the follow-up.

## Landed binary on the card (`build-landed`, source `87d9e5963`, SHA-256 `bc668f488eb54978...`)

Built natively at the landing ref (16 s incremental, `dirty.txt` empty). The first sitting's three GPU
cells were REFUSED by the collector before anything ran, verbatim `REFUSED: [Errno 11] Resource
temporarily unavailable`: another lane's collector cell held the canonical `/tmp/memra-gpu.lock`
(the lock doing its job; nothing of that lane was touched). The refused chain is kept under
`pro-single-day15/refused-lockbusy/` (its `landed.log`, the three empty cell dirs and driver logs). The
second sitting (`run-landed2.sh`, bounded 90 s retries, never killing a holder) waited two attempts
(`gate-landed attempt 0: lock busy at 09:29:35`, `attempt 1: lock busy at 09:31:05`) and ran on the third:

| Cell | Result |
| --- | --- |
| `tools/prefix-newest-turn-fits-gate.py` on the landed binary (`gate-landed-retry2`, 1024 MiB budget, cohort 2800/3000/3200, 8 turns 9,200 to 11,300; the day-14 twin, its V4 now "a cohort eviction and never a self-eviction") | `PREFIX-NEWEST-TURN-FITS: budget_bytes=1073741824 cohort_bytes=737943552 turns=8 cold_turns_after_1=0 cached_ok=7/7 lines_ok=8/8 evictions=9 cohort_evictions=3 self_evictions=0 refused_or_skipped=0 effective_free_ok=8/8 V1=ok V2=ok V3=ok V4=ok -> PASS`; boot line `policy plain-LRU (global oldest unleased entry first; leases untouchable)`; V3 state error 0 on all 8 turns; completions identical to the cache-off boot 8/8 and identical to day 14's recorded digests on all 8 turns (`254a65a01730e58b`, `24a97a2867768b2d`, `65eeb1ef8f716af1`, `64a99158cb668b0c`, `c6b9d167a76a942e`, `8e4798b352770d9a`, `f34ee12b3db5cb9c`, `ee53848835a29d90`): the landed default produces the bytes the segmented fix and the pre-fix base produced. Victims, in order: the cohort's 2800, then 3000 and 3200, then the loop's own 9200, 9500, ... 10700 (oldest first, never the newcomer) |
| `tools/serve-smoke.sh <Qwen3.8 artifact> /nonexistent-draft` through the collector (`serve-smoke`; the battery rebuilt `memra-server` from the checkout at `87d9e5963`, `bc668f488eb54978...`, byte-identical to `bins/landed`) | `serve-smoke: 0 failed`; spec, gemma4 and Q35 arms `SKIP` (no draft or model on this box), as on day 14 |
| `tools/cache-meter-gate.py --n 5 --k 256` against the landed binary, native path, `MEMRA_SERVE_SPEC=0` (`cache-meter`) | `cache-meter-gate: 0 failed` |
| `tools/tier-battery.py --validate` on `ab-smoke`, `ab-full`, `gate-landed-retry2`, `serve-smoke`, `cache-meter` | rc 0 on all five; every capture `executed-not-qualified`, `"qualification": false`, telemetry `captured-unvalidated` (ab-smoke 447 s, ab-full 1,535 s, gate 72 s, serve-smoke 22 s, cache-meter 7 s) |

## Checks actually run

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` (landing tree) | PASS after one `cargo fmt` pass over the edited tests (`local-checks/fmt.log`) |
| `cargo test -p memra-server --offline --no-fail-fast` (dev, local, `CPUQuota=1200% MemoryMax=28G`; `local-checks/test-server.log`) | 748 passed, 0 failed, 6 ignored (five tests fewer than day 14: the SLRU-only tests deleted with the arm); the adapted prefix-cache, kv-flex and host_glm tests included |
| `cargo clippy -p memra-server --offline --all-targets -- -D warnings` (dev, local, CPU quota; `local-checks/clippy-server.log`) | PASS |
| `bash tools/check-flags.sh` | PASS (no uncovered runtime `MEMRA_*` name; two names fewer: 865) |
| `bash tools/docs-registry-census.sh` | PASS (58 tables, 901 rows, every row matches its header) |
| `git diff --check` | PASS |
| Native release builds, one RTX PRO 6000 Blackwell (`build-tip/` at `9466b891` for the A/B, `build-landed/` at `87d9e5963`) | exit 0 / exit 0, `dirty.txt` empty |
| `tools/prefix-policy-ab.py --pairs 1` (smoke) and `--pairs 5` (the campaign), target card, collector-locked | `-> SMOKE` (2/2 pairs lru, digests 28/28) and `-> WINNER=lru` (10/10 pairs, digests 28/28); collector exit 0 both |
| `research/spill-b-20260919/verify-day15.py` (offline replay of the mirrored receipts) | `DAY15 REPLAY OK`: verdict re-derived from the per-run rows, schedule checked as the interleaved AB/BA pairing with five pairs per order, digest identity re-checked, binary bound to `build-tip`, 600 W rig line, shape gates, landed twin gate PASS with day-14 digests |
| Landed-binary battery on the card (twin gate, serve-smoke, cache-meter, validate) | PASS / 0 failed / 0 failed / rc 0, table above |
| Full GPU exactness battery (`kernel-check`, `run-gen`, `run-spec`), local 5090 cells | NOT RUN (no kernel or numeric change: victim selection only; the A/B's 28/28 digest identity across both arms and the twin gate's restored == cold identity are the byte receipts) |

Receipt directories under `pro-single-day15/` (mirror of the target card's `b-day15`, binaries and bundles
excluded): `build-tip/`, `build-landed/`, `ab-smoke/`, `ab-full/` (collector journal, `command.gpu.csv` 250 ms
telemetry, `cell/` with `plan.json`, `shape.json`, `cold/`, `runs/<nn>-<pair>-<arm>/{run.json,server.log}`,
`RUNS.md`, `PAIRS.md`, `REQUESTS.md`, `summary.json`, `VERDICT.txt`), `refused-lockbusy/`, `gate-landed-retry2/`,
`serve-smoke/`, `cache-meter/`, the driver scripts (`run-ab.sh`, `build-arm.sh`, `run-landed.sh`,
`run-landed2.sh`, `cache-meter-cell.sh`), driver logs, exit files, `validate-*.log`, and `local-checks/`.

## Boundaries and record

- Every GPU command on the target card went through `tools/tier-battery.py --rig pro-single` with the
  canonical lock (the A/B and the twin gate with `--external-lock`, inherited FD, proof in each cell's
  `LOCK.json`; serve-smoke and cache-meter under the collector's own hold). No third lock name; no bare GPU
  run; no `--no-verify`; no skip variable; no touch of `/root/artifacts`, `/root/memra-spill`, other lanes'
  worktrees or `main`; the other lane's lock hold was waited out, never interrupted. Local builds and tests
  under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`. No local GPU cell.
- Native checkout `/root/wt-b` synced by git bundle (left detached at `87d9e5963`, clean; no server, lock
  free). Receipts mirrored from the target card's `b-day15` to `pro-single-day15/`.
- Pushes: the pre-push hook allowed `a8c79d896` (the merge), `9466b891` (the pre-registration) and
  `87d9e5963` (the landing): perf board, flags census, releasability and docs-registry censuses, public
  boundary all OK.
- The harness's one wrong assumption is recorded, not hidden: the offline model unpinned the hit entry at
  retire; the server releases the restore lease at the restore fence, before publication. The predicted
  8,700 became the measured 9,600, and the verdict rule did not depend on either number.

