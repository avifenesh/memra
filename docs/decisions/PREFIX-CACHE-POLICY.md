# Prefix-cache eviction policy: plain LRU, the segmented arm deleted (2026-09-21)

**Status:** decided on the target card (one RTX PRO 6000 Blackwell, 600 W); landed as the naked
default with the `MEMRA_PREFIX_CACHE_POLICY` and `MEMRA_PREFIX_CACHE_PROTECTED_PCT` doors removed
(memra#523 item 2, lane `research/spill-b-20260919/DAY15.md`). The local RTX 5090 run of the same
replay is a follow-up: the primary metric below is a function of bytes and policy, not of the card,
so the verdict is not a timing default, but the owner's rule that one rig's result sets at most one
rig's default is stated here; the 5090 cell confirmed it on day 20 (below).

## Question

The 2026-09-05 incident: one agent loop growing 105k to 168k tokens by about 300 tokens per turn
beside other tenants' promoted 30k conversations ran cold on every turn under the segmented policy
(`slru`, the default since 2026-08-13), while a launcher's `MEMRA_PREFIX_CACHE_POLICY=lru` served it
from cache. Day 14 fixed the defect inside SLRU (the newest turn fits: protected LRU oldest first
once probation is exhausted, memra#523 item 1, PR #596). Item 2 asked which policy should be the
default on that shape, decided by an interleaved A/B in both orders with N >= 5.

## Measured

Harness `tools/prefix-policy-ab.py` (source at `9466b891`; deleted with the door), binary built
from `9466b891` (SHA-256 `5e544691205585bbeaf6c9d1e3596f915f6f328efb803da327357057e761f2da`), artifact
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, budget 2048 MiB, four cohort tenants of 7,800 to 8,400 ids seeded
twice (1,589,723,136 B, 74 % of the budget), one loop tenant 27,300 to 30,600 ids over 12 turns
(entries 45 % to 50 % of the budget; two consecutive turns fit together), a cohort tenant continuing
its conversation after every third loop turn and every cohort tenant once more after the loop; 28
requests per run, greedy, `max_tokens=8`, plain path. Schedule `AB-0, BA-0, ..., AB-4, BA-4` (A =
`slru`, B = `lru`), 20 runs, one lock hold, one thermal window, plus a cache-off boot for the cold
digests. Receipts: `research/spill-b-20260919/pro-single-day15/ab-full/` (collector journal, 250 ms
telemetry, per-run server logs and rows), replayed offline by `verify-day15.py`.

Verdict line, verbatim:

```text
PREFIX-POLICY-AB: budget_bytes=2147483648 cohort_tenants=4 cohort_bytes=1589723136 turns=12 start_tokens=27300 grow=300 return_every=3 pairs_per_order=5 runs=20 requests_per_run=28 digests_identical=28/28 computed_tokens slru_median=132300 lru_median=122700 (N=10 each) pairs_slru_better=0/10 pairs_lru_better=10/10 ties=0/10 return_cached slru_median=0 lru_median=8700 loop_cold_after_1 slru_max=0 lru_max=0 refusals slru=0 lru=0 temp_c=46.0..50.0 power_limit_w=600.0 -> WINNER=lru
```

Per pair (computed tokens, lower is better; every pair adjacent, same binary, same ids):

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

Digest identity held on all 28 requests across the 20 runs and the cache-off boot: the policy moved
residency, never bytes. TTFT on the loop turns (server `first_decode_ms`) was the same in both arms
to within a few ms at p50; the p95 difference is the one extra 300-token prefill per return under
SLRU.

## Why the arms differ, read from the per-request rows

- Both arms serve the loop identically on the turns after the day-14 fix: every turn hits its
  predecessor, no cold turn after turn 1, the cohort evicted oldest first at turns 1 and 2.
- After a cohort tenant's return, SLRU's probation-first order evicts the loop's NEWEST entry (the
  only probation member; the hit entry was promoted), so the next loop turn hits the turn before it
  and recomputes 600 tokens instead of 300. Plain LRU evicts the older hit entry (its lease ends at
  the restore fence, before the new entry publishes, so it is the older of the two) and the next
  turn hits the newest entry. Three returns, 900 tokens.
- After the loop stops, SLRU protects the loop's last promoted entry, which is never reused again,
  over the fresh one-hit cohort entry, so the last cohort tenant's final turn is cold under SLRU
  and a hit under LRU (8,700 tokens). This is the `research/slrutarget-20260813/` losing shape in
  miniature: a stale promoted entry outliving a fresh one.
- Under the newest-turn-fits rule every unleased byte was already reclaimable, so SLRU's protected
  share bought no scan resistance on this shape (entries larger than the free share evict the
  cohort under both arms); it only decided which of two entries went first, and it decided wrong
  both times.

## Decided

Plain global LRU (oldest unleased entry first, leases untouchable) is the only policy. Deleted in
the same PR: the `MEMRA_PREFIX_CACHE_POLICY` and `MEMRA_PREFIX_CACHE_PROTECTED_PCT` reads, the
probation/protected segments, promotion on reuse, demotion at the protected target, the segmented
victim functions (`capacity_victim_with`, `room_victim_with`), the SLRU share arithmetic in pinned
admission, the `insert probation` and `demote (protected bytes)` log lines (the insert line is now
`[prefix-cache] insert (<why>): ...`, the evict lines `evict (LRU)` and `evict (snapshot preflight,
LRU)`), the SLRU-only unit tests, and the A/B harness itself (its arms no longer exist). The
newest-turn-fits invariant, the two typed refusal lines and their throttle, the leased preflight,
the pressure-relief order and the day-14 twin gate stay; the gate now asserts a cohort eviction and
no self-eviction instead of a protected eviction.

## Rejected, and what it costs

- Keeping SLRU as the default: lost every pair on the incident's shape.
- Keeping `lru` as a door with SLRU deleted, or SLRU as a door with LRU the default: a door exists
  to be decided, not kept (door hygiene, 2026-09-05). The winner is naked.
- The trade, stated: `research/slrucache-20260813/` measured SLRU's hot-set reuse at 115/120 against
  LRU's 107/120 with hot thrash 13 -> 5 on a small-entry, one-hit-scan shape. That win is given up.
  On the serving shapes this engine is deployed for (long conversations whose entries are a large
  share of the budget) the day-14 rule had already made SLRU's protection reclaimable, so the
  mechanism that produced that win no longer existed in the default path.
- Re-deciding on timing: the primary metric is computed tokens (the bill and the prefill work),
  which the policy determines deterministically from bytes; TTFT is recorded, not the verdict.

## Follow-ups

- The same replay on the local RTX 5090 (compatibility, not a re-decision: the byte arithmetic does
  not depend on the card): done, day 20, `WINNER=lru` at 10/10 pairs (below).
- Live telemetry of `prefix_cache_hits` / `cached_tokens` on production shapes with small entries,
  where the `slrucache-20260813` result would show if it mattered; a new receipt on a real shape is
  what would bring segmentation back.

## Confirmation on the second card class (2026-09-21, day 20; the day-16 no-verdict smoke superseded)

The decision holds on both card classes. The follow-up cell on the local RTX 5090 Laptop GPU (24,463 MiB) first
stopped on its precondition on day 16 (`digests_identical=26/28 -> DIGEST-FAIL`, one restored-suffix request per
arm lineage; the cause, the off-grid restored-suffix prime, is classified below and was fixed at both capture
sites on days 18 and 19, memra#602). Redone on day 20 on the same harness and two-arm binary from `9466b891` with
the three capture commits cherry-picked on top (`269178070`, `49d79b946`, `fa5d7a014`; every crate file merged by
git), the shape moved onto the 32-token GDN prime grid with the target's byte shares preserved (cohort
1,248/1,344/1,440/1,536 ids, 73.8 % of the budget; loop 10,912 + 160 ids per turn, entries 44.7 to 49.5 %;
budget 1024 MiB, ctx 16384; the harness's own seed assertions require on-grid prompts under the capture law, and
the token counts were always derived from the shares), `AB-0, BA-0, ..., AB-4, BA-4`, 20 runs, one lock hold,
250 ms telemetry, the cache-off boot for the cold digests, verbatim:
`PREFIX-POLICY-AB: budget_bytes=1073741824 cohort_tenants=4 cohort_bytes=792920064 turns=12 start_tokens=10912
grow=160 return_every=3 pairs_per_order=5 runs=20 requests_per_run=28 digests_identical=28/28 computed_tokens
slru_median=31776 lru_median=29600 (N=10 each) pairs_slru_better=0/10 pairs_lru_better=10/10 ties=0/10
return_cached slru_median=0 lru_median=1696 loop_cold_after_1 slru_max=0 lru_max=0 refusals slru=0 lru=0
temp_c=66.0..74.0 power_limit_w=None -> WINNER=lru` (N=10 runs per arm, 56 to 88 C over the cell, no power
limit reported by nvidia-smi on that laptop part). lru was better at every one of the ten pairs in both orders,
by 2,176 computed tokens per run (3 x 160 on the post-return turns plus the 1,696-token final hit): the same
three mechanisms as above, the same per-request residency story row for row; every digest identical across the
20 runs and the cache-off boot; every `cached_tokens` row equal to the offline prediction (280/280 per arm).
The day-16 confound (the admission reclaim ladder evicting 12 prefix entries per run under VRAM pressure) is
absent: zero `[admit-oom]` lines in all 21 boots. The scored cell ran with `MEMRA_REUSE_POOL=0`: the whole-session
continuation pool is inert on a `prompt_ids` replay (every day-16 row read `plain-affinity: declined`, a turn
extends the previous prompt and never the parked session's committed sequence) and its parked sessions were the
VRAM the ladder took from the prefix cache on that 24 GB card; sizing the shape alone cannot remove it there
(the arithmetic is in DAY20.md). A smoke at the default pool on the same binary read 28/28 digests and the same
rows with the ladder firing 11 times per run, so the pool moved residency, not bytes. No timing is compared with
the target card. Record: `research/spill-b-20260919/DAY20.md`, receipts `rtx5090-day20/`, replay
`verify-day20.py`; the day-16 smoke stays in `DAY16.md` and `rtx5090-day16/`.

## The digest precondition, classified (2026-09-21, day 17)

The 5090 follow-up reproduced the day-16 mismatch with the twin gate on the day-16 lru shape (turn 10, 12,350 =
12,200 restored + 150: restored `22f02397`, cold `4415b7e3`, the same digests on a second binary) and named the
two programs from the server's own `[primeseg]` receipts and two same-binary controls: the cold prime's calls
start on the 32-token GDN WY-chunk grid (`grid_off=0`), while the restored render restores at the prompt-end
seed boundary, an arbitrary position, and primes the suffix from there off the grid (`start=12200 take=150
grid_off=8`), which the engine's `align_prime_ranges_to_gdn` law says is not bit-identical. On-grid restores of
the same prompt reproduced the cold bytes and off-grid restores flipped a greedy near-tie at generated token 2
(`RESTORE-POINTS ... identical=3/5`); under the split-invariant sequential scan the whole chain was identical
(`RESTORE-VS-COLD: arm=gdn-sequential ... identical=12/12 -> IDENTICAL`), so the restore itself is exact. The
class is card-independent and which request flips is card-dependent: the 28/28 precondition above held on the
target card because none of its 28 requests was a near-tie that flipped there, not because the restored and cold
programs are identical. It does not move the primary (computed tokens, a bookkeeping quantity the policy decides
identically under either program) and the decision stands; the defect is tracked as its own bug and fix lane.
Record: `research/spill-b-20260919/DAY17.md`, receipts `rtx5090-day17/`, replay `verify-day17.py`.
