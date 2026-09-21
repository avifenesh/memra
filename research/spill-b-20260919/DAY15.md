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
