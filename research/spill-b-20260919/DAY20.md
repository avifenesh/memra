# WP-B day 20: the day-16 confirmation redone on the local RTX 5090 with both capture sites on the grid (#523 item 2)

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`, on `origin/main` `2a589903d` (days 17 to 19
merged through #606). Day 16 replayed the day-15 policy A/B on the second card class and stopped on its
precondition (`digests_identical=26/28 -> DIGEST-FAIL`); day 17 named the cause (the off-grid restored-suffix
prime, memra#602) and days 18 and 19 fixed it at both capture sites. This day redoes the day-16 cell on a binary
that carries the fix: the pre-registered rule, the day-16 scaled shape (budget 1024 MiB, byte shares preserved,
ctx 16384), `AB-0, BA-0, ..., AB-4, BA-4`, 20 runs, one lock hold, 250 ms telemetry, the cache-off boot for
the cold digests. It changes no code, no default, no gate and no byte. Written in two parts: the
**pre-registration** below was committed before any GPU run of the day; the **result** sections follow it and
quote the runs verbatim.

## Pre-registration (committed before the runs)

### The binary: the pre-registration tree plus the capture fix, off the lane

The harness `tools/prefix-policy-ab.py` and the two-arm binary exist only at `9466b8912`; the capture fix does
not exist there. A DETACHED worktree of `9466b8912` (`git worktree add --detach ~/projects/wt-spill-b-ab
9466b8912`, removed with its target dir when the day closes) took the three capture commits by cherry-pick:

| lane commit | detached commit | binary side | conflicts |
| --- | --- | --- | --- |
| `269178070` (day 18: `seed_capture_boundary`, the plain seed on the grid) | `fe28c5c07` | `lib.rs`, `worker.rs` auto-merged | `tools/kv-host-tenant-reclaim-gate.sh` (absent at `9466b8912`, modify/delete) and `tools/prefix-newest-turn-fits-gate.py` (content): both resolved to the incoming version |
| `49d79b946` (day 19: the spec session's `capture_at` and republish on the grid) | `74ee484bc` | `worker.rs`, `spec.rs` auto-merged | none |
| `fa5d7a014` (day 19: the #379 gate's site list, lane A's equalities, SERVING and TESTING) | `a843686ce` | no crate touched | `docs/SERVING.md`, `docs/TESTING.md`, `tools/kv-host-tenant-reclaim-gate.sh`: resolved to the incoming version |

No hunk was hand-ported: every crate file merged by git. `crates/memra-engine/src/spec.rs` in the detached
tree is byte-identical to the lane tip `ca062f170`; `crates/memra-server/src/worker.rs` differs from the lane
tip by the SLRU arm and its door (deleted at `87d9e5963`, which is not applied here on purpose: the harness
needs both arms), by the `% c == 0` spelling of `66c0aea89`'s `is_multiple_of` (identical arithmetic, not
applied: that commit touches no capture site), and by the main merges the lane took after `9466b8912`. Build:
`build-day20.sh`, receipt `rtx5090-day20/build-tip/` (`source.txt` = `a843686ce`, `source-log.txt` the four
commits, `dirty.txt` empty, `exit` 0, `binary.sha256`), release, own `CARGO_TARGET_DIR`, under
`systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`. Every GPU cell runs through
`tools/tier-battery.py --rig rtx5090` from that worktree with the canonical `/tmp/memra-5090.lock`
inherited by the harness (`run-day20-ab.sh`; bounded retries, 6 x 90 s, never kills a holder). No bare GPU
run. The harness is not modified.

### The shape: the day-16 shape on the 32-token grid, byte shares preserved

The harness asserts, from its own rows, that every cohort tenant publishes an entry of exactly
`prompt_tokens` on its first send and restores exactly `prompt_tokens` on its second (the promotion check),
and it fits this card's bytes-per-token from those `insert` lines. Under the capture law a 1,250-id prompt
now publishes 1,216 tokens (day 19, this card: `insert (seed): 1216 tokens` for the 1,250 cohort) and its
re-send restores 1,216 of 1,250, so the day-16 token counts would make the harness REFUSE (exit 2) on the
first run. Those assertions are gates and are not relaxed. The token counts were never the shape's
invariant: day 16 chose them so that every entry's SHARE of the budget matched the target card's. So every
prompt of this day is a multiple of 32 (the seed then publishes at the prompt end, `SeedCapture::AtPromptEnd`,
and the restored length equals the previous prompt), and the shares are what is preserved. Entry bytes are the
exact day-15 `/metrics` deltas, `156,893,184 B + 29,696 B/token`:

| quantity | day 16 | day 20 | target (day 15) |
| --- | --- | --- | --- |
| cohort ids | 1,250 / 1,350 / 1,450 / 1,550 | **1,248 / 1,344 / 1,440 / 1,536** | 7,800 / 8,000 / 8,200 / 8,400 |
| cohort bytes, share of 1,073,741,824 B | 793,870,336 B, 73.9 % | **792,920,064 B, 73.8 %** | 74.0 % |
| loop | 11,000 + 150 per turn, 12 turns, to 12,650 | **10,912 + 160 per turn, to 12,672** | 27,300 + 300, to 30,600 |
| entry 1 | 483,549,184 B, 45.0 % | **480,935,936 B, 44.8 %** | 45.0 % |
| entry 12 | 532,547,584 B, 49.6 % | **533,200,896 B, 49.7 %** | 49.6 % |
| growth over the loop | 4.6 % | **4.9 %** | 4.6 % |
| entry 11 + entry 12 <= budget, margin | 1.2 % | **1.1 %** | 1.2 % |
| cohort + entry 1 > budget | yes | **yes (1,273,856,000 B)** | yes |
| returns | cohort prompt + 150 after turns 3, 6, 9, 12; every tenant once more after the loop | **+ 160, same structure** | + 300 |
| requests per run, roles, order | 28 | **28, the same roles in the same order** | 28 |
| `MEMRA_CTX`, budget, `MEMRA_SERVE_SPEC`, greedy, `max_tokens`, `prompt_ids` | 16384, 1024 MiB, 0, yes, 8, yes | **unchanged** | 32768, 2048 MiB |

`day20-predict.py` (the day-16 model, which reproduces day 15 and day 16 to the token) predicts the same three
mechanisms on these bytes: **slru 31,776 vs lru 29,600** computed tokens per run, lru better by **2,176** (3 x 160
on the post-return turns plus the 1,696-token final hit) at every pair, deterministically; return cached slru 0,
lru 1,696; loop cold turns after turn 1, 0 in both arms; policy evictions 21 and 19; refusals 0; digests 28/28.

### The day-16 confound, sized from this card's own receipts

Day 16 recorded the admission reclaim ladder (`[admit-oom] reclaim-on-defer`) evicting 12 prefix entries per
run in both arms. The trigger is `worker.rs`'s admission test, `effective free < cost + reserve`, where
`effective free` is driver free plus pool-cached bytes, `cost` is the `[admission] request cost` line (KV at
29,696 B/token, the prefill workspace, the fixed residual: 2,744 to 2,847 MB for the loop prompts) and the plain
path's reserve is `min(cost, SPEC_SHRINK_RESERVE)` = 1,611 MB (`admission_reserve`, `1536 << 20`). Replaying
that test over the day-16 lru run's own settled `/metrics` rows (`effective free` after request k-1 against
`cost` of request k plus 1,611 MB) reproduces its eleven reclaim events exactly (requests 9, 10, 12, 13, 14, 16,
17, 18, 20, 21, 22; none at the cohort, return or final requests). The card's non-cache headroom
(`effective free + prefix_cache_bytes`) read 5,050 to 5,130 MB across the steady loop turns and 4,690 to
4,790 MB right after a cohort return; a loop request needs 4,380 to 4,460 MB; with the cache at 990 to 1,060 MB
the ladder must fire. What holds the missing headroom is the whole-session continuation pool
(`MEMRA_REUSE_POOL`, default 2 per namespace): every request parks its session (the seed2 rows, which publish
nothing, each cost 349 to 359 MB of effective free; a parked 11k-token loop session about 660 MB), two loop
sessions stay parked through the loop and one cohort session after each return, and none of them ever served a
request: every row of days 15, 16 and 19 reads `plain-affinity: declined (no checkpoint retained ...)`, because a
`prompt_ids` turn extends the previous PROMPT, not the parked session's committed sequence (prompt plus its 8
generated tokens), so a token-prefix resume can never match. The pool is inert on this workload and is the VRAM
the ladder takes from the prefix cache.

Sizing the SHAPE alone cannot remove it on this card, and the arithmetic is stated so the lead can check it:
the reclaim needs `cache + parked <= headroom - required`; with two parked loop sessions (about 1,320 MB) that
leaves under 700 MB for the cache at 11k-token prompts, and the shape's own mechanism needs the cache full
(two loop entries plus a return entry must exceed the budget, or the return evicts nothing and the arms tie,
which is why the loop entries are 45 to 50 % of the budget); shrinking the loop below about 9,400 tokens at this
budget removes the mechanism, and shrinking the budget below about 850 MiB makes four 157 MB cohort entries
exceed the 80 % protected share. So the scored cell runs with **`MEMRA_REUSE_POOL=0`** (documented in
`docs/FLAGS.md` as "park nothing, pooling off"), set in the collector's environment and inherited by the
harness's server boot (the harness passes its environment through and pins every other server variable itself).
Predicted headroom with no parked session: about 6,450 MB against a required 4,460 MB with the cache at
1,061 MB, a margin of about 900 MB at every loop turn. The pool touches residency of whole sessions, not a
byte of any prompt, entry or completion; the digest precondition and the row-for-row match with the prediction
are what prove that on this card. Every `[admit-oom]` and `[admit-trim]` line of every boot is counted by
`verify-day20.py`; the scored cell must read zero reclaim events in every run or it is not clean.

### Schedule

1. `ab-smoke-default`: one pair per order (4 runs plus the cache-off boot, no verdict by design), the pool at
   its default, the day-16 configuration on the new binary: the dry run the digest precondition is confirmed on
   (all 28 restored digests equal to the cache-off boot in all four runs) and the census of the reclaim on this
   shape with the harness's own parked sessions.
2. `ab-smoke-pool0`: the same smoke with `MEMRA_REUSE_POOL=0`: zero reclaim events expected, rows identical to
   the default smoke's.
3. `ab-full`: `AB-0, BA-0, ..., AB-4, BA-4`, 20 runs plus the cache-off boot, `MEMRA_REUSE_POOL=0`, one lock
   hold, one thermal window, 250 ms telemetry from the collector.

### The rule, unchanged

The day-15 pre-registered rule, verbatim from DAY15.md: precondition, every completion digest identical across
the twenty runs and the cache-off boot on all 28 requests (a mismatch is a FAIL of the day, exit 1); primary,
total computed tokens across the replay, lower is better; secondary, the cohort tenants' `cached_tokens` on
their returns, stated whichever way it points; win, better on the primary at every one of the ten pairs in both
orders, a tie at any pair is not "better", otherwise INCONCLUSIVE. No timing is compared with the target card
or any other box. `WINNER=lru` confirms the decision on the second card class; anything else is reported to
the lead as a disagreement, plainly, and the decision text is not touched.

## Result 1: the dry run at the default pool (`ab-smoke-default`): the precondition holds, the confound is present

Binary `bd1423b0016e1a544b9029cfe81d4cabbf8e446cad1d6e64f7e369c196c727e4` built from `a843686ce` in the
detached worktree (`build-tip/`, `dirty.txt` empty, exit 0 in 2 min 50 s with the rig's sccache), artifact
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, NVIDIA GeForce RTX 5090 Laptop GPU (24,463 MiB, driver 595.84, `power.limit`
`[N/A]`, `power.max_limit` 175 W), through `tools/tier-battery.py --rig rtx5090` with the canonical
`/tmp/memra-5090.lock` inherited by the harness (proof in `lock.json` and `cell/LOCK.json`, same device and
inode), the lock free on the first attempt, the continuation pool at its default. One pair per order, four runs
plus the cache-off boot, 28 requests each, 386 s wall, 1,538 telemetry samples at 250 ms (54 to 88 C over the
cell, peak draw 182.6 W, peak `memory.used` 23,065 MiB of 24,463; the first and last samples read 15 MiB, so the
card was alone), collector `executed-not-qualified`, exit 0. Verbatim:

```text
PREFIX-POLICY-AB: budget_bytes=1073741824 cohort_tenants=4 cohort_bytes=792920064 turns=12 start_tokens=10912 grow=160 return_every=3 pairs_per_order=1 runs=4 requests_per_run=28 digests_identical=28/28 computed_tokens slru_median=31776 lru_median=29600 (N=2 each) pairs_slru_better=0/2 pairs_lru_better=2/2 ties=0/2 return_cached slru_median=0 lru_median=1696 loop_cold_after_1 slru_max=0 lru_max=0 refusals slru=0 lru=0 temp_c=67.0..74.0 power_limit_w=None -> SMOKE
```

| pair | order | slru computed | lru computed | slru - lru | better | slru return cached | lru return cached | slru loop cold | lru loop cold | slru ttft loop p50/p95 ms | lru ttft loop p50/p95 ms | slru temp C | lru temp C |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| AB-0 | AB | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 186.301/281.798 | 184.494/185.217 | 74.0->69.0 | 69.0->67.0 |
| BA-0 | BA | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 195.675/285.217 | 183.919/185.197 | 68.0->72.0 | 67.0->68.0 |

- **The digest precondition holds: 28/28 in every run, both arms, against the cache-off boot** (day 16 on this
  card: 26/28, one restored-suffix request per arm lineage). The shape gates held from this card's own `insert`
  lines (fit 29,583 B/token + 157,070,000 B from the MB-rounded lines; cohort 792,920,064 B = 73.8 %, turn 1
  44.7 %, turn 12 49.5 %, every gate true). Every published entry equals its prompt (the seed lands at the
  on-grid prompt end), every restored length equals the previous prompt and is a multiple of 32
  (`verify-day20.py`: 62 hit lines). The primary, the secondary and all 28 `cached_tokens` per run equal
  `day20-predict.py`'s table (56/56 rows per arm).
- **The confound is present, exactly as on day 16.** `[admit-oom] reclaim-on-defer` ran 11 times per run in
  every run, evicting 12 prefix entries per run before the loop-turn prefills (the policy's own evict lines fell
  to 9 under slru and 7 under lru against the predicted 21 and 19), plus 5 reclaim events in the cache-off boot
  (parked sessions only, no cache to take from) and 3 `[admit-trim]` lines per run; `plain-affinity: declined`
  on 19 rows per boot. The non-cache headroom read 3,233 to 8,307 MB over a run. The pre-registered model
  (`effective free < cost + 1,611 MB`) predicted every one of those events.

## Result 2: the same smoke with `MEMRA_REUSE_POOL=0` (`ab-smoke-pool0`): the confound is gone, nothing else moved

Same binary, artifact, card, collector and lock (free on the first attempt), the pool off. Four runs plus the
cache-off boot, 386 s wall, 1,537 samples (65 to 89 C, peak draw 194.0 W, peak `memory.used` 20,441 MiB, first
and last samples 15 MiB), collector `executed-not-qualified`, exit 0. Verbatim:

```text
PREFIX-POLICY-AB: budget_bytes=1073741824 cohort_tenants=4 cohort_bytes=792920064 turns=12 start_tokens=10912 grow=160 return_every=3 pairs_per_order=1 runs=4 requests_per_run=28 digests_identical=28/28 computed_tokens slru_median=31776 lru_median=29600 (N=2 each) pairs_slru_better=0/2 pairs_lru_better=2/2 ties=0/2 return_cached slru_median=0 lru_median=1696 loop_cold_after_1 slru_max=0 lru_max=0 refusals slru=0 lru=0 temp_c=67.0..75.0 power_limit_w=None -> SMOKE
```

| pair | order | slru computed | lru computed | slru - lru | better | slru return cached | lru return cached | slru loop cold | lru loop cold | slru ttft loop p50/p95 ms | lru ttft loop p50/p95 ms | slru temp C | lru temp C |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| AB-0 | AB | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 183.931/280.552 | 182.229/183.787 | 75.0->70.0 | 70.0->68.0 |
| BA-0 | BA | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 182.033/280.11 | 181.427/182.597 | 67.0->68.0 | 68.0->67.0 |

Zero `[admit-oom]` lines and zero `[admit-trim]` lines in every boot, the cache-off boot included; no
`parked` line at all. The policy made every eviction itself: 21 per slru run and 19 per lru run, the predicted
counts, with the same victims in the same order as the target card's day-15 rows. Digests 28/28, every
`cached_tokens` row equal to the default smoke's and to the prediction, the same primary and secondary: the
pool moved VRAM residency of whole sessions and nothing else. Non-cache headroom 7,533 to 8,655 MB over a run,
the predicted ~900 MB clear of the admission test at every loop turn. This is the configuration the scored
cell ran in.

## Result 3: the 20-run cell with `MEMRA_REUSE_POOL=0` (`ab-full-retry1`): `WINNER=lru`, the decision holds on both card classes

Same binary, artifact, card and collector. The first attempt at 15:06:53 UTC was REFUSED by the collector
(`REFUSED: [Errno 11] Resource temporarily unavailable`: the canonical lock was still busy three seconds after
the pool-0 smoke's exit file appeared); the driver waited 90 s as designed and attempt 1 held the lock for the
whole cell (`ab-full-retries.log`, the refused stub `ab-full/`, the cell `ab-full-retry1/`). `AB-0, BA-0, ...,
AB-4, BA-4`, 20 runs plus the cache-off boot, 28 requests each, 1,248 s wall, 4,966 telemetry samples at 250 ms
(56 to 88 C over the cell, 66 to 74 C at run boundaries, peak draw 181.3 W, peak `memory.used` 20,441 MiB of
24,463; the first sample read 15 MiB, so the card was alone at the start, and it read 15 MiB again after the
collector exited), collector `executed-not-qualified`, exit 0, `tools/tier-battery.py --validate` on
`CELL.jsonl` and on the capture: `CAPTURE INTEGRITY MATCH; command status=executed-not-qualified; NOT
qualification`. Verbatim:

```text
PREFIX-POLICY-AB: budget_bytes=1073741824 cohort_tenants=4 cohort_bytes=792920064 turns=12 start_tokens=10912 grow=160 return_every=3 pairs_per_order=5 runs=20 requests_per_run=28 digests_identical=28/28 computed_tokens slru_median=31776 lru_median=29600 (N=10 each) pairs_slru_better=0/10 pairs_lru_better=10/10 ties=0/10 return_cached slru_median=0 lru_median=1696 loop_cold_after_1 slru_max=0 lru_max=0 refusals slru=0 lru=0 temp_c=66.0..74.0 power_limit_w=None -> WINNER=lru
```

Per pair (computed tokens, lower is better; every pair adjacent, same binary, same ids; N=10 runs per arm):

| pair | order | slru computed | lru computed | slru - lru | better | slru return cached | lru return cached | slru loop cold | lru loop cold | slru ttft loop p50/p95 ms | lru ttft loop p50/p95 ms | slru temp C | lru temp C |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| AB-0 | AB | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 183.914/282.005 | 182.415/184.167 | 74.0->69.0 | 69.0->67.0 |
| BA-0 | BA | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 182.302/279.243 | 181.518/182.828 | 66.0->67.0 | 67.0->66.0 |
| AB-1 | AB | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 182.454/279.04 | 182.043/195.605 | 67.0->67.0 | 67.0->66.0 |
| BA-1 | BA | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 188.681/280.04 | 182.009/182.592 | 66.0->67.0 | 66.0->66.0 |
| AB-2 | AB | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 182.856/280.382 | 181.439/182.606 | 67.0->67.0 | 67.0->66.0 |
| BA-2 | BA | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 183.335/279.825 | 182.116/183.925 | 66.0->67.0 | 66.0->66.0 |
| AB-3 | AB | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 185.837/279.805 | 181.848/183.975 | 67.0->67.0 | 67.0->66.0 |
| BA-3 | BA | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 182.605/279.607 | 181.708/183.311 | 69.0->67.0 | 66.0->69.0 |
| AB-4 | AB | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 182.301/280.342 | 181.584/182.571 | 67.0->67.0 | 67.0->66.0 |
| BA-4 | BA | 31776 | 29600 | 2176 | lru | 0 | 1696 | 0 | 0 | 182.649/285.708 | 181.95/185.141 | 66.0->67.0 | 66.0->66.0 |

- **Precondition:** every completion digest identical across the 20 runs and the cache-off boot on all 28
  requests (`digests_identical=28/28`; `verify-day20.py` also checks identity across runs of the same arm,
  28/28). The day-16 failure of this precondition on this card is gone on the binary that carries the capture
  fix; the shape's prompts are on the grid, so the fix's `AtPromptEnd` arm is the one exercised here (the
  off-grid `AtBoundary` arm is the twin and restore gates' business, green on both cards on day 19).
- **Primary:** lru better at every one of the ten pairs in both orders, by 2,176 computed tokens per run
  (slru 31,776, lru 29,600, no variance across the ten runs of either arm). **Secondary:** return cached slru 0,
  lru 1,696 (the last tenant's final served from cache under lru, cold under slru). Loop cold turns after
  turn 1: 0 in both arms. Refusals 0. Policy evictions 21 per slru run and 19 per lru run, every one the
  policy's own.
- **Mechanisms:** the same three the target card showed on day 15, read from `REQUESTS.md`: the loop served
  identically by both arms except after a cohort return, where slru evicts the loop's newest (probation) entry
  and the next turn computes 320 instead of 160 (rows 12, 16, 20), and after the loop, where slru protects the
  dead last loop entry over the fresh cohort-4 entry so cohort-4's final is cold (1,856 computed) while lru
  serves it from cache (`hit 1696`, 160 computed). Every one of the 28 `cached_tokens` rows in all 20 runs
  equals `day20-predict.py`'s table (280/280 per arm).
- **Confound:** zero `[admit-oom]` lines and zero `[admit-trim]` lines in all 21 boots; no parked session; the
  admission reclaim ladder did not fire. The cell is clean by the brief's definition.
- **Agreement with the target card:** `WINNER=lru`, 10/10 pairs, 28/28 digests, the same three mechanisms, the
  same per-request residency story row for row (scaled). The decision in `docs/decisions/PREFIX-CACHE-POLICY.md`
  holds on both card classes. No timing is compared with the target card; the TTFT columns are this card's own
  record of its own runs (the slru p95 carries the one extra 160-token prefill per return).

## Checks actually run

| Check | Result |
| --- | --- |
| Native release build, detached worktree at `a843686ce` (= `9466b8912` + the three capture picks), own target dir, CPU quota (`build-tip/`) | exit 0 in 2 min 50 s, `dirty.txt` empty, SHA-256 `bd1423b0...` |
| `ab-smoke-default` (pairs 1, pool default) through the collector, inherited canonical lock | `-> SMOKE`, exit 0, digests 28/28 in 4/4 runs, reclaim 11 events / 12 entries per run (the day-16 confound, present) |
| `ab-smoke-pool0` (pairs 1, `MEMRA_REUSE_POOL=0`) | `-> SMOKE`, exit 0, digests 28/28, 0 reclaim events in 5 boots, rows equal to the default smoke's |
| `ab-full-retry1` (pairs 5, `MEMRA_REUSE_POOL=0`) | `-> WINNER=lru`, exit 0, 10/10 pairs, digests 28/28 across 20 runs, 0 reclaim events in 21 boots |
| `verify-day20.py` on each of the three cells (offline replay) | `DAY20 REPLAY OK` three times; prediction matched 56/56, 56/56 and 280/280 rows per arm |
| `tools/tier-battery.py --validate` on `CELL.jsonl` and `command.capture.json` of each cell | exit 0; `CAPTURE INTEGRITY MATCH; command status=executed-not-qualified; NOT qualification` |
| `cargo fmt --all -- --check`, `bash tools/docs-registry-census.sh`, `git diff --check` (lane tree, CPU quota) | PASS, PASS, clean (`rtx5090-day20/fmt-check.log`; the census output in the commit log line) |
| Full GPU exactness battery | NOT RUN: no code change on the lane (docs and receipts only) |

## Boundaries and record

- Every GPU command on this card went through `tools/tier-battery.py --rig rtx5090` with the canonical
  `/tmp/memra-5090.lock` (inherited FD, proof in `lock.json` and `cell/LOCK.json` of every cell); one attempt
  refused on a busy lock, one 90 s wait, no holder signalled; no third lock name; no bare GPU run; no
  `--no-verify`; no skip variable; no other lane's worktree touched, `main` untouched; nothing of V4.1; no
  external dependency; no captured, restored or served byte changed (the digest precondition is the proof);
  the harness and the SLRU arm were used from the detached worktree only and never returned to the lane; the
  harness was not modified; no gate and no rule relaxed. Build and cells under `systemd-run --user --scope -p
  CPUQuota=1200% -p MemoryMax=28G`.
- The one deviation from day 16's configuration is stated in the pre-registration: `MEMRA_REUSE_POOL=0` on the
  server the harness boots for the scored cell, with the default-pool smoke beside it as the control that the
  pool moved no byte and no row.
- The detached worktree and its target dir were removed when the day closed; the binary's SHA-256 and source
  refs are in `build-tip/`.
- No timing is compared with the target card.
