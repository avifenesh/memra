# WP-B day 31 D4: the door's host-tier demotion (b) and its bounded refusal (c), one boot

Pre-registered before the D4 boot. Section 1 does not change after a number is seen; results go in section 2.
`executed-not-qualified`; no default moves.

## 1. Pre-registration

### 1.1 What is read

DAY31.md 1.1 (b) and (c) name the two door behaviours the day-31 cell cannot show, because no arm there sets
`MEMRA_KV_HOST_MB`. This boot arms the host tier on top of one swept arm, and reads both.

- (b) Reclaim demotion. When an arrival does not fit, the worker flushes unleased prefix-cache entries
  (worker.rs, the reclaim-on-defer block). Door ON, `evict_all_demoting` copies entries to the host tier until the
  copied bytes reach `demote_budget_bytes(decide(required, tiers, 0, budget))`, which is the shortfall when
  `short_by <= min(demotable, host_free)` and 0 otherwise, then drops the rest. It prints
  `[admit-mem] reclaim demoted D of E device prefix entries to the host tier (M MB kept warm; host H MB of B MB resident)`
  only when D > 0. Each copy prints `[prefix-host] demote copy complete off the tick: ... X ms from submission to
  completion ...`, and a second demote inside the same flush first settles the pending one
  (`[prefix-host] demote hashing settled synchronously by ...: ... Y ms since the hand-off ...`).
- (c) Bounded refusal. `decide()` returns `Refuse` iff `need > device_free`, `short_by > min(demotable, host_free)`
  and `waited_ms >= MEMRA_ADMIT_DEFER_BUDGET_MS` (8000 default). That predicate is what "both tiers exhausted"
  means in this code. The refusal is a 429 whose `Retry-After` equals the logged `retry_after_s`.

### 1.2 Boot

Local RTX 5090, the day-31 binary `target/day31/memra-server` (sha256
`daccda3bcd36e4a9eebfd2bb72ec6dae20b05c595ef725da26b9491f49c4efef`), Qwen3.5-9B NVFP4 MTP, `MEMRA_CTX=65536`, one
collector hold on `/tmp/memra-5090.lock` (`tools/tier-battery.py --rig rtx5090 --external-lock`), run only after the
day-31 local chain has finished, with the same idle checks.

| cell | env on top of every day-31 boot's env (DAY31.md 1.2) |
|---|---|
| `D4-host` | `MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=32768 MEMRA_KV_HOST_MB=8192` |

32768 is used because it is the swept value whose charge (about `P + 32840` tokens at 16,704 B/token, about 0.6 GB
per burst request) makes a 32-request burst exceed this card's headroom; it is not a pick. 8192 MiB is used because the
default tenant share (`MEMRA_KV_HOST_TENANT_PCT=50`) then admits 4096 MiB, which holds the whole 2052 MB device prefix
cache. Workload: `day31-client.py` exactly as DAY31.md 1.4, inner order AB, N = 5, burst 32. The control is the
day-31 boot `O1-on32768` on the same card: same binary, same arm, same order, host tier unarmed.

### 1.3 Readings (`d4-read.py`)

- **D4-B** (reading, no pass mark). Every `[admit-mem] reclaim demoted` line: D, E, M, H, B. Per line, the flush
  window: from the stamp of the last line before the contiguous run of `[prefix-host]` lines that ends at the
  reclaim line, to the reclaim line's stamp, in ms; plus the sum of the copy ms and settle ms printed inside that
  window. Totals across the boot. Also the count of `[admit-oom] VRAM defer` lines in the burst window, both boots.
- **D4-C** (judged). PASS iff at least one burst 429 was returned, every burst 429 carries `Retry-After` in 1..=60,
  the number of burst 429s equals the number of `verdict=refuse` lines in the burst window, and every refuse line has
  `waited_ms >= 8000` and `short_by > min(demotable, host_free)` with `host_free > 0` somewhere in the boot's
  `[admit-mem] id=` lines (the tier was armed, not absent). Zero 429s reads "not observed", which is neither PASS
  nor FAIL, and (c) stays owed with that reason.
- **D4-ID** (judged). For every non-burst tag where `D4-host` and `O1-on32768` both returned 200 with equal
  `prompt_sha256`: equal `message_sha256`, G and `finish_reason`. PASS iff every such twin matches; otherwise FAIL
  naming the tags. This checks that arming the host tier does not change a token.
- Burst statuses, `Retry-After` values and `active_sessions_max` for both boots, side by side.

### 1.4 Failures

As DAY31.md 1.8: causes quoted from the captured stderr, a rerun only as the whole boot under a new cell name with
the reason recorded in section 2.

## 2. Results

Written after the boot; section 1 is unchanged. Receipts: `rtx5090-day31-d4/` (`chain.log`, `order.log`,
`LOCK-D4.json`, `collector-D4/`, `boots/D4-host/`), reading `rtx5090-day31-d4/SUMMARY.txt`
(`d4-read.py boots/D4-host ../rtx5090-day31/boots/O1-on32768`), fault list `rtx5090-day31-d4/FAULTS.txt`.

### 2.1 Run

One boot, `D4-host`, 2026-09-23T03:08:58Z to 03:42:51Z, after the day-31 local chain printed `chain done` at
03:08:25Z. One collector hold on `/tmp/memra-5090.lock` (lock proof rc=0); `compute-apps-before.csv` and
`compute-apps-after.csv` list no process. Binary `daccda3b...c4efef` (the day-31 binary); `source.txt` is
`c426a8be2`, a research-only commit on top of the pre-registration. Boot lines:
`[admit-mem] door=ON open_output_tokens=32768 defer_budget_ms=8000` and `[prefix-host] on: budget 8590MB pinned
cacheable host RAM (MEMRA_KV_HOST_MB, startup budget policy), plain byte-LRU, demote on device capacity eviction,
promote on exact-prefix probe; verify=off (MEMRA_KV_HOST_VERIFY); tenant share cap 50% = 4295MB
(MEMRA_KV_HOST_TENANT_PCT)` (server.log line 16). `V-BOOT ... -> PASS`.

### 2.2 Verdict and reading lines, verbatim

```
DAY31 D4-B reclaim#1 t=1790134397255 burst=True demoted=1 of=17 kept_warm_MB=104 host_MB=4234/8590 flush_window_ms=57 prefix_host_lines=1 copy_ms_sum=0.0 settle_ms_sum=0.0
DAY31 D4-B totals reclaim_lines=1 demoted=1 evicted=17 kept_warm_MB=104 flush_window_ms_sum=57
DAY31 D4-B D4-host vram_defer_lines_in_burst=36
DAY31 D4-B control vram_defer_lines_in_burst=40
DAY31 D4-C r429=11 refuse_lines=11 retry_after_in_1_60=True refuse_predicate_all=True host_tier_seen_armed=True sent_retry_after={'60': 11} logged_retry_after_s={'60': 11} -> PASS
DAY31 D4-ID twins=49 equal=49 differ=0 -> PASS
DAY31 D4-BURST D4-host B=32 status={503: 2, 429: 11, 200: 19} retry_after={'5': 2, '60': 11}
DAY31 D4-BURST control B=32 status={503: 2, 429: 11, 200: 19} retry_after={'5': 2, '60': 11}
```

### 2.3 What they say

- (b) observed. One reclaim flush, at the burst's first memory defer: server.log line 8569
  `[admit-mem] reclaim demoted 1 of 17 device prefix entries to the host tier (104MB kept warm; host 4234MB of 8590MB
  resident)`. The flush window is 57 ms. The line inside it is `[prefix-host] demote: 3072 tokens, 105.0MB in 55.6ms
  (host resident 4234.3MB / 8590MB, model q9)` (line 8568). `copy_ms_sum` and `settle_ms_sum` read 0.0 because
  this boot ran the synchronous demote path (`verify=off`), which prints `demote: ... in X ms`, not the off-tick
  `demote copy complete` and `settled synchronously` lines that 1.1 and the reader name. By hand, the tick cost of
  the flush is that one 55.6 ms copy. The demote budget was the shortfall, so one 105 MB entry covered it and the
  other 16 were dropped. Across the whole boot the tier took 50 demotes (5145.5 MB, 2360.0 ms, at most 88.6 ms each)
  and 15 promotes (1643.0 MB, 894.7 ms, at most 96.7 ms); the other 49 demotes fall before the burst, outside any
  reclaim flush. The host tier already held 4234.3 MB when the burst began (the last demote before it, stamp 1790134227760, reports that resident size), and the flush demote's own line reports the same 4234.3 MB after adding 105.0 MB, under the 4295 MB tenant share cap.
- (c) observed. 11 burst 429s, each `Retry-After: 60` equal to the logged `retry_after_s=60`, one `verdict=refuse`
  line per 429. Every refuse line has `waited_ms` 8001 to 8042, `demotable=0` and `host_free=4355682560`, for
  example `short_by=133309460 device_free=2738572004 demotable=0 host_free=4355682560 waited_ms=8002
  retry_after_s=60`. So the host tier was armed and had 4.36 GB free, but nothing was left to demote: the flush had
  already emptied the device prefix cache, and the shortfall is live-session KV, which the host tier does not hold.
  That is what "both tiers exhausted" means in this code (1.1 (c)).
- Identity. 49 non-burst twins equal to the control in `message_sha256`, G and `finish_reason`. Arming the host tier
  changed no token.
- Burst. Statuses and `Retry-After` values are the same as the control's: 19 x 200, 11 x 429 (60 s), 2 x 503 (5 s).
  The two 503s are again prefill OOMs (lines 8651 and 8652,
  `[engine-error] class=Overloaded prefill error: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`).
- The same pre-fix defect shows here as in the control: `iv-b-r0` and `iv-b-r1` returned 500 on the #87 trap at
  pos 32851 and 32852 (lines 6439 and 7464), the allocation being 32854. No panic in this boot.

D4 (b) and D4 (c) are not owed any more. Both were read on the pre-fix program (DAY31.md 2.5). No panic occurred in
this boot, and the flush and the refusals are decided at admission, within about 9 s of the burst's release, long before
any burst request can reach its 32,776-token allocation.
