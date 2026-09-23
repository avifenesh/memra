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
