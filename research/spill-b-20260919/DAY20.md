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
