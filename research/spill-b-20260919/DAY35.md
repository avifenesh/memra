# WP-B day 35: memra#680's remaining term, the post-prime prefix seed

Day 34's target-card part A (the memra#680 PRO boots on BOX4) read NOT-GREEN on one term: under the day-33 fix, a
64-request burst at open output 32768 still produced three prefill CUDA OOMs. The day-33 prefill-OOM park caught all
three, and no request answered 503 (DAY34.md 2). The lead's call: book that term under the door, pre-registered
before code. This day is authorized to change engine and server code under the default-OFF door only; door OFF stays
byte-identical. Every cell is `executed-not-qualified`; no default moves.

## 1. Pre-registration

Committed and pushed before any day-35 code and before any day-35 boot.

### 1.1 Diagnosis (from the day-34 receipts and code at main `25bbb91f5`)

- The receipt: `pro-single-day34/box/p680/boots/green-R64/server.log`. From the burst's first primes, each finished
  session seeds a prefix entry (`[prefix-cache] insert (seed): 1376 tokens, 197.8MB`, then 1344 tokens at 196.8 MB).
  The resident prefix cache grows from 7086.9 MB to 8861.0 MB in about 1.4 s, and the next three prefills OOM
  (lines 746 to 748, `[admit-mem] prefill OOM parked session back to queue ... CUDA_ERROR_OUT_OF_MEMORY`). The step-OOM
  reclaim then evicts 23 prefix entries (4316 MB), and the parked sessions are admitted again.
- The code. The day-33 booking (`pending_prime_bytes`, `worker.rs` at `25bbb91f5`) counts each still-priming
  session's `prefill_workspace_bytes(queued rows)` and nothing else. The seed entry is sized by
  `prefix_snapshot_bytes(cache)` (20236): KV rows times per-token bytes, plus the recurrent conv and SSM state in f32,
  plus any latent planes. It is allocated when the prime completes (`maybe_prefix_seed`, 20432, then
  `prefix_insert_from_session`, 20281), after the session was admitted. A session is armed to seed when
  `seed_prefix` is true and `seed_at` names its grid boundary; `maybe_prefix_seed` clears `seed_prefix` whether the
  insert lands, is refused or is skipped (20442). The admission gate sees the entry only once it exists, so a burst
  still overcommits by the seeds of every session admitted before its prime completed.
- On the 27B an entry is about 197 MB for about 1360 tokens, about 155 MB of it recurrent state. On the 9B it is
  72.2 MB for 1312 tokens.

### 1.2 The fix under test: option (a), argued

The two options the lead named:

- **(a) Book the seed.** Each still-priming session's future seed-entry bytes are counted in the booked reduction
  until the insert lands.
- **(b) Gate the seed.** The insert is deferred or skipped when booked headroom is short.

I choose (a). Under (b) the prefix cache's contents would depend on the load at the moment each prime completes. A
later request's route (a prefix restore or a cold prime, a known pair under the one-numeric-program rule) would then
depend on burst timing, and so would the numeric program it runs. Under (a) the cache holds exactly the entries it
holds without the door; only admission timing moves, and admission timing already moves under the door. (a) is also
the same mechanism as the day-33 workspace term, one more addend.

Under the door only:

- `pending_seed`: the sum, over active sessions with `seed_prefix`, `seed_at = Some(b)`, no vision input and a live
  cache, of the entry size at `b` rows. The size uses `prefix_snapshot_bytes`'s own arithmetic with the row count set
  to `b`: KV layers at `b` rows, the recurrent state as allocated, latent planes at `b` rows as an upper bound. Each
  session's term disappears when `maybe_prefix_seed` clears `seed_prefix`.
- Every headroom reading of the admission block is reduced by `pending_prime + pending_seed`.
- The admit, defer and refuse lines carry `pending_seed=`, and `device_free` is the reading after both terms.
- Unarmed, both terms are 0 and the reduction is the identity. No `[admit-mem] id=` line prints, and
  `prefix_snapshot_bytes` itself is not changed.
- CPU unit tests, one per arm (1.6).

### 1.3 Shapes, arms, binaries

Workload `day33-client.py` on the day-33 environment (DAY33.md 1.3). Binaries: `red` = main `25bbb91f5` (the day-33
tree as merged, #692), `green` = the lane commit carrying the day-35 fix. Each is built once per card in a detached
worktree.

| shape | door | client args | card |
|---|---|---|---|
| `G2` | ON, open output 8192 | `--chars 5000 --skip-long --burst 64` | 5090 |
| `L64` | ON, open output 32768 | `--chars 5000 --skip-long --burst 64` | 5090 |
| `R64` | ON, open output 32768 | `--chars 5000 --skip-long --burst 64` | BOX4 |
| `off` | removed | `--chars 5000 --skip-long --burst 0` | both |

- **Local RTX 5090** (`/tmp/memra-5090.lock`, one boot per hold, bounded idle waits). Run `red-G2`, then `red-L64`.
  The local red shape is the first of the two whose red boot reads R-OOM RED (any `CUDA_ERROR_OUT_OF_MEMORY` line,
  park receipts included, or any 503). Then run `green-<shape>-r1`, `red-off`, `green-off` and `green-<shape>-r2`.
  If neither shape is red on the day-33 tree, run the same green boots on `G2`, and section 2 records that the 5090
  has no local red for this term.
- **BOX4** (`/tmp/memra-gpu.lock` on the box, only after lane A's `LANE-A-BOX4-DONE`): `red-R64`, `red-off`,
  `green-R64-r1`, `green-off`, `green-R64-r2`.
- **The gate** `tools/admit-mem-burst-gate.sh` already counts every OOM line, parks included. If the local red shape
  is `L64`, its defaults move to `L64` (open output 32768, burst 64). It must fail once on `red` and pass twice on
  `green` at its defaults before the move is committed.

### 1.4 Readings and verdicts

`day33-compare.py` unchanged (sha256 `4493164c202e71ed091be48496518dbbea1f8f4942a916c85fb3087dc9381f10`), with the
DAY33.md 1.6 terms: R-OOM, G-NOOM, G-BOOK, V-ID-FIX, V-ID, V-OFF and the card verdict. G-NOOM counts every
`CUDA_ERROR_OUT_OF_MEMORY` line, park receipts included, as the acceptance requires. Per card, `SUMMARY.txt` and
`FAULTS.txt` (`day31-faults.py`).

Acceptance, per card: R-OOM RED on the red boot of the card's shape (for the 5090, only if a local red exists);
G-NOOM, G-BOOK, V-ID-FIX, V-ID and V-OFF PASS on both green runs; card verdict GREEN.

### 1.5 Expected readings (not rules)

- BOX4: `red-R64` shows three or more OOM lines (the three parks of day 34 part A's `green-R64`, or more). Both
  green runs show zero, with the booked seed turning those three admissions into defers or typed 429s.
- 5090: on the day-33 tree G2 read zero OOM lines four times (DAY33.md 2.3), so `red-G2` is expected NOT-RED.
  `red-L64` is uncertain: the 9B's entries are 72 MB and its card is 32 GB.

### 1.6 Unit tests (CPU)

- The seed rows owed: `Some(b)` while armed with a live cache and no vision; 0 once `seed_prefix` is cleared, when
  `seed_at` is `None`, or with vision.
- The entry size at `b` rows from layer descriptors: KV rows times per-token bytes, the recurrent state counted once.
- The booked reading with both terms: admits when the need fits, defers when the seed term alone makes it short (the
  day-34 shape), and is the identity at zero.
- The receipt renders `pending_seed=`.
- The existing day-33 tests stay green.

### 1.7 After day 35

The clean door rerun on the day-35 tree, both cards, is a fresh pre-registration: day 34's with its binary updated.
That is the owner's decision cell.

## 2. Results

Written after the runs. Section 1 is unchanged.
Every number below is read from the committed receipts named next to it.

### 2.1 Commits and timeline (UTC, 2026-09-23)

| commit | time | what |
|---|---|---|
| `164777767` | 20:37:58 | section 1, before any day-35 code or boot |
| `809c16444` | 20:48:48 | `fix(server)`: the seed term, the CPU tests, the FLAGS row |
| `1de7ee2da` | 21:01:28 | the local runner and chain, the BOX4 chain, the build records |
| (first boot) | 21:16:03 | local `red-G2` start (`rtx5090-day35/run.log`; the chain waited from 20:59:03 behind lane A's 5090 hold) |
| `5a5b6319a` | 22:17:15 | local receipts, `SUMMARY.txt`, `FAULTS.txt` |
| this commit | after 22:37:48 | BOX4 receipts and reading, this section, `STATE.md`, the INDEX row |

Local 21:16:03 to 22:09:54. BOX4 22:16:37 to 22:37:48, after lane A's `LANE-A-BOX4-DONE` (22:10:54). Binaries: local
red `4375ce6e...` and green `0a940566...`; BOX4 red `3063862a...` and green `573482b6...`.

### 2.2 Notes (no rule, arm, value or reader changed)

- **Resync after a session block.** The session stopped on a server-side API error during `green-G2-r2`. On resume
  (22:06:25) the chain, `green-G2-r2`'s cell and the BOX4 watcher were still running under their own pids, and
  `green-G2-r1`, `red-off` and `green-off` had finished with receipts. Nothing was restarted; the resync line is in
  `rtx5090-day35/run.log`.
- **BOX4 relaunch.** The watcher's first launch (22:11:05) failed before any build or boot:
  `bash: research/spill-b-20260919/pro-single-day35/chain.sh: No such file or directory`. The watcher had not
  fast-forwarded `/root/wt-b` from `7a0c73aa3`. I fast-forwarded it to `1de7ee2da` and launched the unchanged chain
  at 22:16:37 (`pro-single-day35/start.log`).
- **No local red.** Both local red shapes read NOT-RED on the day-33 tree (`red-G2` 36 x 200 and 28 x 429; `red-L64`
  19 x 200 and 45 x 429; no OOM line). As 1.3 pre-registered, the green boots ran on G2, and the seed term has no
  red on the 5090. The gate script was not moved.
- BOX4 is DAY34.md 2.2's Workstation card; red and green pair within BOX4 only.

### 2.3 Verdict lines, verbatim

`rtx5090-day35/SUMMARY.txt` (V-BOOT PASS on all 6):

```
== R-OOM / G-NOOM / G-BOOK (ON boots)
DAY33 R-OOM card=rtx5090 boot=red-G2 role=red shape=G2 v=8192 oom_lines=0 burst={429: 28, 200: 36} first_oom=none -> NOT-RED
DAY33 G-NOOM card=rtx5090 boot=red-G2 role=red shape=G2 oom_lines=0 burst_503=0 crash_lines=0 burst_200=36 other_non200=0 r429=28 refuse_lines=28 retry_after_in_1_60=True -> PASS
DAY33 G-BOOK card=rtx5090 boot=red-G2 role=red shape=G2 admit_lines=52 admit_lines_in_burst=36 est_over_booked_free=0 -> PASS
DAY33 R-OOM card=rtx5090 boot=red-L64 role=red shape=L64 v=32768 oom_lines=0 burst={429: 45, 200: 19} first_oom=none -> NOT-RED
DAY33 G-NOOM card=rtx5090 boot=red-L64 role=red shape=L64 oom_lines=0 burst_503=0 crash_lines=0 burst_200=19 other_non200=0 r429=45 refuse_lines=45 retry_after_in_1_60=True -> PASS
DAY33 G-BOOK card=rtx5090 boot=red-L64 role=red shape=L64 admit_lines=35 admit_lines_in_burst=19 est_over_booked_free=0 -> PASS
DAY33 R-OOM card=rtx5090 boot=green-G2-r1 role=green shape=G2 v=8192 oom_lines=0 burst={429: 28, 200: 36} first_oom=none -> NOT-RED
DAY33 G-NOOM card=rtx5090 boot=green-G2-r1 role=green shape=G2 oom_lines=0 burst_503=0 crash_lines=0 burst_200=36 other_non200=0 r429=28 refuse_lines=28 retry_after_in_1_60=True -> PASS
DAY33 G-BOOK card=rtx5090 boot=green-G2-r1 role=green shape=G2 admit_lines=52 admit_lines_in_burst=36 est_over_booked_free=0 -> PASS
DAY33 R-OOM card=rtx5090 boot=green-G2-r2 role=green shape=G2 v=8192 oom_lines=0 burst={429: 28, 200: 36} first_oom=none -> NOT-RED
DAY33 G-NOOM card=rtx5090 boot=green-G2-r2 role=green shape=G2 oom_lines=0 burst_503=0 crash_lines=0 burst_200=36 other_non200=0 r429=28 refuse_lines=28 retry_after_in_1_60=True -> PASS
DAY33 G-BOOK card=rtx5090 boot=green-G2-r2 role=green shape=G2 admit_lines=52 admit_lines_in_burst=36 est_over_booked_free=0 -> PASS
== V-ID-FIX (green ON against red ON, same shape, sequential rows)
DAY33 V-ID-FIX card=rtx5090 shape=G2 green=green-G2-r1 red=red-G2 eligible=16 equal=16 differ=0 -> PASS
DAY33 V-ID-FIX card=rtx5090 shape=G2 green=green-G2-r2 red=red-G2 eligible=16 equal=16 differ=0 -> PASS
== V-ID (the day-32 term: green ON against green OFF)
DAY33 V-ID card=rtx5090 shape=G2 on=green-G2-r1 off=green-off eligible=16 equal=16 differ=0 -> PASS
DAY33 V-ID card=rtx5090 shape=G2 on=green-G2-r2 off=green-off eligible=16 equal=16 differ=0 -> PASS
== V-OFF (green OFF against red OFF: the default-OFF program unchanged)
DAY33 V-OFF card=rtx5090 green=green-off red=red-off rows=16 equal=16 differ=0 admit_mem_lines=0 -> PASS
DAY33 VERDICT card=rtx5090 boots=6 v_boot_all=True green_noom_book_all=True v_id_fix_all=True v_id_all=True v_off_all=True -> GREEN
```

`pro-single-day35/box/SUMMARY.txt` (V-BOOT PASS on all 5):

```
== R-OOM / G-NOOM / G-BOOK (ON boots)
DAY33 R-OOM card=pro6000 boot=red-R64 role=red shape=R64 v=32768 oom_lines=3 burst={429: 15, 200: 49} first_oom=server.log:743 1790202433730 [admit-mem] prefill OOM parked session back to queue (model q38, retry 1/3): DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory") -> RED
DAY33 G-NOOM card=pro6000 boot=red-R64 role=red shape=R64 oom_lines=3 burst_503=0 crash_lines=0 burst_200=49 other_non200=0 r429=15 refuse_lines=15 retry_after_in_1_60=True -> FAIL
DAY33 G-BOOK card=pro6000 boot=red-R64 role=red shape=R64 admit_lines=68 admit_lines_in_burst=52 est_over_booked_free=0 -> PASS
DAY33 R-OOM card=pro6000 boot=green-R64-r1 role=green shape=R64 v=32768 oom_lines=0 burst={429: 20, 200: 44} first_oom=none -> NOT-RED
DAY33 G-NOOM card=pro6000 boot=green-R64-r1 role=green shape=R64 oom_lines=0 burst_503=0 crash_lines=0 burst_200=44 other_non200=0 r429=20 refuse_lines=20 retry_after_in_1_60=True -> PASS
DAY33 G-BOOK card=pro6000 boot=green-R64-r1 role=green shape=R64 admit_lines=60 admit_lines_in_burst=44 est_over_booked_free=0 -> PASS
DAY33 R-OOM card=pro6000 boot=green-R64-r2 role=green shape=R64 v=32768 oom_lines=0 burst={429: 20, 200: 44} first_oom=none -> NOT-RED
DAY33 G-NOOM card=pro6000 boot=green-R64-r2 role=green shape=R64 oom_lines=0 burst_503=0 crash_lines=0 burst_200=44 other_non200=0 r429=20 refuse_lines=20 retry_after_in_1_60=True -> PASS
DAY33 G-BOOK card=pro6000 boot=green-R64-r2 role=green shape=R64 admit_lines=60 admit_lines_in_burst=44 est_over_booked_free=0 -> PASS
== V-ID-FIX (green ON against red ON, same shape, sequential rows)
DAY33 V-ID-FIX card=pro6000 shape=R64 green=green-R64-r1 red=red-R64 eligible=16 equal=16 differ=0 -> PASS
DAY33 V-ID-FIX card=pro6000 shape=R64 green=green-R64-r2 red=red-R64 eligible=16 equal=16 differ=0 -> PASS
== V-ID (the day-32 term: green ON against green OFF)
DAY33 V-ID card=pro6000 shape=R64 on=green-R64-r1 off=green-off eligible=16 equal=16 differ=0 -> PASS
DAY33 V-ID card=pro6000 shape=R64 on=green-R64-r2 off=green-off eligible=16 equal=16 differ=0 -> PASS
== V-OFF (green OFF against red OFF: the default-OFF program unchanged)
DAY33 V-OFF card=pro6000 green=green-off red=red-off rows=16 equal=16 differ=0 admit_mem_lines=0 -> PASS
DAY33 VERDICT card=pro6000 boots=5 v_boot_all=True green_noom_book_all=True v_id_fix_all=True v_id_all=True v_off_all=True -> GREEN
```

### 2.4 What the rows say

| card, shape | red (`25bbb91f5`) burst | green (`809c16444`) burst |
|---|---|---|
| BOX4, R64 (32768, B = 64) | 49 x 200, 15 x 429, 3 parked prefill OOMs | 44 x 200, 20 x 429, no OOM line; the same on both runs |
| 5090, G2 (8192, B = 64) | 36 x 200, 28 x 429, no OOM | 36 x 200, 28 x 429, no OOM, both runs |
| 5090, L64 (32768, B = 64) | 19 x 200, 45 x 429, no OOM | not run (not the shape) |

- On BOX4 the booked seed turns the three admissions that used to OOM into typed 429s. `pending_seed` reached 8.49 GB
  in `green-R64-r1` (1.75 GB on the 5090 at G2). The prefix cache's contents are unchanged: the insert is booked,
  never gated.
- The red boot here is day 34 part A's green binary on the merged main (`25bbb91f5`). It reproduces that boot's three
  parks and its 49 x 200 and 15 x 429 exactly.
- The fix moved no token: V-ID-FIX 16/16 against red and V-ID 16/16 against OFF on every green run on both cards, and
  V-OFF 16/16 with no `[admit-mem] id=` line on the OFF boots.
- CPU (`cargo test -p memra-server --tests`): 891 passed, 0 failed, and 7 `argv_boot`; clippy `-D warnings` and fmt
  clean. New tests: `admit_memory::tests::booked_seed_defers_what_the_workspace_booking_alone_admits`,
  `admit_memory::tests::admit_arm_renders_the_seed_term`,
  `worker::tests::pending_seed_rows_are_the_armed_boundary_only`,
  `worker::tests::seed_entry_bytes_is_rows_times_kv_plus_state_once`,
  `worker::tests::pending_seed_is_armed_only_and_joins_the_booked_reduction`.

### 2.5 Owed

| item | why | price |
|---|---|---|
| The owner's decision cell (DAY36.md) | started 22:39 on both cards; its start condition (this section's BOX4 GREEN) holds | running |
| A local red for the seed term | the 9B's 72 MB entries never OOM the 5090 at these shapes; the local gate keeps G2 | none unless a shape is found |

### 2.6 Cleanup

The local detached worktrees `wt-b35-red` and `wt-b35-green` are removed with this commit. `target/day35/green` stays
until day 36 has used its copy; `target/day35/red` is deleted. The BOX4 chain removed its worktrees. Its receipt
root, and `bins/green` (the day-36 binary's source), stay.
