# How long a failed GB10 context creation takes to recover, fixed 50 ms against the ladder (memra#1330)

Date: 2026-11-18. Follows `research/gb10-boot-oom-20261114/` (memra#1316, memra#1326), whose
retry this measures. Unit: `spark-cb33` (GB10), one sparklock lease 11:18 to 13:34.

**Verdict: no code change.** 59 context creations failed their first try in the state where the
failures are (the page cache full, eight readers, a 24 GiB device holder that exits), 31 retried
at a fixed 50 ms cadence and 28 on the shipped ladder (50, 100, 200, 400, then 800 ms), arms
alternating creation by creation. Every one recovered, the longest after 12.0 s. The cadence does
not change the time to the context: median 556 ms with the fixed cadence and 337 ms with the
ladder, p90 2,076 and 2,389 ms, max 12,007 and 6,861 ms (Mann-Whitney p = 0.89; the median
difference is +219 ms, 95 % bootstrap interval -519 to +371 ms). The fixed cadence only adds
calls: 135 against 98 in all, at most 24 against 12 in one creation, and 4.3 against 2.8 calls a
second in the recoveries over 1 s. The time goes to the driver call itself: a failing call took
up to 4.7 s and a call that succeeded up to 11.8 s, so a recovery is made inside a call, while
the kernel reclaims and compacts, not between two of them.

- **`RECLAIM_RETRY`'s 90 s stays.** No recovery came near it (12.0 s at most; the shipped
  policy would have given up on none), it costs nothing on a boot that recovers, and the 50 ms
  memory records of these cells hold a stretch of 45.1 s with no free 2 MiB block, longer than
  memra#1316's 32.2 s. 59 recoveries do not bound the tail beyond what they show: by the rule of
  three, the share of failed creations that take longer than 12 s is under about 5 % (3 / 59).
- **The ladder stays.** It recovers as fast as a fixed 50 ms cadence and makes fewer calls in
  a long stretch.

The issue asked for 100 failing creations per arm. The coordinator of the Spark lanes set an
interim readout at about 30 per arm, to stop there if it answered both decisions; it does, so
the run stopped at 31 and 28. p99 is the maximum at this size.

## The cell

`research/gb10-boot-oom-20261114/tools/recovery_cell.sh`, started and stopped from the
workstation by `tools/cell_1330.sh` (detached in its own session on the unit, its pid in a file,
its end moved with the lease), is `transition_cell.sh` with driver-level creations instead of
server boots:

- The page cache is filled to MemFree under 3,000 MB and eight readers keep running
  (`fill_live.sh`, `cat` over the unit's model files, 1.3 TB).
- A round: the holder (`ctx_probe.py hold 24 2 ARM`) backs 24 GiB of device memory, keeps it
  2 s and exits; the round waits (30 s at most) until MemFree is back under 2,500 MB; six
  creations follow, 1.5 s apart (`ctx_probe.py round`), each `ctx_probe.py recover ARM` in a fresh
  process: `cuInit`, `cuDeviceGet`, then `cuDevicePrimaryCtxRetain` until it succeeds, with no
  cap, MemFree, MemAvailable and the Normal zone's free blocks of order 9 and up read right
  before each try, then `cuCtxSetCurrent`, `cuMemGetInfo` and the release.
- A round whose wait times out makes no creation and ends the episode: readers off 60 s, a
  refill, 20 s with no creation, and the next round. An episode here had one or two rounds
  with creations and then one that timed out (`c2`: 31 rounds with creations in 27 episodes),
  about three minutes in all.
- **Arms.** `fixed50` waits 50 ms before every retry; `ladder` waits as `RECLAIM_RETRY` does
  (`crates/memra-runtime/src/ctx_boot.rs`: 50 ms doubling, 800 ms at most) without its 90 s
  bound, and the record says where that bound would have stopped. Inside a round the arms
  alternate creation by creation, the round's first arm alternates over the rounds that make
  creations, and the holders alternate round by round.
- **Strata.** `first`: the round's first creation, the one the issue names (made when MemFree is
  back under 2,500 MB). `later`: the round's other five. `holder`: the holder's own creation,
  made 1.5 s after the previous round's last creation when the episode goes on, or after the
  pause, the refill and the settling when an episode starts. All three are the same call in the same state, and the tables pool them; each is
  also shown alone.

| cell | what | rounds / episodes | creations (+ holders) | first try out of memory (+ holders) |
|---|---|---|---|---|
| `cb33/smoke1` | the kit end to end, three rounds | 3 / 2 | 6 (+3) | 0 (+0) |
| `cb33/c1` | the campaign, 11:28 to 12:11 | 31 / 14 | 102 (+31) | 12 (+2) |
| `cb33/c2` | the campaign with the first arm's alternation fixed, 12:12 to 13:34 | 58 / 27 | 186 (+58) | 37 (+8) |

**`c1`'s first creations are unbalanced.** In `c1` the first creation's arm alternated by round
number, and since every episode ended on a round that timed out, most rounds with creations had
an even number: 12 first creations went to `fixed50` and 5 to `ladder`. `c2` alternates over the
rounds that make creations. `c1`'s other strata alternate inside the round or round by round and
are balanced. `tables-balanced.txt` leaves `c1`'s first creations out (24 and 26 failed
creations); it reads the same: medians 493 and 316 ms, max 12,007 and 6,861 ms, p = 0.82.

## Read-out

`tools/tab_1330.py cb33/c1 cb33/c2` (`tables-all.txt`; per cell `tables-c1.txt`,
`tables-c2.txt`). Times are from the start of the first try to the end of the try that succeeded.

| arm | creations | first try out of memory | ms to the context: median / p90 / p99 = max | driver calls: median / p90 / max, total | ms of waits, mean | recoveries over 1 / 2 / 5 / 10 s |
|---|---|---|---|---|---|---|
| fixed50 | 188 | 31 (16.5 %) | 556 / 2,076 / 12,007 | 3 / 9 / 24, 135 | 169 | 8 / 4 / 1 / 1 |
| ladder | 189 | 28 (14.8 %) | 337 / 2,389 / 6,861 | 2 / 6 / 12, 98 | 573 | 10 / 5 / 2 / 0 |

By stratum (failed / creations; median and max ms to the context):

| stratum | fixed50 | ladder |
|---|---|---|
| first | 12 / 28; 590, 1,285 | 8 / 20; 620, 1,382 |
| later | 12 / 116; 539, 12,007 | 17 / 124; 653, 6,861 |
| holder | 7 / 44; 335, 4,835 | 3 / 45; 261, 306 |

- **The calls, not the waits.** The ladder waits more (573 ms a recovery on the mean, against
  169) and arrives as soon. A failing call took a median 73 ms at the fixed cadence (104 calls,
  up to 929 ms) and 40 ms on the ladder (70 calls, up to 4.7 s): calls 50 ms apart are slower
  each. The call that succeeded took a median 225 and 187 ms, up to 11.8 s. The 12.0 s recovery is two calls (139 ms
  out of memory, 50 ms of wait, then 11,818 ms in a call that succeeded); a 6.3 s ladder
  recovery holds a failing call of 4.7 s. The longest ladder recovery, 6.9 s, is 12 calls of
  23 to 33 ms with MemFree flat at 730 to 757 MB and no free 2 MiB block, the shape of
  memra#1316's 8.5 s one.
- **What a try sees.** Before the 174 failing tries the free blocks of order 9 and up read 0
  (4 at most), MemFree 1,899 MB median (to 12.3 GB), MemAvailable about 117 GB. Before the 59
  tries that succeeded they read 0 in the median too (8 at most): the free list does not say when a try
  will pass, so a host reading cannot replace the tries.
- **The rate.** The round's first creation failed in 20 of 48 (42 %), against 8 of 57 first
  server boots after an exit in memra#1316. A driver-level creation reaches the call about
  100 ms after its process starts (`init_ms`); a server boot starts later.
- **The kernel log** has one `kgrctxAllocCtxBuffers` line per failed try, 174 tries against
  180 lines. One of the six other lines is 1.2 s from a failed try; the other five are 45 to
  73 s away from any try of these cells, so another process opened a context on the unit then
  (not identified).

## Limits

- **59, not 200.** The issue's size was 100 per arm; the run stopped at the interim readout
  because both decisions were answered. A difference of a few hundred milliseconds between the
  arms at the median, or any difference at p99, is not resolved at this size.
- **One unit, one load.** `spark-cb33` only, the page cache full with `cat` readers, the
  holder at 24 GiB. memra#1316 ran on both units; the rate and the times under another load
  were not measured.
- **Driver-level.** No `memra-server` boot ran here; the call and the state are the ones of
  memra#1316's boots, and the ladder is the shipped policy's waits, not its code.
- **The 90 s is not tested at its edge.** Nothing came close to it, so this record does not
  show what a boot that runs out of budget does; memra#1316's CPU tests cover that.

## Reproduce

```bash
# on the workstation, under `sparklock acquire --unit cb33 ...`
T=research/gb10-boot-oom-20261114/tools
bash $T/cell_1330.sh spark-cb33 start c2 'READERS=8 K=6 GAP_MS=1500 ARMS="fixed50 ladder" HOLD_GIB=24 ARM_MB=2500 ARM_WAIT_S=30 PAUSE_S=60 SETTLE_S=20 UNTIL=<unix, inside the lease>'
bash $T/cell_1330.sh spark-cb33 status c2            # rounds, creations, failures so far
bash $T/cell_1330.sh spark-cb33 until c2 <unix>      # after each lease renewal
bash $T/cell_1330.sh spark-cb33 stop c2              # ends after the round in progress (kill: at once)
rsync -a spark-cb33:.cache/memra-1330/runs/c2 <dir>/ && python3 $T/tab_1330.py <dir>/c2
```

The cells' 50 ms memory record (`memwatch.jsonl`: MemFree, MemAvailable, Cached and the
free-block counts) is 24 MB, so each cell keeps two parts of it: `memwatch-failed.jsonl`, the
samples from 10 s before to 10 s after every failed creation, and `no-order9-stretches.json`,
every stretch of 1 s and over with no free block of order 9 and up, computed from the whole
record.
