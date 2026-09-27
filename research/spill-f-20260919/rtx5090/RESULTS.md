# 5090 half results (OWED 19, 23, 17, 18; M1-PREREG.md sections D, E, F)

Rig: the local laptop RTX 5090 (24 GB, power.limit `[N/A]`, max 175 W), storage `/data` proven
`nvme-local-direct` (`proof/`, re-proven after the 07:28Z reboot as mount id 250;
`proof-prereboot-mount242/` covers capped rounds 1 to 4). Binaries: `build/` (engine source equal
to BOX27's) for B3; `build-g2/` for the G2 probe; `../owed17/build/` and `../owed18/build/` for
the two doors. Every visit runs inside a 20 GiB swapless `systemd-run` scope, `CPUQuota=1200%`.

## B3 capped regime (ten rounds, 60 visits): unscored under the registered gate

Registered verdict (`capped/pool.log`, `capped/pooled-summary.json`):

```
M1-5090-VERDICT regime_scored=False contaminated={'worker16': 6, 'mmap-random': 8, 'mmap-normal': 6, 'pread16': 6, 'worker2': 7}
rounds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] visits=60 refused=['direct16'] gpu_cotenant_unclean=0
```

The post-hoc B1 read gate also leaves the regime unscored (4 or 5 contaminated visits per arm).
No challenger gets a verdict on this rig from this regime.

Why: on this rig `/data` and the worktrees (`/home/avifenesh/projects`) are one LVM volume, so
other lanes' builds, git traffic and the desktop's caches reach the proven device during visits.
Foreign shares ran up to 48% of a visit's device bytes (limit 2%). The attribution sampler
(`../m1-io-attribution.py`, diagnostic only, started after capped) shows, in its first sample, a
foreign `rustc` writing about 29 MB per 5 s during a run-gen visit. The GPU co-tenant gate saw no
other compute app in any visit.

direct16 is refused on correctness: rounds 1 and 2 fell back to mmap 4 and 2 times, quoted
`[spill-pread] falling back to mmap: worker read ring is busy` (the registered direct gate allows
zero fallbacks). The same mechanism hits `worker2` in every visit here (431 to 746 fallbacks) and
`worker16` in two visits (5 and 6); the gate does not check those arms, so their rows below are
partly mmap. On BOX27 `worker2` was affected the same way (correction in `../box27/RESULTS.md`).
The mechanism, with a fix candidate, is OWED 26.

Descriptive only (not a verdict; every visit correct unless noted; token ids equal the oracle):

| Arm | Visits | Median tok/s | Range | Contaminated | Max foreign share |
|---|---|---|---|---|---|
| worker16 | 10 | 16.60 | 14.94 to 17.44 | 6 | 28.7% |
| mmap-random | 10 | 7.78 | 7.14 to 7.95 | 8 | 47.7% |
| mmap-normal | 10 | 17.87 | 15.73 to 19.42 | 6 | 48.3% |
| pread16 | 10 | 13.38 | 12.65 to 13.92 | 6 | 28.6% |
| worker2 | 10 | 9.15 | 8.30 to 9.88 | 7 | 30.0% |
| direct16 (refused) | 10 | 6.56 | 6.08 to 6.95 | 0 | 1.7% |

The direction matches BOX27's cold window 2 (mmap-normal ahead of worker16, every other arm
behind), but on this rig that is a description, not a result.

Record of the run: round 5's first attempt was interrupted by the requested reboot
(`capped/interrupted-round-05-reboot`); rounds 5 to 10's second attempt was refused by the
collector on the stale proof (`capped/refused-round-*-stale-proof`); both are kept, never scored.
The waits and every blocker are in `capped/waits.jsonl`.

## B3 bounded regime (ten rounds, 60 visits, MemoryMax 7,864,223,232): unscored

Registered verdict with the D and F fallback amendment (`bounded/pool.log`):

```
M1-5090-VERDICT regime_scored=False contaminated={'worker16': 10, 'mmap-random': 10, 'mmap-normal': 4, 'pread16': 5, 'worker2': 10}
rounds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] visits=60 refused=['direct16'] gpu_cotenant_unclean=2 visits_with_mmap_fallbacks=29
```

Two registered causes, per arm below. The B3 build (engine source equal to BOX27's, as section D
requires) predates the OWED 26 fix, so under the bounded page cache every `worker16` visit took
ring-busy mmap fallbacks (10 to 2,294 per visit) and every `worker2` visit too (1,248 to 14,492);
`direct16` fell back in 9 visits and is refused. Foreign device traffic on the shared volume
contaminated 3 to 10 visits per arm. `pread16`, the blocking arm, had zero fallbacks. The residency
bound held in all 60 visits (below 50% of the artifact's pages at visit end). Per-visit GPU telemetry
shows no active throttle reason; maximum temperature 83 C.

Descriptive only:

| Arm | Median tok/s | Range | Visits with fallbacks | Foreign share > 2% | Unclean |
|---|---|---|---|---|---|
| worker16 | 8.81 | 8.07 to 11.88 | 10 | 4 | 10 |
| mmap-random | 1.46 | 0.93 to 1.53 | 0 | 10 | 10 |
| mmap-normal | 4.30 | 2.79 to 4.41 | 0 | 4 | 4 |
| pread16 | 4.57 | 3.34 to 5.31 | 0 | 5 | 5 |
| worker2 | 4.02 | 1.47 to 5.23 | 10 | 6 | 10 |
| direct16 (refused) | 6.30 | 2.07 to 6.51 | 9 | 3 | 3 |

The first bounded cell was killed by the cgroup OOM killer on the runner's whole-file hash
(`refused-bounded-oom-runner-hash`, fixed before any visit); bounded rounds 1 to 7 held the card
back to back, rounds 8 to 10 ran with the 300 s card-sharing yield.

Reading: on this rig the B3 comparison cannot be scored with the pre-fix build; the fixed build's
worker path is what the OWED 17 cell and the PRO sitting measure.

## G2: pinned vs pageable host copies, all ten sizes (OWED 23)

`RESULT {"campaign": "G2-5090", "status": "all-visits-complete", "samples": 400, "n_per_size_direction_arm": 10, "ab_pairs": 5, "ba_pairs": 5, "power_envelope": {"power.limit": "[N/A]", "power.max_limit": "175.00 W"}, "qualification": false}`

One idle-gated cell, 794.9 s; `m1-g2-5090-summary.py` replayed every raw probe log against its
samples and hashes, the fixed calibration, N=5 in each order, the constant power fields, and
250 ms collector telemetry (median 251 ms, max 297 ms; GPU 59 to 72 C, SM 1,590 to 2,805 MHz,
PCIe gen 5 x8). Medians of 10 (`g2/round-01/g2-summary.json`):

| Bytes | h2d pageable GiB/s | h2d pinned GiB/s | h2d pinned/pageable time | d2h pageable GiB/s | d2h pinned GiB/s | d2h pinned/pageable time |
|---|---|---|---|---|---|---|
| 4,096 | 0.35 | 0.31 | 1.113 | 0.32 | 0.35 | 0.918 |
| 16,384 | 1.34 | 1.23 | 1.089 | 1.15 | 1.36 | 0.839 |
| 65,536 | 3.93 | 4.39 | 0.896 | 3.25 | 4.83 | 0.673 |
| 262,144 | 9.46 | 11.87 | 0.797 | 6.91 | 12.68 | 0.545 |
| 1,048,576 | 15.03 | 20.14 | 0.746 | 8.77 | 18.23 | 0.481 |
| 4,194,304 | 19.36 | 24.63 | 0.786 | 10.64 | 21.15 | 0.503 |
| 16,777,216 | 16.25 | 25.94 | 0.626 | 9.66 | 19.64 | 0.489 |
| 67,108,864 | 14.58 | 26.50 | 0.550 | 12.32 | 22.09 | 0.558 |
| 268,435,456 | 12.63 | 26.58 | 0.475 | 11.84 | 21.43 | 0.552 |
| 1,073,741,824 | 12.68 | 26.58 | 0.477 | 11.97 | 20.74 | 0.577 |

Reading: pinned wins both directions from 64 KiB up (host to device 26.6 GiB/s against 12.7 at
1 GiB), and device to host at every size. At 4 KiB and 16 KiB host to
device, pageable is faster on this card in all 10 visits of each order (pinned takes 1.100 to 1.157x
and 1.073 to 1.120x the time); BOX27's PRO 6000 had pinned ahead at every size (`../box27/RESULTS.md`
B6). A per-rig difference in the small-copy path; expert slices (about 860 KB) and KV frames sit far
above that range.

## Scratch loss, 2026-09-27 (recorded at 19:33Z)

Between the 19:20Z poll (the handoff 8 GiB cell's `waits.jsonl` read, 8 rounds done) and the next
one seconds later, the whole local scratch `~/spill-f-5090` disappeared, and the queue and its
driver ended, most likely when they next tried to create a file under the missing path. Nothing in
the user journal, the shell histories, or the other session that names the path shows a command
that removed it; the cause is unknown. Kept, because mirrored earlier: every regime above, the G2
cell, the OWED 17, 18 and 26 cells under `../owed17/5090`, `../owed18/5090`, `../owed26/5090`, and
the private store (BOX27, BOX36, proofs, the OWED 20 heads). Lost: the handoff 8 GiB cell's rounds 1
to 8 (never mirrored), the post-deletion pool-cell step (it had not started), the diagnostic
attribution samples, the queue logs after the last mirror, and the frozen binaries (their hashes
stay in each `build*/` record). The handoff 8 GiB cell reruns from round 1 on a rebuild of the
recorded commit, and every later cell is mirrored as soon as it ends.
