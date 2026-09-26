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
