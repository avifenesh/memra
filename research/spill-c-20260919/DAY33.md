# Session C day 33: does the Move 1 demote's on-tick hash read write-combined memory on the RTX 5090 class, and at what rate

Lane `lane/spill-c-20260919`, checkout `wt-spill-c`. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the hook prints `UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-c-20260919
at <sha>; no GPU qualification claimed` and records the skip in the clone's `.git/memra-gate-skips.log`). Today's card is
the local RTX 5090 Laptop GPU under the collector's lock (`/tmp/memra-5090.lock`, `--rig rtx5090`), shared with the lead's
battery and hit-gate runs; every cell is `executed-not-qualified`; nothing here is a support state or a qualification
claim; every median is quoted with its N and regime; this card's figures are compared to nothing from the target card.
No engine change: the one code edit is an opt-in arm in the `hash-micro` diagnostic bin (below). No commit on main, no PR.

## Merge (first action)

`git fetch origin`; no PR from `lane/spill-integ42-20260922` existed and the branch was not on origin, so the local ref
`lane/spill-integ42-20260922` (`af92791f6`, the lead's integ42: C day 31, C day 32, A day 26, #645) was merged with
`--no-ff` as `596dded90` (clean, no conflict, so no side of `HOSTPREFIX-DOOR.md` had to be chosen) and pushed:
`UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-c-20260919 at 596dded90cf2ad35c1c54fa39a42afbaddd1b90b; no GPU
qualification claimed`, `pre-push: skip recorded`. Tree for today's cells: this merge plus the instrument's arm.

## The instrument's new arm (`hash-micro --two-step`; the diagnostic bin only, no engine path)

`crates/memra-engine/src/bin/hash_micro.rs` (day 18's diagnostic, `e16bc69e8`) gains `--two-step`: the write-combined
buffer hashed in place by `memra_tier::contracts::checksum` (the door's hash; its access pattern is one sequential
single-pass read, `Sha256::update` over the whole slice after the framed domain and length prefixes, `contracts.rs:75-86`)
against `copy_from_slice` of the write-combined buffer into a cached pinned buffer (`cuMemHostAlloc` flags 0) followed by
`checksum` over that copy, the two steps timed separately and summed; two orders (`wc, wc_copy` then `wc_copy, wc`), N
passes per kind per order, one `pass` line per pass (`kind=wc_copy` lines carry `memcpy_ms` and `hash_ms`) and one
`HASH-MICRO two-step rule` line; the digests of both kinds must agree or the process exits non-zero. Without the flag the
bin's output is day 18's, byte-for-byte in shape (`day18-replay.py`'s checks still apply to it). No engine path reads it,
no default moves, no `MEMRA_*` read (no `docs/FLAGS.md` row), no `.cu` file (no `docs/KERNELS.md` row). `cargo fmt` clean.

## 1. Pre-registration (written and pushed before any GPU run today)

### The question

`HOSTPREFIX-DOOR.md` section D item 6 (the census question) as it stands on this card after day 31: the ON demote's `in`
figure minus its `from submission to completion` figure is 21 to 23 ms on every one of the 12 ON demotes of the pair cell
(`rtx5090-day31/pair/`), and that is the part of the demote that runs after the copy landed (the tick-top poll, the two
host hashes, the publish); the day-18 hash micro-cell read `wc_ms=1431.613` per 160 MiB pass over write-combined memory on
this card (`rtx5090-day18/hashmicro/`), which puts about 490 ms per pass at 54.8 MB; and the pair cell printed the
premise `PINNED-DEFAULT device="NVIDIA GeForce RTX 5090 Laptop GPU" kind=write-combined flags=4` inside its hold. So: does
the Move 1 demote's on-tick hash read write-combined memory on this card, and at what rate? Lane A day 27 reads the same
question from the code side (where each hash runs and over which memory); this side is the measurement. The two demote-side
hash sites are named here only as pointers for A's census, not read: `crates/memra-engine/src/tier_transfer.rs:1697-1711`
(the completion checksum: a `receipt_digest_from_lanes` branch and a `checksum(item.host...bytes())` branch) and
`crates/memra-server/src/worker.rs:9344` (the bundle checksum at take, `checksum(bytes)` per plane).

### The hypotheses, named before running

- **H1.** The hash reads the write-combined pinned destination, and the micro-cell's 1431.6 ms per 160 MiB rate does not
  apply to the hash's access pattern (for example, a sequential read of write-combined memory at the entry's size, or in
  the server's per-plane pieces, runs far faster than the micro-cell's pass; or a `memcpy` of the write-combined bytes
  into cached memory followed by the hash of the copy is far cheaper than hashing in place).
- **H2.** The hash reads a cached copy, a staging buffer, or the device bytes (a device-side digest), not the write-combined
  destination.
- **H3.** The destination is not write-combined on this tree for this class despite the `PINNED-DEFAULT` line (the door's
  lease allocation would then not go through the site the transfer gate prints from).

### What each predicts for the cells below

- **H1** predicts at least one write-combined read route at 54,800,000 B whose cost fits inside the ON demote's wall time:
  either the single-pass hash over write-combined memory or the two-step `memcpy` plus hash reads at or under the slowest
  day-31 ON demote `in` figure (60.4 ms). It also predicts that the 160 MiB single pass reproduces day 18's order of
  magnitude only if the size is what changes the rate; if the sequential rate at 54.8 MB is the same 0.1 GB/s as at
  160 MiB, H1's first form is out, and only the two-step form remains to be read.
- **H2 and H3** predict the same numbers from these cells: the write-combined single pass at 54,800,000 B near 490 ms
  (the day-18 rate carried to this size), the two-step route also well above 60.4 ms (the `memcpy` step reads the same
  uncached memory), and the cached and heap passes near 12 ms (the day-18 cached rate, 4.49 GB/s, at this size), so that
  two cached passes (about 24 ms) sit near the day-31 `in` minus completion figure (21 to 23 ms). These cells cannot
  separate H2 from H3: both say the hashes do not read write-combined memory at this rate; which memory they do read, and
  whether the door's leases are the printed arm, is the code census (A day 27), not this measurement. The cells can only
  refute H1 or let it stand.

### The cells (one collector lock hold on `/tmp/memra-5090.lock`, `day33-cell.sh` under `day33-local-run.sh`; receipts `rtx5090-day33/hashwc/`)

Inside the hold, in this order, fixed now: the premise receipt `tier-transfer-gate roundtrip` (`ev/pinned-default.log`, the
`PINNED-DEFAULT device= kind= flags=` line from the production allocator, as day 31), then four `hash-micro` invocations,
each through the line stamper into `ev/<label>.log` with its exit status in `ev/<label>.exit`:

- (a) `micro-54m`: `hash-micro --bytes 54800000 --n 5` (the pair cell's entry, the server's `54.8MB` printed at 1e6 per
  MB, so 54,750,000 to 54,849,999 B; 54,800,000 is used and SHA-256 throughput is length-independent at this size) over
  cached pinned, write-combined pinned and heap memory, day 18's shape: N=5 per kind per order, two orders (`cached, wc,
  heap` then `heap, wc, cached`), N=10 pooled per kind.
- (b) `twostep-54m`: `hash-micro --two-step --bytes 54800000 --n 5`: the single-pass hash over write-combined memory
  against the `memcpy` into cached pinned memory plus the hash of the copy; N=5 per kind per order, two orders, N=10
  pooled per kind, the `memcpy` and hash steps reported separately.
- (a) `micro-160m`: `hash-micro --bytes 167772160 --n 5` (day 18's size, re-run in the same hold as the 54.8 MB cells so
  the two sizes share one regime).
- (b) `twostep-160m`: `hash-micro --two-step --bytes 167772160 --n 5`.
- (c) The day-31 pair receipts re-read by `day33-reading.py` (`rtx5090-day31/pair/wc-pair/ev/{o1,o2}-{on,off}-server.log`):
  per ON demote, the `demote submitted` line, the `D2H receipt` line (present or absent; by format it carries issuer,
  seq, epochs, items, complete, require, checksums and the retire word, and no timing, so there is nothing to tabulate
  for it and nothing missing, hence no new boot), the `from submission to completion` figure, the `in` figure and their
  difference, N=6 per ON boot, N=12 pooled, with the OFF boots' `in` beside them (N=12). No new boot unless a line is
  missing; the dry run of the re-read on the day-31 receipts before this push found none missing.

Regime: the collector's 250 ms `command.gpu.csv` plus the cell's own 1 s `ev/card.during.csv`, `card.{before,after}.csv`,
`compute-apps.{before,after}.csv`, `loadavg.{before,after}.txt`. The sitting runs inside `systemd-run --user --scope -p
CPUQuota=1200% -p MemoryMax=28G` (the hash is single-threaded, so the quota does not bound it). Bounded wait as day 31: no
compute app and at least 20000 MiB free before the hold, a refused lock retried, 15 waits of 120 s in total; if that ends
busy the cell is `NOT RUN` with the card's last snapshot (`hashwc/NOT-RUN.txt`) and nothing on the card is inspected
beyond `nvidia-smi`'s own listing or signalled.

### Admissibility (an inadmissible cell decides nothing and is reported as such)

The lock proof `acquired=true` on `/tmp/memra-5090.lock`; collector status `executed-not-qualified`; the premise line
reads `kind=write-combined` (if it reads `cached` the premise of the question is wrong and the numbers say so); no compute
app in the before and after snapshots; every invocation exits 0 with exactly one rule line; 10 passes per kind; the rule
line's medians agree with the pass lines within 0.01 ms; digests equal across kinds in every invocation; the driver's
write-combined bit set on the write-combined buffer only (and not on the cached copy in the two-step arm); `bytes` and
`n_per_order` as ordered; 12 ON demotes with both an `in` and a completion figure and 12 OFF demotes with an `in` figure in
the day-31 receipts. Applied by `day33-reading.py`, which prints one `DAY33 HASH-WC VERDICT:` line.

### The clauses (read by the script, verbatim)

- **H1-single**: the FASTEST single-pass write-combined hash at 54,800,000 B (N=20, both invocations' `wc` passes) is at
  or under the SLOWEST day-31 ON demote `in` (60.4 ms). True means a single-pass read of the destination fits the wall time.
- **H1-twostep**: the FASTEST `memcpy` plus hash of the write-combined buffer at 54,800,000 B (N=10) is at or under 60.4 ms.
- If both are false: `H1 refuted (no WC read route fits the ON demote's wall time; H2 or H3 stands, separated by the code
  census, not by this cell)`. If either is true: `H1 stands`. The fastest pass and the slowest demote are used on purpose:
  the clause asks whether the route is possible at all, not whether it is typical.
- Arithmetic beside the clauses, no verdict: two cached passes at 54,800,000 B against the day-31 `in` minus completion
  median (21.6, N=12, IQR 1.1); one cached pass; the 160 MiB write-combined pass against day 18's 1431.6 (cross-sitting,
  same box, stated as such).

### What I will NOT do

No tuning after a result; a second attempt happens only for a lock refusal, an idle wait that ends busy, or a process
failure, and is reported as an attempt. No change to the sizes, N, orders or clauses after the run. No third lock name; no
bare GPU run (every invocation inside the collector's hold); no touch of other sessions' processes (the lead's battery
and hit-gate runs share this card and take the same lock); nothing under other lanes' worktrees; no engine change; no
cross-card comparison.

## 2. The run (tree `71bd494db`, `hash-micro` `5b3d09181ef90db6…` built under the CPU quota from this tree, `rtx5090-day33/local/build.log` rc=0; `tier-transfer-gate` `01c2d4c49db8c49b…`, day 31's binary in this worktree's `target/`; receipts `rtx5090-day33/hashwc/`)

**Wait and hold.** The card carried other sessions' processes before the run (a `decode-batch-gate` at 18.4 GB, then a
`memra-server` at 15.2 GB and 8.9 GB in the snapshots I took while writing; none touched). The runner's first idle probe
passed and its lock attempt was refused at 14:48:38Z (`hashwc/waits.log` `attempt 0: lock busy`; the lead's gate holds the
flock between its boots); while it slept I stopped that runner (my own process, holding nothing) because it ran under the
tool's 10-minute cap, which could have cut a hold, and relaunched it detached at 14:49:15Z (the restart is a line in
`waits.log`; the first attempt's `hash-wc-driver.log` was overwritten by the second, as the runner names them the same). The
detached runner's attempt 0 took the lock at once: collector window 14:49:18.39Z to 14:50:21.82Z (63 s), `hash-wc/lock.json`
`acquired: true` on `/tmp/memra-5090.lock` (`inherited-flock-same-open-description`, inode 166260), `ev/LOCK.json` the same
file, `CELL.jsonl` `status: executed-not-qualified`, `exit_code: 0`, one attempt, zero idle waits, one lock refusal. Inside the
hold, before the invocations, the premise receipt (`ev/pinned-default.log`, rc=0): `PINNED-DEFAULT device="NVIDIA GeForce RTX
5090 Laptop GPU" kind=write-combined flags=4`, then `kind=write-combined driver_flags=6` at every roundtrip size. The four
invocations by the marks: `micro-54m` 14:49:30.4 to 14:49:35.7, `twostep-54m` to 14:49:43.3, `micro-160m` to 14:49:58.8,
`twostep-160m` to 14:50:20.7, every `rc=0`.

**Regime.** Collector `command.gpu.csv`: 252 samples at 250 ms, 56 to 58 C, 9.5 to 31.7 W (`power.limit [N/A]`), SM clock 180
to 1635 MHz, memory used 15 to 533 MiB. The cell's `ev/card.during.csv`: 63 samples at 1 s, 57 to 58 C, 9.5 to 31.7 W.
`card.before.csv`: 15 MiB used, 57 C, 31.69 W, P0; `card.after.csv`: 15 MiB, 56 C, 11.21 W, P8; `compute-apps.{before,after}.csv`
empty. Host load average 2.47 before, 1.66 after (1 min; the sitting inside `systemd-run --scope -p CPUQuota=1200% -p
MemoryMax=28G`). The hash is single-threaded and the card idles through it: this is a host-memory cell under a CUDA context.

**Admissibility.** `day33-reading.py rtx5090-day33 rtx5090-day31` (`hashwc/reading.log`): 41 `ok`, 0 `FAIL`, `admissible=True`.
Every clause of section 1 held: lock proof, collector status, premise `write-combined`, no compute app, four exits 0, one
rule line each, 10 passes per kind, medians agreeing with the pass lines, digests equal in every invocation, the driver's
write-combined bit on the write-combined buffer only (and not on the two-step copy), `bytes` and `n_per_order` as ordered,
12 ON and 12 OFF demotes in the day-31 receipts. One reading-script fix after the run, stated: the first pass of the script
looked for `acquired` in `ev/LOCK.json`, which the lock-proof tool does not write (it writes owner, lock, mechanism, device,
inode); the collector's own `lock.json` in the same cell carries `acquired: true` for the same inode. The script now reads the
collector's file for `acquired` and checks the cell's proof names the same lock file; the first output is kept verbatim as
`hashwc/reading.first.log` (40 ok, 1 FAIL on that check alone, every number identical). No cell was re-run; nothing was tuned.

**The rule lines, verbatim.**

- `micro-54m`: `HASH-MICRO rule device="NVIDIA GeForce RTX 5090 Laptop GPU" bytes=54800000 n_per_order=5 pooled=10 orders=2 digest_equal=true wc_bit_cached=false wc_bit_wc=true cached_ms=11.862 wc_ms=473.846 heap_ms=11.939 cached_range=11.786..12.053 wc_range=468.756..476.108 heap_range=11.783..13.526 cached_o1=11.912 cached_o2=11.846 wc_o1=473.700 wc_o2=474.727 heap_o1=11.834 heap_o2=12.846 cached_gbps=4.620 wc_gbps=0.116 heap_gbps=4.590 wc_over_cached=39.948 cached_over_heap=0.994 two_hashes_cached_ms=23.723 one_hash_cached_ms=11.862`
- `twostep-54m`: `HASH-MICRO two-step rule device="NVIDIA GeForce RTX 5090 Laptop GPU" bytes=54800000 n_per_order=5 pooled=10 orders=2 digest_equal=true wc_bit_wc=true wc_bit_copy=false wc_ms=473.409 wc_copy_ms=224.178 wc_copy_memcpy_ms=212.329 wc_copy_hash_ms=11.837 wc_range=469.934..476.108 wc_copy_range=222.009..225.832 wc_copy_memcpy_range=210.188..213.998 wc_copy_hash_range=11.821..11.959 wc_o1=473.144 wc_o2=473.673 wc_copy_o1=223.409 wc_copy_o2=225.163 wc_gbps=0.116 wc_copy_gbps=0.244 wc_copy_over_wc=0.474`
- `micro-160m`: `HASH-MICRO rule device="NVIDIA GeForce RTX 5090 Laptop GPU" bytes=167772160 n_per_order=5 pooled=10 orders=2 digest_equal=true wc_bit_cached=false wc_bit_wc=true cached_ms=36.585 wc_ms=1450.827 heap_ms=37.223 cached_range=36.296..38.160 wc_range=1442.075..1456.979 heap_range=34.934..39.313 cached_o1=36.546 cached_o2=37.100 wc_o1=1451.463 wc_o2=1450.191 heap_o1=37.464 heap_o2=36.982 cached_gbps=4.586 wc_gbps=0.116 heap_gbps=4.507 wc_over_cached=39.657 cached_over_heap=0.983 two_hashes_cached_ms=73.169 one_hash_cached_ms=36.585`
- `twostep-160m`: `HASH-MICRO two-step rule device="NVIDIA GeForce RTX 5090 Laptop GPU" bytes=167772160 n_per_order=5 pooled=10 orders=2 digest_equal=true wc_bit_wc=true wc_bit_copy=false wc_ms=1450.024 wc_copy_ms=686.817 wc_copy_memcpy_ms=650.234 wc_copy_hash_ms=36.628 wc_range=1444.339..1455.452 wc_copy_range=680.107..691.253 wc_copy_memcpy_range=643.744..653.727 wc_copy_hash_range=36.136..39.006 wc_o1=1448.638 wc_o2=1451.433 wc_copy_o1=686.637 wc_copy_o2=687.173 wc_gbps=0.116 wc_copy_gbps=0.244 wc_copy_over_wc=0.474`

**The medians (ms, N=10 pooled per kind, N=5 per order; the two orders agree within 1.1 ms at 54.8 MB and 2.8 ms at 160 MiB on every kind), this card, this sitting, 56 to 58 C.**

| size | cached pinned | write-combined, single pass | heap | write-combined, memcpy into cached then hash | the memcpy step | the hash step |
|---|---|---|---|---|---|---|
| 54,800,000 B | 11.862 | 473.846 (the two-step invocation's own `wc` arm: 473.409) | 11.939 | 224.178 | 212.329 | 11.837 |
| 160 MiB | 36.585 | 1450.827 (the two-step invocation's own `wc` arm: 1450.024) | 37.223 | 686.817 | 650.234 | 36.628 |

**(c) The day-31 pair receipts re-read (`rtx5090-day31/pair/wc-pair/ev/*-server.log`; the logs carry no per-line timestamps, so
the `D2H receipt` line has no timing of its own by format and nothing was missing: no new boot).**

| boot | seq | `demote submitted` | `D2H receipt` (items, require) | `from submission to completion` ms | `in` ms | `in` minus completion ms |
| o1-off | (door OFF) | n/a | n/a | n/a | 23.1 | n/a |
| o1-off | (door OFF) | n/a | n/a | n/a | 23.8 | n/a |
| o1-off | (door OFF) | n/a | n/a | n/a | 21.8 | n/a |
| o1-off | (door OFF) | n/a | n/a | n/a | 5.1 | n/a |
| o1-off | (door OFF) | n/a | n/a | n/a | 4.8 | n/a |
| o1-off | (door OFF) | n/a | n/a | n/a | 5.3 | n/a |
| o1-on | 3 | 64 tok, 54.8 MB, 18 items | seq=3 items=18 (8 KV planes, draft) complete=18 require=ok | 34.8 | 57.8 | 23.0 |
| o1-on | 5 | 64 tok, 54.8 MB, 18 items | seq=5 items=18 (8 KV planes, draft) complete=18 require=ok | 37.4 | 60.4 | 23.0 |
| o1-on | 8 | 64 tok, 54.8 MB, 18 items | seq=8 items=18 (8 KV planes, draft) complete=18 require=ok | 37.2 | 58.9 | 21.7 |
| o1-on | 11 | 64 tok, 54.8 MB, 18 items | seq=11 items=18 (8 KV planes, draft) complete=18 require=ok | 18.3 | 39.7 | 21.4 |
| o1-on | 14 | 64 tok, 54.8 MB, 18 items | seq=14 items=18 (8 KV planes, draft) complete=18 require=ok | 18.0 | 39.5 | 21.5 |
| o1-on | 17 | 64 tok, 54.8 MB, 18 items | seq=17 items=18 (8 KV planes, draft) complete=18 require=ok | 18.1 | 39.7 | 21.6 |
| o2-on | 3 | 64 tok, 54.8 MB, 18 items | seq=3 items=18 (8 KV planes, draft) complete=18 require=ok | 35.3 | 57.7 | 22.4 |
| o2-on | 5 | 64 tok, 54.8 MB, 18 items | seq=5 items=18 (8 KV planes, draft) complete=18 require=ok | 36.3 | 58.8 | 22.5 |
| o2-on | 8 | 64 tok, 54.8 MB, 18 items | seq=8 items=18 (8 KV planes, draft) complete=18 require=ok | 35.5 | 56.9 | 21.4 |
| o2-on | 11 | 64 tok, 54.8 MB, 18 items | seq=11 items=18 (8 KV planes, draft) complete=18 require=ok | 18.1 | 39.4 | 21.3 |
| o2-on | 14 | 64 tok, 54.8 MB, 18 items | seq=14 items=18 (8 KV planes, draft) complete=18 require=ok | 17.3 | 38.5 | 21.2 |
| o2-on | 17 | 64 tok, 54.8 MB, 18 items | seq=17 items=18 (8 KV planes, draft) complete=18 require=ok | 17.5 | 38.7 | 21.2 |
| o2-off | (door OFF) | n/a | n/a | n/a | 18.2 | n/a |
| o2-off | (door OFF) | n/a | n/a | n/a | 24.1 | n/a |
| o2-off | (door OFF) | n/a | n/a | n/a | 22.6 | n/a |
| o2-off | (door OFF) | n/a | n/a | n/a | 5.4 | n/a |
| o2-off | (door OFF) | n/a | n/a | n/a | 6.5 | n/a |
| o2-off | (door OFF) | n/a | n/a | n/a | 17.6 | n/a |

Pooled: ON `in` `median 48.3 (N=12, min 38.5, max 60.4, IQR 18.8)`; ON `from submission to completion` `median 26.5 (N=12, min
17.3, max 37.4, IQR 17.8)`; ON `in` minus completion `median 21.6 (N=12, min 21.2, max 23.0, IQR 1.1)`; OFF `in` `median 17.9
(N=12, min 4.8, max 24.1, IQR 17.5)`. (Day 31 pooled r2 to r6 for its `in` medians, N=10; this table pools all six demotes per
boot, N=12; the raw lists are the same lines.)

**The verdict line, verbatim (`hashwc/reading.log`).** `DAY33 HASH-WC VERDICT: 54.8MB cached 11.9 wc 473.8 heap 11.9 wc_copy 224.2
(N=10 each); 160MiB cached 36.6 wc 1450.8 heap 37.2 wc_copy 686.8 (N=10 each); day31 ON in median 48.3 max 60.4 (N=12)
in_minus_completion median 21.6 (N=12); H1-single fits=False H1-twostep fits=False -> H1 refuted (no WC read route fits the ON
demote's wall time; H2 or H3 stands, separated by the code census, not by this cell); pinned=write-combined; admissible=True`

## 3. The reading (the hypotheses against the numbers; nothing tuned)

- **H1 is refuted on this card, both forms.** Clause H1-single: the fastest single-pass hash over 54,800,000 B of
  write-combined memory in the sitting is 468.8 ms (N=20), 7.8x the SLOWEST day-31 ON demote `in` (60.4 ms) and 9.7x its
  median (48.3). Clause H1-twostep: the fastest `memcpy` of the write-combined buffer into cached pinned memory plus the hash
  of the copy is 222.0 ms (N=10), 3.7x the slowest ON demote. The memcpy reads the same uncached memory at 0.258 GB/s
  (212.3 ms for 54.8 MB, 650.2 for 160 MiB), 2.2x faster than the byte stream sha2 reads but the same order; the two-step
  route is 0.474 of the single pass at both sizes. Neither route fits inside the whole ON demote, let alone inside the 21 to
  23 ms that follow completion.
- **The write-combined rate is size-independent here.** `wc_gbps=0.116` at 54.8 MB and at 160 MiB (473.8 against 1450.8 ms,
  the ratio 3.06 against the byte ratio 3.06); `wc_over_cached` 39.9 and 39.7. So the "sequential read at the entry's size
  behaves differently" form of H1 is out by measurement. The "per-plane pieces" form (18 items of about 3 MB) was NOT
  measured today; by arithmetic it hashes the same bytes with the same uncached reads plus 17 more finalizations, and
  cannot be 8x cheaper, but that is arithmetic and is stated as such.
- **H2 or H3 stands; this cell does not separate them.** Both predicted exactly these numbers: write-combined near 490 ms
  (473.8 measured), cached and heap near 12 ms (11.9 and 11.9), two cached passes near the post-completion segment
  (`two_hashes_cached_ms=23.723` against `in` minus completion `median 21.6, max 23.0`). That segment is consistent with
  up to two hashes of the entry over cached or heap memory (or one hash of 11.9 ms plus about 10 ms of poll, insert and
  publish work) and inconsistent with any read of 54.8 MB of write-combined memory by either route measured. Which memory
  the two demote-side hashes read (`tier_transfer.rs:1697-1711`: a lanes-derived receipt digest or a host-read checksum;
  `worker.rs:9344`: the bundle checksum at take), and whether the door's leases are the printed arm (H3), is A day 27's
  code census; nothing here infers it.
- **The premise held again.** `kind=write-combined flags=4` from the production allocator inside the hold, `driver_flags=6`
  at every roundtrip size, as day 31.
- **Day 18 reproduced within the cross-sitting spread (same box, stated as such, not a same-window figure).** 160 MiB
  write-combined 1431.6 then, 1450.8 now (+1.3 %); cached 37.3 then, 36.6 now; heap 37.5 then, 37.2 now.
- **What stays open.** The promote-side half of item 6 (the completion checksum over the host source at promote, a cached
  77.9 ms pass on the target card against a measured 1.4 ms promote-share delta) was not touched today. The per-plane form
  of H1 is arithmetic, not a cell. H2 against H3 is code, not measurement.

## 4. Records and checks

`DAY33.md` (this file), `STATE.md`, `research/INDEX.md` row `spill-c-20260919/day33`, `HOSTPREFIX-DOOR.md` section D item 6 (the
RTX 5090 measurement; A's code-side answer lands separately). Receipts `rtx5090-day33/` with `.gitattributes` (`*.log
-whitespace`), `local/` (build log, binary hashes, runner log) and `hashwc/` (collector output, `ev/`, `reading.log`,
`reading.first.log`, `waits.log`, `progress.log`). Checks at close: shellcheck clean on `day33-cell.sh` and
`day33-local-run.sh`; `tools/check-flags.sh` every runtime `MEMRA_*` name resolves (no new read); `tools/check-conflict-markers.sh`
OK; `git diff --check` clean; `cargo fmt` clean on the bin; zero em dashes in this lane's lines; no host, id, location or
cost in tracked files. Boundaries: no engine change (the diagnostic bin's opt-in arm only), no V4.1 code, no external
dependency, no `--no-verify`, no third lock name, no bare GPU run, no touch of other sessions' processes or worktrees, no
cross-card comparison, no medians without N and regime, no cell tuned after a result. At close NOTHING RUNNING for C: no
server, no lock, no GPU process; `/tmp/spill-c-day33-checkflags.log` removed. Budget: about 2.4 agent-hours against 3.

## Push section

- `596dded90`: the integ42 merge (`UNQUALIFIED DEVELOPMENT: refs/heads/lane/spill-c-20260919 at 596dded90cf2ad35c1c54fa39a42afbaddd1b90b; no GPU qualification claimed`, `pre-push: skip recorded`).
- `71bd494db`: the pre-registration, the bin's arm, the scripts (`UNQUALIFIED DEVELOPMENT: ... at 71bd494db4227559fe8ea0467873638a5094c369; no GPU qualification claimed`), before any GPU run.
- The day-docs tip: the receipts, this reading and the records; its SHA is the commit that carries this line.
