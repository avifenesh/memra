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

## 2. The run

(filled after the run)

## 3. The reading

(filled after the run)

## 4. Records and checks

(filled at close)
