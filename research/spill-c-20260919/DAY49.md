# WP-C day 49 (2026-09-24): the MoE slot cache door, improvement I7: the install's record pass in parallel

`OWED.md` C1 step (b). Day 40 read the door's install at 60.66 s per process on the RTX 5090 host (`install_s`, the
`installed` line minus the load's last line), of which the record pass was 55.77 s (`records_ns`: every retained
record compared byte for byte with the loaded `HostExps` and checksummed, one record at a time) and the artifact SHA
lock 5.08 s. I9 (day 44) makes the compare read page-cache bytes instead of write-combined memory; the pass is still
serial over 30,720 records. Written before any I7 code; tree at start: `2e900b115` (through I5).

## 1. Pre-registration

**The design.**

- (a) **The SHA lock stays first and serial.** The whole-file SHA-256 over the opened inode runs before anything else
  exactly as today, so a non-approved artifact still refuses before any catalog, bank, CUDA slot or record read (the
  day-18 `hashlock` cell's property).
- (b) **The record pass in parallel.** Inside `bank_projection`, the per-expert work that does not depend on order
  (the byte compare with `HostExps` and the contract `checksum`) runs on `min(16, available_parallelism)` scoped
  threads over contiguous expert ranges; every result is gathered by expert id, and everything order-dependent (the
  records digest chain, the catalog entries, the id map, the refusals) is then done serially in the same order as
  today. The first mismatch in expert order refuses with today's error, whichever thread saw it first.
- (c) Unchanged: every record is still compared and checksummed; the `catalog_sha256`, `records` and
  `records_sha256` values printed by the installer are identical to the serial pass's; no refusal text changes.

**Correctness.** A CPU test drives the parallel helper and a serial reference over the same synthetic slabs (with and
without a planted mismatch) and asserts the same digests in order and the same first-mismatch expert. On the cells the
installer's `catalog_sha256`, `records=` and `records_sha256=` must equal day 40's printed values
(`catalog_sha256=2204b159...defde8 records=30720 records_sha256=8084706a...0a459d`).

**The cell `install` (RTX 5090 first).** Day 48's shape and budget; arms OFF (the I7 binary, no door), I5 (the day-48
binary), I7 (the I7 binary), every door arm with `--expert-bank-stages`; order 1 (OFF, I5, I7) x 5, order 2 reversed
x 5, one collector hold. Integrity as day 48's plus the three installer identity values on every door run.

**Clauses.** (i) I7's `records_ns` below a quarter of I5's; (ii) I7's `install_s` below I5's by more than the larger
IQR in both orders; (iii) no window regression (`median(I7 window) - median(I5 window) <= noise` pooled and both
orders). I7 stays if all three hold.

**What each card can decide.** The RTX 5090 decides the clauses here; the target card reads them in the ladder (its
host runs SHA-256 at 2.15 GB/s against this host's 4.49, `DAY18.md`).

## 2. Results, cell `install` (RTX 5090 Laptop GPU, `rtx5090-day49/install/`)

One collector hold, 00:09:54Z to 00:16:22Z, 30 runs, tree `cf86f3771`, binaries `run-gen-i5` `2b5a738e...` and
`run-gen-i7` `3eab35aa...`, the approved artifact, the runner under the 1200% cap. Regime (`regime.log`, 250 ms,
N=1511): SM 1582 to 2782 MHz, power 28.4 to 163.9 W, 59 to 73 C. Collector `--validate` rc=0. The installer identity
values equal day 40's on every door run (the reader's integrity term).

Verbatim (`install/reading.log`):

`DAY49 INSTALL CHECKS rig=rtx5090 runs=30 integrity=ok`

`DAY49 ARM i5 install_s median=12.87 iqr=0.57 records_s median=4.31 sha_s median=5.25 window_s median=0.353 iqr=0.005`

`DAY49 ARM i7 install_s median=9.74 iqr=1.11 records_s median=0.96 sha_s median=5.24 window_s median=0.354 iqr=0.006`

`DAY49 CLAUSE (i) records_s i5=4.31 i7=0.96 rule i7 < 0.25 x i5 -> PASS`

`DAY49 CLAUSE (ii) install_s i7_minus_i5 o1=-3.52 o2=-2.58 noise=1.11 rule < -noise both orders -> PASS`

`DAY49 CLAUSE (iii) window i7_minus_i5 pooled=+0.001 o1=+0.002 o2=+0.001 noise=0.006 rule <=noise -> PASS`

`DAY49 INSTALL rig=rtx5090 integrity=ok clause_i=PASS clause_ii=PASS clause_iii=PASS`

I7 stays: the record pass 4.31 to 0.96 s, the install 12.87 to 9.74 s; the SHA lock (5.2 s) is now most of it.
