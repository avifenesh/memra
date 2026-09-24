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
