# WP-C day 79 (2026-09-26): OWED C11, I18, a ticket's records held by position instead of by cloned id, before any code

Lead: "C11 (the next cut to the host-hit demand, 0.155 ms per window token)". I17 read `flat` and stays (`DAY77.md`
section 3); the door loses to REF by 10 ms over 32 tokens on the 285K class. Tree at start: `962ba2b9f`.

## 0. Where the host-hit demand's time goes, from source and the day-61 profile

The owner's demand is the bank's `stage` then `publish` (the demand then the lease). Per block on the local CPU at
I15 (`cpu-day64/window.log`, P3 split): `stage` 755 to 771 ns, of which `stage_cache` 347 to 349, `stage_charge` 211
to 213 (the governor's per-ticket charge, the protocol's own), `stage_lookup` 44; `publish` 253 to 260, of which
`publish_policy` 171 to 173. `stage_cache` is the largest single part, and it is almost all bookkeeping around the host
cache, not the cache: `stage` (`residency.rs`) collects the batch's ids into a `BTreeSet<BankId>` (a clone of every
`BankId`, whose `TensorId` holds a `String`, and a tree node per id), then builds the ticket's records as a
`BTreeMap<BankId, BankLease>` (a node per record); `publish` then finds each output lease by `BankId` in that map
(each comparison walks two `Digest`s and a `String`), and the SLRU pass does it again. A host-hit prefetch of one
expert does this for three records every time.

## 1. Pre-registration: I18

**The change (memra-tier `residency.rs` only).** The ticket keeps its records by position:
- `stage` deduplicates the batch's ids by comparing them in place (a batch holds at most `MAX_GROUP` ids), keeps each
  unique id's first position, reads the host cache once per unique id as today, and records `Vec<Option<BankLease>>`
  in first-occurrence order plus each batch position's index into it; the missing records are sorted by `BankId`
  exactly as the `BTreeSet` ordered them, so the read plan, the charges and the checksums are in today's order;
- `publish` places each published lease at the position whose id it carries (a lease for an id outside the batch is
  kept at the end, as the map kept it), outputs by position, and the SLRU pass reads each id's lease by position; the
  path without an SLRU inserts into the host cache in `BankId` order, as the map iterated;
- `can_release` and the retire paths walk the vector where they walked the map's values.
No `BankId` is cloned for a host-hit record, no tree node is allocated, and every answer, order and refusal is today's.

**Why the program is the same.** The host cache is read the same number of times for the same ids, the leases are the
same `Rc` bodies, the output is in batch order as before, the SLRU sees the same ids in the same order, and the
missing records (and so the reads) keep their order. The host demand sequence must be byte-for-byte I17's.

**CPU gates before any card** (`day79-cpu/`): the tier bank suite, with a new test that a batch with a duplicated id,
a partly cached batch and a fully missing batch publish the same leases in the same order as the map did (checked
against a copy of the old logic kept in the test); the engine library; the day-61 profile's P3 split and P9 at I17
and at I18 in one window (both orders, the per-block medians); clippy and fmt.

## 2. Pre-registration: the card cell `i18` (the 285K class, then the RTX 5090; before its script)

DAY77's cell with I18 in I17's place: binaries `i17=d4ab19f1d` (REF's arm and the door before I18) and `i18` (named in
section 2a); arms REF (`run-gen-i17`, `MEMRA_MOE_PREFETCH=1`), I17, I18, I18C (I18 with `--moe-dispatch-clock`), 40
runs in both orders, then the profiled pair (REF, I18, I18, REF). Integrity as DAY77's, with I18's host demand sequence
equal to I17's; the admissibility clause; I18 against I17 (`improves`, `regresses`, `flat`, gen-only primary), the door
(I18, or I17 if I18 `regresses`) against REF; `regresses` on either card reverts I18 with its receipt.

## 2a. I18 on the CPU, the binary, and the sitting, before any cell

I18 landed as `c7294b912` (`residency.rs`: `TicketRecords`; `stage` builds it with the missing records sorted by id,
`publish` places leases by id and outputs by position, the SLRU pass and `can_release` read it, the path without an
SLRU inserts in `BankId` order). CPU gates (`day79-cpu/gates.log`): the tier suites green, the engine library 574
passed, clippy and fmt clean. One deviation from section 1, stated: the new test (`tests/bank/day79.rs`) checks the
map's contract (every output position gets the record whose id sits there, with the group all missing, partly cached,
all cached, and reversed) rather than a copy of the old code kept in the test; a batch with duplicated ids stays covered
by the existing rows test, which passes.

**The CPU profile** (`day79-cpu/profile.log`, the engine library's test binaries at I17 `c380dca2d` and at I18
`c7294b912`, one pinned core, order I17, I18, I18, I17, I17, I18; the rig shared with other lanes' agents): the
grouped cycle P9 (the door's path) reads 2394.7 to 2424.0 ns per block at I17 and 2234.6 to 2277.8 at I18, about 150 ns
less; in the single-record split `stage` drops from 856 to 889 to 741 to 788, `publish` rises from 298 to 308 to 331 to
354. About 0.45 us per prefetched expert, 0.04 ms per window token at the card's 88.6 prefetches: near the target
card's resolution again. The cell decides.

Binaries: `i17=d4ab19f1d` (REF's arm and the door before I18), `i18=c7294b912`. `day79-cell.sh`, `day79-read.py`
(DAY77's reader one step on: I18's host demand sequence equal to I17's) and `day79-box.sh`; dry checks
(`day79-cpu/`): the reader on DAY77's BOX32 receipts relabelled one step (`dry-check-reader.log`, meaningless), the
cell's control flow (22 runs per binary), the driver.

Run as `D79_BUILDS="i17=d4ab19f1d i18=c7294b912" bash /root/wt-c/research/spill-c-20260919/day79-box.sh` on a Core
Ultra 9 285K host with one RTX PRO 6000 Blackwell Workstation Edition (nsys). Expected: two builds about 10 minutes,
the cell about 25. The RTX 5090's half: queue v14 (`rtx5090-queue-v14-20260926.sh`).
