# WP-C day 77 (2026-09-26): OWED C11, I17, one owner entry per group for residency and for staging, before any code

`DAY75.md` section 3: I16 regressed and was reverted; the door stays I15 and still loses to REF by 0.28 ms per
generated token and 0.19 per window token. `DAY72.md` section 3 placed the gap: the GPU waits on the door's prefetch
path, which costs 0.266 ms per window token more than REF's on the CPU, and between I10 and I15 each cut of that path
moved the window gap about one for one (the prefetch path's `pf_demand_ns` 0.464 to 0.154, the window gap 0.437 to
0.172 ms per token). So the next improvement cuts the path's CPU time without moving any work later. Tree at start:
`417e0a15d`.

## 0. Where the path's time is, from source and the day-61 profile at I15 (`cpu-day64/window.log`)

`prefetch_banked_group` (`moe_cache.rs`) takes the next expert's three blocks as a group (I15), but still enters the
owner registry once per block to ask `host_resident` (the proxy's `access`: the thread check, the thread-local
registry, the `RefCell`, the dynamic call), and once per block again in `with_bytes_at` to stage each copy. Per block
on the local CPU the proxy's `host_resident` reads 443 to 466 ns against 151 to 163 ns for the same query without the
proxy (P1 against P4b), and `with_bytes` 269 to 285 ns; on the target card the residency part alone is 0.046 ms per
window token (`pf_resident_ns`, 0.52 us per prefetched expert). Per prefetched expert that is six registry entries
where two would do.

## 1. Pre-registration: I17

**The change.** Two batched proxy calls, each one registry entry, and the grouped prefetch rewritten to use them:
- `ExpertBankProxy::host_resident_many(&[ExpertDispatchId]) -> Result<[bool; MAX_GROUP]>`: each record's
  `host_resident`, in order, in one `access`;
- `ExpertBankProxy::with_bytes_each(&ExpertGroupToken, f)`: lends each record's bytes of the group to `f(index,
  bytes)`, in order, in one `access`, the same checks as `with_bytes_at` (the group's identity, each lease's).
`prefetch_banked_group` validates as today, asks residency once for all wanted blocks, then reserves a slot for each
resident block in the same order as today, demands the chosen blocks in one call as today, and stages every chosen
block's copy in one `with_bytes_each`, recording each pending block exactly as today.

**Why the program is the same.** Host residency does not change between the per-block queries of today (reserving a
GPU slot touches no host state), so the same blocks are chosen in the same order, into the same slots, demanded in the
same call, and copied in the same order on the copy stream. So the host demand sequence must be byte-for-byte I15's,
and the tokens identical.

**CPU gates before any card** (`day77-cpu/`): the tier bank suite with new tests for both calls (the order, the
identity checks, a foreign or stale token refused, the error path leaving the lease as `with_bytes_at` leaves it); the
engine library; the day-61 profile's P8 (the grouped cycle) on the local CPU at I15 and at I17 in one window, both
orders, 5 repeats each, with the per-block medians; clippy and fmt.

## 2. Pre-registration: the card cell `i17` (the 285K class, then the RTX 5090; before its script)

DAY75's shape with I17 in I16's place: binaries `i15=2243b1fe2` and `i17` (named in section 2a); arms REF
(`run-gen-i15`, `MEMRA_MOE_PREFETCH=1`), I15, I17, I17C (I17 with `--moe-dispatch-clock`); order 1 (REF, I15, I17,
I17C) x 5, order 2 reversed x 5, 40 runs; then REF and I17 under Nsight Systems (REF, I17, I17, REF), window rows kept,
reports by hash.
- **Integrity** as DAY75's, plus: I17's host demand sequence equal to I15's (section 1's claim; a difference voids).
- **Admissibility** (DAY64 section 5's clause), then I17 against I15 (`improves`, `regresses` or `flat`, gen-only
  primary) and the door (I17, or I15 if I17 `regresses`) against REF (`beats`, `matches` or `loses`); I17C's brackets and
  Part B beside, deciding nothing.
- **What follows.** `regresses` on either card: I17 is reverted with its receipt; `flat` or `improves`: it stays.

## 2a. I17 on the CPU, the binary, and the sitting, before any cell

I17 landed as `d4ab19f1d`: `owner_proxy.rs` gains `host_resident_many` and `with_bytes_each`; `moe_cache.rs`
`prefetch_banked_group` validates every wanted member first, asks the group's residency once, reserves the resident
members' slots in the same order, demands them in one call, and stages them in one `with_bytes_each`, recording each
pending block as before (the error paths keep their meaning: a failed copy releases the later members' slots, a
refused borrow leaves the group's lease open). CPU gates (`day77-cpu/gates.log`): the engine library 574 passed; the
tier suites all green, the two new proxy tests among them (`tests/bank/day77.rs`); the censuses updated (`day48`'s
memo site, `day50`'s prefetch path: the two group calls present, the per-member calls gone); the day-4 fixture
re-pinned (no SLRU statement changed); clippy and fmt clean.

**The CPU profile** (`day77-cpu/profile.log`, the local CPU, one pinned core, three invocations; the rig was shared
with other lanes' agents at the time, so the absolute values sit above DAY64's): the day-61 profile's P8 (I15's
grouped cycle) and the new P9 (I17's) read in one window, P8 again after P9: P9 2386.5 to 2413.2 ns per block against
P8 2456.7 to 2566.7 before and after it. About 70 to 150 ns per block, 0.2 to 0.45 us per prefetched expert: smaller
than section 0's reading of the proxy entries suggested (most of `host_resident`'s 450 ns is the query inside the
owner, not the entry). At the card's 88.6 prefetches per window token that is about 0.02 to 0.04 ms per token, near
the target card's noise (0.001 s over 32 tokens); the cell decides.

Binaries: `i15=2243b1fe2`, `i17=d4ab19f1d`. `day77-cell.sh`, `day77-read.py` (DAY75's reader with the host demand
sequence equality of section 2) and `day77-box.sh` were written after section 2. Dry checks (`day77-cpu/`): the reader
on two synthetic cells from DAY75's receipts (`dry-check-reader.log`: with I16's runs as I17 the sequence check voids,
as it must; with I15's runs as I17 it reads); the cell's control flow (`dry-check-cell.log`: 22 runs per binary); the
driver (`dry-check-driver.log`).

Run as `D77_BUILDS="i15=2243b1fe2 i17=d4ab19f1d" bash /root/wt-c/research/spill-c-20260919/day77-box.sh` on a Core
Ultra 9 285K host with one RTX PRO 6000 Blackwell Workstation Edition (nsys; box needs as `DAY72.md` section 1a).
Expected: two builds about 10 minutes, the cell about 25. The RTX 5090's half: queue v13
(`rtx5090-queue-v13-20260926.sh`).

## 3. The target card (BOX32, the 285K class, run by the lead as registered; `pro-single-day77/`)

BOX32 is BOX29's machine re-rented (a Core Ultra 9 285K, 188 GB, one RTX PRO 6000 Blackwell Workstation Edition).
`D77_BUILDS="i15=2243b1fe2 i17=d4ab19f1d" bash .../day77-box.sh` on the tree `1d4f94cba`, 10:54Z to `box done
2026-09-26T11:08:31Z`. Receipts: 219 of 219 `OK` against the box manifest (re-checked), ELFs and profiles by hash.
Regime: 32 to 47 C, SM median 2610 MHz, N=2024. Verbatim (`i17/reading.log`):

- `DAY77 host demand sequence i15 sha256 4bdc2610c3534e42 lines=[22077]`, and the same for `i17` and `i17c`
- `DAY77 I17 CHECKS rig=pro-single runs=40 integrity=ok`
- `DAY77 ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0005 max_iqr_window=0.0010 failing=[] -> admissible`
- `DAY77 gen-only decode medians (N=10 each): ref=0.254 i15=0.264 i17=0.264 i17c=0.264`
- `DAY77 STEP i17_vs_i15 gen-only decode: pooled=+0.0000 o1=+0.0000 o2=+0.0000 noise=0.0005 -> flat`
- `DAY77 STEP i17_vs_i15 steady window: pooled=-0.0010 o1=-0.0010 o2=+0.0000 noise=0.0010 -> flat`
- `DAY77 DOOR i17_vs_ref gen-only decode: pooled=+0.0100 o1=+0.0100 o2=+0.0100 noise=0.0005 -> loses`
- `DAY77 VERDICT rig=pro-single integrity=ok i17=flat door=i17 vs_ref=loses (window: i17=flat vs_ref=loses)`

**Read as registered: I17 `flat`, so it stays; the door (I17) `loses` to REF** (0.264 against 0.254 gen-only, 0.231
against 0.226 window). The host demand sequence is byte-for-byte I15's, as section 1 claimed. As the CPU profile
predicted, the saving is below this card's resolution: I17C's residency bracket reads 0.0426 ms per window token
against I15's 0.046, and the profiled GPU still idles 0.41 ms per window token more than REF's.

## 3a. The RTX 5090 (queue v13, 2026-09-26 10:49Z to 11:00Z; `rtx5090-day77/i17/`)

`run-gen-i17` rebuilt locally from `d4ab19f1d`. Regime: 60 to 81 C, SM median 1687 MHz, N=2429 (warmer than earlier
5090 holds; the rig ran other lanes' work). Verbatim: `DAY77 ADMISSIBILITY rig=rtx5090 ceiling=0.005 max_iqr_gen=0.0120
max_iqr_window=0.0195 failing=[...every arm...] -> inadmissible`, `DAY77 VERDICT rig=rtx5090 integrity=ok -> void
(inadmissible) [as read: i17=flat door=i17 vs_ref=matches (window: i17=flat vs_ref=matches)]`. Decides nothing; the
same host demand sequence held here too.
