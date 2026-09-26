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
