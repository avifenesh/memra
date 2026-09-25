# WP-C day 61 (2026-09-25): the door's host-hit lease, profiled on the CPU, then tuned (lead owed item 2), registered before code

Lead, integ60 resume: "tune the door to match or beat it ..., each improvement with its own pre-registration and
cell". `DAY60.md` registers the attribution of the door's gap to REF on both cards; the RTX 5090 needs its reset and
no target card is up, so that cell waits. This day works the one lead the existing receipts already give, on the CPU,
where no card is needed. Tree at start: `da649107c`.

## 0. What is known, from receipts and source

- The target card's tuned door (`DAY58.md` tip arm, `--expert-bank-stages`, N=10, `pro-single-day52/smallfix/`), per
  window token: 92.3 host-hit demands (88.6 of them the door's prefetches, 3.8 demand copies), `inner_demand_ns`
  0.423 ms, the bank's `stage()` 0.333 ms of it (3.6 us per demand), `publish` 0.031 ms, the bank's retire side
  0.053 ms, `wait_ns` 0. The window gap to REF is 0.44 ms per token (`DAY51.md` section 4: 0.240 s against 0.226 s
  over 32 tokens). REF makes no lease at all.
- Per door prefetch of a host-resident block (`moe_cache.rs` `prefetch_banked`, `owner_proxy.rs`,
  `native.rs` `TracedDispatch`, `expert_dispatch.rs` `SlruExpertDispatch`, `residency.rs` `BankService`): the proxy's
  `host_resident` (a registry borrow, a map lookup, an SLRU lookup); the proxy's `demand`: `TracedDispatch` drains
  up to 32 fill completions, looks the record up in the SLRU twice, calls `SlruExpertDispatch::demand`, which
  validates again, clones the `BankId` into a one-id `BankBatch`, clones the request, and runs the whole ticket
  protocol for a record that is already cached: `stage` (the request's validation, the layout lookups, a
  `BTreeSet` of the ids, the read plan of an empty miss list, two wire encodings per id to size the ticket's
  metadata, a governor reservation, the ticket's bookkeeping maps), `progress`, `publish` (the output lease, the
  hotness, the SLRU hit); then the trace line; then `with_bytes` and, once the copy lands, `finish`:
  `finish_host_use`, `retire`, `acknowledge` (the governor release), `collect_evicted`.
- Nothing measures which of those parts carry the 3.6 us. Guessing the change from the list would be the shortcut the
  owner's order forbids.

## 1. Pre-registration: the CPU profile of one host-hit lease (an instrument; decides nothing)

**The profile** is one `#[ignore]` test in `crates/memra-engine/src/banked_residency/native.rs`,
`day61_profile::host_hit_lease_profile`, run on the local CPU under the 1200% cap
(`cargo test --release -p memra-engine --lib day61_profile -- --ignored --nocapture`), one thread, stderr (the host
trace) to `/dev/null`, stdout teed to `research/spill-c-20260919/cpu-day61/profile.log`. It builds the door's owner
stack as the installer does, from synthetic bytes: the pressure model's catalog shape (40 layers x 3 projections x
256 experts, 30720 records, one payload segment each, through `host_exps_catalog` as `map_host_exps` calls it, a
`PerRecord` catalog), the installer's governor dimensions, limits (`items` 1, `tickets` `BANKED_INFLIGHT + 1`), the
no-op `Heat`, an SLRU that holds every record, a `FileReader` over a temporary file, `TracedDispatch` with no fill
and the trace buffer, registered with `ExpertBankOwner`. Every record is demanded and finished once (the fill), then
it runs host-hit cycles in one fixed seeded order that routes 8 experts per layer as the model does:

- **P1, the door's prefetch sequence through the proxy**, per cycle: `host_resident`, `demand`, `with_bytes`
  (reads one byte), `finish`, each bracketed; 200000 cycles, 5 repeats, the median repeat's per-cycle ns for the
  whole cycle and each step.
- **P2, the same loop without brackets** (the brackets' own cost is P1 minus P2).
- **P3, the bank alone**: the same cycles on a second stack whose `SlruExpertDispatch` is called directly, with the
  bank's stage clock installed, so `stage`, `publish`, the retire side and `collect` are split per cycle.
- **P4, the parts inside `stage` that its clock does not split**, each timed alone over the same ids in the same
  order: the two wire encodings, a `BankId` clone, a catalog lookup, a request clone and validation, one governor
  reservation and its release.

**Reading.** Per cycle ns for every step of P1 to P4, and each part's share of P1's `demand` plus `finish`. The
parts named here decide what section 2 registers, before any change's code: a change targets the parts that carry
the cost, keeps every contract the protocol states (a lease pins its host bytes until `finish`, a record is
released only after every ticket that holds it retired, the SLRU and hotness see the same demands in the same
order, the same bytes reach the same slot), and gets its own cell on both cards.

These are this CPU's numbers (the local rig's), a profile for the design, not a card result.

## 1a. The profile's result (`cpu-day61/`, the local CPU, one thread under the cap)

Two runs, the second after adding three log-only parts to P4 (P4b: the SLRU lookup, the std hash of one `BankId`, the
owner stack's `host_resident` without the proxy); the profile decides nothing, so the addition moves no bound. Tree:
`4c9a7fe53` plus the profile module. Per-cycle medians of 5 repeats of 200000 host-hit cycles:

| part (ns per cycle) | run 1 | run 2 |
|---|---:|---:|
| P1 cycle through the proxy | 5687.5 | 5411.7 |
| P1 `host_resident` / `demand` / `with_bytes` / `finish` | 658.8 / 4263.3 / 253.7 / 490.3 | 598.1 / 4064.0 / 234.6 / 494.0 |
| P2 cycle unbracketed | 5371.1 | 5370.3 |
| P3 bank alone: `demand` / `finish` | 3676.0 / 443.6 | 3663.4 / 451.6 |
| P3 stage clock: `stage` / `publish` / retire side / `collect` | 2499.4 / 367.3 / 339.4 / 14.2 | 2487.8 / 364.7 / 346.6 / 13.2 |
| P4 two wire encodings (with their catalog lookup) | 1108.3 | 1061.3 |
| P4 catalog lookup / `ids` map get / `BankId` clone into a vec / one-id `BTreeSet` | 190.9 / 57.2 / 101.6 / 88.5 | 210.7 / 56.7 / 87.6 / 87.2 |
| P4 request clone and validate / governor reserve and release | 20.4 / 320.3 | 19.6 / 317.3 |
| P4b SLRU lookup / std hash of a `BankId` / `host_resident` without the proxy | | 92.3 / 39.7 / 182.2 |

Read against section 0: a host-hit prefetch costs about 5.4 us of CPU here. The bank's `stage` is 2.5 us of it, and
the largest single part of `stage` is the two wire encodings that size the ticket's metadata (about 0.85 us without
their catalog lookup), a value that is a pure function of the immutable catalog entry and is recomputed on every
ticket. The rest is spread: the same record looked up several times in the catalog (twice in `stage`, once more in
`demand`'s validation), the host cache twice, the SLRU up to five times (the prefetch's `host_resident`, twice in
`TracedDispatch::demand`, twice in `publish`), the registry entered twice where the prefetch's two calls run back to
back, and the governor's per-ticket reservation (0.32 us, the protocol's own charge). In a tight loop the lookups
read cheaper than inside the cycle (182 ns against the proxy call's 598 ns), so the P1 brackets are the reading.

One more term the CPU profile cannot show, from the target card's tip receipts: the cache side of the door's stage
clock spends `retire_ns` 0.118 ms per window token. `admit_banked` runs `retire_banked` on every admission (471 per
token, 378.7 of them GPU hits that take no lease), and each call with a lease in flight queries the front copy event
and finishes, through the proxy, every lease whose copy landed. REF pays a sibling term: `reap_copy_sources` runs on
every dispatch and prefetch and queries the copy event of every consumed prefetch still in its keepalive list (its
pinned sources carry a keepalive, `model.rs` `expert_source`). So this term is not one REF lacks; it is work the door
does on admissions that take no lease, where the bound it serves is not in play.

## 2. Pre-registration: I11 and I12, the lease protocol's repeated work removed (before any of their code)

Both keep every contract the protocol states: a lease pins its host bytes until `finish`; a record's backing is
released only after every ticket holding it retired; the SLRU and the hotness see the same demands in the same order;
the same bytes reach the same GPU slot; the in-flight bound holds at every lease taken; the registry refuses what it
refused. No flag, no second path: each change removes work the one protocol repeats.

**I11, the host side** (memra-tier bank and the door's owner stack), one commit per change, in this order:
1. The catalog computes each record's ticket metadata allowance (the two encodings plus 1024, today recomputed in
   `stage` and in `resident_charge_bytes`) once, at `Catalog::new`, and `stage`, `resident_charge_bytes` and the
   fill's admission read it. Construction refuses exactly what it refused.
2. `stage` reads each id's catalog entry once (its layout for the logical bytes and its allowance) and the host cache
   once (a cached lease or a missing record), where it read each twice.
3. `publish` asks the SLRU once per id on a demand ticket (`hit` answers whether the record is resident).
4. `SlruExpertDispatch::demand` looks its id up once.
5. `TracedDispatch::demand` reuses the slot its pre-demand lookup found for a host hit (a hit does not move a record's
   host slot); a miss still reads the slot after the demand.
6. The prefetch's residency check, its GPU slot step and its lease run in one owner call,
   `ExpertBankProxy::demand_if_resident`, in the order the three calls ran: not resident, no slot step and no lease;
   no slot, no lease; a refused lease hands the slot back to be released. The dispatch clock's door prefetch then
   counts residency and demand together in `pf_demand_ns` (`pf_resident_ns` reads 0 for the door from I11 on; the
   slot step keeps `pf_reserve_ns`); its `docs/FLAGS.md` text says so in the same commit.

**I12, the cache side** (`moe_cache.rs`): `retire_banked` runs where a lease is taken (the miss path before `demand`,
the prefetch before its owner call) and at teardown, not on every admission. A GPU hit and a prefetched block's
consumption take no lease, so the bound (`BANKED_INFLIGHT`, waited on at a lease point) is unchanged; a finished
copy's lease is finished at the next lease point instead of at the next admission.

**CPU gates, per change** (`cargo test` under the cap): the memra-tier bank suite and the engine lib suite green at
every commit, clippy `-D warnings`, and one pin per change: (1) the memoized allowance equals the recomputed one for
every record of the profile's catalog and of the bank fixtures; (2) and (3) the bank suite's ticket, publish and SLRU
tests unchanged and green; (4) and (5) a test that, over the profile's routed cycles, the reused slot equals a fresh
lookup after every demand and the trace text equals the text the day-48 format builds from a fresh lookup; (6) the
day-50 census updated to the one call, plus owner-proxy tests of each outcome (not resident: gate not called; gate
declines: no lease; refused: gate value returned, registry pending unchanged); (I12) a census that `admit_banked`'s hit
and consumption arms do not call `retire_banked` and that every `bank.demand` and `demand_if_resident` call site is
preceded by one.

**The CPU ladder** (the day-61 profile at each commit, same harness): P1 or, from change 6, P5 (the prefetch sequence
through `demand_if_resident`) per cycle, one row per change, teed to `cpu-day61/ladder-<n>.log`. It shows each
change's CPU share; it decides nothing on a card.

**The card cell `i11`** (both cards; `day61-cell.sh`, reader `day61-read.py`, both written before any cell). Day 18's
pressure shape as in `DAY60.md` (`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986`, prompt `55 88 13`), one
collector hold, 250 ms telemetry, the 1200% cap. Arms: REF (`MEMRA_MOE_PREFETCH=1`, the `c60` binary), ON (the door
at the full-bank host budget, the `c60` binary), I11 (the same door, the binary at I11's last commit), I12 (the binary
at I12's commit). Order 1 (REF, ON, I11, I12) x 5, order 2 reversed x 5, 40 runs.

- **Integrity** (any failure voids the card's reading): every run exit 0 and `MATCH`; one `tokens:` tape across all
  four arms; every door run reports its fill complete before decode and `physical_reads=0`; the host demand
  sequence (`[expert-host-slru]` lines with the slot number removed, since the threaded fill's completion order
  assigns host slots) identical across every door run of ON, I11 and I12 (the target card's 30 day-58 runs of three
  binaries show one such sequence).
- **Readings** (`noise` the larger IQR of the two arms compared; gen-only decode the primary reading, the steady
  window beside it): for each step, I11 against ON and I12 against I11: `improves` iff the newer median is below the
  older by more than `noise` in order 1, order 2 and pooled; `regresses` iff above by more than `noise` in both orders;
  `flat` otherwise.
- **The door against REF**: the door arm read is I12, or I11 if I12 `regresses` against I11: `beats` iff its median
  is below REF's by more than `noise` in both orders and pooled; `loses` iff above by more than `noise` in both orders;
  `matches` otherwise. Recorded plainly whatever it reads; the owner reads it against 2026-10-04.
- **What follows.** A step that `regresses` on either card is reverted by this lane in the change that records it,
  with its receipt; `flat` or `improves` keeps it (the CPU ladder shows the work it removes). If the door still
  `loses` on the target card, the next improvement is registered from the attribution (`DAY60.md`) and this cell's
  readings, never by moving a bound here.
