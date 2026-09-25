# WP-C day 63 (2026-09-25): the owner demand on the prefetch path, split and tuned (lead owed item 2, next step), registered before code

Lead, after the days 59 to 61 sitting: "Continue with DAY63 (the owner demand on the prefetch path: the governor's
per-ticket reservation, the lease clones, publish plus finish), same rules as before: pre-register before code, CPU
ladder then card cells, no bound moved after a result." Tree at start: `e6217375e` (being integrated as integ61).

## 0. Where the demand stands (receipts)

- Target card, BOX12 (`DAY60.md` section 2): the door's prefetch path costs 0.607 ms per window token more than REF's,
  the owner demand 0.464 of it (`pf_demand_ns`, 5.2 us per issued prefetch) on the `c60` door; I11 then took the door
  from 0.240 to 0.234 s per 32-token window and from 0.276 to 0.267 s gen-only, and it still `loses` to REF by 0.25 ms
  per window token and 0.34 per generated token (`DAY61.md` section 3).
- Local CPU, the day-61 profile at I11's last change (`cpu-day61/ladder-5.log`): 3620 ns per host-hit prefetch cycle;
  the bank's `stage` 1164.5, `publish` 296.0, the retire side 334.9 (`finish_host_use` + `retire` + `acknowledge`),
  `collect` 13.0; the governor's reservation and release 317 ns alone; the rest of `SlruExpertDispatch::demand`, the
  proxy and `TracedDispatch` about 1.2 us between them.
- From source, per host-hit ticket: `stage` reserves the ticket's queue charge from the governor (the request's
  validation, the fit test, two `TierBudget` temporaries, the issuer's `Arc<Mutex>` record and its map entries, the
  tenant maps) and `acknowledge` releases it; a `BankLease` clone copies its `BankId` and its whole `RecordLayout`
  (the segment and requirement vectors and the tensor names), and a host-hit cycle clones a lease at least three
  times (the ticket's record map, `publish`'s output vector, the proxy's `finish` rebuilding the `ExpertDemand`);
  `finish` looks the ticket up in the pending map four times (`finish_host_use`, `retire`, twice in `acknowledge`) and
  once more to remove it.

## 1. Pre-registration: the split (an instrument; decides nothing)

**The instrument.** The bank's stage clock (`BankStageTimes`, installed only by `with_stage_clock`, the
`--expert-bank-stages` diagnostic) gains fields appended to its line, every existing field unchanged in meaning and
order: inside `stage`, `stage_lookup_ns` (the per-id catalog loop), `stage_cache_ns` (the unique set and the host cache
pass), `stage_charge_ns` (the governor reservations); inside `publish`, `publish_output_ns` (the output leases) and
`publish_policy_ns` (the hotness and the SLRU); on the retire side, `host_use_ns`, `retire_only_ns` and `ack_ns`, one per
call, their sum what `retire_ns` has always counted, and `ack_release_ns` (the governor release inside
`acknowledge`). A census pins that the clock's new writes are counter additions only. Without the stage clock nothing
runs differently.

**The profile additions** (the day-61 `#[ignore]` profile, same harness, same catalog and routed cycles): P3 prints the
new fields per cycle; P7 times, over the same routed ids, one `BankLease` clone and its drop, taken from a cached lease
(`BankedResidency::resident`).

**Reading.** Per cycle ns of every new field, and each named part's share of P3's `demand` plus `finish`. The parts
decide what section 2 registers, before any change's code. The same contracts bind as `DAY61.md` section 2: a lease
pins its host bytes until `finish`; a record is released only after every ticket holding it retired; the governor
charges and releases exactly what it did (the same dimensions, the same refusals, in the same order); the SLRU and
the hotness see the same demands in the same order; the same bytes reach the same slot. These are the local CPU's
numbers, a profile for the design, not a card result.

## 1a. The split's result (`cpu-day63/split.log`, `split-run2.log`; commit `ad73d242c`)

Two runs, one thread under the cap, the median of 5 repeats of 200000 host-hit cycles. Another tenant's CPU work
shared the rig in both (load average 3.8 at the second run's start), so P2's absolute cycle reads 3949.8 and 3800.5 ns
against ladder-5's 3619.9 on the same program; the split's shares are the reading, not the absolute numbers.

| part (ns per cycle, P3, the bank alone with its clock) | run 1 | run 2 |
|---|---:|---:|
| `stage` whole | 1288.8 | 1263.1 |
| `stage_lookup` (the per-id catalog loop) | 158.3 | 140.4 |
| `stage_cache` (the unique set and the host cache pass) | 703.9 | 696.2 |
| `stage_charge` (the governor's queue reservation) | 271.0 | 274.6 |
| `publish` whole / `publish_output` / `publish_policy` | 372.2 / 68.8 / 241.7 | 370.1 / 68.3 / 241.1 |
| retire side whole / `host_use` / `retire_only` / `ack` (`ack_release` inside it) | 352.0 / 32.3 / 31.9 / 287.8 (152.5) | 349.9 / 31.7 / 31.5 / 286.7 (152.3) |
| P7, one lease clone and its drop | 68.7 | 74.5 |

Read against the lead's list: the governor's per-ticket reservation and release is 425 ns a cycle (271 + 152); the
retire side's pending-map work beside that release is about 135 ns; `publish` is mostly the SLRU (the demand ticket's
`hit`, 241 ns), its output lease 68; a lease clone is about 70 ns, three a cycle. The largest single part is not on
the list: `stage_cache`, about 700 ns, a one-record set and one lookup in the host cache's `BTreeMap<BankId, _>` of
30720 records (each comparison walks two 32-byte digests and a tensor name). The per-id lookups keyed by the full
`BankId` recur through the cycle: the catalog (twice), the host cache, the SLRU (three times).

## 2. Pre-registration: I13, the per-ticket and per-lease work tuned (before any of its code)

Every change keeps the contracts of section 1 exactly: the same charges in the same dimensions, the same refusals in
the same order, the same state transitions, the same bytes. One commit per change, in this order:

1. **The governor without temporaries.** `reserve` tests the fit without collecting the three value vectors,
   issues the lease, then adds the request into `used` in place (checked before any component changes); `release`
   subtracts in place the same way; `prune_fairness` returns before building its sets when `last_served` holds no
   more tenants than `queue_limit` (then no idle tenant can exceed the limit, so it would remove nothing).
2. **One body per lease.** `BankLease`'s fields (id, layout, class, charge, backing) move into one `Rc`; a clone is a
   count increment, every accessor returns what it returned.
3. **The retire side in one lookup.** `BankService::finish_ticket` runs `finish_host_use`, `retire` and `acknowledge`
   on one pending entry with the same checks, flags, charge releases and removals, and the same errors at the same
   points (a ticket not yet retired refuses `NotReady` after marking host use done, as the three calls did);
   `SlruExpertDispatch::finish` calls it. The three public calls stay for every other caller. The stage clock keeps
   `host_use_ns`, `retire_only_ns`, `ack_ns` and `ack_release_ns` as brackets inside it and `retire_ns` as their sum.
4. **The SLRU's maps hashed deterministically and cheaply.** `SlruPolicy`'s `table` and `reserved` take an in-crate
   Fx-style hasher (the rotate, xor, multiply step rustc uses) instead of the randomly keyed SipHash; neither map is
   iterated anywhere, so no order is observable, and the keys are catalog records, not untrusted input.

**CPU gates, per change:** the memra-tier suite and the engine lib suite green, clippy `-D warnings`, fmt; and one pin
per change: (1) a differential test of the in-place add and subtract against `TierBudget::checked_add` and
`checked_sub` over randomized budgets (the same values, the same errors, nothing changed on an error), and a governor
test that pruning still evicts idle tenants past the limit; (2) a clone shares its body (`Rc::ptr_eq`) and every
accessor reads the same; (3) `finish_ticket` against the three calls on twin banks over the refusal cases (unknown
ticket, not published, not retired) and the success path: the same results and the same governor totals; (4) the SLRU
oracle tests (`slru_oracle`, day 43's randomized oracle) unchanged and green.

**The CPU ladder:** the day-61 profile at each commit (`cpu-day63/ladder-<n>.log`), P2 and P3's split per row, read
beside `split.log`. The rig's other load is recorded with each row (`uptime`); a row read under a different load is
marked, not compared as if equal.

**The card cell** is registered in section 4 before its script is written, with I13 against I12 and against REF, and
any further improvement that section 3 registers from this ladder.
