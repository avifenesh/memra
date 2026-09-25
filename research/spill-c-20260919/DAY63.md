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
