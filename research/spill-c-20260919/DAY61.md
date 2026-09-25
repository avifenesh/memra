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
