# WP-C day 64 (2026-09-25): the door's structural improvements, registered before code (lead owed item 2, next step)

Lead, after day 63's card cell: "Next per your ledger: read DAY63 into its record, then the structural improvement
registered from the card's split before any code. Same rules as before." Tree at start: `13e14fa6b` (main `d6515742f`,
#725, merged in) plus day 63's card record.

## 0. What the card's split says (`DAY63.md` section 4, the target card, I13, per 32-token window)

The door still loses to REF by 0.19 ms per window token (0.28 per generated token). Its owner side spends 0.216 ms per
window token on 92.3 host-hit demands (88.6 of them prefetches). Two kinds of work remain, by the split:

- **Work per record, keyed by the full `BankId`.** `stage_cache` 0.081 ms (0.87 us a demand: a one-record set and one
  walk of the host cache's `BTreeMap<BankId, BankLease>` over 30720 records), `stage_lookup` 0.015 (the catalog's
  `BTreeMap`), `publish_policy` 0.013 (the SLRU's hit), and outside the bank clock the validation's second catalog
  walk and the residency check's SLRU lookup. On the local CPU (`cpu-day63/ladder-4.log`) the same parts read 478 ns
  (`stage_cache`), 123 (`stage_lookup`) and 207 (`publish_policy`) of a 2940 ns cycle.
- **Work per ticket and per lease.** The governor's charge (0.031) and release (inside the retire side's 0.036), the
  rest of `stage` (0.022) and `publish`, the dispatch adapter and proxy around each demand (about 0.045), and on the
  cache side the retire of each finished lease through the proxy (`retire_ns` 0.075 with the stage clock on). Every
  prefetched expert takes three tickets, three leases and three retires, one per block (gate, up, down), issued back
  to back in `hybrid_forward.rs` `moe_prefetch_expert`.

Neither is the GPU side (`DAY60.md`: `cpu_side`). Both are registered below, one improvement each, in this order.

## 1. Pre-registration: I14, the record lookups hashed (before any of its code)

The same contracts as `DAY61.md` section 2 and `DAY63.md` section 1. One commit per change:

1. **The catalog indexed by hash.** `Catalog` keeps its records in `BankId` order in a vector (every ordered iteration,
   `max_id`'s and the source validation's, reads the same order) and finds a record through an Fx-hashed index from
   `BankId` to its position; `record` and `entry` refuse exactly as they did (`validate`, `NotFound`, `MaskedId`).
   Construction refuses what it refused, in the same order. The Fx hasher moves from `slru.rs` into its own module
   for the bank's three users.
2. **The host cache indexed by hash.** `CacheIndex.map` becomes an Fx-hashed map. Its one ordered use, `trim`'s victim
   (the lowest hotness score, and among equal scores the first in `BankId` order), states the tie-break explicitly
   (`min` over `(score, id)`), so the victim is the same record.

**CPU gates:** the memra-tier suite and the engine lib suite green at each commit, clippy `-D warnings`, fmt; pins:
(1) over the bank fixtures and the door-shaped 30720-record catalog, every id's `record` and `entry` and every refusal
equal a `BTreeMap` built from the same entries, and the ordered iterations yield the same sequence; (2) `trim`'s victim
equals a `BTreeMap` reference's first minimum over randomized scores with ties, and the day-4 and day-43 SLRU and
bank traces stay green.

## 2. Pre-registration: I15, one ticket per prefetched expert (before any of its code)

**The change.** Under the door, `moe_prefetch_expert`'s three blocks go through one cache call,
`MoeSlotCache::prefetch_expert`, and the legacy cache keeps its per-block calls unchanged (REF's program does not
move). For the door the call does, in the per-block order it replaces: for each block (gate, up, down) the table and
pending skips, the validate memo, the residency check and the GPU slot reservation with the same `keep` set (so the
same victims are evicted in the same order); then ONE owner call leases every block that passed, as one ticket with
up to three records (`ExpertBankProxy::demand_many`, `SlruExpertDispatch::demand_many`, `TracedDispatch` writing one
trace line per record in block order, the bank's limits admitting three items per ticket); then each lease's bytes
are staged on the copy stream in block order, each block pending with its own copy event. The ticket's leases are
finished together, once, after the last of its blocks is consumed and that block's copy event (the last copy on the
stream, so every earlier one has landed) completes; `retire_all_banked` finishes an unconsumed group once. The
in-flight bound counts leases as before (a group counts its blocks).

**What it keeps, stated:** the same GPU slot sequence, the same copies in the same order, the same host SLRU hits in
the same order (a ticket publishes its ids in order), the same trace lines in the same order, a lease never finished
before every copy that reads it is proven complete, an unknown stream keeps the whole ticket open (fail closed),
every refusal of a block before the owner call leaves that block to the demand path as today. The demand path (a
GPU miss) keeps one record per ticket.

**CPU gates:** the suites, clippy and fmt at each commit; pins: owner-proxy tests of `demand_many` (identity per
record, a lying bank refused, the pending bound per ticket, `with_bytes` by index, `finish` once); a census that the
legacy path's per-block `prefetch_source` calls are unchanged and that the door's grouped lease is finished only from
`retire_banked` and `retire_all_banked`; and the host-hit profile gains P8, the grouped prefetch of three records
through the proxy, beside P2's three single ones.

**The CPU ladder** for I14 and I15: the day-61 profile at each commit (`cpu-day64/ladder-<n>.log`) and one window
re-reading every row in both orders at the end (`cpu-day64/window.log`), as day 63's.

## 3. Pre-registration: the card cell `i15` (both cards; before its script)

Binaries: `c60=da649107c` (REF), `i13=c9379c051`, `i14` and `i15` (each improvement's last commit, named in section 3a
before any cell). The cell (`day64-cell.sh`, reader `day64-read.py`): day 18's pressure shape as days 61 and 63, one
collector hold. Arms: REF, I13, I14, I15, and I15S (I15 with `--expert-bank-stages`, read for its split only). Order 1
(REF, I13, I14, I15, I15S) x 5, order 2 reversed x 5, 50 runs.

- **Integrity** as `DAY63.md` section 3 (every run exit 0 and `MATCH`; one tape across every arm; every door run's
  fill complete and `physical_reads=0`; one host demand sequence without slot numbers across every door run; I15S's
  stage lines carry the day-63 fields).
- **Readings:** I14 against I13 and I15 against I14, `improves`, `regresses` or `flat` as `DAY61.md` section 2 defines
  them, gen-only decode the primary reading and the window beside it.
- **The door against REF:** the door arm is I15, or I14 if I15 `regresses`, or I13 if I14 also `regresses`: `beats`,
  `matches` or `loses`, recorded plainly.
- **What follows.** A step that `regresses` on either card is reverted with its receipt; `flat` or `improves` stays.
