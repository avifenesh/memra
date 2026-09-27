# WP-C day 89 (2026-09-27): OWED C2, the door's promotion, option 1's first cut: I23, the budget governor's per-ticket map work, registered before any code

Lead: "start registering the cuts for option 1 (the governor, then the retire side), text only, so the owner's answer
does not wait on registration. No code lands until the owner answers." `DAY88.md` section 6b's option 1 closes the
285K class's 2.5 ms gen-only gap with further cuts of the door's host-hit protocol, then runs one `promo` sitting on
both classes. Tree at start: `64326c89c`. This file is text only, and its code waits for the owner's answer. The
retire side is `DAY90.md`.

## 0. Where the governor's time is, from the split and the source

In situ at I22 on the local host (`rtx5090-day85/split22/`, the queue v19 split of `DAY85.md` section 3b; figures
recomputed by `day89-cpu/section0.py` through `day83-read.py`'s own terms, `day89-cpu/section0.log`), per generated
token:
- `bank_stage_charge` is 43.2 us: the ticket's queue reserve plus one record reserve per host miss (`stage` in
  `bank/residency.rs`).
- `bank_ack_release` is 14.6 us: the queue charge's release at the acknowledgement. The reader computes this term but
  does not print it; it sits inside `bank_ack` (29.4).
- Together they make 57.8 us, the "about 58" in `DAY88.md` section 6b, over 2443 tickets in the generate phase: 76 per
  token, 0.76 us per ticket.

What a ticket does in `Governor` (`tier/governor.rs`) and `LeaseIssuer` (`contracts.rs`):
- The reserve runs the queue's priority check, then the deadline, the fit and the checked add over the budget's
  vectors. The issue clones the request's budget (three heap vectors: device, peer, replicas), allocates the charge
  record (`Arc<Mutex<ChargeRecord>>`) and inserts it into the issuer's `records` map. Then comes the governor's
  `charged_tenants` insert (lease id to tenant) and the per-tenant count.
- The release looks the lease up in `records` and removes it after the capability checks, applies the checked
  subtract, removes the lease from `charged_tenants`, decrements the per-tenant count, and runs `prune_fairness`, which
  returns at once while the fairness history stays within the queue limit.

`records` and `charged_tenants` hold one entry per live charge, and every cached record keeps its charge
(`BankLease::from_backend` in `fill`, `stage`'s record reserves). Under the qualified door, with the whole bank host
resident, that is 30,720 records plus the open tickets. So each ticket makes two inserts and two removals in maps of
that size, both on the standard library's SipHash, with the cache misses such maps bring between two launches.

## 1. Pre-registration: I23

**The change** (`memra-tier`: `contracts.rs`, `tier/governor.rs`; no engine code).
1. **The tenant rides with the charge.** The governor's issue records the request's tenant in the charge record it
   already allocates, and the release reads it back there after the issuer's capability checks. `charged_tenants`
   goes. The lease does not grow, and leases from the other issuers (the owner proxy's, the contract tests') carry no
   tenant.
2. **The issuer's live-charge map goes.** A lease's liveness is its record's own state. `record()` still answers
   `ForeignLease` for another issuer's lease and `AlreadyReleased` for a released one (state `Released`, the state
   `release` sets on exactly the records it removes from the map today). Then `mark`, `release` and their refusals
   (`Busy` for a pinned or unretired charge) read the same record as before.
   - One divergence, recorded here and not hidden: a poisoned record of an already released lease would answer
     `Quarantined` where the map answered `AlreadyReleased`. No path panics while holding a record's lock (every lock
     section is a state read or write, a checked add, or a live pin's decrement), so no reachable sequence tells them
     apart.
   - A lease dropped without release frees its record now instead of leaking it in the map. Its bytes stay charged in
     `used` either way ("Dropping it does not credit quota").
3. **Nothing else.**
   - These stay as they are: the queue's priority and deadline checks, the fit against capacity and mandatory
     headroom, the budget clone in the lease (the release subtracts it), the pins and states, the per-tenant count,
     `last_served`, `dirty`, and every refusal in its order.
   - The remaining maps stay on SipHash. They are keyed by tenant digests from served requests (`memra-kv`), bounded
     by the tenant count, and cost two hashes per ticket. The Fx hasher's charter (`bank::fx`) covers catalog-record
     keys, so this lane does not widen it for a few microseconds per token.

**Its reach, stated.** `Governor` and `LeaseIssuer` are shared: `memra-kv`'s host prefix tier (the contracts door,
lanes A and B) and the tier's object store build the same governor, and the owner proxy issues its own charges. The
change keeps every reachable answer, so their programs are unchanged. Their suites are part of the gates below, and
the lead's GPU battery covers their GPU cells.

**CPU gates before any card** (`day89-cpu/`):
- **The trace fixture, recorded first, at `64326c89c`, before any I23 line.** A seeded sequence runs against one
  governor: reserves, enqueues, dispatches, expiries, marks, pins, releases (foreign, double, pinned, unretired among
  them) and cancels, over several tenants and priorities. Every outcome, lease id, `used()` and the fairness history
  go to a fixture. After the change the same sequence must reproduce it exactly, and the owner proxy's issuer gets the
  same treatment.
- The existing suites: the governor's and the issuer's unit tests, the tier suites, `memra-kv`'s tiered tests, the
  engine library, clippy (`-D warnings`, all targets), fmt and `rc-scan.py --live`.
- The day-61 profile at I22 and I23 (P1, P9, P10) in one window.
- The local RTX 5090's in-situ split, deciding nothing: `DAY83.md`'s cell shape, arms `i22s` and `i23s`, reads whether
  `bank_stage_charge` and `bank_ack_release` fell. The local GPU check reads I22's tape and host demand sequence.

**No card cell of its own.** I23 and I24 (`DAY90.md`) go to the target cards together, as the promoted binary's
`promo` sitting on both classes under `DAY88.md` section 5's rule, unchanged: `naked` neither regresses against the
door as qualified nor loses to its rollback. `promo-res` follows section 6a. This registration claims no size for the
cut; the split measures it.

## 2. The owner's rulings (2026-09-27, via the lead), and the gates and split as they run, registered before any code

**The rulings.**
- On `DAY88.md` section 6b: "option 1 only. Close the 285K gap with the cuts, then rerun `promo` on both classes. The
  prefetch default does not land early." I23 and I24 are built now as registered.
- On process, verbatim: "we should stop overcomplicating our local CI, we are making all our progress too slow. we
  should do deeper measurement where it is relevant."

**The gates, narrowed by that ruling before any result.** Section 1's lists and `DAY90.md`'s are replaced by the
following; GitHub CI runs the rest.
- **The equivalence proofs stay**, because they are what makes each cut the same program:
  - I23's trace fixture: governor and issuer, recorded at the pre-change tree (crates equal to `0155bc69f`) before any
    I23 line;
  - I24's fixture: proxy and bank, recorded at I23 before any I24 line, with its failure-retention case.
- **The affected crates' tests and clippy**: `memra-tier` (tests, clippy `-D warnings`, all targets) and `memra-kv`'s
  lib tests (its tiered tier builds the governor). I24 adds `memra-engine`'s `banked_residency` tests and its clippy,
  for `TracedDispatch`. Plus fmt.
- **Dropped:** the day-61 CPU profile, the full engine library, the day44 and day50 censuses beyond the
  `banked_residency` tests, and `rc-scan.py` on scripts this work does not touch.

**The measurement that sizes the cuts: one local queue (v21, `rtx5090-day89/`).**
- **The check** (`check/`): `run-gen-p88` and `run-gen-i24` with the door and no clocks, in the order p88, i24, i24,
  p88. `day85-cpu/gpu-check-read.py` reads it: `MATCH`, one tape, one host demand sequence.
- **The split** (`split24/`):
  - arms `p88s`, `i23s` and `i24s` run both clocks (`--moe-dispatch-clock --expert-bank-stages`);
  - arms `p88d`, `i23d` and `i24d` run the dispatch clock alone (`pf_retire` without the stage clock's settle);
  - order 1 is p88, i23, i24 for each clock pair, x 5; order 2 is reversed, x 5; 60 runs, the argv of queue v19;
  - `p88` is the direct parent of I23 (I22's door plus phase 1; `DAY88.md` read `naked` against `q22` flat), so it
    replaces section 1's `i22s`.
  - `day83-read.py --check --change p88s,i23s` and `--change i23s,i24s` read the leaves, and the `d` arms'
    `pf_retire` is read beside them.
- **The sizing rule, registered now.** 6b sized the gap in the 285K class's door-only work: roughly 100 to 150 us per
  generated token must go. This host's door-only work at I22 is about 1.55 times the 285K's (346 against 223 us per
  token, `DAY85.md` sections 3 and 3b), so 100 us there is about 155 us here.
  - The cuts are **short** if the door-only leaves (`day83-read.py`'s LEAVES), summed, fall by less than 155 us per
    generated token from `p88s` to `i24s`. Then the next candidates register before any card (`DAY90.md`: the proxy's
    registry entry, `validated`, the host cache).
  - At 155 or more, NEED TARGET CARD for the one `promo` sitting on both classes.
