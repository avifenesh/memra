# WP-C day 90 (2026-09-27): OWED C2, the door's promotion, option 1's second cut: I24, the retire side's finish, registered before any code

The lead's order is in `DAY89.md`: register the cuts for option 1 (the governor, then the retire side), text only, and
land no code until the owner answers. Tree at start: `64326c89c`. I24 builds on I23 and changes nothing I23 touches:
the governor's release inside the acknowledgement is I23's.

## 0. Where the retire side's time is

**On the target classes, without the stage clock.** `--moe-dispatch-clock` brackets the prefetch path's
`retire_banked` calls (`pf_retire`), and the stage clock's work stays out of it. At I20 the generate phase reads 52.4
us per generated token on the 285K class (BOX39, `box39-i20c`) and 44.9 on the 9950X class (BOX40, `box40-i20c`),
0.28 and 0.24 us per prefetched block (`day83-cpu/section0.log`). At I22 the window reads 25.9 and 21.0 us per window
token (`DAY85.md` section 3, `I22C`). These are the section 6b retire figures that went to the owner, and they stand.

**On the local host, with both clocks** (`rtx5090-day85/split22/`, `day89-cpu/section0.log`, I22, per generated
token):
- `retire_outer` is 57.4 us: the engine's retire outside the bank's finish, with no wait.
- `bank_retire` is 33.8 us: `bank_ack` 29.4, of which the governor's release is 14.6 (I23), `bank_host_use` 3.4 and
  `bank_retire_only` 0.9. `bank_collect` is 1.2.
- Under `--expert-bank-stages`, `retire_banked` also runs `settle_copy_timings`, which reads each landed copy's
  timing events (a query and an elapsed-time read per copy). Those events exist only under the stage clock, so the
  naked binary runs none of it, and `retire_outer` overstates the naked walk by that much. Section 1's local
  measurement reads the retire on a clock without it.

**What one finish does** (from the source):
- **The walk** (`moe_cache.rs` `retire_banked`) makes one `cuEventQuery` per in-flight entry it pops, plus one on an
  incomplete front per call.
- **The proxy** (`ExpertBankProxy::finish` and `finish_group`) runs, in order:
  - the owner-thread check, the registry's thread-local borrow and its owner lookup;
  - the pending lookup;
  - the identity check (`require`, `require_group`), which recomputes the lease's or the group's identity from each
    lease's record id and artifact;
  - a clone of the pending lease (a single) or of its lease vector (a group: one heap vector plus one reference-count
    increment per lease);
  - the adapter's `finish` or `finish_many`: the bank's `finish_ticket`, then the clone's drop, then `collect_evicted`;
  - the registry's removal, which drops the original.
- **The bank's ticket map** (`BankService::pending`, `HashMap<TransferTicket, Pending>` on SipHash, keyed by the
  bank's own minted tickets, at most `limits.tickets` entries) takes at least five operations per ticket: the insert
  at `stage`, one lookup per `progress` call, the lookups at `publish` and `finish_ticket`, and the removal at the
  acknowledgement.

## 1. Pre-registration: I24

**The change** (`memra-tier`: `bank/expert_dispatch.rs`, `bank/owner_proxy.rs`, `bank/residency.rs`, `bank/fx.rs`;
`memra-engine`: `TracedDispatch` in `banked_residency/native.rs`; `moe_cache.rs` untouched, so its source pin
stands).
1. **Finish by reference.**
   - `ExpertDispatchBank::finish` and `finish_many` take `&ExpertDemand` and `&ExpertDemands`, and the proxy hands them
     the registry's own entry, with no clone. The registry removes and drops its entry after a successful finish, as
     now, and on a failure the entry is untouched, so the retention the clone existed for holds by construction.
   - The aliases a finish can see are the same. Today the clone drops before `collect_evicted` while the registry's
     original lives through it. After the change only the original exists, and it lives through it too.
   - Release decisions read pins, views and pending references, never an alias count
     (`borrowed_view_and_alias_survive_busy_retirement`).
   - The trait's three implementations change together: the adapter, the engine's `TracedDispatch` and the proxy's
     test bank.
2. **The identity memo.** The registry entry keeps the identity its token was minted from, the same
   `identity`/`group_identity` value `demand` and `demand_many` computed and checked. `finish` and `finish_group`
   compare the token against it. It is the same comparison with the same `ForeignLease` refusal, without walking each
   lease again, and it is exact because a pending entry's demands never change while pending.
3. **The bank's ticket map on the Fx hasher.**
   - `pending` becomes an `FxMap`. Its keys are tickets the bank mints (issuer, sequence, epochs), not untrusted input,
     and the `bank::fx` charter extends to them in its doc comment.
   - Its readers are order-free: `is_empty`, `len`, lookups, `values().any(..)`, and `tickets()`, whose callers use
     the result as a set (on SipHash its order already varies per process).
4. **Nothing else.** These stay as they are: the walk, its event queries, its FIFO and its stop at the first incomplete
   entry; the in-flight bound; `finish_ticket`'s order (host use, retire, acknowledge); `collect_evicted`; every
   refusal in its order.

**Not in I24, and why.** The walk's event queries may be the largest retire piece, but nothing counts them today.
Cutting them exactly means keeping the FIFO's stop rule, because the retire timing sets when charges and pins release,
and so which host records can be evicted under a budget smaller than the bank. They get measured first: a log-only
count and time of the walk's queries under `--moe-dispatch-clock`. That edits `moe_cache.rs`, so the tier's day-4
source pin needs a re-pin as its own reviewed change. The cut registers on its own (I25) only if the count times the
cost is material.

**CPU gates before any card** (`day90-cpu/`):
- Retention, finish by reference: a finish that fails partway leaves the registry entry, and a retry finishes it once.
  A foreign, unknown or mismatched token refuses as before. This is proven at `64326c89c` and after the change.
- A twin-bank equivalence, as `tests/bank/day85.rs` does:
  - an I23 bank and an I24 bank take the same seeded demand, finish, cancel and failure sequence;
  - every outcome must be identical: the host demand sequence, `cached_records`, `owned_leases`, `tickets()` as a set,
    `used()`, and every refusal.
- The existing suites: the owner proxy's and the adapter's unit tests, the tier suites (day 43, 47, 64, 84, 85 among
  them), `memra-kv`'s tiered tests, the engine library, the day44 and day50 source censuses, clippy (`-D warnings`, all
  targets), fmt and `rc-scan.py --live`.
- The day-61 profile at I23 and I24 in one window.
- The local RTX 5090, deciding nothing:
  - the split in `DAY83.md`'s shape, arms `i23s` and `i24s`, for the leaves;
  - two arms with the dispatch clock alone, `i23d` and `i24d`, for `pf_retire` without the stage clock's settle;
  - the local GPU check, reading I22's tape and host demand sequence.

**The card sitting.** I23 and I24 go to the target cards together as one `promo` sitting on both classes under
`DAY88.md` section 5's rule, unchanged, with `promo-res` under section 6a. If the split shows the two cuts short of the
100 to 150 us per token section 6b sized, the next candidates register in turn before any sitting: the proxy's
registry entry, the adapter's `validated`, then the host cache. This registration claims no size.
