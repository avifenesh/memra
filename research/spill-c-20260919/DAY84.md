# WP-C day 84 (2026-09-26): OWED C11, I21, the lease path's residency read by record position, before any code

`DAY83.md` section 2 named `outer` (171.0 us per generated token, 26 percent of the door-only leaves in situ at I20)
as I21's target, and placed about 70 percent of it on one read that `pf_resident` (the second leaf, 125.0) makes too:
per record, the adapter's `ids` map and the SLRU policy's `table`. Tree at start: the DAY83 reading's commit.

## 0. The read, from the source

Both lookups run over every record the door holds (30720 here) with wide keys:
- `ids`, in the traced adapter (`TracedDispatch`, `banked_residency/native.rs`) and in the dispatch adapter
  (`SlruExpertDispatch`, `memra-tier` `bank/expert_dispatch.rs`), is a `BTreeMap` from the record's local id
  `(layer, projection, expert)` to its `BankId`; a lookup walks four or five tree levels and returns a `BankId`
  (about 130 bytes: two 32-byte digests and a heap string) stored inline in the leaf;
- the SLRU policy's `table` (`bank/slru.rs`) is an Fx-hashed map from that `BankId` to the host slot: the lookup hashes
  the whole key and compares it against the stored one (its string on the heap).
In a decode, between two leases the forward runs; each lookup touches a dozen or more cache lines, and the day-61
harness puts one residency read at 445 to 455 ns over the same record count. `outer` makes it once per demanded record
(the trace's pre-demand slot), `pf_resident` once per prefetched record (the host residency check), and the trace's
slot map (another `BTreeMap`, slot to local id) takes one insert per record inside `outer`.

## 1. Pre-registration: I21

**The change.** A record's catalog position replaces the hashed id on these reads; the maps stay, authoritative, for
everything else.
- `Catalog::position(&BankId)` (the index it already keeps; a hashed read, used at install and on the SLRU's
  admission path only) and `Catalog::len()`.
- `SlruPolicy` keeps, beside `table`, the host slot of each catalog position and the position of each slot (two
  `u32` vectors, installed by the bank when it takes the policy, empty policy only), updated at every `table` change:
  `publish_at(id, position)` in place of `publish(id)` where the view is installed (a plain `publish` on an indexed
  policy refuses `Unsupported`, so the view cannot go stale), the eviction in `reserve` and `remove` by slot. New read
  `resident_at(position)`: the same answer as `resident(id)` for the record at that position, by construction and by
  test. `BankService` calls `publish_at` with `catalog.position(id)` at its two SLRU publications (the fill admission
  and a missing record's publication).
- The SLRU's metadata charge (`slru_metadata_bytes`) counts the view: 4 bytes per catalog record and 4 per slot, added
  to today's formula, so the installer reserves what the view holds (about 240 KB here, against a 16 GiB host tier).
- `SlruExpertDispatch` resolves each local id to its position once, in `new`, into a dense table indexed by
  `(layer, projection, expert)` arithmetic; `host_resident(local)` reads the position and `resident_at`; a new
  `resident_slot(local)` returns the slot the same way, with the errors the traced adapter's read gave (`NotFound` for
  an unknown local id, then `Incomplete` without a policy).
- `TracedDispatch`'s pre-demand reads (single and grouped demand) and the trace's read for a missing record's
  published slot use `resident_slot`; the trace's slot map is a vector indexed by host slot, growing on demand, with the
  same replace-and-compare that yields each line's `victim`.

Not in I21 (named for later, each its own registration): the bank's own hashed reads on the same path (the catalog
entry in `stage`, the host cache in `stage_cache`, the SLRU `hit` in `publish_policy`, the dispatch adapter's
`validated`), and `outer`'s other parts (the proxy's registry entry, identity checks and pending insert).

**The same program.** No decision reads anything new: every answer is the one the hashed read gave, so the host demand
sequence (the trace, including each `victim`) and the tokens must equal I20's.

**CPU gates before any card** (`day84-cpu/`):
- the SLRU view against `table` on the day-4 synthetic fixture (every row) and the oracle's randomized trace: after
  every operation, `resident_at(position(id)) == resident(id)` for every id; a plain `publish` on an indexed policy
  refuses;
- `Catalog::position`: distinct, in the catalog's order, `NotFound` for an unknown id;
- the dispatch adapter: `host_resident` and `resident_slot` equal the map-based reads for every local id, before and
  after a fill admission and an eviction; an unknown local id `NotFound`;
- the day-61 profile at I20 and I21 in one window (P1 and P9, both orders), and P10's allocation census;
- the tier suites, the engine library, clippy (`-D warnings`, all targets) and fmt;
- a local RTX 5090 check: I20 and I21, both orders, `MATCH`, the same tape and host demand sequence; then the in-situ
  split of I21 beside I20 (DAY83's cell with arms `i20s` and `i21s`, 10 runs each in both orders, pinned to the
  P-cores), read by `day83-read.py`, deciding nothing: whether `outer` and `pf_resident` moved as section 0 says.

## 2. Pre-registration: the card cell `i21`

DAY82's cell with I21 for I20's step: binaries `i15=2243b1fe2`, `i20=8efea3a54` and `i21` (named in section 2a); arms
REF (`run-gen-i20` with `MEMRA_MOE_PREFETCH=1`), I15, I20, I21, I21C (I21 with `--moe-dispatch-clock`); 50 timed runs in
both orders and the profiled pair (REF, I21); integrity with I15's, I20's and I21's host demand sequences equal;
admissibility; I21 against I20 (`improves`, `regresses`, `flat`, gen-only primary, the window beside it), I21 against
I15 beside it deciding nothing, the door against REF. `regresses` on either class reverts I21 with its receipt. On the
285K class, then a 9950X.

## 2a. I21 on the CPU, before any card

I21 landed as `b555b4141` (memra-tier `bank/slru.rs`, `bank/types.rs`, `bank/residency.rs`, `bank/expert_dispatch.rs`;
memra-engine `banked_residency/native.rs`). One thing the source gave that section 1 did not name: with its reads by
position, the traced adapter no longer reads its own copy of the id map, so that copy (a second `BTreeMap` of every
record's `BankId`) is gone; the dispatch adapter's map stays for `validated`.

**CPU gates** (`day84-cpu/gates.log`): the tier suites (the bank suite 92 passed, with the five `day84` tests: the view
against the table after every one of the day-43 trace's 250,000 operations and the day-4 fixture's 2,013 rows, the
refusals of an indexed policy, the catalog's positions and the metadata charge, the dispatch adapter's reads through two
fills and 400 demands with misses and evictions), the engine library (574 passed), clippy (`-D warnings`, all targets)
and fmt clean, `git diff --check` clean.

**The CPU profile** (`day84-cpu/profile.log`, the engine library's test binaries at I20 and I21, one pinned P-core,
order I20, I21, I21, I20, I20, I21, the host shared with other lanes' work, load average 4.6 to 5.9): P1's
`host_resident` 574 to 607 ns per call at I20, 47 to 51 at I21; P9's grouped cycle 2325 to 2422 ns per block at I20,
1782 to 1907 at I21 (about 23 percent less); P1's single `demand` 2001 to 2081 at I20 and 1961 to 2055 at I21, not
resolved. P10: 16 allocations per grouped cycle at both. So the residency check lost nearly all its cost, and the
demand did not lose the pre-demand read's: that read had also brought the record's SLRU `table` entry into the cache
for the bank's own `hit` in `publish` (the same entry, still hashed), which now pays for it. The in-situ split below
reads where the demand's part went; the bank's own hashed reads are the next registration's (section 1, "Not in I21").

**The local check and the in-situ split** are queued together (queue v18, `rtx5090-queue-v18-20260926.sh`, dry-checked
under stubs in `day84-cpu/dry-check-queue.log`): the check's four runs, then 20 runs of `i20s` and `i21s`, one lock hold,
pinned to the P-cores; `day83-read.py` gained a `--change a,b` line for it (additive, deciding nothing).
