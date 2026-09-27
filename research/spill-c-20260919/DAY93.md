# WP-C day 93 (2026-09-27): OWED C2, option 1: I26's design (the host hit without a transfer ticket), registered before any code

`DAY91.md` section 1 names I26 and the questions its design must answer from the source before code. The lead:
"I26: design first, as you wrote." This file is that design. Its code waits for the lead's read of it; the answers
below are what the code must keep. Tree at start: `eb1ec5fc2` (I25 landed).

## 0. What a host hit does today (`bank/residency.rs`, `bank/expert_dispatch.rs`)

The adapter's `demand` and `demand_many` stage a ticket for every demand, then drive the full lifecycle.
- **`stage_at`** checks, in order:
  - the batch is not empty;
  - the batch is at most `limits.items` ids, and the pending tickets are fewer than `limits.tickets`;
  - the request is valid;
  - a prefetch-priority batch finds fewer than `limits.tickets - 1` pending;
  - no device, peer, replica or pinned bytes are asked for;
  - each id is the catalog's, and the logical bytes stay within `limits.batch_bytes`.

  For a batch with no misses, it then reads the host cache per unique record and clones each cached lease. It
  reserves the queue charge: pageable equal to the batch's metadata allowance (the records' ticket metadata; no
  output and no slot, since nothing is read), staging 0, and in flight 1. It inserts a `Pending` with an empty read
  plan.
- **`progress`** finds nothing to read.
- **`publish`** makes the outputs, the cached leases in block order. For demand priority it records `heat.demand` per
  id, then per id in order the SLRU `hit_at` (demand) or `resident_at` (prefetch); each answers resident.
- **`finish_ticket`** marks host use done, retires (nothing unpublished), and acknowledges: it releases the queue
  charge and removes the ticket.
- **`collect_evicted`** then releases any evicted lease no unretired ticket references.

**What the ticket protects.** An evicted lease is released by `collect_evicted` only when `can_release` finds no
unretired pending ticket referencing it. So the ticket is also the in-use guard: the bytes a copy is still reading
are not released.

**The door's configuration** (`install_expert_bank_gate`):
- `open_leases = BANKED_INFLIGHT + 1 = 33` sets four things: the proxy's pending limit, the bank's `limits.tickets`,
  the governor's in-flight capacity and its staging capacity (33 records).
- The door's governor is its own instance: its clock always reads 0, every request's deadline is `u64::MAX`, and
  nothing enqueues.
- Its pageable capacity is the planned payload, plus 4096 bytes per host slot, plus 512 MiB of headroom.

Under the qualified door every decode demand is a host hit (`DAY91.md` section 0: `host_misses=0`).

## 1. The design: I26

**Scope.** A demand, single or grouped, of demand or prefetch priority, whose every record the host cache holds
when the demand is staged. Any miss, anywhere in the batch, takes today's ticket path unchanged, and so do the
fill and every refusal path.

**The hit path** (in the bank, crate-private, called by the adapter by catalog position).
1. **The same checks, in the same order, with the same answers**: empty, items, the ticket limit, the request's
   validity, the prefetch reservation, the device and pinned refusal, the ids, the batch bytes.
   - The two ticket-count checks read `pending.len() + open_hits`, where `open_hits` counts the hit demands not yet
     finished. A hit occupies the same slot a ticket did.
2. **The same policy effects, in the same order**: for demand priority `heat.demand` per id, then per id the SLRU
   `hit_at` or `resident_at`, exactly as `publish` runs them.
3. **The same outputs**: the cached leases, cloned in block order.
4. **The in-use guard.** Each leased record's use count goes up (a map from the lease's charge id to its count, on
   the Fx hasher), and `can_release` answers `Busy` while that count is nonzero, as it does for an unretired ticket
   today. The hit demand gets a hit ticket from the bank's ticket sequence, with the same epochs and a distinct flag,
   so a finish finds its records and never a `Pending`.
5. **The finish** (`finish_ticket` on a hit ticket): the use counts go down, `open_hits` goes down, then
   `collect_evicted`, the same point as today. An unknown or already finished hit ticket refuses as an unknown ticket
   does (`UnknownTicket`).

**What goes, and why each is provably inert in the door's configuration.**
- **The pending ticket, the empty plan, the `progress` loop, the unpublished and expected sets, and the retire and
  acknowledge bookkeeping.** For a batch with no misses they hold nothing, and the in-use guard (4) takes the one
  role that matters.
- **The queue charge (the batch's metadata allowance pageable, in flight 1).**
  - **In flight.** The governor's in-flight capacity is 33, and the bank's ticket limit (also 33, now counting
    hits) bounds tickets plus hits. So no reservation of the door's could be refused for in-flight capacity with or
    without the hits' charges.
  - **Pageable.** The hits' charges together are at most `open_leases x MAX_GROUP x` the largest ticket metadata
    allowance, a few hundred KiB against 512 MiB of headroom. "About" is not a proof, so the install computes the
    bound. The worst case of every other pageable charge is:
    - the record charges of the planned slots (payload plus metadata);
    - the SLRU metadata charge;
    - the evicted leases open tickets may still hold;
    - the misses' ticket charges.

    If the headroom left over covers the hits' bound, the hit path drops the charge. Otherwise it keeps the queue
    charge as today, and one install line says which. The answers are then the same in every configuration, by
    construction.
  - **Order and time.** The door's governor has no queue and no clock, so its `Busy` and `Deadline` answers never
    arise.
  - **What it changes.** Only `used()` during a hit, and nothing in the door reads it.
  - This argument holds for the door's governor only. A bank built on another governor (`memra-kv`'s, the object
    store's) never takes the hit path, because the hit path is the adapter's, and the adapter is the door's alone.
- **The misses keep everything.**

**What the proxy, the engine and the trace see.** Nothing different. The proxy registers and finishes the same
demands with the same tokens, the engine's retire walk and its events are unchanged, and `TracedDispatch`'s
pre-demand reads and lines stay above the bank.

## 2. The gates, registered now

- **The fixture first, at I25, before any I26 line.** A seeded trace through the adapter, over a bank with fewer
  host slots than records, so hits, misses, evictions and pinned evicted leases all occur. Singles and groups, both
  priorities, injected finish failures, and refusals at the ticket and batch limits. After every operation the
  transcript records:
  - every outcome, including the refusal kinds;
  - the leased records and their bytes;
  - each demanded record's SLRU slot and the SLRU queue orders;
  - the heat of each id;
  - `cached_records` and `owned_leases`;
  - `used()` outside every hit's flight (after each finish).

  The ticket inventory and `used()` during a hit are excluded by design (section 1); the transcript says so in its
  header. The install's bound line and both of its branches get their own test. I26 must reproduce the transcript exactly.
- **The existing suites.** I24's proxy fixture changes by design (it records tickets and the in-flight charge), so
  it is re-recorded at I26 with the reason in its header. Its I25 transcript stays banked beside it. Then
  memra-tier's tests, memra-engine's `banked_native` tests, and clippy on both crates.
- **The local queue v23:** the check (p88 and I26 traced, one tape and one host demand sequence), then the split of
  `p88s`, `i25s` and `i26s` under DAY89 section 2's sizing rule, cumulative from `p88s`.
- **The card.** At `reaches`, NEED TARGET CARD for the `promo` sitting with the traced twin (`DAY92.md` section 1).

## 3. The lead's read (2026-09-27): approved with two conditions, registered before any I26 line

The lead: "Lead read of DAY93 (I26): approved, code it, with two conditions."

**The ordering argument (the lead's check, recorded here with its lines).** The adapter stages one ticket per call
and drives it to publication inside that same call (`bank/expert_dispatch.rs`, at I25):
- `demand` (line 249) stages at line 254, runs `progress` at line 263 and `publish` at line 264;
- `demand_many` (line 302) stages at line 317, runs `progress` at line 326 and `publish` at line 327.

So nothing can interleave between a demand's stage and its publication. A hit that applies `heat.demand` and the SLRU
effects at its stage lands at the same logical point as today's publication, in the same order.

**Condition 1: the hit finish is all-or-nothing.**
- Every fallible step of a hit finish comes before its first mutation: the hit ticket's lookup (`UnknownTicket`), and
  each leased record's use count checked nonzero.
- The mutations are three, and none can fail: the use counts down, the hit ticket removed, `open_hits` down.
- So a failure, injected or real, leaves every count and `open_hits` unchanged. A retried finish succeeds once and
  never decrements twice, and a finish after success refuses `UnknownTicket`.
- `collect_evicted` runs after the finish, in the adapter, as today.
- **The fault seam.** The fixture needs a finish that fails and is retried, so the bank gets a fault-injection
  method of the check (`#[doc(hidden)] inject_finish_failure`, forwarded by the adapter).
  - It makes the next `finish_ticket` return `NotReady` at the last fallible point before any mutation: after the
    lookup, on the ticket path and the hit path alike.
  - It is a fault-injection door of a check, which door hygiene keeps; it has no env read and no flag.
- **The fixture adds a scripted case to each seed**, before any I26 line and recorded at I25, with the owned leases
  and cached records in the transcript at every step:
  - a group of three host hits is held open;
  - misses evict its records while it is open, and its leases stay owned (the in-use guard: `collect_evicted` skips
    them, `Busy`);
  - its first finish is injected to fail, and the leases stay owned;
  - the retried finish succeeds, and `collect_evicted` releases them;
  - a second finish refuses `UnknownTicket`.

**Condition 2: the install's pageable bound names hits.** Tickets plus hits are at most `open_leases` (33), so the
evicted leases that open tickets or hits may pin are at most `open_leases x MAX_GROUP` records at the largest record
charge, the same count as today. The install's bound line names the term so, beside the hits' own charge bound. Its
test covers both branches: the charge dropped, and the charge kept.

**What reads `used()` (the list the "nothing reads it" claim needs; grep at `6d126a48b`).**
- **The door's governor** is created at `banked_residency/native.rs:1095`, handed to the bank at `:1122` and used
  once for the SLRU metadata reserve at `:1170`. `BankedExpertGate` keeps it only to release that charge at close
  (`:793`).
- **No door code reads its `used()`:** nothing in `banked_residency.rs`, `banked_residency/native.rs`,
  `moe_cache.rs` or `memra-tier/src/bank/`.
- **Inside the governor** (`tier/governor.rs:178-210`), `used` feeds its own admission: `fits`, `check_combine`,
  `combine_in_place`. Section 1's capacity argument covers exactly these.
- **Every other reader reads another governor, never the door's:**
  - `memra-kv/src/tiered/` (the host prefix tier and its tests);
  - `memra-server/src/worker.rs` (its tier dimensions and tests);
  - `memra-engine/src/tier_transfer.rs`, `ple_rows_tier.rs` and `qwen4exp_gpu.rs`;
  - the gate binaries `tier_transfer_gate`, `kv_tier_gate` (`active.rs`, `fault.rs`) and `storage_bench`;
  - `memra-tier/src/conformance/revision_v11.rs`.
- **The bank tests** read their stand-in governor's `used` field. I24's fixture (`tests/bank/day90.rs`) records it
  and is re-recorded at I26 by design, with its I25 transcript banked beside it, as registered. I26's fixture reads
  it only with no demand open.

## 4. I26 landed, CPU-gated (2026-09-27)

- **The fault seam and the fixture, first** (`bd4c72223`, at I25, before any I26 line): `inject_finish_failure` and
  `tests/bank/day93.rs` re-recorded, 6293 lines. Each seed ends with the scripted hit case of section 3 (on a fresh
  bank, so its SLRU state is its own), and on every seed it reads:
  - the held group of three host hits, evicted while open: owned 8 against cached 5;
  - the injected finish refuses `NotReady` and changes nothing: owned 8;
  - the retry succeeds, and `collect_evicted` releases the three: owned 5;
  - a second finish refuses `UnknownTicket`.
- **I26** (`963270775`):
  - `BankService::stage_hit_at` and `finish_hit`, as sections 1 and 3 write them.
  - The ticket limits count `pending` plus `hits`; `can_release` reads the use counts; `tickets()` lists the open
    hits.
  - `hit_charge_bound`, with `HitChargeBound::inert`.
  - The adapter tries the hit path first for `demand` and `demand_many`.
  - The door's install prints `[experts-via-tier] hit queue charge dropped|kept: ...` with every term named, the
    evicted leases' term as `evicted_leases_held_by_open_tickets_or_hits`.
  - The stage clock counts a stage when the hit path takes or refuses a demand. An attempt that finds a record not
    held keeps its time in `stage_ns`, and the `stage_at` after it counts the stage.
- **The gates** (`day93-cpu/gates.log`):
  - memra-tier's tests: day 93's fixture reproduces exactly, so does I24's day-90 proxy fixture (the generic bank
    keeps the queue charge by default, so its tickets and charges read as before), and day 85's twin-bank
    equivalence of the positioned adapter against the hashed ticket path holds;
  - the bound's both-branch test `the_hit_charge_bound_names_hits_and_both_branches_hold`: kept charges a held hit
    in flight 1 plus pageable metadata, dropped charges nothing, both return to the same charge, and dropping is
    refused `Busy` with a demand open;
  - memra-engine's `banked_native` tests, memra-kv's lib tests, clippy on memra-tier and memra-engine, fmt.
  - `moe_cache.rs` is untouched.
- **I24's proxy fixture needs no re-record.** DAY93 section 2 expected one; it reproduces unchanged, so nothing was
  re-recorded.
- **The local binary** `run-gen-i26` is `6c584cd3...`. Queue v23 (`rtx5090-queue-v23-20260927.sh`, dry-checked)
  waits behind queue v22, which took the card at 23:44Z:
  - its check: p88 against I26 traced, and I26 untraced for the tape;
  - its split: p88s, i24s, i25s, i25t, i26s and i26t in one window, read by `day93-cpu/split-read.py`.

  `DAY91.md` section 3 retired DAY89's rule as the 285K predictor, so v23 sizes the local cut and decides no card;
  the 285K decision is `DAY94.md`'s cell.
