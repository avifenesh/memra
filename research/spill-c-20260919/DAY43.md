# WP-C day 43 (2026-09-24): the MoE slot cache door, improvement I6: the host tier at the machine's scale

`OWED.md` C1 step (b), first improvement in the order day 40's rule set (`DAY40.md` section 5: `demand` is the top
stage, 16.36 of 20.47 ms per token on the RTX 5090's window, and at the 16-record tier every GPU miss is a host miss
and a physical read). Tree at start: `9085bdf50`. Every cell is `executed-not-qualified`. Nothing moves a default.

## 1. Pre-registration (committed before any I6 code)

**What is wrong at scale, from source (tree `9085bdf50`).**

1. `host_bank_slots` (`banked_residency.rs`) refuses above 256 MiB ("qualification ceiling") and clamps to 16
   records; the installer builds one SLRU class of `max_bytes` (860,160 B) slots, so a 450,560 B record takes an
   860,160 B slot; the governor's pageable capacity is a fixed 512 MiB.
2. `SlruPolicy` (`crates/memra-tier/src/bank/slru.rs`) is "a readable CPU oracle, not a tuned O(1) replacement":
   `hit` and `remove` scan `VecDeque`s (`position`, `retain`), and its maps are `BTreeMap<BankId, _>`.
3. `BankService::collect_evicted` (`residency.rs`), called by every `finish`, compares every owned lease with every
   cached one (`owned.values().filter(|l| !cache.values().any(..))`): quadratic in the tier's size. `cache_bytes()`
   sums the cache on every call.

At 16 records none of this shows. At the budget day 40's profile asks for (a window LRU hit fraction of 0.439 from
8 GiB up, 0.000 at 2 GiB or less) it does.

**The design.**

- (a) **The plan.** `--expert-bank-host-bytes=N` is planned into one SLRU class per exact record size in the catalog
  (three here: 450,560, 557,056 and 860,160 B), each class's slot count proportional to its record count under N
  (the Hamilton remainder pass of `moe_cache.rs::size_class_plan`, each class capped at its record count), so the
  tier holds records, not max-size slots. Refused (the typed `ExpertBankRefusal`, exit 2) when N cannot hold one slot
  of the largest record size, and when N exceeds three quarters of the host's `MemAvailable` read at install (a
  machine fact, `/proc/meminfo`; no environment variable). The 16-record clamp and the 256 MiB ceiling are removed;
  the default stays 256 MiB. The governor's pageable capacity is the planned bytes plus the metadata and staging it
  already charges. The installer prints `[experts-via-tier] host_bank_plan requested=N planned=P classes=[(bytes,
  slots), ...] records_held=R ceiling=C`.
- (b) **O(1) host SLRU.** `SlruPolicy` keeps its interface and its decisions; its segments become intrusive
  doubly-linked lists over slot indices (the `moe_cache.rs` `SlruList` shape) and its maps `HashMap`s. The current
  `VecDeque` implementation moves to the tier's tests as the oracle; every existing SLRU test runs on the new policy,
  and a randomized trace (at least 200,000 operations over three classes with `keep` sets, reserve, publish, hit,
  remove and abort) must produce the oracle's decision (`slot`, `evicted`) at every step and its `orders()` at every
  1,000th.
- (c) **Bookkeeping in O(evicted).** Leases the SLRU evicts from `cache` go on an explicit list; `collect_evicted`
  tries to release only those (a `Busy` lease stays listed); `cache_bytes` is maintained on insert and remove. A CPU
  test drives a bank of 4,096 records through 20,000 demands with evictions and asserts, after every `finish`, that
  every owned lease is either cached or held by an open ticket.
- (d) Unchanged: the per-record verify, the lease identity, the trace line's form (its `slot=` field now ranges over
  the planned slots), `physical_reads`, one lease at a time, the drains. I6 changes where bytes can stay, not how a
  miss is served.

**The cell `resid` (RTX 5090 first; the target card in the ladder sitting).** Day 40's shape unchanged
(`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986`, prompt `55 88 13`, the approved artifact). Four arms,
every door arm with `--expert-bank-stages` (its cost was within bound on day 40): OFF (no door); BASE (the day-40
binary `b73bb4d3...`, default budget, today's program); I6D (the I6 binary, default 256 MiB); I6G (the I6 binary,
`--expert-bank-host-bytes=8589934592`). Order 1 (OFF, BASE, I6D, I6G) x 5, order 2 (I6G, I6D, BASE, OFF) x 5; N=5 per
arm per order, 40 runs in one collector hold, `/tmp/memra-5090.lock`, 250 ms telemetry, the runner waiting for no
compute app and at least 46 GiB `MemAvailable` (the 8 GiB tier beside the loader's 15 GB of pinned slabs).

**Integrity (a failure voids the reading, the clause named).** All 40 runs exit 0 with `MATCH`; one `tokens:` tape
and one STEADY-STATE line across the 40; `slots=9986` everywhere; door arms print `installed`, `physical_reads=` and
the four stage lines; I6 arms print the plan line and BASE does not; in every door run `physical_reads` equals the
count of `hit=false` trace lines and the trace line count equals the host demands.

**Readings and the decision rule.** Per arm: the window's door cost per token (median over 10 of the arm's window
minus OFF's, over 32), per-order medians, and day 40's R2 stage lines (host hits per token, `verify`, `step`,
`stage`, `demand`). Let `noise` be the larger IQR of the two arms compared (window seconds, N=10 each).

- **I6 at the default budget does not regress:** `median(I6D window) - median(BASE window) <= noise` pooled and in
  both orders. If this fails, I6 is fixed or reverted before anything builds on it.
- **The budget's effect (a reading with a direction registered):** host hits per window token on I6G above 0 and the
  window door cost on I6G below BASE's by more than `noise` in both orders, `resid_budget_helps`; otherwise
  `resid_budget_flat` or `resid_budget_hurts`, stated as it reads. Day 40's LRU profile predicts roughly 40 host hits
  per token (0.44 of 92.3) and a `demand` drop of the same fraction; SLRU is not LRU, so the number is a direction,
  not a bound.
- I6 stays in the tree if the no-regression clause holds: it is the tier the fill (the next improvement) fills.

**What each card can decide.** The RTX 5090 decides I6's regression clause on this card and reads the budget's
effect here. The target card reads both in the ladder sitting; its budget arm is registered there with that host's
`MemAvailable`.
