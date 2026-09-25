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

## 1a. I14 on the CPU

Both changes' CPU gates passed at their commits (the memra-tier suite 303 and 304, engine lib 567, clippy `-D warnings`,
fmt): `the_hashed_catalog_answers_as_the_ordered_one` (every answer and refusal against a `BTreeMap` of the same
entries, both layout classes, a duplicate still refused), `ids_and_records_iterate_in_bank_id_order`, the door-shaped
30720-record catalog's every `record` found through the index, `trim_evicts_the_ordered_maps_victim` (400 randomized
demands with ties against a `BTreeMap` reference, the cached set compared after every ticket), and the SLRU and bank
trace tests unchanged. Commits: change 1 `8e7faf4ec`, change 2 `83f03d9b7`.

## 2a. I15 on the CPU

I15's CPU gates passed at `2243b1fe2` (the memra-tier suite 305, engine lib 568, clippy `-D warnings`, fmt, and
`cargo check --workspace --all-targets`): `a_grouped_demand_leases_its_records_in_order` (the token names every
record in order, `with_bytes_at` lends each record the bytes a single demand lends, the pending bound counts the
group as one ticket, `finish_group` once, single and group tokens refuse each other, an empty and a four-record group
refused, a bank that publishes the records reversed refused with its group finished through it), the day-46, day-48,
day-50 and day-61 censuses read against the grouped prefetch (the demand path still demands one record behind its
retire; the prefetch retires before its bound and its one grouped demand; the memo's guard at both sites), and the new
census `a_group_is_finished_once_on_its_proven_paths`. Section 2 said the grouped lease is "finished only from
`retire_banked` and `retire_all_banked`"; that was incomplete: as the single prefetch always did, the grouped one also
finishes its group in its two refusal paths (a group naming other records, and a staging refusal on a drained copy
stream before any member was staged), and the cache's `Drop` finishes every group after both streams drain. The
census pins exactly those sites and no other. The day-4 fixture was re-pinned to the new `moe_cache.rs`
(no SLRU statement changed).

**The CPU ladder** (`cpu-day64/`; each row as run in `ladder-<n>.log`, all four binaries re-read in one window in both
orders in `window.log`; P2 is one host-hit prefetch per block through the proxy, P8 the grouped form per block):

| row | commit | change | P2 as run | P2 same window | P8 same window |
|---|---|---|---:|---:|---:|
| 0 | `c9379c051` | I13 | 2975.1 (`cpu-day63/ladder-4.log`) | 2995.0 | |
| 1 | `8e7faf4ec` | I14 change 1, the catalog hashed | 2606.8 | 2637.8 | |
| 2 | `83f03d9b7` | I14 change 2, the host cache hashed | 2458.0 | 2535.9 | |
| 3 | `2243b1fe2` | I15, one ticket per expert | 2606.4 | 2550.2 | 1997.0 |

I14 takes the single host-hit prefetch from 2995 to 2536 ns in one window (`stage_lookup` 123 to 46 ns, `stage_cache`
478 to 351, `ladder-2.log`); I15 does not touch the single form (2550) and the grouped form costs 1997 ns per block,
a third less than I13's 2995. Day 61 began this program at 5370 ns.

## 3a. The binaries and the sitting, named before any cell

Labels: `c60=da649107c` (REF), `i13=c9379c051`, `i14=83f03d9b7` (I14's last change), `i15=2243b1fe2`. `day64-cell.sh`
and `day64-read.py` written after section 3; the reader dry-checked for mechanics only on day 63's target receipts
relabelled (one host demand sequence `4bdc2610...` across the door arms; its verdict there means nothing). The driver
`day64-box.sh` (builds by `day63-box-build.sh`, `run-gen` only), dry-checked (`day64-cpu/dry-check-driver.log`). Run as
`D64_BUILDS="c60=da649107c i13=c9379c051 i14=83f03d9b7 i15=2243b1fe2" bash /root/wt-c/research/spill-c-20260919/day64-box.sh`.
Expected: four builds about 20 minutes, the cell about 10 (50 runs at about 11 s). Box needs as `DAY63.md` section 3a.
The RTX 5090's cell is queued (queue v8, behind v7) with `run-gen-i14` (`dc406de3...`) and `run-gen-i15`
(`a909193e...`) built locally (`/tmp/c61-build/build-i1{4,5}.log`).

The integrity gate carries I15's main claim: one host demand sequence (the trace without slot numbers) across I13,
I14 and I15, so the grouped demand must reproduce the per-block order of host hits, misses and victims exactly.

## 4. The target card, first sitting (BOX15, run by the lead as registered; `pro-single-day64/`)

The lead ran `day64-box.sh` exactly as section 3a names it on the tree `6d099c0fd`. The Core Ultra 9 285K class of
BOX12 and BOX13 was taken, so it ran on BOX15: one RTX PRO 6000 Blackwell Workstation Edition at 600 W (driver
595.84) on an AMD Ryzen 9 9950X host, accepted by the lead with no brake and a 90 GiB alloc, idling at 26.7 W against
the 285K boxes' 15.7 to 16.0 W, FMA spin 2671 to 2687 MHz against 2855. 07:34Z to `box done 2026-09-25T07:50:16Z`;
binaries built on the box (`box-binaries.sha256`); the runner pinned with `taskset -c 0-11`. The lead mirrored the
receipts; read here: 229 of 229 files `OK` against `box-mirror-manifest.sha256` (`MIRROR-CHECK.txt`). Regime
(`regime.txt`): 25 to 41 C, SM median 2610 MHz. Verbatim (`i15/reading.log`):

- `DAY64 host demand sequence sha256 4bdc2610c3534e42 lines=[22077]`
- `DAY64 I15 CHECKS rig=pro-single runs=50 integrity=ok`
- `DAY64 gen-only decode medians (N=10 each): ref=0.245 i13=0.322 i14=0.283 i15=0.307 i15s=0.314`
- `DAY64 STEP i14_vs_i13 gen-only decode: pooled=-0.0385 o1=-0.0060 o2=-0.0020 noise=0.0742 -> flat`
- `DAY64 STEP i15_vs_i14 gen-only decode: pooled=+0.0245 o1=-0.0110 o2=+0.0590 noise=0.0692 -> flat`
- `DAY64 steady window medians (N=10 each): ref=0.218 i13=0.279 i14=0.247 i15=0.271 i15s=0.275`
- `DAY64 DOOR i15_vs_ref gen-only decode: pooled=+0.0625 o1=+0.0620 o2=+0.0630 noise=0.0622 -> matches`
- `DAY64 DOOR i15_vs_ref steady window: pooled=+0.0530 o1=+0.0530 o2=+0.0530 noise=0.0532 -> matches`
- `DAY64 VERDICT rig=pro-single integrity=ok i14=flat i15=flat door=i15 vs_ref=matches (window: i14=flat i15=flat vs_ref=matches)`

The one host demand sequence held across all 40 door runs of I13, I14 and I15 (I15's grouped demand reproduced the
per-block order of hits, misses and victims exactly), and every run's tape matched.

**The verdict reads as the rule reads it, and it stands in the record as `flat`, `flat`, `matches`.** Its noise terms
(0.053 to 0.074 s) are the size of the effects, against 0.001 on BOX13, so the placement below is recorded beside it.

**Where the noise comes from** (the receipts, placed before any conclusion). Every door arm is bimodal by boot, and REF
is not (`i15/ev/marks.tsv` and the logs, in run order):

- REF: all 10 runs 0.245 or 0.246 s gen-only, 0.217 or 0.218 window.
- The door, every arm: each boot reads either 0.248 to 0.251 s gen-only (0.219 to 0.221 window) or 0.307 to 0.325 s
  (0.271 to 0.280), in both orders and at every position of the interleave: I13 4 fast and 6 slow, I14 5 and 5, I15 4
  and 6, I15S 3 and 7.
- The mode is set per boot on the host CPU, not the card. In the slow boots the host fill before decode takes 1188 to
  1350 ms (all but one of them 1331 or more) against 1113 to 1131 in the fast ones; I15S's stage clock reads every CPU-side part about 1.85 times larger
  (the owner's demand 6.07 to 6.73 ms per window against 3.37 to 3.55, the cache's lease retire 1.90 to 2.01 against
  1.06 to 1.10, the bank's `stage` 3.67 to 4.15 against 2.11 to 2.24), while the GPU time of each banked copy is the
  same or lower (17.6 to 18.5 us against 19.6 to 20.0) and the SM clock reads 2610 MHz in both modes.
- The likely cause, not measured: the 12 pinned CPUs span both of the 9950X's core complexes (0 to 7 and 8 to 11), and
  the door's decode is CPU-side-bound (`DAY60.md`), so its owner thread's placement moves its speed and REF's lighter
  CPU work does not. No receipt records the owner thread's CPU, so this stays a hypothesis.
- What the fast boots show, deciding nothing: there the door runs within about 3 to 6 ms of REF over 32 tokens (I15
  0.248 against 0.245 gen-only). What the slow boots show: the same door 62 ms behind. Both orders put the door behind
  REF by 0.062 to 0.063 s in medians.

This box's regime cannot decide the steps or the door against REF by the registered rule: its noise is the door's own
host-CPU bimodality. The 9950X-class finding (the door's sensitivity to its owner thread's host placement) is recorded
as a finding of this cell and becomes item C12 of `OWED.md`.

## 5. Pre-registration: the rerun on the 285K class, with an admissibility clause (before it runs)

**The clause, new, for this rerun and every later door timing cell of this lane.** A timing cell is admissible only
if every arm's gen-only IQR and steady-window IQR are each at most 0.005 s (a fifth of a millisecond per token; BOX12
and BOX13 read 0.000 to 0.001 across all their arms). An inadmissible cell's step and door readings decide nothing and
are recorded as they read, with the arms that failed the clause. The clause does not change section 4's reading.

**The rerun.** The same cell and binaries (`c60`, `i13`, `i14`, `i15` from the same commits, built on the box) on the
Core Ultra 9 285K host class of BOX12 and BOX13, as cell `i15b` (the same arms and order as `i15`), read by
`day64-read.py --admissibility`, which prints the clause's line (`DAY64 ADMISSIBILITY ... -> admissible` or
`-> inadmissible` with the failing arms) before the verdict and, when inadmissible, ends in `DAY64 VERDICT ... ->
void (inadmissible)`. One log-only addition for the placement question: the cell samples every 250 ms, with `ps`, the
CPU each `run-gen` process last ran on (`ev/placement.tsv`); it decides nothing.

**Post-hoc, deciding nothing:** the clause, run over section 4's receipts, is printed once into
`pro-single-day64/admissibility-posthoc.log` for the record.

**Script changes for the rerun, before it runs.** `day64-read.py` gains `--admissibility` (without it the reader
prints exactly what it printed on BOX15: re-read, byte-equal to `pro-single-day64/i15/reading.log`); `day64-cell.sh`
gains the cell `i15b` (the arms and order of `i15`, plus the placement sampler, stopped by its own pid when the arms
end); the driver `day64b-box.sh` runs `i15b` into `/root/spill-receipts/c-day64b` and reads it with
`--admissibility`, dry-checked (`day64-cpu/dry-check-driver-b.log`). The RTX 5090's queued `i15` cell (queue v8) is
read with `--admissibility` when it is recorded. The post-hoc line over BOX15 (`admissibility-posthoc.log`):
`DAY64 ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0742 max_iqr_window=0.0590 failing=[...] ->
inadmissible` for every door arm, REF inside the ceiling. Run as
`D64_BUILDS="c60=da649107c i13=c9379c051 i14=83f03d9b7 i15=2243b1fe2" bash /root/wt-c/research/spill-c-20260919/day64b-box.sh`
on a Core Ultra 9 285K host with one RTX PRO 6000 Blackwell Workstation Edition (box needs as section 3a).
