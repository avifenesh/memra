# WP-C day 83 (2026-09-26): OWED C11, the door's per-block CPU split in situ at I20, before the next cut

Lead: "Continue C11 (the door's remaining gap)". DAY82 read I20 `flat` on both classes and the door still `loses` to
REF gen-only by 3 ms over 32 tokens (BOX39, BOX40). Tree at start: `7e4bf9668`.

## 0. What the card cells' own clocks say (`day83-cpu/section0.log`)

The door's dispatch clock (`--moe-dispatch-clock`, the clocked arm of every card cell since DAY60) brackets the
prefetch path's door-only steps: `pf_demand` (the owner's lease), `pf_resident` (the host residency check) and
`pf_retire` (finishing leases whose copies are done). REF runs none of them. Generate phase (generate minus gate),
medians of each clocked arm's 10 runs, per prefetched block (6007 blocks over the 32 generated tokens in every run):

| cell, host | program | pf_demand | pf_resident | pf_retire | door-only | pf_stage (both programs) |
|---|---|---|---|---|---|---|
| gap15, BOX29 (285K) | REF | 0 | 0 | 0 | 0 | 2.29 us |
| gap15, BOX29 (285K) | I15 | 1.72 us | 0.50 | 0.32 | 2.54 | 2.55 |
| i18, BOX34 (285K) | I18 | 1.74 | 0.46 | 0.28 | 2.48 | 2.49 |
| i20, BOX39 (285K) | I20 | 1.67 | 0.46 | 0.28 | 2.41 | 2.50 |
| i20, BOX40 (9950X) | I20 | 1.40 | 0.41 | 0.24 | 2.05 | 2.42 |

Three things follow.

- **The cuts since I15 moved the door's own clock by about 0.1 us per block** (2.54 to 2.41 across three 285K hosts,
  so host variation is inside that): I17, I18 and I20 were real (the local profile resolved I17's and I18's), but small
  against what is left.
- **What is left is large, and the wall sees only part of it.** On BOX39 the door-only work is 453 us per generated
  token (14.5 ms over 32 tokens) against a wall gap of 3 ms; on BOX29 at I15 it was about 17 ms against 9. The GPU
  hides between a half and four fifths of it by host. So a cut the card resolves has to remove a large share of those
  2.4 us per block, not another 5 percent.
- **Where it is, as far as the receipts go.** The stage clock (`--expert-bank-stages`) splits the owner's lease inside
  the bank. The last cell that carried it on the target card is DAY64b's `i15s` arm (I15, BOX14, a 285K, N=10), read
  now: per generated token the owner's inner demand is 254.8 us, of which the bank's `stage` 162.2 (the host cache
  read `stage_cache` 106.4, the budget charge 26.6, the rest 20.9, the catalog lookup 8.3), `publish` 37.6 (its policy
  27.4) and the dispatch adapter's own work 55.0; the retire side 33.8 (the acknowledgement 29.7); the trace 13.3.
  `stage_cache` (per unique id: the host cache's lookup, and a clone of each cached lease) is the largest single leaf
  at I15, 3.4 ms over 32 tokens, larger than the whole gen-only gap. I18 changed part of that step (the ticket's records
  by position); the card clock above says little of it moved. No cell has split the lease at I20, on any host.

So the next step is the split itself at I20, in situ (the real decode, where the forward runs between two leases),
before choosing I21. A hot-loop profile (the day-61 P9 harness) runs the same calls back to back; it resolved I17 and
I18 but does not say what each step costs between two launches of a real decode.

## 1. Pre-registration: the cell `split20` on the local RTX 5090 (a measurement; no code, no default)

**Why here.** The split is a CPU reading of the owner thread, not a wall comparison. The local CPU is a Core Ultra 9
275HX (Arrow Lake, the 285K's core design), and the local card runs the same program. No target card is needed to
name I21's term; the target card measures I21 afterwards, by wall, as always.

**Binaries.** `run-gen-i15` (`2243b1fe2`) and `run-gen-i20` (`8efea3a54`), built by `c-local-build.sh` into
`target/c-bins`, hashed into the cell.

**Arms.** The DAY82 cell's door argv (`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986`, prompt `55 88 13`,
`--experts-via-tier --expert-bank-host-bytes=17179869184`) with:
- `i15s`: `run-gen-i15` with `--moe-dispatch-clock --expert-bank-stages`;
- `i20s`: `run-gen-i20` with `--moe-dispatch-clock --expert-bank-stages`;
- `i20c`: `run-gen-i20` with `--moe-dispatch-clock` only (to read the stage clock's own cost on `pf_demand`).

Order 1 (i15s, i20s, i20c) x 5, order 2 reversed x 5: 30 runs. Each run pinned to the local CPU's eight P-cores
(`taskset -c 0-7`, from `/sys/devices/cpu_core/cpus`) inside the 1200% CPU cap, so the owner thread never lands on an
E-core. One hold of `/tmp/memra-5090.lock` for the 30 runs, taken after the idle wait (the lock free, no compute app
on the card, at least 40 GiB MemAvailable); the host's load average recorded before and after, since other lanes'
CPU work can share this host.

**Reader** (`day83-read.py --check`, written before the cell's script). Integrity: every run exits 0 with `MATCH`, one
tape across the 30 runs, one host demand sequence across them (the same program, as DAY82 read on both classes), 10
runs per arm; a failing integrity decides nothing. Then per arm and phase (generate minus gate, window minus warm),
medians over the arm's 10 runs, in us per generated token: the leaves of the door-only CPU:
- `outer`: the engine's demands (`pf_demand` plus the dispatch-time `demand_ns`) outside the owner's inner demand and
  trace: the proxy's registry entry, its identity checks and pending insert, the traced adapter's pre-demand lookups;
- `dispatch_inner`: the owner's inner demand outside the bank's `stage` and `publish` (validated ids and their clones,
  the batch, the progress loop);
- the bank's `stage` leaves (`stage_lookup`, `stage_cache`, `stage_charge`, the rest), `publish` leaves (`publish_output`,
  `publish_policy`, the rest), the retire side (`host_use`, `retire_only`, `ack`, `collect`), and `retire_outer` (the
  engine's `pf_retire` outside the bank's finish);
- `pf_resident` and the trace.

Then `DAY83 LARGEST`: the largest leaf in `i20s`'s generate phase, with its share of the leaves' sum; when its median
minus its IQR does not clear the second's median plus its IQR, both are named. Beside it, deciding nothing: the stage
clock's own cost (`pf_demand` in `i20s` against `i20c`) and each leaf's change from I15 to I20 on this host.

**What it decides.** Only I21's target. I21 is registered in `DAY84.md` before any code, under I20's rule: remove the
named leaf's work by the smallest change that keeps every answer, order and refusal (borrowed or reused structures,
work done once at install instead of per lease, a denser key in place of a hashed one; never a skipped protocol step).
Then CPU gates, and the card cell in DAY82's shape on the 285K class, then a 9950X.

## 1a. Before the cell: one leaf's scope corrected

Found in the reader's dry check (`day83-cpu/dry-check-queue.log`, stub binaries replaying one BOX39 I20C run and
DAY64b's first I15S stage lines, meaningless), before any cell: `pf_retire` brackets only the prefetch path's
`retire_banked` calls, while the bank's retire counters count every finish, the dispatch path's too, so `retire_outer`
as section 1 wrote it mixed two scopes. It is now the engine stage clock's `retire_ns` (every `retire_banked` call)
minus its in-flight-bound `wait_ns` and the bank's `retire` and `collect`, one scope. `pf_retire` stays beside it. No
other leaf changes; `outer` already pairs `pf_demand` with the dispatch path's `demand_ns`, the owner clock's scope.
The queue's control flow under the same stubs: 30 runs in the registered order, 10 per arm with the registered flags,
the lock held across them, the reader run into the cell.
