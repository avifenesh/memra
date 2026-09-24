# WP-C day 45 (2026-09-24): the MoE slot cache door, improvement I3: the host fill

`OWED.md` C1 step (b). Day 40's profile: 1,659 of the window's 2,955 host demands are the first demand of their record
in the process, and a GPU miss is for a record evicted long ago or never loaded, so host residency alone reaches a
window hit fraction of 0.439 at 8 GiB and above (the day-43 dry check read 40.5 host hits per window token of 92.3).
First touches need the record in the host tier before it is demanded. Written before any fill code; tree at start:
`21a742625` (I6 with its governor fix, I9).

## 1. Pre-registration

**The design.**

- (a) **Fill workers.** When the door installs with a host budget that holds at least one record, the gate starts
  `min(8, available_parallelism / 2)` fill threads over the catalog's retained records in catalog order (trunk blocks
  by layer, then gate, up, down, then expert id). Each thread owns a clone of the artifact's opened inode
  (`Arc<File>`, the one the SHA lock authenticated), takes the next record from a shared atomic cursor, `pread`s its
  exact range into a fresh buffer, computes the contract checksum (`memra_tier::contracts::checksum`, the function
  `BankService::progress` verifies with) and sends `(record, bytes, digest)` over a bounded channel (at most 64
  records in flight, so the fill's heap is bounded). The workers touch no CUDA state, no bank state and no lease.
- (b) **Admission on the owner thread.** At the start of every host demand (`TracedDispatch::demand`) the owner drains
  up to 32 finished fills from the channel without blocking and offers each to the bank through a new
  `BankService::admit_filled(id, bytes, digest)`: the digest must equal the catalog's checksum for the record and the
  storage tail must be zero (the same two checks `progress` applies to a read), the record must not be resident or
  pending, and the SLRU must have a FREE slot of a class that holds it; then it is charged to the governor, becomes a
  `BankLease`, and is published into the host cache exactly as a demand's read would be. The fill never evicts and
  never replaces: a record that finds no free slot, or is already resident, is dropped. A digest mismatch is refused,
  counted and printed (`[experts-via-tier] fill refused <record>: checksum mismatch`), never admitted.
- (c) **Stop.** The workers stop when the cursor passes the last record or the owner signals stop; the gate's `Drop`
  signals stop, drains the channel and joins every worker before the bank closes. A worker's read error is counted and
  ends that worker; nothing is retried.
- (d) **Counts.** The stage line (with `--expert-bank-stages`) and the close line gain `fill_reads`, `fill_admitted`,
  `fill_dropped`, `fill_refused`, `fill_ns` (the owner's admission time). `physical_reads` keeps counting the demand
  path's reads only, so day 43's trace consistency clause (`physical_reads` equals the `hit=false` lines) holds as
  before.
- (e) Unchanged: every demand still verifies what it reads; a fill-admitted record is bytes a worker read from the
  authenticated inode and the owner accepted against the catalog digest; the numeric program is unchanged.

**The cell `fill` (RTX 5090 first).** Day 40's shape. Three arms, every door arm with `--expert-bank-stages` and
`--expert-bank-host-bytes=17179869184` (16 GiB, above the whole bank's 15.2 GB, so every record fits): OFF (the fill
binary, no door); I9G (the day-44 binary at 16 GiB, no fill); FILL (the fill binary at 16 GiB). Order 1 (OFF, I9G,
FILL) x 5, order 2 reversed x 5, one collector hold, the runner waiting for 40 GiB `MemAvailable`.

**Integrity.** As day 44's (30 runs, `MATCH`, one tape, one STEADY-STATE line, the trace consistency), plus
`fill_refused=0` on every FILL run.

**Clauses.** Let `noise` be the larger IQR of the two arms compared (window seconds).
- (i) **FILL beats I9G in the window**: `median(FILL window) < median(I9G window) - noise` in both orders.
- (ii) **The fill reaches the window**: FILL's host hits per window token at least 0.9 of its GPU misses per token.
- A reading, direction registered: FILL's gen-only door cost below I9G's (the gen span holds more first touches).
The fill stays if (i) holds. If (i) fails with (ii) holding, the fill works but its admission or residency costs what
it saves, recorded and investigated before the next improvement; if (ii) fails, the fill did not finish before the
window, recorded with `fill_admitted` at `phase=warm`.

**What each card can decide.** The RTX 5090 decides (i) and (ii) here; the target card reads them in the ladder
sitting (its host SHA runs at 2.15 GB/s, so the fill takes longer there; `fill_admitted` at `phase=gate` says how far
it got by the first forward).

## 2. Results, cell `fill` (RTX 5090 Laptop GPU, `rtx5090-day45/fill/`)

One collector hold, 22:56:27Z to 23:03:38Z, 30 runs, tree `a25a78008`, binaries `run-gen-i9` `878aa1ff...` and
`run-gen-fill` `19bcfd59...`, the approved artifact, the runner under the 1200% cap. Regime (`regime.log`, 250 ms,
N=1718): SM 180 to 2790 MHz, power 21.8 to 168.1 W, 59 to 80 C. Collector `--validate` rc=0.

Verbatim (`fill/reading.log`):

`DAY45 FILL CHECKS rig=rtx5090 runs=30 integrity=ok`

`DAY45 ARM i9g window_door_ms_per_token=22.50 gen_door_ms_per_token=57.25 window_s median=1.051 iqr=0.006 | per window token: gpu_misses=92.3 host_hits=40.5 host_misses=51.8 demand=19.161 verify=5.290 step=11.448 finish=0.104`

`DAY45 ARM fill window_door_ms_per_token=5.17 gen_door_ms_per_token=13.66 window_s median=0.496 iqr=0.011 | per window token: gpu_misses=92.3 host_hits=92.3 host_misses=0.0 demand=0.867 verify=0.000 step=0.000 finish=0.084 fill_admitted gate=26254 warm=29556`

`DAY45 CLAUSE (i) fill_minus_i9g window o1=-0.554 o2=-0.562 noise=0.011 rule < -noise both orders -> PASS`

`DAY45 CLAUSE (ii) fill host_hits/gpu_misses per window token median=1.000 rule >=0.9 -> PASS`

`DAY45 READING gen fill_minus_i9g=-1.395 s -> gen_lower`

`DAY45 FILL rig=rtx5090 integrity=ok clause_i=PASS clause_ii=PASS`

The fill stays. Every window miss is a host hit (92.3 of 92.3 per token), the demand falls 19.16 to 0.87 ms per token
(no read, no verify on the owner thread), and the window door cost falls 22.50 to 5.17 ms per token; gen-only decode
falls by 1.395 s over 32 tokens.
