# WP-A day 39: OWED item 3, the same-tick fill (design F) on slower CPUs

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Rig: the local RTX 5090 Laptop GPU's host (Intel Core Ultra 9
275HX) for the survey's reading; the deciding cells need a slower-CPU host with an RTX PRO 6000 and a 9950X-class
host (the lead's rentals). Every cell `executed-not-qualified`.

## 1. The survey, pre-registered before any cell or code

**The item** (`DAY34.md` section 7 and finding 2; `DAY35.md` section 4; rulings 49, 52, 53). Design F fills the
promote's pinned staging from the resident planes (`Arc<Vec<f32>>`) in ONE host function on the copy stream,
single-threaded (`SpanFillTask::run`), then the 96 span copies follow on the copy stream. On BOX4's CPU the fill of the
27B's 156.9 MB took 11.4 ms (`156.9MB filled by the hash helper in 11.4ms`, the day-32 helper's single-thread fill of
the same bytes), and the fill plus the spans outlast the 13.1 ms from the probe's submission to the next tick top
(`polls [2] (counts [90])`), so the promote lands one tick later and F reads flat there (`promote in` 28.40 against
28.40). On the 5090 host F lands inside the probe's tick and wins one tick per promote (DAY35 section 4).

**The candidates.**

- **T, a threaded fill.** The host function splits the fill's bytes across T threads (each a contiguous share of the
  planes, the same `copy_nonoverlapping` program, the same bytes), joined before it returns; T fixed per host.
- **O, an overlapped fill.** The fill runs per plane on a separate fill stream (one host function per plane group), and
  each span copy on the copy stream waits on its own group's fill event, so the copies of early groups overlap the fill
  of later ones: the copy stream's work ends near max(fill, copies) instead of fill + copies.
- **T with O.**

**The survey cell (CPU only; a reading on this host now, the deciding reading on each rented host in its sitting).** A
detached probe (`day39-fill-survey/`): the 27B's staging shapes exactly (48 buffers of 3,145,728 B and 48 of 122,880 B,
156,893,184 B, the day-31 fill line's bytes), cached pinned destinations (`cuMemHostAlloc` flags 0, the staging set's
kind), heap `Vec<f32>` sources already touched; the fill timed at T = 1, 2, 4, 8 and 12 threads (scoped threads, each
a contiguous share of the byte list), N=5 each after one warm run, plus the host's memcpy bandwidth reading at T = 1.
The 9B's shapes (48 buffers, 52,690,944 B) the same way. Bitwise: every destination compared with its source after
each run. With `--gpu` (a card present, under its lock) the probe also times the span copies themselves: every staging
buffer to a device buffer on one stream, bracketed by events, N=5, the value the rule below subtracts.

**What the survey decides, stated now.** Nothing about slower hosts from this host: it prices the scaling on this CPU
and checks the probe. On each rented host the same probe is its first cell, and the design is chosen there by this
rule: T alone if some T at or below the host's physical cores fills the 27B's 156.9 MB in at most 13.1 ms minus the
host's measured span-copy time minus 1.0 ms of margin (the copy-stream work then ends before the next tick top on the
day-34 timeline); O with T if no T does. The chosen design is then pre-registered with its acceptance (the promote's
copy landing in the probe's tick on the slower host, `polls [1]` on at least 80 of 90 steady promotes, and F's day-35
e2e margin against the day-32 helper fill there) before its code.

**Budget.** The survey 0.05 agent-day here.

## 2. The survey on this host, as it ran (`rtx5090-day39/survey/`): a reading, deciding nothing for slower hosts

- Probe `day39-fill-survey/` (hash in `survey/binary.sha256`), in the same 5090 hold as DAY38's unit rerun (16:45:19Z),
  `--gpu`. `available_parallelism=24`. Verbatim (`survey.log`), every run `bitwise=true`:
  - 27B (156,893,184 B): `threads=1 .. median=12.677` ms (12.38 GB/s), `threads=2 .. 8.979`, `threads=4 .. 6.494`,
    `threads=8 .. 5.929` (26.46 GB/s), `threads=12 .. 5.895`; `SPANS shape=27B .. median=5.621` ms (27.91 GB/s).
  - 9B (52,690,944 B): `threads=1 .. 3.738`, `threads=4 .. 1.871`, `threads=8 .. 1.619`, `threads=12 .. 1.666`;
    `SPANS shape=9B .. median=1.913`.
- Read against section 1's rule as if this host carried the 27B: T=1 needs 12.68 ms against a budget of 13.1 - 5.62 -
  1.0 = 6.48 ms, so the single-thread fill of design F would miss the probe's tick here too; T=4 (6.49 ms) sits at the
  bound and T=8 (5.93 ms) inside it. On the 9B this host needs no thread (3.74 ms against 13.1 - 1.91 - 1.0). Neither is
  the rented hosts' reading; the probe is their first cell.

## 3. The survey on the rented hosts, pre-registered before it runs

The same probe (`day39-fill-survey`, built on the box by `pro-single-day38/build.sh`) is the first item-3 cell on every
rented host, beginning with DAY38 section 10's sitting (`fill-survey.sh`, under the collector's hold after the unit
cells): the 27B's and the 9B's shapes at T = 1, 2, 4, 8, 12, bitwise, and the span copies on that card (`--gpu`), with the
host's CPU model and core counts banked beside it (`host-shape.txt`). Section 1's rule is applied to each host's own
reading and nothing else; the design it picks for that host class is pre-registered with its acceptance before its code.
A 9950X-class host and a slower-CPU host (BOX4 class) are both owed readings; whichever class this sitting's box is, the
other stays owed.

## 4. BOX7, the slower-CPU host (BOX4 class), as it ran (`pro-single-day38/box/fill/`)

- Host: an AMD EPYC 9B14 class part (96 cores, 2 threads per core, 192 CPUs, a cgroup quota of 92 CPUs), 440 GB RAM;
  one RTX PRO 6000 Blackwell Workstation Edition. The probe ran under the sitting's collector hold after the unit cells
  (`fill-survey rc=0`, 18:11:50Z). Verbatim (`fill/survey.log`), every run `bitwise=true`:
  - `FILL header available_parallelism=92 gpu=true`
  - 27B (156,893,184 B): `threads=1 N=5 ms median=10.985 min=10.867 max=11.071 gbps=14.28`, `threads=2 .. median=8.421
    min=6.747 max=9.326`, `threads=4 .. median=7.625 min=4.172 max=8.937`, `threads=8 .. median=6.701 min=3.092
    max=7.220`, `threads=12 .. median=3.889 min=3.214 max=6.184 gbps=40.35`; `SPANS shape=27B .. median=2.880
    min=2.845 max=2.955 gbps=54.48`.
  - 9B (52,690,944 B): `threads=1 .. median=3.034`, `threads=2 .. 2.101`, `threads=4 .. 1.370`, `threads=8 .. 2.380`,
    `threads=12 .. 2.424`; `SPANS shape=9B .. median=1.121`.
- **The rule of section 1, applied to this host's reading and nothing else**: the budget is 13.1 - 2.880 - 1.0 = 9.22 ms.
  T=1 (10.985 ms, design F as it runs today) misses it, as BOX4's 11.4 ms helper fill did; T=2 (8.421) is the first T
  inside it and T=12 (3.889) the fastest. **T alone is picked for this host class**; O is not built. The medians
  between T=2 and T=8 carry wide spreads here (T=8: 3.09 to 7.22 ms); T=12's maximum, 6.18 ms, is inside the budget
  too.
- The 9950X-class reading stays owed (no such host in this sitting).

## 5. Design T, pre-registered before any T code

**The design.** `SpanFillTask::run` (the copy stream's host function, design F) splits the fill's byte list into
`T_eff` contiguous byte shares (the survey probe's `shares`: one byte list across the planes in attach order, cut into
equal shares, a plane split across two shares where a cut falls inside it), runs shares 1 to `T_eff - 1` on scoped
threads (`std::thread::scope`, spawned by the host function) and share 0 on the driver's callback thread, and returns
after every share joined. Every byte is written by exactly one `copy_nonoverlapping` from the same source bytes, so
the staging contents are bitwise those of T=1 (one numeric program; nothing but the host threads changes). The copies
behind the host function are unchanged.

- `T_host = min(12, max(1, available_parallelism / 2))`, read once when the engine is built and fixed for its life
  (the probe's T=12 is the fastest or tied on both hosts read: 5.895 against T=8's 5.929 ms on the 5090's host, 3.889
  on BOX7; `available_parallelism / 2` keeps T at or below the physical cores on an SMT host). Per fill, `T_eff =
  min(T_host, max(1, bytes / 4 MiB))`, so a fill under 8 MiB stays on one thread. No new env read: the door stays
  `MEMRA_KV_HOST_CONTRACTS`, and T=1 is the arm below, not a runtime switch.
- A CPU census cell: `shares` covers every byte of the list exactly once, each share contiguous and in order, shares
  equal to within the last one, `T_eff` as stated for 0 B, 1 B, 4 MiB - 1, 8 MiB, the 27B's and the 9B's lists, with
  `T_host` 1, 2 and 12; and the fill of a list through `SpanFillTask::run` bitwise equal to its sources at T 1, 2, 5 and
  12 (heap destinations; CPU only).
- The engine's native filled-batch cell (`h2d_span_filled_batch_fills_on_the_copy_stream_before_its_copies`) gains a
  batch whose fill runs at `T_eff > 1` with a cut inside a plane, landed bitwise on the device.

**The arms** (built from one tree, the lane tip carrying T and, if it has landed by then, the fix of DAY38's hump; the
arm list and each binary's hash banked before the cell):

- `FT`: the tip as built.
- `F1`: the tip with `T_host` fixed at 1 by a one-line patch (design F as it runs today).
- `HK`: the tip with `rtx5090-day35/hk-revert.patch` ported onto it (the day-32 helper fill with design K; the port
  resolved by DAY35 section 1's rule, one test-section conflict, banked with its base as `hk-revert-tip.patch`). A
  measurement arm, not a delivery: its bar is a clean build, clippy clean and the server lib suite.
- `OFF`: the `FT` binary with the door off.

**The target-card cell (BOX7, this host class).** DAY35 section 1's cell with a fourth arm and ten promotes per boot:
`stall_cell.py --mode promote --n 10` per boot, every receipt replayed; order o1 = `HK FT F1 OFF` five times, o2 = `OFF
F1 FT HK` five times: 40 boots, N=5 boots per arm per order; the PRO promote environment of DAY34's sitting; one
collector hold; the 27B NVFP4 artifact (its sha256 banked); 250 ms telemetry. Metrics as DAY35 section 1 (E2E, PIN,
pair noise), plus the polls of every steady promote (`day33-reading.py`'s `published` polls: the second and later
promote of each boot, 9 per boot, 90 per arm pooled over both orders).

**Acceptance on the target card** (section 1's, unchanged):

- (a) `FT`'s promote lands in the probe's tick: `polls [1]` on at least 80 of its 90 steady promotes.
- (b) F's day-35 margin against the day-32 helper fill: in BOTH orders `median(HK) - median(FT)` exceeds the order's
  pair noise for E2E AND for PIN.
- (c) correctness: all 40 boots ready, `STALL REPLAY: PASS` 40 of 40 and `errors=0` on every receipt; on `FT`'s binary
  the gate set (identity default and plain, OFF and ON; failure OFF and ON; the fault gate default and plain; the hit
  gate OFF and ON) ALL GREEN, and the unit cells green.
- Readings, not clauses: `F1` against `FT` (the lever alone), `F1`'s polls, DAY28 1b's `on_minus_off` for every ON arm.
- An incomplete cell is repeated whole once in a new hold; nothing is read from a partial one.

**The 5090 cells** (before the target sitting; nothing about the slow host is decided here): the CPU census, the
engine's native cells, the gate set on `FT`'s binary (identity x4, failure x2, fault default and plain, hit OFF and ON,
the unit cells), and DAY35's cell with the arms `FT F1 OFF` (o1 = `FT F1 OFF` x5, o2 reversed, 30 boots, `--n 5`, the
9B NVFP4 MTP artifact, the day-35 environment). The 5090 clause: (e) in both orders `median(FT) - median(F1)` at most
the order's pair noise for E2E and for PIN (T costs nothing on a host where the fill already fits: 3.74 ms at T=1 on the
9B here).

**Decision.** (a), (b), (c) on BOX7 and (e) plus the gates on the 5090 all pass: T is kept, item 3 closes for this host
class, and the 9950X-class reading stays owed. (a) fails: T is refuted on this host class as it reads, reverted in one
commit, and O with T is pre-registered next (section 1's rule). (a) passes and (b) fails: recorded as it reads, T kept
only if (e) passes (it is not worse anywhere), and the question of F against the helper fill goes to the lead as an
owner item. (e) fails: the per-fill `T_eff` floor is revised under a new pre-registration before the target sitting.

**Budget.** 0.5 agent-day: code and CPU cells 0.1, the HK port 0.1, the 5090 cells 0.2, the target cell's share of a
sitting 0.1.

## 5a. Amendment to section 5, before any T cell runs: the stall cell's `--n`

Section 5 wrote `stall_cell.py --mode promote --n 10` for the target cell and, in the same paragraph, "9 per boot, 90 per
arm pooled over both orders". Those disagree: the harness's `--n` is its n per order (`STALL rule .. n_per_order=5
pooled=10`), so `--n 5` gives ten promote runs per boot, nine steady (DAY35's cell: `e2e N=50` per arm per order from five
boots; DAY34's BOX4 sitting: `promote in-ms steady N=90`), and `--n 10` would give nineteen. The count the clause is
written on (90 steady promotes per arm, `polls [1]` on at least 80) is kept, so both cells run `--n 5`, as DAY34 and
DAY35 did. Found by dry-running the reader (`day39-reading.py`) on DAY35's banked cell; no T cell has run.
