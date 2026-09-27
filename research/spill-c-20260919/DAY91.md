# WP-C day 91 (2026-09-27): OWED C2, option 1 after I23 and I24 read short: the next candidates, and the I24 `promo` sitting, registered before any code or card

`DAY89.md` section 4: I23 and I24 remove 42.4 us per generated token of door-only work on the local host, against the
155 the sizing rule asks, so the next candidates register before any card (`DAY89.md` section 2). The owner's order
stands: option 1 only (close the 285K gap with the cuts, then `promo` on both classes). Tree at start: `d9a45ae5c`
plus section 4's record.

## 0. Where the remaining door-only work is, and what the candidates can reach

At I24, generate phase, local host (`rtx5090-day89/split24/`, us per generated token, medians of 10), the leaves sum
to 281.2:

| leaf | us | what it is |
|---|---|---|
| `retire_outer` | 51.8 | the engine's retire walk, its event queries, the proxy's finish entry |
| `outer` | 46.5 | the proxy's registry entries, the traced adapter's pre-demand slot reads |
| `bank_stage_cache` | 38.4 | the host cache's unique-id pass and cached-lease clones |
| `dispatch_inner` | 37.9 | the adapter's batch: validated ids, their clones, the request clone, the progress loop |
| `bank_ack` | 20.7 | the acknowledgement: the queue charge's release and the ticket's removal |
| `bank_publish_policy` | 16.7 | the SLRU's hits in publication order |
| `own_trace` | 16.4 | the host demand trace's lines |
| `bank_stage_charge` | 15.8 | the ticket's queue reserve |
| `stage_rest` | 12.3 | the plan, the pending insert, the rest of `stage` |
| `pf_resident` | 11.0 | the prefetch's host residency reads |
| the rest | 13.6 | publish output and rest, stage lookup, host use, collect, retire only |

The rule needs 112.6 more.
- DAY90's three named candidates (`outer`, `dispatch_inner`, `bank_stage_cache`) hold 122.8 together, so trimming
  them cannot reach it: they would have to all but vanish.
- **What decides the path:** under the qualified door the whole bank is host resident, and after the fill every
  decode demand is a host hit. `o1-i24s-r1`: `host_misses=0` in every phase, `host_hits=22077`.
- So the full ticket lifecycle runs for records already in host memory: stage, plan, progress, publish, finish,
  acknowledge, and the queue charge's reserve and release.
  - Its bank leaves make 117.5 of the 281: stage 69.1, publish 23.4, the bank's retire 23.7 and collect 1.3.
  - The adapter's batch and progress loop inside `dispatch_inner` (37.9) come on top.

## 1. The candidates, each answer-preserving, each to be registered with its own code gates

**I25a, the trace as a gate-set diagnostic.**
- **What it removes:** the naked door writes the complete host demand trace, 22077 lines per 32-token run, to stderr.
  That is `own_trace` (16.4), the traced adapter's pre-demand slot reads (inside `outer`), and stderr writes the
  split does not clock. REF writes none of it.
- **The change:** the trace runs only under a CLI flag the gates and cells set (`--expert-bank-trace`; a FLAGS row as
  a gate-set flag), and the default door traces nothing. The numeric program is unchanged.
- **The cell implication, registered with it:** `promo`'s integrity (one host demand sequence in the door arms) is
  read from traced arms. `q22` already traces (I22), and a traced twin of `naked` runs untimed, while the timed
  `naked` runs untraced.
- This changes what the default door prints, so the owner sees it before it lands.

**I25b, the registry entry.**
- **What it removes:** every proxy call clones the thread handle (`thread::current()`, a reference count up and down)
  to check the owner thread, then borrows the thread-local registry. The engine enters the registry four times per
  ticket (`host_resident_many`, `demand_many`, `with_bytes_each`, `finish_group`).
- **The change:** the owner check compares a thread-local copy of the thread id, with the same `WrongOwner` answer.

**I26, the host hit without a transfer ticket (structural; its design registers before its code).** When every
record of a demand is host resident, the lease is taken from the host cache directly: no queue charge, no plan, no
progress loop, no pending ticket, no publish, finish or acknowledgement. It reaches most of those 117.5 and part of `dispatch_inner`. Before
code, the design must answer each of these from the source:
- **The SLRU hit order.** Publication's `hit_at` calls must run in the same order.
- **The pins.** The bytes must not be reused while the copy reads them, so the lease's own pin must hold until the
  copy's event completes, released at the same retire point as today.
- **The queue charge's admission effect.** While a hit is in flight it holds queue bytes. Dropping it changes `used()`,
  which matters only where capacity binds. Either prove it never binds for a hit, or keep an equivalent charge.
- **The host demand trace.** Its lines must stay the same, so the integrity instrument reads one sequence.
- **The misses and mixed groups.** A miss or a mixed group keeps today's ticket path unchanged.

Its equivalence proof cannot be I24's fixture, whose state lines include the ticket inventory by design. It is a new
fixture over what a caller can observe: the bytes lent, the SLRU orders, the evictions, the host demand sequence,
every refusal, and `used()` outside the hit's flight.

**Held unless I26 is refused:** the adapter's batch without cloned ids and request (`dispatch_inner`), and the host
cache's unique-id pass by position (`bank_stage_cache`). Under I26 both run only on misses, which the qualified door
does not have in decode.

**Order.** I25a and I25b first (small, answer-preserving, one split), then I26's design, its code and one split. Each
reads under `DAY89.md` section 2's sizing rule, cumulative from `p88s`. NEED TARGET CARD for the `promo` sitting comes
when the rule reads `reaches`.

## 2. The I24 `promo` sitting, registered now: what the 285K class returns for a local cut

The lead's order after the split: register the next candidates, then NEED TARGET CARD for the `promo` sitting on both
classes. Section 1 is text, so this sitting measures I23 and I24 as landed.
- **What it can decide:** it lands phase 1 only if every cell passes under `DAY88.md` section 5 on both classes.
- **The expected reading:** the local sizing reads `short`, so the 285K class's `naked` is expected to lose still.
- **What it measures besides:** `naked` (I24) against `q22` (I22) on the 285K class is the wall's return for a
  measured local cut of 42.4 us per token. That is the translation section 2's threshold assumed (100 us there for
  155 here), and it sizes I25 and I26 honestly.

**The sitting.** `day89-box.sh`, with the cells `promo`, `promo-res` and `promo-spec`.
- **What changed from day 88:** the scripts are copies of day 88's with the promoted label `p88` renamed `i24`
  (`day89-cell.sh`, `day89-box-build.sh`), and receipts go to `c-day89`.
- **Builds:** `D89_BUILDS="i22=4b378a064 i24=1fd4b24c0"`.
- **The reader:** `day88-read.py --6a` reads all three cells. `--6a` applies section 6a (`promo-res` gen-only) inside
  the full read; without it the reader reproduces the first sitting as it read.
- **Unchanged from `DAY88.md` section 5:** arms, orders, integrity, admissibility and readings.
- **Where:** the 285K class and a 9950X, one RTX PRO 6000 Blackwell Workstation Edition each, at least 48 GB
  MemAvailable, about 40 minutes each.
- **Mirrors:** `pro-single-day89/` and `pro-single-day89-9950x/`.

**Prepared and checked before the card (`day91-cpu/`):**
- **The cell under stubs.** `dry-check-cell.log`: `day89-cell.sh` makes exactly the calls of `day88-cell.sh`, with the
  label as the only difference.
- **The scripts.** Shellcheck is clean on the new ones.
- **The reader.** `day88-read.py` without `--6a` reproduces `pro-single-day88/reading.log` byte for byte.
- **The 285K resident shape is at the edge.** Applied to the first sitting's `promo-res`, `--6a` reads `+0.0010 ...
  noise=0.0003 -> regresses` (0.153 against 0.152 s), where DAY88b's admissible rerun read `flat` at noise 1 ms.
  - This sitting's `promo-res` on that class can go either way.
  - A `regresses` there fires section 6a's last clause: the resident load's fix registers on its own.
  - Recorded here before the card so that no reading is argued after it.

## 3. The I24 `promo` sitting, read as registered (run by the lead, tree `ba8a5f4cc`; `pro-single-day89/`, `pro-single-day89-9950x/`)

Both halves are copied in byte for byte: 389 files each, the lead's manifests re-checked here OK. `day88-read.py
--rig pro-single --6a`, re-run on the copies, gives exactly the boxes' `reading.log`. Every cell's integrity is ok,
and both classes read one host demand sequence per program (door `4bdc2610c3534e42`, `nopf` `0e220d04f52d13e9`).

**285K class** (BOX46, the 285K machine BOX44 ran on, one RTX PRO 6000 WS, driver 580.173.02; builds `run-gen-i24`
`0866c7a6...` and `run-gen-i22` `4a9a2ecd...` from their commits; 22:22Z). Verbatim:
- `DAY88 PROMO ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0013 max_iqr_window=0.0013 failing=[] -> admissible`
- `DAY88 PROMO gen-only decode medians (N=10 each): naked=0.314 q22=0.313 legacy=0.311 alloc=0.314 nopf=0.380 legnopf=0.379`
- `DAY88 PROMO naked_vs_q22 gen-only decode: pooled=+0.0010 o1=+0.0000 o2=+0.0010 noise=0.0012 -> flat`
- `DAY88 PROMO naked_vs_legacy gen-only decode: pooled=+0.0030 o1=+0.0030 o2=+0.0030 noise=0.0010 -> loses`
- `DAY88 PROMO naked_vs_legacy steady window: pooled=+0.0000 o1=+0.0010 o2=+0.0000 noise=0.0010 -> matches`
- `DAY88 PROMO-RES naked_vs_legacy gen-only decode: pooled=+0.0000 o1=+0.0000 o2=+0.0000 noise=0.0010 -> flat`
- `DAY88 PROMO-SPEC spec-naked rc=0 self_consistency=PASS installed=True off_rollback=False` and `spec-legacy ... PASS`
- `DAY88 VERDICT rig=pro-single -> phase1_does_not_land (naked loses to legacy)`

**9950X class** (BOX45, driver 595.91.07; builds `0e1f9d6f...` and `6dc3bc30...`; 22:12Z). Verbatim:
- `DAY88 PROMO ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0010 max_iqr_window=0.0002 failing=[] -> admissible`
- `DAY88 PROMO naked_vs_q22 gen-only decode: pooled=+0.0000 o1=+0.0000 o2=+0.0000 noise=0.0003 -> flat`
- `DAY88 PROMO naked_vs_legacy gen-only decode: pooled=-0.0010 o1=-0.0010 o2=-0.0010 noise=0.0010 -> matches`
- `DAY88 PROMO naked_vs_legacy steady window: pooled=-0.0020 o1=-0.0020 o2=-0.0020 noise=0.0000 -> beats`
- `DAY88 PROMO-RES naked_vs_legacy gen-only decode: pooled=+0.0000 o1=+0.0000 o2=+0.0000 noise=0.0000 -> flat`
- `DAY88 VERDICT rig=pro-single -> phase1_lands`

**Read as registered.**
- Phase 1 lands on the 9950X class and does not land on the 285K class, where the door loses to its rollback by
  3.0 ms over 32 tokens gen-only and matches it on the steady window. Section 5 decides on both classes, so phase 1
  does not land.
- The resident shape reads `flat` on both: the borderline registered in section 2 did not regress.
- The prefetch wins 66 to 80 ms on both programs and both classes, as before.

**The calibration of `DAY89.md` section 2's rule (recorded before I26 is sized).** The rule assumed the 285K's gap is
the door-only CPU the local split measures, with a share of it reaching the wall.
- **This sitting cannot test it.** Translated by the rule, I24's local cut of 42.4 us per token predicts about 0.3
  ms on the 285K wall, a quarter of the cell's noise. It measured +1.0 ms at noise 1.2, `flat`.
- **The ladder can.** Each gap below is door against REF gen-only, same-window, per sitting; the local door-only
  work comes from the RTX 5090 splits.

| rung | local door-only (us/token) | 285K gap (ms) | 9950X gap (ms) | sittings |
|---|---|---|---|---|
| I20 | 656.7 | +3.0 | +3.0 | `DAY82.md` |
| I21 | about 507 (-150) | +2.0 | +2.0 | `DAY84.md` |
| I22 | 346.6 (323.7 at p88) | +2.0 (noise 2.0) | +1.0 | `DAY85.md` |
| p88 (I22's door) | 323.7 | +2.5 | -1.0 | `DAY88.md` |
| I24 | 281.2 | +3.0 | -1.0 | this sitting |

- **What it shows.** Across a local cut of 375 us per token, about 57 percent of the door-only work, the 285K class's
  gap did not close (+3.0 to +3.0 ms), while the 9950X class's closed by about 4 ms (+3.0 to -1.0).
  - The two classes run the same program on the same bytes. So on the 9950X the door-only CPU was the gap, and on
    the 285K the remaining gap is something that CPU cut does not reach.
  - Each gap is a same-window reading; comparing gaps across sittings carries sitting-to-sitting drift, which is
    this table's scope.
- **So the rule is refuted as a predictor for the 285K class.** Neither `short` nor `reaches` says whether a further
  door-only cut closes that class's gap. I26 keeps its approval as the owner's option 1 and a real CPU cut (it closed
  the 9950X class's gap in kind), but no local split can size it against the 285K gap. The 285K gap must first be
  located on the 285K itself.
- **Where the evidence points, not a finding.** On the 285K class the window matches in every sitting (`DAY79.md` to
  here), so the loss sits in the generate phase's cold part: after prefill, more GPU misses (440 against 120 per 32
  tokens locally) and a colder prefetch pipeline. The 285K box also has no systemd scope, so every cell ran under
  `taskset -c 0-11`, 12 of its 24 cores. The 285K is a hybrid P- and E-core part, its core-type layout is recorded
  in no receipt, and the local host's splits pin to P-cores. The door does more CPU work between launches than the
  legacy, so a latency-bound thread on an E-core would cost the door more than the legacy. That is a hypothesis
  until measured.

**The deciding measurement, proposed for the owner and the lead (registered in `DAY94.md` before any card):** one
sitting on the 285K class.
- The provenance records the host's P- and E-core lists (`/sys/devices/cpu_core/cpus`,
  `/sys/devices/cpu_atom/cpus`).
- `naked` and `legacy` run under the box's `taskset -c 0-11`, and again pinned to the P-cores alone, both orders,
  x 5.
- Dispatch-clock twins of both run pinned alike.
- The reading asks whether P-pinning closes the gap, and where the generate phase's extra time is on that host.
