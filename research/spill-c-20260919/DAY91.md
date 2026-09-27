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
