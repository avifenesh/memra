# Serving c8 on a 24 GB card: queue order, prefill order, seed batching, cache headroom

Date: 2026-11-07. Lane `fix/5090-c8-serving`, receipts `research/c8-serving-20261107/`. Card: one RTX 5090
Laptop GPU (24 GB), Qwen3.8-27B NVFP4 + Q5_K with its MTP head, `MEMRA_CTX=32768`.

## Problem

At agentic c8 (29 turns of 18 coding-agent sessions, 1.4k-17k prompt tokens) main failed requests. It
answered 1-2 HTTP 408s per prime pass at the 90 s first-token deadline. In the greedy measured pass, one
CUDA OOM per run ended all eight in-flight streams after their first tokens. On the spec-cache-mixed shape
at c8, the prefix cache cost throughput and doubled batch TTFT against the cache-off arm.

## Chosen

1. **Head-of-line memory order.** A request the VRAM gate defers holds every later request of that
   admission pass on a device it shares: interactive requests keep their order, dark-lane ones shed. The
   hold comes after the refusals that need no memory (slot capacity, the template, the first-token gate).
   It applies only to a request that can fit once its device's sessions retire and the evictable prefix
   entries go. A request that cannot fit even then keeps main's behavior: it waits alone, and the idle-box
   check refuses it. The cause it removes: later requests that fit were admitted past the deferred one, so
   a 17,166-token request saw free memory hover under its 5,048 MB charge for its whole 90 s window while
   15 later requests got in.
2. **Earliest-deadline-first prefill, while a prefill is at risk** (`deadline_prefill_rows`). The round robin
   stays the default. A tick advances rows in deadline order only when some row would miss its deadline
   under the round robin but would make it alone. With requests waiting, the estimate assumes finishing
   peers are replaced. Only sessions the tick can advance are rows (n>1 choice members and fanout rows
   still forming are left out and never held back). A row without a wire deadline orders by its arrival
   plus the default streaming deadline. The cause it removes: a 12k-17k prompt shared the device four or
   five ways for its whole prompt (42-68 s of priming).
   Update, 2026-11-09: with no request waiting, the estimate no longer lets finishing peers leave. It
   keeps the tick's current takes for every take the row still needs
   ([`PREFILL-WORKSPACE-HELD.md`](PREFILL-WORKSPACE-HELD.md)). The held prefill charge admits a cold
   burst instead of queueing it, so the old form was in use there, and the slowest first token sat at
   85-89 s in nine of nine runs.
3. **Concat rows take the single prime's take** (`batch_row_take`). A cold session whose grid-aligned seed
   lies inside its prompt used to prime alone for the whole prompt; it now joins the concat batch with its
   take ending on the seed. The cause it removes: at c8, eight unique 100-token prompts each primed alone
   up to their 64-token seed, so `prime_wait` p50 was 420 ms with the cache against 4 ms without. The same
   function merges a remainder under 16 tokens into the call, as the single path does. Before, a batched
   1030-token row primed 1024 tokens and fed its last 6 through tokenwise `decode_step`, which a single
   prime of the same prompt never did.
4. **The prefix cache yields the transient reserve after an insert** (`prefix_yield_transient_reserve`).
   Before a decode phase, if the cache published since the last check, a discrete card reads effective free
   memory (driver free plus the pool's cached bytes, after a fence). It compares that with the transient
   reserve admission charges every admitted request (`admission_reserve`). On a shortfall it evicts the
   oldest unleased entries and trims their bytes back to the driver. The cause it removes: inserts made
   after admission spent the room the admitted sessions' decode transients rely on, and nobody re-checked
   it before the next decode step. Driver free fell from 3.8 GB to 83 MB, and the fa partial pool's next
   grow failed inside the batched decode step.

What changes numerically. No prime call's boundaries change, except the sub-16-token tail in item 3,
which now matches the single path. What does change is which calls run concatenated and which run alone,
because seed rows join batches and the deadline order picks rows. That puts those prompts in the
documented batched-prime near-tie class (`docs/SERVING.md` "First-token cross-config drift"), as any
change in batch composition does on main. The greedy comparisons below measure it.

## Measured

Scored agentic c8 A/B: main against this tree, six fresh boots interleaved A B B A A B, T=1, 768 new
tokens, a prime pass and then a measured pass of 29 turns each. Medians are over the three runs per arm. A
408 counts as a wait that ended when its error arrived.

| agentic c8 | main | this tree |
|---|---|---|
| failed requests, both passes, 6 x 29 per arm | 15 (all 408) | 0 |
| measured pass: first-token wait p50 | 19.5 s | 35.7 s |
| measured pass: first-token wait p99 | 90.0 s | 65.0 s |
| measured pass: median wait, turns of 12k+ tokens / under 4k | 85.8 s / 8.7 s | 48.2 s / 20.5 s |
| prime pass: first-token wait p50 / p99 | 26.7 s / 90.0 s | 41.3 s / 81.5 s |
| measured pass: tokens served / tok/s over its wall | 9,910 / 41.0 | 10,623 / 40.0 |
| device free floor, lowest run | 691 MiB | 1,203 MiB |

The cost is the median. Short turns now wait behind a deferred or at-risk long one, where on main they
overtook it and the long one timed out. That is not lost work: at equal work the mean wait is unchanged.
The traced runs gave main a 900 s first-token deadline so it serves every turn too (two runs per arm,
measured pass, percentiles and mean pooled over the 58 turns, tok/s the median run, a separate window from
the table above):

| agentic c8, every turn served | p10 | p50 | p90 | p99 | mean | turn total mean / p99 | tok/s |
|---|---|---|---|---|---|---|---|
| main, 900 s deadline | 5.1 s | 19.4 s | 111.4 s | 152.1 s | 34.5 s | 75.0 s / 166.2 s | 35.6 |
| this tree | 12.7 s | 37.4 s | 53.9 s | 69.6 s | 33.3 s | 68.5 s / 141.4 s | 40.0 |

Main keeps its 19.4 s median at equal work, so the order does delay turns that would have been fine
(turns under 4k: 10.3 s against 20.7 s). It moves the wait from the long turns to the short ones. Main's
tail is the price of its median: at its real 90 s deadline those p90 and p99 waits are the 408s. The order
also finishes the same work 12% faster, because main's starved long turns stretch the pass, and whole
turns end sooner: mean 68.5 s against 75.0 s, with the median turn equal (66.5 s against 67.3 s). This record
takes that trade: no request fails and the tail halves; the short turns wait about 10 s longer. It is the
order vLLM keeps (its V1 scheduler stops at the first waiting request it cannot allocate). c4, c1, the
mixed shape and the greedy cell on the final tree are in
[`research/c8-serving-20261107/RESULTS.md`](../../research/c8-serving-20261107/RESULTS.md).

The single-run diagnostics below are the ones the choices were made on
([`research/c8-serving-20261107/README.md`](../../research/c8-serving-20261107/README.md)).

| c8 prime pass (29 turns), single runs | served | max first token |
|---|---|---|
| main, cache on, 90 s deadline | 27/29 (2 x 408) | 82.7 s |
| main, 900 s deadline | 29/29 | 110.1 s (3 past 90 s) |
| hold on every defer, no deadline order | 29/29 at 90 s; 95.8 s max without the deadline | 84.1 s |
| hold + deadline order (replacement estimate) + seed rows + the cache guard on the driver floor (`cand6`), 90 s | 29/29 | 74.7 s |

Greedy c8 (prime pass, then a 256-token measured pass): main failed 11 of 58 requests in each of two runs
(3 x 408 and one decode OOM ending 8 streams); `cand6` served 58 of 58. The rows both
runs served were byte-identical to main's (prime 27/27, measured 20/20). Mixed shape, cache on: batch
TTFT p50 1,304 ms on main, 935 ms with seed batching, 736 ms with the cache off; greedy text identical
on 95/95 rows in each pairing.

## Rejected

- **A hold that gives way while the deferred request's deadline allows it, plus a prefill rescue that
  shares the device with every row still fitting beside the at-risk ones** (2026-11-08, measured as
  `cand9`, three traced runs). A later request went past the head when an upper bound on the head's first
  token still met its deadline. The bound ran every session on the device to its whole remaining budget
  at the slow decode step. The at-risk prefill kept the shortest other rows while every deadline in the
  set still fit. It cut the measured-pass median to 26.7 s with no failure, but raised everything above
  it (p75 56.3 s against 41.5 s, p90 69.7 s against 53.9 s, p99 86.2 s) and cost 8% of tok/s (37.0
  against 40.0). The mean was the same (34.8 s). Short turns gained nothing (20.6 s under 4k); the gain
  went to mid-size turns. Both passes put their slowest first token at 86-90 s against the 90 s deadline,
  because a set packed to its deadlines has no margin for the estimate's error. An independent review
  then found optimistic terms in the bound: the head's own spared cache entry counted as free, per-call
  step costs stood in for a session's per-token wall, and held requests behind the head were overtaken
  unchecked. A corrected bound can only be more conservative, so it lies between the two measured points
  on the same trade line. Dropped as a trade with no clear win.
- **Holding the queue only while the deferred request's deadline is at risk** (its prefill need against
  its slack). It was meant to keep main's median: the hold alone, in a single run, raised the prime pass's
  median first token from 22.9 s to 35.3 s. It did not keep it (41.1 s with the earliest-deadline order,
  which sets the median here). In a decode-heavy measured pass it had no term for the wait until decoding
  sessions free memory: a 13,232-token request looked safe, was overtaken for about 50 s, was held too
  late and was answered 408. The unconditional hold has no estimate to get wrong.
- **Earliest-deadline-first without the replacement estimate.** It predicted a 17k prompt would finish in
  about 31 s under the round robin. It took 60-69 s, because every finishing peer was replaced at once. The
  order engaged three times briefly and the slowest first token stayed at 87.4 s.
- **Pure oldest-first prefill.** It bounds the tail (simulated max 64-70 s against 96-137 s), but a short
  chat behind a long prompt waits for the whole long prefill. That is the case the round robin exists for.
- **A longer streaming deadline on discrete cards** (the unified-memory derivation applied here). The
  formula gives 90 s at 32,768 context on 82 SMs, so it changes nothing on this card. A larger value hides
  the scheduling and admission causes behind a policy change, and both causes are fixable.
- **A smaller default prefix budget on 24 GB.** The 408s happen with the cache off too (1 of 29), so the
  budget is not their cause. The mixed-shape cost is the solo seed prime, not the cache's bytes. A fixed
  smaller budget would give up hits where memory is free. The headroom yield shrinks the cache only while
  the driver is short.
- **The step-OOM retry door (`MEMRA_BATCH_OOM_RECOVER`) as the decode-OOM fix.** The failing allocation
  comes after the step's first KV append (step guard `Touched`), so the retry cannot run there.
- **A cache guard on the 2 GiB driver-headroom floor** (driver free only, trim first, then evict). On the
  mixed shape the pool keeps a residue the trim cannot return, so the driver sits 14 MiB under that floor
  in steady state. The guard would evict an entry on every insert there, with two syncs per tick.
- **The unified-memory live-free insert clamp on discrete cards** (rejected by memra#1090 for losing the
  shared entry under load). The reserve yield differs in two ways. It acts once per decode phase after
  inserts, not inside every insert. It frees only the shortfall against the reserve admission already
  charged, not against the driver headroom plus the reserve.
- **Keying these to the RTX 5090 class.** None is a tuned default: each engages only under the condition it
  fixes (a memory defer, a prefill that would miss its deadline, a seed inside a batched prompt, effective
  free under the reserve). Where that condition never occurs, the code path is the old one. One-card
  evidence is what this record has; a PRO 6000 run is owed for the measured effect there, not for
  correctness.
