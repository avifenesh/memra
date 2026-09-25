# WP-A day 51: OWED item 17, design P (a pre-touched payload reserve on the hash helper; item 8's remedy)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, on `73e366a7f` (DAY49 and DAY50 recorded). Behind
`MEMRA_KV_HOST_CONTRACTS` (default OFF). Every cell `executed-not-qualified`.

## 1. Pre-registration (committed before any P code)

**What DAY49 placed** (section 3). The hash helper copies each landed staging buffer into a fresh heap `Vec<f32>`
(`staged.as_f32_slice().to_vec()`), and while no host entry frees, every page of that `Vec` is new: 38306 minor faults
per 156.9 MB, 23.73 ms of copy against 7.64 ms where the LRU frees entries and the heap reuses their memory. The 16.09
ms sit on the helper's job, which the publication waits for: the steady wall t0 to publication reads 101.0 to 101.5 ms
against 88.4 to 88.8 (landed after 2 tick-top polls against 1). Every demote pays it until the host tier reaches its
budget. The faults are the price of new resident memory; they can be moved, not avoided, so the design moves them off
the job and into the helper's idle time.

**Design P.**

1. **The reserve.** The hash helper owns a reserve of heap `Vec<f32>` buffers whose every page is already written, keyed
   by exact length. Its target is the staged payload lengths of the last `Hash` job the helper served (one image's
   recurrent planes), clamped so the reserve never holds more than the host tier's budget in bytes.
2. **The copy.** A staged payload of `L` floats takes a reserve buffer of length `L` if one is there and writes it with
   `copy_from_slice` from the staged bytes (every element); otherwise it allocates as today (`to_vec`). The payload's
   bytes are the same either way: one numeric program.
3. **The refill.** On the helper thread itself, at the top of its job loop, before it blocks for the next job: while
   the reserve is short of its target, the helper allocates and writes one buffer at a time (`resize` with a value the
   compiler cannot see is zero, so every page is written), and checks its job channel (`try_recv`) before each buffer. A
   job that is waiting is served first; the refill resumes after it. No new thread; no job waits behind more than one
   buffer (at most 3 MiB of writes on the 27B).
4. **Retarget.** After the payload copies of a `Hash` job, before its reply: the reserve drops every buffer the job did
   not take, releases its charge, and takes the job's staged lengths as its new target. A job of another shape (another
   model) finds no buffer of its lengths, copies as today, and retargets the reserve.
5. **The charge.** The reserve is resident memory, so it is charged on the tier governor's pageable ledger under its own
   tenant (a digest in a domain disjoint from every `tenant_salt`, as the staging set's is), for the target's whole
   byte count, BEFORE its first buffer is allocated. A refused charge allocates nothing: the reserve stays empty until
   the next retarget and every copy takes today's path. The charge is released at the retarget (the image's own
   pageable charge, taken at its pre-submit, already covers the payloads the job took) and when the helper exits (the
   latch, shutdown, the `hash-helper-gone` arm). The ledger's pageable capacity gains one term, one host budget (twice
   becomes three times), so the reserve's charge is never what refuses a demote the OFF arm would have made (ruling 15's
   admissibility: the reserve holds at most one budget).
6. **Lines** (log only). The helper split gains `; reserve H of N staged`; a finished refill prints `[prefix-host]
   payload reserve ready: N buffers, B MB in X ms (minflt +M, yielded Y time(s)), charged to the governor's pageable
   ledger`; a refused charge prints the governor's refusal. The first demote of a context has no reserve (no shape yet).
7. **Stated limits, not owed.** The reserve holds one image's recurrent bytes of resident heap beside the staging set's
   pinned copy of the same size, from the first demote on (about 157 MB on the 27B). A context serving two models whose
   demotes alternate retargets on every job and gains nothing.

**Acceptance, stated before any code.** Arms: `base` (this commit's tree: the tip after DAY49's split lines) and `p`
(P's tip). The demote, promote and hump cells are S4's (DAY48) and the chain cell item 15's (DAY43), with S's bounds;
two new clauses, (b) and (f), and (c)'s wall made a gain.

- (a) Semantics. CPU cells: a census `day51_the_payload_reserve_is_the_copy_program` (the copy writes a reserve buffer
  only through `copy_from_slice` of the staged slice and otherwise `to_vec`; the refill runs only at the loop's top and
  calls `try_recv` before every buffer; the charge is reserved before the first allocation and dropped at the retarget;
  the pageable capacity term; no decision reads the new split fields or the refill figures), and unit cells on the
  reserve itself: a hit's bytes are bitwise the staged bytes (`-0.0`, NaN payloads, denormals); a waiting job is
  returned before any buffer is allocated, and a job sent mid-refill is returned before the reserve completes; the
  governor's pageable use equals the target's bytes from the first buffer on and returns to its baseline after the
  retarget and after the helper exits; a refusing governor leaves the reserve empty and `take` misses; the reserve never
  holds more than its cap; a shape change frees the old buffers. On the target card: the unit cells (S4's set, the
  door's `option_b_` and `option_c_` cells, the new ones) and every gate on P's binary green: identity x4, failure x2,
  the fault gate default and plain, twin x2, the hit gate OFF and ON, and the pause gate.
- (b) The mechanism (the demote A/B below, P's boots, the steady demotes: the second and later of each boot), per
  order: P's copy minflt median at most 0.25 x pages (pages = copy bytes / 4096), P's copy ms median at most base's
  minus 8.0 ms, and at least 90% of P's steady demotes read `reserve N of N staged`.
- (c) The price P exists to cut, per order: the steady `wall .. t0 to publication` median on P at most base's **minus
  8.0 ms**, and the demoting intruder's e2e median on P at most base's plus 1.0 ms.
- (d) The promote A/B, per order: PIN (the second and later `promote: .. in Y ms` of each boot) median on P at most
  base's plus 1.0 ms, and the promoting intruder's e2e median at most base's plus 1.0 ms.
- (e) The hump (S's cell: four door-ON boots `xgpp xp xp xgpp`, `stall_cell.py --mode demote --n 8`): P's median HUMP at
  most 0.15 ms, with the G'' control humping (above 0.15) in the same hold; a control that does not hump makes the cell
  unread, and it repeats once.
- (f) Where the first touch is already absent (the demote A/B at `MEMRA_KV_HOST_MB=480`, the LRU freeing from the
  fourth demote on; the steady demotes the fourth and later), per order: the wall median on P at most base's plus 2.0 ms
  and the copy ms median at most base's plus 1.0 ms.
- (g) The chain (item 15's `promote-long` cell, where a hit parks on a `Demoting` entry and promotes the moment it
  publishes, so its `Sources` job meets the refill): per order, the chained request's e2e median on P at most base's
  plus 1.0 ms and the first intruder's e2e median at most base's plus 1.0 ms.
- Readings, no clause: the refill line per demote (time, faults, yields) in each regime; each boot's server `VmRSS`
  and `VmHWM` before its stop (the reserve's resident cost); the chain's steady helper time (DAY43's receipts read the
  chain's helper at 123.5 ms from its fourth demote on: its demotes replace freed entries, so the chain is expected to
  gain nothing from P).

**The cells** (one RTX PRO 6000 Blackwell, the 27B NVFP4 MTP artifact; `pro-single-p/`, one collector hold per cell):
the demote A/B (`--mode demote --n 5`, `MEMRA_KV_HOST_MB=8192`, `MEMRA_PREFIX_CACHE_MB=256`), the free A/B (the same at
`MEMRA_KV_HOST_MB=480`), the promote A/B (`--mode promote --n 5`), the chain A/B (`--mode promote-long --n 5`,
`MEMRA_PREFIX_CACHE_MB=448`): each 20 boots, `o1 = base p x5`, `o2 = p base x5`, door ON, each boot's start
temperature and SM clock recorded, 250 ms telemetry; then the hump cell, the gates, the hit gate, the pause gate and
the unit cells. One reader, `day51-reading.py`, written before the cells run, prints every clause.

**The rule.** P becomes the door's copy program on the RTX PRO 6000 class if (a) to (g) hold, each in both orders.
Otherwise P is reverted in one commit with its red receipts banked, the failed clause recorded as read, and any
revision pre-registered anew. No bound here moves after a result.

**Predictions.** (b): copy about 7.6 ms, faults near 0, every steady demote a full hit; (c): the wall about 12.7 ms
shorter (landed after 1 poll instead of 2), e2e flat; (d), (f) and (g) flat within 0.5 ms; (e) about +0.02; the
refill about 16 to 20 ms with about 38300 faults at `8192` and a few ms with about none at `480`; VmRSS about 157 MB
higher on P.

**What each card decides.** Each card its own. The target card first, on the class DAY49 read (the 9950X class); a
different host class reads its own verdict. The 5090 half after the card's reset (owed with S4's and V's halves).

**Budget.** 0.5 agent-day: the code, census and unit cells 0.2, the sitting and its reader 0.1, the card 0.2 (about
2.5 hours of card time).
