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

## 2. P as built (`d82738c14`), the CPU cells, and the sitting prepared

- Built as section 1 states, in `crates/memra-server/src/worker.rs`: `HostPayloadReserve` (`take`, `retarget`,
  `refill_step`, `refill_until_job`), owned by the helper thread (`HostHashWorker::spawn(fault, reserve)`, the one
  production reserve `HostPayloadReserve::new(governor.clone(), hpx.budget as u64)`); the helper's `for job in jobs_rx`
  is now a loop whose top is `reserve.refill_until_job(&jobs_rx)`, then the blocking `recv`; the copy is
  `match reserve.take(src.len()) { Some(mut v) => { v.copy_from_slice(src); .. } None => src.to_vec() }`; the retarget
  runs after the payload map and before the reply is built, on every `Hash` job; `capacity.pageable =
  thrice(host_budget)`. The charge is one `ResidentCharge` for the target's bytes under `host_payload_reserve_tenant()`,
  taken by the first refill step of a target. A yield is counted when a job interrupts a refill that has written at
  least one buffer of its target. No new env read, no new fault value.
- Censuses moved with the code (their subject changed, no bound): the spawn string and the two signature lookups of
  `every_path_that_meets_a_hashing_demote_meets_it_through_the_same_settle`, the signature lookup of
  `day41_the_sources_faults_key_on_the_first_sources_job`, the copy line `the_d2h_spans_ride_the_ticket_in_the_stated_order`
  and `day49_the_split_lines_are_log_only` pin (now the reserve's `take` line, still before the hash), and day 49's
  `thread_minflt()` count, now 7 (the definition, two reads around each of the two copies, two around the refill's
  buffer). The test sites that spawn a helper pass a reserve on its own governor (`test_payload_reserve`).
- New: the census `day51_the_payload_reserve_is_the_copy_program` and the cells
  `day51_a_reserve_hit_is_the_staged_bytes_bitwise`, `day51_the_refill_yields_to_a_waiting_job`,
  `day51_the_reserve_charge_the_cap_and_the_shape_change` (section 1 (a)'s CPU half, every item).
- CPU cells, green: server lib `925 passed; 0 failed; 25 ignored`; clippy `-D warnings` (memra-server, all targets);
  `cargo fmt --all -- --check`; `git diff --check`; `tools/check-flags.sh` (`no uncovered runtime names`). The engine and
  tier crates are unchanged.
- The reader `day51-reading.py` was checked on a synthetic fixture built from DAY49's and item 15's mirrored receipts
  (relabelled arms, a `reserve 97 of 97 staged` suffix added to one arm's lines): every clause parses and prints, and
  a boot removed, a control that does not hump and a red gate exit read INCOMPLETE, UNREAD and FAIL as they must. The
  fixture is gone.
- The sitting `pro-single-p/`: `build.sh <tip> e4de9c804` (p, base, gpp from one clone; base is section 1's commit, the
  tree before P's code; the tip's test binaries), `driver.sh` (the four A/B cells `demote`, `free`, `promote`, `chain`
  through `ab.sh`, the hump cell, the gates with the pause gate, the hit gate, the unit cells, then the reader), about
  2.2 hours of card time on one RTX PRO 6000 Blackwell with the 27B artifact at
  `/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`. The host class is recorded (`host-shape.txt`); the 9950X class is
  DAY49's, so it is the class this reading is registered for.

## 3. P on the target card, as it ran (BOX22; `pro-single-p/box/`)

- The host reads `AMD Ryzen 9 9950X 16-Core Processor` (32 CPUs, 123 GB): the 9950X class, a different machine from
  DAY49's (a 9950X3D2), so its arms are compared only with each other. One RTX PRO 6000 Blackwell Workstation Edition.
  The model sha256 `1facf36c2db359dc..`, verified in every cell.
- Build `rc=0` (11:46Z to 11:57Z): p `2692df5612b141a7..` (tree `a2db13d67`), base `b9ca4480015a1531..` (`e4de9c804`),
  gpp `541acb7e25248598..` (`358749c9f`); markers `p reserve-ready wording: 1`, `base .. 0`, `gpp .. 0`. The build
  step log notes `the p rebuild differs in bytes`, as S2's, S3's and S4's builds did; the arms run the first p build,
  the unit cells the tip's test binaries.
- Cells, one collector hold each: `ab-demote-cell rc=0` 12:14:59Z, `ab-free-cell rc=0` 12:33:03Z, `ab-promote-cell rc=0`
  12:51:00Z, `ab-chain-cell rc=0` 13:10:09Z, `hump-cell rc=0` 13:15:24Z, `gates rc=0` 13:36:22Z, `hitgate-off rc=0`,
  `hitgate-on rc=0`, `unit-cell rc=0` 13:37:57Z. Every A/B cell 20 boots, 20 of 20 `STALL REPLAY: PASS`.
- Thermal regime: boot starts 28 C (the hold's first boot, the card idle) and 57 to 68 C after; the 250 ms telemetry 28
  to 75 C, SM up to 2842 to 2865 MHz.
- Mirror: 1041 files, 1040 of 1040 manifest entries OK, 0 mismatched (`MIRROR-CHECK.txt`); the three ELFs by hash only
  (`BINARIES.box.sha256`). Box scratch removed.

**The reading, verbatim** (`box/reading-day51.log`, the box's run of the reader registered in section 1):

```
DAY51 P (b) cell=demote order=o1 minflt p-median (0.25 x pages)=+0.00 rule <=+9576.42 | copy p-minus-base=-15.17 rule <=-8.00 | hits fraction of steady demotes=+1.00 rule >=+0.90 -> PASS
DAY51 P (b) cell=demote order=o2 minflt p-median (0.25 x pages)=+0.00 rule <=+9576.42 | copy p-minus-base=-15.32 rule <=-8.00 | hits fraction of steady demotes=+1.00 rule >=+0.90 -> PASS
DAY51 P (c) cell=demote order=o1 wall p-minus-base=-12.40 rule <=-8.00 | e2e p-minus-base=+0.37 rule <=+1.00 -> PASS
DAY51 P (c) cell=demote order=o2 wall p-minus-base=-12.40 rule <=-8.00 | e2e p-minus-base=+0.32 rule <=+1.00 -> PASS
DAY51 P (d) cell=promote order=o1 pin p-minus-base=+0.20 rule <=+1.00 | e2e p-minus-base=+0.12 rule <=+1.00 -> PASS
DAY51 P (d) cell=promote order=o2 pin p-minus-base=+0.20 rule <=+1.00 | e2e p-minus-base=+0.12 rule <=+1.00 -> PASS
DAY51 P (f) cell=free order=o1 wall p-minus-base=+0.10 rule <=+2.00 | copy p-minus-base=-0.01 rule <=+1.00 -> PASS
DAY51 P (f) cell=free order=o2 wall p-minus-base=+0.10 rule <=+2.00 | copy p-minus-base=-0.00 rule <=+1.00 -> PASS
DAY51 P (g) cell=chain order=o1 chain p-minus-base=+1.40 rule <=+1.00 | first p-minus-base=+0.23 rule <=+1.00 -> FAIL
DAY51 P (g) cell=chain order=o2 chain p-minus-base=+1.54 rule <=+1.00 | first p-minus-base=+0.05 rule <=+1.00 -> FAIL
DAY51 P (e) hump xp=+0.022 rule <=0.15 control xgpp=+0.514 (humps) -> PASS
DAY51 P -> FAIL
```

- (a), read by the same reader: every gate exit 0, and every gate's own line green: identity x4 `KV-HOST-SPILL IDENTITY
  GATE: ALL GREEN (teeth=0)`, failure x2 `ALL GREEN`, the fault gate default and plain `KV-HOST-CONTRACT-FAULT GATE: ALL
  GREEN` (255 ok each), twin x2 `cached_ok=7/7`, `KV-HOST-PAUSE-DEMOTE GATE: ALL GREEN` (40 ok), `SPEC-ON-CACHE-HIT
  GATE: ALL GREEN (qwen)` OFF and ON (61 and 68 ok); the unit cells `unit-cells parallel=3/3 engine-serial-rc=0
  door-rc=0 cpu-rc=0 engine-census-rc=0 tier-rc=0` (the door cells 18 passed, the CPU censuses 38 with the four day-51
  cells). P passes (a).

**Verdict, as registered: P FAILS (g) in both orders** (the chained request +1.40 / +1.54 ms against +1.0) and passes
(a) to (f). By section 1's rule P is not the door's copy program on this class, and it is reverted in one commit.

**What the cell read** (readings, no clause):

1. The mechanism works as designed where the tier fills: every steady demote a full hit (`reserve 96 of 96 staged`),
   the copy 7.91 ms against 23.08 / 23.23 with no faults, the publication one poll earlier (wall 88.40 / 88.50 against
   100.80 / 100.90). The refill runs 21.3 ms per demote there (38306 faults, off the job, never yielded in that cell).
2. Where entries free (the free cell, the promote cell, the chain), the refill reuses freed memory (7.6 to 8.0 ms at
   5376 to 6144 faults, 5.1 ms at none in the chain) and the copy is the same 7.9 ms on both arms: P buys nothing
   there, as section 1 predicted for the chain.
3. Where the chain's +1.4 ms sits (`readings/publication-split.log`, `day51-publication-split.py`, written after the
   verdict): the chain's demote holds the owner thread 36.82 / 37.16 ms on P against 35.96 / 35.92 on base, and the
   difference is in the publication's `take-back bind and publish` segment, 10.28 / 10.43 ms against 9.39 / 9.36 (+0.89
   / +1.07). The pre-submit (26.00 / 26.11 against 25.81 / 26.01) and the helper (122.6 against 122.5) are flat. In
   the chain, every publication replaces the previous host copy of the same prompt, so that segment frees the replaced
   entry's heap payloads and its 32 pinned leases on the owner thread. The cause of the extra millisecond on P is not
   placed by these lines (the segment is not split).
4. The reserve's resident cost, per boot at its stop: VmRSS +149 to +153 MB on P in every cell (3038 against 2889 MB
   in the demote cell), one image's recurrent bytes, as stated.
5. Two owner-thread prices this shape shows on the base arm, beside the design: a long demote's pre-submit holds the
   owner 25.8 to 26.0 ms, 25.3 to 25.5 of it in the pinned lease allocations (OWED item 19), and each publication that
   replaces a host entry holds it 9.4 ms in `take-back bind and publish` (the replaced entry's frees: OWED item 14's
   subject). Both are the tenant's tick.

**What follows.** P is reverted (`git revert` of `d82738c14`, code and its TESTING.md entry, in one commit). Item 17
stays open: a revision is pre-registered anew before its code. Its obvious shape, from reading 2, is a reserve that
refills only while the copy it replaces would fault (the tier filling) and holds nothing where freed memory is reused;
it must first place the chain's extra millisecond (a split of the publication segment), because a revision that simply
avoids the regime would not say whether reserve-provenance buffers cost more to free.
