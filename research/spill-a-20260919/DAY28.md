# WP-A day 28: option (a) landed under ruling 39: the bundle hash off the tick (Move 1 owed item 2)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Start `8bfd3d103` = remote. Merged `origin/main` `22f7489a3`
(after #647: my day 26, C days 31 and 32, the boundary rule) `--no-ff`, then the lead's local integ43 ref
`lane/spill-integ43-20260922` `7cd2fc95d` (no remote existed; my day 27 and C day 33 as the lead integrated them)
`--no-ff`; the one `HOSTPREFIX-DOOR.md` conflict took the integ side; merge `c3e417a12`, pushed in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (logged; no qualification claimed). Rig: the target card (BOX3, one
RTX PRO 6000 Blackwell, 96 GB, 600 W, `BOX-ACCESS.md`) and the local RTX 5090 rig where its lock allows; no
cross-card comparison. Every cell `executed-not-qualified`.

## 1. Pre-registration (this section is committed before any engine code moves)

**Ruling 39 (lead): option (a) is approved as `DAY27.md` section 2 states it**, with these additions, restated here as
the contract the code below is written to:

1. The receipt term stays SHA-256 over the same bytes: `memra_engine::cache::tiered::checksum` (the `valid-bytes`
   frame) over `f32s_as_bytes(payload)`, the same `StateBundle`, the same wire. No program change: the helper runs the
   one function the owner thread runs today, over the same `Vec<f32>` bytes, and hands the digest back; the layout is
   rebuilt with the handed-in sums, byte for byte the same program over the same bytes. What the identity gate's
   receipt term proves before and after is identical: per KV plane the bundle checksum equals the D2H receipt (hash 2 =
   hash 1, and hash 3 at promote); for the recurrent state, the logits and the hidden row, the bundle checksum is the
   digest of the bytes that were in the heap at publication and nothing else.
2. `Hashing` is a state of `PendingDemote` (`Demoting` still; the entry is in neither index, a hit on its prompt is a
   miss) with the same fail-closed discipline as every other pending state. Every path that settles a `Demoting` entry
   meets `Hashing` through the SAME settle (`host_demote_settle_with` dispatches on the phase before the contract
   step): the tick-top `Poll`, and the `Block` waits of a second demote of any route (the eviction sink and the
   admission reclaim ladder), a promote (the hook and the admission probe), and a tenant purge. A `Block` wait prints one
   line naming what it waits on: the hash helper's reply for that ticket; no CUDA event (the KV planes settled and the
   ticket retired and acknowledged BEFORE the hand-off) and no join (the helper joins at shutdown and at the tier's
   latch, never inside a settle). The parked-only wait and the orphan grace gain no arm: a demote parks no request.
   Trims gain no arm: a `Hashing` demote holds no device bytes in flight and its heap payloads are not pool memory (the
   day-17 census: trims never met a `Demoting` entry, and still do not). Shutdown drops a pending demote typed (nothing
   published after a stop, the capture drain's rule) and joins the helper. `host.disable` (every latch) joins the
   helper. The idle wait already caps at 2 ms while `hpx.demoting` is `Some`, so a `Hashing` entry publishes on an idle
   box without waiting for a request.
3. One long-lived helper per `HostTierContext` (`HostHashWorker`, spawned when the door builds the context, one
   `std::thread`, no pool, never a thread per demote), a job channel in and a reply channel out; the helper OWNS the
   heap payloads while it hashes (`Vec<f32>` moved out of the image, moved back with the digests; no copy). The KV
   planes' checksums (about 0.9 ms on the target card, the leases hold an `Rc` and a CUDA event) stay on the owner
   thread inside the bind; `Pinned` f32 payloads (only with an arena, which the door refuses) are hashed on the owner
   thread as today. An image with no heap payload to hand off publishes directly as today (no `Hashing` phase).
4. Fail-closed arms, each a typed `demote failed (...)` line, nothing published, the entry dropped whole (its shell's
   planes return to the pool), the tier latched off (`TIER DISABLED`), no ticket to leak (retired before the hand-off):
   (i) `hash-helper-gone`: the reply channel is closed (the helper exited or panicked); (ii) `hash-never-lands`: the
   digests have not landed `HOST_HASH_DEADLINE` after the hand-off. **N, pre-registered: a wall deadline of 10 s from
   the hand-off, checked at every tick-top poll and used as the `Block` wait's `recv_timeout`; not a tick count, because
   a tick count latches a busy 100 ms tick late and an idle 2 ms poll early; the line reports the polls and the wait.**
   10 s is about 130x the target host's hash of the largest image class (157 MB at 2.15 GB/s = 73 ms; DAY27) and about
   270x the 5090 host's; (iii) a reply whose ticket or byte count is not the payload's (the same latch).
5. The two fault values land as one-shot arms of the EXISTING `MEMRA_KV_HOST_FAULT` door, read once at boot into the
   helper (`hash-helper-gone`: the helper exits on its first job, dropping it; `hash-never-lands`: the helper hashes its
   first job, discards the reply and stays alive), with the `docs/FLAGS.md` row's value list extended in the same
   commit. No new `MEMRA_*` name; no flag for the phase: the door is the switch.
6. Lines. The copy's settle keeps its figure and gains a hand-off tail: `[prefix-host] demote copy complete off the
   tick: ticket seq=S complete after P poll(s), X ms from submission to completion (mode); N heap payloads (M MB)
   handed to the hash helper` (the day-17 line `demote published off the tick: ...` stays for the direct path only,
   since publication now follows the digests). Publication prints the day-15 `[prefix-host] demote: T tokens, M MB in
   W ms (...)` line unchanged (`W` is wall time t0 to publication, as today) and then one ledger line: `[prefix-host]
   demote digests landed off the tick: ticket seq=S, N payloads (M MB) hashed in H ms on the hash helper, landed after
   Q poll(s) (mode); the owner thread held O ms across the demote: pre-submit A, copy settle B over P poll(s), hashing
   polls C, take-back bind and publish D; owner in-completion I ms; wall W ms t0 to publication`, with `I = A + C + D`.
   The `Block` wait's line: `demote hashing settled synchronously by <why>: waiting for the hash helper's reply for
   ticket seq=S (...)`. The join: `hash helper joined (<why>)`.
7. Bitwise digest unit cell (CPU, `worker::tests`): a fixture image's heap payloads (recurrent planes of several
   sizes, logits, hidden; deterministic fill) hashed on the test thread through the one `checksum` program in the order
   the bind consumes them, and by a real spawned helper; every digest equal bitwise, every byte count equal, every
   payload back byte-identical in its slot. Plus CPU cells for the state machine: a `Poll` before the reply keeps the
   state and publishes nothing; the reply reaches publication (on the CPU the bind refuses by name, the day-17 proof
   that publication was reached); `hash-helper-gone` and `hash-never-lands` (a short injected deadline) each a typed
   `Failed`, latched, `demoting` consumed, under `Poll` and under `Block`; a forged mismatched reply refuses and
   latches; the source census of item 2; the `from_door` table.
8. Fault gate cells (`tools/kv-host-contract-fault-gate.sh`), the shape of the existing demote cells: r1 P_A seeds E_A;
   r2 P_B seeds E_B, the byte budget evicts E_A whose demote takes the fault after its copy completes; r3 P_C seeds
   E_C and evicts E_B. Assertions: three completions served (the tick program serves); door ON; exactly one
   `demote copy complete off the tick ... handed to the hash helper` line (the hand-off happened once); exactly one
   typed injected refusal (`demote failed (tier hash helper gone ...)` or `demote failed (tier hash digests never
   landed ...)`); `TIER DISABLED` exactly once; nothing published in the boot (no `[prefix-host] demote: ` line, no
   `digests landed` line); no quarantine (`no longer whole` absent); no ticket leaked (`Capacity`, `leaked` absent);
   the only other host-tier refusal allowed is r3's `demote refused: the tier latched off while settling the pending
   demote` (expected exactly once in the never-lands cell, where r3's eviction meets the `Hashing` entry through the
   `Block` wait and rides out the deadline; zero in the helper-gone cell, where the tick top latched during r2).

**Acceptance gate, verbatim from `DAY27.md` section 2:** "(1) the day-26 double-park cell, one hold, both orders, N=5
boots per arm per order, `stall_median(ON) <= stall_median(OFF) + 2.0` on the promote-then-hit shape (day 26: 81.8
against 85.3 with the hash tick still stretched to 95.3), the request's e2e `on_minus_off <= +20.0` (day 26: +91.4; the
token-emission reading expects one tick plus the slack, about +15.8), the demote's `in - completion` on the owner
thread `<= 12.0` ms (from 74.8; the pre-submit f32 D2H and the KV part stay); (2) identity x4, failure x2, fault, twin
and hit gates `ALL GREEN` in both arms on both cards; (3) two new fault cells, `hash-helper-gone` and
`hash-never-lands`, each a typed line, nothing published, the tier latched; (4) a CPU unit cell proving the helper's
digests equal `bind_tier_image`'s on-thread digests bitwise over a fixture image (same program, same bytes); (5) no
flag: the door is the switch, and the census gains no `MEMRA_*` read."

**How each figure is read (`day28-reading.py`, fixed now).** The cell is `pro-single-day26/double-park.sh` unchanged
(harness `stall_cell.py` byte-for-byte, `--mode promote --n 5`, twenty boots, both orders), read by
`day25-double-park-reading.py` and `day26-reading.py` as on day 27, plus `day28-reading.py` for the clauses:
- Clause 1a, stall: `stall_median(ON) <= stall_median(OFF) + 2.0` per order from the DAY25 `stall` line's per-arm
  medians (day 27: 81.9 / 85.2 and 81.8 / 85.3).
- Clause 1b, e2e: the DAY25 `e2e` line's `on_minus_off <= +20.0` per order (day 27: +91.3 / +91.3).
- Clause 1c, `in - completion` on the owner thread: on the day-27 tree everything between the copy's completion and the
  publication ran on the owner thread, so the reader's wall `demote_in - demote_completion` (74.8) IS the owner-thread
  figure by construction. On today's tree the two diverge (the publication waits for the helper), so the ledger line's
  `owner in-completion I` (= pre-submit + hashing polls + take-back, bind and publish: the same segments the 74.8 held,
  measured on the owner thread) is the clause's figure, median over the ON runs, `<= 12.0`; the wall `in - completion`
  is reported beside it and is expected to stay near or above 74.8 (it now contains the helper's hash and a tick of
  slack) and is NOT the clause. A reading that had to use the wall figure on today's tree would be a FAIL, not a
  substitute.
- Clause 2: every gate's own verdict line, verbatim, both arms, on the target card in one sitting; on the local RTX
  5090 the hit gate OFF/ON and the fault gate's new cells where the lock allows within a bounded wait, else NOT RUN,
  stated. The hit gate's ON census must equal day 24's (`12/12/13/13`, `2/2/3/3`, 30 route submissions; the hash
  helper changes no route count: it is inside the demote's publication, and the hit gate's ON boots submit zero
  demotes on this tree).
- Clauses 3 to 5: the fault gate's two new cells (each assertion named above), the unit cell's `test result: ok`, and
  `tools/check-flags.sh` finding no uncovered runtime name plus the source census asserting no `MEMRA_` read beside
  the existing `kv_host_fault()`.
- Reported beside the clauses, not clauses: the tenant's largest and second gaps (day 27: 95.3 and 92.4; expected to
  fall to about the decode plus the KV part of the bind), `demote_completion` (97.5), the helper's `H ms` (expected 70
  to 80 on this host: 157 MB at 2.15 GB/s), the count of hashing settles by mode (`tick-top poll` against `settled
  synchronously by ...`: a `Block` wait costs the owner thread the REMAINDER of the hash, at most about 73 ms here,
  and is expected zero times in this cell, whose promotes are spaced further than one hash), and the 5090's own
  figures where run (never compared to the target card's).

**What is NOT changed.** The D2H and H2D contract routes, their receipts (`hash 1`, `hash 3`), the promote, the
capture and restore routes, the tenant-share reclaim, the identity lease and the wire. The KV planes' bundle checksums
(hash 2's KV part) stay on the owner thread. Nothing generates tokens here: bytes are hashed, so one numeric program per
request holds by construction, stated in the census.

Cost expectation, pre-registered: the owner thread's `in - completion` from 74.8 to under 12 on the target card (the
pre-submit f32 D2H about 6 ms, the KV part of the bind about 1 ms, the insert and the ledger); the request's e2e
`on_minus_off` from +91.3 to about +15.8; the tenant's largest gap from 95.3 to about 21. On the 5090 class the same
shape at that host's rates (its 21 to 23 ms `in - completion` is mostly the heap pass at 4.5 GB/s plus the
write-combined KV share; C's DAY33 refuted the write-combined read of the whole image); no cross-card number.
