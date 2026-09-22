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

## 2. The code (`45f824a75`; scripts `f9284a711`)

`crates/memra-server/src/worker.rs` only, plus the fault gate and the FLAGS row. No engine crate, no tier crate, no
wire, no new numeric program, no `unsafe`, no new `MEMRA_*` name.

- **`HostHashWorker`**, one per `HostTierContext` (`hasher`), spawned by `host_tier_context` at boot (a refused spawn
  refuses the door) and by the two test constructors: one named `std::thread` (`memra-host-hash`), a job channel in
  (`HostHashJob { seq, payloads: Vec<HostHashPayload> }`), a reply channel out (`HostHashReply { seq, hashed: Vec<(payload,
  bytes, Digest)>, helper_ms }`). The helper's loop runs `host_hash_payload_digest(&data)` = `(bytes.len(),
  memra_engine::cache::tiered::checksum(f32s_as_bytes(data)))`, THE program the bind ran on the tick, and sends every
  payload back with its digest. `submit`, `try_reply` (`Poll`), `reply_within` (`Block`, bounded), `join(why)` (drop
  the sender, join the thread, one line; idempotent). The helper only ever blocks on its job channel, so the join
  cannot hang.
- **`PendingDemote`** gains `hashing: Option<PendingHashing { seq, payloads, bytes, handed }>` and `owner:
  DemoteOwnerLedger { presubmit_ms, copy_settle_ms, hash_polls_ms, hash_polls }`. `host_hash_take_payloads` moves every
  `HostF32::Heap` payload (conv, ssm, logits, hidden) out of the image, leaving an emptied `Vec` in its slot;
  `host_hash_restore_payload` puts one back and refuses a slot that is not the emptied heap `Vec`.
- **The driver.** `host_demote_settle_with` = `host_demote_settle_with_deadline(.., HOST_HASH_DEADLINE)` (10 s). The
  driver takes the pending demote and, BEFORE the poll count and the contract step, dispatches a `Hashing` entry to
  `host_demote_settle_hashing`. The `Done` arm of the copy applies the `flip-demote` fault, puts the KV planes into the
  image, takes the heap payloads, submits the job, prints `demote copy complete off the tick: ... handed to the hash
  helper` and, under `Poll`, parks the `Hashing` state; under `Block` it continues straight into
  `host_demote_settle_hashing` in the same call (a `Block` caller expects the slot free on return). An image with no heap
  payload publishes at once as on day 17 (`demote published off the tick`).
- **`host_demote_settle_hashing`**: `Poll` reads the reply channel; `Block` prints the wait line and waits
  `reply_within(deadline - elapsed)`. The reply is checked (ticket, payload count, every byte count, every slot on
  restore); the tier's latch is re-read; the payloads go back; `host_demote_publish(.., Some(&digests))` runs the
  day-15 tail with the digests handed to `bind_tier_image`, whose `add` consumes a handed-in digest for a slot only
  after `n == bytes.len()` and hashes every other payload (the KV planes, the draft, the metadata) as before. On
  `Demoted` the ledger line prints. Fail-closed arms, each `demote failed (...); nothing published; the tier latches
  off` then `host.disable`: the reply channel closed (`tier hash helper gone`), the deadline (`tier hash digests never
  landed`, under `Block` at the bounded wait's end, under `Poll` at the first poll past the deadline), a reply that does
  not describe the image (`tier hash reply mismatch`), and a `Hashing` entry without its shell (unreachable, stated).
- **`HostPrefixCache::disable`** joins the helper (`hash helper joined (the tier latched off)`).
  **`host_demote_drain_at_shutdown`** (the run loop's last statement, after the restore drain) drops a pending demote
  typed (`demote dropped at shutdown: ...; nothing published`) and joins the helper (`hash helper joined (shutdown)`).
- **Faults.** `HostHashFault::{HelperGone, NeverLands}`, `from_door("hash-helper-gone" | "hash-never-lands")`, read
  once from the existing `kv_host_fault()` into the helper at spawn; `HostContractFault::from_door` returns `None` for
  both (the contract routes never take them). `docs/FLAGS.md`: the `MEMRA_KV_HOST_FAULT` row's value list and text
  extended in the same commit. `tools/kv-host-contract-fault-gate.sh`: `hcell` with the two cells (section 1 item 8).
- **CPU cells** (`worker::tests`, all green with the whole server lib: `830 passed; 0 failed; 14 ignored`):
  `hash_helper_digests_equal_the_owner_thread_digests_bitwise` (six payloads, 4096 to 786,432 f32 with `None` slots
  between, the 151,936-wide logits and a 5120 hidden row: every digest equal bitwise to `checksum` on the test thread,
  every byte count equal, every payload back in its slot byte-identical, two restore refusals, the join idempotent),
  `hashing_demote_keeps_its_state_until_the_digests_land_then_reaches_publication` (three polls keep the state with
  the test playing the helper; the reply reaches the bind, which refuses by name on the CPU; the `Block` shape),
  `hash_helper_gone_is_a_typed_refusal_that_latches_under_poll_and_under_block`,
  `hash_digests_never_landing_latch_at_the_deadline_under_poll_and_under_block` (a 60 ms injected deadline; the
  `Block` wait rode it out; the latch joined the helper),
  `hash_reply_that_does_not_describe_the_image_is_a_typed_refusal_that_latches` (five forged replies),
  `hash_fault_door_values_are_the_helpers_and_never_the_contract_routes`, and the source census
  `every_path_that_meets_a_hashing_demote_meets_it_through_the_same_settle` (section 1 item 2, statement by statement;
  the parked-only wait and the orphan grace read no demote state; trims settle no demote; one helper spawned once
  from the existing door read; the bind's byte-count check precedes the consumed digest). The day-17 census test moved
  its driver needle to `host_demote_settle_with_deadline`; the day-21 census now expects the demote drain as the run
  loop's last statement; the reclaim census needle is `host.bind_tier_image(&mut e, hashed)`.
- **Checks on `45f824a75`**: `cargo fmt --all -- --check` clean; clippy `-D warnings` all targets on tier, engine and
  server `Finished`; the GPU-less `DOCS_RS=1 --target x86_64-unknown-linux-gnu` clippy pass `Finished`;
  `check-flags: no uncovered runtime names` (a first draft's assertion string spelled a name-shaped literal and the
  census caught it: the assertion now pins the read to `HostHashFault::from_door(kv_host_fault())`); markers `OK`;
  `git diff --check` clean; zero em dashes in the day's lines.

## 3. The sitting (target card, BOX3, one RTX PRO 6000 Blackwell, 600 W; `pro-single-day28/box/`)

Tree `f9284a711` on `/root/wt-a` (branch `lane-a-day28`, clean; the code is `45f824a75`, `f9284a711` adds the cell
scripts only), binary `52756f946dca77042c7a8ce6…` (`double-park/ev/binary.sha256`, `gates/binary.sha256`, one binary
for every cell), harness `stall_cell.py` byte-for-byte day 26's. Three collector holds and the hit gate's own `flock`
in one sitting, 16:12Z to 16:45Z, zero lock retries (no `lock-retries.log`), no compute app on the card before or
after any cell, every `CELL.jsonl` `status: executed-not-qualified`, `qualification: false`, `exit_code 0`:
`double-park` (twenty boots 16:12:36Z to 16:31:59Z; 4,639 samples at 250 ms; 33 to 52 C; 31.7 to 359.6 W; 0 to
17,109 MiB), `gates` (16:32Z to 16:40:49Z; 2,116 samples; 39 to 62 C; 86.6 to 509.8 W; up to 21,939 MiB), the hit gate
16:41Z to 16:42Z, `unit-cell` to 16:45:08Z. Every double-park receipt `STALL REPLAY: PASS` (20 of 20), `errors=0`,
one tenant text SHA per boot, `server_promote_ms` count 10, no `demote failed|promote failed|promote refused|TIER
DISABLED` line in any boot: `ADMISSIBLE all_receipts=True` (day 25's reader) and `DAY26 DOUBLE-PARK ADMISSIBLE
all_receipts=True`.

## 4. The reading, verbatim (`box/reading-day25.log`, `box/reading-day26.log`, `box/reading-day28.log`)

Day 25's reader:

- `DAY25 DOUBLE-PARK stall order=o1 on_minus_off=-3.4 unc=0.1 -> isolated (on 81.8, off 85.2)`;
  `order=o2 on_minus_off=-3.4 unc=0.1 -> isolated (on 81.9, off 85.2)`.
- `DAY25 DOUBLE-PARK e2e order=o1 on_minus_off=+16.9 unc=3.3 -> isolated (on 132.3, off 115.4)`;
  `order=o2 on_minus_off=+16.9 unc=1.1 -> isolated (on 132.2, off 115.4)`.
- `DAY25 DECOMPOSITION arm=on N_runs=100 parked_per_run=[1] ... promote_completion median=19.6 promote_in median=26.1
  demote_completion median=nan demote_in median=183.9 demote_in-completion median=nan idle_p50(tick)=13.47 ... | tenant
  top gaps: largest median=95.3 second median=19.0 sum median=114.3`; `arm=off N_runs=100 parked_per_run=[0] promote_in
  median=10.6 demote_in median=6.2 | tenant top gaps: largest median=98.6 second median=16.6`. (The `nan`: day 25's
  reader keys `demote_completion` on the day-17 line `demote published off the tick`, which this tree prints only for an
  image with nothing to hand off; the day-28 reader reads the same figure from the `demote copy complete off the tick`
  line, below.)

Day 26's reader (its own clauses, not today's gate; run on the receipts root as on day 27):

- `DAY26 CLAUSE 1 arm=on N_runs=100 parked_per_run=[1] restore_submitted_per_run=[0] not_routed_per_run=[1]
  runs_with_parked_1_submitted_0_not_routed_1=100 -> PASS`.
- `DAY26 CLAUSE 2 e2e order=o1 on_median=132.3 off_median=115.4 on_minus_off=+16.9 unc=3.3 expected=+15.8 (day 25:
  +105.85 minus 90.1) |d-expected|=1.1 -> PASS`; `order=o2 ... on_minus_off=+16.9 unc=1.1 expected=+15.8 ...
  |d-expected|=1.1 -> FAIL` (the day-26 rule is `|d - expected| <= unc`; 1.1 against 1.1 at the printed precision, the
  second order's tighter IQR; today's clause 1b is the `<= +20.0` rule below).
- `DAY26 CLAUSE 3 stall order=o1 on_stall_medians=[81.8, 81.8, 81.9, 81.8, 81.7] on_cell_median=81.8 IQR=0.0
  off_cell_median=85.2 on_minus_off=-3.4 unc=0.1 -> isolated`; `order=o2 ... [81.9, 81.9, 81.9, 81.8, 81.8] ... 81.9
  ... -3.4 unc=0.1 -> isolated`.
- `DAY26 CLAUSE 5 hitgate-off: SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen) (ok=61 FAIL=0)`; `hitgate-on: ... ALL GREEN
  (qwen) (ok=68 FAIL=0)`; `hitgate-on counts against day 24: spec_on_census_equal=True spec_off_census_equal=True
  route_submissions=30 (day 24: 30) spec_boundary_captures_with_draft_plane=11 (day 24: 11) not_routed_lines=0 -> PASS`.
  (`CLAUSE 4 restore-arm: NO RECEIPT -> FAIL` by construction: the restore arm was not part of today's cell.)

Day 28's reader (the pre-registered clauses; the report line's wall figures from the `copy complete` and ledger lines):

- `DAY28 CLAUSE 1a stall order=o1 N_boots_on=5 N_boots_off=5 on_cell_median=81.8 off_cell_median=85.2 rule on<=off+2.0
  -> PASS`; `order=o2 ... on_cell_median=81.9 off_cell_median=85.2 ... -> PASS`.
- `DAY28 CLAUSE 1b e2e order=o1 N_runs_on=50 N_runs_off=50 on=132.3 off=115.4 on_minus_off=+16.9 rule <=+20.0 -> PASS`;
  `order=o2 ... on=132.2 off=115.4 on_minus_off=+16.9 ... -> PASS`.
- `DAY28 CLAUSE 1c owner in-completion N=100 median=7.40 min=7.17 max=45.01 runs_with_demote_without_ledger=0 rule
  <=12.0 -> PASS`.
- `DAY28 REPORTED wall in-completion median=86.5 (N=100); demote_completion median=97.5; helper hashed_in_ms median=73.2
  min=73.0 max=73.4; payloads=[98] mb=[157.9]; hash_polls median=6 max=6; settle modes={'tick-top poll': 100}; tenant top
  gaps: largest median=95.3 second median=19.0`.
- `DAY28 VERDICT clauses_failed=0 -> ALL PASS`.

## 5. Verdicts, clause by clause (the gate of section 1, verbatim from `DAY27.md` section 2)

**Clause 1, the double-park cell.** (1a) `stall_median(ON) <= stall_median(OFF) + 2.0`: 81.8 against 85.2 and 81.9
against 85.2, `on_minus_off=-3.4 unc=0.1 isolated` both orders: **PASS**, unchanged from day 27 (81.9/81.8 against
85.2/85.3). (1b) e2e `on_minus_off <= +20.0`: **+16.9 / +16.9** (132.3 / 132.2 against 115.4), `isolated`, from +91.3
on day 27: **PASS**, and within 1.1 of the +15.8 the token-emission reading predicted (one tick plus the slack). (1c)
`in - completion` on the owner thread `<= 12.0`: the ledger's `owner in-completion` median **7.40** (N=100, min 7.17,
max 45.01), from 74.8: **PASS**. Read with the decomposition: `owner in-completion` = pre-submit + hashing polls +
take-back, bind and publish; the hashing polls are 0.01 ms and the take-back, bind (the KV planes' checksums) and
publish 1.08 to 1.13 ms in every run, so the figure IS the pre-submit segment: about 6 ms at steady state (the same
6.2 the OFF demote's whole `in` reads: the f32 D2H of 157 MB on the owner stream plus the 32 lease allocations) and 40
to 45 ms on the first two demotes of every boot (b01-on's `pre-submit 39.28` and `43.39`; day 17's first-touch step on
the pinned allocations), which are the max. The segment counts t0 to the submission's RETURN, and the code order is
`reserve_image`, the KV submission (32 `alloc_host`, 32 `register_device`, the fence, `submit_batch`, then the
`submitted` stamp), then the 98 recurrent planes plus the logits and the hidden row copied one `clone_dtoh` and
`synchronize` at a time into pageable `Vec`s (`host_entry_from_device`): so the stamp sits INSIDE the pre-submit
segment and the baseline's `completion` (97.5 median, unchanged today) contained the f32 D2H that the ledger's
`pre-submit` now also counts. The comparison is therefore conservative (today's figure carries a segment the 74.8 did
not), and it passes. What the tenant sees: the second gap fell from 92.4 to **19.0** (the demote's tick is now the
decode plus about 6 ms), the largest gap 95.3 is the intruder's own prime and stands in both arms (OFF 98.6). The
wall `in - completion` rose from 74.8 to 86.5 (the publication waits for the helper's 73.2 ms and a tick top: 6 polls
median), exactly as pre-registered; it is not the clause. Every one of the 100 landings was a `tick-top poll`; no
`Block` wait met a `Hashing` entry in this cell.

**Clause 2, the gates `ALL GREEN` in both arms on both cards.** Target card, verbatim: `identity-default-off`
`KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)`; **`identity-default-on` `KV-HOST-SPILL IDENTITY GATE: 4 FAILURE(S)
(teeth=0)`** (8 ok; the four: `E_A promoted back on the r3 probe (named log line)`, `MEMRA_KV_HOST_VERIFY digest
matched across the round trip`, `metrics: demotions >= 1, promotions >= 1, zero rejected allocs`, `r3 served a
strict-prefix hit through the promoted entry`); `identity-plain-off` `ALL GREEN (teeth=0)`; `identity-plain-on` `ALL
GREEN (teeth=0)`; `failure-off` `KV-HOST-SPILL FAILURE GATE: ALL GREEN`; **`failure-on` `KV-HOST-SPILL FAILURE GATE: 1
FAILURE(S)`** (14 ok; the one: `the promote caught it: VERIFY FAILED, loud and named`, the `digest` cell; `alloc` and
`poolfull` green); **`contract-fault` `KV-HOST-CONTRACT-FAULT GATE: 23 FAILURE(S)`** (per cell: `presubmit` 9 ok 1 fail
and `postpublish` 9 ok 1 fail, each the one check `the next demote publishes`; `promote-presubmit` 6 ok 5 fail,
`promote-postpublish` 6 ok 5 fail, `promote-readyview` 6 ok 5 fail, `promote-reject` 8 ok 6 fail, each missing its
injected refusal and everything after it; `d2d-capture` 12 ok, `d2d-restore` 14 ok, `hash-helper-gone` 14 ok,
`hash-never-lands` 14 ok, all 0 fail); `twin-off` and `twin-on` `PREFIX-NEWEST-TURN-FITS: ... cached_ok=7/7
lines_ok=8/8 ...` (rc 0 both); `hitgate-off` `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 ok); `hitgate-on` `ALL
GREEN (qwen)` (68 ok) with the ON census equal to day 24's exactly: `armed=1 door_on=1 capture_submitted=12
capture_published=12 restore_submitted=13 restore_landed=13 demote_submitted=0 promote_submitted=0
refused_contracts_door=0 restore_refused=0 latched=0` (spec-on), `capture_submitted=2 capture_published=2
restore_submitted=3 restore_landed=3` (spec-off), `ok: door arm: 30 route submission(s) across the two boots`.
**Clause 2 FAILS on the target card**, in the default (spec) arm: three gates red, one cause (the finding below). The
local RTX 5090 half is in section 6.

**Clause 3, the two fault cells.** **PASS**: `hash-helper-gone` 14 ok 0 FAIL and `hash-never-lands` 14 ok 0 FAIL, every
pre-registered assertion (section 1 item 8) by name; the servers' typed lines, verbatim: `[prefix-host] demote failed
(tier hash helper gone: the reply channel closed before the digests of ticket seq=3 landed (98 payloads, 157.9MB, 1
poll(s), tick-top poll)); nothing published; the tier latches off`, then `TIER DISABLED: tier hash helper gone: ...`,
then `hash helper joined (the tier latched off)`; and `[prefix-host] demote hashing settled synchronously by a second
demote: waiting for the hash helper's reply for ticket seq=3 (98 payloads, 157.9MB, 472.1ms since the hand-off; no
CUDA event and no join: ...)`, `[prefix-host] demote failed (tier hash digests never landed: ticket seq=3 waited 10.0s
past the hand-off, deadline 10s (98 payloads, 157.9MB, 2 poll(s), settled synchronously by a second demote)); nothing
published; the tier latches off`, `TIER DISABLED: ...`, `hash helper joined (the tier latched off)`, `demote refused:
the tier latched off while settling the pending demote` (exactly once, r3's, as pre-registered). Both `hand-off
ticket(s) ['3'], refusal ticket(s) ['3']`; no `[prefix-host] demote:` line, no `digests landed` line, no `no longer
whole`, no `Capacity`, no `leaked`; three completions served in each.

**Clause 4, the bitwise digest unit cell.** **PASS**: `hash_helper_digests_equal_the_owner_thread_digests_bitwise ...
ok` on the target card's host (`box/unit/cargo-test-hash.log`: `test result: ok. 8 passed; 0 failed`, the eight day-28
cells with the two census tests) and on the local rig (the whole server lib, `830 passed; 0 failed; 14 ignored`). The
box's GPU cells: `option_b_`/`option_c_` `test result: ok. 8 passed; 0 failed` (`cargo-test.log`), the engine's `d2d_`
cells `ok. 5 passed; 0 failed` (`cargo-test-engine.log`).

**Clause 5, no flag.** **PASS**: no new `MEMRA_*` name (`check-flags: no uncovered runtime names`); the census test
pins the helper's only door value to the existing `kv_host_fault()` read; the two fault values sit on the existing
`MEMRA_KV_HOST_FAULT` row.

### The finding: a hit that arrives inside the `Hashing` window is a miss, and the door's default-arm gates land there

Every red cell on the target card has one shape, read from the server logs (`box/gates/identity-default-on/
host-on-server.log`, line numbers): line 60 r2's `insert (spec-boundary)` evicts E_A; 63 `demote submitted off the
tick ... ticket seq=3`; 65 `demote copy complete off the tick: ticket seq=3 ... 54.0ms from submission to completion
(tick-top poll); 98 heap payloads (157.9MB) handed to the hash helper`; **69 r3 admitted, `[spec-k] ... prompt=102
cached=0 lcp=0`**: the host lookup misses E_A, which is in neither index while `Hashing`, so r3 primes cold; 72 and
73 `demote: 64 tokens, 159.9MB in 521.8ms` and `demote digests landed off the tick: ticket seq=3 ... hashed in 73.2ms
on the hash helper, landed after 1 poll(s)`: E_A publishes at the tick top AFTER r3's prime. On the day-26 tree the
same boot reads (`pro-single-day26/box/gates/identity-default-on/host-on-server.log`) line 63 `demote submitted`, 65
`demote published off the tick: ticket seq=3 ... 54.4ms from submission to completion`, 66 `demote: ... in 239.2ms`,
then 69 `promote submitted`, 80 `[spec-k] ... prompt=102 cached=64 lcp=64`: there the bind ran INSIDE the tick top
that observed the copy complete, before admission, so r3 always saw E_A. Under option (a) the entry stays `Demoting`
for the helper's 73 ms plus a tick, and the day-17 rule "a `Demoting` entry is a miss" (harmless while the window was
one copy and one tick) now covers r3 in every gate whose next request must hit the just-demoted entry immediately
after a spec-boundary insert: the identity gate's r3 (no promote, so no verify round trip, no promotions counter, no
strict-prefix hit through the promoted entry), the failure gate's `digest` cell (no promote, so `flip-demote` is never
caught by `VERIFY FAILED`), and the fault gate's four promote-side cells (no promote, so the injected promote fault
never fires and nothing follows it). The plain arm (`MEMRA_SERVE_SPEC=0`) inserts at the seed, r2's prefill, so E_A's
demote lands hundreds of ms before r3 and every plain-arm cell is green; the fault gate's two demote-side cells fail
only their last check, `the next demote publishes`: r3 is the boot's last request, its eviction demote hands off at
its boundary and the gate's `stop` arrives before the 73 ms land (the server log ends at `demote copy complete`).
The hit gate (no promote-then-hit shape) and the twins are untouched.

What this is: the pre-registered semantics of option (a) ("the entry stays `Demoting` until the digests land"),
implemented as stated, measured against the door's own correctness gates, which put a hit inside the window. Not
tuned today (the instruction: report the failing clause with its decomposition). **What is left, for the lead's
ruling:** a hit on the one `Hashing` entry's own prompt must not miss. Two shapes: (i) the probe parks the request one
tick, as the promote does, re-admitted when the digests land (nothing on the owner thread; the existing parking
machinery; the request's e2e grows by the remainder of the hash, at most about 73 ms here, only when it arrives inside
the window); (ii) the probe settles the `Hashing` demote synchronously (the `Block` shape: the owner thread waits the
remainder of the hash, at most about 73 ms here, and the tick stretches by it). (i) keeps the owner thread free and
is the recommendation. Either is engine code with its own pre-registration and the same gate set; the fault gate's
demote cells also want a bounded wait for the boot's last publication before `stop` (a gate-shape change, or the
same park: the idle loop publishes it 73 ms later either way). Until then the door's default arm is RED on the target
card and the day-28 code must not be integrated as the door's serving path; the state machine, the helper, the fault
arms and the digest cell are landed and green, and the clause-1 cost figures stand.

### The second finding: the demote's remaining owner-thread cost is the pre-submit segment, and it is not 6 ms at first touch

DAY27 priced the pre-submit f32 D2H at about 6 ms from the OFF demote's 6.2. At steady state that holds (`owner
in-completion` median 7.40, min 7.17). On the first two demotes of every boot it is 39 to 45 ms (the max 45.01), the
first-touch step of the 32 pinned lease allocations day 17 measured on the arena pair; and with `MEMRA_KV_HOST_VERIFY=1`
(the identity and failure gates' diagnostic arm) the pre-submit segment reads 149 to 151 ms, because `t0` precedes the
verify digest (`host_roundtrip_digest`, a second D2H and SHA of the whole entry). The wall `completion` (97.5) is
mostly the f32 D2H and the tick wait, not the KV copy (about 1 ms). Move 2 owed item 1 (the recurrent f32 state off
the tick) is now the demote's whole remaining owner-thread cost; the ledger line prices it per demote from here on.

## 6. Local RTX 5090 (the acceptance gate's "both cards"; `rtx5090-day28/`)

`battery-5090.sh`: the hit gate OFF then ON and the fault gate in the default and plain arms (C's day-23 shape on this
card, `MEMRA_HOSTGATE_CACHE_MB=64`), the tree's local release binary (`build.log` rc=0 under the CPU quota) and the 9B
NVFP4 MTP artifact, each cell after a bounded idle wait (15 x 120 s) and under the gate's own `flock
/tmp/memra-5090.lock`; the lead's battery and lane C shared the card (eleven waits before the first cell, four before
the second, every one logged with `nvidia-smi`'s own listing, nothing signalled). RTX 5090 Laptop GPU, `power.limit
[N/A]`. Outcomes, verbatim from `battery.log` and the gate logs:

- `hit-off` (16:34:23Z): **`server died during boot`**, an empty `qwen-on-server.log`. Read from the gate, not inferred:
  the hit gate launches its server under `flock -w 300 /tmp/memra-5090.lock` (`LAUNCH=(flock -w 300 "$GPU_LOCK")` when
  it owns the lock), another lane's gate held the lock through the five minutes (its `memra-server` is the compute app in
  the launcher's next wait listing, 16:39Z), the flock timed out without launching and the boot loop read a dead pid; the
  card was never touched by that cell. The re-run (`rerun-hit-off.sh`, the same gate unwrapped from the first pass's
  `MemoryMax` scope, after the same bounded wait) is recorded below when it lands.
- `hit-on` (16:47:27Z to 16:47:53Z): `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (68 ok), census `armed=1 door_on=1
  capture_submitted=12 capture_published=12 restore_submitted=13 restore_landed=13 demote_submitted=0
  promote_submitted=0 refused_contracts_door=0 restore_refused=0 latched=0` (spec-on), `capture_submitted=2
  capture_published=2 restore_submitted=3 restore_landed=3` (spec-off), `ok: door arm: 30 route submission(s) across the
  two boots`: equal to day 24's and to the target card's today.
- `fault-default` (16:47:53Z to 16:49:22Z): **`KV-HOST-CONTRACT-FAULT GATE: 23 FAILURE(S)`** (98 ok): the same cells and
  the same checks as the target card's default arm (`presubmit` and `postpublish` the one check `the next demote
  publishes`; the four promote cells their injected refusal and everything after it; `d2d-capture`, `d2d-restore`,
  `hash-helper-gone`, `hash-never-lands` all green). The finding of section 5 on this card too, at this host's 9B
  window (about 12 ms of heap hash plus a tick), which is still wider than the gap between r2's boundary insert and r3's
  admission.
- `fault-plain` (16:49:22Z to 16:50:39Z): `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (121 ok), the two hash cells 14 ok
  each on this card as well.

No number from this card is compared to the target card's.

## 7. Checks, budget, cleanup

On the records commit: `cargo fmt --all -- --check` clean (no Rust moved after `45f824a75`, whose battery is in section
2); `check-flags: no uncovered runtime names`; `check-conflict-markers: OK`; `git diff --check` clean;
`.gitattributes` (`*.log -whitespace`) in `pro-single-day28/box/` and `rtx5090-day28/`; zero em dashes in the day's
own lines (the only em dashes in the day's additions are inside banked server logs, the engine's `[gpu-watch] Xid
source` boot line, verbatim receipts). Every push in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode,
logged to the gate-skips ledger; no qualification claimed. Budget: about 4.5 agent-hours of 5 at the records commit
(the local hit-off re-run's bounded wait runs on). Box: `/root/wt-a` at `f9284a711` on `lane-a-day28`, clean;
`/root/spill-receipts/a-day28/` mirrored to `pro-single-day28/box/` (bins not mirrored; the three reading logs and
`reading-day28.log` re-read on the mirror after the reader's report-only parse of the `copy complete` line, clauses
unchanged); both bundles removed on both ends; no server or process of mine left running on the box; nothing of other
lanes touched. Local: the release build and the battery under the CPU quota; the collector never held the 5090 lock
outside the gates' own `flock`; no scratch left in `/tmp`.
