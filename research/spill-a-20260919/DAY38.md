# WP-A day 38: OWED item 2, hash 1 (the demote's D2H receipt digest) off the owner thread

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, tip `925b33d3d` (DAY37 finding 5 closed on the 5090). Rig: the
local RTX 5090 Laptop GPU (Intel Core Ultra 9 275HX host); the target card's half rides the next rented sitting.
Every cell `executed-not-qualified`. Every engine push in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development`
mode. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF).

## 1. The survey, pre-registered before any cell or code

**What hash 1 is today** (`tier_transfer.rs` `progress`): when a D2H item's copy event is observed complete at a
tick-top poll, the owner thread computes `checksum(landed host lease bytes)` (SHA-256 over the frame
`memra-tier\0v1\0`, the domain length and `valid-bytes`, the payload length, the payload) and makes it both the item's
completion checksum and its expectation, so the D2H `require` passes by construction. It becomes the plane's receipt:
M''s bind re-hash (hash 2, on the helper) must equal it (the `flip-demote` fault writes between the two), and K's
promote-side SHA-256 of the host source must equal it. Price: `copy settle` 8.33 ms per demote on the 5090
(write-combined leases, DAY35 section 8), 0.70 ms on BOX5 (cached, DAY36 section 4). What it does not do: witness the
transfer. Hash 1 and hash 2 both read the landed bytes; neither reads the source, so a copy that landed wrong bytes is
not refused anywhere before publication.

**Why design M failed** (DAY35 section 6): M1 handed the landed views to the helper at the landing poll and took the
reply at the NEXT poll, so every copy phase gained a poll; behind a long tick the demote published late, and a hit
arriving in the copy phase serves cold by the day-17 rule (DAY29 parks the `Hashing` phase only), so the gates'
promote-after-demote never ran.

**The candidates.**

- **P, the copy-phase park, with M1.** A hit on a `Demoting` entry in its copy phase parks as a `Hashing` hit does
  (DAY29's "one predicate change"), which removes M's gate failures; hash 1 then goes to the helper as M1 did. By
  construction it keeps M1's added poll: every demote publishes one tick later (M's wall +24.25 / +26.50 ms on the 5090
  carried it).
- **L, the helper hash at landing.** A host function on the copy stream behind the D2H items hands the landed views to
  the helper when the copies complete (no CUDA call in it; design F's mechanism in the other direction), so the helper
  hashes from the landing instant, not from the next poll; the landing poll finds the digests supplied when the hash is
  shorter than the time to the next tick top, and misses (M1's extra poll) otherwise, so it needs P's park as its
  fallback. Its margin is the helper's hash time: 8.10 ms on the 5090's write-combined leases (M1's receipts job, DAY35
  section 6), about 1.1 to 1.4 ms on cached leases (K's helper, DAY34), against a 5090 tick of about 8 ms.
- **G, the device-side form of DAY27 option (c).** The copy stream computes the SAME program (the framed SHA-256) over
  each KV item's DEVICE source, behind the producer fence, before the item's copy; one D2H of the 32-byte digests into
  pinned memory and a receipt event close the batch (slice 3's D2D receipt scratch, reused); `progress` lands an item
  only with its digest read after the receipt event. The owner reads 32 bytes per item. Hash 1 becomes the source's
  digest; M''s hash 2 on the helper (the landed bytes' digest, already computed) must equal it at the bind, so the
  bind now witnesses landed equal to source, which today nothing does. No poll is added while the kernel ends before
  the tick top. The receipt program, K's comparison and the `flip-demote` naming stay as they are.

**The survey cells (the 5090 now; G's kernel is re-read on the target card in the sitting).**

1. **G's kernel, price and identity.** A Memra-native framed SHA-256 kernel in the survey probe (one thread per item;
   nvrtc, the probe crate), checked bitwise against `memra_tier::contracts::checksum`'s program (the `sha2` crate with
   the frame) on every size below, and timed on its own stream with events: 16 items x 60 KiB (the 9B 64-token entry),
   32 x 60 KiB (the 27B's), 32 x 1 MiB, 32 x 4 MiB (a 4096-token 27B entry is about 3.8 MB per item), N=5 each.
2. **The write-combined read rate on this host** (for L's margin and for a separate 5090 improvement, not for the pick):
   the framed SHA-256 over 1.1 MiB of write-combined pinned memory read directly, against the same after a streaming
   copy (`_mm_stream_load_si128`) into cached memory, N=5 each.

**The pick, stated now.** G is picked if its kernel is bitwise equal to the program on every size and takes at most
2.0 ms for 32 x 60 KiB on the 5090 (well inside one tick, so the landing stays at its poll on the gates' 64-token
shapes). Otherwise L with P's park is picked if the 5090's write-combined hash, streamed or not, is at most 6.0 ms per
1.1 MiB (inside the tick with margin); otherwise P with M1. **Whatever is picked, P's park is built with it**: under G a
large entry's kernel time lengthens its copy phase (about 32 x 3.8 MB at one thread per item), and a hit arriving there
should wait for the publication, not re-prime cold. Every design is then pre-registered with its own acceptance before
its code, DAY35 section 2's (a) to (d) the template.

**What the 5090 can decide.** The kernel's identity (bitwise), its price on this card, and the write-combined read rate
of this host. It cannot price the kernel on the target card; the sitting re-reads it.

**Budget.** The survey 0.1 agent-day; the design, build and 5090 cells about 1 agent-day.

## 2. The survey, as it ran (`rtx5090-day38/survey/`), and the pick

- Probe `day38-hash-survey/` (binary hash in `survey/binary.sha256`), one hold 12:46:19Z to 12:46:23Z, no compute app;
  card 52 C, 24.6 W before; 53 C, 52.6 W after. Verbatim (`survey/survey.log`):
  - `SURVEY IDENTITY checked=56 mismatches=0 -> BITWISE` (14 sizes from 0 to 1 MiB + 3, four source offsets each).
  - `SURVEY G items=16 bytes_each=61440 N=5 ms median=1.417 min=1.416 max=1.418 bitwise=true`
  - `SURVEY G items=32 bytes_each=61440 N=5 ms median=1.516 min=1.515 max=1.517 bitwise=true`
  - `SURVEY G items=32 bytes_each=1048576 N=5 ms median=24.900 min=22.875 max=25.470 bitwise=true`
  - `SURVEY G items=32 bytes_each=4194304 N=5 ms median=107.120 min=107.108 max=107.131 bitwise=true`
  - `SURVEY WC kind=write-combined bytes=1153434 N=5 direct_ms median=9.841 .. streamed_ms median=0.341 min=0.338
    max=0.669`; `kind=cached .. direct_ms median=0.236 .. streamed_ms median=0.254`.
- **The pick, by section 1's rule: G** (bitwise on every size; 1.516 ms for 32 x 60 KiB against the 2.0 ms bound), with
  P's park built beside it. The price grows with the item: one thread per item is a sequential chain, about 40 MB/s per
  thread (so a 4096-token 27B entry, about 3.8 MB per item, costs about 100 ms of copy-stream time instead of about 60
  ms of owner-thread SHA-256 on a cached host); that lengthens a large entry's copy phase, which is P's reason.
- **A separate finding, for its own ledger item** (`OWED.md` item 12): on this host a streaming copy (`movntdqa`) makes
  the framed SHA-256 over 1.1 MiB of write-combined pinned memory 0.341 ms instead of 9.841 ms (29x); every CPU hash the
  door runs over the 5090's write-combined leases (K's promote checksums, M''s hash 2 on the helper, the verify arm)
  pays the direct rate today.

## 3. Design G with P, pre-registered before any code

**Engine** (`crates/memra-engine`).

1. A kernel `d2h_receipt_sha256` in `cu/tier_receipt.cu`: the survey's program (the framed SHA-256 of
   `memra_tier::contracts::checksum`), one thread per item, up to 64 items per launch passed by value (item count,
   device pointers, lengths), 32 bytes out per item. A `docs/KERNELS.md` row.
2. `CudaTransfers::submit_batch`, for a D2H batch on the copy stream (the engine was built with the receipt kernels):
   the copy stream waits on every accepted item's producer event, the kernel digests every accepted item's DEVICE
   source, one D2H of the digests into a cached pinned twin, the batch's receipt event, and then the copies exactly as
   today. A batch on the owner stream (`CudaTransfers::new`, no kernels) keeps the CPU checksum in `progress`.
3. `progress`: an item of such a batch lands when its copy event AND the receipt event are observed complete; its
   completion checksum and expectation are its source digest, read from the twin (32 bytes); the owner thread hashes
   nothing. The receipt program, K's promote comparison against it, and the bind's comparison are unchanged in form;
   the bind's hash 2 over the landed bytes now witnesses landed equal to source.
4. The red arm's hook, `inject_d2h_source_flip()` (one-shot): the next such batch flips one byte of its first item's
   source on the copy stream after the digest and before the copy (a one-byte kernel `tier_flip_byte`).
5. The tier rule `conformance/d2h_device_receipt.rs` (additive, unversioned, `WIRE_VERSION` stays 1): a device-receipt
   D2H item is not landed until its copy and the batch receipt are observed; its checksum is its source digest; the red
   arm, a binding that lands an item on its copy alone, fails the schedule. CPU binding.
6. Native cells: `d2h_device_receipt_lands_with_the_source_digest` (every item's receipt bitwise equal to the CPU
   `checksum` of its source and of its landed lease bytes; a 300 ms hold before the batch shows the item not landed
   while the digest has not run); `d2h_source_flip_is_witnessed_by_the_landed_bytes` (under the hook the receipt
   differs from the CPU `checksum` of the landed bytes by exactly the flipped byte's effect, and equals the unflipped
   source's).

**Server** (`crates/memra-server/src/worker.rs`, `tools/kv-host-contract-fault-gate.sh`, `docs/FLAGS.md`).

7. The D2H receipt line names where hash 1 ran: `; receipts on the copy stream (source digests)`.
8. Two values on the existing `MEMRA_KV_HOST_FAULT` row, one-shot, demote side, no new name: `d2h-source-flip` (arms
   the hook for the next off-tick demote) and `d2h-delay` (the next off-tick demote's copies wait 3 s behind a
   copy-stream spin, `delay_on`, so its copy phase is long enough for a hit to arrive in it).
9. **P, the copy-phase park.** `host_hashing_hit` becomes a hit on the one `Demoting` entry in EITHER phase (the same
   pool key, token-prefix and depth rules); the parked ids live on `PendingDemote` and ride into `Hashing`; a first park
   in the copy phase prints `hit parked on a Demoting entry in its copy phase: request R (P tokens) hits the Demoting
   entry's N tokens (ticket seq=S, submitted X ms ago); the request waits for the publication`; the parked-only bounded
   wait's guard covers any `Demoting` entry; a latch during either phase names the parked requests (they re-admit to a
   cold prime). The re-admitted request finds the published entry and takes the unmodified promote park, then the device
   hit: one numeric program (it generated nothing before the park).
10. Fault gate cells, each two boots (door ON with the fault, then door OFF as the byte reference): `source-flip` (r2's
    demote refused at the bind with `plane checksum differs from its D2H contract receipt`, nothing published, the tier
    on, r3 served cold, the next demote publishing, r1 to r4 byte-equal to OFF) and `copy-phase-hit` (r3 hits the
    delayed demote's entry in its copy phase: exactly one copy-phase park line, then the publication, the promote and a
    device hit for r3, no cold prime of r3, r1 to r4 byte-equal to OFF).

No new `MEMRA_*` name, no new numeric program (bytes are hashed; no token is produced differently).

**Acceptance, stated before any code** (the 5090 now; the target card's sitting re-reads (b) to (d) there).

- (a) Verification semantics: the tier rule and its red arm; the two native cells; the fault gate's `source-flip` cell;
  the failure gate's `digest` cell on the door ON arm unchanged (`plane image checksum differs from its D2H receipt as
  injected` at the bind, `VERIFY FAILED` at the promote).
- (b) On the 5090: identity x4, failure ON, the fault gate default and plain (every cell, the two new ones included),
  hit OFF and ON with the day-24 census, the unit cells (the door's GPU cells, the engine's native cells in parallel,
  the CPU censuses): ALL GREEN.
- (c) The owner's `copy settle` segment on G's boots: median at most 1.5 ms and max at most 3.0 ms over at least 20
  steady demotes (the base, DAY35 section 8: 8.33).
- (d) Per order: the steady demotes' `wall .. t0 to publication` median on G at most the base's plus 5.0 ms (G adds its
  kernel, about 1.5 ms, to the copy phase; a poll added to the copy phase would cost about a tick, 8 ms or more), and
  the demoting intruder's e2e median at most the base's plus 1.0 ms.
- (e) P: its CPU cells (a hit in each phase parks once and re-parks silently, the ids ride into `Hashing` and are
  consumed at publication and at a latch) and the fault gate's `copy-phase-hit` cell.

**The 5090 A/B cell**: DAY35 section 2's (`stall_cell.py --mode demote --n 5` byte for byte, o1 = `base G` five times,
o2 = `G base` five times, door ON, 20 boots, the day-35 5090 environment, one bounded hold), base = the lane tip before
G's code, G = the lane tip after it; read by `day38-reading.py` for (c) and (d); the gates on G's binary after the A/B.

**Predictions.** (c) `copy settle` about 0.3 ms (the landing poll reads 32 bytes per item, then the take-back and the
spans' take). (d) the wall 1 to 2 ms longer (the kernel in the copy phase) and the e2e 5 to 8 ms lower (the owner's 8
ms hash leaves the demoting intruder's path), unless the kernel pushes the landing past the tick top, in which case the
wall grows by a tick and (d) fails as written. The tenant's demote-mode stall falls by about the 8 ms the owner stops
holding.

**What each card decides.** The 5090: (a) to (e) on this card. The target card: its own (b) to (d) and the kernel's
price there. No figure is compared across cards.

## 4. The 5090 half of G with P, as it ran (`rtx5090-day38/g/`, `engine-cells/`)

- Build: G = the lane tip `0f5c2d2f0` (binary `24588464c4ad4abc..`, `strings` count of `receipts on the copy stream`: 1),
  base = `80039a8de` in a scratch worktree (binary `995682a695faba3d..`, count 0); the tip's test binaries. The engine's
  native cells ran first in their own hold (`engine-cells/`, 13:22:23Z): `test result: ok. 13 passed` in three parallel
  runs and one serial run, the two new cells included (`D2H DEVICE RECEIPT cell flip=false items=3 gpu_ms=22.906`
  serially, the 1 MiB items as the survey priced them).
- The cell (`g-card-run.sh`): the hold taken 13:53:37Z after two bounded busy attempts behind lane B, released 14:10:41Z;
  no compute app at the start or the end; card telemetry 4091 samples, 58 to 88 C, 25.7 to 188.5 W. 20 boots, `STALL
  REPLAY: PASS` 20 of 20.
- **(c), verbatim** (`g/reading-day38.log`): `DAY38 G C copy-settle N=80 median=0.15 min=0.12 max=0.31 rule N>=20
  median<=1.5 max<=3.0 -> PASS` (base `copy-settle N=80 median=8.40 min=8.16 max=9.11`). **PASS.**
- **(d), verbatim**: `DAY38 G D order=o1 wall base=59.60 g=52.00 g-minus-base=-7.60 rule <=+5.0 | e2e base=111.06
  g=103.52 g-minus-base=-7.54 rule <=+1.0 -> PASS`; `order=o2 wall base=60.30 g=52.95 g-minus-base=-7.35 .. e2e
  base=111.63 g=104.99 g-minus-base=-6.64 .. -> PASS`. **PASS.**
- Readings: the owner's hold per steady demote `owner-held` 9.50 to 1.32 ms; `take-back` 0.06 on both (M'); the helper
  37.90 / 37.80 ms; 90 of 90 G receipt lines name the copy stream, the kernel `receipt-kernel-ms N=90 median=3.51
  min=2.15 max=4.03`; the tenant's demote-mode stall 42.05 / 42.08 to 37.74 / 38.48 ms.
- **(a) and (b), verbatim**: identity x4 `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok each); failure ON
  `KV-HOST-SPILL FAILURE GATE: ALL GREEN` (15 ok; the `digest` cell's bind line `contracts door D2H receipt: Key plane
  image checksum differs from its D2H receipt as injected (MEMRA_KV_HOST_FAULT=flip-demote)` and `VERIFY FAILED:
  promoted digest .. != demote digest ..`); hit OFF and ON `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 and 68 ok);
  the fault gate default `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (190 ok, both new cells in, the copy-phase park line
  `hit parked on a Demoting entry in its copy phase: request .. (89 tokens) hits the Demoting entry's 64 tokens (ticket
  seq=3, submitted 37.2ms ago)`); **the fault gate plain `KV-HOST-CONTRACT-FAULT GATE: 3 FAILURE(S)`** (187 ok): the
  `copy-phase-hit` cell's `FAIL: copy-phase-hit: exactly one copy-phase park line (r3)`, `FAIL: .. the entry published
  after the park` and `FAIL: .. r3 promoted after its park (not a cold prime)`; its other checks, the `source-flip`
  cell in both arms (15 ok each), and r1 to r4 byte-equal to the door-OFF boot all green. The unit cells did not run:
  the script's unit steps `cd` into the crate before redirecting to an output root I passed relative, so their logs had
  no directory (`unit server rc=1`, no result line); the A/B and the gates run from the repository root and are
  unaffected. **(b) FAILS as registered.**
- **The cause of the plain arm's failure, placed from the log** (`g/fault-plain/ev/copy-phase-hit-server.log`): the
  `d2h-delay` spin sits on the SHARED copy stream, so r2's own seed capture, submitted after the delayed demote, queued
  behind it; at r2's session retire the capture seam settled it `Block`: `capture published off the tick (seed): 64
  tokens complete after 50 poll(s), 2984.6ms .. (settled synchronously by a session retire; the settle held the owner
  thread 2607.12ms ..)`. The owner thread was held until the spin ended, so r3 was admitted only after the delayed copy
  landed (`complete after 50 poll(s), 3005.5ms`) and parked on the `Hashing` phase (`hit parked on a Hashing entry:
  request .. hits the Demoting entry's 64 tokens (ticket seq=2 ..)`), then promoted (the ledger's `1 hit(s) parked`
  and two promotes read green). The default arm's shape publishes its capture through the spec-boundary route, not the
  retire seam, and met the copy-phase park. So P held where the shape reached it; the red arm's delay, being a spin on
  the shared stream, also delayed every later copy-stream consumer and let a `Block` settle hold the owner thread.
- **What this also says about G as built**: G's kernel runs on the copy stream AHEAD of the demote's copies, so every
  later copy-stream consumer (a seed capture, a promote's fill and spans, a restore) queues behind it; at 64 tokens it is
  about 1.5 to 3.5 ms, at 4096 tokens about 100 ms (DAY38 section 2), and a retire seam's `Block` settle of a capture
  queued behind it would hold the owner thread for the remainder. Day 25 priced the retire seam at 0.4 ms on a landed
  copy; a copy queued behind other copy-stream work is not one (a separate ledger item, `OWED.md` item 13).

## 5. Design G' pre-registered (G revised; the red arm revised), before any G' code

**What changes, and only this.**

1. **The receipt stream.** `CudaTransfers::new_with_copy_stream` creates a third stream of the same context, the
   receipt stream, beside the copy stream. `seal_d2h_device_receipt` issues on it: the waits on every accepted item's
   producer event and on the lanes' zero-fill, the digest kernels, the lanes' D2H and the receipt event. The item copies
   stay on the copy stream exactly as before G, NOT behind the kernel: each waits on its producer fence only. The kernel
   and the copies read the same sources concurrently (reads only). An item lands when its copy event AND the receipt
   event are observed complete (G's `progress` rule, unchanged; the tier rule unchanged). The copy stream's other
   consumers no longer queue behind the kernel.
2. **The red arms on the receipt stream.** `d2h-delay`: the spin is queued on the RECEIPT stream ahead of the digest,
   so the demote's receipt (and so its landing) comes at least 3 s after the submission while the copy stream's other
   work is not delayed. `d2h-source-flip`: the flip is queued on the receipt stream after the digest, one event is
   recorded after it, and for that batch only the copy stream waits on that event before the first copy (the fault's
   definition: the copy after the flip). Without a fault the copy stream waits on nothing of the receipt stream.
3. The census follows: the receipt stream's order (waits, the delay, the digest, the flip and its event, the lanes, the
   receipt event) and the copy stream's items waiting on their producer fences and, under the flip only, on the flip's
   event. Everything else of section 3 (the kernel, `progress`, the hooks' names, P, the server lines, the fault values,
   the gate cells and their checks) stays as built.

**Acceptance: section 3's (a) to (e), verbatim and whole, re-run on G'** (the A/B base `80039a8de` against G', the gate
set, and the unit cells from an absolute output root). No clause, bound or check moves.

**Predictions.** (c) about 0.15 ms as G. (d) the wall at or below G's (the demote's copy phase becomes the longer of the
kernel and the copies, not their sum; about -8 ms against the base) and the e2e as G's. The plain arm's `copy-phase-hit`
cell: r2's capture no longer queues behind the delay, r3 arrives in the delayed copy phase and parks there.

**What each card decides.** As section 3.

## 5a. Amendment to section 5, before any G' cell runs: the receipt line names the stream it now runs on

Under G' the receipt runs on the receipt stream, so the D2H receipt line's term `; receipts on the copy stream (source
digests, X.XXms)` would misname it. It becomes `; receipts on the receipt stream (source digests, X.XXms)`; the fault
gate's `source-flip` check that the receipts ran on the device and `day38-reading.py`'s kernel regex follow the new
words, and so do the server census and the FLAGS, KERNELS and TESTING rows. No rule, bound, check, cell or order moves;
section 4's receipts keep the words they were written with.

## 6. G' before its cell: a defect found by reading, fixed, and the queued cell restarted

- G' built at `80349d455` (binary `befc684a9f97d604..`), its cell queued at 14:32:18Z behind lane B's hold. Before the
  hold was taken, a read of the landing paths found that `CudaTransfers::synchronize` waited on every item's event, the
  D2D receipt and the spans, but not on the NEW D2H receipt's event. Under G the receipt preceded the copies on one
  stream, so the items' events implied it; under G' the receipt runs beside the copies and the items can complete first,
  so a `Block` settle (a second demote, a promote or a purge meeting the Demoting entry), the abort's
  synchronize-then-retire and every by-reference route (`ContractD2h::OnTick`) would read the batch unlanded after its
  one host wait and latch the tier: integ38's D2D-receipt shape, for this class.
- The queued cell was stopped by this lane while still in its lock wait (its script and its own `flock -w 120 9` child,
  both this lane's processes; the hold was never taken, no boot ran; the stopped cell's log is kept as
  `rtx5090-day38/gp-stopped-prehold/`).
- The fix `c12e80e19`: `synchronize` also waits on the D2H receipt's event; the census pins it; the native cell's
  landing is now one host wait on the ticket while the receipt still sits behind its 300 ms hold (the `Block` settle's
  shape), so the cell fails on the unfixed code. Engine lib 548 passed, clippy clean. The G' binary rebuilt
  (`ded2dd0903719..`), the cell relaunched unchanged in its script, its acceptance section 3's.
- A second read against the spill review patterns, while the relaunched cell was still in its lock wait, found two more
  G' lifetimes, and the queued cell was stopped again pre-hold (`rtx5090-day38/gp-stopped-prehold-2/`): an unretired
  entry's drop leaked its items, D2D receipt and spans but FREED the D2H receipt scratch, whose lanes the receipt stream
  may still write (fixed in `e2ce1911b`: leaked like every other in-flight input, census); and the sealer took the
  scratch by value, so an error after its first enqueue dropped it under a pending write (move-then-match; fixed in
  `12c3f69d7`: the scratch rides `ManuallyDrop` through the enqueues and is handed out on success only, census). Engine
  lib 548 and server lib 895 passed, clippy clean. The cell relaunched on `12c3f69d7` (G' binary `5a88e47f4a7ade0b..`).

## 7. G' on the RTX 5090, as it ran (`rtx5090-day38/gp/`): (c), (d) and (e) pass; (b) fails again, a real defect placed

- G' at `12c3f69d7` (binary `5a88e47f4a7ade0b..`), base `80039a8de` (`995682a695faba3d..`). The hold taken 15:14:34Z
  after bounded waits behind lane B, released 15:31:46Z; no compute app at either end; card telemetry 4121 samples, 63
  to 88 C, 24.4 to 197.2 W. 20 boots, `STALL REPLAY: PASS` 20 of 20.
- Unit cells (from the absolute output root this time): `unit server rc=0 test result: ok. 18 passed` (the door's GPU
  cells, serial); `unit engine rc=0 test result: ok. 13 passed` (the thirteen native cells in parallel).
- **(c)**: `DAY38 G C copy-settle N=80 median=0.14 min=0.12 max=0.18 rule N>=20 median<=1.5 max<=3.0 -> PASS` (base
  8.41). **(d)**: `DAY38 G D order=o1 wall base=60.25 g=52.80 g-minus-base=-7.45 rule <=+5.0 | e2e base=111.74 g=105.18
  g-minus-base=-6.56 rule <=+1.0 -> PASS`; `order=o2 wall base=59.80 g=52.80 g-minus-base=-7.00 .. e2e base=111.76
  g=105.39 g-minus-base=-6.37 .. -> PASS`. Readings: `owner-held` 9.56 to 1.21 ms; the receipt kernel `N=90
  median=3.44 min=1.92 max=3.64` ms on the receipt stream, 90 of 90 receipt lines naming it; the tenant's demote-mode
  stall 42.25 / 42.32 to 38.19 / 38.22 ms.
- **(b)**: identity x4 ALL GREEN (12 ok each), failure ON ALL GREEN (15 ok), hit OFF and ON ALL GREEN (61, 68 ok), the
  fault gate default ALL GREEN (190 ok, the copy-phase park line `hit parked on a Demoting entry in its copy phase:
  request .. (89 tokens) hits the Demoting entry's 64 tokens (ticket seq=3, submitted 36.9ms ago)`); **the fault gate
  plain `KV-HOST-CONTRACT-FAULT GATE: 4 FAILURE(S)`** (186 ok), all four in `copy-phase-hit` (`exactly one copy-phase
  park line (r3)`, `the entry published after the park`, `the published entry's ledger names the parked hit`, `r3
  promoted after its park (not a cold prime)`); r1 to r4 still byte-equal to door OFF. **(b) FAILS as registered.**
- **The cause, placed from the log** (`gp/fault-plain/ev/copy-phase-hit-server.log`). Under G' the demote's copies are
  not behind the spin, and r2's seed capture landed promptly, yet its settle at a TICK-TOP POLL held the owner thread for
  the rest of the spin: `capture published off the tick (seed): 64 tokens complete after 2 poll(s), 2984.2ms ..
  (tick-top poll; the settle held the owner thread 2944.94ms, entered 39.2ms after submission)`. A `Poll` settle does not
  wait on the copy. What it does at the acknowledge is drop the capture batch's receipt scratch, whose pinned twin
  (`PinnedBacking`, allocated per batch by `receipt_scratch_bytes`) is freed with `cuMemFreeHost`, and DAY37's probe
  measured that call waiting for EVERY stream's queued work in the context (`free-host`, same thread: about 280 ms behind
  a 300 ms spin): here the 3 s receipt-stream spin. The owner thread was held there, r3 was admitted after the delayed
  copy had landed and its entry published, and it took a plain host hit (no park, a promote).
- **The defect, named**: the door allocates a pinned receipt twin per batch (every D2D capture and restore since day 22,
  every D2H demote under G) and frees it with `cuMemFreeHost` on the owner thread at the batch's acknowledge; that free
  waits for all queued device work in the context, so any long work on another stream (a large entry's receipt kernel,
  about 100 ms at 4096 tokens; F's fill host function, 11.4 ms on BOX4; a delayed copy) holds the owner thread at the
  next batch's end. Section 4's 2607.12 ms retire-seam hold under G likely paid the same free after its copy wait; the
  two are not separated there.
- **OWED**: item 13 is restated around the free (the seam's `Block` and the twin's free), and the host tier's lease
  frees (32 per host entry leaving the tier) are the same call on the same thread: a new item 14.

## 8. Design G'' pre-registered (G' plus pooled receipt twins), before any G'' code

1. **Receipt twins are pooled per `CudaTransfers`.** `receipt_scratch_bytes` takes a pinned twin of exactly the
   batch's lane bytes from the engine's twin pool when one is free, else allocates one; the twin is zero-filled as today.
   When a batch's entry is ACKNOWLEDGED (retired, every write to the twin observed), its twin goes back to the pool
   instead of being freed; the device lanes still drop (a stream-ordered free, which DAY37's probe measured holding
   nothing). An unretired entry's drop still leaks its scratch whole. The pool frees its twins only when the
   `CudaTransfers` drops (the latch or shutdown). Distinct sizes stay few (items x 32 or 64 bytes); the pool is bounded
   by distinct sizes times the batches in flight (at most one per class). Both receipt classes use it (D2D captures and
   restores, D2H demotes).
2. The d2h-delay arming line's words are corrected (`the receipt waits .. behind a receipt-stream spin (the copies do
   not)`). No check reads that phrase.
3. Censuses: `acknowledge` returns the twins to the pool before the entry drops; `receipt_scratch_bytes` takes from it;
   the unretired drop still forgets; no `PinnedBacking` of a receipt scratch is dropped on the retired path. A native
   cell: two consecutive device-receipt batches of the same shape reuse one twin (the same host pointer), and the second
   receipt is still the program over its own sources.
4. Everything else is G' as built.

**Acceptance: section 3's (a) to (e), verbatim and whole, re-run on G''.** No clause, bound or check moves.

**Predictions.** The plain arm's capture settles at its poll in about 0.4 ms; r3 arrives in the delayed copy phase and
parks there; (c) and (d) as G'.

## 9. G'' on the RTX 5090, as it ran (`rtx5090-day38/gpp/`, `gpp/unit-r2/`): every clause of section 3 passes

- G'' at `a266da656` (binary `c8133ce1a5d50f1a..`), base `80039a8de` (`995682a695faba3d..`). The hold taken 16:04:30Z
  after bounded waits behind lane B, released 16:21:47Z; no compute app at either end; card telemetry 4141 samples, 63
  to 89 C, 23.8 to 183.7 W. 20 boots, `STALL REPLAY: PASS` 20 of 20.
- **(c)**: `DAY38 G C copy-settle N=80 median=0.15 min=0.12 max=0.19 rule N>=20 median<=1.5 max<=3.0 -> PASS` (base
  8.34). **PASS.**
- **(d)**: `DAY38 G D order=o1 wall base=60.60 g=54.70 g-minus-base=-5.90 rule <=+5.0 | e2e base=113.00 g=107.52
  g-minus-base=-5.48 rule <=+1.0 -> PASS`; `order=o2 wall base=64.30 g=57.10 g-minus-base=-7.20 .. e2e base=119.61
  g=112.14 g-minus-base=-7.47 .. -> PASS`. **PASS.** Readings: `owner-held` 9.59 to 1.33 ms per steady demote; the
  receipt kernel median 3.59 ms on the receipt stream, 90 of 90 receipt lines naming it; the tenant's demote-mode stall
  42.44 / 45.86 to 38.92 / 41.06 ms.
- **(a), (b), (e)**: identity x4 `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` (12 ok each); failure ON `ALL GREEN`
  (15 ok, the `digest` cell's bind line and `VERIFY FAILED` as before); hit OFF and ON `SPEC-ON-CACHE-HIT GATE: ALL
  GREEN (qwen)` (61, 68 ok); the fault gate default `ALL GREEN` (190 ok) and **plain `ALL GREEN` (190 ok)**: the
  `source-flip` cell's `demote failed (tier image Key plane checksum differs from its D2H contract receipt); nothing
  published`, and the `copy-phase-hit` cell in both arms with its park line (plain: `hit parked on a Demoting entry in
  its copy phase: request .. (89 tokens) hits the Demoting entry's 64 tokens (ticket seq=2, submitted 434.8ms ago)`),
  r2's capture now settling at its poll in 0.06 and 0.09 ms (`the settle held the owner thread 0.06ms`; G' read
  2944.94 ms), r1 to r4 byte-equal to door OFF.
- **The unit cells, as they ran, then as rerun.** In the hold the engine's native cells read `rc=101 .. 11 passed; 2
  failed`: my own G'' addition to `d2h_device_receipt_cell` submitted its second batch with no producer fence, which the
  contract's validation refuses (`called Result::unwrap() on an Err value: NotReady`, `producer_fence.ok_or(
  Error::NotReady)`), a fixture error in the new test lines, not the mechanism. Fixed in `e5e8ba81e` (the fence recorded
  after the uploads; test code only), and the unit cells rerun alone in one hold (16:45:01Z, no compute app) with the
  same production binary under test: `engine parallel 1..3 rc=0 test result: ok. 13 passed`, `engine serial rc=0 ..
  13 passed` (the second batch reuses the pooled twin, `D2H DEVICE RECEIPT cell flip=false items=3 gpu_ms=22.896`),
  `server door rc=0 .. 18 passed`.
- **Verdict line**: `DAY38 HASH1 G'' (5090) (a) PASS (b) PASS (c) copy-settle 8.34 -> 0.15 ms PASS (d) wall -5.90 / -7.20
  e2e -5.48 / -7.47 ms PASS (e) PASS; hash 1 leaves the owner thread, the bind witnesses landed equal to source; target
  card owed`.
- What G'' carries beyond hash 1, stated: the receipt twins of every D2D capture and restore batch (since day 22) are no
  longer freed per batch on the owner thread; that free was a context-wide wait behind any stream's queued work.

## 10. The target-card sitting for items 1 and 2 (and item 3's first reading), pre-registered before it runs

`origin/main` `d61012658` (#699, #715, #716: dsv4 only; the `worker.rs` hunks are the dsv4 route contract's lanes,
disjoint from the door) merged as `71afb6e6f`, no conflict; on the merged tree `cargo fmt --all -- --check` clean, server
lib 910, engine lib 548 and tier contracts 101 passed, clippy `-D warnings` on tier, engine and server clean,
`check-flags` and `git diff --check` clean (`rtx5090-day38/merge-cpu/`). The 5090 receipts of section 9 are on
`a266da656` and `e5e8ba81e`; between them and the merged tree sit only main's dsv4 files (checked by file set).

**The box, as needed.** One RTX PRO 6000 Blackwell Workstation Edition (600 W) through the lead's acceptance (the clock
spin and the idle power), the 27B NVFP4 MTP artifact staged by the lead at `/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`
(its sha256 banked; day 36's was `1facf36c2db359dc..`), access only through the wrapper the lead sends,
`/tmp/memra-gpu.lock`. RAM at least 96 GB, free disk at least 150 GB (the clone, two release servers, the test binaries
and the artifact), the CUDA 13.1 and Rust toolchains as on BOX5, network for the clone and the crate fetch. Any CPU class
serves items 1 and 2; the host's CPU model and core counts are banked (`host-shape.txt`) because item 3 decides per host
class, and a slower-CPU host (BOX4 class) makes this sitting item 3's first slow-host reading as well.

**Build** (`pro-single-day38/build.sh <tip>`, outside any hold): the base `80039a8de` and the tip (the merged tree
carrying this section), release servers each, the tip's test binaries, the DAY39 fill probe; the binaries' hashes and
the G' marker count (`receipts on the receipt stream`: 1 in the tip, 0 in the base) banked.

**The cells, one sitting** (`driver.sh`): `ab.sh` under ONE collector hold (G'''s A/B: `stall_cell.py --mode demote --n
5`, o1 = `base g` five times, o2 = `g base` five times, door ON, the PRO demote environment of DAY36 section 3, readiness
bounded to 480 s; read by `day38-reading.py`); `gates.sh` under the collector (identity x4, failure OFF and ON, the fault
gate default and plain with all fourteen cells including `source-flip` and `copy-phase-hit`, twin OFF and ON);
`hitgate.sh` (OFF, then ON with the day-24 census); `unit-cells.sh` under the collector (DAY37's all arm: the thirteen
native cells in one process at the default thread count, 20 runs; then once serially; the door's GPU cells; the CPU
censuses `day31_` to `day38_` with the hash, Hashing and Demoting censuses; the engine censuses with
`native_cells_own_their_context`; the tier's span, H2D and device-receipt rules); `fill-survey.sh` under the collector
(DAY39 section 3).

**The rules, verbatim.** DAY38 section 3's (b), (c) and (d) on this card: (b) the gate set and the unit cells ALL GREEN;
(c) the owner's `copy settle` median at most 1.5 ms and max at most 3.0 ms over at least 20 steady demotes on G'''s boots;
(d) per order, the steady demotes' `wall .. t0 to publication` median on G'' at most the base's plus 5.0 ms and the
demoting intruder's e2e median at most the base's plus 1.0 ms. DAY37 section 1's target-card clause: the all arm 20 of
20 (`DAY37 FINDING5 TARGET all-arm green=N of 20 rule 20 of 20`). Nothing on this card is compared with the 5090.

**Expected, stated before the sitting.** (c) `copy settle` from about 0.70 ms (BOX5's base) to about 0.15 ms. (d) the
wall within about -1 to +1 ms of the base (the owner's 0.7 ms hash leaves; the receipt kernel, about 1.5 ms for the 27B's
32 items, runs beside the copies) and the e2e about 0.5 ms lower. The receipt kernel's time about 1 to 2 ms. The gates
green, the copy-phase park line in both fault arms. The all arm 20 of 20.

**Timing, for the rental.** The builds about 30 minutes, the A/B about 50 minutes (20 boots of the 27B), the gates about
40 minutes (the fault gate's two new cells add four boots), the hit gate about 10 minutes, the unit cells and the fill
probe about 10 minutes: about 2 hours 20 minutes from access.

## 11. BOX7, the A/B as it read (the rest of the sitting still running): (c) PASS, (d) FAILS on the e2e clause

- BOX7: one RTX PRO 6000 Blackwell Workstation Edition (600 W), the lead's acceptance (idle 14.7 W, the FMA spin a steady
  2317 MHz effective, a host clock policy of the BOX4 machine class); host an EPYC 9B14 class part, 92 CPUs of quota,
  440 GB RAM. The sitting launched 17:13:34Z (`build.sh b214bd2cf`, `rc=0`; `driver.sh`), the A/B in one collector hold
  17:27:57Z to 17:48:54Z, 20 boots, `STALL REPLAY: PASS` 20 of 20; the artifact `1facf36c2db359dc..`.
- Verbatim (`reading-day38-target.log`, mirrored with the sitting):
  - `DAY38 G C copy-settle N=80 median=0.52 min=0.45 max=0.73 rule N>=20 median<=1.5 max<=3.0 -> PASS` (base
    `copy-settle N=80 median=1.57`).
  - `DAY38 G D order=o1 wall base=182.80 g=185.40 g-minus-base=+2.60 rule <=+5.0 | e2e base=207.66 g=209.08
    g-minus-base=+1.42 rule <=+1.0 -> FAIL`; `order=o2 wall base=182.70 g=185.30 g-minus-base=+2.60 .. e2e base=207.64
    g=209.01 g-minus-base=+1.36 .. -> FAIL`.
  - Readings: `owner-held` 4.06 to 3.06 ms per steady demote; the helper 162.25 / 161.85 ms per 157.9 MB on this host;
    the receipt kernel median 3.33 ms.
- **(d) FAILS as registered on the target card.** The per-run receipts place why (`stall_cell` receipts, both
  orders, every boot): in every G'' boot the TENANT's time to its 24th token (`fired_at_ms`) grows run over run (363.7,
  362.9, 363.0, 365.5, 367.8, 369.7, 371.7, 373.4, 375.5, 376.9 ms in `o1/b02-g`) and the intruder's wall with it (205.4
  to 210.4 ms), while every base boot stays flat (362.8 to 364.1 ms). Something G'' does per demote accumulates within a
  process and slows the next decode steps.
- **The 5090 receipts say which change**, read back from the three banked 5090 cells (tenant pre-fire ITL median,
  run 2 to run 10, medians over the arm's boots): G (`rtx5090-day38/g/`, the receipt on the copy stream) 7.34 to 7.36 ms,
  flat; G' (`gp/`, the receipt on its own stream) 7.37 to 7.66; G'' (`gpp/`, plus pooled twins) 7.37 to 7.77; every
  base arm flat. The growth arrived with G', the separate receipt stream, and the 5090's (d) passed because G''s gains
  there (about 8 ms per demote) outweighed it within ten runs. Section 9's 5090 PASS stands as it read; the defect it
  did not see is real.

## 12. The diagnosis, pre-registered before it runs (BOX7, after the sitting's cells)

- **Arms.** X1: the sitting's tip binary (G''). X2: the same tip with `pro-single-day38/diag-one-stream.patch` (one
  line: the receipt stream IS the copy stream, so the receipt's kernels queue ahead of the copies as under G; everything
  else G''), built by `diag-build.sh` from the box's clone, the tree returned to the tip.
- **The cell** (`diag.sh`, one collector hold): four long door-ON boots X1 X2 X1 X2, each `stall_cell.py --mode demote
  --n 15` (30 demote runs), the PRO demote environment; 250 ms telemetry.
- **The reader** (`day38-growth-reading.py`, dry-checked on the 5090 receipts): per boot, EARLY = the median over demote
  runs 2 to 6 of the tenant's pre-fire ITL median, LATE = over the last five, GROWTH = LATE minus EARLY; an arm GROWS if
  the median of its boots' GROWTH exceeds 0.10 ms. `x1 grows, x2 flat -> the separate receipt stream`; `both grow ->
  elsewhere in G''`; `x1 flat -> not reproduced`.
- **What follows from each verdict** (before any fix code): the stream verdict -> one X1 boot under Nsight Systems on this
  box to place the mechanism (the decode step's wait early against late), and a fix pre-registered on that trace; the
  elsewhere verdict -> the next arm pre-registered (the twin pool removed, then the `synchronize` and drop changes);
  not reproduced -> the A/B is re-run once on the unchanged binaries in a new hold. G'' is not integrated with this
  defect; (d) is re-run on the fixed design, whole.

## 12a. The diagnosis, as it ran (BOX7 `diag/`, one hold after the sitting's cells)

- X2 built by `diag-build.sh` (`rc=0`, the tree returned to the tip); four boots 18:17Z to 18:28:04Z, each 30 demote
  runs. Verbatim (`diag/reading-growth.log`): `GROWTH arm=x1 boots=2 median-growth=-0.113 grows=False`, `GROWTH arm=x2
  boots=2 median-growth=+0.003 grows=False`, **`GROWTH VERDICT x1 flat -> not reproduced`**, as registered.
- **What the registered windows could not see**: the per-run series of both X1 boots rise and fall back, `itl=[12.33,
  12.36, 12.33, 12.44, 12.52, 12.61, 12.72, 12.77, 12.83, 12.9, 12.83, 12.76, 12.68, 12.62, 12.53, 12.41, 12.34, ..
  12.33 ..]` (`b01-x1`; `b03-x1` the same shape, peak 12.92 at run 10); both X2 boots stay at 12.32 to 12.37 on every
  run. The effect is a transient hump over the first 16 demote runs of a boot, peak about +0.6 ms per decode step, not an
  unbounded growth; the section-3 A/B's 10-run boots sit on its rising side. The verdict stands as it reads; the hump is
  recorded beside it, and it appears with the separate receipt stream only (X2 is G'' with the receipt on the copy
  stream).

## 13. Pre-registered before it runs: the registered follow-up, and a trace to place the hump

1. **Section 12's registered consequence of `not reproduced`**: the section-3 A/B re-run once, unchanged binaries, in a new
   hold (`diag2.sh` step 1: `ab.sh`'s cell verbatim, output redirected to `g-rerun/`), read by `day38-reading.py`; (c)
   and (d) as registered.
2. **The trace** (`diag2.sh` step 2, the same hold): one X1 and one X2 boot under Nsight Systems (`--trace=cuda,osrt`,
   no sampling), `stall_cell.py --mode demote --n 8` each (16 demote runs), exported to sqlite and read by
   `day38-nsys-reading.py` (written before the traces: the receipt kernel's launches as demote markers; per interval the
   owner stream's kernel count, GPU-busy ms, launch gaps, the five most frequent kernels' mean durations, and every other
   stream's busy ms). A reading that places the hump as GPU-side (owner kernels longer, or other streams busy) or
   host-side (launch gaps longer); the fix is pre-registered on it.
