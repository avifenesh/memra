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
