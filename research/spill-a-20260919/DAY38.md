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
