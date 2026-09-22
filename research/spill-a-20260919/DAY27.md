# WP-A day 27: where the demote's hash tick actually runs (Move 1 owed item 2 read from the code), the three options pre-registered, the digest micro-cell

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Start: `645e649a6` = remote. No integ42 PR existed; the lead's
local `lane/spill-integ42-20260922` (`af92791f6`, contains main `bc6e2f44d` and my day 26) merged `--no-ff` as
`cb9fa5ef3`, clean, pushed in the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (logged; no
qualification claimed). Ruling 37 (lead, integ42): the day-26 code stays; the 74.8 ms hash tick in a one-token
request's path is Move 1 owed item 2, not a new item. Today: no engine code. Section 1 reads the code; section 2
pre-registers the options and measures the one thing measurable without engine code; section 3 is the day-26
double-park baseline if the budget allows. Every cell `executed-not-qualified`. Rig: the target card's host
(one RTX PRO 6000 Blackwell, `BOX-ACCESS.md`), and the local RTX 5090 rig's host; no cross-card comparison.

## 1. Where each hash runs, over which memory, how many bytes, on which thread (file:line on `cb9fa5ef3`)

**The owed-item text is wrong about what the 74.8 ms hashes.** Owed item 2 (`OWNER-THREAD-OFFLOAD.md`) reads
"two checksums over the entry's pinned bytes ... 77.9 ms per 160 MB of cacheable pinned memory". The code says
the two Move 1 receipt hashes run over the KV planes only, about 1.9 MB on the 27B's 64-token entry (`DAY21.md`:
"the recurrent state, copied on the owner stream, is about 157 MB and the 32 KV planes about 1.9 MB"), and the
74.8 ms is a third thing: `bind_tier_image`'s `StateBundle` checksum pass over the WHOLE image, 157 MB of which
is the recurrent f32 state in pageable heap memory that never touched the contract. In order:

1. **Hash 1, the completion checksum.** `CudaTransfers::progress`, `crates/memra-engine/src/tier_transfer.rs:1620`
   to `1725`; the hash at `1710` to `1712`: for every D2H item whose event is done, `checksum(item.host.bytes())`
   over the item's destination lease, the whole `CudaPinnedLease` (`bytes()` at `251` to `258` is
   `PinnedBacking::as_slice` after `event.synchronize()`, `150` to `156`). Reached through `poll` (`1998` to
   `2001`), which the worker calls at `crates/memra-server/src/worker.rs:10407` inside
   `host_kv_planes_settle_contract` (`10372`), itself called by the tick-top poll
   `host_demote_settle_pending(&mut hpx, ContractWait::Poll, "the tick top")` (`21323`, via
   `host_demote_settle_with` `12329`). Thread: the CUDA owner thread, which is the worker thread inside the tick
   (`check_thread()` at `1621`). Memory: the destinations `alloc_host` hands out (`644` to `645`) in the device's
   `pinned_default` (`446`), `PinnedKind::for_device` (`58` to `64`): `Cached` on the RTX PRO 6000 Blackwell
   class, `WriteCombined` on the RTX 5090 class. Bytes: the KV planes only, K and V per attention layer plus the
   MTP draft plane's K and V when present: 32 items = 16 KV planes on the 27B (about 1.9 MB at 64 tokens), 18
   items = 8 KV planes plus the draft pair on the 9B (no record splits its 54.8 MB; the same geometry rule, a few
   MB at most). Where it lands in the receipts: INSIDE the demote's `from submission to completion` figure, because
   `copy_ms = submitted.elapsed()` is stamped after the settle returns `Done` (`12416`), after `poll`,
   `take_destination`, `require`, `record_consumer`, `retire_source`, the planes back, the owner-stream
   `synchronize`, `retire` and `acknowledge` (`10372` to `10600`).
2. **The demote's `require`.** `10460` to `10483`: the expectations are the completion's own checksums (the
   comment at `10460`: "there is no pre-copy hash of the device bytes short of a second read"), so `require`
   proves shape, status, epochs and length, not bytes; the byte check is item 3's per-plane comparison.
3. **Hash 2, the bundle checksum.** `bind_tier_image` (`9296`), `let sum = checksum(bytes)` at `9344` inside
   `add`, called for every KV plane (K q8_0 and V q5_1, `9440` to `9451`, each compared to its D2H receipt, a
   mismatch a typed refusal except under `flip-demote`), for every recurrent f32 plane (`Role::Recurrent`,
   `9453`, NO receipt to compare against), the logits (`9456`), the hidden row (`9463`) and the draft plane
   (`9473`, `9480`). Called from `host_demote_publish` (`12258`, the bind at `12265`) right after the
   `demote published off the tick` line (`12432` to `12441`), on the worker thread inside the same tick; its
   duration is inside `demote: ... in X ms` (`t0` at `12186`, before `host_entry_from_device` `11669`) and
   OUTSIDE `from submission to completion`. Memory: the KV planes are the contract leases of item 1; the recurrent
   f32 planes are `HostF32::Heap(Vec<f32>)` (`8261` to `8268`: `reserve_image` returns `HostPlaneLeases(None)`
   when `arena` is `None`, `8594` to `8596`, and the door refuses the arena; `host_glm::read_f32` is a
   `clone_dtoh` into a `Vec` plus a stream synchronize, `worker/host_glm.rs:13` to `17`): pageable heap, not
   pinned at all, on both cards. Bytes: the whole image, about 157 MB recurrent plus 1.9 MB KV plus the logits
   (n_vocab f32) and the hidden row on the 27B (159.8 MB total); 54.8 MB on the 9B.
4. **Under OFF there is no bundle hash.** `hpx.tier = Some(tier)` only under `kv_host_contracts && hpx.budget > 0`
   (`20963` to `20978`); with `tier` `None`, `bind_tier_image` returns at `9298` to `9300` before any hash. The
   OFF demote (6.2 ms, day 25) copies the same 159.8 MB and hashes nothing. The door's tick cost is a
   `StateBundle` checksum the OFF arm never pays; Move 1 moved the KV copy (1.9 MB) off the tick and the copy it
   moved was never the cost.
5. **The promote's receipt.** `host_kv_planes_settle_promote` (`11216`): `poll` at `11253` runs `progress` over
   the H2D items, so the engine hashes the SOURCE host bytes of each KV lease once more (hash 3, 1.9 MB), and
   `require(&ticket, &receipts, true)` at `11319` compares each item's completion checksum to the D2H receipt
   carried on `HostPlaneBytes::Contract.receipt` (`8208` to `8224`; `PromotePlanned.ck`/`.cv`). The server does
   not hash again. The recurrent f32 state is not receipted at promote: the restore copies it on the owner stream
   (`DAY21.md`), and only `MEMRA_KV_HOST_VERIFY=1` digests it (`prefix_entry_state_digest`, default off).

**Reconciled with the receipts.** Target card, 27B, 159.8 MB entry (`DAY25.md` decomposition, `DAY26.md`):
`in - completion` = 74.8 = (t0 to submission: the f32 `clone_dtoh` of about 157 MB on the owner stream, 32 lease
allocations with their 1.9 MB zero fill, the registration) + (bind: SHA-256 over 159.8 MB, 157 MB of it heap) +
reclaim and insert. C's day-18 micro-cell on this host: 2.153 GB/s cached, 2.151 heap, so one pass over 159.8 MB
is 74.2 ms. The 74.8 is ONE SHA-256 pass over the whole image inside `bind_tier_image`; two passes over the entry
(C's `two_hashes_cached_ms=155.8`) are excluded by the receipts twice over: `completion` (22.3) contains hash 1
and cannot hold a 74 ms pass, and `in - completion` cannot hold two. Move 1's two receipt hashes are about 0.9 ms
each on this host (1.9 MB at 2.15 GB/s), hash 1 inside the 22.3, hash 2 inside the 74.8's KV part; the promote's
hash 3 is inside `promote_completion` 19.6. Local RTX 5090 rig, 9B, 54.8 MB entry, write-combined leases
(`DAY31.md` of lane C): hash 1 reads write-combined memory for the KV planes only; the recurrent state is heap.
`in - completion` 21 to 23 ms = bind's SHA over about 54 MB of heap at the micro-cell's 4.476 GB/s heap rate
(12.2 ms) + the KV planes' share read from write-combined memory at 0.117 GB/s (8.5 ms per MB) + the pre-submit
f32 D2H and the insert. That sums into the 21 to 23 band for a KV share of about 0.5 to 1 MB; the 9B's KV byte
count is the one term no record measures (no `[prefix-host]` line prints the per-class split, and adding one is
engine code). **Answer to C's section D item 6, "which memory the hashes read":** pageable heap (`Vec<f32>`) for
about 98 percent of the hashed bytes on both cards; the card's pinned kind (cached on the target, write-combined
on the 5090) for the KV planes only. The micro-cell's write-combined pass (1431.6 ms per 160 MiB) does not apply
to the 5090's demote because only the KV planes live in write-combined memory and they are one to two percent of
the entry; the 21 to 23 ms is one heap-rate SHA pass over the entry plus a write-combined read of the small KV
share. Day 16's write-combined delta (136 to 140 ms against 6 to 8) on this card is the same reading: the bind
pass at the then-larger entry plus the write-combined KV share.

**What this does to owed item 2.** The item's decision cell ("hash on a helper thread against the completion
event, or a GPU-side digest of the source bytes recorded on the copy stream") was priced against 160 MB of
contract-routed pinned bytes; there are 1.9 MB of those. The 74.8 ms belongs to the bundle checksum over bytes that
(i) never cross the contract, (ii) live in heap, and (iii) have no receipt to agree with. Any option that only
changes the D2H receipt term moves at most 2 x 0.9 ms on the target card (and the write-combined KV share on the
5090). The options below are stated against that reading.

## 2. Owed item 2: the three options, pre-registered before the cell ran

What the identity gate's receipt term proves today under ON: the bytes the H2D reads at promote are the bytes the
D2H wrote at demote (hash 1 = hash 3, per KV plane, `require`), and the KV planes published into the bundle are
those same bytes (hash 2 = receipt per plane). For the recurrent state, the logits and the hidden row the bundle
checksum is a self-consistency record over bytes no second party ever read: it proves the `StateBundle` was built
from the bytes that were in the heap at publication and nothing else.

**(a) The bundle hash on a helper thread; the entry stays `Demoting` until the digests land.** State added: one
phase in `PendingDemote` after the contract settles (`Hashing`: the `HostF32` heap payloads, the logits and the
hidden row moved to a helper thread that owns them while it hashes; the KV leases cannot move, they hold an `Rc`
and a CUDA event, `8208` and `tier_transfer.rs:150`, so the owner thread keeps hash 2's KV part, 0.9 ms on the
target card); a helper thread spawned once with the tier context; a channel each way. The tick-top poll of the
phase takes the digests back, rebuilds the layout with the precomputed sums (the `add` closure's `checksum` call
replaced by the sum handed in, byte for byte the same program over the same bytes) and publishes. Every other
route that settles the pending demote first (a second demote, a promote, a purge, the pause sweep, the handoff)
meets the phase and blocks on the channel, the day-17 `Block` shape. Fail-closed arms: the helper is gone or the
channel closed (the entry drops whole, a typed `demote failed` line, the tier latches off: a submitted image
that cannot be hashed cannot be published and cannot be retried without its bytes); the digests never land within
N tick tops (the same latch, N pre-registered at the cell); the helper reports a digest for a byte count that is
not the payload's (refusal, latch). The receipt term stays exactly: SHA-256 with the `valid-bytes` frame over the
same bytes, the same `StateBundle`, the same wire; the identity gate proves what it proves today and stops proving
nothing. Cost moved off the tick: about 73 ms on the target card (the heap part of the 74.8), about 12 ms on the
5090; what stays on the tick: the pre-submit f32 D2H (about 6 ms of the 74.8 on the target card), the KV part of
hash 2, hash 1 inside the settle, the insert. Price: about one agent-day including its cells.

**(b) The D2H receipt term replaced by the slice-3 program** (GPU four-lane digest of the device source on the copy
stream before the copy, CPU four-lane digest of the host bytes at the poll, `receipt_digest`,
`crates/memra-tier/src/conformance/d2d_receipt.rs:57` to `84`, the kernel `cu/tier_receipt.cu`). What it moves:
hash 1 and hash 3, 1.9 MB each on the target card, and the write-combined KV share on the 5090; nothing of the
74.8 unless the bundle checksum's program for `Role::Recurrent`, `Logits` and `Hidden` also becomes the four-lane
digest (call that (b'), a wire change of the `StateBundle` checksum's meaning for those roles). What the receipt
would stop proving: the four-lane digest is a wrapping sum of mixed words, order-independent by construction, not
collision resistant; it proves the copy did not corrupt the bytes (the transfer-integrity term slice 3 was
written for), it does not name the bytes the way SHA-256 does; the identity gate's byte-identity clause is
unaffected (it compares texts), the failure gate's `flip-demote` cell is unaffected (a flipped byte moves every
lane), the door's `checksums_sha256=<hex over the ordered item checksums>` receipt line would carry lane digests.
What the cell measures today (no engine code): the CPU four-lane digest's cost against SHA-256's over 160 MiB of
heap, on each card's host, `research/spill-a-20260919/day27-digest-micro/` (a detached cargo project calling the
engine's two programs through a path dependency, `memra_tier::contracts::checksum` and
`memra_tier::conformance::receipt_digest`; heap is the memory kind of the 157 MB, and C's day 18 read heap equal
to cached pinned within 0.4 percent on both hosts), `--bytes 167772160 --n 5`, two orders (SHA first, then lanes
first), interleaved call by call, pooled N=10 per program, through the collector in a rig hold on each host
(`pro-single-day27/digest-micro.sh`). Reported: per-program medians, ranges, per-order medians, GB/s,
`lanes_over_sha`, digest stability across passes. Reading rule, fixed now: (b) or (b') lowers a host's hash cost
iff `lanes_over_sha < 1` with the two ranges disjoint; a `lanes_over_sha >= 1` reads "the four-lane program as
compiled today is not cheaper than SHA-NI on this host" and (b) is a 5090 write-combined follow-up at most. The
smoke run on the local rig's host at 16 MiB (not the cell) read `lanes_over_sha=1.418`; the cell decides. The
four-lane digest over write-combined memory is NOT measured today: the micro-cell allocates heap only, and a
write-combined arm needs a pinned allocation through the driver, which is C's `hash-micro` shape with a lanes arm
added, an engine-bin change.

**(c) The hash on the copy stream entirely: a GPU digest of the pinned host bytes read back over PCIe.** The
existing kernel `d2d_receipt_digest(const u8* p, u64 n, u64* out)` takes a device pointer; the engine launches it
only over `CudaSlice` device memory (`tier_transfer.rs:535`), and `PinnedBacking::alloc` sets no
`CU_MEMHOSTALLOC_DEVICEMAP` (asserted at `2188`), so a host lease is device-addressable only through the driver's
device pointer for it, which no engine code obtains. A cell needs a bin that maps a lease and launches the digest
over it: engine code, so (c) is UNMEASURED today. Arithmetic, not a measurement: a kernel read of 160 MB over
PCIe at the D2H's own rate (the OFF demote moves 159.8 MB in 6.2 ms, day 25) is about 6 ms on the copy stream and
zero owner-thread time; the digest would be the four-lane program (SHA-256 has no GPU implementation here), so the
receipt term changes as in (b'). The stronger form of (c) is the slice-3 shape itself applied to the recurrent
state: digest the DEVICE source planes before the D2H (no PCIe read, about 0.1 ms at device bandwidth) and make
that the bundle checksum, which needs the recurrent f32 planes to cross the contract first (Move 2 owed item 1,
the 157 MB on the owner stream). What (c) would prove: the bytes that left the device; what it would stop
proving: that the host bytes at publication are those bytes (today hash 2 reads them; under (c) a host-side
corruption before promote is caught only by the promote's digest, the `flip-demote` fault's shape).

**Recommendation to the lead (for ruling; not implemented).** (a), scoped to the heap payloads, with (b) held as
the 5090 write-combined follow-up and (c)'s device-side form queued behind Move 2 owed item 1. Reason: the 74.8 is
a SHA-256 pass over heap bytes with no receipt partner, so it can leave the owner thread without touching the
copy stream, the contract, the tier crate's conformance or the wire; the receipt term is unchanged; the fail-closed
arms are the day-17 shapes. Acceptance gate for (a), pre-registered: (1) the day-26 double-park cell, one hold,
both orders, N=5 boots per arm per order, `stall_median(ON) <= stall_median(OFF) + 2.0` on the promote-then-hit
shape (day 26: 81.8 against 85.3 with the hash tick still stretched to 95.3), the request's e2e `on_minus_off <=
+20.0` (day 26: +91.4; the token-emission reading expects one tick plus the slack, about +15.8), the demote's
`in - completion` on the owner thread `<= 12.0` ms (from 74.8; the pre-submit f32 D2H and the KV part stay);
(2) identity x4, failure x2, fault, twin and hit gates `ALL GREEN` in both arms on both cards; (3) two new fault
cells, `hash-helper-gone` and `hash-never-lands`, each a typed line, nothing published, the tier latched; (4) a
CPU unit cell proving the helper's digests equal `bind_tier_image`'s on-thread digests bitwise over a fixture
image (same program, same bytes); (5) no flag: the door is the switch, and the census gains no `MEMRA_*` read.

### 2b. The digest cell, run after the pre-registration above was committed (`ad4f229e0`)

Target card's host (`pro-single-day27/box/digest-micro/`, one collector hold, no compute app on the card before or
after, host load 0.06 before the cell, the server build finished and idle first; binary
`65ffd3cf59c22ce3e26cb9309b92d17f1508f0358947bc1fc0476016d8c9ee0b`, source `day27-digest-micro/src/main.rs` at
`ad4f229e0`; the box resolved the detached project's lockfile offline from its own cargo cache, three patch versions
apart from the committed lock, banked as `micro-Cargo.lock.box`):

`DIGEST-MICRO rule host="AMD EPYC 9555 64-Core Processor" bytes=167772160 n_per_order=5 pooled=10 orders=2 memory=heap
sha_ms=77.589 lanes_ms=70.737 sha_range=77.538..77.877 lanes_range=70.662..71.044 sha_o1=77.591 sha_o2=77.570
lanes_o1=70.783 lanes_o2=70.713 sha_gbps=2.162 lanes_gbps=2.372 lanes_over_sha=0.912 sha_stable=true lanes_stable=true`

Read by the pre-registered rule: `lanes_over_sha=0.912` with disjoint ranges, so the four-lane program as compiled today
IS cheaper than SHA-256 on this host, by 9 percent, and both are compute-bound near 2.2 to 2.4 GB/s (neither runs at
memory speed; the scalar per-word `mix64` x4 costs about what SHA-NI costs). Consequence for (b'): the bundle checksum's
program change would take about 7 ms off the 74.8 on the target card, not the 74.8; it does not replace (a). The SHA
figure repeats C's day 18 on this host (77.9 cached, 78.0 heap) within 0.5 percent, so the two cells agree on the rate
that the section 1 arithmetic used. `executed-not-qualified`.

Local RTX 5090 rig's host: pending a bounded wait on the card (another lane's `memra-server` held the card from the
start of the day's waits; the result line goes here or the cell is recorded NOT RUN).
