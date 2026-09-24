# WP-A day 40: OWED item 4, the strong-form receipt of the recurrent spans (both directions)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Rig: the local RTX 5090 Laptop GPU; the target card's half rides a
later sitting (BOX7 is running DAY38 section 10's). Every cell `executed-not-qualified`. Every engine push in the
announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF).

## 1. What is owed, and the survey, pre-registered before any cell or code

**The weak form today.** A demote's recurrent f32 planes ride its ticket as spans (DAY30): a span's receipt is its
event observed, its length equal to the submission and its take-back exactly once; the hash helper's SHA-256 over the
landed staging becomes the payload's bundle share, compared with nothing. A promote's spans (DAY32, filled on the copy
stream since DAY33): the event, the length and the take-back behind the reader wait. No span's BYTES are witnessed in
either direction: a span that lands wrong bytes at the demote, a resident plane that changes in the host tier, a fill
that stages wrong bytes or an H2D copy that lands them all publish silently.

**The named strong forms.** D2H (`DAY30.md` sections 3 and 9): "the device four-lane digest of each span source plus the
CPU oracle over the landed bytes; about +71 ms on the target card's helper". H2D (`DAY31.md` section 2, `DAY32.md`
section 7): "the helper's SHA-256 of the staged bytes required equal to the plane's share recorded at the demote, about
+73 ms on the helper per promote on the target card". Both put a whole-image CPU pass (about 157 MB on the 27B) on the
helper, and the H2D one inside the promote's critical path (the promote cannot publish before it).

**The device-side form, named here.** The day-22 four-lane receipt program (`d2d_receipt_digest`, its CPU oracle
`memra_tier::conformance::receipt_digest`) costs about 0.17 ms per 158 MiB on the target card's device memory (DAY22
cell (v)). Under unified addressing every `cuMemHostAlloc` allocation is readable by a kernel through its host pointer,
flags or not, so the same kernel can read a pinned staging buffer over PCIe. That gives both witnesses on the device:

- D2H at the demote: the source digest (device memory) and the landed staging digest (pinned host memory, read over
  PCIe after the span's copy) on the receipt stream; the settle compares them before anything is published.
- H2D at the promote: the destination digest (device memory, after the span's copy) against the SOURCE digest recorded
  at the demote (kept with the host entry); the settle compares them before the reader wait's publication. The chain is
  device state at demote equal to device state at promote, every hop between them covered (the landing, the heap
  residency, the fill, the H2D copy).

**The survey cells (5090 now; the target card re-reads the kernel costs in its sitting).** A detached probe
(`day40-span-receipt-survey/`, nvrtc, the kernel source of `cu/tier_receipt.cu`'s `d2d_receipt_digest` verbatim):

1. **The UVA read.** The four-lane kernel over a cached pinned buffer (`cuMemHostAlloc` flags 0, the staging set's kind)
   through its host pointer: bitwise against the CPU oracle (`memra_tier::conformance::receipt_digest`, through a path
   dependency) on 1 byte to 3 MiB + 3 at offsets 0 to 7; a write-combined buffer the same way.
2. **The prices**, N=5 each after one warm run, on the copy-stream shape (one stream, events): the 27B's span shapes
   (48 x 3 MiB and 48 x 120 KiB, 156,893,184 B) and the 9B's (24 x 2 MiB and 24 x 96 KiB): (a) device-memory sources,
   (b) pinned staging over PCIe, one launch per span, and (c) the same with every span's launch in flight back to back.
3. **The CPU oracle's price on this host** over the same bytes (the named D2H form's helper cost here), N=5.

**The pick, stated now.** The device-side form is picked for both directions if (1) is bitwise on every size, offset
and kind, and on this card (2b) plus (2a) for the 27B's shape costs at most 15 ms of receipt-stream time per demote and
(2a) at most 1.0 ms per promote. Otherwise the named forms (the CPU passes on the helper) are built as named. Either
way the design is pre-registered with its acceptance before its code.

**What the 5090 can decide.** The UVA read's identity and the prices on this card and host. It cannot price the target
card; the sitting re-reads (2) there.

**Budget.** The survey 0.1 agent-day; the design, build and 5090 cells about 1 agent-day.

## 2. The survey, as it ran (`rtx5090-day40/survey/`), and the pick

- Probe `day40-span-receipt-survey/` with the engine's own `cu/tier_receipt.cu` compiled by `build-fatbin.sh 120a`
  (source sha256 `6347cbe006ab5f3b..`, fatbin `3a7824de82e25c1c..`); one 5090 hold ending 17:50:55Z after bounded waits
  behind lane B. Verbatim (`survey.log`):
  - `SURVEY UVA checked=128 mismatches=0 -> BITWISE` (8 sizes, 8 offsets, cached and write-combined).
  - 27B spans (96, 156,893,184 B): `arm=device .. median=0.588` ms (266.7 GB/s); `arm=staging .. median=6.725`
    (23.3 GB/s, one launch per span, each observed); `arm=staging-queued .. median=6.290` (24.9 GB/s);
    `CPU-ORACLE .. median=51.983` (3.02 GB/s on this host).
  - 9B spans (48, 52,690,944 B): device 0.204, staging 2.462, queued 2.219, CPU oracle 17.153 ms.
  - Every price run `bitwise=true`.
- **The pick, by section 1's rule: the device-side form**, both directions: (1) bitwise on every size, offset and kind;
  (2b) + (2a) for the 27B's shape 6.725 + 0.588 = 7.31 ms against the 15 ms bound; (2a) 0.588 ms against 1.0. The CPU
  oracle the named D2H form would put on the helper is 52 ms here, about 70 ms on the day-27 target host.
- **Held before the design**: DAY38's target-card sitting found that G' and G'' (a D2H receipt on its own receipt
  stream) slow the tenant's decode a little more with each demote in a boot (DAY38 section 11). The device-side span
  form would put more work on that stream, so its design waits for that cause to be placed and fixed.

## 3. Design S pre-registered (the device-side span receipts on G4's one side stream), before any S code

The hold of section 2 ends here: the hump's cause is placed (DAY38 sections 13 to 16) and the placement fixed as G4
(section 17: every piece of side work on the copy stream). S is built on G4; if G4's own cells move its placement, S is
re-registered on the new one before its code.

**The engine** (`crates/memra-engine/src/tier_transfer.rs`, `cu/tier_receipt.cu` unchanged: the four-lane
`d2d_receipt_digest` is the program, its CPU oracle `memra_tier::conformance::receipt_digest`).

1. **D2H spans (the demote).** `submit_d2h_spans`, on the copy stream behind its existing fence: for every span, the
   SOURCE digest (`d2d_receipt_digest` over the span's device f32 source) into its lanes, then the span copies exactly
   as today, then for every span the LANDED digest over the span's pinned staging destination read through its device
   address (`cuMemHostGetDevicePointer`; the day-40 survey read it bitwise, 128 of 128), then one D2H of the lanes into a
   pooled pinned twin and a span-receipt event. A span batch lands when every span's copy event AND the span-receipt
   event are observed; `take_d2h_spans` hands each span back with its `(source, landed)` digest pair. A span whose
   landed digest differs from its source's is not refused by the engine (as the D2D receipt, the verdict is the
   caller's gate): `take_d2h_spans` returns them with the pair and the caller refuses.
2. **H2D spans (the promote).** `attach_h2d_spans` (filled or not), on the copy stream: after the span copies, for
   every span the DESTINATION digest over its device f32 buffer, one D2H of the lanes, a span-receipt event; the batch
   lands with it; `take_h2d_spans` hands each span back with its destination digest.
3. The lanes and twins: `receipt_scratch` (64 bytes per span for D2H, 32 for H2D), the G'' twin pool, the unretired
   entry's leak, `synchronize` covering the span-receipt event, a census.

**The server** (`worker.rs`).

4. **The demote's settle** compares each span's pair: equal, and the SOURCE digest is kept with the host entry beside
   the plane (one 32-byte term per recurrent plane); unequal, the demote is refused typed (`demote failed (tier image
   <role> span landed bytes differ from their device source); nothing published`), the tier stays on, nothing is
   published, the next request primes cold.
5. **The promote's settle**, before the reader wait's publication, compares each span's destination digest with the
   plane's kept source digest: equal, the promote publishes as today; unequal, the promote is refused typed (`promote
   failed (tier image <role> span landed on the device differs from its demote's source)`), the host twins stay intact,
   the request primes cold, the tier stays on (the day-16 `promote-reject` shape).
6. **Red arms** on the existing `MEMRA_KV_HOST_FAULT` row (no new name), one-shot: `span-flip-landed` (one byte of the
   first span's pinned staging flipped by a one-byte kernel on the copy stream after its copy and before its landed
   digest: the demote refuses) and `span-flip-resident` (one byte of the first resident recurrent plane flipped on the
   host after its demote published and before the next promote's fill: the promote refuses).
7. **The tier crate**: two rules, `d2h_span_receipt` (a D2H span lands with its source and landed digests; a caller
   that publishes a span whose pair differs fails the schedule) and `h2d_span_receipt` (an H2D span lands with its
   destination digest; a promote that publishes against a different kept source digest fails), each with a CPU binding
   and its red arm; additive, `WIRE_VERSION` unchanged.

**Acceptance, stated before any code** (each card its own):

- (a) Semantics: the two tier rules and their red arms; native cells (every source, landed and destination digest
  bitwise the CPU oracle over the same bytes, 1 B to 3 MiB + 3 at offsets 0 to 7; each flip witnessed by exactly its
  own span); the fault gate's `span-flip-landed` and `span-flip-resident` cells (one typed refusal each, nothing
  published from the flipped bytes, the tier on, r1 to r4 byte-equal to door OFF).
- (b) The gate set ALL GREEN (identity x4, failure OFF and ON, fault default and plain with every cell, twin OFF and
  ON on the target card, hit OFF and ON) and the unit cells.
- (c) Price, demote: the S tip against the G4 tip (the demote-mode A/B of DAY38 section 3, 20 boots): the steady
  demotes' `wall` median at most +8.0 ms (the survey's 7.31 ms of kernel time on the 5090's copy stream) and the
  demoting intruder's e2e median at most +1.0 ms, per order.
- (d) Price, promote: the S tip against the G4 tip (DAY35's promote cell, 20 boots, `--n 5`): PIN (the promote's
  in-ms, steady) median at most +1.0 ms and the promoting intruder's e2e median at most +1.0 ms, per order.
- (e) DAY38's hump clause (f) on the S tip (the census keeps one side stream; the cell reads it).

**What each card decides.** Each card its own (a) to (e). The survey's prices were the 5090's; the target card's are
read in its sitting.

**Budget.** 1.5 agent-days: the engine and the tier rules 0.5, the server and the fault cells 0.5, the 5090 cells 0.3,
the target sitting 0.2.
