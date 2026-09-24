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
