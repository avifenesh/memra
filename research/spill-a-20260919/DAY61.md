# WP-A day 61: OWED item 12, the helper's hashes over write-combined pinned memory (a streamed read)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Behind `MEMRA_KV_HOST_CONTRACTS` (default OFF). Every cell
`executed-not-qualified`. Pre-registered while DAY59's cells wait for a card; its code follows DAY59's design in order.

## 1. Pre-registration (committed before any code)

**The item** (DAY38 section 2, the survey on the 5090): `SURVEY WC kind=write-combined bytes=1153434 N=5 direct_ms
median=9.841 .. streamed_ms median=0.341`; `kind=cached .. direct_ms median=0.236 .. streamed_ms median=0.254`. On the
5090 the pinned leases read as write-combined (the lane's memory note: CPU checksums of them run about 0.116 GB/s), so
every CPU hash over them runs at the direct rate: the promote's source checksums on the helper (`H2dSourceView::digest`,
design K), the bind's KV re-hash on the helper (`PinnedLeaseView::digest`, design M'), and the verify arm's host
digest. On the target card the same leases read as cached.

**Design W.** One function, `checksum_streamed(bytes)`, the same program as `memra_tier::contracts::checksum`
(`digest("valid-bytes", bytes)`: SHA-256 over the domain, the little-endian length, then the bytes), fed through a
64 KiB cached bounce buffer that an SSE4.1 streaming copy (`movntdqa`, the survey's `stream_copy`) fills from the
source, chunk by chunk; a tail shorter than 16 bytes is copied plainly. `PinnedLeaseView::digest` and
`H2dSourceView::digest` call it; the heap payload hashes stay on `checksum` (heap memory is cached). On a CPU without
SSE4.1 (checked at run time) it falls back to `checksum`, the same digest.

**Acceptance.**

- (a) Bitwise: `checksum_streamed` equals `checksum` on every size from 0 to 1 MiB plus 3 at four source offsets (the
  survey's identity set), on cached and on pinned memory; the fault and identity gates door ON default and plain on
  each card.
- (b) The 5090 (the price): the helper's `Sources` job time (`promote published off the tick .. helper`) and the bind's
  re-hash share of the demote's helper time, W against the tip, the promote and demote stall cells, N=5 per arm per
  order, both orders: each helper time at most half of the tip's.
- (c) The target card (no regression on cached leases): the same cells, each helper time at most the tip's plus 10%.
- (d) The tenant's stall and the intruders' e2e in (b) and (c) at most the tip's plus 1.0 ms.

**The rule.** W becomes the helper's hash for pinned sources if (a) to (d) hold; otherwise it is reverted in one commit.
A 5090-only win with a target regression past (c) makes it a per-card choice keyed on the measured read rate, which is
its own pre-registration.

**What each card decides.** Each its own clauses: the 5090 (b), the target (c); both (a) and (d).

**Budget.** 0.4 agent-day: the function and its identity cell 0.15, the census 0.05, the two sittings 0.2.

## 2. Design W as built (`34a348fd2`), and its two sittings prepared

- `memra_tier::contracts::ChecksumStream`: `checksum`'s frame, fed in chunks. `digest` and the stream share one
  framing fn, and `finish` refuses a feed shorter or longer than the framed length.
- `tier_transfer::checksum_streamed`: when SSE4.1 is present (checked at run time), the unaligned head (under 16
  bytes) is hashed plainly. The aligned body is then streamed (`movntdqa`, 16 bytes at a time) into a per-thread
  64 KiB cached bounce buffer and hashed from it, and the tail (under 16 bytes) is hashed plainly. Otherwise it calls
  `checksum`. `PinnedLeaseView::digest` and `H2dSourceView::digest` call it; nothing else does. The heap payloads
  keep `checksum`.
- One addition to the registration, stated before any cell runs: the registered text names only a short tail. Here
  the head is also peeled (under 16 bytes, read plainly) so the streamed loads are aligned for any source offset.
- Cells:
  - `day61_the_streamed_checksum_is_the_checksum` (CPU): the survey's identity sizes (0 to 1 MiB + 3), plus the
    bounce buffer's chunk boundaries, at offsets 0 to 3; a short and a long feed are refused.
  - `day61_the_streamed_checksum_is_the_checksum_on_pinned_memory` (card): both pinned arms (write-combined and
    cached), through `PinnedLeaseView`, at offsets 0 to 3.
  - The census `day61_the_pinned_views_hash_streamed_and_nothing_else_does`. `NATIVE_CELLS` goes from 15 to 16
    (the context-pool census).
- The red arm (`day61/red-arm.patch`: the streamed copy skips each chunk's second vector, and a printed marker)
  fails the CPU cell (`day61/red-arm.log`: `assertion left == right failed: len 63 off 0`, binary
  `eb9c41328a30db35` carrying the marker). The filtered log keeps the assertion rather than the marker lines.
- CPU: engine lib `575 passed; 0 failed; 49 ignored`, tier `301 passed`, server lib `935 passed; 0 failed; 26
  ignored`; clippy `-D warnings` on all three, all targets; fmt clean (`day61/`).
- Read while building, not in scope: the demote's `Hash` job also copies each landed f32 span's pinned staging into
  a heap `Vec` (`staged.as_f32_slice().to_vec()`, DAY49's `copy` term). On the 5090 that staging is write-combined
  too, so the same streamed read would apply. It is recorded here as a candidate for its own pre-registration, not
  folded into W.
- **The sittings**, both reading `w-reading.py` (written before either runs):
  - **The target**, `pro-single-w/`, receipts `/root/spill-receipts/a-w`: `build.sh <tip> <W's parent>` (w,
    red and base from one clone), then `driver.sh`. The unit cells run green and red; the gates are the identity
    gate default and plain door OFF and ON plus the contract fault gate default and plain; then the paired cell,
    base against w by demote and promote, 40 boots, the P2 cells' environment (`MEMRA_MAX_SESSIONS=4`). Its line:
    `W VERDICT card=target -> ..` (clauses (a), (c), (d)).
  - **The 5090**, `rtx5090-w/`: `build-local.sh <out> <tip> <W's parent>`, under the CPU cap, with scratch
    worktrees removed at the end; then `card-run.sh <out> <model>` in one bounded hold of `/tmp/memra-5090.lock`
    with the idle rule. The same cells run, the gates through `--external-lock 9`. Its line:
    `W VERDICT card=5090 -> ..` (clauses (a), (b), (d)). Each boot also records the compute apps at its start.
  - The reader was dry-run on P2's demote and promote receipts mapped as two arms (`INCOMPLETE` there, from the
    mapping's boot counts). It read the target's re-hash share at about 0.74 ms and its `Sources` job at 1.2 ms.
