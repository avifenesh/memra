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
