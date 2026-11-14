# An idle server refuses by the request's size, not the moment's memory (2026-11-13)

Date: 2026-11-13. Issue memra#1304, branch `fix/1304-admit-class`, receipts
`research/admit-idle-class-20261113/`. Device: one DGX Spark (GB10, unified memory), Qwen3.8-27B
GGUF (NVFP4 trunk, Q5_K output head) with its masked MTP draft, 262,144 context, the served
defaults.

## Problem

On GB10 a fresh, idle server whose host `MemAvailable` was about 2 GB answered a 1k request with
HTTP 400 `context_length_exceeded` (memra#1257's short cells `01-T` and `02-B`):

```text
[admit-oom] capacity reject: model="q38" ctx=1607 does not fit an IDLE box (available 0MB) — HTTP 400 context_length_exceeded
```

With `MEMRA_PREFIX_CACHE_MB=0`, in the same state, the request was admitted and completed (cells
`11-T`, `12-B`). Two things were wrong.

1. **The booking.** On unified memory admission books the prefix cache's allowed growth (budget
   less resident) as committed (2026-11-02, the GB10 quality panel: eight 131k-token sessions
   under a cache that grew 4 GB per session took `MemAvailable` to about 1 GiB with the booking
   off; ledger in [FLAGS-HISTORY.md](../FLAGS-HISTORY.md#removed-doors-2026-11-02-gb10-unified-memory-prefix-booking-seam-booking-kept)).
   The derived budget is half the post-load pool, about 52 GB here, so with the cache empty and
   `MemAvailable` near 2 GB the booked reading was 0. On an idle box the booking protects no one:
   it exists so that admitted sessions are not starved by the cache growing underneath them.
2. **The class.** The idle reject's reasoning was "nothing else is running, so the request does
   not fit the box at any arrival rate". That holds on a discrete card whose free memory is a
   property of the box. It does not hold when the reading counts host memory other processes hold
   and a booking for growth that has not happened. A client told its context is too long shortens
   or gives up when it should retry. The "largest admissible context" hint came from the same
   reading.

## Chosen

- **No growth booked on an idle box.** `unified_prefix_growth_booking` books 0 while no session is
  active (the same `active.is_empty()` the gate drains on) and the whole allowed growth otherwise.
  Until the next admission decision the cache can grow only by the arrival's own entries, and
  each of them reserves its room before it allocates, against live free memory less the driver
  headroom and the transient reserve (`unified_insert_budget`; the reserve is the one admission
  charges the arrival). The next arrival finds a session active and books the whole growth, so
  the overload the booking guards (sessions admitted under a growing cache) is guarded as before.
  With ample memory nothing changes: an idle arrival fit the booked reading before and fits the
  larger one now.
- **An entry that copies reserves its room first** (review fold). The first version of this record
  said every insert is clamped to live free memory. That was true of the entry's place in the
  cache and false of its bytes: the plain snapshot checked before allocating, the MTP publisher
  lends the session's rows and allocates no plane, but the DFlash and GLM-5 boundary publishers
  (trunk planes, the drafter tail, latent slices), an MTP publication whose plane cannot lend, and
  a host promotion allocated their copy first and were clamped by the insert afterwards. With the
  growth booked, admission had pre-funded those bytes; an idle arrival admitted with nothing
  booked could copy gigabytes into memory the box did not have. On unified memory
  `prefix_copy_preflight` now counts what the publication will allocate
  (`spec_publication_bytes`, the tail's size from `DflashKv::tail_bytes`) and makes room with the
  snapshot's own preflight before the first byte; the publisher hands the tail's export to the
  publication, which runs it after the reservation. A refused publication prints its typed
  `[prefix-cache] insert refused:` line and the request is served unpublished. A discrete card
  keeps its path (copy, then evict to the budget): a copy that does not fit a full card fails as
  a CUDA out-of-memory error, which the publisher already handles by dropping the publication.
- **The idle refusal follows the request's size.** After the boot calibration the worker reads each
  model-owned device once (`read_idle_capacity`: the effective free memory admission reads, and the
  pool's live bytes), with the weights loaded, no session and the cache empty. An idle, drained
  refusal reads the same plans at that capacity, less what memra itself has come to hold at rest
  since (pool bytes still live on the idle box) and never below the reading now
  (`idle_capacity_free`). A request that does not fit it gets the 400 `context_length_exceeded`,
  its "largest admissible context" computed from that capacity. A request that fits it is short of
  the moment's memory and gets 503 `overloaded` with `Retry-After` (5 s), the class the server
  already uses for running out of VRAM; `[admit-oom] idle memory short:` logs it. The boot line
  `[admission] idle capacity (post-load, cache empty)` states the reading.
- **Both refusals name the largest context that fits now** (review fold). The capacity subtracts
  only what memra itself has come to hold, so memory another process keeps for good (another
  process on a discrete card) left a request near the capacity with a 503 that no retry clears,
  and the 400's "largest admissible context" named a size that then got that 503. The same
  arithmetic on the reading now (`largest_context_within`) gives the size that fits the moment:
  the 503's message carries it ("the largest context that fits now is N tokens"), and the 400
  carries it beside the capacity's figure when it is the smaller.
- **A refusal by the live reading says so** (review fold). An insert the live clamp refuses is
  logged as `insert refused: entry N exceeds budget M (live free memory holds the insert budget
  under its configured C; ...)` and no longer spends the one-time "first prefix insert is larger
  than the entire byte budget" warning, which is about a shape the configured budget cannot hold.

## Rejected

- **429 for the transient refusal.** 429 is the QoS shed: the caller is over a budget and a retry
  in a couple of seconds works. An idle server short of memory is the box out of capacity, which a
  router should see (and fail over on) as 503; the busy box's memory defer past its budget stays
  429 because there the caller competes with load.
- **Waiting on an idle box.** Nothing the server runs will free the memory; only other processes
  can. Queueing would hold the request against the client's deadline with no bound on the wait.
- **Booking only the arrival's own entry on an idle box.** The entry is optional (the insert guard
  refuses it when it does not fit, and the request is served cold next turn), so refusing a request
  because its cache entry would not fit inverts the priority.
- **Changing the booking on a busy box.** The panel's failure is the multi-session case, measured;
  this change leaves it as it is.
- **The copy preflight on a discrete card** (review fold). The hazard is an allocation that
  succeeds past the driver headroom and takes host memory, which only unified memory has. Moving
  the eviction ahead of the copy on a discrete card would change the publishers' order on the
  primary target with no measurement behind it.

## Measured

GB10, one fresh server per cell, `mem_holder.py` holding `MemAvailable` at the cell's level, then
memra#1221's g1k request (charged 2,170 MB: cost, draft state and the 1,536 MiB reserve). The
post-load capacity read at boot was 104,140 to 104,956 MB. Main is `01960ee1b1`; the branch ran on
two builds (the second moves one `let` so a source pin reads it; same program), with the same
outcome in every cell pair.

| `MemAvailable` | main | branch |
|---|---|---|
| about 1.8 GB (the issue's recipe) | 400 `context_length_exceeded`, `available 0MB` | 503 `overloaded`, `Retry-After: 5`: 2,025 MB free with nothing booked, under the charge |
| about 1.5 GB | 400 | 503 `overloaded`, `Retry-After: 5` |
| about 2.6 GB | 400, `available 0MB` | admitted: 127 tokens, `stop`, the text every admitted g1k wrote |
| about 100 GB (ample) | admitted, the same text | admitted, the same text |
| about 20 GB, a g28k then a g1k 3 s later | both 400 | the g28k admitted idle; the g1k defers while it runs (`VRAM defer: 1 active, effective free 0MB`, the whole growth booked), then completes with the same text |
| ample, the charge raised to about 210 GB (`MEMRA_ADMIT_RESERVE_MB=200000`) | not run | 400 `context_length_exceeded` against the 104,951 MB capacity |

No out-of-memory line in any cell, in the server or the kernel. Red arms (tests kept, behavior
restored): the booking ignoring idle, the refusal always a 400, and the reject branch answering
`EngineError::context_length` each fail their test. Receipts:
[research/admit-idle-class-20261113/](../../research/admit-idle-class-20261113/README.md).

## Limits

- The capacity is read once at boot. A server booted under host pressure has a smaller capacity,
  and a request between that and the box's real capacity gets the 400 as before this change.
- Memory memra holds outside its pool after boot (lazily captured graphs, driver workspaces), and
  memory another process keeps for good, count as transient, so a request within that margin of
  the capacity gets a 503 that retrying does not clear. Its message names the largest context
  that fits now, which is the caller's way out.
- The copy preflight is unified-memory only. On a discrete card a copying publisher still
  allocates before the cache evicts for it; there the allocation fails with a CUDA out-of-memory
  error when the card is full and the publication is dropped
  (`a_discrete_card_keeps_its_copy_publication_order` pins that path as it was).
- No GB10 cell reaches the 400 with this model on its own: the largest charge at 262,144 context
  is about 8.3 GB against a capacity near 100 GB. The 400 path is shown with the teeth door
  `MEMRA_ADMIT_RESERVE_MB` and in the unit tests.
