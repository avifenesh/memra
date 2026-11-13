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
  every insert is clamped to live free memory less the driver headroom and the transient reserve
  (`unified_insert_budget`), the reserve admission charges the arrival. The next arrival finds a
  session active and books the whole growth, so the overload the booking guards (sessions admitted
  under a growing cache) is guarded as before. With ample memory nothing changes: an idle arrival
  fit the booked reading before and fits the larger one now.
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
- Memory memra holds outside its pool after boot (lazily captured graphs, driver workspaces)
  counts as transient, so a request within that margin of the capacity gets a 503 that retrying
  does not clear.
- No GB10 cell reaches the 400 with this model on its own: the largest charge at 262,144 context
  is about 8.3 GB against a capacity near 100 GB. The 400 path is shown with the teeth door
  `MEMRA_ADMIT_RESERVE_MB` and in the unit tests.
