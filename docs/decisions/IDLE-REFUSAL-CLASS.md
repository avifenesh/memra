# An idle server refuses by the request's size, not the moment's memory (2026-11-13)

Date: 2026-11-13, review folds 2026-11-14. Issue memra#1304, branch `fix/1304-admit-class`,
receipts `research/admit-idle-class-20261113/`. Devices: one DGX Spark (GB10, unified memory),
Qwen3.8-27B GGUF (NVFP4 trunk, Q5_K output head) with its masked MTP draft, and with the DFlash2
drafter for the copy-publisher cells, 262,144 context, the served defaults; one RTX 5090 Laptop
(discrete), Qwen3.5-9B Q8_0, for the foreign-pressure cell, and Qwen3.8-27B GGUF with the DFlash2
drafter for the discrete publisher cell.

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
  publication, which runs it after the reservation. The counting and the reservation are
  engine-free (the publisher passes the cache's geometry and the device's reading in), so the
  tests drive them with 31-G's numbers; what a reservation evicted is reaped before the copy
  allocates on it (memra#1278), and an entry that ends up larger than its reservation is said
  aloud at the insert. A refused publication prints its typed
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
- **Both refusals name the largest request that fits now** (review fold). The capacity subtracts
  only what memra itself has come to hold, so memory another process keeps for good (another
  process on a discrete card) left a request near the capacity with a 503 that no retry clears,
  and the 400's "largest admissible context" named a size that then got that 503. The same
  arithmetic on the reading now gives the size that fits the moment: the 503's message carries
  it ("the largest prompt plus max_tokens that fits now is N tokens"), and the 400 carries it
  beside the capacity's figure when it is the smaller.
- **The figure is one the caller can send** (second review). It is found by re-keying the
  charge's context term, the cost model's own exact context allocation, to a smaller context
  with everything else held, and bisecting (`largest_context_within`); no slope is assumed. That
  takes in the ring geometry (the first fold added the ring's bytes per token to a slope that
  already held them), every state a choice group builds (its rows and one snapshot: with one
  row's term a group's overshoot shed more tokens than the request has, and no size was named,
  the first RTX 5090 take), and each device's own charge and free bytes where the charge is per
  device (the figure is what fits on all of them). It is then put in the caller's units: the
  charged context less the request's slack over prompt plus max_tokens (8 rows, 64 on a route
  that can speculate), because a caller that sent the charged context as prompt plus
  max_tokens was charged the slack on top and refused again with the same figure
  (`largest_request_within`). When nothing fits there is no figure.
- **A promotion owns the caller's hit** (second review, a bug the first fold introduced). `admit`
  resolved its device hit to an index and then ran the host promotion. With the reservation in
  front of it, a promotion on unified memory could evict (a removal swaps the pool's last entry
  into the hole) and then fail, where before a failed promotion had evicted nothing; the caller
  read its old index: outside the pool, or at another prompt's entry. `host_promote_prefix_hit`
  now takes the hit by reference and leaves it at the promoted entry or at the same entry's
  current index. On unified memory the hit is leased across the promotion, so the reservation
  cannot take the entry the request would otherwise be served from; on a discrete card it is
  followed by its id and no lease is taken (a lease's release moves recency). The candidate's
  recency in the host tier moves only when the reservation will go ahead.
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

Review fold (2026-11-14), measured on the fold's code as build id `408ad3a2d420`, before the
branch's rebase onto main `63c351a2f4`, which changes no file of `crates/memra-server` and none
the cells execute:

| cell | result |
|---|---|
| GB10, the branch's six cells again on the fold build (1.8 and 1.5 GB, 2.4 GB, ample, the 20 GB pair, the teeth door) | the same status, code, `Retry-After`, tokens and text in each |
| GB10, DFlash2 drafter, a 131k prompt, ample memory | the boundary entry is copied and published: 4,131.5 MB, the bytes the preflight counted (`preflight_bytes=4131520512 entry_bytes=4131520512`) |
| the same, `MemAvailable` held at the 10,176 MB charge plus 600 MB by a holder that gives memory back as the server grows | admitted with nothing booked; 12,137 MB free at the publication, budget 8,535 MB, the copy fits and is published; the same text |
| the same, the charge plus 1,500 MB held by a fixed amount of other memory | admitted; 6,366 MB free at the publication, budget 2,765 MB: `insert refused: entry 4131520512 exceeds budget 2765296228 (live free memory holds the insert budget under its configured 49153134592; snapshot preflight, model q38)` before anything is allocated; the request completes with the same text; `MemAvailable` never under 4,642 MiB |
| RTX 5090 Laptop, the native-choices `kv` phase: another process takes all but 1 GiB of the card, then an `n = 8` request on the idle server | 503 `overloaded`, `Retry-After: 5`, "the largest context that fits now is 712 tokens" (available 1,193 MB, required 1,517 MB, capacity 16,424 MB); the stored receipt of the same cell on the earlier server is a 400; 200 once the memory is freed |

Sixteen source red arms and the gate's (the stored 400 receipt under the new verifier) each fail
for their stated reason; the list is in the research README.

Second review fold (2026-11-14), on main `0db086f87a`, measured on build id `f41206bfaadf`:

| cell | result |
|---|---|
| RTX 5090 Laptop, the native-choices `kv` phase again | 503 `overloaded`, `Retry-After: 5`, "the largest prompt plus max_tokens that fits now is 659 tokens" (available 1,194 MB, required 1,517 MB, capacity 16,423 MB; the request is charged its prompt plus `max_tokens` plus 64 rows, so the first fold's 712 was a size the caller could not send) |
| RTX 5090 Laptop, Qwen3.8-27B with the DFlash2 drafter, prefix cache on, one request twice, main `0db086f87a` against the head | the same in both arms: `insert (dspark-boundary): 7968 tokens, 477.7MB`, the second request restored from it with the drafter tail, four identical texts; the head counts the publication (`preflight_bytes=477724672 entry_bytes=477724672`) and reserves nothing on a discrete card |
| GB10, the first fold's eight cells again (1.8 and 1.5 GB, 2.4 GB, ample, the 20 GB pair, the teeth door, the DFlash copy with ample memory and under a fixed holder) | the same status, code, `Retry-After`, tokens and text in each; the 1.8 GB refusal names "218" (2,129 MB free under the 2,170 MB charge), the 1.5 GB one names nothing because the charge's fixed part is over the free memory; the copy is refused before allocating at 6,319 MB free (budget 2,718 MB) |
| GB10, the host tier on (`MEMRA_KV_HOST_MB=8192`), the host-tier identity gate, MTP route and DFlash2 drafter | all green in both: the promotion's room is made first (`evict (snapshot preflight, LRU)`), the entry that gives way is demoted, the promoted state's digest matches, and the request served through the round trip writes the bytes of the same request served cold with the tier off |

Thirty-two source red arms (the sixteen above, the second review's six mutations that had
passed the suite, ten for this fold) and the verifier's each fail for their stated reason.

## Limits

- The capacity is read once at boot. A server booted under host pressure has a smaller capacity,
  and a request between that and the box's real capacity gets the 400 as before this change.
- Memory memra holds outside its pool after boot (lazily captured graphs, driver workspaces), and
  memory another process keeps for good, count as transient, so a request within that margin of
  the capacity gets a 503 that retrying does not clear. Its message names the largest request
  (prompt plus `max_tokens`) that fits now, which is the caller's way out.
- The copy preflight is unified-memory only. On a discrete card a copying publisher still
  allocates before the cache evicts for it; there the allocation fails with a CUDA out-of-memory
  error when the card is full and the publication is dropped
  (`a_discrete_card_keeps_its_copy_publication_order` pins that path as it was).
- No GB10 cell reaches the 400 with this model on its own: the largest charge at 262,144 context
  is about 8.3 GB against a capacity near 100 GB. The 400 path is shown with the teeth door
  `MEMRA_ADMIT_RESERVE_MB` and in the unit tests.
