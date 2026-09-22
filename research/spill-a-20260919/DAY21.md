# WP-A day 21: Move 2 slice 2, the hit restore off the tick behind rule 3's reader fence

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, resumed from `62aa92279`; `origin/main` `4bb2afb63` merged
`--no-ff` as `2c656a3d4` (clean; marker census OK). Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses an engine-touching range otherwise);
no qualification is claimed for any cell; every cell below is `executed-not-qualified`.

## Task 1, pre-registration (this section is committed before any slice-2 code)

### The op shape: a borrowed destination, a pinned borrowed source

Day 20 found that a capture's source is the live session cache's plane (borrowed) and named the engine's
`D2dCapture` (borrowed source, owned registered destination). The restore has the mirror problem and one more: its
DESTINATION is the admitted request's fresh session cache (`pp::new_cache_planned`), owned by the session for its
whole life and never moved into the engine's registry; its SOURCE is a published device entry's `PrefixPlane` (a
plain `CudaSlice<u8>` the device LRU owns), which today is NOT registered with the engine either (a captured plane
comes back through `take_plane` into a `PrefixPlane` before publication). Registering the source would mean moving
the entry's planes out of the entry for the copy's duration, and the entry must stay servable to other hits while a
restore reads it. So the restore class is a same-device copy from a BORROWED source span into a BORROWED destination
span, with nothing owned by the registry on either side, and its typed op is the engine's `D2dRestore` with
`CudaTransfers::submit_d2d_restore`. What makes the borrowed source safe is the device LRU's PIN: `PrefixCache::pin`
takes the entry out of the LRU index (`oldest_evictable` never names it; `evict_all`, `evict_all_demoting`, the trims
and the tenant purge's device half skip pinned entries by construction), so no path can free the source under the
running copy. The pin is taken at submit and becomes the request's serving pin at re-admission ("the pin count carries
it"). What makes the borrowed destination safe is the park: the request that owns the cache is parked on the requeue
and nothing reads or writes its cache until the owner stream's wait on the copy's completion event is installed.

### The contract in `memra-tier` terms

- Ticket: one per restore (every KV plane of the entry, K and V, one `Epochs`), on the copy stream behind a
  producer fence recorded on the owner stream at submit (after the recurrent-state copies, below).
- Producer-side guarantee: the source entry is published (its bytes complete: an on-tick capture is stream-ordered
  before this submit on the owner stream; a copy-stream capture published only after its events) and PINNED from
  submit to acknowledge. The pin is the contract's producer guarantee for a borrowed source.
- Reader fence (rule 3, `conformance/reader_fence.rs`, re-bound to a D2D): the restore's items are NOT fenced at
  submit. At the settle, after every completion event is observed complete, the OWNER stream's wait on each item's
  event is installed (`install_consumer_wait`, now over unfenced D2D items too) and only then is the destination
  `ready`; the parked request's first prime chunk is issued on the owner stream after re-admission, behind that wait.
  A prime issued before the wait is unordered whatever the copy's state (the contract orders through the fence, not
  through observation).
- Receipt: byte count, epochs, completion event; no checksum term (slice 3's decision, as for the capture), so the
  host-contract gate refuses a restore item by construction and no path publishes a restore through it.
- Retire: `retire(ticket, None)` after the wait is installed (an unpublished-in-the-engine's-sense, landed ticket:
  the destination leaves through the caller, not through `take_destination`), then `acknowledge`, then the pin
  passes to the serving session.

The frozen schedule: `d2d_restore_ready` (a landed copy without the installed wait is not `ready`; a prime issued
before the wait is unordered and fails the schedule; the source pin is held from submit through acknowledge) with
CPU bindings and a red arm (`prime_early`: a prime issued between the landing and the wait).

### The recurrent state in this slice

The KV rows are the byte-span class (`CudaSlice<u8>` planes, `bytes` from the head). The recurrent state
(`conv_state`, `ssm_state`, `CudaSlice<f32>`) is not in that class and this slice keeps it on the OWNER stream at
submit (`engine.copy_into`, exactly the OFF statements), stream-ordered before the producer fence and so before the
first prime chunk. On the 27B that is about 157 MB of a 4096-token entry's 278 MB: the slice moves the rows, not the
fixed term. Stated here so the stall cell's reading is bounded correctly; moving the f32 planes is a later slice once
the class admits typed f32 spans.

### The `Restoring` state, on the session

`Restoring` is a state of the parked REQUEST, not of the entry: the entry stays published, servable and pinned. The
worker holds at most one (`HostPrefixCache::restoring`): the request's id (`request_id`, the HTTP envelope's), its
pool key, the source pin, the entry's tokens and boundary logits, the destination cache, the ticket, and `ready`
(the wait installed, the ticket retired and acknowledged; the cache is complete and waits for its request).

| path | on a `Restoring` request |
|---|---|
| admission of the SAME request (its re-admission) | pending: parked again (`requeue.push_back`, FIFO, never shed); ready: admitted, and the hit site takes the ready cache and the restore's pin instead of allocating and copying; the hit accounting is the OFF hit's (`hits`, `hit_tokens`, the `hit:` line) |
| admission of another request | its own whole-entry device hit takes the tick program (the synchronous restore); one restore in flight per worker; a second hit on the same entry is served from the same published source, the pin count carries both |
| the tick top | polls the ticket after the capture poll; on every event complete installs the owner-stream wait, releases the producer fence, retires, acknowledges: `ready`; a ready restore whose request has not re-admitted within three tick tops is dropped typed (its cache freed, its pin released) |
| eviction, trim, reclaim, the eviction sink | the source is pinned: not a candidate anywhere (by construction, unchanged code); the reclaim and every trim settle the restore first (`Block`) so no copy is running when a trim quiesces the pool |
| tenant purge | settles first (`Block`); the purged tenant's ready restore is dropped (cache freed, pin released) so a revoked tenant's bytes are never primed on after the purge's receipt; another tenant's stays ready |
| shutdown | a host wait on the events, then drop; nothing is primed after a stop |
| the tier latched off, the capture or Move 1 paths | independent: a restore reads device memory only; a `Latched` restore settle latches the tier off (below) |
| a second restore while one is pending | never submitted: the probe answers "through" and the request takes the tick program |

### The retire seam (the lead's note on revuto #634, received mid-day, after the first worker commit)

Slice 1's finding: a capture's source is a live session's plane and the engine retains none of it, so the lead
now settles a pending `Capturing` entry `Block` before any session leaves `active` (integ36, `bd6584f9b`, merged
into this lane as `14b2d7b0f` before the refusal-path mirror and the census below). The restore's mirror is built into its ownership rather than added at
the seam: its DESTINATION cache is owned by the pending state (`HostPrefixCache::restoring`) and never by a
session while the copy is in flight; the request that will own it is parked on the requeue, not in `active`, so
no retire, park, rewind or prime can meet the cache before `host_restore_take_ready` hands it over, and that
hand-over requires `ready` (every completion event observed complete, the owner-stream wait installed). Its
SOURCE is pinned from submit through the hand-over or the typed drop (a drop with a pending contract settles
`Block` first), and a pinned entry is out of every eviction index by construction. The census
`every_path_that_meets_a_restoring_request_settles_or_ignores_it_as_stated` pins both: the cache leaves the
pending state exactly once, under the `ready` check, and the pin is taken before the submit and carried to the
serving session. The lead's second item, a refused submission whose producer event is still pending, is mirrored
verbatim: the restore's refusal path drains the owner stream, releases the fence, and latches the route off typed
if it still will not release (never `let _`); the same census pins drain, then release, then latch.

### Failure paths, typed, one line each

- A refusal before or at submit (the fresh cache's allocation, the OFF validation, the producer fence, the engine's
  `submit_d2d_restore`): the cache is dropped, the pin released, one `[prefix-cache] restore refused (contracts door):
  ..; tick program` line, a one-tick memo (`restore_cold`, by request id) so the same admission takes the OFF program
  (the synchronous restore, or the cold prime when the OFF path would) without a second attempt or a second line.
- A lost observation at the settle (`Latched`): the copy may still be writing the destination and reading the
  source, so NEITHER is freed: the destination cache is forgotten and the source pin is kept forever (a leak by
  design, the frozen `Entry::drop` rule mirrored), the tier latches off, and the restore path latches to the tick
  program for the boot: `[prefix-cache] RESTORE OFF-TICK DISABLED: ..`. The parked request re-admits and takes the OFF
  program on a fresh cache.
- A ready restore whose request does not consume it (the request failed a later admission check, was refused, or
  disconnected): dropped typed at the next probe or after three tick tops (cache freed, pin released).
- A ready restore whose entry no longer matches its request (the entry gone, or the prompt no longer a whole-entry
  hit on it): dropped typed; the OFF program serves.
- The fail-closed arm (#622 mirrored): a pending restore missing its cache or its ticket never re-admits its request
  onto it; with a submitted ticket and no cache the ticket is settled and retired through the engine where reachable
  and the tier and the restore path latch off; with a cache and no ticket the cache drops whole (nothing submitted).

Nothing half-restored is ever primed on: the only cache a request primes on is the OFF path's (validated and copied
synchronously before the prime) or a ready restore's (every event observed complete, the owner-stream wait
installed before the request was re-admitted).

### The one-program law

The restored rows are the bytes a synchronous D2D would have produced: the same `memcpy_dtod` per plane from the
same source plane at the same offsets and lengths, into the same fresh cache the OFF path allocates, the same
`copy_into` for the recurrent state, the same `len`, `len_d` and `pos`; the request then primes its suffix on the
same program. The hit gate's identity clause (`spec-on-cache-hit-gate.sh`, plain and spec arms: a hit's bytes
equal the cold bytes) and the identity gate are the proof; any digest difference is a FAIL of the slice, never a
tolerance. The delayed-copy-stream fault (cell (iii)) is owed with slice 3's fault cells and not run today.

### Cells (target card, BOX3, one RTX PRO 6000 Blackwell, 600 W; the collector; `executed-not-qualified`)

Gates as on day 20: identity default and plain, OFF and ON; failure OFF and ON; the fault gate; the twin gate OFF and
ON; the hit gate OFF and ON (its own flock); the GPU unit cells `option_b_*`, `option_c_*`, the engine's
`d2d_capture_*` and the new `d2d_restore_*`. If identity, fault or hit is red in either arm the slice does not stand:
it stays `wip:` with the red receipt.

**The restore stall cell (memra#536 Move 2 cell (ii)).** `stall-cell.sh off|on` boots the day-20 server (prefix cache
ON at 1024 MB, host tier 8 GiB, `MEMRA_SERVE_SPEC=0`, door OFF or ON); the harness `stall_cell.py --mode restore`
(a NEW arm; the four earlier arms are byte-for-byte unchanged): an untimed setup posts the intruder prompt once so
its grid seed captures and publishes; then in every timed run the intruder re-posts the SAME prompt at the tenant's
24th token with `max_tokens=1`: a whole-entry HIT whose restore copies the entry (about 5088 tokens on the 27B) and
primes the 32-to-35-token suffix on the tick; its `cached_tokens` is recorded per run. N=5 per arm per order, both
orders inside each boot; the two door arms run OFF, ON, ON, OFF in one collector lock hold. Rules, fixed before the
run (ms, the harness's `stall_median`):

- Admissibility: `errors=0`, `tenant_text_identical=True`, and every intruder's `cached_tokens >= prompt_tokens - 64`
  on every receipt (every run a hit). A receipt that fails admissibility decides nothing and is reported as such.
- C1, ON against OFF, same hold, per pass: `on_under_off` if `stall_median(ON) <= stall_median(OFF) - 6.0`; `flat`
  within 6.0 either way; `on_over_off` above (the day-18 promote arm's threshold).
- C2, the copy's own duration: the ON arm's `server_restore_ms` (the `restore landed off the tick` lines, ms from
  submission to the observing poll), a list with its median; a reading, no threshold.
- What the cell can and cannot say: the intruder's suffix prime is on the tick in both arms, the recurrent state
  (about 157 MB) stays on the owner stream in both arms, and only the KV rows (about 150 MB at 5088 tokens) move; on
  this card a 150 MB D2D is well under a millisecond, so the expected reading is `flat`, which records that the rows'
  on-tick share is under the cell's resolution. A stall difference larger than the bound in either direction is a
  finding to explain, not a verdict on the door. Nothing here decides the door.
