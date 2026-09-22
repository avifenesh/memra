# WP-A day 24: Move 2 owed item 2, the capture half: the spec-boundary capture through the door

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, resumed from `0ebdde372`; the lead's
`lane/spill-integ39-20260922` (`1d57f4a57`: A day 23, C day 28, the lead record) merged `--no-ff` as `540fd2d8c`
(clean; PR #642 was OPEN at the start of the day, so `origin/main` was not merged separately). Every push today in
the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses an engine-touching range
otherwise); no qualification is claimed for any cell; every cell below is `executed-not-qualified`.

## Task 1, pre-registration (this section is committed before any day-24 code)

### What this slice moves

Slice 1 (day 20) routes only `prefix_insert_from_session`, the `seed` and `lcp-split` publishes of a plain-primed
session. Census item 10 (lane C, `HOSTPREFIX-DOOR.md`) counted what stays on the tick: every
`prefix_insert_from_spec_boundary` publish, one per prime stop of an MTP spec session at the drain sweep. On the hit
gate that is 11 of the spec-on boot's 12 entries (`insert (spec-boundary)`), each carrying a draft plane. Day 23
landed the restore half of the draft plane; this slice lands the capture half, under `MEMRA_KV_HOST_CONTRACTS=1`
(default OFF, decide-by 2026-10-05), following item 10 point by point.

### The OFF program this slice must equal, byte for byte

`prefix_insert_from_spec_boundary` (worker.rs), called from the drain sweep of the tick (the `for cap in
mtp_captures` loop, after the burst committed past the boundary) with the LIVE spec session's `cache_ref()`,
`draft_plane_ref()` and a `SpecBoundaryCapture` the spec engine took at the prime stop. In order: the early returns
(`pp_host_bounce_active`, `pos == 0`, `pos > committed.len()`, empty boundary logits, an SWA ring); the latent arm
(a latent-bearing cache publishes only with boundary tails and the latent flag, else a loud refusal); per trunk
layer `l` of the live cache: an MTP head layer (`len == 0`) absent, `l.len < pos` a loud refusal, otherwise
`alloc_u8(pos x k_tok_bytes)` and `alloc_u8(pos x v_tok_bytes)` then `copy_u8_into` of rows `[0..pos)` from `l.k`
and `l.v` on the OWNER stream (the live plane's `len` runs PAST `pos`: the burst appended rows after the boundary;
rows below `pos` are append-only for the session's lifetime, the same law slice 1's seed route relies on); the
recurrent state and the boundary logits come OWNED from the capture (`cap.snap.conv`, `cap.snap.ssm`,
`cap.logits`, `cap.last_h`: taken by the spec engine at the boundary, no copy at publish); the draft plane: from
`draft_plane_ref` (`None` when ring-backed), a source shorter than `pos x tok_bytes` publishes TRUNK-ONLY with a
typed line, otherwise `alloc_u8` twice and `copy_u8_into` of rows `[0..pos)` from the scratch's `k` and `v` on the
owner stream (an alloc or copy failure drops only the plane, trunk-only entry, silent); `bytes` = trunk + recurrent
(f32 x 4) + draft + `last_h x 4` (+ latent, + the DSPARK tail, both absent here); the entry
(`toks = committed[..pos]`, `pos`, `last_logits = cap.logits`, `draft`, `last_h`);
`trace_prefix_entry_state(.., "spec-snapshot", why)`; `px.insert_demoting(pool_key, e, "spec-boundary", ..)`.
No budget preflight (`prepare_snapshot`) exists on this publisher; `insert_demoting` evicts at insert.

### The ON program, as designed by item 10

1. **One core, two publishers.** The slice-1 route's submit half (fresh planes from the OFF allocator registered
   at the destination generation with retained twins; the producer fence; ONE batch on the copy stream behind it;
   the one-shot `d2d-delay-capture` arm; the refusal that unwinds every registered plane) is factored into
   `host_capture_submit` over a `CaptureSubmit` description (the boundary `pos`, the borrowed trunk sources with
   their per-token bytes, the owned recurrent planes, the optional borrowed draft source, the owned logits and
   `last_h`, the entry class for the tenant salt, the trace role). `prefix_capture_off_tick` (seed, lcp-split) keeps
   its admission, its recurrent clones and its lines unchanged and calls the core; the new
   `prefix_spec_capture_off_tick` is the spec-boundary publisher's route, called from
   `prefix_insert_from_spec_boundary` after its early returns and before the latent arm, and answers
   `OnTick(cap)` (the capture handed back untouched for the OFF program), `Submitted` or `Refused`.
2. **What the spec route refuses by name (`OnTick`, the OFF program runs, its own refusals kept):** the door OFF,
   no transfer engine or the capture path latched; a latent-bearing cache or a capture with latent tails (the
   `glm5-boundary` shape); a TP cache; a trunk layer with `0 < len < pos` (the OFF program's loud refusal fires);
   a cache with no KV plane at the boundary. The DFlash tail (`dspark-boundary`, `export_tail`) is a different
   publisher (`drain_dspark_prefix_capture`) and never reaches this route. A draft source shorter than `pos` rows
   mirrors the OFF program: the OFF line (`spec publish: draft plane shorter than boundary {pos}; entry published
   trunk-only`) and the trunk is routed alone.
3. **Two borrowed source spans, one batch.** The trunk items are rows `[0..pos)` of each live trunk plane
   (`pos x k_tok_bytes`, `pos x v_tok_bytes`; slice 1's `D2dCapture` class, borrowed source into an owned
   registered destination) and the draft items are rows `[0..pos)` of the spec session's draft scratch
   (`scratch.kv.k`, `scratch.kv.v` through `draft_plane_ref`), the same class, two more items after the trunk
   rows, so no new engine op shape is needed (`items=34` on the 27B: 16 layers x 2 + 2). One ticket, one
   producer fence, one receipt.
4. **The producer fence** is one owner-stream event recorded at the drain sweep, after the burst committed past
   `pos`: every kernel that wrote a trunk row or a draft-head row below `pos` was issued on the owner stream
   before it (the boundary's last draft-head write included). The recurrent state and logits are already owned
   by the capture and need no copy and no fence.
5. **The `Capturing` entry owns the fresh draft planes beside the fresh trunk planes.** `CapturePlane` gains its
   class (trunk slot or draft); the settle (`host_kv_planes_settle_capture`) takes every plane back through its
   twin and answers `Done { kv, draft }`; the shell's `draft` is filled beside its `kv` slots; a plane of EITHER
   class that does not come back is `Latched` (the whole entry drops, nothing published): both or neither. The
   shell carries `pos`, `toks = committed[..pos]`, the owned recurrent planes, `last_logits`, `last_h`, and
   `bytes` equal to the OFF entry's (trunk + recurrent + draft + `last_h x 4`).
6. **Slice 3's receipt term** covers the draft items as two more lanes of the same `ReceiptScratch` (source digest
   behind the fence, the copy, the destination digest); the `D2D capture receipt` line's `items=` counts them
   (34 against 32); a refused receipt takes every fresh plane of both classes back and drops it, publishes
   nothing, and latches the tier and the capture route (the day-22 `ReceiptMismatch` arm, unchanged in shape).
7. **Publication** is the slice-1 publication: at a tick top after every item's event is observed complete,
   through `insert_demoting` with `why = "spec-boundary"` (the same `[prefix-cache] insert (spec-boundary)` line
   the gates' census counts) and `trace_prefix_entry_state(.., "spec-snapshot", ..)` (the trace role rides the
   pending state). The spec session's re-arm on the next hit reads the published draft rows only through the
   day-23 restore route or the OFF constructor; nothing here changes the reader.
8. **The borrow's lifetime (item 10's open question).** The sources are the live session's trunk cache and its
   draft scratch. Slice 1's retire seam already settles a pending capture `Block` before any session leaves
   `active` (a retire, a park into `SpecReuseEntry`, an OOM teardown). The one path that frees a source INSIDE
   the tick is the MTP demotion (`s.spec.take()` then `into_demoted`, which drops the `MtpScratch`), and it
   runs right after the drain sweep. This slice settles the pending capture `Block` before the first spec
   session is consumed by the demotion sweep (`"a spec demotion"`), the DSPARK demotion the same (its cache moves
   without a free, stated for completeness). A census test names every settle site.
9. **Under alloc pressure the ON route differs from OFF in one stated way:** an alloc or `register_device`
   failure of a DRAFT destination refuses the whole capture (typed `capture refused (contracts door)`, every
   registered plane back, nothing published) where OFF publishes trunk-only silently. No token moves: a missing
   entry is a cold prime; the identity gates prove the two arms equal.

### Fail-closed arms (mirroring #638 and integ38)

- The one-shot `d2d-delay-capture` arm is spent before any admission refusal (the engine takes it first in
  `submit_d2d_capture`; the worker's `take_d2d_fault(true)` at the arming site is shared by both publishers).
- The `Block` settle names every event it reads (`synchronize` covers the items and the receipt event since
  `269ef2cec`); this slice adds no event to the landing.
- No new pending or ready state: the draft rides the existing `Capturing` entry, so the parked-only wait guard,
  the idle waits and the orphan handling gain no new arm.
- No new `MEMRA_*` read. No new numeric program: the copy is not one; the published bytes are the OFF entry's
  bytes and the reader is unchanged.

### The tier crate first

`memra_tier::conformance::d2d_capture` gains the draft rule beside `d2d_capture_publish`:
`D2dDraftCaptureFixture` (the capture fixture plus the class of every item, a per-class landing so the trunk
items can land with the draft still running, and the witnessed receipt term per class) and the schedule
`d2d_capture_draft_publish`: one batch carries both classes under one ticket; a publish with the trunk landed and
the draft unlanded is refused `NotReady` (both or neither); after the landing every item of both classes is
witnessed and the host-contract gate admits the batch; publication happens exactly once, retire with no consumer
fence, acknowledge, every destination of both classes comes back. The red arm
`d2d_capture_draft_published_with_the_draft_unlanded_fails`: a caller that publishes on the trunk's landing alone
fails the schedule. CPU bindings in `tests/contracts/d2d_capture_bindings.rs` (18- and 34-item batches; the red
arm under `catch_unwind`). Then the worker (no engine seam: the draft source is a `&CudaSlice<u8>`, slice 1's
class).

### Acceptance (pre-registered)

On the target card (BOX3, one RTX PRO 6000 Blackwell, 600 W), through the collector, one sitting where the card
allows (lane C shares the card today; the lock is retried boundedly):

- **Hit gate**, OFF and ON (the ON arm armed per C day 27): `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` in BOTH
  arms; the ON arm's spec-on boot must show `capture submitted off the tick (spec-boundary)` lines with `draft
  plane N rows (X KB) in the batch` for its draft-bearing publishes (count them against the gate's own census, 11
  `insert (spec-boundary)`), each followed by a `D2D capture receipt ... items=34 ... require=ok` and a `capture
  published off the tick (spec-boundary)`; the day-23 restores still engage (`restore submitted off the tick ...
  draft plane N rows ... in the batch`, `draft plane ready`, `spec restore: ... + draft plane from cache`); zero
  `capture refused`, zero `CAPTURE OFF-TICK DISABLED`, zero `TIER DISABLED`, zero `capture dropped`; the
  identity clause (spec-on text equal to spec-off text) holds on every namespace. The `ok: door arm: N route
  submission(s)` count grows by the routed spec-boundary captures (19 on day 23).
- **Identity gate** default and plain, OFF and ON: `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` x4; the
  default ON arm shows the spec-boundary captures routed with the draft plane in the batch.
- **Failure gate** both arms: `KV-HOST-SPILL FAILURE GATE: ALL GREEN` x2, unchanged.
- **Fault gate**, all cells: `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`; the `d2d-capture` cell (plain arm, seeds)
  still refuses (`capture receipt refused: source_digests_sha256=.. destination_digests_sha256=..`, two different
  digests); the `d2d-restore` cell the same.
- **Twin gate** OFF and ON: `-> PASS` x2.
- **Unit cells**: server `8 passed` or more; engine `d2d_*` under the lock, all passed; the tier contracts suite
  with the new bindings.

If the ON arm shows the spec-boundary capture refused by name (a `capture refused (contracts door)` line on a
publish the gate expected to route, or a `capture dropped` line), the refusal is quoted and the day stops at the
finding; nothing is relaxed.

### What stays outside, by name

The DFlash tail (`dspark-boundary`, the drafter's `export_tail`, owned and fenced by the drafter); TP shards and
latent planes (`glm5-boundary`, refused by name to the OFF program); the fanout leader's snapshot and the pause
sweep's boundary snapshot (`prefix_snapshot` direct); the recurrent f32 state's copy at the boundary (taken by the
spec engine at prime time, on the owner stream, before this publisher runs: Move 2 owed item 1 as before).

### Budget

5 agent-hours. If the slice cannot land whole, the tier conformance lands as a `wip:` with the gates green and the
remainder stated.
