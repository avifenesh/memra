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

## Task 2, what landed (under `MEMRA_KV_HOST_CONTRACTS=1`, default OFF, decide-by 2026-10-05)

Commits, in order: the merge `540fd2d8c`; the pre-registration `dc5eb64bc`; the tier rule `46eed44d0`; the worker
route and the FLAGS.md sentence `0a4ff4b0b`; the box scripts `185c57b4f` (the tree the target card ran); records
after it.

**Tier crate** (`crates/memra-tier/src/conformance/d2d_capture.rs`,
`tests/contracts/d2d_capture_bindings.rs`). `D2dDraftCaptureFixture` extends the capture fixture with the class of
every item (`Role::Key`/`Role::Value` trunk rows, `Role::Draft` draft rows), a per-class landing
(`trunk_completes` fires the trunk items' events and leaves the draft items running) and the witnessed receipt term
per class. The schedule `d2d_capture_draft_publish`: one batch carries both classes under one ticket; while the
copies run nothing is landed, publishable or retirable and nothing is witnessed; with the TRUNK landed and a draft
item still running the batch is still not landed, a publish is refused `NotReady` (both or neither), `retire` is
`Busy`, the witnessed classes are the trunk's alone and the host-contract gate stays closed; once every event
completes the bytes are exact per item, both classes are witnessed, the gate admits the batch, publication happens
exactly once, the ticket retires with no consumer fence and acknowledges, and every destination of both classes
comes back. The red arm `d2d_capture_draft_published_with_the_draft_unlanded_fails`: a caller that publishes on the
trunk's landing alone fails the schedule, and the binding shows the shape of that failure (the index names an entry
whose draft plane has not landed, `retire` `Busy`, the draft class unwitnessed). CPU bindings: `DraftCaptureBatch`
with the modelled per-item digest, the 27B's 34-item batch and the 9B's 18-item batch, the red arm under
`catch_unwind`, and the per-class receipt clause on its own. Contracts suite: `87 passed` (84 before).

**Worker** (`crates/memra-server/src/worker.rs`; no engine seam was needed: a draft source is a
`&CudaSlice<u8>`, slice 1's own class). Slice 1's submit half is now `host_capture_submit` over a `CaptureSubmit`
description (the boundary `pos`, the borrowed trunk sources, the OWNED recurrent planes, the optional borrowed
draft source, the owned logits and `last_h`, the program class for the tenant salt, the trace role, the bytes the
shell already owns, and what the submitted line says about the recurrent state); `prefix_capture_off_tick` keeps
its admission, its recurrent clones and its lines and calls the core with `draft: None`,
`class: HostTierEntryClass::Plain`, `trace_role: "snapshot"` and the day-20 recurrent note verbatim. The trunk loop
now copies rows `[0..pos)` rather than `[0..l.len)`: on the seed route those are the same (it requires
`l.len == cache.pos`), on the spec-boundary route `pos` sits below a live plane the burst appended past.

`prefix_spec_capture_off_tick` is the spec-boundary publisher's route, asked by
`prefix_insert_from_spec_boundary` after its early returns (`pp_host_bounce_active`, `pos == 0`,
`pos > committed.len()`, empty logits, an SWA ring) and before the latent arm. It answers `OnTick` with the
capture handed back untouched (the OFF program runs unchanged, its own refusals included) for: the door OFF, no
transfer engine, the capture path latched, a latent-bearing cache, a capture carrying latent tails, a TP cache, a
capture whose snapshot is not at `pos`, a trunk layer with `0 < len < pos`, or a cache with no KV plane at the
boundary. Otherwise a pending capture settles first (`"a second capture"`, never two in flight), the key is
re-checked after that settle (a `capture skipped (contracts door)` line where the settled capture already published
this prefix), the draft source is read exactly as the OFF program reads it (shorter than the boundary prints the OFF
line `spec publish: draft plane shorter than boundary {pos}; entry published trunk-only` and the trunk is routed
alone; `None` on a ring-backed scratch is silent, as OFF), the class is `MtpDraft` when a draft plane rides and
`Plain` otherwise, `bytes` counts the recurrent planes and `last_h` the capture already owns, and the core takes it
with `trace_role: "spec-snapshot"`.

In the core, the draft plane is step 2b: rows `[0..pos)` of the scratch as two more `D2dCapture` items of the SAME
batch after the trunk rows, under the same producer fence, ticket and receipt (`items=34` on the 27B against 32
plain), with the same fresh-plane discipline as the trunk (the OFF allocator, `register_device` at the destination
generation, a retained twin, every refusal unwinding every registered plane of both classes through its twin).
`CapturePlane` carries its class (`CapturePlaneClass::Trunk` with its slot, or `Draft`); the settle answers
`Done { kv, draft }` and a plane of EITHER class that does not come back is `Latched` (the entry drops, nothing
published: both or neither); the landing fills `shell.draft` beside the trunk slots before the state turns `ready`;
publication prints the OFF publisher's own trace role, which rides the pending state. Both demotion sweeps settle a
pending capture BLOCKING before they consume the session whose planes the copy stream reads: the MTP demotion
(`"a spec demotion"`), because `into_demoted` drops the `MtpScratch` the draft items borrow, and the DSPARK
demotion (`"a dspark demotion"`) for the same reason, stated even though its publisher never routes. The retire,
park, trim, purge and shutdown seams already settled it since slice 1. CPU census
`day24_spec_boundary_capture_paths_are_named` pins every path above by its source literal, including that the core
has exactly two callers and that the seed route's line is unchanged.

One stated difference from the OFF program: an alloc or `register_device` failure of a DRAFT destination refuses the
whole capture (typed `capture refused (contracts door)`, every registered plane back, nothing published) where OFF
publishes trunk-only silently. No token moves either way (a missing entry is a cold prime), and the identity gates
prove the two arms equal. No new flag, no new numeric program, no new `unsafe`, no new pending or ready state.

**CPU battery on this tree** (local, under the 1200% CPU quota and a 28G cap): `cargo fmt --all -- --check` clean;
clippy `-D warnings` all targets on tier, engine and server rc=0 with zero errors; the GPU-less cross-target pass
(`DOCS_RS=1`, `x86_64-unknown-linux-gnu`) rc=0 with zero errors; server lib `809 passed; 0 failed; 14 ignored`;
engine lib (CPU) `531 passed; 0 failed; 30 ignored`; tier suites all `ok` (`87 passed` contracts, plus 7, 62, 18, 6,
32, 53 and 4); `tools/check-flags.sh` `no uncovered runtime names`; `tools/check-conflict-markers.sh` `OK (no
conflict marker line in tracked source or docs)`; `git diff --check` clean.

## Task 3, the target card (BOX3, one RTX PRO 6000 Blackwell, 600 W), one sitting

Tree `185c57b4f` on the box worktree `/root/wt-a` (branch `lane-a-day24`, clean), binary
`1a1185fb554dd47b11e2d7b0747975002e7c57c6d4261d16cbb757d1faf5ee23`
(`pro-single-day24/box/gates/binary.sha256`). Every collector cell through
`tools/tier-battery.py --rig pro-single --external-lock` (lock `/tmp/memra-gpu.lock`, `qualification: false`,
`status: executed-not-qualified`, `LOCK.json` owner `collector`, mechanism
`inherited-flock-same-open-description`) and the hit gate under its own `flock` on the same lock
(`pro-single-day24/hitgate.sh`; the gate's `--external-lock` arm landed on C day 28 and is not used here, so this
battery keeps the day-23 shape). Zero lock retries despite lane C sharing the card (its server held 17 GB at
launch and never held the lock when a cell started). Driver `pro-single-day24/driver.sh`. Windows: the gate cell
10:59:54Z to 11:08:00Z (485.6 s), the hit gate 11:08:00Z to 11:09:30Z, the unit cell 11:09:30Z to 11:12:47Z.
Envelope (collector sampler, `command.gpu.csv`): gates 1938 samples, 34.26 to 506.63 W, 33 to 60 C; unit cells 785
samples, 32.04 to 94.73 W, 33 to 40 C. Receipts mirrored to `pro-single-day24/box/` (bins not mirrored).

Verdicts, verbatim, one per cell:

| Cell | Verdict line |
|---|---|
| identity, default (spec) environment, door OFF | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| identity, default environment, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| identity, plain (`MEMRA_SERVE_SPEC=0`), door OFF | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| identity, plain, door ON | `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` |
| failure, door OFF | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` |
| failure, door ON | `KV-HOST-SPILL FAILURE GATE: ALL GREEN` |
| contract fault gate, all cells (ON by construction) | `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN` (93 `ok:`) |
| twin gate (27B, 8 turns), door OFF | `... V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` |
| twin gate, door ON | `... V1=ok V2=ok V3=ok V4=ok V5=ok V6=ok -> PASS` (zero `REFUSED`) |
| hit gate, door OFF | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (61 `ok:`, 0 `FAIL:`) |
| hit gate, door ON, the tier ARMED (C day 27's arm) | `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` (68 `ok:`, 0 `FAIL:`) |
| unit, server (`option_b_*`, `option_c_*`) | `test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 815 filtered out; finished in 0.28s` |
| unit, engine `d2d_*` | `test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 556 filtered out; finished in 3.21s` |

### The acceptance clause, read against the receipts

**The hit gate's ON arm, the spec-on boot** (`pro-single-day24/box/gates/hitgate-on/qwen-on-server.log`).
Engagement: the six `ok: door arm:` lines of both boots (armed, door ON, no latch) and
`ok: door arm: 30 route submission(s) across the two boots (capture, restore, demote or promote off the tick)`
(day 23 read 19 on the predecessor tree: the 11 new ones are the spec-boundary captures). The census is unchanged:
`census spec-on: entries published: 11 draft-bearing (insert (spec-boundary)), 1 plain (insert (seed)), 0 other`
and `census spec-off: entries published: 0 draft-bearing (insert (spec-boundary)), 2 plain (insert (seed)), 0
other`. Counted on the spec-on boot: **11 `capture submitted off the tick (spec-boundary)` lines, all 11 carrying
the draft plane** (the census's 11 draft-bearing publishes, every one routed), 1 `capture submitted off the tick
(seed)` (the plain `samp-noplane` entry), 12 `capture published off the tick`, 12 `D2D capture receipt ...
require=ok` (11 `items=34`, 1 `items=32`), and the day-23 restores still engaging: 13 `restore submitted off the
tick` of which 12 carry the draft plane, 12 `draft plane ready`, 12 `spec restore: ... + draft plane from cache`.
Zero `capture refused`, zero `CAPTURE OFF-TICK DISABLED`, zero `capture dropped`, zero `capture skipped (contracts
door)`, zero `entry published trunk-only`, zero `restore refused`, zero `TIER DISABLED`. Verbatim, with the count
of each shape (the draft rows elided in the table above, quoted here):

- 7 x `capture submitted off the tick (spec-boundary): 64 tokens, 34 planes (158.9MB) on the contracts door's copy
  stream; recurrent state and boundary logits taken by the spec engine at the prime stop; draft plane 64 rows
  (118.8KB) in the batch`
- 3 x the same at `96 tokens, 34 planes (159.9MB)` and `draft plane 96 rows (178.2KB) in the batch`
- 1 x the same at `128 tokens, 34 planes (161.0MB)` and `draft plane 128 rows (237.6KB) in the batch`
- 1 x `capture submitted off the tick (seed): 64 tokens, 32 planes (158.8MB) on the contracts door's copy stream;
  recurrent state cloned at the boundary on the owner stream` (the seed route's line, unchanged)
- receipts: 7 x `items=34 bytes=2019328`, 3 x `items=34 bytes=3028992`, 1 x `items=34 bytes=4038656`, 1 x
  `items=32 bytes=1900544`, every one `require=ok` with equal source and destination digests
- published: 7 x `capture published off the tick (spec-boundary): 64 tokens complete after 1 poll(s), ... (settled
  synchronously by a session retire)`, 3 x the same at 96 tokens, 1 x at 128 tokens, and 1 x `capture published off
  the tick (seed): 64 tokens complete after 2 poll(s), ... (tick-top poll)`

The spec-off twin boot: 2 `capture submitted off the tick (seed): ... 32 planes (158.8MB)`, 2 receipts `items=32
... require=ok`, 2 published, 3 `restore submitted off the tick` with no draft plane, zero refused, dropped or
latched lines (the spec-boundary route never fires there: the boot publishes no draft-bearing entry).

**The identity clause.** Every identity `ok:` of the ON arm holds with the spec side's entries now CAPTURED through
the route as well as restored through it: `ok: r1 spec==plain byte identity`, `ok: r2 spec==plain byte identity`,
`ok: r3 spec==plain byte identity`, `ok: g1 spec==plain byte identity`, `ok: g2 spec==plain byte identity`, `ok: fc
sampled full-cover hit bytes == cold leader bytes (same seed)`, `ok: sx sampled suffix hit reproduces byte-for-byte
at one seed`, `ok: sp penalized sampled hit bytes == cold leader bytes (same seed)`, `ok: g4 reproduces its
publisher's continuation byte-for-byte (snapshot round-trip)`, and the three sampled seeds' `ok: s7 / s1234 /
s99991 sampled hit bytes == cold leader bytes (same seed)` with `ok: ... acceptance == cold acceptance exactly`.
The OFF arm reads the same `ALL GREEN (qwen)` with 61 `ok:`. This is the first hit-gate receipt on any card where
the identity clause covers BOTH halves of the door's D2D program for draft-bearing entries: the capture at
publication and the restore at the hit.

**Identity gate, default environment, door ON** (`pro-single-day24/box/gates/identity-default-on/`): the
spec-boundary route engaged there too: 2 x `capture submitted off the tick (spec-boundary): 64 tokens, 34 planes
(158.9MB) ...; draft plane 64 rows (118.8KB) in the batch`, receipts `seq=1` and `seq=2` `items=34 bytes=2019328
... require=ok` with equal digests (`1ef7121cc4e7f088434198c481650acd...` and `272e34c504a43f4f9939b45b41a0b729...`),
2 `capture published off the tick (spec-boundary)`, and the day-23 restores beside them (2 x `restore submitted off
the tick: ... 34 planes ...; draft plane 64 rows (118.8KB) in the batch`, 2 x `draft plane ready`, 2 x `spec
restore: 64 of 102 prompt tokens + draft plane from cache`), `ALL GREEN (teeth=0)`.

**The fault gate's two D2D cells still refuse** (verbatim): `capture receipt refused:
source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=1462aed093ef5fee..` and `restore receipt
refused: source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=1462aed093ef5fee..`, the same two
digests as day 23, with every clause of both cells `ok` (including `ok: d2d-capture: the capture route latched
typed on the mismatch`, `ok: d2d-capture: nothing was published off the tick`, `ok: d2d-capture: r2's capture ran
on the tick after the latch (an insert follows the refusal)`, and the restore twin's `ok: d2d-restore: r1's capture
receipt was accepted (the green arm, live)`). 93 `ok:`, unchanged from day 23: the fault gate's cells run the PLAIN
arm (`MEMRA_SERVE_SPEC=0`), whose entries carry no draft plane, so the spec-boundary route is not in this gate's
reach (stated, not claimed).

### Findings

1. **Every spec-boundary publish the gate makes routed, with the draft plane, and the identity clause held.** All
   11 `insert (spec-boundary)` entries of the spec-on boot were captured off the tick as 34-item batches carrying
   their draft plane, each with a witnessed receipt over both classes, each published, and every spec-against-plain
   byte identity of the gate holds. Zero refusals, zero trunk-only publishes, zero latches. Item 10's count (11 of
   the boot's 12 entries) is exactly the routed count.
2. **On this gate's shape the spec-boundary capture settles at the SESSION RETIRE, not at a tick-top poll.** All 11
   published `after 1 poll(s)` with `settled synchronously by a session retire`, 106.3 to 204.6 ms from submission
   to completion, while the seed capture of the same boot published at a `tick-top poll` `after 2 poll(s)`. That is
   slice 1's retire seam behaving as designed and as this slice requires: the sources are the LIVE session's trunk
   planes and draft scratch, so a session leaving `active` must settle the copy first. The consequence is that for
   a session which retires in the same tick as its prime stop (the gate's shape: one turn, publish, retire) the
   owner thread still pays a host wait for the 159 MB copy at the retire, so the moved share is not the copy's full
   cost on this shape. A session that keeps decoding past the drain sweep publishes at a later tick top instead.
   Pricing that difference is Move 2 owed item 3 (the isolating stall cell); nothing here is tuned or claimed.
3. **The draft plane's share at publication matches the restore side exactly.** A 64-token spec-boundary entry's
   capture batch is 2,019,328 B against the plain seed's 1,900,544 B: 118,784 B for the draft plane, 1,856 B per
   token, the same figure day 23 measured on the restore side and the same 1.9 KB per token census item 10
   estimated. The 96- and 128-token shapes scale linearly (178.2 KB and 237.6 KB).
4. **No finding against the engine or the gates.** First sitting green in every cell; no new engine seam was
   needed for this slice (the draft source is slice 1's own borrowed-source class).

### Budget

About 2.9 agent-hours against 5 (the merge and pre-registration to 10:31Z; the tier rule and the worker by 10:50Z;
the box build 10:56Z to 11:00Z; the card 10:59Z to 11:13Z; records after). The slice landed whole: nothing is left
as `wip:`.

### Box state and cleanup

`/root/wt-a` at `185c57b4f` on `lane-a-day24` (clean); `/root/spill-receipts/a-day24/` kept on the box and mirrored
here (bins not mirrored); the transfer bundle and the mirror tarball removed on both ends; no server of mine
running at the end. Lane C's server on the shared card was never touched. The local RTX 5090 was not touched today
(the door gates on this tree are owed there with the lock, as on days 18 to 23).
