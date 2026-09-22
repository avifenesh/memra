# WP-A day 23: Move 2 owed item 2, the restore half: the draft-bearing restore through the door

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, resumed from `b94bee809`; the lead's
`lane/spill-integ38-20260922` (`643ecbb28`: A day 22 plus the lead fixes `269ef2cec`, `9d4b17761`, `cc754b476`,
and C days 26 and 27) merged `--no-ff` as `bee1ca332` (clean; marker census OK). PR #639 (integ38 to main) was OPEN
at the start of the day, so `origin/main` was not merged separately. Every push today in the announced
`MEMRA_RELEASE_QUALIFICATION_MODE=development` mode (the #589 hook refuses an engine-touching range otherwise); no
qualification is claimed for any cell; every cell below is `executed-not-qualified`.

## Task 1, pre-registration (this section is committed before any day-23 code)

### What this slice moves

Slice 2 (day 21) refuses a draft-bearing entry BY NAME at the probe's class check (`e.draft.is_some()`, silent,
`return false`): every `insert (spec-boundary)` entry restores on the tick program. Census item 11 (lane C day 26,
`HOSTPREFIX-DOOR.md`) counted what that is on the hit gate: 11 of 14 entries and 12 of 16 hits per gate run are
draft-bearing on both rigs. C day 27 armed the hit gate's ON arm; its identity clause now covers the route for the
plain hits and the draft-bearing restore stayed owed. This slice lands it, under `MEMRA_KV_HOST_CONTRACTS=1` (default
OFF, decide-by 2026-10-05), following item 11's design point by point.

### The OFF program this slice must equal, byte for byte

`HybridModel::spec_session_from_restored_deferred` (spec.rs): after the trunk cache was restored (the carrier), the
pre-checks (`ensure_usable`, an MTP head, `pos > 0`, `cache.pos == pos`, `draft_len == pos`, generation room), then
`MtpScratch::new` ALLOCATES the draft scratch inside the constructor, the geometry checks (a ring-backed scratch
refused, the entry's `k_tok_bytes`/`v_tok_bytes` equal to the scratch layout, `pos <= cap`, source planes at least
`pos x tok_bytes`), then `copy_u8_into(scratch.kv.k, 0, draft.k, pos x k_tok_bytes)` and the V twin on the OWNER
stream, then `scratch.set_len(pos)` (host `len` and the 4-byte `len_d`), then the session is built and the deferred
prime's first draft-head read walks rows `[0..pos)` of that scratch. The decision that this REQUEST takes the draft at
all is `spec_restore_refusal` at admission (sampled under the load guard or the penalty window, an entry without
`last_h`, and so on), after `spec_eligible` (K policy, grammar, MTP head, no DFlash drafter).

### The ON program, as designed by item 11

1. **The scratch is allocated at the probe** (`host_restore_park_probe`), before any session exists, through a new
   engine seam `HybridModel::alloc_restored_draft_scratch` that runs the SAME geometry checks in the same order and
   returns a `RestoredDraftScratch` (a pub wrapper over the crate-private `MtpScratch`; `MtpScratch` itself stays
   private). The scratch is owned by the pending `Restoring` state (`PendingRestore::draft`) beside the trunk cache,
   never by a session while the copy is in flight. `set_len(pos)` runs at the probe on the owner stream, before the
   producer fence (the trunk's `len`/`len_d` statement, mirrored; a different buffer than the K/V bytes, so its order
   against the copy cannot move a byte).
2. **The destination** is a borrowed `CudaViewMut<u8>` of exactly `pos x k_tok_bytes` and one of exactly
   `pos x v_tok_bytes` into the scratch's K and V planes (`RestoredDraftScratch::destinations`), the slice-2 restore
   class (`D2dRestore`, `submit_d2d_restore`); no new op shape, no registry, no new `unsafe`.
3. **The source** is borrowed from the pinned entry's `draft.k`/`draft.v` under the SAME pin the trunk restore takes
   (`px.pin` at the probe; no second guarantee).
4. **The producer fence** is the one owner-stream event `host_restore_submit` records after the recurrent-state copies
   and the `len`/`len_d` statements; the draft items are two more ops in the SAME batch (one ticket, one receipt,
   `items=34` on the 27B and `items=18` on the 9B against 32 and 16 plain).
5. **Rule 3's reader wait** is installed at the settle over every unfenced item of the batch (`install_consumer_wait`
   already fences every non-D2H item, the draft items included), BEFORE the parked request re-admits. The wait
   therefore precedes session construction; the constructor (`spec_session_from_restored_ready`) takes a ready,
   pre-filled scratch and runs no copy. The restored draft rows become readable only after the installed wait: the
   deferred prime's first draft-head read is issued by the session, which exists only after `host_restore_take_ready`
   handed the scratch over under `ready`.
6. **`spec_restore_refusal` is evaluated at the probe, before the submit**, with the probe's inputs: the whole-entry
   hit (`fed_len == e.pos`, `prompt_len`), the request's sampler (`Sampler::new(req.sampler_cfg.clone())`: greedy,
   penalty window), the doors (`spec_restore_sampled_on`, `spec_pen_session_on`), the load guard over the loop's own
   `active` and `queue + requeue` counts (`spec_load_demand(n_active + 1 + n_pending)`, the same reading `admit`
   takes), `last_h` and `last_logits`. Ahead of it the probe's spec pre-decision reuses the admission gate's own
   estimate (`estimate_spec`, `admission_request_may_spec`: MTP-capable, K policy over the projected wave,
   `MEMRA_SERVE_SPEC`, the peer probe), plus no grammar and no armed DFlash drafter for the model. A request that
   would serve plain on a draft-bearing entry submits the TRUNK restore alone and records why
   (`PendingRestore::draft_declined`); it never submits a draft plane it will not read. The admission body's decision
   stays authoritative: the two agree by construction on every input both read, and the two disagreement arms are
   typed and both correct (no numeric program is involved in either): (a) the probe submitted the draft and `admit`
   serves plain (a constraint or eligibility clause the probe cannot see): the ready scratch is DROPPED with one line
   naming the reason; (b) the probe declined the draft and `admit` takes the draft (the load reading moved while the
   request was parked): the OFF constructor copies the plane on the owner stream from the still-pinned entry, as
   today, with one line naming the probe's reason. Both arms are counted in the gates' logs; a nonzero count is a
   finding to report, never a clause to move.
7. **The geometry checks move to the probe**: a refusal there (alloc failure, ring, layout, cap, truncation) never
   refuses the TRUNK route; it declines the draft with the reason (`draft_declined`) and the request restores plain
   through the door, exactly what the OFF program does when the same check fails in its constructor (the hit serves
   PLAIN on the restored trunk).
8. **Slice 3's receipt term** covers both plane classes: the draft items are two more receipt lanes in the same
   `ReceiptScratch` (source digest behind the fence, the copy, the destination digest), `Completion::require`'s clause
   compares them, the `D2D restore receipt` line's `items=` counts them; a refused receipt drops the cache AND the
   scratch (the copy landed) and releases the pin (the day-22 `ReceiptMismatch` arm, one more field).

### Fail-closed arms (mirroring #638 and integ38)

- The pin is released at every drop that releases it today; the scratch travels with the cache: dropped where the
  cache drops (a ready orphan, a stale match, a tenant purge, shutdown, a refused receipt, a ready state its request
  serves plain), FORGOTTEN where the cache is forgotten (`Latched`: never a free under a possibly running copy).
- The fail-closed settle arms (a ticket without a cache, a cache without a ticket) are unchanged in shape; a scratch
  found beside a missing cache is forgotten when the ticket did not settle and dropped when it did.
- The retire seam's `Block` settle comes first, as today; the `Block` wait covers every event the settle reads (ruling
  34: the receipt event is in it since `269ef2cec`), and this slice adds no event to the landing.
- No new pending or ready state: the draft rides the existing `Restoring` state, so the parked-only wait guard and the
  orphan grace gain no new arm (they cover the state as a whole).
- No new `MEMRA_*` read. No new numeric program: the copy is not one, and the session's first token is produced by the
  same deferred prime over the same bytes in both arms.

### The tier crate first

`memra_tier::conformance::d2d_restore` gains the draft rule beside `d2d_restore_ready`: `D2dDraftRestoreFixture`
(the restore fixture plus the item classes as `Role`s, the deferred prime's first draft-head read on the reader
stream, and the witnessed receipt term per class) and the schedule `d2d_restore_draft_ready`: one batch carries both
classes (`Role::Key`/`Role::Value` and `Role::Draft`) under one ticket and one pin; landing is not readiness for
either class; the installed wait makes both ready; a draft-head read after it is ordered; the receipt term is
witnessed for BOTH classes after the landing (the destination digest equal to the source digest per item), so the
host-contract gate admits the batch; the red arm `d2d_restore_draft_read_before_its_wait_is_unordered`: a prime that
reads draft rows before the installed wait is unordered, landed copy or not, and a later wait does not order it. CPU
bindings in `tests/contracts/d2d_restore_bindings.rs` (a `witnessed` fixture mode) with the red arm run under
`catch_unwind`. Then the engine seam (`RestoredDraftScratch`, `alloc_restored_draft_scratch`,
`spec_session_from_restored_ready`, the OFF constructor refactored to call the same tail after its own alloc and
copies), then the worker.

### Acceptance (pre-registered)

On the target card (BOX3, one RTX PRO 6000 Blackwell, 600 W), through the collector, one sitting where the card
allows:

- **Hit gate**, OFF and ON (the ON arm armed per C day 27): `SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)` in BOTH arms;
  the ON arm's spec-on boot must show `restore submitted off the tick` for its draft-bearing hits (count the lines
  against the gate's own entry-class census; the submitted line names the draft plane), zero `restore refused`, zero
  `RESTORE OFF-TICK DISABLED`, zero `TIER DISABLED`, zero draft disagreement lines; the identity clause (spec-on text
  equal to spec-off text) holds, which is the byte-identity against OFF for the draft-bearing rows.
- **Identity gate** default and plain, OFF and ON: `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` x4, unchanged.
- **Failure gate** both arms: `KV-HOST-SPILL FAILURE GATE: ALL GREEN` x2, unchanged.
- **Fault gate**, all cells: `KV-HOST-CONTRACT-FAULT GATE: ALL GREEN`; the `d2d-restore` cell still refuses
  (`restore receipt refused: source_digests_sha256=.. destination_digests_sha256=..` with two different digests).
- **Twin gate** OFF and ON: `-> PASS` x2.
- **Unit cells**: server `8 passed` or more; engine `d2d_*` under the lock, all passed; the tier contracts suite with
  the new bindings.

If the door arm shows the draft route refused by name (a `restore refused (contracts door)` or a draft-declined line
on a hit the gate expected to route), the refusal is quoted and the day stops at the finding; nothing is relaxed.

### Budget

5 agent-hours. If the slice cannot land whole, the tier conformance and the probe-side refactor land as a `wip:` with
the gates green and the remainder stated.

## Task 2, what landed (under `MEMRA_KV_HOST_CONTRACTS=1`, default OFF, decide-by 2026-10-05)

Commits, in order: pre-registration `0717ab470`; the tier rule `d16cb66e9`; the engine seam and the worker
`daf9c30fc`; the box scripts `01971da3d`; the clippy fix `2b850b2b0` (the tree the target card ran).

**Tier crate** (`crates/memra-tier/src/conformance/d2d_restore.rs`, `tests/contracts/d2d_restore_bindings.rs`).
`D2dDraftRestoreFixture` extends the restore fixture with the class of every item (`Role::Key`/`Role::Value` trunk
rows, `Role::Draft` draft rows), the deferred prime's first draft-head read on the reader stream, and the witnessed
receipt term per class. The schedule `d2d_restore_draft_ready`: one batch carries both classes under one ticket and
one pin; landing is not readiness for either class; after the landing every item of both classes is witnessed
(destination digest equal to the source digest) and the host-contract gate answers `NotReady` while unfenced and `Ok`
once the wait is installed; the one install fences both classes; a prime and a draft-head read after it are ordered;
retire, acknowledge, then the pin hand-over. The red arm `d2d_restore_draft_read_before_its_wait_is_unordered`: a
draft-head read before the installed wait is unordered, landed copy or not, and a later wait does not order it. CPU
bindings: a `witnessed` fixture mode (the modelled digest per item), 18- and 34-item batches (the 9B's and the 27B's
`items=`), the red arm under `catch_unwind`, and the unwitnessed refusal (`Corrupt`). Contracts suite: `84 passed`
(81 before).

**Engine** (`crates/memra-engine/src/spec.rs`). `RestoredDraftScratch`: a pub value over the crate-private
`MtpScratch` that a worker can own before a session exists; `HybridModel::alloc_restored_draft_scratch` runs the OFF
constructor's geometry checks in the OFF order with the OFF messages verbatim (an MTP head, `MtpScratch::new` at
`max_ctx`, a ring-backed scratch refused, the per-token layout, `pos <= cap`, the source lengths) and allocates;
`destinations()` borrows exactly `pos x k_tok_bytes` and `pos x v_tok_bytes` as `CudaViewMut<u8>` (the slice-2 restore
class, no new op shape); `set_len(e)` is the OFF `set_len(pos)` statement; `copy_from_entry` (private) is the OFF
constructor's two `copy_u8_into` statements. `spec_session_from_restored_ready` builds the session over a READY
scratch (the pre-checks, plus `scratch.kv.len == pos`), running no draft copy; `spec_session_from_restored_deferred`
(the OFF constructor) now calls the same alloc, then `copy_from_entry`, then `set_len`, then the shared tail
`spec_session_from_restored_scratch` (the suffix feed, the draft fill of the suffix rows, the boundary token, the
session assembly: unchanged from the day-22 tree). A source census
(`day23_restored_draft_census::the_ready_constructor_copies_nothing_and_both_constructors_share_the_tail`) pins the
order, the two callers of the tail, the two copy statements in `copy_from_entry` alone, and the OFF messages. No
`unsafe`, no new kernel, no new flag.

**Worker** (`crates/memra-server/src/worker.rs`). The probe (`host_restore_park_probe`) no longer refuses
`e.draft.is_some()` by name (DFlash-tail, TP, latent entries still are). On a draft-bearing entry, after the pin: the
request's sampler is built from its config, `spec_restore_refusal` is evaluated with the whole-entry hit, the penalty
window, the doors, the load guard over the loop's own `active` and `queue + requeue` counts (the same reading `admit`
takes), `last_h` and `last_logits`; `host_restore_draft_decision` (pure) folds in the admission gate's spec estimate
for this request (`estimate_spec`, the loop's own), an armed DFlash drafter for the model, and a grammar. `None`
allocates the scratch (`alloc_restored_draft_scratch` at the request's `ctx_cap`) and sets its length on the owner
stream BEFORE the producer fence; a refusal, a geometry failure or an alloc failure sets `draft_declined` with the
reason and one typed line (`restore draft plane declined at the probe (..)`), never refusing the trunk route.
`host_restore_submit` takes `Option<&mut RestoredDraftScratch>` and pushes the entry's `draft.k` and `draft.v` rows
`[0..pos)` as two more `D2dRestore` items of the SAME batch after the trunk rows, under the same producer fence and
the same pin; `bytes` and `sizes` count them; the receipt term covers them as it covers the trunk (`items=34` on the
27B). The submitted line gains `; draft plane N rows (X KB) in the batch` or `; draft plane not submitted (<why>)`.
`PendingRestore` owns `draft` and `draft_declined`: the settle's `Latched` arm forgets the scratch with the cache
(never a free under a possibly running copy), the `ReceiptMismatch` arm drops it with the cache (the copy landed),
the fail-closed arm drops it when the ticket's host wait settled and forgets it when it did not; every `Dropped`
path drops it with the state; `host_restore_take_ready` hands it over with the cache and the pin, under `ready` only,
and the landed line reads `; draft plane ready`. In `admit`, the ready scratch rides to the spec conversion:
`spec_session_from_restored_ready` when present (no copy), otherwise the OFF constructor with the arm named when the
probe had declined (`spec restore takes the draft the probe declined (<why>); the draft plane is copied on the owner
stream (the OFF program) from the pinned entry`); a ready scratch no spec session consumed drops typed after the
conversion block (`draft plane restored off the tick but the request serves plain (<why>); dropped`). CPU census
`day23_draft_bearing_restore_paths_are_named` pins every path above by its source literal;
`day23_draft_decision_submits_the_draft_only_for_a_request_that_reads_it` covers the pure decision. The day-22
census's `ReceiptMismatch` window is bounded at the arm's end instead of 500 characters (its assertions are
unchanged). `docs/FLAGS.md`: the door row's day-21 sentence names the MTP draft plane as routed since day 23 and a
day-23 sentence states the mechanism and the lines. No new `MEMRA_*` read.

**CPU battery on this tree** (local, under the 1200% CPU quota): `cargo fmt --all -- --check` clean; clippy
`-D warnings` all targets on tier, engine and server `Finished`; the GPU-less pass (`DOCS_RS=1`, cross-target) `Finished`;
server lib `808 passed; 0 failed; 14 ignored`; engine lib (CPU) `531 passed; 0 failed; 30 ignored`; tier suites all
`ok` (`84 passed` contracts); `tools/check-flags.sh` `no uncovered runtime names`; `tools/check-conflict-markers.sh`
OK; `git diff --check` clean.

## Task 3, the target card (BOX3, one RTX PRO 6000 Blackwell, 600 W), one sitting

Tree `2b850b2b0` on the box worktree `/root/wt-a` (branch `lane-a-day23`, clean), binary
`8725826875750242e65fd436f6d8e83b…` (`pro-single-day23/box/gates/binary.sha256`). Every cell through the collector
(`tools/tier-battery.py --rig pro-single --external-lock`, lock `/tmp/memra-gpu.lock`, `qualification: false`,
`disposition: executed-not-qualified`) or, for the hit gate (no `--external-lock` arm), under its own `flock` on the
same lock (`pro-single-day23/hitgate.sh`); zero lock retries (lane C's server left the lock before the driver
started; no compute app on the card at launch). Driver `pro-single-day23/driver.sh`, 09:54Z to 10:08Z. Envelope
(collector sampler, `command.gpu.csv`): gates 1964 samples, 31.97 to 507.02 W, 33 to 60 C; unit cells 15 samples,
53.9 to 99.28 W, 40 to 41 C. Receipts mirrored to `pro-single-day23/box/` (bins not mirrored).

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
| unit, server (`option_b_*`, `option_c_*`) | `test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 814 filtered out; finished in 0.23s` |
| unit, engine `d2d_*` | `test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 556 filtered out; finished in 3.17s` |

### The acceptance clause, read against the receipts

**The hit gate's ON arm (the spec-on boot, `pro-single-day23/box/gates/hitgate-on/qwen-on-server.log`).** Engagement:
`[prefix-host] on: budget 8590MB pinned cacheable host RAM (MEMRA_KV_HOST_MB, ...)`, `[prefix-host] contracts door ON
(MEMRA_KV_HOST_CONTRACTS=1): 1 model program identities ...`; the gate's own lines `ok: door arm: the host tier is
armed on the spec-on boot`, `ok: door arm: the contracts door is ON on the spec-on boot`, `ok: door arm: no latch line
on the spec-on boot`, and across the two boots `ok: door arm: 19 route submission(s) across the two boots (capture,
restore, demote or promote off the tick)` (C day 27 read 7 on this tree's predecessor: the 12 new ones are the
draft-bearing restores). The census the gate prints: `census spec-on: entries published: 11 draft-bearing (insert
(spec-boundary)), 1 plain (insert (seed)), 0 other` and `census spec-off: entries published: 0 draft-bearing (insert
(spec-boundary)), 2 plain (insert (seed)), 0 other` (item 11's counts, unchanged). The spec-on boot's route lines,
counted: 13 `restore submitted off the tick`, of which 12 carry the draft plane and 1 is the plain `np` hit; 0 `draft
plane not submitted`; 12 `draft plane ready`; 13 `D2D restore receipt ... require=ok`; 12 `spec restore: N of M prompt
tokens + draft plane from cache`; 0 `restore refused`, 0 `RESTORE OFF-TICK DISABLED`, 0 `TIER DISABLED`, 0 `restore
draft plane declined at the probe`, 0 `draft plane restored off the tick but the request serves plain`, 0 `spec
restore takes the draft the probe declined` (the two disagreement arms never fired). Verbatim (the ticket sequence
elided), with the count of each shape:

- 8 x `restore submitted off the tick: 64 tokens, 34 planes (158.9MB), ticket seq=N on the contracts door's copy stream;
  recurrent state copied on the owner stream; request parked; draft plane 64 rows (118.8KB) in the batch`
- 3 x `restore submitted off the tick: 96 tokens, 34 planes (159.9MB), ... request parked; draft plane 96 rows (178.2KB)
  in the batch`
- 1 x `restore submitted off the tick: 128 tokens, 34 planes (160.9MB), ... request parked; draft plane 128 rows
  (237.6KB) in the batch`
- 1 x `restore submitted off the tick: 64 tokens, 32 planes (158.8MB), ... request parked` (the plain `np` hit)
- receipts: 8 x `items=34 bytes=2019328 require=ok`, 3 x `items=34 bytes=3028992 require=ok`, 1 x `items=34
  bytes=4038656 require=ok`, 1 x `items=32 bytes=1900544 require=ok`
- landed: 6 x `restore landed off the tick: 64 tokens (158.9MB) complete after 1 poll(s), 2.4ms from submission to
  completion, ... (tick-top poll); draft plane ready`, 2 x the same at `2.3ms`, 3 x `96 tokens (159.9MB) ... 2.4ms ...;
  draft plane ready`, 1 x `128 tokens (160.9MB) ... 2.4ms ...; draft plane ready`, 1 x `64 tokens (158.8MB) ... 2.3ms`
  (plain)
- sessions: 5 x `spec restore: 64 of 106 prompt tokens + draft plane from cache [suffix queued]`, 3 x `64 of 119 ...
  [suffix queued]`, 2 x `96 of 119 ... [suffix queued]`, 1 x `96 of 135 ... [suffix queued]`, 1 x `128 of 128 prompt
  tokens + draft plane from cache [continuation]`

The spec-off twin boot: 3 `restore submitted off the tick: 64 tokens, 32 planes (158.8MB)` (its r2, r3, g2 hits, as
on day 27), 2 captures, 3 receipts `items=32 ... require=ok`, zero refused, disabled or draft lines.

**The identity clause.** Every identity `ok:` of the ON arm holds with the spec side's draft-bearing rows now
restored THROUGH THE ROUTE (the ready scratch, no draft copy) against the spec-off side's rows restored through the
same route: `ok: r1 spec==plain byte identity`, `ok: r2 spec==plain byte identity`, `ok: r3 spec==plain byte
identity`, `ok: g1 spec==plain byte identity`, `ok: g2 spec==plain byte identity`, `ok: fc sampled full-cover hit
bytes == cold leader bytes (same seed)`, `ok: sx sampled suffix hit reproduces byte-for-byte at one seed`, `ok: sp
penalized sampled hit bytes == cold leader bytes (same seed)`, `ok: s7 / s1234 / s99991 sampled hit bytes == cold
leader bytes (same seed)`, `ok: g4 reproduces its publisher's continuation byte-for-byte (snapshot round-trip)`. The
OFF arm reads the same `ALL GREEN (qwen)` with 61 `ok:` (its two boots have no `[prefix-host]` lines). This is the
first hit-gate receipt on any card where the identity clause covers the route for the draft-bearing hits.

**Identity gate, default environment, door ON** (`pro-single-day23/box/gates/identity-default-on/`): the route
engaged on the draft-bearing entries of the spec environment too: 2 x `restore submitted off the tick: 64 tokens, 34
planes (158.9MB), ticket seq=N ...; request parked; draft plane 64 rows (118.8KB) in the batch`, receipts `seq=4` and
`seq=5` `items=34 bytes=2019328 ... require=ok` with equal source and destination digests
(`1ef7121cc4e7f088434198c481650acd…`), `restore landed off the tick: 64 tokens (158.9MB) complete after 1 poll(s),
2.2ms ...; draft plane ready` and one at `77.7ms` (the poll cadence), 2 x `spec restore: 64 of 102 prompt tokens +
draft plane from cache`, zero declined, refused or disagreement lines, `ALL GREEN (teeth=0)`. The plain ON arm is
the day-22 shape (2 restores of `32 planes`, receipts `seq=6`, `seq=7`, `items=32 ... require=ok`).

**The fault gate's `d2d-restore` cell still refuses** (verbatim): `restore receipt refused:
source_digests_sha256=d11e5c4b614f3a68.. destination_digests_sha256=1462aed093ef5fee..`; `ok: d2d-restore: exactly one
refused D2D restore receipt naming two different digests`, `ok: d2d-restore: no restore receipt was accepted`, `ok:
d2d-restore: the restore route latched typed on the mismatch`, `ok: d2d-restore: the tier latched off`, `ok:
d2d-restore: nothing landed off the tick (nothing primed on the refused cache)`, `ok: d2d-restore: the tick program
restored r2 after the refusal`, `ok: d2d-restore: r1's text equals r2's`, `ok: d2d-restore: no ticket leaked (no
Capacity refusal, no leaked wording)`; the `d2d-capture` twin the same (`capture receipt refused: ...d11e5c4b614f3a68..
... 1462aed093ef5fee..`). 93 `ok:` in the gate (67 on day 25's tree plus the day-22 D2D cells' clauses); the plain
arm's entries carry no draft plane, so the draft route is not in this gate's reach (stated, not claimed).

**Cell (v), re-read on this tree as the price cell ran in the unit battery** (not a decision cell today, recorded):
`D2D-RECEIPT PRICE order=copy-first bytes=165675008 n_per_arm=5 ... copy_median=0.159 digest_median=0.168
pair_median=0.336 pair_over_copy=2.11` and `order=digest-first ... copy_median=0.158 digest_median=0.169
pair_median=0.337 pair_over_copy=2.13`: the day-22 reading (2.12x to 2.15x) reproduced within 0.02.

### Findings

1. **The draft route engaged on every draft-bearing hit the gate serves, with the identity clause holding.** 12 of
   the spec-on boot's 13 routed hits carried the draft plane (item 11's 12 of 16 hits, exactly), each landed after one
   poll with a witnessed receipt over 34 items, each built its session over the ready scratch, and every spec-against-
   plain byte identity of the gate held. No line of either disagreement arm fired, so the probe's pre-decision and
   the admission body agreed on every hit of both gates; the arms stay as typed census lines.
2. **The receipt's byte counts name the draft plane's share.** On the 27B a 64-token draft-bearing entry's restore
   batch is 2,019,328 B against the plain 1,900,544 B: 118,784 B for the draft plane (1,856 B per token, one layer's K
   and V), the 1.9 KB per token census item 10 estimated. The recurrent f32 state (about 157 MB per entry, the
   `158.9MB` of the submitted line) still rides the owner stream (Move 2 owed item 1).
3. **No new finding against the engine or the gates.** First sitting green in every cell; the day-22 cudarc finding
   did not recur (the fixture keeps tracking disabled); the unit cells' first-time green on this tree.

### Budget

About 1.6 agent-hours against 5 (the merge and pre-registration 09:20Z to 09:28Z; the tier, engine and worker
commits by 09:52Z; the card 09:54Z to 10:08Z; records after). The slice landed whole: nothing is left as `wip:`.

### Box state and cleanup

`/root/wt-a` at `2b850b2b0` on `lane-a-day23` (clean); `/root/spill-receipts/a-day23/` kept on the box and mirrored
here (bins not mirrored); the transfer bundles removed on both ends; no server of mine running at the end (the card
idle, lock free); local `/tmp/spill-a-day23/` removed. The local RTX 5090 was not touched today (the door gates on this
tree are owed there with the lock, as on days 18 to 22).
