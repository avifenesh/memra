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
