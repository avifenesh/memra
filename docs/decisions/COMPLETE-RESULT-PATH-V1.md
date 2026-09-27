# Complete-result path for long non-streaming requests (2026-09-28)

**Status:** interface frozen, design only. Nothing in this document is built, wired, or
gated. Issue: `avifenesh/memra#550`. Scope: a supported way for a request whose valid
workload cannot finish inside the synchronous non-streaming deadline to receive its
complete output without holding one HTTP connection open for the entire run.

## Problem, grounded in the current code

- The non-stream deadline race lives in `blocking_response_with_receipt`
  (`crates/memra-server/src/lib.rs:11312`-`11428`). A miss with zero tokens produced
  answers `408 deadline_exceeded`, zero-billed (`lib.rs:11352`-`11369`). A miss with some
  tokens produced delivers what exists and bills it under the `deadline_partial` outcome
  (`lib.rs:11378`-`11427`), documented in `docs/SERVING.md:1054`-`1057` and
  `docs/SERVING.md:1109`-`1111`.
- The 90 s non-streaming ceiling (`docs/SERVING.md:1035`-`1044`, `TIMEOUT_MS_DEFAULT` at
  `lib.rs:2319`) is deliberate margin under a fronting proxy's response timeout
  (`docs/SERVING.md:1062`-`1063`). It is honest, and this document does not propose
  raising it for the blocking mode.
- Streaming already removes the runtime bound: once the first token is produced,
  "the parameter is spent; the stream runs to completion" (`docs/SERVING.md:1079`). The
  partial-delivery error message on the non-stream path already tells the caller this: it
  advises setting `"stream": true` for work this long (`lib.rs:11404`-`11405`).
- So the gap issue #550 names is not "no way to receive a long complete result." It is
  "no way to receive one without a single live connection held for the whole run." Any
  caller whose own infrastructure cannot hold a socket open for that duration (a
  serverless invocation with its own short deadline, a batch submission tool, a client
  that backgrounds) has nothing today, streaming included.
- This is exactly the gap OpenAI's `background: true` on `/v1/responses` addresses
  upstream, and memra already recognizes the field name: it is one of the explicit
  SUBSET LAW refusals, `"background responses are not supported"`
  (`crates/memra-server/src/responses_api.rs:152`-`153`), grouped together with
  `previous_response_id`, `store: true`, and `conversation`.
- That grouping conflates two different kinds of state. `previous_response_id`, `store`,
  and `conversation` are CROSS-REQUEST state: they ask the server to remember a prior
  turn so a later, different request can reference it, and the surface is documented as
  stateless translation with no such memory (`responses_api.rs:1`-`21`). `background` is
  SINGLE-REQUEST state: the server already owns this one request's input and is already
  generating its output; the only new requirement is holding that one request's own
  result somewhere other than an open socket until it is collected. It asks the server to
  remember nothing about any other request. Splitting this refusal into two is the core
  move of this design.

## Contract (frozen shape, not yet built)

1. **Opt-in per request, not a global mode.** On any generation surface, `background:
   true` selects this delivery mode. `stream: true` and `background: true` together is a
   400: background exists so the caller does not have to hold a connection open, and
   streaming is the mode that keeps one open, so combining them is a contradiction, not a
   feature.
2. **Admission is unchanged.** Budget, capacity, and deadline-feasibility gates run
   exactly as they do for a synchronous request. `timeout_ms` stops bounding total wall
   time and instead becomes the polled status object's own field (see 3): background is a
   delivery-mode switch on the existing pipeline, not a second pipeline.
3. **Immediate response.** Once the worker is admitted, the handler returns the same
   Responses-vocabulary envelope already implemented (`responses_api.rs:484`-`536`) with
   `status: "queued"` and no `output`, plus the request id. No new object shape on the
   Responses surface.
4. **Poll.** `GET /v1/responses/{id}` extends the existing envelope: its `status` enum
   already spans `in_progress`, `completed`, `incomplete`, and `failed`
   (`responses_api.rs:484`-`490`) and only needs `queued` added. Once status is terminal,
   the response carries the same `output` a synchronous call would have produced.
   `/v1/chat/completions` and `/v1/completions` have no polling-by-id convention today;
   add `GET /v1/jobs/{id}` returning the chat-shaped completion object under the same
   states, so the chat dialects gain the capability without inventing a second envelope
   family.
5. **Cancel.** `POST /v1/responses/{id}/cancel` (chat dialects: `POST
   /v1/jobs/{id}/cancel`) drops the worker's event channel. On the chat/completions
   dialect this reuses the drop-and-deliver sequence that already exists for a deadline
   miss in `blocking_response_with_receipt` (`drop(rx)` at `lib.rs:11351`, then
   `complete_deadline_partial` / `settle_unbilled` at `lib.rs:11356`-`11394`): zero tokens
   produced bills zero, some tokens produced bills what was delivered. On `/v1/responses`
   this is **not** a reuse: that surface's own non-streaming deadline path
   (`responses_api.rs:705`-`753`) always drops `rx` and settles `deadline_exceeded`
   unconditionally today, discarding any partial output by deliberate design
   (`responses_api.rs:708`-`715`, "revisit if a caller asks for partials here
   specifically"). Restoring `background` on `/v1/responses` and then wanting a cancel
   that bills a partial on that same surface is exactly a caller asking for that; it
   requires porting the chat dialect's partial-delivery/billing behavior onto
   `/v1/responses`, which is new work this document is naming, not code already in
   place. A new named outcome, `cancelled`, distinguishes an explicit client cancel from
   a deadline miss in the census on both dialects, mirroring the existing
   `deadline_exceeded` / `deadline_partial` split (`docs/SERVING.md:1109`-`1116`).
6. **Terminal usage.** Satisfied by the existing receipt discipline, not a new webhook.
   `complete`, `complete_deadline_partial`, and `settle_unbilled` already settle exactly
   once, "independently of body polling, before cancelling the worker"
   (`docs/SERVING.md:1088`-`1090`). Background mode inherits that unchanged: the ledger
   row closes the moment generation ends or is cancelled, whether or not the caller ever
   calls `GET`. A caller who never polls does not change what was billed; it only leaves
   a result uncollected.
7. **Uncollected results.** A finished job's buffered output lives in a bounded, TTL'd,
   in-process store. A caller who never polls loses the text after the TTL, the same way
   a caller who disconnects mid-stream today loses nothing it was not already billed for.
   The TTL and the store's memory ceiling are deployment decisions, not part of this
   contract (see "Owed" below).

## Where the new state lives

The one genuinely new piece is holding a request's buffered output between the worker
finishing and the caller's `GET`. This mirrors the existing pluggable-seam pattern
(`metering::Metering` / `metering::Receipt`, `crates/memra-server/src/metering.rs:137`)
rather than hardwiring storage into the handler:

- Add `trait JobStore: Send + Sync` alongside `Receipt`, with `put(id, JobRecord)`,
  `get(id) -> Option<JobRecord>` (read without consuming, for repeated polling before
  terminal state), `take(id) -> Option<JobRecord>` (collect once terminal), `cancel(id)`,
  and a TTL sweep. The stock server ships one in-memory reference implementation (bounded
  map, default TTL, default max resident bytes, both deployment-tunable), the same shape
  as the stock no-ledger `Metering` reference today.
- A deployment that wants a job to survive a process restart, or a store shared across
  replicas behind a router, supplies its own `JobStore`, the same way it supplies its own
  `Metering`. The stock server makes no promise beyond one process's own lifetime. This
  is consistent with the existing statelessness claim in `docs/API-SURFACES.md` and
  `responses_api.rs:1`-`21`: nothing here persists a CONVERSATION. It buffers one
  in-flight request's own output, and only until collected or expired.

## What this explicitly does not change

- `previous_response_id`, `store: true`, `conversation`, `item_reference` input items,
  and `truncation: "auto"` stay refused (`responses_api.rs:129`-`163`). `background`
  restores none of them.
- The synchronous (non-background) deadline behavior, its billing outcomes, and the 90 s
  ceiling are unchanged (`docs/SERVING.md:1030`-`1131`).
- Streaming is unaffected. It remains the answer for a caller that can hold the
  connection open.

## Acceptance mapping (issue #550)

- "Define a supported complete-result contract... with capability and deadline semantics
  visible to the caller": this document.
- "Preserve the full accepted input and output state; continuation must not silently
  truncate context or regenerate an unrelated partial answer": satisfied by
  construction. The buffered `JobRecord` is the same accumulator
  (`text`, `reasoning`, `tokens`, `calls`) `blocking_response_with_receipt` already builds
  (`lib.rs:11322`-`11336`); a `GET` returns that accumulator, never a re-summary or a
  fresh generation.
- "Test a legitimately over-synchronous-budget request through the actual endpoint,
  including status/result retrieval or resume, cancellation, and terminal usage
  callbacks": **not satisfied yet.** This needs the routes, the `JobStore` trait and its
  reference implementation, worker-cancellation wiring, and a real over-90-second
  generation exercised against a running `memra-server` on a GPU box. `memra-server`
  compiles CUDA fatbins and is not buildable in a CPU-only lane; GitHub CI builds it but
  does not run a live server end to end either.
- "Keep partial/deadline failures explicit, and document deployment ownership of
  pricing/accounting policy": addressed by points 5-7 above. A `docs/FLAGS.md` row is
  owed once a flag exists; none is added by this document.

## Owed (why issue #550 stays open)

1. Implementation: the `JobStore` trait plus its in-memory reference implementation, the
   `GET`/`POST` routes on both dialects, and splitting the `background` gate in
   `translate()` (`responses_api.rs:152`-`153`) so it restores this one field while
   `previous_response_id`, `store`, and `conversation` keep refusing. Cancel wiring
   splits by dialect: chat/completions reuses the existing `drop(rx)` +
   `complete_deadline_partial` sequence (`lib.rs:11351`-`11394`); `/v1/responses` needs
   the partial-delivery/billing behavior that surface deliberately does not have today
   (`responses_api.rs:705`-`715`) ported onto it, which is new engine work, not reuse.
2. A default-OFF flag (for example `MEMRA_BACKGROUND_RESPONSES`) with its
   `docs/FLAGS.md` row (default, both arms, rollback seam, receipt pointer, decide-by
   date) once the code exists. Not created by this document.
3. A box run: a real generation submitted with `background: true` that legitimately
   exceeds 90 s, polled through to completion; a second one cancelled mid-generation; and
   a terminal usage row inspected in the ledger for both. Requires a GPU box; not
   runnable in this CPU-only lane.
4. An owner decision on the in-memory `JobStore`'s default TTL and default max resident
   bytes. The memory cost of buffering uncollected output is a capacity decision this
   document does not invent a number for.
