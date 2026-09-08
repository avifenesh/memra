# MTP prime adapter

GPU qualification pending explicit handoff. Default remains OFF.

`spec/prime.rs` owns the frozen segment/range program, trunk cursor, full hidden
stack, capture positions, and subsequent draft-fill cursor. A temporary
`MtpPrimeWalker` binds the state and its `SpecSession` to the execution context.
The supported adapter shape is a single-device GDN MTP plan; other prime routes
keep their existing path. This is an implementation boundary, not qualification
of additional models or hardware.

Trunk operations preserve segment-local offsets, the absolute request end,
GDN range alignment and the existing tokenwise program for sub-floor segments.
Snapshots remain at the same stable boundaries. Draft fill preserves predecessor
pairing, including the previous turn's hidden anchor at the suffix's first row.
The short-prompt quantum completes its existing boundary segments and one draft
fill so a one-chunk cold peer can emit its first token in that service turn.
Long primes return between trunk chunks and between draft-fill chunks.

Draft graph setup was extracted from the existing generator. It runs once before
fill, and the prepared context survives yields. `PreparedMtp` retains boundary
logits, the boundary token and counters, the already-executed init feed, and its
seed/prediction buffers. The consumer checks prompt, sampler, K and constraint
identity, then consumes these fields once. It neither samples/feeds the boundary
twice nor recaptures graphs over already-filled draft KV. Grammar advances once.
Prime timing accumulates executed work, excluding scheduler pauses.

Supported MTP prefix restoration materializes the paid carrier during admission
and queues its suffix without drawing a premature boundary. Both serving arms
then prime through the same walker. The compatibility restoration API retains
its synchronous behavior for other callers. Cached-token accounting, context
capacity and request output budgets stay intact.

CPU schedule tests cover distinct capture stops, non-divisible tokenwise tails,
and the legacy tokenwise override's segment behavior. See `CPU.md` for the check
set. GPU c1/c2 bytes and boundary/capture equivalence remain mandatory before a
latency or exactness verdict.
