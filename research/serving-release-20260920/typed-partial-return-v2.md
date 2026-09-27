# Typed partial prime returns

Lifecycle v2 distinguishes a successful quantum from an intentional partial
return. The producer emits `memra-request-lifecycle-v2` on every record. The
parser retains the exact v1 contract for historical evidence and refuses a
version change within one `(pid, trace_id)` history. Historical parse support
does not transfer evidence to a rebuilt source or binary; the outer release
binding must pin the producer protocol too.

Each non-null v2 `quantum` adds `cancelled_at_boundary`, normally null. Only an
exact `progress::PrimeCancelled` downcast in `prime_observation::observe_call`
emits `prime_quantum_cancelled`, with:

```json
{
  "completed": false,
  "remaining_chunks": null,
  "cancelled_at_boundary": {"chunk": 0, "rows_done": 32, "rows_total": 64}
}
```

`rows_total` must equal the open quantum's input rows, and
`0 <= chunk < rows_done < rows_total`. The chunk index is zero-based. The
producer stamps its own end time and changes phase to `prime_cancelled`;
it never reports `prime_finished` or a known remaining chunk count. These
counts describe the selected call, not total request work or a GPU fence.
No scheduling, numerical operation or error result is changed.

The original `mark_prime_quantum_start` / `mark_prime_quantum_end` APIs retain
their meanings. An ordinary error, including one whose text or nested source
resembles `PrimeCancelled`, still records `failed_prime_quantum` and refuses
target qualification. Saved walker advances/finalization retain their existing
success/error treatment; the new typed seam covers the direct call only.

A v2 partial return can supply **prime cancellation facts** only when the raw
history proves:

1. Complete immutable request/worker bindings and a nonzero open quantum.
2. Exactly one HTTP pending/body drop during that quantum, without normal EOF.
3. The matching typed partial return, followed by observed `receiver_dropped`.
4. Explicit aborted retirement at `ActiveSession`, then final `trace_end`.

New work, decode, another HTTP transition, ordinary failures, requeue, invalid
snapshots or suppression refuse. `EventSender::is_closed` can also mean queue
overflow: typed `PrimeCancelled` alone is not proof of client cancellation, and
an `event_queue_overflow` close cause still refuses C4. Trace destruction does
not invent resource retirement.

The 8192 ordinary-event limit is unchanged. The cancelled return replaces one
ordinary end event; it does not add a second return or coalesce records. The
CPU source-module and Python controls exercise this protocol, including the
original partial-return regression. Full wire/peer/recovery, source/build/model,
controller, physical lease and native scenario validation remain separate.
