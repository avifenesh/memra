# WP-B day 49: O14, a batched decode chunk's OOM recovers its sessions instead of ending them

OWED.md O14, from DAY47 2.1: the step-OOM door on three concurrent streamed requests fired on the batched decode chunk,
and the chunk's error arm ends every session of the chunk with the typed overloaded error. The non-batching step has a
park branch (a session that emitted nothing parks and requeues); the batched chunk has none, and a session that has
streamed tokens cannot park. DAY24 (c)'s acceptance says an OOM at a step leaves peers' streams complete. This day
pre-registers a recovery behind a default-OFF door. No default moves.

## 1. Pre-registration (text only until DAY47 addendum A's reruns read)

Committed and pushed before any day-49 code and before any day-49 cell. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code.

### 1.1 What makes a retry safe (the one-numeric-program argument)

A batched decode step appends each session's new K/V row at `len` and advances `len`, and its linear-attention
layers write the next recurrent state into the ping-pong spare before swapping. A step that fails before any layer
has advanced leaves every session's cache byte for byte as it was (rows past `len` are not state; the unswapped spare is
not state), so running the SAME batched step again is the same program on the same inputs: one numeric program per
request. A step that fails after some layer advanced leaves torn state, and no retry is safe for those sessions.

### 1.2 The arm (`MEMRA_BATCH_OOM_RECOVER`, default unset; decide-by 14 days after its code lands)

Unset: today's error arm. `1`, on a batched decode chunk's error whose text is a quoted CUDA OOM (the same
`is_cuda_oom` match the step-OOM ladder uses):

- **Torn-state check:** every session of the chunk compares its per-layer committed markers (full-attention `len`,
  linear-attention ping-pong parity, `pos`) with the snapshot of those markers taken before the step (a host-side
  copy of integers, no device work).
- **Untouched chunk:** run the step-OOM reclaim ladder once (the prefix cache, parked sessions, the driver trim, as the
  non-batching path does), then run the same batched step once more with the same inputs. A second OOM ends the chunk
  as today (the bounded retry: one).
- **Torn chunk, or not an OOM:** today's error arm (the typed overloaded error per session); a session that emitted
  nothing parks as today.
- **Receipts:** `[admit-oom] batch OOM: <n> sessions untouched; reclaimed <MB>; retried (<ok|failed>)` or `... torn
  (<layers advanced>); ended as today`.

### 1.3 Cells

- **The gate:** `tools/health-fault-gate.sh` arm g-batch (DAY47 addendum A) with the door set: the fault (before any
  device work) makes the chunk untouched, so all three streams must complete with `200`, a `finish_reason` and
  `[DONE]`, digests equal to a no-fault boot's; the DOCUMENTED line of today (unset) stays the before receipt.
- **A red twin:** a new fault position inside the step (`MEMRA_STEP_OOM_FAULT_AT=after-layer-<k>`, a diagnostic door
  that forges the OOM after layer k advanced) must read torn and end as today, with no retry.
- **Serving shape:** DAY38's X shape or DAY44's RX at B = 8 concurrent with the fault, on both cards: no 5xx, no crash,
  every stream completes or ends typed.

### 1.4 Price

Code: about 1 agent-day (the markers snapshot and check, the retry, the inside-step fault door, census and unit
tests). Cells: the gate on each card plus one serving boot pair.

## 2. Results

Written after the runs. Section 1 is unchanged.
