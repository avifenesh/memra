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

### 1.5 Addendum A (2026-09-26, while reading the batched step before any code)

1.2's torn-state check on host-side markers is unsound: the batched step's linear-attention conv ring is updated IN
PLACE by `ssm_conv1d_fused_decode_b` (the pointer table carries each session's `conv_state`), so a step that ran one
layer's conv and then failed changes no marker. The check is replaced by an engine-side guard that knows where state is
written, and the reclaim is named exactly:

- **The step guard (`memra_engine::step_guard`, a thread-local state per call):** the worker arms it before the batched
  call (`Armed`); the batched entry marks `Entered` on arrival; the generic unsplit body marks `Generic` just before its
  pre-layer setup (the pointer table, the embed gather); `decode_batch_layers` marks `Touched` immediately before every
  state-writing statement (the KV append and its `len` advance, the conv ring update, the GDN scan and its ping-pong
  swap). The epilogue runs after the layers, so it is already `Touched`. A census test pins that every state-writing
  call in `decode_batch_layers` is preceded by the mark.
- **Recoverable iff the guard reads `Armed` (the failure came before the engine call: the fault door's injection point)
  or `Generic` (the generic body failed before any state write).** `Entered` (a non-generic batched program: hyper, PP,
  step35, gemma, the B=1 fast path) and `Touched` are not recoverable: today's error arm.
- **The reclaim, once:** the parked sessions of the three pools are dropped, the device prefix cache is evicted to half
  its bytes, the teardown fence runs and the model pools are trimmed back to the driver (the step-OOM ladder's first
  rung, with its own receipt line `[admit-oom] batch OOM: ...`). Then the same batched call runs once more; a second
  failure is today's error arm.
- **The red twin** of 1.3 needs no new fault door: `MEMRA_STEP_OOM_FAULT=2` fires on the first attempt and again on the
  retry, so the recovery is attempted and the chunk ends as today; the `after-layer-<k>` door of 1.3 is dropped (a
  torn chunk is not retried by construction, which the census and a unit test on the guard's rule cover).
- Everything else of section 1 stands.

### 1.6 Addendum B (2026-09-26, the cells as built, before any cell)

- **Code:** `6102fb63a` (the step guard and the worker's one retry; census tests for both; memra-engine 580 and
  memra-server 958 passed, clippy clean), the FLAGS row fixed to its two-column table in `02dbdfa40` (the boot audit
  reads the table, so the cells build `02dbdfa40`).
- **The gate:** arm `i` in `tools/health-fault-gate.sh`: `i-ctrl` (the door on, no fault), `i` (`MEMRA_STEP_OOM_FAULT=1`:
  one `retrying` line, one `retried (ok)` line, all three streams complete with the control's digests, no 5xx, no
  panic) and `i-red` (`=2`: the retry fails too, `retry failed`, and the green assertion fires).
- **The serving shape** is `day45-client.py`'s burst of 8 prompts of 6,144 tokens, `max_tokens=64`, with no warm
  requests (`--warm-n 0`, so the fault lands on the burst's decode), `MEMRA_STEP_OOM_FAULT=1`, arms `off` and `on`
  (`MEMRA_BATCH_OOM_RECOVER=1`) in both orders, on both cards (the 27B on the target card): on `on` every request
  ends `200`; `off` is the before reading. No clause of 1.3 changes.

### 1.7 Addendum C (2026-09-26, the review-pattern reading of the built arm, before any fix code)

The seven spill review patterns read against `6102fb63a` (the retry loop, `batch_oom_reclaim`, the guard):

- **Found: the reclaim skips the VMM reap.** The two reclaim paths that drop the parked pools (the step-OOM teardown
  and the admin trim) call `vmm_reap_for` before `trim_model_device_pools` (DAY37 addendum D: under
  `--kv-allocator vmm` a dropped cache's on-demand planes sit in the graveyard until reaped), and a census test pins
  both. `batch_oom_reclaim` drops the same pools and trims without the reap, so under the VMM door the retry would run
  against a driver that has not got the dropped planes back. The fix: `vmm_reap_for("batch-oom")` before the trim, and
  the census pins the third site.
- **Read and clean:** no move-then-match (the error is borrowed in the guard); nothing parks, so the idle wait is not
  touched; the gate derives its counts from the fault count; the retry-failure exit holds nothing the reclaim took (the
  guard is taken after every attempt); the drop, fence, settle and trim order is the step-OOM rung's; no booking and no
  reply are involved.
- **The registered cells stand on `02dbdfa40`.** `vmm_reap_for` returns at `!kv_vmm::armed()`, and no DAY49 cell sets
  the VMM door, so the fix is unreachable in them: the local runner and the twelfth sitting run as registered.
- **The new cell `i-vmm` (the door combination, the fix's own reading):** the gate's arms `i-ctrl`, `i`, `i-red` with
  `MEMRA_KV_ALLOCATOR=vmm` exported (the gate's server inherits it), on the fix binary and on `02dbdfa40`. Green (the
  fix): arm i's terms of addendum B pass under the door, and each retry is followed by one `[kv-vmm] reap (batch-oom)`
  line before its `retrying` line's retry. Red (`02dbdfa40`): the same terms, and no `reap (batch-oom)` line (the
  defect as it reads). On the 5090 first; the target card in the next B sitting after the twelfth.

## 2. Results

Written after the runs. Section 1 is unchanged.
