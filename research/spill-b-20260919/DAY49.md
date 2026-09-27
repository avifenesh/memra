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
  fix): arm i's terms of addendum B pass under the door, and the server log carries one `[kv-vmm] reap (batch-oom)`
  line per `retrying` line, printed before it (the reap runs inside the reclaim). Red (`02dbdfa40`): the same terms, and no `reap (batch-oom)` line (the
  defect as it reads). On the 5090 first; the target card in the next B sitting after the twelfth.

### 1.8 Addendum D (2026-09-26, after 2.1 and 2.2, before any code of the revision)

2.1 places the target card's FAIL lines on the fault's placement. The door's one budget spends itself at the first
step that reaches any of its three injection points. On the 27B that step is the spec route's solo step, and the one
batched fire held a single session. The revision aims the fault. No clause of 1.3 or addenda B and C is relaxed.

- **The door gains a site value:** `MEMRA_STEP_OOM_FAULT=batch:<n>` fires only at the plain dispatch's batched decode
  chunk, and only on a chunk carrying at least two sessions (O14's subject: one OOM ends every session of the chunk).
  With a site value the spec step and the non-batching step never consume the budget. A plain `<n>` is today's door,
  byte for byte. A malformed value stays OFF with the loud warning. The arming line names the site.
- **The gate's new arm `j`** (arm `i` stays as registered): `MEMRA_SERVE_SPEC=0` (the plain route, where the batched
  chunk is the decode dispatch), `MEMRA_BATCH_OOM_RECOVER=1`, three concurrent streams as arm i. `j-ctrl`: no fault,
  the control digests. `j`: `MEMRA_STEP_OOM_FAULT=batch:1`. Green: exactly one `fired: this batched decode chunk`
  line naming at least 2 sessions, one `retrying` line naming the same count as untouched, one `retried (ok)`, all
  three streams `200` with the control's digests, no 5xx, no panic. `j-red`: `batch:2`, so the retry is faulted too.
  The green assertion must fire (`retry failed`, and the chunk's sessions end with the error event).
- **The serving shape** runs again with `MEMRA_STEP_OOM_FAULT=batch:1`, arms `off` and `on` in both orders, on two
  routes: the spec default, and `MEMRA_SERVE_SPEC=0`. A boot counts only if its fired line names at least 2
  sessions. On `on` every request ends `200`; `off` is the before reading (the chunk's sessions end with the error
  event). If no fire reaches a batched chunk on the spec-default route, that reads as `not reached` for that route,
  with the boot's fired count quoted.
- **Addendum C's `i-vmm` becomes `j-vmm`:** arm j with `MEMRA_KV_ALLOCATOR=vmm`, on the revision (green) and on the
  revision with the `batch-oom` reap reverted by a lane patch (red; `02dbdfa40` predates the site value and cannot aim
  the fault). Green and red read as addendum C registered them.
- **Order:** the code and its CPU gates, then the 5090 (queue-l's items first), then the target card (BOX35, held).

### 1.9 Addendum E (2026-09-27, the gate's default arm list, after 2.1 to 2.5)

Arm i's verdict depends on where its unaimed fault lands. On the 27B that is a solo spec step (2.1, twice); on the 9B on
the 5090 it is a 3-session batched chunk (2.5). A gate arm whose verdict depends on placement does not belong in the
default list that local-ci and the integ batteries run. `tools/health-fault-gate.sh`'s default becomes
`a,b,c,d,e,f,g,h,j`. Arm i stays runnable by name, and its receipts stay as they read. Arm j (addendum D) is the aimed
form of the same check and passed four times on the target card (2.3). No clause changes.

### 1.10 Addendum F (2026-09-27, the default on the RTX PRO 6000 Blackwell class, the owner's ruling, before its code)

The owner's rulings of 2026-09-27, relayed by the lead: `MEMRA_BATCH_OOM_RECOVER` becomes the naked default on the RTX
PRO 6000 Blackwell class now, since the target card is read in full (2.3 and 2.4). It is keyed on the device class under
the per-hardware rule, with `=0` as its seam and a decide-by. The RTX 5090 gets its own flip once its serving boots
(queue-n, 2.6's five unrun boots) read.

- **The program:** unset on a card whose name `HardwareTarget::from_device_name` reads as the RTX PRO 6000 Blackwell
  class (every variant: the receipts are the Workstation Edition at 600 W, and the lead ruled the class) runs the
  recovery (1.2 and addenda A to D). `=0` turns it off on any card, the rollback seam, with `decide-by: 2026-10-11`
  for deleting the seam. `=1` turns it on on any card, which is how the 5090 runs it until its flip. Any other value
  keeps the card's default and says so on the boot line. Unset on every other card stays today's error arm.
- **The boot line:** `[batch-oom] recover=<ON|OFF> source=<pro6000-class-default|MEMRA_BATCH_OOM_RECOVER=<v>|no
  default on this card>`, once, where the worker reads the device name.
- **Why no new cell:** the default changes which arm runs, not the arm. Arm j and its red twin, j-vmm and the serving
  shape read on the target card with the door set (2.3 and 2.4), and the gate's arm j sets
  `MEMRA_BATCH_OOM_RECOVER=1` explicitly, so it reads the same under the new default. A unit test pins the decision
  per device name and value.
- **Records:** `docs/decisions/BATCH-OOM-RECOVER-DEFAULT.md` (what was chosen, the receipts, the 5090's pending
  flip), the FLAGS.md row, and the decisions index.

## 2. Results

Written after the runs. Section 1 is unchanged.

### 2.1 The target card (the twelfth sitting, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, 2026-09-26 17:57 to 18:01Z)

Tree `77fe12114`. The gate's release binary was built on the box from that tree, so it carries `95d35c383` (addendum C's
reap, unreachable here: no VMM door is set). The serving boots ran `bins/tip` from `02dbdfa40`, sha256
`bfb107ad...5f93cf5a0`. The 27B. Receipts at `pro-single-day49/box/` (the lead's `MIRROR-CHECK.txt`: box manifest OK,
no ELF). Note (2026-09-27, from the lead): the `MANIFEST.sha256` in this mirror is the lead's mirror-time manifest, not the sitting's own. The lead's earlier mirror script wrote its manifest over the sitting's, and the originals are gone (the box copies were destroyed or overwritten). Every mirrored file is verified against the lead's manifest, which hashed the box files at mirror time, so no receipt content is affected. The gate lines, verbatim (`g1`, then `g2`):

```
HFG (i) batch-oom-recovers: retry_lines=0 retried_ok=0 completed=3/3 error_events=0 http_5xx=0 panic_lines=0 digests={r0:9544bb8cfe5453f9,r1:d6443173460ba4da,r2:9c2aecc336dfecfc} control={r0:9544bb8cfe5453f9,r1:d6443173460ba4da,r2:9c2aecc336dfecfc} control_completed=3/3 -> FAIL
HFG (i-red) batch-oom-retry-also-fails: retry_lines=1 retry_failed=1 completed=2/3 error_events=1 http_5xx=0 panic_lines=0 green_assertion_fired=true -> PASS
health-fault-gate: arms=i pass=1 documented=0 fail=1 receipts=/root/spill-receipts/b-day49/g1
HFG (i) batch-oom-recovers: retry_lines=0 retried_ok=0 completed=3/3 error_events=0 http_5xx=0 panic_lines=0 digests={r0:9544bb8cfe5453f9,r1:d6443173460ba4da,r2:9c2aecc336dfecfc} control={r0:9544bb8cfe5453f9,r1:d6443173460ba4da,r2:9c2aecc336dfecfc} control_completed=3/3 -> FAIL
HFG (i-red) batch-oom-retry-also-fails: retry_lines=0 retry_failed=0 completed=3/3 error_events=0 http_5xx=0 panic_lines=0 green_assertion_fired=true -> FAIL
health-fault-gate: arms=i pass=0 documented=0 fail=2 receipts=/root/spill-receipts/b-day49/g2
```

- **Arm i FAIL on both runs, and i-red FAIL on `g2`, placed before any fix.** The recovery was never exercised. The
  door's budget is shared by three injection points and spends itself on the first step that reaches any of them. In
  `g1/i`, `g2/i` and `g2/i-red` that step was the spec route's solo step:
  `MEMRA_STEP_OOM_FAULT fired: this step reports a synthetic CUDA OOM (model hfg, generated 0, oom_retries 0/3)` with
  `1 active`, then the existing park and requeue, and every stream completed with the control's digests. Only
  `g1/i-red` reached the batched chunk:
  `MEMRA_STEP_OOM_FAULT fired: this batched decode chunk reports a synthetic CUDA OOM (1 session(s))`, then
  `batch OOM: 1 sessions untouched (Armed) ... retrying the same batched step once`, the second fire, and
  `batch OOM: retry failed (...); ended as today`. That chunk held one session.
- **The serving shape was not exercised either.** All four boots (`O1-off`, `O1-on`, `O2-on`, `O2-off`) fired on the
  spec route's solo step (`this step reports ..., generated 0`), parked, requeued and answered all 9 requests `200` on
  both arms. So `on`'s "every request ends 200" holds, but the fault never reached a batched chunk. `off` shows no
  before reading.
- **What this places:** on this card with the 27B and the spec route on, the first decode step of a concurrent start is
  a solo spec step. The cell as registered cannot aim the fault at a batched chunk carrying more than one session,
  which is O14's subject. Addendum D revises the fault's placement before any rerun.

### 2.2 Addendum C's i-vmm cell on the target card (the thirteenth sitting, the same card, 2026-09-26 18:11 to 18:12Z)

Tree `50bbd44d7`. Green built on the box from `95d35c383` (sha256 `bfa6f49d...ee258054d`), red from `02dbdfa40`
(`f4410984...2d33e80f`), each in its own worktree. Receipts at `pro-single-day49c/box/` (box manifest OK, no ELF). Note (2026-09-27, from the lead): the `MANIFEST.sha256` in this mirror is the lead's mirror-time manifest, not the sitting's own. The lead's earlier mirror script wrote its manifest over the sitting's, and the originals are gone (the box copies were destroyed or overwritten). Every mirrored file is verified against the lead's manifest, which hashed the box files at mirror time, so no receipt content is affected.
Verbatim (`read.log`):

```
DAY49C I-VMM card=pro6000 role=green gate_i=PASS gate_i_red=PASS i-ctrl=[door_on=1 reap_lines=0 retry_lines=0 paired=True] i=[door_on=1 reap_lines=1 retry_lines=1 paired=True] i-red=[door_on=1 reap_lines=1 retry_lines=1 paired=True] expect=one reap per retry, before it -> PASS
  green HFG (i) batch-oom-recovers: retry_lines=1 retried_ok=1 completed=3/3 error_events=0 http_5xx=0 panic_lines=0 digests={r0:9544bb8cfe5453f9,r1:d6443173460ba4da,r2:9c2aecc336dfecfc} control={r0:9544bb8cfe5453f9,r1:d6443173460ba4da,r2:9c2aecc336dfecfc} control_completed=3/3 -> PASS
  green i first reap line: [kv-vmm] reap (batch-oom) released=0 pending_graves=0
  green HFG (i-red) batch-oom-retry-also-fails: retry_lines=1 retry_failed=1 completed=2/3 error_events=1 http_5xx=0 panic_lines=0 green_assertion_fired=true -> PASS
  green i-red first reap line: [kv-vmm] reap (batch-oom) released=0 pending_graves=0
DAY49C I-VMM card=pro6000 role=red gate_i=FAIL gate_i_red=PASS i-ctrl=[door_on=1 reap_lines=0 retry_lines=0 paired=True] i=[door_on=1 reap_lines=0 retry_lines=0 paired=True] i-red=[door_on=1 reap_lines=0 retry_lines=1 paired=False] expect=no reap line (the defect as it reads) -> FAIL
  red HFG (i) batch-oom-recovers: retry_lines=0 retried_ok=0 completed=3/3 error_events=0 http_5xx=0 panic_lines=0 digests={r0:9544bb8cfe5453f9,r1:d6443173460ba4da,r2:9c2aecc336dfecfc} control={r0:9544bb8cfe5453f9,r1:d6443173460ba4da,r2:9c2aecc336dfecfc} control_completed=3/3 -> FAIL
  red HFG (i-red) batch-oom-retry-also-fails: retry_lines=1 retry_failed=1 completed=2/3 error_events=1 http_5xx=0 panic_lines=0 green_assertion_fired=true -> PASS
```

- **Green PASS.** Under `MEMRA_KV_ALLOCATOR=vmm` (`[kv-vmm] door=ON ... grow=inline (pro6000-ws-receipt)`), both
  green boots put the fault on a batched chunk (`1 session(s)`). Each printed `[kv-vmm] reap (batch-oom) released=0
  pending_graves=0` before its `retrying` line. Arm i recovered with the control's digests, and i-red's retry failed
  as its twin should. The reap released nothing: no parked cache existed at the fault.
- **Red reads FAIL as registered, on placement.** Red's arm i fault landed on the solo spec step (`this step reports
  ...`, then the step-OOM rung's own `reap (oom-teardown)`), so its gate terms fail for 2.1's reason. Red's i-red shows
  the defect itself: the batched chunk's fault, `retrying`, and no `reap (batch-oom)` line (`retry_lines=1
  reap_lines=0`).
- Addendum D's placement applies to this cell too.

### 2.3 Addendum D on the target card (the fourteenth sitting, the same card, 2026-09-26 20:58 to 20:59Z)

Tree `9e39bc314`. Green built on the box from `8926ccfb3`, sha256 `707c2ac5...4d1bfe63`. Red is the same commit with
`day49d-noreap.patch` (sha256 `5d3b7239...fb6671ad`), sha256 `4b8501b7...4c052d239`. Both built in worktrees. The 27B.
Receipts at `pro-single-day49d/box/` (the lead's `MIRROR-CHECK.txt`: box manifest OK, no ELF). Note (2026-09-27, from the lead): the `MANIFEST.sha256` in this mirror is the lead's mirror-time manifest, not the sitting's own. The lead's earlier mirror script wrote its manifest over the sitting's, and the originals are gone (the box copies were destroyed or overwritten). Every mirrored file is verified against the lead's manifest, which hashed the box files at mirror time, so no receipt content is affected. The gate lines read the
same on all four runs (`j1`, `j2`, `vmm-green`, `vmm-red`), verbatim once:

```
HFG (j) aimed-batch-oom-recovers: fired_lines=1 other_fired=0 chunk_sessions=3 untouched=3 retry_lines=1 retried_ok=1 completed=3/3 error_events=0 http_5xx=0 panic_lines=0 digests={r0:9544bb8cfe5453f9,r1:d6443173460ba4da,r2:9c2aecc336dfecfc} control={r0:9544bb8cfe5453f9,r1:d6443173460ba4da,r2:9c2aecc336dfecfc} control_completed=3/3 -> PASS
HFG (j-red) aimed-batch-oom-retry-also-fails: fired_lines=2 other_fired=0 chunk_sessions=3 retry_lines=1 retry_failed=1 completed=0/3 error_events=3 http_5xx=0 panic_lines=0 green_assertion_fired=true -> PASS
```

and `read-vmm.log`:

```
DAY49D J-VMM card=pro6000 role=green gate_j=PASS gate_j_red=PASS j-ctrl=[door_on=1 reap_lines=0 retry_lines=0 paired=True] j=[door_on=1 reap_lines=1 retry_lines=1 paired=True] j-red=[door_on=1 reap_lines=1 retry_lines=1 paired=True] expect=one reap per retry, before it -> PASS
DAY49D J-VMM card=pro6000 role=red gate_j=PASS gate_j_red=PASS j-ctrl=[door_on=1 reap_lines=0 retry_lines=0 paired=True] j=[door_on=1 reap_lines=0 retry_lines=1 paired=False] j-red=[door_on=1 reap_lines=0 retry_lines=1 paired=False] expect=no reap line (the defect as it reads) -> PASS
```

- **Arm j PASS on all four runs, and its red twin reads red on all four.** The aimed fault landed on a batched chunk
  of 3 sessions every time, and on nothing else (`other_fired=0`). The retry named the same 3 sessions untouched,
  `retried (ok)` followed, and all three streams completed with the no-fault control's digests. With `batch:2` the
  retry was faulted too: `retry failed`, and all 3 sessions ended with the error event (no 5xx, no panic).
- **O14 is read on the target card:** one OOM on a multi-session batched chunk no longer ends the chunk's sessions
  when `MEMRA_BATCH_OOM_RECOVER=1`. The chunk is run once more and its output equals the no-fault run's.
- **j-vmm PASS on both sides.** Under the VMM door the fix prints one `reap (batch-oom)` line per retry, before it.
  The patched red recovers the same way but prints no reap line: the defect addendum C found, as it reads.
- **The serving boots did not run.** `day49d-run.sh: line 18: cd: /root/projects/wt-spill-b: No such file or
  directory`, then `boots stopped rc=1`. The box chain did not export `WT`, so the runner fell back to its local
  default. The chain now exports `WT` and `RIG_LOCK` and has a `BOOTS_ONLY=1` entry point (`09b15280f`). The boots
  run under that name into `b-day49d-boots`. `read-serve.log` is empty and no serving clause is read here.

### 2.4 Addendum D's serving shape on the target card (the boots-only rerun, the same card, 2026-09-26 21:09 to 21:15Z)

Chain tree `09b15280f` (the sitting records HEAD `6cef1bbe3`). Green was rebuilt on the box from `8926ccfb3`, sha256
`f6252707...761dcf3a6`; the 27B. Burst 8 x 6,144, no warm, `MEMRA_STEP_OOM_FAULT=batch:1`. Receipts at
`pro-single-day49d/box-boots/` (the lead's `MIRROR-CHECK.txt`: box manifest OK, no ELF). Note (2026-09-27, from the lead): the `MANIFEST.sha256` in this mirror is the lead's mirror-time manifest, not the sitting's own. The lead's earlier mirror script wrote its manifest over the sitting's, and the originals are gone (the box copies were destroyed or overwritten). Every mirrored file is verified against the lead's manifest, which hashed the box files at mirror time, so no receipt content is affected. Verbatim (`read-serve.log`):

```
DAY49D SERVE card=pro6000 boot=O1-off arm=off route=spec-default batch_fired=[8] other_fired=0 retry_lines=0 retried_ok=0 retry_failed=0 rows=9 status={200: 1, 503: 8} error_rows=8 -> READING (the before)
DAY49D SERVE card=pro6000 boot=O1-on arm=on route=spec-default batch_fired=[8] other_fired=0 retry_lines=1 retried_ok=1 retry_failed=0 rows=9 status={200: 9} error_rows=0 -> PASS
DAY49D SERVE card=pro6000 boot=O2-off arm=off route=spec-default batch_fired=[8] other_fired=0 retry_lines=0 retried_ok=0 retry_failed=0 rows=9 status={200: 1, 503: 8} error_rows=8 -> READING (the before)
DAY49D SERVE card=pro6000 boot=O2-on arm=on route=spec-default batch_fired=[8] other_fired=0 retry_lines=1 retried_ok=1 retry_failed=0 rows=9 status={200: 9} error_rows=0 -> PASS
DAY49D SERVE card=pro6000 boot=P1-off arm=off-plain route=plain batch_fired=[5] other_fired=0 retry_lines=0 retried_ok=0 retry_failed=0 rows=9 status={200: 4, 503: 5} error_rows=5 -> READING (the before)
DAY49D SERVE card=pro6000 boot=P1-on arm=on-plain route=plain batch_fired=[5] other_fired=0 retry_lines=1 retried_ok=1 retry_failed=0 rows=9 status={200: 9} error_rows=0 -> PASS
DAY49D SERVE card=pro6000 boot=P2-off arm=off-plain route=plain batch_fired=[5] other_fired=0 retry_lines=0 retried_ok=0 retry_failed=0 rows=9 status={200: 4, 503: 5} error_rows=5 -> READING (the before)
DAY49D SERVE card=pro6000 boot=P2-on arm=on-plain route=plain batch_fired=[5] other_fired=0 retry_lines=1 retried_ok=1 retry_failed=0 rows=9 status={200: 9} error_rows=0 -> PASS
DAY49D SERVE-ROUTE card=pro6000 route=plain boots=4 reached=4
DAY49D SERVE-ROUTE card=pro6000 route=spec-default boots=4 reached=4
```

- **The serving clause PASS on every `on` boot, both routes, both orders:** 9 of 9 requests `200`, no error row, one
  retry and one `retried (ok)`. The aimed fault reached the batched chunk on every boot, and nowhere else.
- **The before (`off`):** one batched OOM ends every session of its chunk. On the spec-default route the chunk held 8
  sessions, and 8 of the 9 requests end `503` (1 of 9 `200`). On the plain route it held 5 sessions, and 5 of 9 end
  `503` (4 of 9 `200`). The same on both orders.
- With 2.3, O14 is read on the target card: the gates' 3-session chunk and the serving shape's 5- and 8-session chunks
  all recover under `MEMRA_BATCH_OOM_RECOVER=1`. Without it, every session in the faulted chunk is lost.

### 2.5 The 5090 (the 9B; queue-l, 2026-09-26 19:31 to 22:47Z)

- **Arm i on the registered source (`02dbdfa40`, `rtx5090-day49/g1`, `g2`), verbatim `g1` (`g2` reads the same):**
  ```
  HFG (i) batch-oom-recovers: retry_lines=1 retried_ok=1 completed=3/3 error_events=0 http_5xx=0 panic_lines=0 digests={r0:20e4033303b76715,r1:16e2547b5b1fafb5,r2:1d620d3efb9e2f90} control={r0:20e4033303b76715,r1:16e2547b5b1fafb5,r2:1d620d3efb9e2f90} control_completed=3/3 -> PASS
  HFG (i-red) batch-oom-retry-also-fails: retry_lines=1 retry_failed=1 completed=0/3 error_events=3 http_5xx=0 panic_lines=0 green_assertion_fired=true -> PASS
  ```
  On this card the unaimed fault landed on a batched chunk of 3 sessions in both runs (`this batched decode chunk
  reports a synthetic CUDA OOM (3 session(s))`). Arm i reads PASS as registered: the chunk ran again and matched the
  control, and the red twin ended all three streams.
- **The i-vmm pair (`rtx5090-day49c/`, `read.log` verbatim):**
  ```
  DAY49C I-VMM card=rtx5090 role=green gate_i=FAIL gate_i_red=PASS i-ctrl=[door_on=1 reap_lines=0 retry_lines=0 paired=True] i=[door_on=1 reap_lines=0 retry_lines=0 paired=True] i-red=[door_on=1 reap_lines=1 retry_lines=1 paired=True] expect=one reap per retry, before it -> FAIL
  DAY49C I-VMM card=rtx5090 role=red gate_i=PASS gate_i_red=PASS i-ctrl=[door_on=1 reap_lines=0 retry_lines=0 paired=True] i=[door_on=1 reap_lines=0 retry_lines=1 paired=False] i-red=[door_on=1 reap_lines=0 retry_lines=1 paired=False] expect=no reap line (the defect as it reads) -> PASS
  ```
  Green reads FAIL as registered, on placement: its arm i fault landed on the solo step, the reason 2.1 placed.
  Wherever green did retry, one reap line preceded the retry. Red reads its registered red: two retries and no reap
  line.
- **The unaimed serving shape did not complete.** `O1-off` ran (9 of 9 `200`; the fault parked on a solo step), and
  `O1-on` never started: `rig not idle after 7200 s; not run`. The `boots rc=0` that queue-l logged after it is `date`'s
  exit, as noted in queue-l's log. The unaimed shape is superseded by addendum D's aimed shape, which the 5090 half of
  DAY49D runs (`rtx5090-day49d/run.sh`).

### 2.6 Addendum D on the 5090 (the 9B, `rtx5090-day49d/`, 2026-09-26 23:50Z to 2026-09-27 03:42Z)

Green and red built in worktrees at `8926ccfb3` (red with `day49d-noreap.patch`); sha256s in `binaries.sha256`. The gate
lines read, on `j1` and `j2` alike:

```
HFG (j) aimed-batch-oom-recovers: fired_lines=1 other_fired=0 chunk_sessions=3 untouched=3 retry_lines=1 retried_ok=1 completed=3/3 error_events=0 http_5xx=0 panic_lines=0 ... -> PASS
HFG (j-red) aimed-batch-oom-retry-also-fails: fired_lines=2 other_fired=0 chunk_sessions=3 retry_lines=1 retry_failed=1 completed=0/3 error_events=3 http_5xx=0 panic_lines=0 green_assertion_fired=true -> PASS
```

and `read-vmm.log` reads the pair as registered: green one reap per retry, before it, `-> PASS`; red retries with no reap
line, `-> PASS` (the defect as it reads). The chunk held 2 or 3 sessions on this card.

- **Arm j and j-vmm PASS on the 5090 class,** as on the target card (2.3).
- **The serving shape is partly read.** Three of eight boots ran: `O1-off` (the before: 8 of 9 `503`), `O1-on` and
  `O2-on` (9 of 9 `200`, one retry each). Then `O2-off`'s idle wait ran out (`rig not idle after 7200 s; not run`),
  and the runner stopped there. queue-n asks again for the five unrun boots on green rebuilt at the same commit; that
  reading follows.

### 2.7 Addendum D's serving shape on the 5090, complete (the 9B, `rtx5090-day49d/`, 2026-09-27)

2.6 read three of the eight boots. queue-n ran the other five on green, rebuilt at the same commit `8926ccfb3` (sha256
`a16045ad...` against the first build's `634f9730...`; same source, the build path differs). Burst 8 x 6,144,
`MEMRA_STEP_OOM_FAULT=batch:1`. Verbatim (`read-serve.log`):

```
DAY49D SERVE card=rtx5090 boot=O1-off arm=off route=spec-default batch_fired=[8] other_fired=0 retry_lines=0 retried_ok=0 retry_failed=0 rows=9 status={200: 1, 503: 8} error_rows=8 -> READING (the before)
DAY49D SERVE card=rtx5090 boot=O1-on arm=on route=spec-default batch_fired=[8] other_fired=0 retry_lines=1 retried_ok=1 retry_failed=0 rows=9 status={200: 9} error_rows=0 -> PASS
DAY49D SERVE card=rtx5090 boot=O2-off arm=off route=spec-default batch_fired=[8] other_fired=0 retry_lines=0 retried_ok=0 retry_failed=0 rows=9 status={200: 1, 503: 8} error_rows=8 -> READING (the before)
DAY49D SERVE card=rtx5090 boot=O2-on arm=on route=spec-default batch_fired=[2] other_fired=0 retry_lines=1 retried_ok=1 retry_failed=0 rows=9 status={200: 9} error_rows=0 -> PASS
DAY49D SERVE card=rtx5090 boot=P1-off arm=off-plain route=plain batch_fired=[2] other_fired=0 retry_lines=0 retried_ok=0 retry_failed=0 rows=9 status={200: 7, 503: 2} error_rows=2 -> READING (the before)
DAY49D SERVE card=rtx5090 boot=P1-on arm=on-plain route=plain batch_fired=[8] other_fired=0 retry_lines=1 retried_ok=1 retry_failed=0 rows=9 status={200: 9} error_rows=0 -> PASS
DAY49D SERVE card=rtx5090 boot=P2-off arm=off-plain route=plain batch_fired=[8] other_fired=0 retry_lines=0 retried_ok=0 retry_failed=0 rows=9 status={200: 1, 503: 8} error_rows=8 -> READING (the before)
DAY49D SERVE card=rtx5090 boot=P2-on arm=on-plain route=plain batch_fired=[8] other_fired=0 retry_lines=1 retried_ok=1 retry_failed=0 rows=9 status={200: 9} error_rows=0 -> PASS
DAY49D SERVE-ROUTE card=rtx5090 route=plain boots=4 reached=4
DAY49D SERVE-ROUTE card=rtx5090 route=spec-default boots=4 reached=4
```

- **The serving clause PASS on every `on` boot, both routes, both orders:** 9 of 9 `200`, one retry and one
  `retried (ok)`. The aimed fault reached the batched chunk on every boot and nowhere else. The chunk held 8 sessions,
  or 2 when the burst's arrivals split across two waves (O2-on, P1-off).
- **The before (`off`):** the chunk's sessions end `503`, 8 of 9 (or 2 of 9 on the 2-session chunk).
- **O14 is read on the 5090 as well:** the gates (2.6) and the full serving shape. The owner's ruling put the 5090's own
  flip on these boots (addendum F), so the flip is registered as its own addendum before its code.

