# WP-B day 33: memra#680, the door's burst admission past its own estimate

DAY32.md 2.5 found it and the lead filed it as memra#680. With `MEMRA_ADMIT_BY_MEMORY=1` at open output 32768, a
64-request burst reached 58 in flight on one RTX PRO 6000, and 46 (O1) and 47 (O2) of them died in prefill with
`CUDA_ERROR_OUT_OF_MEMORY`, returned as 503. The local RTX 5090 shows the same shape at a smaller scale: 21 in
flight and 2 prefill OOMs per order. This day is authorized to change engine and server code under the default-OFF
door only. Door-OFF behaviour stays byte-identical. Every cell is `executed-not-qualified`; no default moves.

## 1. Pre-registration

Committed and pushed before the first boot of this day. Nothing in section 1 changes after a number is seen.

### 1.1 Diagnosis (from code at main `c3eb41d12` and the day-32 receipts, before any run)

Line numbers are `crates/memra-server/src/worker.rs` at `c3eb41d12` unless named otherwise.

1. **Which path admitted the 58 without an `[admit-mem]` decision line.** The generic VRAM gate. Each arrival's
   `required = admission_required(cost, reserve)` (23548) is compared against one measured headroom reading
   (`admission_headroom`, 23596; `AdmissionHeadroom::sufficient`, 27939: `free_bytes >= required`). When it fits,
   the request falls through to admission with no door involvement. The door runs only inside the defer branch
   (`if !headroom.sufficient(required)`, 24024; `if admit_memory_cfg.armed`, 24161), and it prints only on a first
   defer or a refusal (`if refusing || waited_ms == 0`, 24211). `admit_memory::decide` has a `MemoryVerdict::Admit`
   arm (`admit_memory.rs:262`), but no admitted request ever reaches it. The day-32 box log shows it: admissions
   every 2 to 4 ms (`pro-single-day32/box/boots/O1-on32768/server.log` lines 6600 to 6739, `active=12` up to 58).
   The receipts-only predictive shadow turns to `verdict=reject-kv` from line 6662 (`booked_bytes=65667440384`
   against its 65,881,157,328-byte budget) and refuses nothing, by design.
2. **Whether the door's device-free reading counts resident prefix cache as free.** No. `device_free` is
   `headroom.limiting_free_bytes()` (24164), which is driver free plus pool-cached (`effective_free_bytes`,
   27840). Resident prefix entries are live allocations, so they count in neither term; they appear separately
   as `demotable` (`px.evictable_bytes()`). The first defer flushed them: line 6702, `evicted_prefix_bytes=
   13288108032`, "effective free 3662MB -> 16949MB" (line 6703). What the reading misses is the opposite: memory
   that admitted sessions will still allocate. After admission each primed session seeds a prefix entry
   (`[prefix-cache] insert (seed): 1344 tokens, 196.8MB`, lines 6776 to 6784, nine of them before the first OOM at
   6785). On the 5090 there were 18 `lcp-split` inserts of 72.2 MB before its OOM
   (`rtx5090-day32/boots/O1-on32768/server.log` up to line 8607).
3. **Whether the charge uses the learned fixed residual.** Yes. `cost = model.estimate(admission_cap,
   prompt_len, estimate_spec)` (23163) is context KV + `prefill_workspace_bytes(prompt_rows)` + `activation_bytes`
   (`AdmissionCostModel::estimate`, 2739). `activation_bytes` is the learned residual (`observe`, 2763: the
   admit-time effective-free delta minus the context allocation, high-water only, applied at 24478). The line
   reads `29696 B/token x ctx + 658MB prefill-workspace + 380MB fixed = 2054MB`, plus the 2301 MB reserve.
   The struct's own comment says why that is not enough (2598 to 2603): `observe` "measures the admit-time
   free-VRAM delta; the workspace grows at PRIME time". The gate books context and residual as they appear in
   the live reading. But the prefill workspace of every admitted, still-priming session has not been allocated
   when the next arrival is measured, so a burst never sees it. Over one tick 58 arrivals each saw
   `free >= 4.35 GB` while 58 x 658 MB of workspace, plus the seed inserts, was still owed.
4. **Why a prefill OOM reaches the client as 503 instead of a defer before prefill.** Admission is the only defer
   point, and it had already passed. The prefill arms send the error terminally:
   `Event::Error(EngineError::engine(format!("prefill error: {err}")))` at 25757 (the interactive prefill loop)
   and 26223 (the dark chunk). `EngineError::engine` promotes the driver's OOM text to `ErrClass::Overloaded`
   (1539 to 1541, `is_cuda_oom`), which is 503 with Retry-After 5. The step-OOM park, which requeues a session
   that has emitted nothing (`step_oom_parkable`, 28120; used at 25253), covers the decode step only.
5. **Why refusals waited 15.6 s against an 8 s budget.** One tick ran from 135.334 s (line 6741, the first
   `[admit-oom] VRAM defer`) to 150.980 s (line 6758). The six refusals are the first decisions the next tick
   made. The budget is checked per tick, so a long tick overshoots it.

### 1.2 The fix under test (door-armed only)

- **Pending-prime booking.** With the door armed, every headroom reading in the admission block is reduced by
  `pending_prime`: the sum, over active sessions that have not finished their prompt prime, of
  `prefill_workspace_bytes(rows still queued)` from the session's own cost model. Unarmed, `pending_prime` is 0
  and the reduction is the identity.
- **One decision line per door admission.** Every arrival the armed gate admits prints
  `[admit-mem] id=... verdict=admit ...` through `memory_line`, with `device_free` = the booked reading (measured
  minus `pending_prime`) and a new field `pending_prime=<bytes>`. The same field joins the defer and refuse lines.
  Unarmed, no `[admit-mem] id=` line is printed, as today.
- **Prefill-OOM park.** With the door armed, a prefill CUDA OOM on a session that has emitted nothing
  (`step_oom_parkable`) parks and requeues like the step-OOM park, so a residual OOM reaches the door's defer and
  refusal instead of the client. Unarmed, both prefill arms are unchanged.
- A CPU unit test per decision arm (1.7).

### 1.3 Shapes, arms, binaries

Workload `day33-client.py` (1.4) on the day-31 server environment (`run-day26-cell.sh`, `CLIENT=day33-client.py`,
`PARSER=day31-parse.py`, `N=5`, `MEMRA_TIMEOUT_MS_MAX=3600000`, `MEMRA_ADMIT_PREDICT_SHADOW=1`,
`MEMRA_COMPAT=openai`; local `MEMRA_CTX=65536`, target card unset).

| shape | door env | client args | card |
|---|---|---|---|
| `G1` | `MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS=2048` | `--chars 5000 --skip-long --burst 64` | 5090 |
| `G2` | `... =8192` | `--chars 5000 --skip-long --burst 64` | 5090 |
| `R32` | `... =32768` | `--chars 5000 --skip-long --burst 32` (day 32's local burst) | 5090 |
| `R64` | `... =32768` | `--chars 5000 --skip-long --burst 64` (day 32's target burst) | PRO 6000 |
| `off` | door variables removed | `--chars 5000 --skip-long --burst 0` | both |

Binaries, one per role per card: `red` = main `c3eb41d12`, built in a detached worktree of that commit; `green` =
the lane commit that carries the fix, built the same way. Each boot records `binary.sha256` and `source.txt`.

Order, local RTX 5090, under `/tmp/memra-5090.lock` (`run-day26-cell.sh` takes it per boot, after a bounded idle
wait: lock free, no compute app, at least 24 GB of host memory):
`red-G1`, `red-G2`, `red-R32`, `red-off`; then, after the fix is built, `green-<gate>-r1`, `green-R32`,
`green-off`, `green-<gate>-r2` (if the gate shape is `R32`, its second run is `green-R32-r2`).

Gate shape: the first of `G1`, `G2`, `R32` whose red boot reads R-OOM RED. If none does, no local gate is wired,
and section 2 says so.

Target card, under `/tmp/memra-gpu.lock` on the box, only after lane A's `LANE-A-PRO-DONE` line:
`red-R64`, `red-off`, `green-R64`, `green-off`.

The red binary is `7497071ead123e999b4a7ea8141e24a6106dde5932b4e12638fb2bed9136b7b3` (`rtx5090-day33/build-red/`).

The gate script (`tools/admit-mem-burst-gate.sh`, the gate shape's burst only): one run on `red` must FAIL, then
two consecutive runs on `green` must PASS before it is wired into `tools/local-ci.sh`. If the two green runs do not
both pass, it is not wired and section 2 says why.

### 1.4 Workload

`day33-client.py` is `day31-client.py` with two opt-in flags (`--skip-long` drops class iv, `--burst 0` drops the
burst); without them it runs day 31's workload byte for byte. Here: warmup, then (i), (ii) and (iii) at L0 (5000
chars), 5 reps each, then the burst of B open (i)-shape requests at 5000 chars, released on one barrier (DAY31.md
1.4). Prompts from `docs/SERVING.md` sha256 `022b1d50620f5b34bb6e6127be05ff79184ce3aa294aca618a31a0dd73887cff`.

### 1.5 Readings

`day33-compare.py --card <label> <role>:<shape>:<boot-dir> ...` over the boots of one card, writing that card's
`SUMMARY.txt`. `day31-faults.py` lists every panic, respawn, FATAL and engine-error line into `FAULTS.txt`. The reader
was checked on the day-32 receipts before any day-33 boot: `red:R32:rtx5090-day32/boots/O1-on32768` reads
`R-OOM ... oom_lines=2 ... -> RED`, and the target card's `O1-on32768` reads `oom_lines=46 ... -> RED`.

### 1.6 Verdicts (fixed now)

- **V-BOOT.** The day-31 parser's door-line check, per boot.
- **R-OOM** (per ON boot, a reading): RED iff the boot's `server.log` has any `CUDA_ERROR_OUT_OF_MEMORY` line or
  its burst has any 503. The acceptance needs RED on every `red` boot of the gate shape and of `R64`.
- **G-NOOM** (per ON boot, judged on `green`): PASS iff zero `CUDA_ERROR_OUT_OF_MEMORY` lines, zero burst 503s,
  zero crash-pattern lines (`panicked`, `[worker] PANIC`, `[worker] FATAL`, `[worker] respawn`, `argmax sentinel`,
  `spec verify refused`), at least one burst 200, no burst status other than 200 or 429, every 429 carrying
  `Retry-After` in 1..=60, and the number of 429s equal to the burst window's `verdict=refuse` lines.
- **G-BOOK** (per ON boot, judged on `green`): PASS iff there is at least one `[admit-mem] ... verdict=admit`
  line, every such line has `est_bytes <= device_free` and a `pending_prime=` field, and the burst window holds at
  least as many admit lines as burst 200s. This is the "in-flight never past the door's budget" term: every
  admission fits the booked reading. A `red` boot has no admit line and reads FAIL, as expected.
- **V-ID-FIX** (per shape, judged): `green` ON against `red` ON of the same shape. Every sequential row where both
  returned 200 on an equal `prompt_sha256` has equal `message_sha256`, G and `finish_reason`, and there is at least
  one such row. The fix moves no token.
- **V-ID** (the day-32 term, judged): `green` ON against `green` OFF. Eligible are warmup, (ii), and open rows whose
  OFF side finished `stop` with `G_off < v`. PASS iff at least one is eligible and all are equal.
- **V-OFF** (judged): `green` OFF against `red` OFF. Every sequential row is equal (status 200, same prompt,
  message, G, finish) with the same row count, the green OFF boot's log has zero `[admit-mem] id=` lines, and its
  V-BOOT passes. This is the door-OFF byte-identity term.
- **Card verdict.** GREEN iff every V-BOOT passes, every `green` ON boot passes G-NOOM and G-BOOK, and V-ID-FIX,
  V-ID and V-OFF all pass. Otherwise NOT-GREEN, naming the failing lines.

### 1.7 Unit tests (CPU, `cargo test -p memra-server`)

One per decision arm, named in section 2 with their result: the booked reading admits when `need <= measured -
pending`; it defers when `measured >= need` but `measured - pending < need` (the memra#680 shape); it refuses after
the budget with the pending term; a zero pending term is the identity (the door-OFF byte-identity); a session's
pending term is its queued-row workspace while priming and 0 once primed or empty; the admit arm renders
`verdict=admit` with `pending_prime=`; the prefill-OOM park is taken only armed, only on a CUDA OOM, and only for a
session that has emitted nothing.

### 1.8 Failures

Causes are quoted from captured stderr, never inferred. A rerun happens only as a whole boot under a new name, with
the reason in section 2.

## 2. Results

Written after the runs. Section 1 is unchanged.
