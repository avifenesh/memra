# WP-B day 25: memra#524 as a serving-shape gate (`tools/health-fault-gate.sh`, arms a to f) and `phase=warming`

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`. Day start: `git merge --no-ff origin/main 81d75c457`
(#615, the unknown or retired `MEMRA_*` boot refusal; `research/INDEX.md` conflicted on the shared marker, both rows kept,
`tools/check-conflict-markers.sh` `OK`; merge `ace752157`), pushed. Every push of the day whose range touches engine
source is refused `UNQUALIFIED` by the #589 hook on a plain push and runs as `MEMRA_RELEASE_QUALIFICATION_MODE=development
git push` (`UNQUALIFIED DEVELOPMENT`, logged in `.git/memra-gate-skips.log`). **No qualification is claimed anywhere in
this record**; every GPU cell is `executed-not-qualified`; N=1 per arm per run; no number here is a timing claim and no
timing is compared across cards.

Rig discipline as on every prior day: every GPU command on the local RTX 5090 Laptop GPU runs under `flock -w 1800`
`/tmp/memra-5090.lock` (`run-day25-gate.sh`; the lock was free at every acquisition today), CPU-heavy work under
`systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`, `nvidia-smi --query-compute-apps` read before and
after every boot (all empty: no co-tenant on the card during any cell), and no process this lane did not start is touched
(the gate signals only a `memra-server` whose pid it spawned AND whose cwd is this tree; a lane-A `driver.sh` and a
`cargo test` were visible in the process table and were left alone).

Reading order followed: `STATE.md`, `DAY24.md` section 5, integ27 in the lead's `INTEGRATION-DAY12.md` (read from
`lane/spill-integ27-20260922`, PR #619 still open at day start; revuto's round moved only the `RequestCharge` wording:
the eager arm books its draft state, conservative), memra#524 and #526 in full.

## 1. What the code says (read at `ace752157`, before any change)

- Routes: `/health` == `/livez` (liveness) and `/readyz` (readiness); **there is no `/healthz` route**
  (`lib.rs` route table, `docs/SERVING.md` "The full route table"). The task's `/healthz` is read as `/health`.
- Phases: `PHASE_LOADING=0`, `IDLE=1`, `BUSY=2`, `DEAD=3` (`health.rs`); `/health` and `/readyz` bodies carry
  `worker.phase` through `phase_name`, so readiness DOES report a phase and task 2 is implementable, not a proposal.
- Order inside `worker::run`: model load, `run_boot_calibration(...)`, `ready_tx.send(Ok(...))`, `health.mark_ready()`.
  `worker::spawn` blocks on `ready_rx`, and `serve_with` binds the listener only after `spawn` returns
  (`lib.rs` `TcpListener::bind(bind_addr)` follows `worker::spawn`). So on a FIRST boot the probe window has no socket at
  all: a readiness probe reads connection-refused, exactly as the `loading` row of SERVING.md's phase table says. The
  window IS reachable over HTTP during a respawn, because the supervisor re-runs `run` (load, probe, ready) under a live
  listener with `mark_respawning` having set `PHASE_LOADING`.
- Fault doors that already exist, each with its `docs/FLAGS.md` row: `MEMRA_PANIC_AFTER=<n>` (one-shot worker panic
  on the n-th completed request; the G5 ladder `catch_unwind` -> `mark_dead` -> `/health` 503 with the quoted payload ->
  respawn -> exit 70; `WORKER_RESPAWN_MAX=1`, `MEMRA_WORKER_RESPAWN`), the gpu-watch canary with `MEMRA_GPU_WATCH_S`
  and `MEMRA_GPU_PROBE_TIMEOUT_S` (a steady-state probe that outlives the deadline latches `mark_gpu_fault`; the latch
  is never cleared: `live()` reads `gpu_faulted` first), `MEMRA_DRAIN_S` (SIGTERM: `DRAINING` flag, `/readyz` 503
  `draining` with `Retry-After`, `/health` 200 `draining`, new completions 503 `code: draining`, in-flight streams
  finish, exit 0). `MEMRA_STEP_OOM_FAULT` and `MEMRA_KV_HOST_FAULT` exist too and were not needed today.
- **No arm needed a new fault hook.** Arm (e)'s injection is a `PATH`-shadowed `nvidia-smi` in the server's environment
  only (a file, not a door); everything else is an existing door or a signal. Nothing was pre-registered as a missing
  door and no serving path gained a hook.

## 2. `phase=warming` (task 2): landed, CPU-tested, observed on the wire

`crates/memra-server/src/health.rs`: `PHASE_WARMING = 4`, `phase_name` renders `warming`, `live()` answers
`Err("worker is warming: boot calibration probe in flight (readiness follows its completion)")`, and
`mark_warming()` moves LOADING -> WARMING through `compare_exchange` so a late or repeated call can never demote a
serving worker (IDLE and BUSY stay put). `mark_ready()` (IDLE) ends it; `mark_respawning()` (LOADING) restarts the
order on a respawn. `crates/memra-server/src/worker.rs`: `run_boot_calibration` takes `&SharedHealth` and calls
`health.mark_warming()` after every skip `return` (`MEMRA_ADMIT_CALIBRATE=0`, `!serve_spec_enabled()`,
`MEMRA_ADMIT_RESERVE_MB`) and after the per-model `continue` (a model with no spec route), immediately before the
probe's own `t0`: the phase is entered only when a probe actually runs, so a probe-skipped boot never says `warming`
and is not made to look warm. No new numeric program, no flag, no env read.

CPU tests: `health::tests::warming_sits_between_loading_and_ready` (LOADING -> WARMING -> IDLE; warming is neither
live nor ready and names itself; `mark_warming` from IDLE and from BUSY is a no-op; the respawn walks LOADING ->
WARMING -> IDLE with generation 1) and the day-24 source-order test `worker::tests::readiness_follows_the_boot_calibration_probe`
extended: exactly one `health.mark_warming();` inside `run_boot_calibration`, after its last `return;` and before
`dspark_spec_session_burst(`. `cargo test --release -p memra-server --lib -- health::tests worker::tests::readiness_follows_the_boot_calibration_probe`:
`test result: ok. 18 passed; 0 failed` (`rtx5090-day25/build/cpu-tests-health-worker.log`).

Docs: SERVING.md phase table gains the `warming` row; FLAGS.md `MEMRA_ADMIT_CALIBRATE=0` row names the phase and the
three skipped paths; `MEMRA_PANIC_AFTER` row points at arm d.

On the wire (run1 and run2, arm d's respawn window sampled at 100 ms): `not_ready_phase_sequence=dead>loading>warming
loading_samples=7 warming_samples=8 loading_then_warming=true`; run1 timeline from the trigger: `3951 ms dead`,
`5756 ms loading`, `6824 ms warming`, `8028 ms ready/idle`. The first boot's probe window stays connection-refused
(31 samples of `000`, section 3 arm a), as the code order predicts.

## 3. `tools/health-fault-gate.sh` (task 1): arms, assertions, verdicts

Shape: boots the real `memra-server` (`MEMRA_COMPAT=openai`, the Qwen3.5-9B NVFP4 MTP GGUF serve-smoke uses, port
8186 behind `tools/port-guard.sh` pre-flight and `memra_port_owned` post-boot), one boot per arm (a and b share one),
polls `/readyz` at 100 ms from launch and records every sample (`t_ms,code,status,phase,detail`), asserts HTTP codes
AND body fields (`status`, `worker.phase`, `detail`, `choices.0.finish_reason`, `usage.completion_tokens`,
`worker.generation`, `error.code`, `Retry-After`) plus the server log's line order, and writes one verbatim verdict
line per arm to `VERDICTS.txt`. Exit 0 with no FAIL (DOCUMENTED arms do not fail the gate), 1 on any FAIL, 2 on setup.
Rig lock: not taken by the gate (serve-smoke's convention; `local-ci.sh` holds the lock for the run; the collector
holds it on a PRO box). The gate never signals a process it did not spawn. Mapping to the DAY24 section 5
pre-registration: today's (a)/(b) are the readiness half of DAY24's (a) and (f), (c) is DAY24's (f) made explicit,
(d) is DAY24's (b) with the real panic door, (e) is DAY24's (a) turned into the latch question, (f) is DAY24's (e).
DAY24's (c) (step OOM under a tiny headroom, `MEMRA_STEP_OOM_FAULT`) and (d) (client disconnect mid-stream) were not in
the lead's day-25 list and remain pre-registered only.

Harness attempts, kept verbatim as labelled failures (no assertion changed between them and the green runs):

- `rtx5090-day25/attempt1-harness-clock-bug/`: every boot read `FATAL: not ready within 420s` in about 10 ms, all nine
  lines `boot failed -> FAIL`. Cause, quoted: this rig's `date` is `date (uutils coreutils) 0.8.0` and `date +%s%3N`
  prints `1790031363654135641` (nanoseconds), so the deadline arithmetic expired at once. Fix: `now_ms` reads
  `$EPOCHREALTIME` (bash 5). Server logs are empty (SIGTERM before the first line); the servers were never at fault.
- `rtx5090-day25/attempt2-harness-jget-bug/`: the server passed every arm; five lines read FAIL because `jget` indexed
  `choices.0` with a string key on a list and returned `finish_reason=` empty while the body carried
  `"finish_reason":"length"`, `"completion_tokens":32`. Fix: list indices as integers. The four lines that did not
  depend on `jget` (a, a2, e, f) were already PASS with the same values the green runs show.

**run1** (`rtx5090-day25/run1/`, binary `run1/gate/binary.sha256`, tree `ace752157` plus the day's uncommitted change;
committed as `1b18164d8`), verbatim:

```
HFG (a) readiness-before-probe: probe_done_line=30 listening_line=33 pre_ready_samples=31 pre_ready_codes={000:31,503:0} ready_samples_not_ready=0 ready_while_warming=0 first_ready_phase=idle -> PASS
HFG (b) ready-then-first-request: readyz_before=200/ready request_http=200 finish_reason=length completion_tokens=32 request_ms=181 ready_to_completion_ms=291 readyz_after=200/ready (N=1, not a timing claim) -> PASS
HFG (c) probe-skipped c-calibrate0 [MEMRA_ADMIT_CALIBRATE=0]: skip_line=16 probe_done_line=0 warming_samples=0 ready_ms=2441 first_request_http=200 first_request_ms=235 (N=1) -> DOCUMENTED (ready without warmup; the first request pays the cold route)
HFG (c) probe-skipped c-spec0 [MEMRA_SERVE_SPEC=0]: skip_line=16 probe_done_line=0 warming_samples=0 ready_ms=2333 first_request_http=200 first_request_ms=333 (N=1) -> DOCUMENTED (ready without warmup; the first request pays the cold route)
HFG (c) probe-skipped c-reserve [MEMRA_ADMIT_RESERVE_MB=1536]: skip_line=17 probe_done_line=0 warming_samples=0 ready_ms=2328 first_request_http=200 first_request_ms=229 (N=1) -> DOCUMENTED (ready without warmup; the first request pays the cold route)
HFG (d) panic-respawn-truthful-health: trigger_http=200 health_503_quoted_after_ms=198 readyz_while_dead=503/not_ready/dead untruthful_samples=0 recovered_after_ms=4367 generation_after=1 request_after_http=200 health_after=200/ok log_panic_line=44 log_respawn_line=45 -> PASS
HFG (a2) warming-phase-on-respawn: not_ready_phase_sequence=dead>loading>warming loading_samples=7 warming_samples=8 loading_then_warming=true -> PASS
HFG (e) fatal-fault-not-cleared-by-timeout: watch_on_line=5 control_200_samples=3/3 latched_after_ms=3047 shim_answering_again_for_ms=8185 health_after=503/unhealthy readyz_after=503/not_ready health_200_after_clear=0 critical_lines=1 request_http_while_latched=200 (observed, bounded, not asserted) -> PASS
HFG (f) sigterm-drains-and-flips-readiness-first: stream_frames_at_sigterm=15 readyz_during_drain=503/not_ready retry_after_s=60 health_during_drain=200/draining new_request_http=503 new_request_code=draining new_request_retry_after_s=60 stream_curl_rc=0 stream_frames=258 stream_done=1 stream_finish_reason=length stream_finished_after_sigterm_ms=1248 exit_code=0 exit_after_sigterm_ms=1495 drain_complete_line=48 deadline_hit_line=0 -> PASS
health-fault-gate: arms=a,b,c,d,e,f pass=6 documented=3 fail=0
```

**run2** (`rtx5090-day25/run2/`, same binary), verbatim where a value moved: `(b) ... request_ms=183
ready_to_completion_ms=300`, `(c) c-calibrate0 ... ready_ms=2218 ... first_request_ms=229`, `c-spec0 ... ready_ms=2439
... first_request_ms=325`, `c-reserve ... ready_ms=2224 ... first_request_ms=229`, `(d) ...
health_503_quoted_after_ms=195 ... recovered_after_ms=4320`, `(e) ... shim_answering_again_for_ms=8188`, `(f) ...
stream_finished_after_sigterm_ms=1250 exit_after_sigterm_ms=1511`; every other field identical to run1;
`health-fault-gate: arms=a,b,c,d,e,f pass=6 documented=3 fail=0`. Regime: laptop GPU, 59 to 61 C, 23 W idle to
142 W at the end of a run, power limit reported `[N/A]`, driver 595.84, no co-tenant on the card.

What each arm establishes, in plain words:

- **(a)** On the armed first boot the `[admit-cal] boot calibration done:` line (log line 30; probe `1.2s`, `transient
  floor 1536MB ... measured 1266MB; probe kv charge 67MB, draft-state 41MB, drafted 48 accepted 47`) precedes
  `[server] listening on` (line 33); all 31 readiness samples before the first 200 were connection-refused; the first
  200 carried `status=ready phase=idle`; no 200 ever carried a non-ready status or the `warming` phase.
- **(b)** `/readyz` 200 `ready` before and after the first request, which completed 200 with `finish_reason=length`
  and 32 completion tokens, 181 ms request time, 291 ms from the first 200 to completion (N=1).
- **(c)** The three probe-skipped boots each printed their documented skip line (`boot calibration disarmed
  (MEMRA_ADMIT_CALIBRATE=0)`, `boot calibration skipped: spec serving disabled`, `boot calibration skipped:
  MEMRA_ADMIT_RESERVE_MB teeth door`), never printed a `boot calibration done` line, never reported `warming`, were
  ready about 2.2 to 2.4 s after launch, and served the first request 200. Recorded as DOCUMENTED: readiness there
  means "weights resident, route cold"; the doors' FLAGS rows now say so. Not a pass and not relaxed.
- **(d)** The `MEMRA_PANIC_AFTER=1` request itself returned 200 (the panic fires after the completion counts);
  `/health` read 503 with `detail` containing `worker thread panicked` 198 ms later; `/readyz` read `503/not_ready/dead`;
  across every `/health` and `/readyz` sample of the arm, zero were untruthful (200 with a status other than
  `ok`/`ready`, or non-200 with `ok`/`ready`); the respawn recovered in 4.37 s with `worker.generation=1`; a request
  then completed 200; `/health` `200/ok`; log order `[worker] PANIC in the GPU worker thread` (line 44) then
  `[worker] respawn attempt 1/1` (line 45).
- **(e)** `[gpu-watch] on: every 2s, probe deadline 2s` (line 5); three control `/health` samples 200 over 6 s with the
  shim answering; one hung probe latched `503/unhealthy` with `detail` `nvidia-smi did not answer within 2s` 3.05 s
  after the flag file appeared; the shim answered again for 8.2 s (more than three intervals) and `/health` stayed
  `503/unhealthy` with the same detail, `/readyz` `503/not_ready`, zero 200 samples after the clear, exactly one
  `[gpu-watch] CRITICAL` line. The request path served 200 while latched (observed and recorded, not asserted: the
  latch is a supervisor signal and the worker is alive; whether a latched process should refuse work is a product
  question the owner decides). Clean SIGTERM exit 0.
- **(f)** SIGTERM with a stream 15 frames in: `/readyz` `503/not_ready` with `detail` `draining` and `Retry-After: 60`
  (`MEMRA_DRAIN_S=60` for the arm), `/health` `200/draining`, a new request 503 with `error.code=draining` and
  `Retry-After: 60`; the stream ran to 258 frames including `data: [DONE]` with `finish_reason=length` 1.25 s after
  the signal; exit 0 1.5 s after the signal; `[server] drain complete in` (line 48) and no deadline line. This is the
  re-runnable receipt SERVING.md's graceful-drain bullet lacked (#526 ask 4); the bullet now names the gate.

**Wired into `tools/local-ci.sh`** after serve-smoke, `MEMRA_CI_HEALTH_FAULT=0` skips (FLAGS row added). Why: two
consecutive runs on this rig read identically in every asserted field (only the recorded millisecond values moved,
and none is asserted); the arms are boots and injected faults against fixed doors, not timings, so nothing in them
depends on load; the whole gate is about 2 min on the 9B. The condition the lead set ("every implemented arm is
deterministic on this rig") is met on the evidence of two runs; a third run on the target card is section 5.

## 4. What is not claimed

- N=1 per arm per run; the millisecond fields are records, not medians, and no threshold is set on them.
- `executed-not-qualified` everywhere; the #589 hook's `UNQUALIFIED DEVELOPMENT` announcement stands on every push.
- The request path under a latched gpu fault (arm e) is recorded, not judged.
- DAY24's (c) step-OOM arm and (d) client-disconnect arm remain pre-registered only (not in today's list; both have
  their doors: `MEMRA_STEP_OOM_FAULT`, and a client close, so neither needs a new hook).
- `phase=warming` is observable over HTTP only on the respawn window; a first boot binds the listener after the probe,
  by design (SERVING.md `loading` row). Exposing the probe window on a first boot would mean binding before the load,
  a serving-order change the owner decides; not done.

## 5. Target card (one RTX PRO 6000 Blackwell, 96 GB, 600 W)

The box was free at first check (lane A's cargo test was in the process table and untouched; the gpu lock read free).
`/root/wt-b` at `1b18164d8` (the gate commit; the local-ci wiring commit `8e1fcd3b9` changes no engine file), built
there (`pro-single-day25/build/build.log`, binary `ba1b4d27...` in `build/binary.sha256`; a first attempt through a
non-login shell found no `cargo` and built nothing, quoted `bash: line 1: cargo: command not found`, rerun under
`bash -lc`), then one collector cell: `python3 tools/tier-battery.py --rig pro-single --timeout 1500 --out .../cell
--execute env HFG_OUT=.../gate tools/health-fault-gate.sh <the day-23 staged 9B>`; `lock.json`
`{"rig": "pro-single", "lock": "/tmp/memra-gpu.lock", "acquired": true}`, `CELL.jsonl` `"status":
"executed-not-qualified"`, `"qualification": false`, `elapsed_seconds 34.9`, exit 0. Regime: `NVIDIA RTX PRO 6000
Blackwell Server Edition`, driver 580.178.04, power limit 600 W, 32 C / 33 W before to 39 C / 238 W after, no
co-tenant on the card before or after (`compute-apps-*.csv` empty). Receipts `research/spill-b-20260919/pro-single-day25/`
(pulled over the lead's socket; grep for the box's hostname, any non-loopback IP, provider or ssh tokens: 0 hits;
`provider_instance_id: null`). Verbatim:

```
HFG (a) readiness-before-probe: probe_done_line=30 listening_line=33 pre_ready_samples=17 pre_ready_codes={000:17,503:0} ready_samples_not_ready=0 ready_while_warming=0 first_ready_phase=idle -> PASS
HFG (b) ready-then-first-request: readyz_before=200/ready request_http=200 finish_reason=length completion_tokens=32 request_ms=143 ready_to_completion_ms=229 readyz_after=200/ready (N=1, not a timing claim) -> PASS
HFG (c) probe-skipped c-calibrate0 [MEMRA_ADMIT_CALIBRATE=0]: skip_line=16 probe_done_line=0 warming_samples=0 ready_ms=1095 first_request_http=200 first_request_ms=166 (N=1) -> DOCUMENTED (ready without warmup; the first request pays the cold route)
HFG (c) probe-skipped c-spec0 [MEMRA_SERVE_SPEC=0]: skip_line=16 probe_done_line=0 warming_samples=0 ready_ms=1102 first_request_http=200 first_request_ms=232 (N=1) -> DOCUMENTED (ready without warmup; the first request pays the cold route)
HFG (c) probe-skipped c-reserve [MEMRA_ADMIT_RESERVE_MB=1536]: skip_line=17 probe_done_line=0 warming_samples=0 ready_ms=1102 first_request_http=200 first_request_ms=166 (N=1) -> DOCUMENTED (ready without warmup; the first request pays the cold route)
HFG (d) panic-respawn-truthful-health: trigger_http=200 health_503_quoted_after_ms=192 readyz_while_dead=503/not_ready/dead untruthful_samples=0 recovered_after_ms=3469 generation_after=1 request_after_http=200 health_after=200/ok log_panic_line=44 log_respawn_line=45 -> PASS
HFG (a2) warming-phase-on-respawn: not_ready_phase_sequence=dead>loading>warming loading_samples=5 warming_samples=4 loading_then_warming=true -> PASS
HFG (e) fatal-fault-not-cleared-by-timeout: watch_on_line=5 control_200_samples=3/3 latched_after_ms=2727 shim_answering_again_for_ms=8157 health_after=503/unhealthy readyz_after=503/not_ready health_200_after_clear=0 critical_lines=1 request_http_while_latched=200 (observed, bounded, not asserted) -> PASS
HFG (f) sigterm-drains-and-flips-readiness-first: stream_frames_at_sigterm=5 readyz_during_drain=503/not_ready retry_after_s=60 health_during_drain=200/draining new_request_http=503 new_request_code=draining new_request_retry_after_s=60 stream_curl_rc=0 stream_frames=258 stream_done=1 stream_finish_reason=length stream_finished_after_sigterm_ms=886 exit_code=0 exit_after_sigterm_ms=1074 drain_complete_line=48 deadline_hit_line=0 -> PASS
health-fault-gate: arms=a,b,c,d,e,f pass=6 documented=3 fail=0
```

Same verdict in every asserted field as the two local runs; the log line numbers (30/33, 16/17, 44/45, 5, 48) are
identical on both cards, which is the source-order reading of section 1 seen from the log. The sample counts differ
(17 pre-ready samples against 31, 5 loading and 4 warming against 7 and 8, 5 stream frames at SIGTERM against 15)
because the card class loads and primes faster; no cross-card ratio is drawn from them.

## 6. Checks before the final push

`cargo fmt --all -- --check` clean; `cargo clippy --release -p memra-server --all-targets -- -D warnings` `Finished`
(`rtx5090-day25/build/clippy-memra-server.log`); `git diff --check` clean; `tools/check-flags.sh` `every runtime MEMRA_*
name resolves against 'docs/FLAGS.md' (no grandfather list)`; `tools/check-conflict-markers.sh` `OK`;
`python3 tools/check-public-boundary.py check` `582 matches (582 grandfathered, 0 new)`.

## 7. Budget

Agent time against the 4-hour budget: about 2 h 35 min end to end (reading 35 min, code and gate 55 min, three local
runs and two harness fixes 30 min, target card 20 min, write-up and checks 15 min).
