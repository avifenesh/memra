#!/usr/bin/env bash
# tools/health-fault-gate.sh: serving-shape health, readiness and lifecycle gate (memra#524; the
# fault/lifecycle half of memra#526). Boots the real memra-server on the real model, injects the
# faults the server already has doors for, and asserts HTTP codes plus body fields on `/readyz`,
# `/health` (== `/livez`; there is no `/healthz` route) and the request path. Every verdict is one
# line, verbatim, in `$OUT/VERDICTS.txt`; every arm keeps its server log, its readiness samples
# and `nvidia-smi --query-compute-apps` before and after.
#
# Arms (pre-registered research/spill-b-20260919/DAY24.md section 5, restated by the lead for
# day 25):
#   a  readiness before the boot calibration probe completes reads not-ready. On a FIRST boot the
#      listener binds after the probe (docs/SERVING.md, phase table), so every pre-ready sample is
#      connection-refused (code 000); the arm asserts the log order (`[admit-cal] boot calibration
#      done:` before `[server] listening on`) and that no sample ever read 200 before that order
#      held. The probe window is observable over HTTP during a RESPAWN: arm d records the phase
#      sequence there (verdict line `a2`, `loading` then `warming`, memra#524 phase=warming).
#   b  readiness after the probe reads ready and the first request completes (200, finish_reason,
#      completion_tokens > 0), `/readyz` 200 before and after.
#   c  the probe-skipped boots (MEMRA_ADMIT_CALIBRATE=0, MEMRA_SERVE_SPEC=0, MEMRA_ADMIT_RESERVE_MB)
#      state what readiness means there: ready WITHOUT warmup, never `warming`. Recorded as the
#      DOCUMENTED behaviour of those doors, not as a pass.
#   d  a request that panics the worker (MEMRA_PANIC_AFTER=1, the one-shot fault door with its
#      FLAGS row; MEMRA_WORKER_RESPAWN=1) leaves `/health` truthful: 503 with the quoted payload
#      within seconds, never 200 while the worker is dead or reloading, 200 with generation 1 after
#      the respawn, and a request completes on the respawned worker.
#   e  a fatal fault of the gpu-watch class (memra#516) is not cleared by a timeout-only recovery:
#      a PATH-shadowed `nvidia-smi` (server process only) hangs ONE steady-state probe past
#      MEMRA_GPU_PROBE_TIMEOUT_S, the latch fires (`/health` 503, `[gpu-watch] CRITICAL`), the shim
#      answers again, and three probe intervals later the latch still holds. No new door: the shim
#      is the injection and MEMRA_GPU_WATCH_S / MEMRA_GPU_PROBE_TIMEOUT_S are documented knobs.
#   f  a graceful shutdown flips readiness first and drains: SIGTERM with a stream open ->
#      `/readyz` 503 `draining` + Retry-After, `/health` 200 `draining`, a new request 503 with
#      `code: draining`, the stream runs to `[DONE]` with a finish_reason, exit 0, `drain complete`.
#   g  a step OOM (MEMRA_STEP_OOM_FAULT=1, the synthetic-OOM door) on one non-streamed request's own
#      non-batching step parks the session back to the queue and it completes; no 5xx (WP-B DAY47
#      1.1, addendum A). Red twin g-red: MEMRA_STEP_OOM_FAULT=4 (one past the default retry budget)
#      walks it into the bounded-retry honest error, and the green assertion must fire there.
#      g-batch (DOCUMENTED): three concurrent streams share the batched decode chunk the fault lands
#      on; the chunk's error arm ends every one of them (DAY47 2.1, owed O14).
#   h  a client that closes its stream mid-generation is retired within 1,000 ms (`[abort] client
#      disconnected:`), its peer completes, and the box idles clean (DAY47 1.2). Red twin h-red: the
#      same shape with no close, and the green assertion must fire there.
#
# Usage: tools/health-fault-gate.sh [model.gguf]
#   HFG_PORT (8189; 8186 is serve-gemma4-batch-gate.sh's, revuto on #621; census of tools/ before choosing a default), HFG_ARMS (a,b,c,d,e,f,g,h; a and b share one boot), HFG_OUT (receipt dir).
#   Run under the rig lock (`flock /tmp/memra-5090.lock`, or the collector on a PRO box); the
#   gate boots seven servers in sequence and never takes the lock itself, like serve-smoke.
#   Exit 0 when no arm FAILED (DOCUMENTED arms do not fail the gate); 1 on any FAIL; 2 on setup.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

MODEL="${1:-/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf}"
[ -f "$MODEL" ] || { echo "health-fault-gate: SKIP (no model at $MODEL)"; exit 0; }
PORT="${HFG_PORT:-8189}"
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
ARMS="${HFG_ARMS:-a,b,c,d,e,f,g,h}"
OUT="${HFG_OUT:-/tmp/health-fault-gate-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
VERDICTS="$OUT/VERDICTS.txt"
: > "$VERDICTS"
NVSMI="$(command -v nvidia-smi || true)"
[ -n "$NVSMI" ] || { echo "health-fault-gate: FATAL: nvidia-smi not on PATH (arm e needs the real tool to shadow)"; exit 2; }
READY_WAIT_S="${HFG_READY_WAIT_S:-420}"

. tools/port-guard.sh
memra_port_guard health-fault-gate "$PORT" HFG_PORT || exit 2

# Build UNCONDITIONALLY (serve-smoke's rule: a gate that can run a stale binary is a rotted gate).
cargo build --release -p memra-server || { echo "health-fault-gate: build FAILED"; exit 2; }
BIN=target/release/memra-server
sha256sum "$BIN" > "$OUT/binary.sha256"
{ echo "commit $(git rev-parse HEAD)"; echo "model $MODEL"; "$NVSMI" --query-gpu=name,driver_version,power.limit --format=csv,noheader; date -u +%FT%TZ; } > "$OUT/source.txt"

PASSN=0; DOCN=0; FAILN=0
verdict() { # <line ending in -> PASS|DOCUMENTED|FAIL>
  echo "$1" | tee -a "$VERDICTS"
  case "$1" in
    *"-> PASS") PASSN=$((PASSN+1));;
    *"-> DOCUMENTED"*) DOCN=$((DOCN+1));;
    *) FAILN=$((FAILN+1));;
  esac
}
now_ms() { echo $(( ${EPOCHREALTIME/./} / 1000 )); }  # bash 5; `date +%s%3N` prints nanoseconds with this date(1)
jget() { # <file> <dotted.path> -> value or ""
  python3 - "$1" "$2" <<'PY' 2>/dev/null
import json,sys
try:
    v=json.load(open(sys.argv[1]))
    for k in sys.argv[2].split('.'):
        v=v[int(k)] if isinstance(v,list) else v[k]
    print("" if v is None else v)
except Exception:
    print("")
PY
}
gpu_apps() { "$NVSMI" --query-compute-apps=pid,process_name,used_memory --format=csv > "$1" 2>&1; }

SPID=0; D=""; T_LAUNCH=0; T_READY=0
# sample <route> <samples.csv>: one HTTP read, appended as t_ms,code,status,phase,detail
sample() {
  local body="$D/.sample.$$" code
  code=$(curl -s -o "$body" -w '%{http_code}' -m 2 "$BASE$1" 2>/dev/null) || code=000
  [ -n "$code" ] || code=000
  local status phase detail
  if [ -s "$body" ]; then
    status=$(jget "$body" status); phase=$(jget "$body" worker.phase); detail=$(jget "$body" detail)
  else status=""; phase=""; detail=""; fi
  echo "$(( $(now_ms) - T_LAUNCH )),$code,$status,$phase,\"${detail//\"/\'}\"" >> "$2"
  LAST_CODE=$code; LAST_STATUS=$status; LAST_PHASE=$phase; LAST_DETAIL=$detail
}
# boot <label> [ENV=VAL ...]: launch, poll /readyz at 100 ms until 200, record every sample.
boot() {
  local label=$1; shift
  D="$OUT/$label"; mkdir -p "$D"
  gpu_apps "$D/compute-apps-before.csv"
  T_LAUNCH=$(now_ms)
  env MEMRA_COMPAT=openai MEMRA_MODELS="hfg=$MODEL" MEMRA_ADDR=$ADDR "$@" "$BIN" > "$D/server.log" 2>&1 &
  SPID=$!
  echo "$*" > "$D/env.txt"
  : > "$D/readyz-samples.csv"
  local deadline=$(( T_LAUNCH + READY_WAIT_S * 1000 ))
  while :; do
    sample /readyz "$D/readyz-samples.csv"
    if [ "$LAST_CODE" = "200" ]; then T_READY=$(now_ms); break; fi
    if ! kill -0 "$SPID" 2>/dev/null; then echo "  [$label] server died during boot; log tail:"; tail -5 "$D/server.log"; return 1; fi
    if [ "$(now_ms)" -gt "$deadline" ]; then echo "  [$label] FATAL: not ready within ${READY_WAIT_S}s"; return 1; fi
    sleep 0.1
  done
  memra_port_owned health-fault-gate "$PORT" "$SPID" || return 1
  echo "  [$label] ready after $(( T_READY - T_LAUNCH )) ms (pid $SPID)"
}
# stop: SIGTERM only a memra-server this gate started (pid AND cwd), record its exit code.
stop() {
  local label=$1 rc
  if kill -0 "$SPID" 2>/dev/null; then
    if [ "$(readlink /proc/$SPID/cwd 2>/dev/null)" = "$PWD" ]; then kill -TERM "$SPID" 2>/dev/null
    else echo "  [$label] REFUSING to signal pid $SPID: cwd is not this tree"; fi
  fi
  wait "$SPID" 2>/dev/null; rc=$?
  echo "$rc" > "$D/exit"
  echo "  [$label] exit $rc"
  gpu_apps "$D/compute-apps-after.csv"
  rm -f "$D/.sample.$$"
  SPID=0
}
cleanup() { [ "$SPID" != 0 ] && kill -0 "$SPID" 2>/dev/null && [ "$(readlink /proc/$SPID/cwd 2>/dev/null)" = "$PWD" ] && kill -TERM "$SPID" 2>/dev/null; true; }
trap cleanup EXIT

# chat <max_tokens> <stream:true|false> <outprefix> [timeout_s]: writes .hdr .body, prints the code
chat() {
  local code
  code=$(curl -s -m "${4:-180}" -D "$3.hdr" -o "$3.body" -w '%{http_code}' "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"hfg\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about a mutex.\"}],\"max_tokens\":$1,\"temperature\":0,\"stream\":$2}") || code=000
  echo "${code:-000}"
}
completes() { # <prefix>: non-stream body has finish_reason in (stop,length) and completion_tokens>0
  local fr ct
  fr=$(jget "$1.body" choices.0.finish_reason 2>/dev/null); ct=$(jget "$1.body" usage.completion_tokens)
  case "$fr" in stop|length) [ -n "$ct" ] && [ "$ct" -gt 0 ] 2>/dev/null;; *) false;; esac
}
in_arms() { case ",$ARMS," in *",$1,"*) return 0;; *) return 1;; esac; }
lineno() { grep -n -m1 -F -- "$1" "$2" | cut -d: -f1; }

echo "== health-fault-gate: arms=$ARMS port=$PORT out=$OUT =="

# ---------------------------------------------------------------- a + b: armed boot
if in_arms a || in_arms b; then
  echo "--- arm a/b: armed boot (default doors: the calibration probe runs before ready) ---"
  if boot ab; then
    if in_arms a; then
      pre=$(awk -F, '$2!=200' "$D/readyz-samples.csv" | wc -l)
      c000=$(awk -F, '$2==000' "$D/readyz-samples.csv" | wc -l)
      c503=$(awk -F, '$2==503' "$D/readyz-samples.csv" | wc -l)
      early200=$(awk -F, '$2==200 && $3!="ready"' "$D/readyz-samples.csv" | wc -l)
      warm200=$(awk -F, '$2==200 && $4=="warming"' "$D/readyz-samples.csv" | wc -l)
      probe_ln=$(lineno '[admit-cal] boot calibration done:' "$D/server.log"); probe_ln=${probe_ln:-0}
      listen_ln=$(lineno '[server] listening on' "$D/server.log"); listen_ln=${listen_ln:-0}
      first_phase=$(awk -F, '$2==200{print $4; exit}' "$D/readyz-samples.csv")
      v=FAIL
      [ "$probe_ln" -gt 0 ] && [ "$listen_ln" -gt "$probe_ln" ] && [ "$early200" = 0 ] && [ "$warm200" = 0 ] && [ "$pre" -gt 0 ] && v=PASS
      verdict "HFG (a) readiness-before-probe: probe_done_line=$probe_ln listening_line=$listen_ln pre_ready_samples=$pre pre_ready_codes={000:$c000,503:$c503} ready_samples_not_ready=$early200 ready_while_warming=$warm200 first_ready_phase=$first_phase -> $v"
    fi
    if in_arms b; then
      sample /readyz "$D/readyz-b.csv"; before=$LAST_CODE/$LAST_STATUS
      t0=$(now_ms); code=$(chat 32 false "$D/first"); t1=$(now_ms)
      sample /readyz "$D/readyz-b.csv"; after=$LAST_CODE/$LAST_STATUS
      fr=$(jget "$D/first.body" choices.0.finish_reason); ct=$(jget "$D/first.body" usage.completion_tokens)
      v=FAIL
      [ "$before" = "200/ready" ] && [ "$code" = 200 ] && completes "$D/first" && [ "$after" = "200/ready" ] && v=PASS
      verdict "HFG (b) ready-then-first-request: readyz_before=$before request_http=$code finish_reason=$fr completion_tokens=$ct request_ms=$((t1-t0)) ready_to_completion_ms=$((t1-T_READY)) readyz_after=$after (N=1, not a timing claim) -> $v"
    fi
    stop ab
  else
    in_arms a && verdict "HFG (a) readiness-before-probe: boot failed -> FAIL"
    in_arms b && verdict "HFG (b) ready-then-first-request: boot failed -> FAIL"
    stop ab
  fi
fi

# ---------------------------------------------------------------- c: probe-skipped boots
if in_arms c; then
  echo "--- arm c: the probe-skipped boots (documented behaviour, not a pass) ---"
  run_c() { # <label> <expected skip line> ENV=VAL
    local label=$1 skip=$2; shift 2
    if boot "$label" "$@"; then
      skip_ln=$(lineno "$skip" "$D/server.log"); skip_ln=${skip_ln:-0}
      done_ln=$(lineno '[admit-cal] boot calibration done:' "$D/server.log"); done_ln=${done_ln:-0}
      warm=$(awk -F, '$4=="warming"' "$D/readyz-samples.csv" | wc -l)
      t0=$(now_ms); code=$(chat 32 false "$D/first"); t1=$(now_ms)
      v=FAIL
      [ "$skip_ln" -gt 0 ] && [ "$done_ln" = 0 ] && [ "$warm" = 0 ] && [ "$code" = 200 ] && completes "$D/first" && v="DOCUMENTED (ready without warmup; the first request pays the cold route)"
      verdict "HFG (c) probe-skipped $label [$*]: skip_line=$skip_ln probe_done_line=$done_ln warming_samples=$warm ready_ms=$((T_READY-T_LAUNCH)) first_request_http=$code first_request_ms=$((t1-t0)) (N=1) -> $v"
      stop "$label"
    else
      verdict "HFG (c) probe-skipped $label [$*]: boot failed -> FAIL"; stop "$label"
    fi
  }
  run_c c-calibrate0 'boot calibration disarmed (MEMRA_ADMIT_CALIBRATE=0)' MEMRA_ADMIT_CALIBRATE=0
  run_c c-spec0 'boot calibration skipped: spec serving disabled' MEMRA_SERVE_SPEC=0
  run_c c-reserve 'boot calibration skipped: MEMRA_ADMIT_RESERVE_MB teeth door' MEMRA_ADMIT_RESERVE_MB=1536
fi

# ---------------------------------------------------------------- d: worker panic, respawn
if in_arms d; then
  echo "--- arm d: a request panics the worker (MEMRA_PANIC_AFTER=1), the respawn rule (MEMRA_WORKER_RESPAWN=1) ---"
  if boot d MEMRA_PANIC_AFTER=1 MEMRA_WORKER_RESPAWN=1; then
    : > "$D/health-samples.csv"
    trig=$(chat 8 false "$D/trigger" 60)
    t_trig=$(now_ms)
    # 1. the flip: /health 503 with the quoted payload, within 10 s
    flip_ms=""; while [ $(( $(now_ms) - t_trig )) -lt 10000 ]; do
      sample /health "$D/health-samples.csv"
      if [ "$LAST_CODE" = 503 ] && [[ "$LAST_DETAIL" == *"worker thread panicked"* ]]; then flip_ms=$(( $(now_ms) - t_trig )); break; fi
      sleep 0.1
    done
    sample /readyz "$D/readyz-d.csv"; readyz_dead="$LAST_CODE/$LAST_STATUS/$LAST_PHASE"
    # 2. recovery: poll /readyz at 100 ms, record the phase sequence of the not-ready window
    : > "$D/readyz-recovery.csv"; t_rec0=$(now_ms); rec_ms=""
    while [ $(( $(now_ms) - t_rec0 )) -lt $(( READY_WAIT_S * 1000 )) ]; do
      sample /readyz "$D/readyz-recovery.csv"
      if [ "$LAST_CODE" = 200 ]; then rec_ms=$(( $(now_ms) - t_trig )); break; fi
      kill -0 "$SPID" 2>/dev/null || break
      sleep 0.1
    done
    phases=$(awk -F, '$2!=200{print $4}' "$D/readyz-recovery.csv" | uniq | paste -sd'>' -)
    n_load=$(awk -F, '$4=="loading"' "$D/readyz-recovery.csv" | wc -l)
    n_warm=$(awk -F, '$4=="warming"' "$D/readyz-recovery.csv" | wc -l)
    last_load=$(awk -F, '$4=="loading"{n=NR} END{print n+0}' "$D/readyz-recovery.csv")
    first_warm=$(awk -F, '$4=="warming"{print NR; exit}' "$D/readyz-recovery.csv"); first_warm=${first_warm:-0}
    untruthful=$(cat "$D/health-samples.csv" "$D/readyz-recovery.csv" | awk -F, '($2==200 && $3!="ok" && $3!="ready") || ($2!=200 && ($3=="ok" || $3=="ready"))' | wc -l)
    gen=$(curl -s -m 5 "$BASE/readyz" -o "$D/readyz-after.json" -w '' && jget "$D/readyz-after.json" worker.generation)
    after=$(chat 32 false "$D/after" 120)
    sample /health "$D/health-samples.csv"; health_after="$LAST_CODE/$LAST_STATUS"
    panic_ln=$(lineno '[worker] PANIC in the GPU worker thread' "$D/server.log"); panic_ln=${panic_ln:-0}
    respawn_ln=$(lineno '[worker] respawn attempt 1/1' "$D/server.log"); respawn_ln=${respawn_ln:-0}
    v=FAIL
    [ -n "$flip_ms" ] && [[ "$readyz_dead" == 503/* ]] && [ -n "$rec_ms" ] && [ "$gen" = 1 ] && [ "$after" = 200 ] && completes "$D/after" && [ "$health_after" = "200/ok" ] && [ "$untruthful" = 0 ] && [ "$panic_ln" -gt 0 ] && [ "$respawn_ln" -gt "$panic_ln" ] && v=PASS
    verdict "HFG (d) panic-respawn-truthful-health: trigger_http=$trig health_503_quoted_after_ms=${flip_ms:-none} readyz_while_dead=$readyz_dead untruthful_samples=$untruthful recovered_after_ms=${rec_ms:-none} generation_after=$gen request_after_http=$after health_after=$health_after log_panic_line=$panic_ln log_respawn_line=$respawn_ln -> $v"
    v2=FAIL
    [ "$n_load" -gt 0 ] && [ "$n_warm" -gt 0 ] && [ "$first_warm" -gt "$last_load" ] && v2=PASS
    verdict "HFG (a2) warming-phase-on-respawn: not_ready_phase_sequence=$phases loading_samples=$n_load warming_samples=$n_warm loading_then_warming=$([ "$first_warm" -gt "$last_load" ] && echo true || echo false) -> $v2"
    stop d
  else
    verdict "HFG (d) panic-respawn-truthful-health: boot failed -> FAIL"; verdict "HFG (a2) warming-phase-on-respawn: boot failed -> FAIL"; stop d
  fi
fi

# ---------------------------------------------------------------- e: fatal gpu-watch fault latches
if in_arms e; then
  echo "--- arm e: a hung nvidia-smi probe latches the gpu fault; answering again does not clear it ---"
  SHIM="$OUT/e-shim"; mkdir -p "$SHIM"; HANG="$SHIM/hang"; rm -f "$HANG"
  cat > "$SHIM/nvidia-smi" <<SH
#!/usr/bin/env bash
# health-fault-gate arm e: shadows nvidia-smi for the server under test only. While the flag
# file exists the probe hangs (exec, so the deadline kill lands on the sleep itself).
[ -e "$HANG" ] && exec sleep 600
exec "$NVSMI" "\$@"
SH
  chmod +x "$SHIM/nvidia-smi"
  if boot e PATH="$SHIM:$PATH" MEMRA_GPU_WATCH_S=2 MEMRA_GPU_PROBE_TIMEOUT_S=2; then
    : > "$D/health-samples.csv"
    on_ln=$(lineno '[gpu-watch] on: every 2s, probe deadline 2s' "$D/server.log"); on_ln=${on_ln:-0}
    ctrl_ok=0; for _ in 1 2 3; do sample /health "$D/health-samples.csv"; [ "$LAST_CODE/$LAST_STATUS" = "200/ok" ] && ctrl_ok=$((ctrl_ok+1)); sleep 2; done
    touch "$HANG"; t_hang=$(now_ms); latch_ms=""
    while [ $(( $(now_ms) - t_hang )) -lt 20000 ]; do
      sample /health "$D/health-samples.csv"
      if [ "$LAST_CODE" = 503 ] && [[ "$LAST_DETAIL" == *"nvidia-smi did not answer within 2s"* ]]; then latch_ms=$(( $(now_ms) - t_hang )); break; fi
      sleep 0.2
    done
    rm -f "$HANG"; t_clear=$(now_ms)
    sleep 8   # >= 3 probe intervals with the shim answering again
    sample /health "$D/health-samples.csv"; health_late="$LAST_CODE/$LAST_STATUS"; detail_late="$LAST_DETAIL"
    sample /readyz "$D/readyz-e.csv"; readyz_late="$LAST_CODE/$LAST_STATUS"
    req=$(chat 8 false "$D/req" 60)
    crit=$(grep -c -F '[gpu-watch] CRITICAL' "$D/server.log")
    late_ok=$(awk -F, -v t=$((t_clear - T_LAUNCH)) '$1>t && $2==200' "$D/health-samples.csv" | wc -l)
    v=FAIL
    [ "$on_ln" -gt 0 ] && [ "$ctrl_ok" = 3 ] && [ -n "$latch_ms" ] && [ "$health_late" = "503/unhealthy" ] && [[ "$detail_late" == *"did not answer"* ]] && [[ "$readyz_late" == 503/* ]] && [ "$crit" = 1 ] && [ "$late_ok" = 0 ] && v=PASS
    verdict "HFG (e) fatal-fault-not-cleared-by-timeout: watch_on_line=$on_ln control_200_samples=$ctrl_ok/3 latched_after_ms=${latch_ms:-none} shim_answering_again_for_ms=$(( $(now_ms) - t_clear )) health_after=$health_late readyz_after=$readyz_late health_200_after_clear=$late_ok critical_lines=$crit request_http_while_latched=$req (observed, bounded, not asserted) -> $v"
    stop e
  else
    verdict "HFG (e) fatal-fault-not-cleared-by-timeout: boot failed -> FAIL"; stop e
  fi
fi

# ---------------------------------------------------------------- f: graceful drain
if in_arms f; then
  echo "--- arm f: SIGTERM with a stream open flips readiness first and drains ---"
  if boot f MEMRA_DRAIN_S=60; then
    : > "$D/stream.body"
    curl -s -N -m 170 -D "$D/stream.hdr" -o "$D/stream.body" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"hfg","messages":[{"role":"user","content":"Write a long story about a lighthouse keeper."}],"max_tokens":256,"temperature":0,"stream":true}' &
    CPID=$!
    started=false; t_s=$(now_ms)
    while [ $(( $(now_ms) - t_s )) -lt 30000 ]; do
      [ "$(grep -c '^data: ' "$D/stream.body" 2>/dev/null)" -ge 2 ] 2>/dev/null && { started=true; break; }
      sleep 0.05
    done
    frames_at_term=$(grep -c '^data: ' "$D/stream.body")
    kill -TERM "$SPID"; t_term=$(now_ms)
    sample /readyz "$D/readyz-f.csv"; readyz_drain="$LAST_CODE/$LAST_STATUS"; readyz_detail="$LAST_DETAIL"
    ra=$(curl -s -m 2 -o /dev/null -D - "$BASE/readyz" | grep -i '^retry-after:' | tr -d '\r' | awk '{print $2}')
    sample /health "$D/health-f.csv"; health_drain="$LAST_CODE/$LAST_STATUS"
    newreq=$(chat 8 false "$D/during-drain" 20)
    new_code=$(jget "$D/during-drain.body" error.code)
    new_ra=$(grep -i '^retry-after:' "$D/during-drain.hdr" | tr -d '\r' | awk '{print $2}')
    wait "$CPID"; crc=$?
    t_stream_done=$(now_ms)
    frames=$(grep -c '^data: ' "$D/stream.body"); done_seen=$(grep -c '^data: \[DONE\]' "$D/stream.body")
    fr=$(grep '^data: {' "$D/stream.body" | sed 's/^data: //' | python3 -c '
import json,sys
fr=""
for l in sys.stdin:
    try:
        for c in json.loads(l).get("choices",[]):
            if c.get("finish_reason"): fr=c["finish_reason"]
    except Exception: pass
print(fr)')
    wait "$SPID" 2>/dev/null; rc=$?; echo "$rc" > "$D/exit"; SPID=0
    t_exit=$(now_ms)
    gpu_apps "$D/compute-apps-after.csv"
    rm -f "$D/.sample.$$"
    drain_ln=$(lineno '[server] drain complete in' "$D/server.log"); drain_ln=${drain_ln:-0}
    deadline_ln=$(lineno 'drain deadline' "$D/server.log"); deadline_ln=${deadline_ln:-0}
    v=FAIL
    [ "$started" = true ] && [ "$readyz_drain" = "503/not_ready" ] && [[ "$readyz_detail" == *draining* ]] && [ -n "$ra" ] && [ "$health_drain" = "200/draining" ] && [ "$newreq" = 503 ] && [ "$new_code" = draining ] && [ -n "$new_ra" ] && [ "$crc" = 0 ] && [ "$done_seen" = 1 ] && [ -n "$fr" ] && [ "$rc" = 0 ] && [ "$drain_ln" -gt 0 ] && [ "$deadline_ln" = 0 ] && v=PASS
    verdict "HFG (f) sigterm-drains-and-flips-readiness-first: stream_frames_at_sigterm=$frames_at_term readyz_during_drain=$readyz_drain retry_after_s=${ra:-none} health_during_drain=$health_drain new_request_http=$newreq new_request_code=$new_code new_request_retry_after_s=${new_ra:-none} stream_curl_rc=$crc stream_frames=$frames stream_done=$done_seen stream_finish_reason=$fr stream_finished_after_sigterm_ms=$((t_stream_done-t_term)) exit_code=$rc exit_after_sigterm_ms=$((t_exit-t_term)) drain_complete_line=$drain_ln deadline_hit_line=$deadline_ln -> $v"
  else
    verdict "HFG (f) sigterm-drains-and-flips-readiness-first: boot failed -> FAIL"; stop f
  fi
fi

# sstream <prefix> <prompt> <max_tokens>: one streamed chat request in the background; its pid in SPIDS
sstream() {
  : > "$1.body"
  curl -s -N -m 170 -D "$1.hdr" -o "$1.body" -w '%{http_code}' "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"hfg\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"max_tokens\":$3,\"temperature\":0,\"stream\":true}" \
    > "$1.code" 2>/dev/null &
  SPIDS="$SPIDS $!"
}
sframes() { local n; n=$(grep -c '^data: {' "$1.body" 2>/dev/null); echo "${n:-0}"; }
sdone() { local n; n=$(grep -c '^data: \[DONE\]' "$1.body" 2>/dev/null); echo "${n:-0}"; }
sfinish() {
  grep '^data: {' "$1.body" 2>/dev/null | sed 's/^data: //' | python3 -c '
import json,sys
fr=""
for l in sys.stdin:
    try:
        for c in json.loads(l).get("choices",[]):
            if c.get("finish_reason"): fr=c["finish_reason"]
    except Exception: pass
print(fr)'
}
scode() { local c; c=$(cat "$1.code" 2>/dev/null); echo "${c:-000}"; }
serror() { grep -o '"message":"[^"]*"' "$1.body" 2>/dev/null | head -1; }
sok() { [ "$(scode "$1")" = 200 ] && [ "$(sdone "$1")" = 1 ] && [ -n "$(sfinish "$1")" ]; }
P1="Write one sentence about a mutex."; P2="Name three uses of a semaphore."; P3="Describe a spinlock in two sentences."

# ---------------------------------------------------------------- g: a step OOM parks and completes
run_g() { # <label> <fault count> <red:true|false>: one non-streamed request (DAY47 addendum A)
  local label=$1 n=$2 red=$3
  if boot "$label" MEMRA_STEP_OOM_FAULT="$n"; then
    code=$(chat 48 false "$D/solo" 170)
    # either injection point of a solo step (the plain dispatch's or the non-batching step's; DAY47 addendum B)
    fired=$(grep -cE 'MEMRA_STEP_OOM_FAULT fired: this (non-batching )?step reports' "$D/server.log")
    parked=$(grep -c '\[admit-oom\] step OOM parked session back to queue' "$D/server.log")
    panics=$(grep -cE 'panicked|\[worker\] PANIC' "$D/server.log")
    sample /health "$D/health-g.csv"; health_after=$LAST_CODE
    done_ok=false; completes "$D/solo" && done_ok=true
    err=$(jget "$D/solo.body" error.message)
    green=false
    [ "$code" = 200 ] && [ "$done_ok" = true ] && [ "$parked" -ge 1 ] && [ "$panics" = 0 ] && [ "$health_after" = 200 ] && green=true
    if [ "$red" = false ]; then
      v=FAIL; [ "$green" = true ] && [ "$parked" = 1 ] && [ "$fired" = 1 ] && v=PASS
      verdict "HFG (g) step-oom-parks-and-completes: fault=$n fired_lines=$fired parked_lines=$parked http=$code finish_reason=$(jget "$D/solo.body" choices.0.finish_reason) completion_tokens=$(jget "$D/solo.body" usage.completion_tokens) panic_lines=$panics health_after=$health_after -> $v"
    else
      v=FAIL; [ "$green" = false ] && [ "$panics" = 0 ] && [ "$parked" -ge 1 ] && v=PASS
      verdict "HFG (g-red) step-oom-past-the-retry-budget: fault=$n fired_lines=$fired parked_lines=$parked http=$code error={${err:-none}} panic_lines=$panics green_assertion_fired=$([ "$green" = false ] && echo true || echo false) -> $v"
    fi
    stop "$label"
  else
    verdict "HFG ($label) step-oom: boot failed -> FAIL"; stop "$label"
  fi
}
run_g_batch() { # the three-stream shape of DAY47 1.1: a DOCUMENTED reading of the batched chunk (O14)
  if boot g-batch MEMRA_STEP_OOM_FAULT=1; then
    SPIDS=""
    sstream "$D/r0" "$P1" 48; sstream "$D/r1" "$P2" 48; sstream "$D/r2" "$P3" 48
    # shellcheck disable=SC2086
    wait $SPIDS
    fired=$(grep -c 'MEMRA_STEP_OOM_FAULT fired: this batched decode chunk' "$D/server.log")
    parked=$(grep -c '\[admit-oom\] step OOM parked session back to queue' "$D/server.log")
    ok_n=0; err_n=0; c5=0
    for r in r0 r1 r2; do
      sok "$D/$r" && ok_n=$((ok_n+1))
      [ -n "$(serror "$D/$r")" ] && err_n=$((err_n+1))
      case "$(scode "$D/$r")" in 5*) c5=$((c5+1));; esac
    done
    verdict "HFG (g-batch) step-oom-on-a-batched-chunk: batched_fired_lines=$fired parked_lines=$parked completed=$ok_n/3 ended_with_error_event=$err_n/3 http_5xx=$c5 (the chunk's error arm ends every session of the chunk; owed O14) -> DOCUMENTED"
    stop g-batch
  else
    verdict "HFG (g-batch) step-oom-on-a-batched-chunk: boot failed -> FAIL"; stop g-batch
  fi
}
if in_arms g; then
  echo "--- arm g: a step OOM parks, requeues and completes (and the red twin, and the batched reading) ---"
  run_g g 1 false
  run_g g-red 4 true
  run_g_batch
fi

# ---------------------------------------------------------------- h: a client disconnect retires the session
run_h() { # <label> <close:true|false>
  local label=$1 close=$2
  if boot "$label"; then
    SPIDS=""
    sstream "$D/closed" "Write a long story about a lighthouse keeper." 256; cpid=${SPIDS##* }
    sstream "$D/peer" "Write a long story about a harbor pilot." 256
    t_s=$(now_ms); t_close=0
    while [ $(( $(now_ms) - t_s )) -lt 60000 ]; do
      [ "$(sframes "$D/closed")" -ge 8 ] && break
      sleep 0.02
    done
    frames_at_close=$(sframes "$D/closed")
    if [ "$close" = true ]; then kill "$cpid" 2>/dev/null; t_close=$(now_ms); fi
    t_line=0
    while [ $(( $(now_ms) - t_s )) -lt 180000 ]; do
      if grep -q '\[abort\] client disconnected:' "$D/server.log"; then t_line=$(now_ms); break; fi
      [ "$close" = false ] && ! kill -0 "$cpid" 2>/dev/null && break
      sleep 0.02
    done
    # shellcheck disable=SC2086
    wait $SPIDS 2>/dev/null
    abort_lines=$(grep -c '\[abort\] client disconnected:' "$D/server.log")
    sleep 1
    active=$(curl -s -m 2 "$BASE/metrics" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("active_sessions"))' 2>/dev/null)
    sample /health "$D/health-h.csv"; health_after=$LAST_CODE
    peer_ok=false; sok "$D/peer" && peer_ok=true
    latency=none; [ "$t_close" -gt 0 ] && [ "$t_line" -gt 0 ] && latency=$(( t_line - t_close ))
    green=false
    [ "$abort_lines" -ge 1 ] && [ "$latency" != none ] && [ "$latency" -le 1000 ] && [ "$peer_ok" = true ] && [ "$active" = 0 ] && [ "$health_after" = 200 ] && green=true
    if [ "$close" = true ]; then
      v=FAIL; [ "$green" = true ] && v=PASS
      verdict "HFG (h) client-disconnect-retires-within-1000ms: frames_at_close=$frames_at_close abort_lines=$abort_lines close_to_abort_line_ms=$latency peer_complete=$peer_ok peer_finish=$(sfinish "$D/peer") active_sessions_after=$active health_after=$health_after -> $v"
    else
      v=FAIL; [ "$green" = false ] && [ "$abort_lines" = 0 ] && [ "$peer_ok" = true ] && v=PASS
      verdict "HFG (h-red) no-disconnect: frames_at_close=$frames_at_close abort_lines=$abort_lines closed_request_complete=$(sok "$D/closed" && echo true || echo false) peer_complete=$peer_ok green_assertion_fired=$([ "$green" = false ] && echo true || echo false) -> $v"
    fi
    stop "$label"
  else
    verdict "HFG ($label) client-disconnect: boot failed -> FAIL"; stop "$label"
  fi
}
if in_arms h; then
  echo "--- arm h: a client disconnect retires the session within one tick; the peer completes (and the red twin) ---"
  run_h h true
  run_h h-red false
fi

echo "health-fault-gate: arms=$ARMS pass=$PASSN documented=$DOCN fail=$FAILN receipts=$OUT" | tee -a "$VERDICTS"
[ "$FAILN" = 0 ]
